from __future__ import annotations

import json
import shutil
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from timeline_engine.errors import ProjectValidationError
from timeline_engine.generator import (
    decode_frame_marker,
    generate_suite,
)
from timeline_engine.model import (
    Clip,
    load_project,
    validate_project,
)
from timeline_engine.project_factory import (
    build_demo_project,
    build_invalid_project,
)
from timeline_engine.render import RenderConfig, RenderJob
from timeline_engine.source import SourceStore
from timeline_engine.validator import (
    OutputValidator,
    compare_renders,
    decode_lossless_audio,
    decode_lossless_video,
)

WORKSPACE = Path(__file__).resolve().parent.parent / "workspace_test"
SOURCE_DIR = WORKSPACE / "sources"


def _ensure_sources() -> list:
    if not (SOURCE_DIR / "generated_manifest.json").exists():
        return generate_suite(SOURCE_DIR, width=256, height=144)
    entries = json.loads(
        (SOURCE_DIR / "generated_manifest.json").read_text(encoding="utf-8")
    )
    from timeline_engine.generator import GeneratedSpec

    return [
        GeneratedSpec(
            path=Path(item["path"]),
            kind=item["kind"],
            width=item["width"],
            height=item["height"],
            fps=Fraction(item["fps"]["num"], item["fps"]["den"]),
            frame_count=item["frame_count"],
            sample_rate=item["sample_rate"],
            source_id=item["source_id"],
            first_video_pts=Fraction(item["first_video_pts"]),
            first_audio_pts=(
                Fraction(item["first_audio_pts"])
                if item["first_audio_pts"]
                else None
            ),
            pulse_positions=tuple(item["pulse_positions"]),
        )
        for item in entries
    ]


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        cls.specs = _ensure_sources()
        cls.project = build_demo_project(cls.specs, width=256, height=144)
        cls.project.write_json(WORKSPACE / "demo_project.json")

    def _render(self, name: str, segment_frames: int, parallelism: int):
        job_dir = WORKSPACE / name
        config = RenderConfig(
            job_dir=job_dir,
            segment_frames=segment_frames,
            parallelism=parallelism,
            make_mp4=True,
        )
        job = RenderJob(self.project, config)
        report = job.run()
        self.render_report = report
        master = next(job.published_dir.glob("master_lossless_*.mkv"))
        return job_dir, master, report

    def test_01_three_segment_sizes_two_parallelism_match(self) -> None:
        configs = [
            ("whole", 10_000, 1),
            ("seg45_p1", 45, 1),
            ("seg23_p1", 23, 1),
            ("seg17_p2", 17, 2),
            ("seg7_p2", 7, 2),
        ]
        masters = {}
        for name, size, parallel in configs:
            _, master, _ = self._render(name, size, parallel)
            masters[name] = master
        base = masters["whole"]
        for name, master in masters.items():
            base_frames = decode_lossless_video(
                base, 256, 144, WORKSPACE / f"cmp_{name}_base"
            )
            cand_frames = decode_lossless_video(
                master, 256, 144, WORKSPACE / f"cmp_{name}_cand"
            )
            base_audio, _ = decode_lossless_audio(
                base, WORKSPACE / f"cmp_{name}_base"
            )
            cand_audio, _ = decode_lossless_audio(
                master, WORKSPACE / f"cmp_{name}_cand"
            )
            result = compare_renders(
                base_frames, cand_frames, base_audio, cand_audio
            )
            self.assertTrue(result["ok"], msg=(name, result))
            self.assertEqual(result["video_pixel_mismatches"], 0)
            self.assertLessEqual(result["audio_max_abs_error"], 1e-6)

    def test_02_independent_validator_passes(self) -> None:
        _, master, _ = self._render("validated", 45, 1)
        project = load_project(WORKSPACE / "demo_project.json")
        toolchain = self.render_report["toolchain"]
        store = SourceStore(
            WORKSPACE / "validated" / "source_cache",
            project.output,
            toolchain,
        )
        validator = OutputValidator(
            project, store, WORKSPACE / "validated" / "validator_work"
        )
        report = validator.validate_lossless(master)
        self.assertTrue(report.ok, msg=report.to_dict())

    def test_03_invalid_project_rejected_before_render(self) -> None:
        invalid = build_invalid_project(self.specs)
        with self.assertRaises(ProjectValidationError):
            validate_project(invalid)
        invalid_path = WORKSPACE / "invalid_project_tmp.json"
        invalid.write_json(invalid_path)
        with self.assertRaises(ProjectValidationError):
            load_project(invalid_path)

    def test_04_crash_before_done_marker_then_resume(self) -> None:
        job_dir = WORKSPACE / "crash_isolated"
        if job_dir.exists():
            shutil.rmtree(job_dir)
        config = RenderConfig(
            job_dir=job_dir,
            segment_frames=30,
            parallelism=1,
            make_mp4=False,
            crash_before_done_segments=(2,),
        )
        job = RenderJob(self.project, config)
        with self.assertRaises(Exception):
            job.run()
        state = json.loads(
            (job_dir / "job_state.json").read_text(encoding="utf-8")
        )
        self.assertEqual(state["status"], "failed")
        # Orphaned MKV/WAV without done marker must not be treated as cache.
        orphan_segments = list((job_dir / "segments").glob("*/video.mkv"))
        done_markers = list((job_dir / "segments").glob("*/done.json"))
        self.assertGreater(len(orphan_segments), len(done_markers))

        resume_config = RenderConfig(
            job_dir=job_dir,
            segment_frames=30,
            parallelism=1,
            make_mp4=True,
        )
        resume_job = RenderJob(self.project, resume_config)
        resume_report = resume_job.run()
        self.assertTrue(
            list(resume_job.published_dir.glob("master_lossless_*.mkv"))
        )
        self.assertGreaterEqual(len(resume_report["cache_hits"]), 1)

    def test_05_cache_invalidation_scenarios(self) -> None:
        job_dir = WORKSPACE / "cache_base_isolated"
        if job_dir.exists():
            shutil.rmtree(job_dir)
        config = RenderConfig(
            job_dir=job_dir, segment_frames=30, parallelism=1, make_mp4=False
        )
        job = RenderJob(self.project, config)
        first = job.run()
        self.assertGreater(len(first["cache_misses"]), 0)

        # Second identical run: every segment must be a cache hit.
        rerun = RenderJob(self.project, config).run()
        self.assertEqual(len(rerun["cache_misses"]), 0)
        self.assertGreater(len(rerun["cache_hits"]), 0)

        # Change only one transition length: only segments touching it miss.
        changed = RenderJob.__new__(RenderJob)
        modified_project = self._modified_transition_project()
        modified_job = RenderJob(
            modified_project,
            RenderConfig(
                job_dir=job_dir,
                segment_frames=30,
                parallelism=1,
                make_mp4=False,
            ),
        )
        transition_report = modified_job.run()
        affected_segments = len(job.run()["cache_hits"])
        self.assertGreaterEqual(
            len(transition_report["cache_misses"]), 1
        )
        self.assertLess(
            len(transition_report["cache_misses"]),
            affected_segments,
        )

        # Same-name source replacement flips its content hash and invalidates
        # every segment using that media while unrelated segments stay cached.
        replaced_project, target, backup = self._replace_same_named_media()
        try:
            replaced_job = RenderJob(
                replaced_project,
                RenderConfig(
                    job_dir=job_dir,
                    segment_frames=30,
                    parallelism=1,
                    make_mp4=False,
                ),
            )
            replaced_report = replaced_job.run()
            self.assertGreater(len(replaced_report["cache_misses"]), 0)
        finally:
            if backup.exists():
                target.unlink(missing_ok=True)
                shutil.move(str(backup), target)

    def _modified_transition_project(self):
        from timeline_engine.model import Transition

        project = load_project(WORKSPACE / "demo_project.json")
        project.transitions[0] = Transition(
            from_clip=project.transitions[0].from_clip,
            to_clip=project.transitions[0].to_clip,
            overlap=Fraction(1, 4),
        )
        validate_project(project)
        return project

    def _replace_same_named_media(self):
        project = load_project(WORKSPACE / "demo_project.json")
        target = Path(project.media["cfr25"].path)
        backup = target.with_suffix(".backup.mkv")
        if not backup.exists():
            shutil.copy2(target, backup)
        with target.open("ab") as handle:
            handle.write(b"engine-cache-bust")
        validate_project(project)
        return project, target, backup

    def test_06_marker_roundtrip(self) -> None:
        from timeline_engine.generator import build_frame

        for index in (0, 1, 17, 255, 65535, 4_000_000_000):
            frame = build_frame(9, index, 256, 144)
            self.assertEqual(
                decode_frame_marker(frame), (9, index, True)
            )


if __name__ == "__main__":
    unittest.main()
