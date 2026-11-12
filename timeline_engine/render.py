"""Job orchestration: state manifest, crash recovery and atomic publish."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .errors import IntegrityError, RenderError
from .ffmpeg_tools import Toolchain, run_ffmpeg, sha256_file
from .model import Project
from .segment import (
    SegmentCache,
    SegmentRenderer,
    SegmentResult,
    SegmentScheduler,
)
from .source import (
    WORK_AUDIO_CHANNELS,
    WORK_COLOR_RANGE,
    WORK_COLOR_SPACE,
    WORK_PIX_FMT,
    SourceStore,
)
from .timeline import TimelinePlan, TimelinePlanner, build_plan, plan_segments

STATE_NAME = "job_state.json"
PUBLISHED_DIR = "published"


@dataclass
class RenderConfig:
    job_dir: Path
    segment_frames: int = 90
    parallelism: int = 1
    make_mp4: bool = True
    mp4_video_bitrate: str = "2M"
    mp4_audio_bitrate: str = "192k"
    crash_before_done_segments: tuple[int, ...] = ()


class RenderJob:
    """Owns the on-disk job lifecycle and distinguishes artifact states."""

    def __init__(self, project: Project, config: RenderConfig) -> None:
        self.project = project
        self.config = config
        self.job_dir = Path(config.job_dir)
        self.source_dir = self.job_dir / "source_cache"
        self.segment_dir = self.job_dir / "segments"
        self.staging_dir = self.job_dir / "staging"
        self.published_dir = self.job_dir / PUBLISHED_DIR
        for directory in (
            self.job_dir,
            self.source_dir,
            self.segment_dir,
            self.staging_dir,
            self.published_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.job_dir / STATE_NAME
        self.toolchain = Toolchain.capture(__version__).to_dict()
        self.plan: TimelinePlan | None = None
        self.results: list[SegmentResult] = []

    def _resolve_media_paths(self) -> None:
        root = self.project.source_root or Path.cwd()
        for media in self.project.media.values():
            candidate = Path(media.path)
            if not candidate.is_absolute():
                media.path = str((root / candidate).resolve())

    def _write_state(
        self,
        *,
        status: str,
        segment_frames: int,
        parallelism: int,
        completed: dict[str, dict],
        error: str | None = None,
    ) -> None:
        assert self.plan is not None
        state = {
            "schema": "timeline-engine-job/1",
            "status": status,
            "toolchain": self.toolchain,
            "work_format": {
                "pix_fmt": WORK_PIX_FMT,
                "color_range": WORK_COLOR_RANGE,
                "color_space": WORK_COLOR_SPACE,
                "audio_channels": WORK_AUDIO_CHANNELS,
                "audio_sample": "pcm_f32le",
                "scaling": "bicubic",
                "padding": "black letterbox/pillarbox",
                "rounding": "round half up (sample count); floor(x+0.5) pixels",
                "clipping": "hard clip float audio to [-1, 1]",
            },
            "plan": self.plan.to_dict(),
            "segment_frames": segment_frames,
            "parallelism": parallelism,
            "completed_segments": completed,
            "error": error,
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def run(self) -> dict:
        self._resolve_media_paths()
        self.plan = build_plan(self.project)
        store = SourceStore(
            self.source_dir, self.project.output, self.toolchain
        )
        self._register_marker_ids(store)
        bundles = {
            media_id: store.get(media)
            for media_id, media in self.project.media.items()
        }
        planner = TimelinePlanner(self.project, self.plan, bundles)
        segments = plan_segments(self.plan, self.config.segment_frames)
        cache = SegmentCache(
            self.project, self.plan, bundles, self.toolchain
        )
        renderer = SegmentRenderer(
            self.project,
            self.plan,
            planner,
            cache,
            self.segment_dir,
            self.toolchain,
        )
        if self.config.crash_before_done_segments:
            renderer.crash_before_done.update(
                self.config.crash_before_done_segments
            )

        completed = self._load_completed_markers(segments, renderer)
        self._write_state(
            status="running",
            segment_frames=self.config.segment_frames,
            parallelism=self.config.parallelism,
            completed=completed,
        )
        try:
            scheduler = SegmentScheduler(renderer, self.config.parallelism)
            self.results = scheduler.run(segments)
        except Exception as exc:
            self._write_state(
                status="failed",
                segment_frames=self.config.segment_frames,
                parallelism=self.config.parallelism,
                completed=completed,
                error=str(exc),
            )
            raise

        completed = {
            str(result.segment_id): {
                "key": result.key,
                "cache_hit": result.cache_hit,
                "frame_count": result.frame_count,
                "sample_count": result.sample_count,
            }
            for result in self.results
        }
        self._verify_totals()
        self._write_state(
            status="segments_complete",
            segment_frames=self.config.segment_frames,
            parallelism=self.config.parallelism,
            completed=completed,
        )
        publish_meta = self._publish()
        self._write_state(
            status="published",
            segment_frames=self.config.segment_frames,
            parallelism=self.config.parallelism,
            completed=completed,
        )
        result = {
            "plan": self.plan.to_dict(),
            "segments": len(segments),
            "cache_hits": renderer.hits,
            "cache_misses": renderer.misses,
            "published": publish_meta,
            "toolchain": self.toolchain,
        }
        report_path = self.job_dir / "render_report.json"
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _load_completed_markers(self, segments, renderer) -> dict[str, dict]:
        completed: dict[str, dict] = {}
        if not self.state_path.is_file():
            return completed
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("status") == "published":
            # A published prior version is never mutated; reruns create a new
            # version directory below, while segment artifacts may still hit.
            pass
        for segment in segments:
            key = renderer.cache.key_for(segment)
            video_path, audio_path, done_path = renderer._paths(key)
            if renderer._is_complete(
                segment, key, video_path, audio_path, done_path
            ):
                completed[str(segment.segment_id)] = {
                    "key": key,
                    "cache_hit": True,
                    "frame_count": segment.frame_count,
                    "sample_count": segment.sample_count,
                }
        return completed

    def _verify_totals(self) -> None:
        assert self.plan is not None
        frames = sum(result.frame_count for result in self.results)
        samples = sum(result.sample_count for result in self.results)
        if frames != self.plan.frame_count:
            raise IntegrityError(
                f"segment frames {frames} != planned {self.plan.frame_count}"
            )
        if samples != self.plan.sample_count:
            raise IntegrityError(
                f"segment samples {samples} != planned {self.plan.sample_count}"
            )
        ids = [result.segment_id for result in self.results]
        if ids != sorted(ids) or len(set(ids)) != len(ids):
            raise IntegrityError("segment ordering/id set is inconsistent")

    def _register_marker_ids(self, store: SourceStore) -> None:
        """Pick up generator marker ids from generated_manifest.json if present."""
        for media in self.project.media.values():
            source_path = Path(media.path)
            manifest_path = source_path.parent / "generated_manifest.json"
            if not manifest_path.is_file():
                continue
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in entries:
                if Path(entry["path"]).resolve() == source_path.resolve():
                    store.assign_source_id(media.media_id, int(entry["source_id"]))
                    break

    def _publish(self) -> dict:
        assert self.plan is not None
        ordered = sorted(self.results, key=lambda item: item.segment_id)
        version_dir = self.staging_dir / "version_tmp"
        if version_dir.exists():
            shutil.rmtree(version_dir)
        version_dir.mkdir(parents=True)

        video_list = version_dir / "video_concat.txt"
        audio_list = version_dir / "audio_concat.txt"
        video_list.write_text(
            "\n".join(
                f"file '{item.video_path.resolve().as_posix()}'"
                for item in ordered
            ),
            encoding="utf-8",
        )
        audio_list.write_text(
            "\n".join(
                f"file '{item.audio_path.resolve().as_posix()}'"
                for item in ordered
            ),
            encoding="utf-8",
        )
        lossless_mkv = version_dir / "master_lossless.mkv"
        # Stream-copy concatenation of identical FFV1/WAV parameters; no
        # re-encode occurs, so segment boundaries cannot alter pixels/samples.
        run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(video_list),
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(audio_list),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c",
                "copy",
                str(lossless_mkv),
            ]
        )

        published_meta: dict = {
            "lossless_mkv": None,
            "mp4": None,
        }
        version = self._next_version()
        final_lossless = (
            self.published_dir / f"master_lossless_{version}.mkv"
        )
        shutil.move(str(lossless_mkv), final_lossless)
        published_meta["lossless_mkv"] = {
            "path": str(final_lossless),
            "sha256": sha256_file(final_lossless),
        }

        if self.config.make_mp4:
            mp4_tmp = version_dir / "playable.mp4"
            # CPU x264/AAC only.  AAC priming samples are recorded in the
            # metadata so the validator can align against the lossless master
            # instead of treating encoder padding as timeline drift.
            run_ffmpeg(
                [
                    "-i",
                    str(final_lossless),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-vf",
                    "scale=in_range=full:out_range=limited",
                    "-c:a",
                    "aac",
                    "-b:a",
                    self.config.mp4_audio_bitrate,
                    "-movflags",
                    "+faststart",
                    str(mp4_tmp),
                ]
            )
            final_mp4 = (
                self.published_dir / f"playable_{version}.mp4"
            )
            shutil.move(str(mp4_tmp), final_mp4)
            priming = self._aac_priming(final_mp4)
            published_meta["mp4"] = {
                "path": str(final_mp4),
                "sha256": sha256_file(final_mp4),
                "aac_priming_samples": priming,
                "note": (
                    "AAC encoder priming/padding is excluded from timeline "
                    "validation; compare decoded content after skipping the "
                    "declared priming samples."
                ),
            }

        meta_path = (
            self.published_dir / f"publish_{version}.json"
        )
        meta_payload = {
            "toolchain": self.toolchain,
            "plan": self.plan.to_dict(),
            "work_format": {
                "pix_fmt": WORK_PIX_FMT,
                "color_range": WORK_COLOR_RANGE,
                "color_space": WORK_COLOR_SPACE,
            },
            "artifacts": published_meta,
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
        shutil.rmtree(version_dir, ignore_errors=True)
        return published_meta

    def _next_version(self) -> int:
        existing = list(self.published_dir.glob("master_lossless_*.mkv"))
        return len(existing) + 1

    def _aac_priming(self, mp4_path: Path) -> int:
        from .ffmpeg_tools import ffprobe_json

        data = ffprobe_json(
            [
                "-show_streams",
                "-select_streams",
                "a:0",
                "-of",
                "json",
                str(mp4_path),
            ]
        )
        streams = data.get("streams", [])
        if not streams:
            return 0
        # mp4 edts/encoder delay surfaces as start_time in stream seconds.
        start_time = streams[0].get("start_time")
        sample_rate = int(streams[0].get("sample_rate", 48000))
        if start_time in (None, "N/A"):
            return 0
        from fractions import Fraction

        return round(float(Fraction(str(start_time))) * sample_rate)


def render_project(project: Project, config: RenderConfig) -> dict:
    """Convenience entry point used by the CLI and tests."""
    if config.segment_frames <= 0:
        raise RenderError("segment_frames must be positive")
    if config.parallelism <= 0:
        raise RenderError("parallelism must be positive")
    job = RenderJob(project, config)
    return job.run()
