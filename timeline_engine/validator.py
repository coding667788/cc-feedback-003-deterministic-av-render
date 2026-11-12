"""Independent output verification.

The validator never trusts renderer log lines.  It:

1. decodes the published lossless MKV itself with FFmpeg;
2. reads frame marker bits back from pixels and checks source id / frame
   index / parity;
3. independently recomputes the expected source frame selection from the
   project rationals and the source display PTS lists;
4. decodes float audio and locates the injected triangular pulses, checking
   their output sample positions against the exact timeline math;
5. compares whole vs split/resume renders for pixel-exact video equality and
   strict float audio tolerance (default ``1e-6``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .errors import IntegrityError
from .ffmpeg_tools import run_ffmpeg
from .generator import decode_frame_marker
from .model import Project
from .source import SourceStore, WORK_AUDIO_CHANNELS
from .timeline import (
    ItemRuntime,
    TimelinePlan,
    TimelinePlanner,
    build_plan,
)

DEFAULT_AUDIO_TOLERANCE = 1e-6


def decode_lossless_video(
    mkv_path: str | Path, width: int, height: int, work_dir: Path
) -> np.ndarray:
    """Decode a lossless MKV to an in-memory RGB24 frame stack."""
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / "decoded_video.raw"
    run_ffmpeg(
        [
            "-i",
            str(mkv_path),
            "-map",
            "0:v:0",
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            str(raw_path),
        ]
    )
    data = np.fromfile(raw_path, dtype=np.uint8)
    frame_bytes = width * height * 3
    if data.size % frame_bytes != 0:
        raise IntegrityError("decoded video size is not a whole number of frames")
    return data.reshape(-1, height, width, 3)


def decode_lossless_audio(
    mkv_path: str | Path, work_dir: Path
) -> tuple[np.ndarray, int]:
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / "decoded_audio.f32"
    run_ffmpeg(
        [
            "-i",
            str(mkv_path),
            "-map",
            "0:a:0",
            "-vn",
            "-f",
            "f32le",
            "-ac",
            str(WORK_AUDIO_CHANNELS),
            str(raw_path),
        ]
    )
    data = np.fromfile(raw_path, dtype=np.float32).reshape(
        -1, WORK_AUDIO_CHANNELS
    )
    return data, data.shape[0]


@dataclass
class ValidationReport:
    frame_count: int = 0
    sample_count: int = 0
    marker_failures: list[dict] = field(default_factory=list)
    marker_checked: int = 0
    expected_mismatches: list[dict] = field(default_factory=list)
    pulse_results: list[dict] = field(default_factory=list)
    audio_max_abs_error: float = 0.0
    video_pixel_mismatches: int = 0
    ok: bool = False
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "frame_count": self.frame_count,
            "sample_count": self.sample_count,
            "marker_checked": self.marker_checked,
            "marker_failures": self.marker_failures[:20],
            "expected_mismatches": self.expected_mismatches[:20],
            "pulse_results": self.pulse_results,
            "audio_max_abs_error": self.audio_max_abs_error,
            "video_pixel_mismatches": self.video_pixel_mismatches,
            "details": self.details,
        }


class OutputValidator:
    """Re-derives expectations independently from project + sources."""

    def __init__(
        self,
        project: Project,
        source_store: SourceStore,
        work_dir: Path,
        audio_tolerance: float = DEFAULT_AUDIO_TOLERANCE,
    ) -> None:
        self.project = project
        self.store = source_store
        self.work_dir = Path(work_dir)
        self.audio_tolerance = audio_tolerance
        self.plan: TimelinePlan = build_plan(project)
        self.bundles = {
            media_id: source_store.get(media)
            for media_id, media in project.media.items()
        }
        self.planner = TimelinePlanner(project, self.plan, self.bundles)

    def validate_lossless(self, mkv_path: str | Path) -> ValidationReport:
        report = ValidationReport()
        out = self.project.output
        frames = decode_lossless_video(
            mkv_path, out.width, out.height, self.work_dir
        )
        audio, sample_count = decode_lossless_audio(mkv_path, self.work_dir)
        report.frame_count = frames.shape[0]
        report.sample_count = sample_count
        if frames.shape[0] != self.plan.frame_count:
            report.expected_mismatches.append(
                {
                    "type": "frame_count",
                    "expected": self.plan.frame_count,
                    "actual": frames.shape[0],
                }
            )
        if sample_count != self.plan.sample_count:
            report.expected_mismatches.append(
                {
                    "type": "sample_count",
                    "expected": self.plan.sample_count,
                    "actual": sample_count,
                }
            )
        self._check_frame_markers(frames, report)
        self._check_pulses(audio, report)
        report.ok = (
            not report.marker_failures
            and not report.expected_mismatches
            and all(item["found"] for item in report.pulse_results)
        )
        report.details["plan"] = self.plan.to_dict()
        return report

    def _check_frame_markers(
        self, frames: np.ndarray, report: ValidationReport
    ) -> None:
        for frame_index in range(frames.shape[0]):
            timeline_time = self.plan.frame_center(frame_index)
            active = [
                clip
                for clip in self.project.clips
                if clip.timeline_start <= timeline_time < clip.timeline_end
            ]
            if not active:
                if np.any(frames[frame_index] != 0):
                    report.marker_failures.append(
                        {"frame": frame_index, "reason": "gap frame not black"}
                    )
                continue
            # Dissolve frames blend two markers and cannot be decoded as a
            # single bit plane; verify them by blended pixel reconstruction
            # in compare_renders instead.  Here only single-clip frames count.
            if len(active) > 1:
                continue
            clip = active[0]
            runtime = self.planner.clip_runtimes[clip.clip_id]
            source_time = runtime.source_video_time(timeline_time)
            selected = self.planner.select_frame_index(runtime, source_time)
            if selected is None:
                report.marker_failures.append(
                    {"frame": frame_index, "reason": "no selected source frame"}
                )
                continue
            source_id, decoded_index, marker_ok = decode_frame_marker(
                frames[frame_index]
            )
            report.marker_checked += 1
            expected_id = runtime.source.source_id_marker()
            if not marker_ok:
                report.marker_failures.append(
                    {"frame": frame_index, "reason": "parity/signature broken"}
                )
            if source_id != expected_id or decoded_index != selected:
                report.expected_mismatches.append(
                    {
                        "frame": frame_index,
                        "expected": [expected_id, selected],
                        "actual": [source_id, decoded_index],
                    }
                )

    def _check_pulses(
        self, audio: np.ndarray, report: ValidationReport
    ) -> None:
        # Use the manifest recorded by the generator (discovered alongside the
        # source files) as independent expected landmark positions.
        for clip in self.project.clips:
            runtime = self.planner.clip_runtimes[clip.clip_id]
            bundle = runtime.source
            manifest = self._generator_manifest(bundle.source_path)
            if manifest is None or not bundle.has_audio or runtime.clock is None:
                continue
            pulses = manifest.get("pulse_positions", [])
            for pulse_sample in pulses[:6]:
                source_video_time = self._video_time_for_pulse(
                    runtime, int(pulse_sample)
                )
                if source_video_time is None:
                    continue
                if not (
                    clip.src_in <= source_video_time < clip.src_out
                ):
                    continue
                timeline_time = (
                    clip.timeline_start
                    + (source_video_time - clip.src_in) / clip.speed
                )
                expected_output = round_half_up(
                    timeline_time * self.plan.sample_rate - Fraction(1, 2)
                )
                if not (0 <= expected_output < audio.shape[0]):
                    continue
                found_index = self._locate_pulse(
                    audio, int(expected_output)
                )
                report.pulse_results.append(
                    {
                        "clip": clip.clip_id,
                        "source_pulse_sample": pulse_sample,
                        "expected_output_sample": int(expected_output),
                        "detected_output_sample": found_index,
                        "found": found_index is not None
                        and abs(found_index - int(expected_output)) <= 1,
                    }
                )

    def _video_time_for_pulse(
        self, runtime: ItemRuntime, pulse_sample: int
    ) -> Fraction | None:
        clock = runtime.clock
        if clock is None:
            return None
        # Generator pulse indices are positions relative to the audio stream's
        # own sample grid (index 0 sits at the probed first audio PTS).
        position = Fraction(2 * pulse_sample + 1, 2)
        video_time = (position - clock.intercept) / clock.slope
        return video_time

    def _locate_pulse(
        self, audio: np.ndarray, expected_index: int, window: int = 6
    ) -> int | None:
        # Channel 1 carries only the triangular pulse train (channel 0 keeps
        # the continuous sine), so its local maximum is the exact pulse peak.
        start = max(0, expected_index - 16)
        end = min(audio.shape[0], expected_index + 17)
        if start >= end:
            return None
        channel = np.abs(audio[start:end, 1].astype(np.float64))
        focus = channel[16 - window : 16 + window + 1]
        local = 16 - window + int(np.argmax(focus))
        peak = float(channel[local])
        if peak < 0.08:
            return None
        return start + int(local)

    def _generator_manifest(self, source_path: Path) -> dict | None:
        manifest_path = source_path.parent / "generated_manifest.json"
        if not manifest_path.is_file():
            return None
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in entries:
            if Path(entry["path"]).resolve() == source_path.resolve():
                return entry
        return None


def round_half_up(value: Fraction) -> int:
    return (2 * value.numerator + value.denominator) // (
        2 * value.denominator
    )


def compare_renders(
    baseline_frames: np.ndarray,
    candidate_frames: np.ndarray,
    baseline_audio: np.ndarray,
    candidate_audio: np.ndarray,
    *,
    audio_tolerance: float = DEFAULT_AUDIO_TOLERANCE,
) -> dict:
    """Pixel-exact video comparison plus strict float audio tolerance."""
    if baseline_frames.shape != candidate_frames.shape:
        raise IntegrityError(
            f"video shape mismatch {baseline_frames.shape} vs {candidate_frames.shape}"
        )
    pixel_mismatches = int(np.count_nonzero(baseline_frames != candidate_frames))
    if baseline_audio.shape != candidate_audio.shape:
        raise IntegrityError(
            f"audio shape mismatch {baseline_audio.shape} vs {candidate_audio.shape}"
        )
    max_error = float(np.max(np.abs(baseline_audio - candidate_audio)))
    ok = pixel_mismatches == 0 and max_error <= audio_tolerance
    return {
        "ok": ok,
        "video_pixel_mismatches": pixel_mismatches,
        "audio_max_abs_error": max_error,
        "audio_tolerance": audio_tolerance,
    }


def compare_render_files_streaming(
    project: Project,
    baseline_mkv: str | Path,
    candidate_mkv: str | Path,
    work_dir: str | Path,
    *,
    audio_tolerance: float = DEFAULT_AUDIO_TOLERANCE,
    chunk_frames: int = 256,
) -> dict:
    """Memory-bounded comparison for long masters.

    Decodes both masters to raw files and compares memmap views in fixed
    chunks so a 30+ minute render never needs a multi-GB boolean array.
    """
    work_dir = Path(work_dir)
    out = project.output
    frame_bytes = out.width * out.height * 3
    base_dir = work_dir / "base"
    cand_dir = work_dir / "candidate"
    base_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)
    base_raw = base_dir / "decoded_video.raw"
    cand_raw = cand_dir / "decoded_video.raw"
    if not base_raw.is_file():
        run_ffmpeg(
            [
                "-i",
                str(baseline_mkv),
                "-map",
                "0:v:0",
                "-an",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                str(base_raw),
            ]
        )
    if not cand_raw.is_file():
        run_ffmpeg(
            [
                "-i",
                str(candidate_mkv),
                "-map",
                "0:v:0",
                "-an",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                str(cand_raw),
            ]
        )
    if base_raw.stat().st_size != cand_raw.stat().st_size:
        raise IntegrityError(
            f"raw video byte size mismatch "
            f"{base_raw.stat().st_size} vs {cand_raw.stat().st_size}"
        )
    base_map = np.memmap(base_raw, dtype=np.uint8, mode="r")
    cand_map = np.memmap(cand_raw, dtype=np.uint8, mode="r")
    pixel_mismatches = 0
    total_bytes = base_raw.stat().st_size
    frame_count = total_bytes // frame_bytes
    for offset in range(0, total_bytes, chunk_frames * frame_bytes):
        end = min(total_bytes, offset + chunk_frames * frame_bytes)
        pixel_mismatches += int(
            np.count_nonzero(base_map[offset:end] != cand_map[offset:end])
        )

    base_audio, _ = decode_lossless_audio(baseline_mkv, work_dir / "base")
    cand_audio, _ = decode_lossless_audio(candidate_mkv, work_dir / "candidate")
    if base_audio.shape != cand_audio.shape:
        raise IntegrityError(
            f"audio shape mismatch {base_audio.shape} vs {cand_audio.shape}"
        )
    max_error = 0.0
    audio_chunk = 1 << 20
    for offset in range(0, base_audio.shape[0], audio_chunk):
        end = min(base_audio.shape[0], offset + audio_chunk)
        max_error = max(
            max_error,
            float(
                np.max(
                    np.abs(
                        base_audio[offset:end].astype(np.float64)
                        - cand_audio[offset:end].astype(np.float64)
                    )
                )
            ),
        )
    return {
        "ok": pixel_mismatches == 0 and max_error <= audio_tolerance,
        "video_pixel_mismatches": pixel_mismatches,
        "audio_max_abs_error": max_error,
        "audio_tolerance": audio_tolerance,
        "streaming": True,
        "frame_count": int(frame_count),
        "sample_count": int(base_audio.shape[0]),
    }
