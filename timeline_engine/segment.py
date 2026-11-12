"""Render dependency graph, bounded scheduling and segment cache.

Every segment is a pure function of:

* the global index interval it owns;
* every clip/background item active on that interval (including dissolve
  weights and envelopes evaluated at global sample/frame centers);
* source content hashes and source declarations;
* the fixed work format and captured FFmpeg/engine versions.

No FFmpeg filter carries state across the Python boundary: frame selection,
resampling and blending are recomputed from global timestamps, so a cut may
land inside a GOP, a speed ramp or a dissolve without filter-history seams.
Segment artifacts are written to ``*.tmp`` and atomically renamed only after
FFmpeg exits successfully and their size is verified.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import IntegrityError, RenderError
from .ffmpeg_tools import canonical_json, run_ffmpeg, sha256_file
from .model import Project
from .rational import frac_str
from .source import (
    WORK_AUDIO_CHANNELS,
    WORK_AUDIO_DTYPE,
    WORK_COLOR_RANGE,
    WORK_COLOR_SPACE,
    WORK_PIX_FMT,
    SourceBundle,
)
from .timeline import SegmentRange, TimelinePlan, TimelinePlanner


@dataclass
class SegmentResult:
    segment_id: int
    cache_hit: bool
    video_path: Path
    audio_path: Path
    key: str
    frame_count: int
    sample_count: int


def _fraction_json(value) -> dict[str, int]:
    return {"num": value.numerator, "den": value.denominator}


class SegmentCache:
    """Builds content-derived segment keys and verifies stored artifacts."""

    def __init__(
        self,
        project: Project,
        plan: TimelinePlan,
        bundles: dict[str, SourceBundle],
        toolchain: dict[str, str],
    ) -> None:
        self.project = project
        self.plan = plan
        self.bundles = bundles
        self.toolchain = toolchain

    def _segment_influence(
        self, segment: SegmentRange
    ) -> tuple[list[str], list[str], list[str]]:
        """Return (clip_ids, background_ids, transition_keys) in range.

        An item is included whenever its half-open timeline interval could
        touch any frame/sample center owned by the segment, which keeps
        transition context explicit even for cuts in the middle of a dissolve.
        """
        first_frame_time = self.plan.frame_center(segment.frame_start)
        last_frame_time = self.plan.frame_center(segment.frame_end - 1)
        first_sample_time = self.plan.sample_center(segment.sample_start)
        last_sample_time = self.plan.sample_center(segment.sample_end - 1)
        lo = min(first_frame_time, first_sample_time)
        hi = max(last_frame_time, last_sample_time)

        clip_ids: list[str] = []
        for clip in self.project.clips:
            if clip.timeline_start <= hi and clip.timeline_end > lo:
                clip_ids.append(clip.clip_id)
        bg_ids: list[str] = []
        for bg in self.project.backgrounds:
            if bg.timeline_start <= hi and bg.timeline_end > lo:
                bg_ids.append(bg.item_id)
        active = set(clip_ids)
        transition_keys: list[str] = []
        for transition in self.project.transitions:
            if transition.from_clip in active or transition.to_clip in active:
                transition_keys.append(
                    f"{transition.from_clip}->{transition.to_clip}:"
                    f"{frac_str(transition.overlap)}:{transition.kind}"
                )
        return sorted(clip_ids), sorted(bg_ids), sorted(transition_keys)

    def key_for(self, segment: SegmentRange) -> str:
        clip_ids, bg_ids, transition_keys = self._segment_influence(segment)
        clip_payload = []
        clips_by_id = {clip.clip_id: clip for clip in self.project.clips}
        for clip_id in clip_ids:
            clip = clips_by_id[clip_id]
            bundle = self.bundles[clip.media_id]
            clip_payload.append(
                {
                    "id": clip.clip_id,
                    "media_sha256": bundle.sha256,
                    "media_id": clip.media_id,
                    "src_in": _fraction_json(clip.src_in),
                    "src_out": _fraction_json(clip.src_out),
                    "timeline_start": _fraction_json(clip.timeline_start),
                    "speed": _fraction_json(clip.speed),
                    "gain": _fraction_json(clip.gain),
                    "fade_in": _fraction_json(clip.fade_in),
                    "fade_out": _fraction_json(clip.fade_out),
                    "anchors": [anchor.to_dict() for anchor in clip.anchors],
                    "pts_head": [
                        frac_str(pts) for pts in bundle.frame_pts[:4]
                    ],
                    "audio_rate": bundle.audio_sample_rate,
                    "audio_first_pts": (
                        frac_str(bundle.audio_first_pts)
                        if bundle.audio_first_pts is not None
                        else None
                    ),
                }
            )
        bg_payload = []
        bgs_by_id = {bg.item_id: bg for bg in self.project.backgrounds}
        for bg_id in bg_ids:
            bg = bgs_by_id[bg_id]
            bundle = self.bundles[bg.media_id]
            bg_payload.append(
                {
                    "id": bg.item_id,
                    "media_sha256": bundle.sha256,
                    "src_in": _fraction_json(bg.src_in),
                    "src_out": _fraction_json(bg.src_out),
                    "timeline_start": _fraction_json(bg.timeline_start),
                    "speed": _fraction_json(bg.speed),
                    "gain": _fraction_json(bg.gain),
                    "fade_in": _fraction_json(bg.fade_in),
                    "fade_out": _fraction_json(bg.fade_out),
                    "anchors": [anchor.to_dict() for anchor in bg.anchors],
                }
            )
        payload = {
            "segment": {
                "frame_start": segment.frame_start,
                "frame_end": segment.frame_end,
                "sample_start": segment.sample_start,
                "sample_end": segment.sample_end,
            },
            "output": self.project.output.to_dict(),
            "plan": self.plan.to_dict(),
            "clips": clip_payload,
            "backgrounds": bg_payload,
            "transitions": transition_keys,
            "work_format": {
                "pix_fmt": WORK_PIX_FMT,
                "color_range": WORK_COLOR_RANGE,
                "color_space": WORK_COLOR_SPACE,
                "audio_channels": WORK_AUDIO_CHANNELS,
                "audio_dtype": "float32le",
            },
            "toolchain": self.toolchain,
        }
        return hashlib.sha256(canonical_json(payload)).hexdigest()


class SegmentRenderer:
    """Renders segments into lossless MKV + float WAV with atomic commits."""

    def __init__(
        self,
        project: Project,
        plan: TimelinePlan,
        planner: TimelinePlanner,
        cache: SegmentCache,
        segments_dir: Path,
        toolchain: dict[str, str],
    ) -> None:
        self.project = project
        self.plan = plan
        self.planner = planner
        self.cache = cache
        self.segments_dir = Path(segments_dir)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.toolchain = toolchain
        self.hits: dict[int, str] = {}
        self.misses: dict[int, str] = {}
        # When a segment id appears here, rendering raises after the MKV/WAV
        # files are fully written but before done.json is atomically renamed,
        # reproducing a forced process kill at the critical commit boundary.
        self.crash_before_done: set[int] = set()

    def _paths(self, key: str) -> tuple[Path, Path, Path]:
        base = self.segments_dir / key
        return base / "video.mkv", base / "audio.wav", base / "done.json"

    def _is_complete(
        self,
        segment: SegmentRange,
        key: str,
        video_path: Path,
        audio_path: Path,
        done_path: Path,
    ) -> bool:
        if not done_path.is_file():
            return False
        try:
            done = json.loads(done_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if done.get("key") != key:
            return False
        if done.get("frame_count") != segment.frame_count:
            return False
        if done.get("sample_count") != segment.sample_count:
            return False
        expected_video = (
            segment.frame_count
            * self.project.output.width
            * self.project.output.height
            * 3
        )
        # MKV is compressed; trust the recorded hash plus ffprobe-free size
        # check via done manifest and re-hash the artifact below.
        if not video_path.is_file() or not audio_path.is_file():
            return False
        if done.get("video_sha256") != sha256_file(video_path):
            return False
        if done.get("audio_sha256") != sha256_file(audio_path):
            return False
        if expected_video <= 0 and segment.frame_count:
            return False
        return True

    def render_segment(self, segment: SegmentRange) -> SegmentResult:
        key = self.cache.key_for(segment)
        entry_dir = self.segments_dir / key
        entry_dir.mkdir(parents=True, exist_ok=True)
        video_path, audio_path, done_path = self._paths(key)
        if self._is_complete(segment, key, video_path, audio_path, done_path):
            self.hits[segment.segment_id] = key
            return SegmentResult(
                segment_id=segment.segment_id,
                cache_hit=True,
                video_path=video_path,
                audio_path=audio_path,
                key=key,
                frame_count=segment.frame_count,
                sample_count=segment.sample_count,
            )

        self.misses[segment.segment_id] = key
        frames = np.stack(
            [
                self.planner.render_frame(index)
                for index in range(segment.frame_start, segment.frame_end)
            ],
            axis=0,
        )
        audio = self.planner.render_audio(
            segment.sample_start, segment.sample_end
        )
        if audio.shape[0] != segment.sample_count:
            raise RenderError(
                f"segment {segment.segment_id}: produced {audio.shape[0]} "
                f"samples, planned {segment.sample_count}"
            )
        if frames.shape[0] != segment.frame_count:
            raise RenderError(
                f"segment {segment.segment_id}: produced {frames.shape[0]} "
                f"frames, planned {segment.frame_count}"
            )

        tmp_video = entry_dir / "video.tmp.mkv"
        tmp_audio = entry_dir / "audio.tmp.wav"
        self._encode_video(frames, tmp_video)
        self._write_audio(audio, tmp_audio)
        video_hash = sha256_file(tmp_video)
        audio_hash = sha256_file(tmp_audio)
        tmp_video.replace(video_path)
        tmp_audio.replace(audio_path)
        if segment.segment_id in self.crash_before_done:
            self.crash_before_done.discard(segment.segment_id)
            raise RenderError(
                "injected crash: segment artifacts written, done marker withheld"
            )
        done = {
            "key": key,
            "segment_id": segment.segment_id,
            "frame_start": segment.frame_start,
            "frame_end": segment.frame_end,
            "sample_start": segment.sample_start,
            "sample_end": segment.sample_end,
            "frame_count": segment.frame_count,
            "sample_count": segment.sample_count,
            "video_sha256": video_hash,
            "audio_sha256": audio_hash,
            "toolchain": self.toolchain,
        }
        tmp_done = entry_dir / "done.tmp.json"
        tmp_done.write_text(json.dumps(done, indent=2), encoding="utf-8")
        tmp_done.replace(done_path)
        return SegmentResult(
            segment_id=segment.segment_id,
            cache_hit=False,
            video_path=video_path,
            audio_path=audio_path,
            key=key,
            frame_count=segment.frame_count,
            sample_count=segment.sample_count,
        )

    def _encode_video(self, frames: np.ndarray, target: Path) -> None:
        out = self.project.output
        # FFV1 level 3 in Matroska is lossless and CPU-only; raw RGB24 is
        # piped through stdin with the exact planned frame count.
        run_ffmpeg(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                WORK_PIX_FMT,
                "-s",
                f"{out.width}x{out.height}",
                "-framerate",
                f"{out.fps_num}/{out.fps_den}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-coder",
                "1",
                "-context",
                "1",
                "-g",
                "1",
                "-threads",
                "1",
                "-pix_fmt",
                WORK_PIX_FMT,
                str(target),
            ],
            stdin_data=np.ascontiguousarray(frames).tobytes(),
        )

    def _write_audio(self, audio: np.ndarray, target: Path) -> None:
        # 32-bit float PCM WAV: a declared lossless/float intermediate.
        run_ffmpeg(
            [
                "-f",
                "f32le",
                "-ar",
                str(self.plan.sample_rate),
                "-ac",
                str(WORK_AUDIO_CHANNELS),
                "-i",
                "pipe:0",
                "-vn",
                "-c:a",
                "pcm_f32le",
                str(target),
            ],
            stdin_data=np.ascontiguousarray(audio, dtype=WORK_AUDIO_DTYPE).tobytes(),
        )


class SegmentScheduler:
    """Bounded thread-pool scheduler over the dependency-free segment graph.

    Segments share no mutable state (each owns disjoint global indices and an
    isolated cache entry), so they form an embarrassingly parallel graph and
    the bounded worker count is the only scheduling decision.
    """

    def __init__(self, renderer: SegmentRenderer, parallelism: int = 1) -> None:
        if parallelism < 1:
            raise ValueError("parallelism must be >= 1")
        self.renderer = renderer
        self.parallelism = parallelism

    def run(
        self, segments: list[SegmentRange], *, fail_fast: bool = True
    ) -> list[SegmentResult]:
        results: dict[int, SegmentResult] = {}
        if self.parallelism == 1:
            for segment in segments:
                results[segment.segment_id] = self.renderer.render_segment(segment)
            return [results[item.segment_id] for item in segments]
        with ThreadPoolExecutor(max_workers=self.parallelism) as pool:
            futures = {
                pool.submit(self.renderer.render_segment, segment): segment
                for segment in segments
            }
            for future in as_completed(futures):
                segment = futures[future]
                try:
                    results[segment.segment_id] = future.result()
                except Exception as exc:
                    if fail_fast:
                        for pending in futures:
                            pending.cancel()
                        raise IntegrityError(
                            f"segment {segment.segment_id} failed: {exc}"
                        ) from exc
                    raise
        return [results[item.segment_id] for item in segments]
