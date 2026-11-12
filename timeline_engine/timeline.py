"""Exact timeline planning, source time mapping and envelope evaluation.

One global time coordinate (seconds, rational) drives both media:

* output frame ``n`` samples the timeline at
  ``(n + 1/2) * fps_den / fps_num``;
* output sample ``k`` samples it at ``(k + 1/2) / sample_rate``;
* clip boundaries, fades and dissolve weights are evaluated at those same
  global sample times, never re-quantized per segment.

Authoritative boundary math uses :class:`fractions.Fraction`.  Vectorized numpy
math is used only for per-sample interpolation; its coefficients are derived
from the exact rationals and the resulting error stays far below the declared
``1e-6`` absolute float tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .model import (
    Anchor,
    BackgroundAudio,
    Clip,
    Project,
    Transition,
)
from .rational import ceil_div, round_half_up
from .source import SourceBundle


@dataclass(frozen=True)
class TimelinePlan:
    """Quantized output geometry derived from the project duration."""

    duration: Fraction
    frame_count: int
    normalized_duration: Fraction
    sample_count: int
    fps_num: int
    fps_den: int
    sample_rate: int
    channels: int

    def frame_center(self, index: int) -> Fraction:
        return Fraction(2 * index + 1, 2 * self.fps_num) * self.fps_den

    def sample_center(self, index: int) -> Fraction:
        return Fraction(2 * index + 1, 2 * self.sample_rate)

    def to_dict(self) -> dict:
        return {
            "project_duration": f"{self.duration.numerator}/{self.duration.denominator}",
            "frame_count": self.frame_count,
            "normalized_duration": (
                f"{self.normalized_duration.numerator}/"
                f"{self.normalized_duration.denominator}"
            ),
            "sample_count": self.sample_count,
            "fps": {"num": self.fps_num, "den": self.fps_den},
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }


def build_plan(project: Project) -> TimelinePlan:
    """Apply the exact N=ceil(D*fps) and round-half-up sample rules."""
    out = project.output
    duration_fps_num = project.duration.numerator * out.fps_num
    duration_fps_den = project.duration.denominator * out.fps_den
    frame_count = ceil_div(duration_fps_num, duration_fps_den)
    normalized = Fraction(frame_count * out.fps_den, out.fps_num)
    sample_count = round_half_up(normalized * out.sample_rate)
    return TimelinePlan(
        duration=project.duration,
        frame_count=frame_count,
        normalized_duration=normalized,
        sample_count=sample_count,
        fps_num=out.fps_num,
        fps_den=out.fps_den,
        sample_rate=out.sample_rate,
        channels=out.channels,
    )


@dataclass(frozen=True)
class ClockMap:
    """Affine map between source video-clock time and the audio sample index.

    Without anchors the source is assumed muxed on one clock:
    ``sample_position = (video_time - audio_first_pts) * sample_rate``.

    With two anchors, the unique affine mapping through both points is used.
    The slope/intercept are stored both as exact fractions (boundary checks)
    and as doubles (vectorized interpolation).
    """

    sample_rate: int
    audio_first_pts: Fraction
    slope: Fraction  # samples per video second
    intercept: Fraction  # sample index at video_time == 0

    @classmethod
    def derive(
        cls,
        anchors: tuple[Anchor, ...],
        sample_rate: int,
        audio_first_pts: Fraction,
    ) -> "ClockMap":
        if not anchors:
            return cls(
                sample_rate=sample_rate,
                audio_first_pts=audio_first_pts,
                slope=Fraction(sample_rate, 1),
                intercept=-audio_first_pts * sample_rate,
            )
        first, second = anchors
        delta_samples = second.audio_sample - first.audio_sample
        delta_time = second.video_time - first.video_time
        slope = Fraction(delta_samples) / delta_time
        intercept = Fraction(first.audio_sample) - slope * first.video_time
        return cls(
            sample_rate=sample_rate,
            audio_first_pts=audio_first_pts,
            slope=slope,
            intercept=intercept,
        )

    def sample_position(self, video_time: Fraction) -> Fraction:
        return self.slope * video_time + self.intercept

    @property
    def slope_float(self) -> float:
        return float(self.slope)

    @property
    def intercept_float(self) -> float:
        return float(self.intercept)


@dataclass
class ItemRuntime:
    """Precomputed runtime state for one clip or background audio item."""

    item_id: str
    media_id: str
    source: SourceBundle
    timeline_start: Fraction
    timeline_end: Fraction
    src_in: Fraction
    speed: Fraction
    gain: Fraction
    fade_in: Fraction
    fade_out: Fraction
    clock: ClockMap | None
    frame_pts_array: np.ndarray

    def source_video_time(self, timeline_time: Fraction) -> Fraction:
        local = timeline_time - self.timeline_start
        return self.src_in + local * self.speed

    def is_active(self, timeline_time: Fraction) -> bool:
        return self.timeline_start <= timeline_time < self.timeline_end

    def envelope(self, timeline_time: Fraction) -> Fraction:
        """Linear fade envelope in [0, 1] on the global time coordinate."""
        if not self.is_active(timeline_time):
            return Fraction(0)
        local = timeline_time - self.timeline_start
        duration = self.timeline_end - self.timeline_start
        value = Fraction(1)
        if self.fade_in > 0 and local < self.fade_in:
            value = min(value, local / self.fade_in)
        if self.fade_out > 0 and local > duration - self.fade_out:
            value = min(value, (duration - local) / self.fade_out)
        return value

    def envelope_float(self, local_times: np.ndarray) -> np.ndarray:
        duration = float(self.timeline_end - self.timeline_start)
        env = np.ones_like(local_times, dtype=np.float64)
        if self.fade_in > 0:
            env = np.minimum(env, local_times / float(self.fade_in))
        if self.fade_out > 0:
            env = np.minimum(
                env, (duration - local_times) / float(self.fade_out)
            )
        return np.clip(env, 0.0, 1.0)


@dataclass(frozen=True)
class SegmentRange:
    """Half-open global index interval owned by one render segment."""

    segment_id: int
    frame_start: int
    frame_end: int
    sample_start: int
    sample_end: int

    @property
    def frame_count(self) -> int:
        return self.frame_end - self.frame_start

    @property
    def sample_count(self) -> int:
        return self.sample_end - self.sample_start


class TimelinePlanner:
    """Builds runtimes and renders frame/audio intervals deterministically."""

    def __init__(
        self,
        project: Project,
        plan: TimelinePlan,
        bundles: dict[str, SourceBundle],
    ) -> None:
        self.project = project
        self.plan = plan
        self.bundles = bundles
        self.clip_runtimes = {
            clip.clip_id: self._build_clip_runtime(clip)
            for clip in project.clips
        }
        self.background_runtimes = {
            bg.item_id: self._build_background_runtime(bg)
            for bg in project.backgrounds
        }
        self._transition_index = self._index_transitions()

    def _clock_for(
        self,
        item_id: str,
        anchors: tuple[Anchor, ...],
        bundle: SourceBundle,
    ) -> ClockMap | None:
        if not bundle.has_audio:
            if anchors:
                raise ValueError(
                    f"{item_id}: anchors supplied but source has no audio"
                )
            return None
        assert bundle.audio_first_pts is not None and bundle.audio_sample_rate
        return ClockMap.derive(
            anchors, bundle.audio_sample_rate, bundle.audio_first_pts
        )

    def _build_clip_runtime(self, clip: Clip) -> ItemRuntime:
        bundle = self.bundles[clip.media_id]
        pts_array = np.array(
            [float(pts) for pts in bundle.frame_pts], dtype=np.float64
        )
        return ItemRuntime(
            item_id=clip.clip_id,
            media_id=clip.media_id,
            source=bundle,
            timeline_start=clip.timeline_start,
            timeline_end=clip.timeline_end,
            src_in=clip.src_in,
            speed=clip.speed,
            gain=clip.gain,
            fade_in=clip.fade_in,
            fade_out=clip.fade_out,
            clock=self._clock_for(clip.clip_id, clip.anchors, bundle),
            frame_pts_array=pts_array,
        )

    def _build_background_runtime(self, bg: BackgroundAudio) -> ItemRuntime:
        bundle = self.bundles[bg.media_id]
        return ItemRuntime(
            item_id=bg.item_id,
            media_id=bg.media_id,
            source=bundle,
            timeline_start=bg.timeline_start,
            timeline_end=bg.timeline_end,
            src_in=bg.src_in,
            speed=bg.speed,
            gain=bg.gain,
            fade_in=bg.fade_in,
            fade_out=bg.fade_out,
            clock=self._clock_for(bg.item_id, bg.anchors, bundle),
            frame_pts_array=np.array(
                [float(pts) for pts in bundle.frame_pts], dtype=np.float64
            ),
        )

    def _index_transitions(self) -> dict[tuple[str, str], Transition]:
        index: dict[tuple[str, str], Transition] = {}
        for transition in self.project.transitions:
            index[(transition.from_clip, transition.to_clip)] = transition
        return index

    def _dissolve_weight(
        self, clip: Clip, timeline_time: Fraction
    ) -> Fraction:
        """Contribution weight of ``clip`` from explicit transitions only."""
        weight = Fraction(1)
        for (from_id, to_id), transition in self._transition_index.items():
            if transition.kind == "cut":
                continue
            if clip.clip_id not in (from_id, to_id):
                continue
            first = self.project_clips[from_id]
            second = self.project_clips[to_id]
            # The dissolve occupies the final declared-overlap seconds of the
            # shared overlap region on the single global time coordinate.
            dissolve_start = (
                min(first.timeline_end, second.timeline_end) - transition.overlap
            )
            dissolve_end = dissolve_start + transition.overlap
            if dissolve_start <= timeline_time < dissolve_end:
                phase = (timeline_time - dissolve_start) / transition.overlap
                if clip.clip_id == from_id:
                    weight = min(weight, 1 - phase)
                else:
                    weight = min(weight, phase)
        return weight

    @property
    def project_clips(self) -> dict[str, Clip]:
        return {clip.clip_id: clip for clip in self.project.clips}

    def select_frame_index(
        self, runtime: ItemRuntime, source_time: Fraction
    ) -> int | None:
        """Nearest frame whose display PTS <= source_time (VFR-safe)."""
        bundle = runtime.source
        if not bundle.has_video:
            return None
        if source_time < runtime.src_in or source_time >= (
            runtime.src_in
            + (runtime.timeline_end - runtime.timeline_start) * runtime.speed
        ):
            return None
        # Exact boundary-aware binary search on the rational PTS list.
        pts_list = bundle.frame_pts
        low, high = 0, len(pts_list)
        while low < high:
            middle = (low + high) // 2
            if pts_list[middle] <= source_time:
                low = middle + 1
            else:
                high = middle
        index = low - 1
        if index < 0:
            return None
        return index

    def render_frame(self, frame_index: int) -> np.ndarray:
        """Render one output frame at its exact global center time."""
        out = self.project.output
        timeline_time = self.plan.frame_center(frame_index)
        canvas = np.zeros(
            (out.height, out.width, 3), dtype=np.uint8
        )
        accum = np.zeros_like(canvas, dtype=np.float64)
        weight_sum = 0.0
        for clip in self.project.clips:
            runtime = self.clip_runtimes[clip.clip_id]
            if not runtime.is_active(timeline_time):
                continue
            source_time = runtime.source_video_time(timeline_time)
            selected = self.select_frame_index(runtime, source_time)
            if selected is None:
                continue
            weight = float(self._dissolve_weight(clip, timeline_time))
            fade = float(runtime.envelope(timeline_time))
            factor = weight * fade
            frame = runtime.source.read_frame(selected).astype(np.float64)
            accum += frame * factor
            weight_sum += factor
        if weight_sum > 0.0:
            # Weights of two overlapping clips sum to 1; gaps stay black.
            if weight_sum > 1.0 + 1e-12:
                accum /= weight_sum
            canvas = np.floor(accum + 0.5).astype(np.uint8)
        return canvas

    def _render_item_audio(
        self,
        runtime: ItemRuntime,
        timeline_times: np.ndarray,
        apply_transition: bool,
        transition_weights: np.ndarray | None,
    ) -> np.ndarray:
        """Resample one item onto output sample centers via linear interp."""
        count = timeline_times.shape[0]
        out = np.zeros((count, self.plan.channels), dtype=np.float64)
        if runtime.clock is None:
            return out
        local = timeline_times - float(runtime.timeline_start)
        active = (timeline_times >= float(runtime.timeline_start)) & (
            timeline_times < float(runtime.timeline_end)
        )
        if not np.any(active):
            return out
        source_times = (
            float(runtime.src_in) + local * float(runtime.speed)
        )
        positions = (
            runtime.clock.slope_float * source_times
            + runtime.clock.intercept_float
        )
        audio = runtime.source.read_audio()
        total_samples = audio.shape[0]
        lower = np.floor(positions).astype(np.int64)
        fraction = positions - lower
        valid = active & (lower >= 0) & (lower + 1 < total_samples)
        if not np.any(valid):
            return out
        left = np.clip(lower, 0, total_samples - 1)
        right = np.clip(lower + 1, 0, total_samples - 1)
        mixed = (
            audio[left] * (1.0 - fraction)[:, None]
            + audio[right] * fraction[:, None]
        )
        envelope = runtime.envelope_float(local)
        if apply_transition and transition_weights is not None:
            envelope = envelope * transition_weights
        out[valid] = mixed[valid] * envelope[valid, None] * float(runtime.gain)
        return out

    def _transition_weights_for(
        self, clip_id: str, timeline_times: np.ndarray
    ) -> np.ndarray:
        weights = np.ones(timeline_times.shape[0], dtype=np.float64)
        for (from_id, to_id), transition in self._transition_index.items():
            if transition.kind == "cut" or clip_id not in (from_id, to_id):
                continue
            first = self.project_clips[from_id]
            second = self.project_clips[to_id]
            dissolve_start = (
                min(first.timeline_end, second.timeline_end) - transition.overlap
            )
            dissolve_end = dissolve_start + transition.overlap
            start = float(dissolve_start)
            end = float(dissolve_end)
            inside = (timeline_times >= start) & (timeline_times < end)
            if not np.any(inside):
                continue
            phase = (timeline_times[inside] - start) / float(transition.overlap)
            if clip_id == from_id:
                weights[inside] = np.minimum(weights[inside], 1.0 - phase)
            else:
                weights[inside] = np.minimum(weights[inside], phase)
        return weights

    def render_audio(self, sample_start: int, sample_end: int) -> np.ndarray:
        """Render the half-open output sample interval as float32 stereo."""
        count = sample_end - sample_start
        if count <= 0:
            return np.zeros((0, self.plan.channels), dtype=np.float32)
        indices = np.arange(sample_start, sample_end, dtype=np.float64)
        timeline_times = (indices + 0.5) / self.plan.sample_rate
        mix = np.zeros((count, self.plan.channels), dtype=np.float64)

        for clip in self.project.clips:
            runtime = self.clip_runtimes[clip.clip_id]
            weights = self._transition_weights_for(clip.clip_id, timeline_times)
            mix += self._render_item_audio(
                runtime,
                timeline_times,
                apply_transition=True,
                transition_weights=weights,
            )
        for background in self.project.backgrounds:
            runtime = self.background_runtimes[background.item_id]
            mix += self._render_item_audio(
                runtime,
                timeline_times,
                apply_transition=False,
                transition_weights=None,
            )

        # Declared overflow policy: hard clip to the float32 work range.
        clipped = np.clip(mix, -1.0, 1.0)
        return clipped.astype(np.float32, copy=False)

    def segment_times(
        self, segment: SegmentRange
    ) -> tuple[np.ndarray, np.ndarray]:
        frame_indices = np.arange(segment.frame_start, segment.frame_end)
        sample_indices = np.arange(segment.sample_start, segment.sample_end)
        frame_times = (
            (frame_indices.astype(np.float64) + 0.5)
            * self.plan.fps_den
            / self.plan.fps_num
        )
        sample_times = (
            (sample_indices.astype(np.float64) + 0.5) / self.plan.sample_rate
        )
        return frame_times, sample_times


def plan_segments(
    plan: TimelinePlan, segment_frames: int
) -> list[SegmentRange]:
    """Partition global indices with no duplicated/missing boundary ownership.

    Sample cuts are derived from the same frame centers via exact half-open
    rounding, so frame/sample boundaries stay on one coordinate system.
    """
    if segment_frames <= 0:
        raise ValueError("segment_frames must be positive")
    segments: list[SegmentRange] = []
    frame_start = 0
    segment_id = 0
    while frame_start < plan.frame_count:
        frame_end = min(frame_start + segment_frames, plan.frame_count)
        if frame_end >= plan.frame_count:
            sample_end = plan.sample_count
        else:
            # Boundary is the exact midpoint between the centers of the last
            # owned frame and the first frame of the next segment.  Samples
            # whose centers fall in [boundary_prev, boundary_next) are owned.
            boundary = (
                plan.frame_center(frame_end - 1)
                + plan.frame_center(frame_end)
            ) / 2
            # sample k belongs iff (k+1/2)/R < boundary  <=>
            # k < boundary*R - 1/2 ; first excluded index = ceil(boundary*R-1/2)
            threshold = boundary * plan.sample_rate - Fraction(1, 2)
            sample_end = ceil_div(threshold.numerator, threshold.denominator)
        if segment_id == 0:
            sample_start = 0
        if segment_id > 0:
            sample_start = segments[-1].sample_end
        segments.append(
            SegmentRange(
                segment_id=segment_id,
                frame_start=frame_start,
                frame_end=frame_end,
                sample_start=sample_start,
                sample_end=sample_end,
            )
        )
        frame_start = frame_end
        segment_id += 1
    if segments and segments[-1].sample_end != plan.sample_count:
        last = segments[-1]
        segments[-1] = SegmentRange(
            segment_id=last.segment_id,
            frame_start=last.frame_start,
            frame_end=last.frame_end,
            sample_start=last.sample_start,
            sample_end=plan.sample_count,
        )
    return segments
