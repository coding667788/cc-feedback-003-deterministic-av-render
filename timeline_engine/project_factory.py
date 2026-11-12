"""Helpers that build valid projects on top of generated synthetic sources."""

from __future__ import annotations

from fractions import Fraction

from .generator import GeneratedSpec
from .model import (
    BackgroundAudio,
    Clip,
    MediaRef,
    OutputSpec,
    Project,
    Transition,
    validate_project,
)


def media_from_spec(spec: GeneratedSpec) -> MediaRef:
    color_range = "tv" if spec.path.suffix.lower() == ".mp4" else "pc"
    return MediaRef(
        media_id=spec.kind,
        path=str(spec.path.resolve()),
        pixel_format="yuv420p" if color_range == "tv" else "rgb24",
        color_range=color_range,
        color_space="bt709",
    )


def build_demo_project(
    specs: list[GeneratedSpec],
    *,
    width: int = 256,
    height: int = 144,
) -> Project:
    """Project covering VFR, B-frame in-point, 1/3 boundary, speeds, dissolve."""
    by_kind = {spec.kind: spec for spec in specs}
    media = {spec.kind: media_from_spec(spec) for spec in specs}
    output = OutputSpec(
        width=width,
        height=height,
        fps_num=30000,
        fps_den=1001,
        sample_rate=48000,
        channels=2,
    )

    cfr25 = by_kind["cfr25"]
    cfr2997 = by_kind["cfr2997"]
    vfr = by_kind["vfr"]
    bframes = by_kind["bframes"]
    videoonly = by_kind["videoonly"]

    clips = [
        Clip(
            clip_id="c1_cfr25",
            media_id="cfr25",
            # Non-keyframe-friendly in-point expressed as an exact fraction.
            src_in=Fraction(123, 1000) + Fraction(7, 25),
            src_out=Fraction(123, 1000) + Fraction(47, 25),
            timeline_start=Fraction(0),
            speed=Fraction(1),
            fade_in=Fraction(1, 3),
        ),
        Clip(
            clip_id="c2_2997_fast",
            media_id="cfr2997",
            src_in=Fraction(77, 1000) + Fraction(3, 10),
            src_out=Fraction(77, 1000) + Fraction(3, 10) + Fraction(3, 2),
            # 3/2 speed: 1.5 source seconds per output second.
            timeline_start=Fraction(1, 1),
            speed=Fraction(3, 2),
        ),
        Clip(
            clip_id="c3_vfr_slow",
            media_id="vfr",
            src_in=vfr.first_video_pts + Fraction(1, 10),
            src_out=vfr.first_video_pts + Fraction(1, 10) + Fraction(2),
            # 2/3 speed: output duration 3 s for 2 source seconds.
            timeline_start=Fraction(2, 1),
            speed=Fraction(2, 3),
        ),
        Clip(
            clip_id="c4_bframes",
            media_id="bframes",
            # Deliberately not aligned to a 24 fps keyframe/GOP boundary.
            src_in=Fraction(333, 1000) + Fraction(5, 24) + Fraction(1, 48),
            src_out=Fraction(333, 1000) + Fraction(35, 24),
            timeline_start=Fraction(4, 1),
            speed=Fraction(1),
        ),
        Clip(
            clip_id="c5_videoonly",
            media_id="videoonly",
            src_in=Fraction(1, 30),
            src_out=Fraction(21, 30),
            # Gap of 1/3 s before this clip.
            timeline_start=Fraction(6, 1) + Fraction(1, 3),
            speed=Fraction(1),
        ),
    ]
    transitions = [
        Transition(
            from_clip="c1_cfr25",
            to_clip="c2_2997_fast",
            overlap=Fraction(1, 3),
        ),
        Transition(
            from_clip="c3_vfr_slow",
            to_clip="c4_bframes",
            overlap=Fraction(1, 2),
        ),
    ]
    backgrounds = [
        BackgroundAudio(
            item_id="bg_cfr25",
            media_id="cfr25",
            src_in=cfr25.first_video_pts + Fraction(1, 10),
            src_out=cfr25.first_video_pts + Fraction(1, 10) + Fraction(2),
            timeline_start=Fraction(1, 3),
            speed=Fraction(1),
            gain=Fraction(1, 2),
            fade_in=Fraction(1, 6),
            fade_out=Fraction(1, 6),
        )
    ]
    duration = Fraction(8)
    project = Project(
        output=output,
        duration=duration,
        media=media,
        clips=clips,
        transitions=transitions,
        backgrounds=backgrounds,
    )
    validate_project(project)
    return project


def build_invalid_project(specs: list[GeneratedSpec]) -> Project:
    """A triple-overlap project that must be rejected before rendering."""
    spec = specs[0]
    project = Project(
        output=OutputSpec(
            width=256,
            height=144,
            fps_num=30000,
            fps_den=1001,
            sample_rate=48000,
            channels=2,
        ),
        duration=Fraction(3),
        media={spec.kind: media_from_spec(spec)},
        clips=[
            Clip(
                clip_id="a",
                media_id=spec.kind,
                src_in=spec.first_video_pts,
                src_out=spec.first_video_pts + Fraction(1),
                timeline_start=Fraction(0),
                speed=Fraction(1),
            ),
            Clip(
                clip_id="b",
                media_id=spec.kind,
                src_in=spec.first_video_pts,
                src_out=spec.first_video_pts + Fraction(1),
                timeline_start=Fraction(1, 2),
                speed=Fraction(1),
            ),
            Clip(
                clip_id="c",
                media_id=spec.kind,
                src_in=spec.first_video_pts,
                src_out=spec.first_video_pts + Fraction(1),
                timeline_start=Fraction(3, 4),
                speed=Fraction(1),
            ),
        ],
        transitions=[],
        backgrounds=[],
    )
    return project


def build_anchor_project(specs: list[GeneratedSpec]) -> Project:
    """Long 30+ minute project using dual-anchor clock calibration."""
    from .model import Anchor

    by_kind = {spec.kind: spec for spec in specs}
    cfr25 = by_kind["cfr25"]
    output = OutputSpec(
        width=256,
        height=144,
        fps_num=30000,
        fps_den=1001,
        sample_rate=48000,
        channels=2,
    )
    # Reuse the same short source repeatedly; cumulative timeline is 32 min.
    total = Fraction(32 * 60)
    clip_duration = Fraction(2)
    source_window = Fraction(2)
    windows = (
        cfr25.first_video_pts + Fraction(0),
        cfr25.first_video_pts + Fraction(4, 10),
    )
    clips = []
    index = 0
    for cursor_index in range(total // clip_duration):
        cursor = Fraction(cursor_index * 2)
        src_in = windows[index % len(windows)]
        src_out = src_in + source_window
        anchors = (
            Anchor(audio_sample=2400, video_time=src_in + Fraction(1, 100)),
            Anchor(
                audio_sample=98400,
                video_time=src_in + Fraction(1, 100) + Fraction(99, 50),
            ),
        )
        clips.append(
            Clip(
                clip_id=f"long_{index:04d}",
                media_id="cfr25",
                src_in=src_in,
                src_out=src_out,
                timeline_start=cursor,
                speed=Fraction(1),
                anchors=anchors,
            )
        )
        index += 1
    project = Project(
        output=output,
        duration=total,
        media={"cfr25": media_from_spec(cfr25)},
        clips=clips,
        transitions=[],
        backgrounds=[],
    )
    validate_project(project)
    return project
