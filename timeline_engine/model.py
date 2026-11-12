"""In-memory project model and JSON (de)serialization.

Schema version ``1.0``.  Every temporal field is an exact rational; see
:mod:`timeline_engine.rational`.  The loader performs structural validation
and rejects ambiguous or contradictory projects before any render starts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from .errors import ProjectValidationError
from .rational import to_fraction, to_json

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Anchor:
    """One ``source audio sample position -> source video time`` point."""

    audio_sample: int
    video_time: Fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_sample": self.audio_sample,
            "video_time": to_json(self.video_time),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Anchor":
        if not isinstance(data.get("audio_sample"), int) or isinstance(
            data.get("audio_sample"), bool
        ):
            raise ProjectValidationError("anchor.audio_sample must be an integer")
        return cls(
            audio_sample=data["audio_sample"],
            video_time=to_fraction(data["video_time"], field="anchor.video_time"),
        )


@dataclass(frozen=True)
class MediaRef:
    """A source file plus its declared color/pixel declaration."""

    media_id: str
    path: str
    pixel_format: str | None = None
    color_range: str | None = None  # "tv" (limited) or "pc" (full)
    color_space: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.media_id, "path": self.path}
        if self.pixel_format is not None:
            data["pixel_format"] = self.pixel_format
        if self.color_range is not None:
            data["color_range"] = self.color_range
        if self.color_space is not None:
            data["color_space"] = self.color_space
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MediaRef":
        missing = {"id", "path"} - data.keys()
        if missing:
            raise ProjectValidationError(f"media entry missing fields: {sorted(missing)}")
        color_range = data.get("color_range")
        if color_range not in (None, "tv", "pc"):
            raise ProjectValidationError(
                f"media {data['id']}: color_range must be 'tv' or 'pc'"
            )
        return cls(
            media_id=str(data["id"]),
            path=str(data["path"]),
            pixel_format=data.get("pixel_format"),
            color_range=color_range,
            color_space=data.get("color_space"),
        )


@dataclass(frozen=True)
class Clip:
    """A video timeline item.

    Source interval ``[src_in, src_out)`` is placed at ``timeline_start`` and
    played at rational ``speed`` (timeline seconds per source second =
    1/speed).  Optional linear fades apply to audio gain and the clip's video
    contribution weight.
    """

    clip_id: str
    media_id: str
    src_in: Fraction
    src_out: Fraction
    timeline_start: Fraction
    speed: Fraction
    track: int = 0
    gain: Fraction = Fraction(1)
    fade_in: Fraction = Fraction(0)
    fade_out: Fraction = Fraction(0)
    anchors: tuple[Anchor, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> Fraction:
        return (self.src_out - self.src_in) / self.speed

    @property
    def timeline_end(self) -> Fraction:
        return self.timeline_start + self.duration

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.clip_id,
            "media_id": self.media_id,
            "src_in": to_json(self.src_in),
            "src_out": to_json(self.src_out),
            "timeline_start": to_json(self.timeline_start),
            "speed": to_json(self.speed),
            "track": self.track,
        }
        if self.gain != 1:
            data["gain"] = to_json(self.gain)
        if self.fade_in:
            data["fade_in"] = to_json(self.fade_in)
        if self.fade_out:
            data["fade_out"] = to_json(self.fade_out)
        if self.anchors:
            data["anchors"] = [anchor.to_dict() for anchor in self.anchors]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Clip":
        for required in (
            "id",
            "media_id",
            "src_in",
            "src_out",
            "timeline_start",
            "speed",
        ):
            if required not in data:
                raise ProjectValidationError(f"clip missing field '{required}'")
        anchors = tuple(Anchor.from_dict(item) for item in data.get("anchors", []))
        if len(anchors) not in (0, 2):
            raise ProjectValidationError(
                f"clip {data['id']}: calibration requires exactly two anchors"
            )
        return cls(
            clip_id=str(data["id"]),
            media_id=str(data["media_id"]),
            src_in=to_fraction(data["src_in"], field="clip.src_in"),
            src_out=to_fraction(data["src_out"], field="clip.src_out"),
            timeline_start=to_fraction(
                data["timeline_start"], field="clip.timeline_start"
            ),
            speed=to_fraction(data["speed"], field="clip.speed"),
            track=int(data.get("track", 0)),
            gain=to_fraction(data.get("gain", 1), field="clip.gain"),
            fade_in=to_fraction(data.get("fade_in", 0), field="clip.fade_in"),
            fade_out=to_fraction(data.get("fade_out", 0), field="clip.fade_out"),
            anchors=anchors,
        )


@dataclass(frozen=True)
class Transition:
    """Linear dissolve over the overlap of two same-track clips."""

    from_clip: str
    to_clip: str
    overlap: Fraction
    kind: str = "dissolve"

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_clip": self.from_clip,
            "to_clip": self.to_clip,
            "overlap": to_json(self.overlap),
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transition":
        for required in ("from_clip", "to_clip", "overlap"):
            if required not in data:
                raise ProjectValidationError(f"transition missing field '{required}'")
        kind = data.get("kind", "dissolve")
        if kind not in ("dissolve", "cut"):
            raise ProjectValidationError(f"transition kind {kind!r} unsupported")
        return cls(
            from_clip=str(data["from_clip"]),
            to_clip=str(data["to_clip"]),
            overlap=to_fraction(data["overlap"], field="transition.overlap"),
            kind=kind,
        )


@dataclass(frozen=True)
class BackgroundAudio:
    """Independent additive audio item (no video contribution)."""

    item_id: str
    media_id: str
    src_in: Fraction
    src_out: Fraction
    timeline_start: Fraction
    speed: Fraction
    gain: Fraction = Fraction(1)
    fade_in: Fraction = Fraction(0)
    fade_out: Fraction = Fraction(0)
    anchors: tuple[Anchor, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> Fraction:
        return (self.src_out - self.src_in) / self.speed

    @property
    def timeline_end(self) -> Fraction:
        return self.timeline_start + self.duration

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.item_id,
            "media_id": self.media_id,
            "src_in": to_json(self.src_in),
            "src_out": to_json(self.src_out),
            "timeline_start": to_json(self.timeline_start),
            "speed": to_json(self.speed),
        }
        if self.gain != 1:
            data["gain"] = to_json(self.gain)
        if self.fade_in:
            data["fade_in"] = to_json(self.fade_in)
        if self.fade_out:
            data["fade_out"] = to_json(self.fade_out)
        if self.anchors:
            data["anchors"] = [anchor.to_dict() for anchor in self.anchors]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackgroundAudio":
        for required in (
            "id",
            "media_id",
            "src_in",
            "src_out",
            "timeline_start",
            "speed",
        ):
            if required not in data:
                raise ProjectValidationError(
                    f"background audio missing field '{required}'"
                )
        anchors = tuple(Anchor.from_dict(item) for item in data.get("anchors", []))
        if len(anchors) not in (0, 2):
            raise ProjectValidationError(
                f"background {data['id']}: calibration requires exactly two anchors"
            )
        return cls(
            item_id=str(data["id"]),
            media_id=str(data["media_id"]),
            src_in=to_fraction(data["src_in"], field="bg.src_in"),
            src_out=to_fraction(data["src_out"], field="bg.src_out"),
            timeline_start=to_fraction(
                data["timeline_start"], field="bg.timeline_start"
            ),
            speed=to_fraction(data["speed"], field="bg.speed"),
            gain=to_fraction(data.get("gain", 1), field="bg.gain"),
            fade_in=to_fraction(data.get("fade_in", 0), field="bg.fade_in"),
            fade_out=to_fraction(data.get("fade_out", 0), field="bg.fade_out"),
            anchors=anchors,
        )


@dataclass(frozen=True)
class OutputSpec:
    """Normalized render target.  Defaults match the task specification."""

    width: int
    height: int
    fps_num: int
    fps_den: int
    sample_rate: int
    channels: int

    @property
    def fps(self) -> Fraction:
        return Fraction(self.fps_num, self.fps_den)

    @property
    def frame_period(self) -> Fraction:
        return Fraction(self.fps_den, self.fps_num)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": {"num": self.fps_num, "den": self.fps_den},
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OutputSpec":
        data = data or {}
        fps = data.get("fps", {"num": 30000, "den": 1001})
        return cls(
            width=int(data.get("width", 320)),
            height=int(data.get("height", 180)),
            fps_num=int(fps["num"]),
            fps_den=int(fps["den"]),
            sample_rate=int(data.get("sample_rate", 48000)),
            channels=int(data.get("channels", 2)),
        )


@dataclass
class Project:
    """Full validated render project."""

    output: OutputSpec
    duration: Fraction
    media: dict[str, MediaRef]
    clips: list[Clip]
    transitions: list[Transition]
    backgrounds: list[BackgroundAudio]
    source_root: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "output": self.output.to_dict(),
            "duration": to_json(self.duration),
            "media": [item.to_dict() for item in self.media.values()],
            "clips": [clip.to_dict() for clip in self.clips],
            "transitions": [tr.to_dict() for tr in self.transitions],
            "backgrounds": [bg.to_dict() for bg in self.backgrounds],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _item_label(item: Clip | BackgroundAudio) -> str:
    return getattr(item, "clip_id", getattr(item, "item_id", "item"))


def _validate_anchors(item: Clip | BackgroundAudio) -> None:
    if not item.anchors:
        return
    first, second = item.anchors
    if second.audio_sample <= first.audio_sample:
        raise ProjectValidationError(
            f"{_item_label(item)}: anchor audio positions must be strictly increasing"
        )
    if second.video_time <= first.video_time:
        raise ProjectValidationError(
            f"{_item_label(item)}: anchor video times must be strictly increasing"
        )


def validate_project(project: Project) -> None:
    """Cross-field validation; raises :class:`ProjectValidationError`."""
    if project.duration <= 0:
        raise ProjectValidationError("project duration must be positive")
    out = project.output
    if out.width <= 0 or out.height <= 0:
        raise ProjectValidationError("output dimensions must be positive")
    if out.fps_num <= 0 or out.fps_den <= 0:
        raise ProjectValidationError("output fps must be positive")
    if out.sample_rate <= 0 or out.channels <= 0:
        raise ProjectValidationError("sample_rate/channels must be positive")
    if not project.media:
        raise ProjectValidationError("project declares no media")

    clip_ids: set[str] = set()
    for clip in project.clips:
        if clip.clip_id in clip_ids:
            raise ProjectValidationError(f"duplicate clip id {clip.clip_id!r}")
        clip_ids.add(clip.clip_id)
        if clip.media_id not in project.media:
            raise ProjectValidationError(
                f"clip {clip.clip_id}: unknown media {clip.media_id!r}"
            )
        if clip.src_in < 0:
            raise ProjectValidationError(f"clip {clip.clip_id}: src_in must be >= 0")
        if clip.src_out <= clip.src_in:
            raise ProjectValidationError(
                f"clip {clip.clip_id}: zero/negative length source interval"
            )
        if clip.timeline_start < 0:
            raise ProjectValidationError(
                f"clip {clip.clip_id}: timeline_start must be >= 0"
            )
        if clip.timeline_end > project.duration:
            raise ProjectValidationError(
                f"clip {clip.clip_id}: extends past project duration"
            )
        if clip.speed <= 0:
            raise ProjectValidationError(f"clip {clip.clip_id}: speed must be > 0")
        if clip.gain < 0:
            raise ProjectValidationError(f"clip {clip.clip_id}: gain must be >= 0")
        if clip.fade_in < 0 or clip.fade_out < 0:
            raise ProjectValidationError(f"clip {clip.clip_id}: fades must be >= 0")
        if clip.fade_in + clip.fade_out > clip.duration:
            raise ProjectValidationError(
                f"clip {clip.clip_id}: fades exceed clip duration"
            )
        _validate_anchors(clip)

    by_id = {clip.clip_id: clip for clip in project.clips}
    for transition in project.transitions:
        if transition.kind == "cut":
            continue
        if transition.from_clip not in by_id or transition.to_clip not in by_id:
            raise ProjectValidationError(
                f"transition {transition.from_clip}->{transition.to_clip}: unknown clip"
            )
        first = by_id[transition.from_clip]
        second = by_id[transition.to_clip]
        if first.track != second.track:
            raise ProjectValidationError(
                "transition endpoints must be on the same track"
            )
        if transition.overlap <= 0:
            raise ProjectValidationError("transition overlap must be positive")
        overlap_start = max(first.timeline_start, second.timeline_start)
        overlap_end = min(first.timeline_end, second.timeline_end)
        actual_overlap = overlap_end - overlap_start
        if actual_overlap <= 0:
            raise ProjectValidationError("transition clips do not overlap")
        if transition.overlap > actual_overlap:
            raise ProjectValidationError(
                "declared transition overlap exceeds actual clip overlap"
            )

    # Same-track triple overlap rejection: at any timeline point at most two
    # clips may be active on a track.
    tracks: dict[int, list[Clip]] = {}
    for clip in project.clips:
        tracks.setdefault(clip.track, []).append(clip)
    for track_index, track_clips in tracks.items():
        ordered = sorted(track_clips, key=lambda item: item.timeline_start)
        for index, clip in enumerate(ordered):
            active = 1
            for other in ordered[index + 1 :]:
                if other.timeline_start >= clip.timeline_end:
                    break
                if other.timeline_start < clip.timeline_end:
                    active += 1
            if active > 2:
                raise ProjectValidationError(
                    f"track {track_index}: triple overlap near "
                    f"t={clip.timeline_start} is not supported"
                )

    for background in project.backgrounds:
        if background.media_id not in project.media:
            raise ProjectValidationError(
                f"background {background.item_id}: unknown media {background.media_id!r}"
            )
        if background.src_out <= background.src_in:
            raise ProjectValidationError(
                f"background {background.item_id}: zero/negative source interval"
            )
        if background.timeline_start < 0:
            raise ProjectValidationError(
                f"background {background.item_id}: timeline_start must be >= 0"
            )
        if background.timeline_end > project.duration:
            raise ProjectValidationError(
                f"background {background.item_id}: extends past project duration"
            )
        if background.speed <= 0:
            raise ProjectValidationError(
                f"background {background.item_id}: speed must be > 0"
            )
        if background.gain < 0:
            raise ProjectValidationError(
                f"background {background.item_id}: gain must be >= 0"
            )
        if background.fade_in + background.fade_out > background.duration:
            raise ProjectValidationError(
                f"background {background.item_id}: fades exceed item duration"
            )
        _validate_anchors(background)


def load_project(path: str | Path) -> Project:
    """Load, structurally parse and validate a project JSON file."""
    project_path = Path(path)
    raw = json.loads(project_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ProjectValidationError("project root must be a JSON object")
    version = raw.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ProjectValidationError(
            f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}"
        )
    if "duration" not in raw:
        raise ProjectValidationError("project missing 'duration'")

    media_items = [MediaRef.from_dict(item) for item in raw.get("media", [])]
    media = {item.media_id: item for item in media_items}
    if len(media) != len(media_items):
        raise ProjectValidationError("duplicate media id")
    clips = [Clip.from_dict(item) for item in raw.get("clips", [])]
    transitions = [
        Transition.from_dict(item) for item in raw.get("transitions", [])
    ]
    backgrounds = [
        BackgroundAudio.from_dict(item) for item in raw.get("backgrounds", [])
    ]
    project = Project(
        output=OutputSpec.from_dict(raw.get("output")),
        duration=to_fraction(raw["duration"], field="project.duration"),
        media=media,
        clips=clips,
        transitions=transitions,
        backgrounds=backgrounds,
        source_root=project_path.parent,
    )
    validate_project(project)
    return project
