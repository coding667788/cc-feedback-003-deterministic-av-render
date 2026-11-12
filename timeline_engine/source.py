"""Source probing and deterministic decode into the normalized work format.

Work formats (fixed for every render path):

* video: raw ``rgb24``, full (PC) range, SDR Rec.709, already scaled with
  bicubic filtering and black letterbox/pillarbox padding to the project
  output size;
* audio: native source sample rate, stereo, packed IEEE-754 ``float32``
  samples (no resampling yet - the timeline engine owns resampling so sample
  boundaries stay exact).

Frame selection later uses the decoded *display* PTS list captured here; the
average frame rate is never used as a time mapping.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .errors import SourceDecodeError, SourceProbeError
from .ffmpeg_tools import (
    canonical_json,
    probe_container,
    run_ffmpeg,
    sha256_file,
)
from .model import MediaRef, OutputSpec
from .probe_extra import probe_frames, probe_frames_audio, probe_packets

WORK_PIX_FMT = "rgb24"
WORK_COLOR_RANGE = "pc"
WORK_COLOR_SPACE = "bt709"
WORK_AUDIO_CHANNELS = 2
WORK_AUDIO_DTYPE = np.float32


def pts_value(value: object) -> Fraction | None:
    """Parse an ffprobe ``pts_time`` field, tolerating ``N/A``."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "N/A"):
        return None
    return Fraction(text)


@dataclass
class SourceBundle:
    """Decoded source plus the exact timing metadata needed by the planner."""

    media_id: str
    sha256: str
    source_path: Path
    source_id: int
    width: int
    height: int
    frame_pts: list[Fraction]
    has_b_frames: bool
    source_pix_fmt: str | None
    source_color_range: str | None
    source_color_space: str | None
    video_path: Path | None
    audio_path: Path | None
    audio_sample_rate: int | None
    audio_first_pts: Fraction | None
    audio_samples: int
    _video_mem: np.memmap | None = None
    _audio_mem: np.memmap | None = None

    @property
    def has_video(self) -> bool:
        return self.video_path is not None and bool(self.frame_pts)

    @property
    def has_audio(self) -> bool:
        return self.audio_path is not None and self.audio_sample_rate is not None

    @property
    def video_end_pts(self) -> Fraction | None:
        return self.frame_pts[-1] if self.frame_pts else None

    @property
    def audio_end_time(self) -> Fraction | None:
        if not self.has_audio or self.audio_first_pts is None:
            return None
        assert self.audio_sample_rate is not None
        return self.audio_first_pts + Fraction(
            self.audio_samples, self.audio_sample_rate
        )

    def read_frame(self, index: int) -> np.ndarray:
        """Return one HxWx3 RGB24 frame."""
        if not self.has_video:
            raise SourceDecodeError(f"{self.media_id}: no video stream")
        if self._video_mem is None:
            self._video_mem = np.memmap(
                self.video_path, dtype=np.uint8, mode="r"
            )
        frame_bytes = self.width * self.height * 3
        start = index * frame_bytes
        flat = np.asarray(self._video_mem[start : start + frame_bytes])
        return flat.reshape(self.height, self.width, 3)

    def read_audio(self) -> np.ndarray:
        """Return the full stereo float32 track, shape ``(samples, 2)``."""
        if not self.has_audio:
            raise SourceDecodeError(f"{self.media_id}: no audio stream")
        if self._audio_mem is None:
            self._audio_mem = np.memmap(
                self.audio_path, dtype=np.float32, mode="r"
            )
        return np.asarray(self._audio_mem).reshape(-1, WORK_AUDIO_CHANNELS)

    def close(self) -> None:
        for member_name in ("_video_mem", "_audio_mem"):
            member = getattr(self, member_name)
            if member is not None:
                member._mmap.close()  # type: ignore[attr-defined]
                setattr(self, member_name, None)

    def source_id_marker(self) -> int:
        return self.source_id


class SourceStore:
    """Content-addressed source decode cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        output: OutputSpec,
        toolchain: dict[str, str],
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.toolchain = toolchain
        self._bundles: dict[str, SourceBundle] = {}
        self._source_ids: dict[str, int] = {}

    def assign_source_id(self, media_id: str, source_id: int) -> None:
        """Register the generator marker id embedded in a source's frames."""
        self._source_ids[media_id] = source_id

    def get(self, media: MediaRef) -> SourceBundle:
        source_path = Path(media.path)
        if not source_path.is_absolute():
            raise SourceProbeError(
                f"media {media.media_id}: resolve project-relative paths first"
            )
        if not source_path.is_file():
            raise SourceProbeError(
                f"media {media.media_id}: file not found {source_path}"
            )
        digest = sha256_file(source_path)
        entry_dir = self.cache_dir / digest[:16]
        meta_path = entry_dir / "meta.json"
        video_path = entry_dir / "video.rgb24"
        audio_path = entry_dir / "audio.f32"

        key_material = canonical_json(
            {
                "sha256": digest,
                "width": self.output.width,
                "height": self.output.height,
                "pix_fmt": WORK_PIX_FMT,
                "color_range": WORK_COLOR_RANGE,
                "audio_channels": WORK_AUDIO_CHANNELS,
                "media_decl": {
                    "pixel_format": media.pixel_format,
                    "color_range": media.color_range,
                    "color_space": media.color_space,
                },
                "toolchain": self.toolchain,
            }
        )
        key_hash = hashlib.sha256(key_material).hexdigest()

        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("cache_key") == key_hash:
                bundle = self._bundle_from_meta(
                    media, digest, source_path, entry_dir, meta
                )
                self._verify_artifacts(bundle)
                self._bundles[media.media_id] = bundle
                return bundle

        probe = probe_container(source_path)
        streams = probe.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        if not video_streams and not audio_streams:
            raise SourceProbeError(
                f"{media.media_id}: contains no audio or video stream"
            )

        video_meta = self._decode_video(
            media,
            video_streams[0] if video_streams else None,
            source_path,
            entry_dir,
            video_path,
        )
        audio_meta = self._decode_audio(
            media,
            audio_streams[0] if audio_streams else None,
            source_path,
            entry_dir,
            audio_path,
        )
        meta = {
            "cache_key": key_hash,
            "media_id": media.media_id,
            "source_sha256": digest,
            "source_path": str(source_path),
            "source_id": self._source_ids.get(media.media_id, 0),
            "work_format": {
                "pix_fmt": WORK_PIX_FMT,
                "color_range": WORK_COLOR_RANGE,
                "color_space": WORK_COLOR_SPACE,
                "width": self.output.width,
                "height": self.output.height,
            },
            **video_meta,
            **audio_meta,
        }
        tmp_meta = meta_path.with_suffix(".json.tmp")
        tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp_meta.replace(meta_path)
        bundle = self._bundle_from_meta(
            media, digest, source_path, entry_dir, meta
        )
        self._verify_artifacts(bundle)
        self._bundles[media.media_id] = bundle
        return bundle

    def _decode_video(
        self,
        media: MediaRef,
        stream: dict | None,
        source_path: Path,
        entry_dir: Path,
        video_path: Path,
    ) -> dict:
        if stream is None:
            return {
                "has_video": False,
                "width": self.output.width,
                "height": self.output.height,
                "frame_pts": [],
                "has_b_frames": False,
                "source_pix_fmt": None,
                "source_color_range": None,
                "source_color_space": None,
            }
        entry_dir.mkdir(parents=True, exist_ok=True)
        frames = probe_frames(source_path)
        pts_values: list[Fraction] = []
        for frame in frames:
            pts = pts_value(frame.get("pts_time"))
            if pts is None:
                raise SourceProbeError(
                    f"{media.media_id}: video frame without usable display PTS; "
                    "exact frame selection is impossible"
                )
            pts_values.append(pts)
        if not pts_values:
            raise SourceProbeError(
                f"{media.media_id}: video stream has no decodable frames"
            )
        for earlier, later in zip(pts_values, pts_values[1:]):
            if later <= earlier:
                raise SourceProbeError(
                    f"{media.media_id}: display PTS not strictly increasing "
                    f"({earlier} -> {later})"
                )

        packets = probe_packets(source_path)
        has_b_frames = any(
            (pkt_pts := pts_value(item.get("pts_time"))) is not None
            and (pkt_dts := pts_value(item.get("dts_time"))) is not None
            and pkt_pts != pkt_dts
            for item in packets
        )

        source_pix_fmt = stream.get("pix_fmt")
        source_range = media.color_range or stream.get("color_range")
        source_space = media.color_space or stream.get("color_space")
        is_yuv = bool(source_pix_fmt) and not source_pix_fmt.startswith(
            ("rgb", "bgr", "rgba", "bgra", "gbrp", "pal8")
        )
        if is_yuv and source_range not in ("tv", "pc"):
            raise SourceProbeError(
                f"{media.media_id}: YUV source {source_pix_fmt} has undeclared "
                "color range; declare color_range in the project"
            )

        chain: list[str] = []
        if is_yuv:
            in_range = "limited" if source_range == "tv" else "full"
            chain.append(f"scale=in_range={in_range}:out_range=full")
        chain.append(
            "scale="
            f"{self.output.width}:{self.output.height}:"
            "force_original_aspect_ratio=decrease:flags=bicubic"
        )
        chain.append(
            f"pad={self.output.width}:{self.output.height}:"
            "(ow-iw)/2:(oh-ih)/2:color=black"
        )
        chain.append(f"format={WORK_PIX_FMT}")
        tmp_video = video_path.with_suffix(".rgb24.tmp")
        run_ffmpeg(
            [
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                ",".join(chain),
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                WORK_PIX_FMT,
                str(tmp_video),
            ]
        )
        tmp_video.replace(video_path)
        expected_bytes = (
            len(pts_values) * self.output.width * self.output.height * 3
        )
        actual_bytes = video_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise SourceDecodeError(
                f"{media.media_id}: decoded {actual_bytes} video bytes, expected "
                f"{expected_bytes} ({len(pts_values)} frames)"
            )
        return {
            "has_video": True,
            "width": self.output.width,
            "height": self.output.height,
            "frame_pts": [
                f"{p.numerator}/{p.denominator}" for p in pts_values
            ],
            "has_b_frames": has_b_frames,
            "source_pix_fmt": source_pix_fmt,
            "source_color_range": source_range,
            "source_color_space": source_space,
        }

    def _decode_audio(
        self,
        media: MediaRef,
        stream: dict | None,
        source_path: Path,
        entry_dir: Path,
        audio_path: Path,
    ) -> dict:
        if stream is None:
            return {
                "has_audio": False,
                "audio_sample_rate": None,
                "audio_first_pts": None,
                "audio_samples": 0,
            }
        sample_rate = int(stream.get("sample_rate", 0))
        if sample_rate <= 0:
            raise SourceProbeError(
                f"{media.media_id}: audio sample_rate missing"
            )
        entry_dir.mkdir(parents=True, exist_ok=True)
        first_pts = self._audio_first_pts(media, source_path, stream)
        tmp_audio = audio_path.with_suffix(".f32.tmp")
        run_ffmpeg(
            [
                "-i",
                str(source_path),
                "-map",
                "0:a:0",
                "-vn",
                "-af",
                "aformat=sample_fmts=flt:channel_layouts=stereo",
                "-fps_mode",
                "passthrough",
                "-f",
                "f32le",
                str(tmp_audio),
            ]
        )
        tmp_audio.replace(audio_path)
        byte_count = audio_path.stat().st_size
        if byte_count % (4 * WORK_AUDIO_CHANNELS) != 0:
            raise SourceDecodeError(
                f"{media.media_id}: truncated float32 audio decode"
            )
        sample_count = byte_count // (4 * WORK_AUDIO_CHANNELS)
        if sample_count == 0:
            raise SourceProbeError(
                f"{media.media_id}: audio stream decodes to zero samples"
            )
        return {
            "has_audio": True,
            "audio_sample_rate": sample_rate,
            "audio_first_pts": (
                f"{first_pts.numerator}/{first_pts.denominator}"
            ),
            "audio_samples": sample_count,
        }

    def _audio_first_pts(
        self, media: MediaRef, source_path: Path, stream: dict
    ) -> Fraction:
        frames = probe_frames_audio(source_path)
        for frame in frames:
            pts = pts_value(frame.get("pts_time"))
            if pts is not None:
                return pts
        start_time = pts_value(stream.get("start_time"))
        if start_time is not None:
            return start_time
        raise SourceProbeError(
            f"{media.media_id}: audio stream lacks any PTS; cannot align to video"
        )

    def _bundle_from_meta(
        self,
        media: MediaRef,
        digest: str,
        source_path: Path,
        entry_dir: Path,
        meta: dict,
    ) -> SourceBundle:
        frame_pts = [Fraction(text) for text in meta.get("frame_pts", [])]
        first_pts_text = meta.get("audio_first_pts")
        video_name = entry_dir / "video.rgb24" if meta.get("has_video") else None
        audio_name = entry_dir / "audio.f32" if meta.get("has_audio") else None
        return SourceBundle(
            media_id=media.media_id,
            sha256=digest,
            source_path=source_path,
            source_id=int(meta.get("source_id", self._source_ids.get(media.media_id, 0))),
            width=int(meta.get("width", self.output.width)),
            height=int(meta.get("height", self.output.height)),
            frame_pts=frame_pts,
            has_b_frames=bool(meta.get("has_b_frames")),
            source_pix_fmt=meta.get("source_pix_fmt"),
            source_color_range=meta.get("source_color_range"),
            source_color_space=meta.get("source_color_space"),
            video_path=video_name,
            audio_path=audio_name,
            audio_sample_rate=meta.get("audio_sample_rate"),
            audio_first_pts=(
                Fraction(first_pts_text) if first_pts_text else None
            ),
            audio_samples=int(meta.get("audio_samples", 0)),
        )

    def _verify_artifacts(self, bundle: SourceBundle) -> None:
        if bundle.has_video:
            assert bundle.video_path is not None
            expected = len(bundle.frame_pts) * bundle.width * bundle.height * 3
            if not bundle.video_path.is_file() or bundle.video_path.stat().st_size != expected:
                raise SourceDecodeError(
                    f"{bundle.media_id}: cached video artifact missing/corrupt"
                )
        if bundle.has_audio:
            assert bundle.audio_path is not None
            expected = bundle.audio_samples * 4 * WORK_AUDIO_CHANNELS
            if not bundle.audio_path.is_file() or bundle.audio_path.stat().st_size != expected:
                raise SourceDecodeError(
                    f"{bundle.media_id}: cached audio artifact missing/corrupt"
                )
