"""Deterministic synthetic material generator.

Generated video carries machine-decodable frame identifiers in fixed pixel
cells (no fonts / no OCR):

* rows 0..7   : 32-bit source id, MSB first, one bit per column group
* rows 8..39  : 32-bit frame index, MSB first
* row 40      : even/odd parity bit covering all 32 id + 32 index bits
* rows 41..43 : constant signature 101 used as a read sanity marker

Each bit occupies an 8x4 block (8 columns x 4 rows) and bit value ``1`` is
white (255), ``0`` is dark gray (32).  The rest of the frame is filled with a
deterministic per-source base color so dissolves remain visually meaningful.

Audio contains a continuous 440 Hz-ish sine (frequency depends on source id)
plus one-unit triangular pulses at known global source positions, giving the
validator independent timing landmarks.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .ffmpeg_tools import run_ffmpeg
from .probe_extra import probe_frames, probe_frames_audio
from .source import pts_value

BIT_COLUMNS = 32
BIT_BLOCK_W = 8
BIT_BLOCK_H = 4
ID_ROWS = BIT_BLOCK_H
INDEX_ROWS = BIT_BLOCK_H * 4  # 32-bit frame index needs four 8-bit planes
PARITY_ROWS = BIT_BLOCK_H
SIGNATURE_BANDS = 3
DARK_ANCHOR_BANDS = 1
BRIGHT_ANCHOR_BANDS = 1
MARKER_HEIGHT = (
    ID_ROWS
    + INDEX_ROWS
    + PARITY_ROWS
    + (SIGNATURE_BANDS + DARK_ANCHOR_BANDS + BRIGHT_ANCHOR_BANDS)
    * BIT_BLOCK_H
)
BYTE_PLANE_BITS = 10  # leading 1 anchor + leading 0 anchor + 8 data bits


@dataclass(frozen=True)
class GeneratedSpec:
    path: Path
    kind: str
    width: int
    height: int
    fps: Fraction
    frame_count: int
    sample_rate: int | None
    source_id: int
    first_video_pts: Fraction
    first_audio_pts: Fraction | None
    pulse_positions: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "fps": {"num": self.fps.numerator, "den": self.fps.denominator},
            "frame_count": self.frame_count,
            "sample_rate": self.sample_rate,
            "source_id": self.source_id,
            "first_video_pts": (
                f"{self.first_video_pts.numerator}/"
                f"{self.first_video_pts.denominator}"
            ),
            "first_audio_pts": (
                None
                if self.first_audio_pts is None
                else (
                    f"{self.first_audio_pts.numerator}/"
                    f"{self.first_audio_pts.denominator}"
                )
            ),
            "pulse_positions": list(self.pulse_positions),
        }


def marker_dimensions() -> tuple[int, int]:
    width = BIT_COLUMNS * BIT_BLOCK_W
    height = MARKER_HEIGHT
    return width, height


def _paint_bit_plane(
    frame: np.ndarray,
    row_offset: int,
    value: int,
    bits: int = BIT_COLUMNS,
) -> None:
    for bit_index in range(bits):
        bit = (value >> (bits - 1 - bit_index)) & 1
        color = 235 if bit else 64
        x0 = bit_index * BIT_BLOCK_W
        frame[
            row_offset : row_offset + BIT_BLOCK_H, x0 : x0 + BIT_BLOCK_W, :
        ] = color


def build_frame(
    source_id: int, frame_index: int, width: int, height: int
) -> np.ndarray:
    base_color = np.array(
        [40 + (source_id * 17) % 120, 60, 90 + (source_id * 29) % 120],
        dtype=np.uint8,
    )
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :, :] = base_color
    _paint_bit_plane(frame, 0, source_id)
    for plane in range(4):
        byte_value = (frame_index >> (8 * (3 - plane))) & 0xFF
        anchored = 0b10_00000000 | byte_value
        _paint_bit_plane(
            frame, ID_ROWS + plane * BIT_BLOCK_H, anchored, bits=BYTE_PLANE_BITS
        )
    parity = (bin(source_id).count("1") + bin(frame_index).count("1")) & 1
    parity_row = ID_ROWS + INDEX_ROWS
    frame[parity_row : parity_row + BIT_BLOCK_H, 0:BIT_BLOCK_W, :] = (
        235 if parity else 64
    )
    signature = (1, 0, 1)
    for offset, bit in enumerate(signature):
        row = parity_row + BIT_BLOCK_H + offset * BIT_BLOCK_H
        frame[row : row + BIT_BLOCK_H, 0:BIT_BLOCK_W, :] = (
            235 if bit else 64
        )
    dark_anchor_row = (
        parity_row + BIT_BLOCK_H + SIGNATURE_BANDS * BIT_BLOCK_H
    )
    frame[
        dark_anchor_row : dark_anchor_row + BIT_BLOCK_H, 0:BIT_BLOCK_W, :
    ] = 64
    bright_anchor_row = dark_anchor_row + BIT_BLOCK_H
    frame[
        bright_anchor_row : bright_anchor_row + BIT_BLOCK_H,
        0:BIT_BLOCK_W,
        :,
    ] = 235
    return frame


def decode_frame_marker(frame: np.ndarray) -> tuple[int, int, bool]:
    """Read source id / frame index and verify parity + signature."""

    def cell_value(row: int, bit_index: int) -> float:
        x0 = bit_index * BIT_BLOCK_W + BIT_BLOCK_W // 2
        y0 = row + BIT_BLOCK_H // 2
        return float(frame[y0, x0, 0])

    def block_mean(row: int, bit_index: int) -> float:
        x0 = bit_index * BIT_BLOCK_W
        y0 = row
        block = frame[y0 : y0 + BIT_BLOCK_H, x0 : x0 + BIT_BLOCK_W, 0]
        return float(np.mean(block.astype(np.float64)))

    def read_plane(row_offset: int, bits: int = BIT_COLUMNS) -> int:
        values = [block_mean(row_offset, i) for i in range(bits)]
        midpoint = (max(values) + min(values)) / 2.0
        value = 0
        for bit_index in range(bits):
            bit = 1 if values[bit_index] >= midpoint else 0
            value = (value << 1) | bit
        return value

    source_id = read_plane(0)
    frame_index = 0
    for plane in range(4):
        anchored = read_plane(
            ID_ROWS + plane * BIT_BLOCK_H, bits=BYTE_PLANE_BITS
        )
        byte_value = anchored & 0xFF
        frame_index = (frame_index << 8) | byte_value
    parity_row = ID_ROWS + INDEX_ROWS
    parity_on = block_mean(parity_row, 0)
    signature_values = [
        block_mean(
            parity_row + BIT_BLOCK_H + offset * BIT_BLOCK_H, 0
        )
        for offset in range(SIGNATURE_BANDS)
    ]
    dark_anchor = block_mean(
        parity_row + BIT_BLOCK_H + SIGNATURE_BANDS * BIT_BLOCK_H, 0
    )
    bright_anchor = block_mean(
        parity_row
        + BIT_BLOCK_H
        + (SIGNATURE_BANDS + DARK_ANCHOR_BANDS) * BIT_BLOCK_H,
        0,
    )
    reference_max = bright_anchor
    reference_min = dark_anchor
    parity_mid = (reference_max + reference_min) / 2.0
    parity_bit = 1 if parity_on >= parity_mid else 0
    parity_expected = (
        bin(source_id).count("1") + bin(frame_index).count("1")
    ) & 1
    signature_ok = True
    for offset, expected in enumerate((1, 0, 1)[:SIGNATURE_BANDS]):
        actual = (
            1
            if signature_values[offset] >= parity_mid
            else 0
        )
        signature_ok = signature_ok and actual == expected
    return source_id, frame_index, parity_bit == parity_expected and signature_ok


def build_audio(
    source_id: int,
    sample_rate: int,
    duration_seconds: float,
    pulse_positions: tuple[int, ...],
) -> np.ndarray:
    count = int(round(duration_seconds * sample_rate))
    timeline = (np.arange(count, dtype=np.float64) + 0.5) / sample_rate
    frequency = 330.0 + 11.0 * source_id
    wave = 0.25 * np.sin(2.0 * math.pi * frequency * timeline)
    pulse_track = np.zeros_like(wave)
    for position in pulse_positions:
        center = (position + 0.5) / sample_rate
        distance = np.abs(timeline - center) * sample_rate
        pulse = np.clip(1.0 - distance, 0.0, 1.0)
        wave += 0.5 * pulse
        pulse_track += 0.7 * pulse
    stereo = np.stack([wave, pulse_track], axis=1).astype(np.float32)
    return np.clip(stereo, -0.95, 0.95)


def write_av_file(
    target: Path,
    frames: np.ndarray,
    audio: np.ndarray | None,
    fps: Fraction,
    sample_rate: int | None,
    *,
    video_codec: str = "ffv1",
    pixel_format: str = "rgb24",
    b_frames: bool = False,
    output_ts_offset: Fraction | None = None,
) -> None:
    """Mux video and optional audio via named temp raw files (deterministic)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    video_raw = target.with_suffix(".video.raw")
    video_raw.write_bytes(np.ascontiguousarray(frames).tobytes())
    args: list[str] = [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{frames.shape[2]}x{frames.shape[1]}",
        "-framerate",
        f"{fps.numerator}/{fps.denominator}",
        "-i",
        str(video_raw),
    ]
    audio_raw = None
    if audio is not None and sample_rate is not None:
        audio_raw = target.with_suffix(".audio.raw")
        audio_raw.write_bytes(np.ascontiguousarray(audio).tobytes())
        args += [
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            "2",
            "-i",
            str(audio_raw),
        ]
    if output_ts_offset is not None:
        args += [
            "-output_ts_offset",
            f"{float(output_ts_offset):.12f}",
        ]
    args += ["-map", "0:v:0"]
    if audio is not None:
        args += ["-map", "1:a:0", "-c:a", "pcm_f32le"]
    args += ["-c:v", video_codec]
    if video_codec == "libx264":
        args += [
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            pixel_format,
            "-bf",
            "2" if b_frames else "0",
            "-g",
            "12",
        ]
    else:
        args += ["-pix_fmt", pixel_format, "-g", "1"]
    args += [str(target)]
    try:
        run_ffmpeg(args)
    finally:
        video_raw.unlink(missing_ok=True)
        if audio_raw is not None:
            audio_raw.unlink(missing_ok=True)


def _frame_stack(
    source_id: int, count: int, width: int, height: int
) -> np.ndarray:
    return np.stack(
        [
            build_frame(source_id, index, width, height)
            for index in range(count)
        ],
        axis=0,
    )


def write_vfr_file(
    target: Path,
    source_id: int,
    frame_pts: list[Fraction],
    width: int,
    height: int,
    audio: np.ndarray | None,
    sample_rate: int | None,
) -> None:
    """Create a genuinely variable-PTS file via the concat demuxer.

    Each frame is a lossless PNG with an explicit per-frame duration, so the
    resulting presentation timestamps follow ``frame_pts`` exactly instead of
    an average fps.  PNG is used only at generation time; render inputs still
    go through the same decode/probe pipeline as real camera material.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = target.parent / f"{target.stem}_vfr_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines: list[str] = []
    duration_den = 1000
    for index, pts in enumerate(frame_pts):
        frame = build_frame(source_id, index, width, height)
        frame_path = frame_dir / f"frame_{index:05d}.png"
        raw = frame_dir / f"frame_{index:05d}.raw"
        raw.write_bytes(np.ascontiguousarray(frame).tobytes())
        run_ffmpeg(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-i",
                str(raw),
                "-frames:v",
                "1",
                str(frame_path),
            ]
        )
        raw.unlink(missing_ok=True)
        if index < len(frame_pts) - 1:
            duration = frame_pts[index + 1] - pts
        else:
            duration = frame_pts[-1] - frame_pts[-2]
        duration_ms = round(float(duration) * duration_den)
        manifest_lines.append(f"file '{frame_path.resolve().as_posix()}'")
        manifest_lines.append(f"duration {duration_ms / duration_den}")
    manifest_lines.append(
        f"file '{(frame_dir / f'frame_{len(frame_pts) - 1:05d}.png').resolve().as_posix()}'"
    )
    manifest = frame_dir / "concat.txt"
    manifest.write_text("\n".join(manifest_lines), encoding="utf-8")
    concat_args_prefix = [
        "-f",
        "concat",
        "-safe",
        "0",
    ]

    if audio is not None and sample_rate is not None:
        audio_raw = target.with_suffix(".audio.raw")
        audio_raw.write_bytes(np.ascontiguousarray(audio).tobytes())
        try:
            run_ffmpeg(
                [
                    *concat_args_prefix,
                    "-i",
                    str(manifest),
                    "-f",
                    "f32le",
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    "2",
                    "-i",
                    str(audio_raw),
                    "-map",
                    "0:v:0",
                    "-vf",
                    f"setpts=PTS+{int(round(float(frame_pts[0]) * 1000))}/1000/TB",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "ffv1",
                    "-pix_fmt",
                    "rgb24",
                    "-c:a",
                    "pcm_f32le",
                    str(target),
                ]
            )
        finally:
            audio_raw.unlink(missing_ok=True)
    else:
        run_ffmpeg(
            [
                *concat_args_prefix,
                "-i",
                str(manifest),
                "-vf",
                f"setpts=PTS+{int(round(float(frame_pts[0]) * 1000))}/1000/TB",
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "ffv1",
                "-pix_fmt",
                "rgb24",
                str(target),
            ]
        )


def probe_actual_pts(
    target: Path,
) -> tuple[Fraction | None, Fraction | None]:
    """Read back real first video/audio PTS after muxing (edit lists etc.)."""
    video_frames = probe_frames(target)
    first_video = None
    for frame in video_frames:
        pts = pts_value(frame.get("pts_time"))
        if pts is not None:
            first_video = pts
            break
    audio_frames = probe_frames_audio(target)
    first_audio = None
    for frame in audio_frames:
        pts = pts_value(frame.get("pts_time"))
        if pts is not None:
            first_audio = pts
            break
    return first_video, first_audio


def generate_suite(
    output_dir: str | Path,
    *,
    width: int = 256,
    height: int = 144,
) -> list[GeneratedSpec]:
    """Generate the full required source matrix.

    Kinds:
      ``cfr25``     25 fps CFR, 48 kHz audio, non-zero start PTS
      ``cfr2997``   30000/1001 fps, 44.1 kHz audio, audio starts later
      ``vfr``       explicit variable PTS gaps, 48 kHz audio
      ``bframes``   H.264 with B-frames, 48 kHz audio, non-zero PTS
      ``videoonly`` no audio track
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs: list[GeneratedSpec] = []

    def pulses(rate: int, count: int, step_seconds: float = 0.5) -> tuple[int, ...]:
        positions = []
        t = step_seconds
        while t < count / rate - 0.05:
            positions.append(int(round(t * rate)))
            t += step_seconds
        return tuple(positions)

    # 1) CFR 25 fps with a non-zero container start offset on both streams.
    fps25 = Fraction(25, 1)
    frames25 = 90
    rate48 = 48000
    duration25 = frames25 / 25.0
    audio25 = build_audio(
        1, rate48, duration25, pulses(rate48, int(duration25 * rate48))
    )
    path = output_dir / "src_cfr25.mkv"
    write_av_file(
        path,
        _frame_stack(1, frames25, width, height),
        audio25,
        fps25,
        rate48,
        output_ts_offset=Fraction(123, 1000),
    )
    actual_video, actual_audio = probe_actual_pts(path)
    specs.append(
        GeneratedSpec(
            path=path,
            kind="cfr25",
            width=width,
            height=height,
            fps=fps25,
            frame_count=frames25,
            sample_rate=rate48,
            source_id=1,
            first_video_pts=actual_video or Fraction(123, 1000),
            first_audio_pts=actual_audio or Fraction(123, 1000),
            pulse_positions=pulses(rate48, int(duration25 * rate48)),
        )
    )

    # 2) 29.97 fps with 44.1 kHz audio whose first PTS differs from video.
    fps2997 = Fraction(30000, 1001)
    frames2997 = 90
    rate441 = 44100
    duration2997 = frames2997 * 1001 / 30000
    audio441 = build_audio(
        2, rate441, duration2997, pulses(rate441, int(duration2997 * rate441))
    )
    path = output_dir / "src_cfr2997.mkv"
    write_av_file(
        path,
        _frame_stack(2, frames2997, width, height),
        audio441,
        fps2997,
        rate441,
        output_ts_offset=Fraction(77, 1000),
    )
    actual_video, actual_audio = probe_actual_pts(path)
    specs.append(
        GeneratedSpec(
            path=path,
            kind="cfr2997",
            width=width,
            height=height,
            fps=fps2997,
            frame_count=frames2997,
            sample_rate=rate441,
            source_id=2,
            first_video_pts=actual_video or Fraction(77, 1000),
            first_audio_pts=actual_audio or Fraction(77, 1000),
            pulse_positions=pulses(rate441, int(duration2997 * rate441)),
        )
    )

    # 3) Genuine VFR: alternating short/long display durations.
    vfr_pts: list[Fraction] = []
    cursor = Fraction(201, 1000)
    for index in range(80):
        vfr_pts.append(cursor)
        gap = Fraction(1, 20) if index % 2 == 0 else Fraction(3, 40)
        cursor += gap
    vfr_duration = float(cursor - vfr_pts[0]) + float(vfr_pts[1] - vfr_pts[0])
    vfr_audio = build_audio(
        3, rate48, vfr_duration, pulses(rate48, int(vfr_duration * rate48))
    )
    path = output_dir / "src_vfr.mkv"
    write_vfr_file(
        path, 3, vfr_pts, width, height, vfr_audio, rate48
    )
    actual_video, actual_audio = probe_actual_pts(path)
    actual_frame_count = len(probe_frames(path))
    specs.append(
        GeneratedSpec(
            path=path,
            kind="vfr",
            width=width,
            height=height,
            fps=Fraction(0),
            frame_count=actual_frame_count,
            sample_rate=rate48,
            source_id=3,
            first_video_pts=actual_video or vfr_pts[0],
            first_audio_pts=actual_audio or Fraction(0),
            pulse_positions=pulses(rate48, int(vfr_duration * rate48)),
        )
    )

    # 4) H.264 with B-frames and YUV limited range declaration.
    bframe_count = 72
    bframe_duration = bframe_count / 24.0
    bframe_audio = build_audio(
        4, rate48, bframe_duration, pulses(rate48, int(bframe_duration * rate48))
    )
    path = output_dir / "src_bframes.mp4"
    write_av_file(
        path,
        _frame_stack(4, bframe_count, width, height),
        bframe_audio,
        Fraction(24, 1),
        rate48,
        video_codec="libx264",
        pixel_format="yuv420p",
        b_frames=True,
        output_ts_offset=Fraction(333, 1000),
    )
    actual_video, actual_audio = probe_actual_pts(path)
    specs.append(
        GeneratedSpec(
            path=path,
            kind="bframes",
            width=width,
            height=height,
            fps=Fraction(24, 1),
            frame_count=bframe_count,
            sample_rate=rate48,
            source_id=4,
            first_video_pts=actual_video or Fraction(333, 1000),
            first_audio_pts=actual_audio or Fraction(333, 1000),
            pulse_positions=pulses(rate48, int(bframe_duration * rate48)),
        )
    )

    # 5) Video-only source (legal no-audio material).
    path = output_dir / "src_videoonly.mkv"
    write_av_file(
        path,
        _frame_stack(5, 60, width, height),
        None,
        Fraction(30, 1),
        None,
    )
    actual_video, _ = probe_actual_pts(path)
    specs.append(
        GeneratedSpec(
            path=path,
            kind="videoonly",
            width=width,
            height=height,
            fps=Fraction(30, 1),
            frame_count=60,
            sample_rate=None,
            source_id=5,
            first_video_pts=actual_video or Fraction(0),
            first_audio_pts=None,
            pulse_positions=(),
        )
    )

    manifest = output_dir / "generated_manifest.json"
    manifest.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2),
        encoding="utf-8",
    )
    return specs
