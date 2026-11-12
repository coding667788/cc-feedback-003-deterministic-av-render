"""ffprobe helpers split out to keep :mod:`source` focused on decode policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .ffmpeg_tools import ffprobe_json


def probe_frames(path: str | Path) -> list[dict[str, Any]]:
    """Decoded video frames in presentation order (works with B-frames)."""
    data = ffprobe_json(
        ["-show_frames", "-select_streams", "v:0", "-of", "json", str(path)]
    )
    return data.get("frames", [])


def probe_packets(path: str | Path) -> list[dict[str, Any]]:
    """Packets in decode order; PTS!=DTS reveals reordering (B-frames)."""
    data = ffprobe_json(
        ["-show_packets", "-select_streams", "v:0", "-of", "json", str(path)]
    )
    return data.get("packets", [])


def probe_frames_audio(path: str | Path) -> list[dict[str, Any]]:
    data = ffprobe_json(
        ["-show_frames", "-select_streams", "a:0", "-of", "json", str(path)]
    )
    return data.get("frames", [])
