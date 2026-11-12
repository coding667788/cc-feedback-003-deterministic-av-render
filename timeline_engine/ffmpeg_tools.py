"""Thin, deterministic wrappers around the local FFmpeg/ffprobe binaries.

Only codec I/O and elementary filtering are delegated to FFmpeg.  All project
parsing, time mapping and segment planning live in Python.  The exact binary
versions are captured once and become part of every cache key so an FFmpeg
upgrade cannot silently reuse artifacts produced by another build.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .errors import SourceDecodeError, SourceProbeError

# CPU-only software codecs are selected explicitly; no GPU encoders are used.
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def resolve_binary(name: str) -> str:
    """Resolve ``ffmpeg``/``ffprobe`` honoring engine-specific env overrides."""
    env_name = f"TIMELINE_ENGINE_{name.upper()}"
    override = os.environ.get(env_name)
    if override:
        return override
    found = shutil.which(name)
    if not found:
        raise SourceProbeError(f"{name} not found on PATH and {env_name} unset")
    return found


FFMPEG = resolve_binary("ffmpeg")
FFPROBE = resolve_binary("ffprobe")


def _run(
    command: Sequence[str],
    *,
    capture: bool = True,
    stdin_data: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            list(command),
            input=stdin_data,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE,
            check=False,
            creationflags=CREATE_NO_WINDOW,
        )
    except OSError as exc:
        raise SourceDecodeError(f"failed to launch {command[0]}: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace") if completed.stderr else ""
        raise SourceDecodeError(
            f"{Path(command[0]).name} exited {completed.returncode}\n"
            f"command: {' '.join(command)}\n{stderr[-4000:]}"
        )
    return completed


def ffmpeg_version() -> str:
    """Return the full first FFmpeg version banner (build-specific)."""
    completed = _run([FFMPEG, "-version"])
    return completed.stdout.decode("utf-8", "replace").splitlines()[0].strip()


def ffprobe_version() -> str:
    completed = _run([FFPROBE, "-version"])
    return completed.stdout.decode("utf-8", "replace").splitlines()[0].strip()


def run_ffmpeg(
    args: Sequence[str],
    *,
    stdin_data: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run FFmpeg with global deterministic flags prepended."""
    command = [
        FFMPEG,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        *args,
    ]
    return _run(command, stdin_data=stdin_data)


def ffprobe_json(args: Sequence[str]) -> dict[str, Any]:
    command = [FFPROBE, "-hide_banner", "-loglevel", "error", *args]
    completed = _run(command)
    text = completed.stdout.decode("utf-8", "replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SourceProbeError(f"ffprobe returned non-JSON output: {text[:200]}") from exc


def probe_container(path: str | Path) -> dict[str, Any]:
    """Container/stream metadata plus format-level flags."""
    return ffprobe_json(
        [
            "-show_format",
            "-show_streams",
            "-show_chapters",
            "-of",
            "json",
            str(path),
        ]
    )


def probe_packets(path: str | Path) -> list[dict[str, Any]]:
    """Per-packet PTS/DTS/duration/flags in decode order (detects B-frames)."""
    data = ffprobe_json(
        [
            "-show_packets",
            "-select_streams",
            "v:0",
            "-of",
            "json",
            str(path),
        ]
    )
    return data.get("packets", [])


def probe_frames(path: str | Path) -> list[dict[str, Any]]:
    """Per-decoded-frame display PTS in display order.

    ``-show_frames`` forces decoding and reports frames in presentation order,
    which is exactly the order rawvideo output uses, even with B-frames.
    """
    data = ffprobe_json(
        [
            "-show_frames",
            "-select_streams",
            "v:0",
            "-of",
            "json",
            str(path),
        ]
    )
    return data.get("frames", [])


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Content hash used for source replacement detection."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(data: Any) -> bytes:
    """Stable serialization for cache-key hashing (sorted keys, no whitespace)."""
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


@dataclass(frozen=True)
class Toolchain:
    """Version bundle embedded in manifests and cache keys."""

    ffmpeg: str
    ffprobe: str
    engine: str

    @classmethod
    def capture(cls, engine_version: str) -> "Toolchain":
        return cls(
            ffmpeg=ffmpeg_version(),
            ffprobe=ffprobe_version(),
            engine=engine_version,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "ffmpeg": self.ffmpeg,
            "ffprobe": self.ffprobe,
            "engine": self.engine,
        }
