"""Locating and running the pinned ffmpeg binary.

GEM-X shells out to whatever ``ffmpeg`` is first on ``PATH``; frame selection and counts depend on
the ffmpeg version, so afv and GEM-X must run the same binary. One setting decides it for every
module: ``AFV_FFMPEG_DIR``, the directory holding ``ffmpeg`` and ``ffprobe`` (default
``~/.local/bin``, the static 7.0.2 build installed in P0; module C puts the same directory first
on GEM-X's ``PATH``). Without either, ``ffmpeg`` on ``PATH`` is used and ``check_ffmpeg_version``
(run at the start of the ingest stage) catches a mismatch.

The **seek origin** of a file (what ``-ss 0`` means) is the container start time as *this* binary
computes it. PyAV bundles a different libavformat that can disagree (MKV/WebM with AAC or Opus
audio: PyAV says 0, ffmpeg 7.0.2 says -0.023 s), so ``seek_origin_us`` asks the binary.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

DEFAULT_FFMPEG_DIR = "~/.local/bin"
ENV_DIR = "AFV_FFMPEG_DIR"
EXPECTED_VERSION = "7.0.2"  # the build GEM-X ran with in P0

CRASH_HINT = ("this container crashes the pinned ffmpeg; remux it to mp4 first: "
              "ffmpeg -i in -c copy out.mp4")


class MediaError(RuntimeError):
    """A media operation failed; the message is fit to show the user."""


def ffmpeg_dir() -> Path | None:
    """``AFV_FFMPEG_DIR`` if set (must contain ffmpeg), else ``~/.local/bin`` if it has ffmpeg."""
    env = os.environ.get(ENV_DIR)
    if env:
        d = Path(env).expanduser()
        if not (d / "ffmpeg").is_file():
            raise MediaError(f"{ENV_DIR}={env} does not contain an ffmpeg binary")
        return d
    d = Path(DEFAULT_FFMPEG_DIR).expanduser()
    return d if (d / "ffmpeg").is_file() else None


def ffmpeg_path(override: str | os.PathLike[str] | None = None) -> str:
    """The ffmpeg binary: ``override`` (an explicit binary path, mainly for tests), else
    ``ffmpeg_dir()/ffmpeg``, else ``ffmpeg`` on PATH."""
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return str(p)
        raise MediaError(f"ffmpeg binary not found: {override}")
    d = ffmpeg_dir()
    if d is not None:
        return str(d / "ffmpeg")
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise MediaError(f"no ffmpeg: set {ENV_DIR}, install it in {DEFAULT_FFMPEG_DIR} or put it on PATH")


def _crashed(returncode: int) -> bool:
    return returncode < 0 or returncode in (134, 136, 139)


def run_ffmpeg(args: Sequence[str], ffmpeg: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``ffmpeg <args>``; raise MediaError with ffmpeg's stderr on failure."""
    cmd = [ffmpeg_path(ffmpeg), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if _crashed(proc.returncode):
        raise MediaError(f"ffmpeg crashed (exit {proc.returncode}): {CRASH_HINT}\n  {' '.join(cmd)}")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise MediaError(f"ffmpeg failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc


def ffmpeg_version(ffmpeg: str | None = None) -> str:
    """The version token of ``ffmpeg -version`` (``7.0.2`` for ``ffmpeg version 7.0.2-static``)."""
    first = run_ffmpeg(["-hide_banner", "-version"], ffmpeg).stdout.splitlines()[0]
    m = re.match(r"ffmpeg version n?(\d+(?:\.\d+)*)", first)
    return m.group(1) if m else first.strip()


def check_ffmpeg_version(expected: str = EXPECTED_VERSION, ffmpeg: str | None = None) -> str:
    """Fail clearly unless the binary is version ``expected`` (empty ``expected`` skips the check)."""
    got = ffmpeg_version(ffmpeg)
    if expected and got != expected:
        raise MediaError(
            f"ffmpeg {got} at {ffmpeg_path(ffmpeg)}, expected {expected}: GEM-X's frame selection and "
            f"counts depend on the version. Point {ENV_DIR} at the {expected} build (P0 installed it in "
            f"{DEFAULT_FFMPEG_DIR}) or change the expected version deliberately."
        )
    return got


_START_RE = re.compile(r"Duration: [^,]*, start: (-?)(\d+)\.(\d{6})")
_ORIGINS: dict[tuple[str, int, int, str], int] = {}


def seek_origin_us(path: str | os.PathLike[str], ffmpeg: str | None = None) -> int:
    """Container start time in microseconds as the pinned ffmpeg computes it; input ``-ss`` is
    measured from it. Cached per file (path, size, mtime) and binary."""
    p = Path(path).resolve()
    st = p.stat()
    ff = ffmpeg_path(ffmpeg)
    key = (str(p), st.st_size, st.st_mtime_ns, ff)
    if key not in _ORIGINS:
        proc = subprocess.run([ff, "-hide_banner", "-nostdin", "-i", str(p)],
                              capture_output=True, text=True, check=False)
        if _crashed(proc.returncode):
            raise MediaError(f"ffmpeg crashed reading {p.name}: {CRASH_HINT}")
        if "Input #0" not in proc.stderr:
            tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
            raise MediaError(f"ffmpeg cannot read {p}:\n{tail}")
        m = _START_RE.search(proc.stderr)
        us = 0 if m is None else int(m.group(2)) * 1_000_000 + int(m.group(3))
        _ORIGINS[key] = -us if m is not None and m.group(1) == "-" else us
    return _ORIGINS[key]
