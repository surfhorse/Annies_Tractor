"""Fixtures for afv.media tests; video generators live in media_synth.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from media_synth import GEMX_PY, GEMX_VIDEO, colour_cuts_frames, write_video

from afv.media.ffmpeg import MediaError, ffmpeg_path


@pytest.fixture(scope="session")
def gemx_python() -> Path:
    """The gem-x env's python (has OpenCV and GEM-X's video.py); tests skip without it."""
    if not (GEMX_PY.exists() and GEMX_VIDEO.exists()):
        pytest.skip("gem-x environment not installed")
    return GEMX_PY


@pytest.fixture(scope="session")
def ffmpeg() -> str:
    try:
        return ffmpeg_path()
    except MediaError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("synth")


@pytest.fixture(scope="session")
def cuts_video(synth_dir: Path) -> Path:
    """30 fps: shots of 40, 35, 50 frames (cuts at frames 40 and 75)."""
    return write_video(synth_dir / "cuts.mp4", colour_cuts_frames([40, 35, 50]))


@pytest.fixture(scope="session")
def vfr_video(synth_dir: Path) -> Path:
    rng = np.random.default_rng(5)
    dts = rng.choice([1 / 24, 1 / 30, 1 / 50, 1 / 15], size=90)
    times = np.concatenate([[0.0], np.cumsum(dts)[:-1]])
    frames = colour_cuts_frames([45, 45], seed=7)
    return write_video(synth_dir / "vfr.mp4", frames, times=list(times))
