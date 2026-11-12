"""Supplied-box handling of the GEM-X worker (pure numpy part; runs on CPU in the afv env)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_WORKER = Path(__file__).parents[2] / "src/afv/models/workers/gemx_worker.py"
_spec = importlib.util.spec_from_file_location("gemx_worker_under_test", _WORKER)
gw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gw)

W, H, T = 1280, 720, 60


def _boxes() -> np.ndarray:
    b = np.full((T, 3, 4), np.nan, np.float32)
    b[:, 0] = [100, 100, 300, 600]
    b[:, 1] = [800, 120, 1000, 620]
    b[:, 2] = [500, 50, 560, 300]  # referee
    return b


def test_gaps_are_interpolated_and_unobserved():
    b = _boxes()
    b[20:45, 0] = np.nan  # fighter 0 hidden (clinch) on 20..44
    b[45, 0] = [150, 100, 350, 600]
    sel, observed, others, everyone = gw.prepare_boxes(b, 0, W, H)
    assert not observed[20:45].any() and observed[:20].all() and observed[45:].all()
    assert sel[32, 0] == pytest.approx(100 + 50 * (32 - 19) / 26, abs=1e-3)  # linear in the gap
    assert len(others[30]) == 2 and len(everyone[30]) == 2  # fighter 0 absent on frame 30
    assert len(everyone[10]) == 3


def test_partly_nan_box_is_a_gap_not_a_nan_crop():
    """Review regression: fill_gaps used to test column 0 only, so a NaN x1 leaked into the crops."""
    b = _boxes()
    b[5, 0, 2] = np.nan
    sel, observed, _, everyone = gw.prepare_boxes(b, 0, W, H)
    assert np.isfinite(sel).all() and not observed[5] and observed[4] and observed[6]
    assert len(everyone[5]) == 2
    filled = gw.fill_gaps(
        np.array([[0, 0, 10, 10], [np.nan, 1, 11, 11], [2, 2, 12, np.nan], [4, 4, 14, 14]], float)
    )
    # rows 1 and 2 are both gaps (one NaN each): interpolated between t=0 and t=3
    assert np.allclose(filled[1], np.array([0, 0, 10, 10]) + 4 / 3)
    assert np.allclose(filled[2], np.array([0, 0, 10, 10]) + 8 / 3)


def test_boxes_are_clipped_and_empty_ones_dropped():
    b = _boxes()
    b[50:, 0] = [-80, 100, 150, 760]  # partly outside: clipped
    b[10, 0] = [100, -300, 300, -10]  # entirely above the frame: empty after clipping
    sel, observed, _, _ = gw.prepare_boxes(b, 0, W, H)
    assert np.allclose(sel[55], [0, 100, 150, H - 1])
    assert not observed[10] and np.allclose(sel[10], [100, 100, 300, 600])  # held from neighbours


def test_all_missing_track_raises():
    b = _boxes()
    b[:, 0] = np.nan
    with pytest.raises(ValueError, match="empty on every frame"):
        gw.prepare_boxes(b, 0, W, H)


def test_worker_puts_pinned_ffmpeg_first(tmp_path, monkeypatch):
    (tmp_path / "ffmpeg").write_text("#!/bin/sh\n")
    (tmp_path / "ffmpeg").chmod(0o755)
    monkeypatch.setenv("AFV_FFMPEG_DIR", str(tmp_path))
    monkeypatch.setenv("PATH", f"/usr/bin:{tmp_path}:/bin")
    assert gw.ensure_ffmpeg_first_on_path() == str(tmp_path / "ffmpeg")
    import os

    assert os.environ["PATH"].split(os.pathsep)[:2] == [str(tmp_path), "/usr/bin"]
