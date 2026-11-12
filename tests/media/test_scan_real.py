"""Scan frames of real candidate shots are byte-identical to ``prepare_frames``' frames (CPU, slow).

Uses the owner's ``Full Fight.mp4`` (git-ignored; skipped when missing): candidate intervals found by
``afv shots detect`` on it, including its first (0.58 s) and last shots.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from afv.media import shots as ms

REPO = Path(__file__).parents[2]
MEDIA = Path(os.environ.get("AFV_MEDIA_SAMPLES") or next(
    (p for p in (REPO / "Media_Samples", REPO / "Media Samples") if p.is_dir()), REPO / "Media_Samples"))
FULL_FIGHT = MEDIA / "Full Fight.mp4"
# (start, duration) of candidates 001, 002, 036, 041, 087 and 126 of Full Fight.mp4
CANDIDATES = [(0.0, 0.583933), (0.984317, 16.649982), (287.73745, 29.329316), (342.959283, 5.305317),
              (720.87015, 4.738083), (978.794483, 2.385716)]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not FULL_FIGHT.is_file(), reason=f"needs the owner's clip {FULL_FIGHT}"),
]


@pytest.mark.parametrize(("start", "duration"), CANDIDATES)
def test_real_candidate_scan_frame_is_exact(tmp_path, start, duration):
    from afv.models.track import prepare_frames

    k = ms.extract_scan_frame(FULL_FIGHT, start, duration, tmp_path / "f.jpg", tmp_path / "t.jpg")
    fr = prepare_frames(FULL_FIGHT, start, duration, tmp_path / "cache")
    assert k == min(ms.SCAN_FRAME, fr.num_frames - 1)
    a = hashlib.sha256((tmp_path / "f.jpg").read_bytes()).hexdigest()
    assert a == hashlib.sha256((fr.frames_dir / f"{k:05d}.jpg").read_bytes()).hexdigest()
    assert ms.read_jpeg(tmp_path / "t.jpg").shape == (270, 480, 3)
