"""The candidate scan gives exactly ``track.detect_prompt``'s prompt (GPU, real track worker, opt-in).

On real candidates of the owner's ``Full Fight.mp4`` (git-ignored): one ``scan_shots`` request for all of
them, then for each shot the way ``afv run``/``afv pick`` get a prompt (``prepare_frames`` +
``detect_prompt`` on the scan's frame): same frame, same boxes in the same order, same scores.

Run with ``AFV_GPU_TESTS=1 ~/afv/afv-env/bin/python -m pytest tests/models/test_scan_gpu.py``. About two
minutes; holds the GPU lock through ``run_worker``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from afv.core.project import Project, load_scan
from afv.media import shots as ms

REPO = Path(__file__).parents[2]
MEDIA = Path(os.environ.get("AFV_MEDIA_SAMPLES") or next(
    (p for p in (REPO / "Media_Samples", REPO / "Media Samples") if p.is_dir()), REPO / "Media_Samples"))
FULL_FIGHT = MEDIA / "Full Fight.mp4"
# candidates 001 (0.58 s), 036 (fight, 3 people), 041 (fight), 087 (post-fight crowd), 126 (last, nobody)
CANDIDATES = {"001": (0.0, 0.583933), "036": (287.73745, 29.329316), "041": (342.959283, 5.305317),
              "087": (720.87015, 4.738083), "126": (978.794483, 2.385716)}

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("AFV_GPU_TESTS") != "1", reason="set AFV_GPU_TESTS=1 to run GPU tests"),
    pytest.mark.skipif(not Path("~/afv/track/.venv/bin/python").expanduser().exists(), reason="no track env"),
]


def test_scan_prompts_equal_detect_prompt(tmp_path):
    from afv.cli import _probe
    from afv.models import track

    if not FULL_FIGHT.is_file():
        pytest.fail(f"missing {FULL_FIGHT} (set AFV_MEDIA_SAMPLES to the clips' folder)")
    probe, _ = _probe(FULL_FIGHT)
    proj = Project.create(tmp_path / "ff.afv", FULL_FIGHT, probe=probe)
    for sid, (start, dur) in CANDIDATES.items():
        proj.add_shot(start, dur, shot_id=sid, state="candidate")
    rep = ms.scan_shots(proj, proj.shots())
    assert sorted(rep.scanned) == sorted(CANDIDATES) and not rep.failed and not rep.error
    people = {}
    for sid, (start, dur) in CANDIDATES.items():
        scan = load_scan(proj.shot(sid))
        assert scan is not None
        fr = track.prepare_frames(FULL_FIGHT, start, dur, tmp_path / "frames" / sid)
        got = track.detect_prompt(FULL_FIGHT, tmp_path / "prompts" / sid, [scan["frame"]],
                                  threshold=scan["prompt"]["threshold"], shot_frames=fr)[0]
        want = track.PromptInfo.from_json(scan["prompt"])
        assert got.frame == want.frame == scan["frame"]
        assert got.boxes.tolist() == want.boxes.tolist()  # same people, same numbering
        assert got.scores.tolist() == want.scores.tolist()
        people[sid] = len(got.boxes)
    assert people["036"] >= 2 and people["126"] == 0  # a fight shot; the closing card has nobody
