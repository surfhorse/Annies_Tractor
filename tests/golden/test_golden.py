"""Golden-file regression on three Media Samples shots: the whole pipeline (ingest -> export) with
``deterministic=True``, compared stage by stage against ``golden_shots.json`` (numbers with
tolerances, ids, file names, the normalised shot's sha256; no footage or frames are stored).

Opt-in: ``AFV_GPU_TESTS=1`` (GPU, model environments, Blender, the clips in ``Media Samples``).
Each shot takes several minutes; run one at a time, e.g. ``-k close_boxing``.
``AFV_GOLDEN_UPDATE=1`` rewrites the shot's entry instead of comparing (review the diff);
``AFV_GOLDEN_DUMP=<dir>`` also writes this run's values to ``<dir>/<shot>.json`` (tolerance calibration).
Projects go to ``$AFV_GOLDEN_DIR`` (default ``<repo>/projects/_golden``, git-ignored; on the Windows
drive so Blender can write the exports) and are rebuilt from scratch each time, unless
``AFV_GOLDEN_RESUME=1`` (continue an interrupted run from its cached stages, e.g. after a timeout while
another job held the GPU).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from golden import compare, export60_summary, summarise_shot
from typer.testing import CliRunner

from afv.cli import app
from afv.core.project import Project
from afv.core.runner import Runner

REPO = Path(__file__).resolve().parents[2]
# The sample clips' folder (git-ignored): AFV_MEDIA_SAMPLES, else Media_Samples/ (round 5 name), else the
# older "Media Samples/".
MEDIA = Path(os.environ.get("AFV_MEDIA_SAMPLES") or next(
    (p for p in (REPO / "Media_Samples", REPO / "Media Samples") if p.is_dir()), REPO / "Media_Samples"))
GOLDEN = Path(__file__).with_name("golden_shots.json")
# (clip, start, duration, fighters, extra, prompt boxes of the picks on frame 0 for a sanity check)
SHOTS: dict[str, dict] = {
    "close_boxing": {"clip": "Close boxing short.mp4", "start": 0.0, "duration": 3.036366,
                     "fighters": [1, 0], "extra": [2]},
    "ufc_cross": {"clip": "UFC cross.mp4", "start": 0.0, "duration": 10.043366,
                  "fighters": [1, 0], "extra": [2]},
    "who_am_i": {"clip": "Who Am I.mp4", "start": 0.0, "duration": 4.045708,
                 "fighters": [0, 1], "extra": []},
}  # fmt: skip

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.blender,
    pytest.mark.skipif(not os.environ.get("AFV_GPU_TESTS"), reason="GPU golden runs: set AFV_GPU_TESTS=1"),
]


def _golden() -> dict:
    return json.loads(GOLDEN.read_text()) if GOLDEN.is_file() else {}


@pytest.mark.parametrize("name", sorted(SHOTS))
def test_golden_shot(name):
    spec = SHOTS[name]
    src = MEDIA / spec["clip"]
    if not src.is_file():  # opted in (AFV_GPU_TESTS) but the clip is gone: fail, never pass silently
        pytest.fail(f"missing sample clip {src} (set AFV_MEDIA_SAMPLES to the clips' folder)")
    root = Path(os.environ.get("AFV_GOLDEN_DIR", REPO / "projects" / "_golden")) / name
    if os.environ.get("AFV_GOLDEN_RESUME") and (root / "project.json").is_file():
        shot = Project(root).shot("1")
        shot.edit_path("fighters").unlink(missing_ok=True)  # re-picked below with the same numbers
    else:
        shutil.rmtree(root, ignore_errors=True)
        shot = Project.create(root, src).add_shot(spec["start"], spec["duration"])
    info = shot.info
    info.settings.params = {"export": {"formats": ["fbx", "glb", "bvh"]}}
    shot.save_info(info)
    first = Runner(shot, deterministic=True).run(to="detect_track")
    assert first.outcome == "needs_pick", first.message
    want = _golden().get(name, {})
    prompt = json.loads((shot.stage_dir("detect_track") / "prompt.json").read_text())
    picks = spec["fighters"] + spec["extra"]
    boxes = [prompt["boxes"][k] for k in picks]
    if "pick_boxes" in want and not os.environ.get("AFV_GOLDEN_UPDATE"):  # same people under the numbers
        assert np.allclose(boxes, want["pick_boxes"], atol=6.0), (boxes, want["pick_boxes"])
    args = ["pick", shot.id, "--fighters", ",".join(map(str, spec["fighters"])), "-p", str(root)]
    if spec["extra"]:
        args += ["--extra", ",".join(map(str, spec["extra"]))]
    res = CliRunner().invoke(app, args, catch_exceptions=False)
    assert res.exit_code == 0, res.output
    rep = Runner(shot, deterministic=True).run(to="export")
    assert rep.outcome == "ok", rep.message
    got = {"pick_boxes": [[round(float(v), 1) for v in b] for b in boxes], **summarise_shot(shot)}
    # the same motion exported again at 60 fps (retime) as BVH only
    rep60 = Runner(
        shot, deterministic=True, params={"export": {"fps": 60.0, "formats": ["bvh"], "name": "golden60"}}
    ).run(to="export")
    assert rep60.outcome == "ok", rep60.message
    got.update(export60_summary(shot, "golden60"))
    if os.environ.get("AFV_GOLDEN_DUMP"):  # keep this run's values, e.g. to calibrate tolerances
        dump = Path(os.environ["AFV_GOLDEN_DUMP"])
        dump.mkdir(parents=True, exist_ok=True)
        (dump / f"{name}.json").write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
    if os.environ.get("AFV_GOLDEN_UPDATE"):
        data = _golden()
        data[name] = got
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        return
    assert want, f"no golden entry for {name}: run with AFV_GOLDEN_UPDATE=1 and review it"
    got.pop("pick_boxes")
    want = {k: v for k, v in want.items() if k != "pick_boxes"}
    diffs = compare(got, want)
    assert not diffs, f"{name}: {len(diffs)} differences from the golden file:\n" + "\n".join(diffs)
