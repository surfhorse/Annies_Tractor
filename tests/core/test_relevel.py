"""The runner's gravity re-level pass: after pose3d, ``afv.models.camera.relevel_override(pose3d qc)``
may ask for camera params (``up_cam0``); camera and pose3d then run again, exactly once, the record
goes to ``relevel.json`` + the camera meta (provenance), and later runs redo nothing."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import BaseModel

import afv.models.camera  # noqa: F401  (imported by the runner's re-level check; register it now, not
from afv.core.provenance import format_provenance, project_provenance  # into the test's fake registry)
from afv.core.runner import RELEVEL_FILE, Runner
from afv.core.stage import register
from afv.core.types import ActorMotion, BodyMotion, CameraTrack

T = 10
UP = [0.0, -0.99, 0.12]
BAD = {"pitch_difference_deg": 6.0, "height_change_if_gemx_m": 1.4, "gemx_up_cam0": UP}
GOOD = {"pitch_difference_deg": 0.3, "height_change_if_gemx_m": 0.05, "gemx_up_cam0": UP}


def _tl():
    return {"frames": np.arange(T), "timestamps": np.arange(T) / 30.0}


@pytest.fixture
def stages(fakes):
    """Camera with ``up_cam0``; pose3d whose qc gravity disagrees unless the camera was re-levelled (or
    always, with ``state["always_bad"]``)."""
    state = {"camera": [], "pose3d": 0, "always_bad": False, "never_bad": False, "force": []}

    @register("vipe")
    class Camera:
        name, version, env, inputs, licences, weight_hash = "camera", "1", "cpu", ("detect_track",), (), ""

        class Params(BaseModel):
            focal: float = 1000.0
            up_cam0: list[float] | None = None

        def run(self, ctx, inputs, params):
            state["camera"].append(params.up_cam0)
            state["force"].append(("camera", ctx.force))
            (ctx.work_dir / "qc.json").write_text(json.dumps({"summary": {"mode": "vipe"}}))
            tilt = 0.0 if params.up_cam0 is None else params.up_cam0[2]
            return CameraTrack(**_tl(), focal_px=np.full(T, 1000.0 + tilt, np.float32))

    @register("default")
    class Pose:
        name, version, env, inputs, licences, weight_hash = (
            "pose3d", "1", "cpu", ("detect_track", "camera"), (), "")  # fmt: skip

        class Params(BaseModel):
            pass

        def run(self, ctx, inputs, params):
            state["pose3d"] += 1
            state["force"].append(("pose3d", ctx.force))
            relevelled = float(inputs["camera"].focal_px[0]) != 1000.0
            bad = state["always_bad"] or (not relevelled and not state["never_bad"])
            flags = ["gravity_pitch_disagrees", "camera_below_floor"] if bad else []
            qc = {"summary": {"flags": flags}, "gravity": BAD if bad else GOOD}
            (ctx.work_dir / "qc.json").write_text(json.dumps(qc))
            rot = np.zeros((T, 2, 4), np.float32)
            rot[..., 3] = 1
            return BodyMotion(**_tl(), skeleton="soma_provisional", fps=30.0,
                              actors={"A": ActorMotion(rot=rot, root=np.zeros((T, 3), np.float32))})

    return state


@pytest.fixture
def picked(shot, do_pick):
    do_pick(shot)
    return shot


def test_relevel_runs_camera_and_pose3d_again_once_and_is_cached(stages, picked):
    rep = Runner(picked).run(to="pose3d")
    assert rep.outcome == "ok"
    assert stages["camera"] == [None, UP] and stages["pose3d"] == 2
    rec = json.loads((picked.dir / RELEVEL_FILE).read_text())
    assert rec["override"] == {"up_cam0": UP} and "gravity_pitch_disagrees" in rec["trigger"]["flags"]
    assert rep.relevel["override"] == {"up_cam0": UP}
    meta = picked.meta("camera")
    assert meta["params"]["up_cam0"] == UP and meta["relevel"]["override"] == {"up_cam0": UP}
    assert meta["qc"]["relevel"]["override"] == {"up_cam0": UP}
    ran = [s.name for s in rep.stages if s.state == "ran"]
    assert ran.count("camera") == 2 and ran.count("pose3d") == 2
    # provenance shows it
    prov = project_provenance(picked.project, [picked.id], hash_exports=False)
    assert prov["shots"][0]["stages"]["camera"]["relevel"]["override"] == {"up_cam0": UP}
    assert "re-levelled from pose3d's gravity" in format_provenance(prov)

    # a second run redoes nothing, and status agrees
    rep2 = Runner(picked).run(to="pose3d")
    assert stages["camera"] == [None, UP] and stages["pose3d"] == 2
    assert all(s.state in ("fresh",) for s in rep2.stages) and rep2.relevel is not None
    plan = {s.name: s.state for s in Runner(picked).plan(to="pose3d").stages}
    assert plan["camera"] == "fresh" and plan["pose3d"] == "fresh"


def test_relevel_happens_at_most_once_per_run(stages, picked):
    stages["always_bad"] = True  # even after the re-level pose3d still disagrees
    assert Runner(picked).run(to="pose3d").outcome == "ok"
    assert stages["camera"] == [None, UP] and stages["pose3d"] == 2
    Runner(picked).run(to="pose3d")  # and later runs keep the one re-level
    assert stages["camera"] == [None, UP] and stages["pose3d"] == 2


def test_no_relevel_when_gravity_agrees_and_stale_record_is_dropped(stages, picked):
    stages["never_bad"] = True
    stale = {"base_camera_key": "old", "override": {"up_cam0": UP}}
    (picked.dir / RELEVEL_FILE).write_text(json.dumps(stale))
    rep = Runner(picked).run(to="pose3d")
    assert stages["camera"] == [None] and stages["pose3d"] == 1 and rep.relevel is None
    assert not (picked.dir / RELEVEL_FILE).exists()


def test_new_camera_inputs_make_a_new_decision(stages, picked, do_pick):
    Runner(picked).run(to="pose3d")
    old = json.loads((picked.dir / RELEVEL_FILE).read_text())["base_camera_key"]
    do_pick(picked, fighters=(0, 1))  # tracking changes: the old record no longer applies
    Runner(picked).run(to="pose3d")
    assert stages["camera"] == [None, UP, None, UP] and stages["pose3d"] == 4
    assert json.loads((picked.dir / RELEVEL_FILE).read_text())["base_camera_key"] != old


def test_users_own_up_cam0_wins(stages, picked):
    mine = [0.0, -1.0, 0.0]
    rep = Runner(picked, params={"camera": {"up_cam0": mine}}).run(to="pose3d")
    assert rep.outcome == "ok" and stages["camera"][0] == mine
    assert all(u == mine for u in stages["camera"])  # a recorded override never replaces it
    assert not (picked.dir / RELEVEL_FILE).exists()  # and none is recorded


def test_ctx_force_only_for_user_forced_stages_never_for_the_relevel_pass(stages, picked):
    Runner(picked).run(to="pose3d")  # stale runs and the automatic re-level pass: not forced
    assert stages["force"] == [("camera", False), ("pose3d", False), ("camera", False), ("pose3d", False)]
    stages["force"].clear()
    Runner(picked).run(to="pose3d", from_stage="camera")  # afv run --from camera
    assert stages["force"] == [("camera", True), ("pose3d", True)]  # re-level record still applies: once


def test_forced_run_that_triggers_a_relevel_does_not_force_the_relevel_pass(stages, picked):
    Runner(picked).run(to="detect_track")
    stages["force"].clear()
    Runner(picked).run(to="pose3d", from_stage="detect_track")  # forced, and no re-level yet
    assert stages["force"] == [("camera", True), ("pose3d", True), ("camera", False), ("pose3d", False)]


def test_force_after_a_relevel_stays_on_for_later_forced_stages(stages, picked):
    """--from camera: camera and pose3d forced in pass 1; the re-level pass runs camera and pose3d
    unforced; smooth (after pose3d, not run in pass 1) is still forced in pass 2."""

    @register("default")
    class Smooth:
        name, version, env, inputs, licences, weight_hash = "smooth", "1", "cpu", ("pose3d",), (), ""

        class Params(BaseModel):
            pass

        def run(self, ctx, inputs, params):
            stages["force"].append(("smooth", ctx.force))
            return inputs["pose3d"]

    Runner(picked).run(to="detect_track")
    stages["force"].clear()
    Runner(picked).run(to="smooth", from_stage="camera")
    assert stages["force"] == [
        ("camera", True), ("pose3d", True), ("camera", False), ("pose3d", False), ("smooth", True)]
    stages["force"].clear()
    Runner(picked).run(to="smooth")  # nothing out of date, nothing forced
    assert stages["force"] == []
    Runner(picked, params={"camera": {"focal": 999.0}}).run(to="smooth")  # out of date: not forced
    assert all(f is False for _, f in stages["force"]) and stages["force"]


def test_a_record_of_another_decision_rule_is_decided_again(stages, picked):
    """[Z, round 3] relevel.json stores the decision rule (``afv.models.camera.RELEVEL_RULE``); a record
    made by another rule is not applied: camera and pose3d run without it and the decision is made
    again (here: the same re-level, recorded with the current rule)."""
    from afv.models.camera import RELEVEL_RULE

    assert Runner(picked).run(to="pose3d").outcome == "ok"
    rec = json.loads((picked.dir / RELEVEL_FILE).read_text())
    assert rec["rule"] == RELEVEL_RULE
    rec["rule"] = "0"  # an older rule
    (picked.dir / RELEVEL_FILE).write_text(json.dumps(rec))
    Runner(picked).run(to="pose3d")
    assert stages["camera"] == [None, UP, None, UP]  # decided again, from the camera without it
    assert json.loads((picked.dir / RELEVEL_FILE).read_text())["rule"] == RELEVEL_RULE
