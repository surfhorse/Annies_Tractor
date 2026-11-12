"""The cleanup tools (add-on slice 2) in headless Blender against a fake ``afv serve``
(``fake_serve.py``, run by Blender's own Python): every tool writes its edit, sends the right
requests, re-imports the new solve (motion only, or everything after a re-run from an early stage)
and keeps the tweak track; a paused run goes on with "Run anyway"; Cancel; the one-shot fallback."""

from __future__ import annotations

import json

import numpy as np
import pytest
from take_helpers import TESTS_DIR, addon_module, synthetic_shot
from test_blender_take import run_check

from afv import take as tk
from afv.core.types import Tracks
from afv.export import blender as bl
from afv.motion.rotations import yup_to_zup_vec

edits = addon_module("edits")

pytestmark = [
    pytest.mark.blender,
    pytest.mark.slow,
    pytest.mark.skipif(bl.find_blender() is None, reason="Blender not installed"),
]

N = 24


def _tracks(bm):
    n = bm.num_frames
    boxes = np.zeros((n, 2, 4), np.float32)
    boxes[:, 0] = (100, 40, 250, 330)
    boxes[:, 1] = (350, 30, 500, 340)
    t = Tracks(frames=bm.frames, timestamps=bm.timestamps, track_ids=np.array([1, 2]), boxes=boxes,
               scores=np.ones((n, 2), np.float32), fighters={"A": 1, "B": 2})
    return tk.tracks_on_frames(t, bm.frames, (1.0, 1.0), (640, 360))


@pytest.fixture(scope="module")
def shot(tmp_path_factory):
    out = tmp_path_factory.mktemp("bl_cleanup")
    proj = out / "proj"
    sdir = proj / "shots" / "007"
    sdir.mkdir(parents=True)
    (sdir / "shot.json").write_text(json.dumps({"id": "007", "start": 0.0, "duration": N / 30, "settings": {
        "impl": {}, "params": {"smooth": {"strength": 0.75}}}}))
    takes = []
    for i, seed in enumerate((0, 5, 7)):
        bm, cam = synthetic_shot(n_frames=N, seed=seed)
        inp = tk.TakeInputs(motion=bm, camera=cam, shot_id="007", project_dir=str(proj), shot_dir=sdir,
                            tracks=_tracks(bm))
        takes.append(tk.write_take(sdir / "take" if i == 0 else out / f"next{i}", inp))
        if i == 0:  # the fake worker overwrites the shot's take: keep the first one to compare with
            first = tk.write_take(out / "first", inp)
    return {"out": out, "shot_dir": sdir, "takes": takes, "first": first}


def _cfg(out, name, exe, **kw):
    log = out / f"{name}.jsonl"
    p = out / f"{name}.json"
    p.write_text(json.dumps({"log": bl.host_path(log, exe), **kw}))
    return bl.host_path(p, exe), bl.host_path(log, exe)


@pytest.fixture(scope="module")
def result(shot):
    exe = bl.find_blender()
    out = shot["out"]
    takes = [bl.host_path(t.parent, exe) for t in shot["takes"][1:]]
    cfg, log = _cfg(out, "main", exe, takes=takes)
    cfg_r, log_r = _cfg(out, "review", exe, takes=takes, review_once=True)
    cfg_s, log_s = _cfg(out, "slow", exe, takes=takes, run_s=60)
    why = "smooth: pin 0: bone 'Tail' is not in the rig"
    cfg_f, _log_f = _cfg(out, "fail", exe, takes=takes, fail_run=why)
    job = {"take": bl.host_path(shot["takes"][0], exe), "shot_dir": bl.host_path(shot["shot_dir"], exe),
           "fake": bl.host_path(TESTS_DIR / "fake_serve.py", exe), "cfg": cfg, "log": log,
           "cfg_review": cfg_r, "log_review": log_r, "cfg_slow": cfg_s, "log_slow": log_s, "cfg_fail": cfg_f}
    return run_check("cleanup", job, out / "cleanup")


def test_import_sets_the_panels(result):
    a = result["after_import"]
    assert a["range"] == [0, N - 1] and a["smooth"] == [0.75, 1.0] and a["tracks"] and a["prefix"] == ""


def test_pin_reads_the_takes_local_rotations(result, shot):
    """An unposed pin equals the take's ``rot`` (the pins.json convention) and the hips head."""
    meta, arrays = tk.load_take_arrays(shot["first"])
    f = meta["fighters"][0]
    names = [b["name"] for b in f["bones"]]
    p = result["pin_all"]
    sel = [p["bones"].index(n) for n in names]
    got = np.array(p["rot"])[sel]
    want = arrays["A"]["rot"][6]
    err = np.minimum(np.abs(got - want).max(-1), np.abs(got + want).max(-1))
    assert err.max() < 1e-5, err.max()
    hips = names.index(f["hips"])
    assert p["hips"] == f["hips"]
    np.testing.assert_allclose(p["root"], arrays["A"]["pose_head"][6, hips], atol=1e-5)


def test_paint_contact(result, shot):
    r = result["paint"]
    assert r["op"] == ["FINISHED"] and not r["error"], r["status"]
    assert r["contacts"] == {"spans": [{"role": "A", "start": 3, "end": 9, "state": "plant", "foot": "left"}]}
    assert r["mode"] == "POSE"  # still posing after the re-import
    meta, arrays = tk.load_take_arrays(shot["takes"][1])  # what the worker's ``take`` delivered
    names = [b["name"] for b in meta["fighters"][0]["bones"]]
    heads = np.array(r["heads"])[[r["bones"].index(n) for n in names]]
    err = np.linalg.norm(heads - yup_to_zup_vec(arrays["A"]["pose_head"][5]), axis=-1)
    assert err.max() < 1e-4, err.max()  # the new solve is what shows
    assert r["camera_same"] and r["frame"] == 5  # motion only: camera and frame untouched
    assert r["scene_end"] == 15  # ... also when the user shortened the scene range
    assert r["tracks"] == ["AFV solved", "AFV tweaks"] and r["tweak_action"]
    steps = [n for n, _ in r["timings"]["steps"]]
    assert steps == ["edit contacts", "run contacts 3-8", "take", "re-import"]
    assert r["timings"]["total"] > 0 and "afv serve: running" in r["worker"]
    log = result["log"]
    assert [x["cmd"] for x in log[:4]] == ["ping", "edit", "run", "take"]
    run = log[2]["args"]
    assert run["from"] == "contacts" and run["to"] == "hands_face" and run["frames"] == "3:9"
    assert run["shot"] == "007" and run["project"].endswith("/proj") and "shot_dir" not in run
    assert log[1]["args"]["name"] == "contacts" and "shot_dir" not in log[1]["args"]


def test_pins(result):
    r = result["pin"]
    assert r["op"] == ["FINISHED"], r["status"]
    (pin,) = r["pins"]["pins"]
    assert (pin["role"], pin["frame"], pin["radius"], pin["preset"]) == ("A", 10, 8, "unity_humanoid")
    assert list(pin["bones"]) == ["LeftForeArm"] and pin["root"] is None
    want = edits.basis_to_local(np.array(r["rest"]), np.array(r["basis"]))
    np.testing.assert_allclose(pin["bones"]["LeftForeArm"], want, atol=1e-6)
    pins = result["pin_root"]["pins"]["pins"]
    assert [p["frame"] for p in pins] == [10, 12]
    assert set(pins[1]["bones"]) == {"LeftForeArm", result["pin_all"]["hips"]}
    np.testing.assert_allclose(pins[1]["root"], result["pin_root"]["head"], atol=1e-5)
    assert all(p["root_motion"] == "world" and p["prefix"] == "" for p in pins)
    merged = result["pin_merged"]["pins"]  # a second pin at frame 12: bones merged, root kept
    assert [p["frame"] for p in merged] == [10, 12]
    assert set(merged[1]["bones"]) == {"LeftForeArm", "RightArm", result["pin_all"]["hips"]}
    assert merged[1]["bones"]["LeftForeArm"] == pins[1]["bones"]["LeftForeArm"]
    assert merged[1]["root"] == pins[1]["root"]
    assert [p["frame"] for p in result["pin_deleted"]["pins"]] == [12]
    runs = [x["args"] for x in result["log"] if x["cmd"] == "run" and x["args"]["from"] == "smooth"]
    assert len(runs) >= 3 and not any(r_.get("frames") for r_ in runs[:3])  # pins: smooth in full


def test_smoothing_hands_rerun_and_swap(result):
    smooth = [x["args"] for x in result["log"] if x["cmd"] == "run" and x["args"].get("sets")]
    assert smooth and smooth[0]["from"] == "smooth" and "frames" not in smooth[0]
    assert {"smooth.strength=0.5", "smooth.region_strength.arms=0.25"} <= set(smooth[0]["sets"])
    assert "smooth.region_strength.legs=1" in smooth[0]["sets"]
    t = result["smooth"]["timings"]
    assert t is not None, result["smooth"]["status"]
    assert [n for n, _ in t["steps"]] == ["run smooth", "take", "re-import"]
    sm = result["smooth"]
    assert sm["tweak_before"] and sm["tweak_after"] and sm["tweak_action"]  # NLA tweak mode kept
    assert result["hands"] == {"spans": [{"role": "A", "start": 3, "end": 9, "side": "left"}]}
    runs = [x["args"] for x in result["log"] if x["cmd"] == "run"]
    froms = [r["from"] for r in runs]
    i = froms.index("hands_face")
    assert runs[i]["frames"] == "3:9" and froms[i + 1] == "pose3d" and froms[i + 2] == "detect_track"
    assert result["rerun"]["camera_rebuilt"] and "frames" not in runs[i + 2]
    assert result["ids"]["ids"] == {"swaps": [{"start": 9, "end": None, "objects": [1, 2]}]}
    takes = [x["args"] for x in result["log"] if x["cmd"] == "take"]
    assert takes and all(a.get("check_plan") is False for a in takes)


def test_revert_last_fix(result):
    r = result["revert"]
    assert r["ops"] == [["FINISHED"]] * 3, r["status"]
    assert r["history_before"][-3:] == ["smoothing applied", "hands off A left 3-8", "swap #1 and #2 from 9"]
    assert r["after_swap"]["ids"] is None and r["hands"] is None  # both files had not existed
    assert r["history_after"] == r["history_before"][:-3]
    assert r["history_file"] and not r["old_file"]  # shots/<id>/.addon/history.json, not in edits/
    # smoothing: the shot had strength 0.75 and no region values; Apply set all seven; Revert puts
    # 0.75 back and removes the region keys (no explicit defaults)
    assert r["smooth_set"]["strength"] == 0.5 and r["smooth_set"]["region_strength"]["legs"] == 1
    assert r["smooth_reverted"]["strength"] == 0.75 and not r["smooth_reverted"].get("region_strength")
    assert r["sliders"] == [0.75, 1.0]
    runs = [x["args"] for x in result["log"] if x["cmd"] == "run"]
    assert [r_["from"] for r_ in runs[-3:]] == ["detect_track", "hands_face", "smooth"]
    assert runs[-2]["frames"] == "3:9"
    assert {"smooth.strength=0.75", "smooth.region_strength.arms=null"} <= set(runs[-1]["sets"])
    edits_sent = [x["args"] for x in result["log"] if x["cmd"] == "edit"]
    assert [(e["name"], e["data"]) for e in edits_sent[-2:]] == [("ids", None), ("hands", None)]


def test_failed_fix_is_reported_and_reverted(result):
    f = result["failed"]
    assert f["failed"]["step"].startswith("run contacts") and "not in the rig" in f["failed"]["message"]
    assert {"role": "A", "start": 20, "end": 23, "state": "free", "foot": "left"} in f["written"]["spans"]
    assert f["reverted"] == f["before"]


@pytest.fixture(scope="module")
def rigs(tmp_path_factory):
    """Takes on other rigs: unity_generic (a root bone) shown with world or extracted root motion,
    and a bone-name prefix."""
    out = tmp_path_factory.mktemp("bl_rigs")
    bm, cam = synthetic_shot(n_frames=12, seed=3)
    takes = {}
    for key, rig in (("generic_world_prefix", tk.RigSettings(preset="unity_generic", root_motion="world",
                                                              prefix="mixamorig:")),
                     ("generic_extracted", tk.RigSettings(preset="unity_generic", root_motion="extracted"))):
        inp = tk.TakeInputs(motion=bm, camera=cam, shot_id="007", project_dir=str(out / "proj"))
        takes[key] = tk.write_take(out / key, inp, rig)
    exe = bl.find_blender()
    job = {"takes": {k: bl.host_path(v, exe) for k, v in takes.items()}, "frame": 7,
           "tweak": "generic_world_prefix"}
    return {"takes": takes, "res": run_check("pins2", job, out / "pins2")}


@pytest.mark.parametrize("key", ["generic_world_prefix", "generic_extracted"])
def test_pins_on_other_rigs(rigs, key):
    """Canonical bone names (no prefix), no root bone, the take's root_motion/prefix in every pin,
    rotations = the take's ``rot`` and root = the hips head."""
    meta, arrays = tk.load_take_arrays(rigs["takes"][key])
    f = meta["fighters"][0]
    r = rigs["res"][key]
    prefix, mode = meta["rig"]["prefix"] or "", meta["rig"]["root_motion"]
    assert r["rec"] == {"preset": "unity_generic", "prefix": prefix, "root_motion": mode}
    pin = r["pin"]
    assert pin["prefix"] == prefix and pin["root_motion"] == mode and pin["preset"] == "unity_generic"
    names = [b["name"] for b in f["bones"]]
    root_bone = f["root_bone"]
    assert root_bone and edits.canonical(root_bone, prefix) not in pin["bones"]  # pin the hips instead
    for i, n in enumerate(names):
        if n == root_bone:
            continue
        c = edits.canonical(n, prefix)
        assert ":" not in c
        got, want = np.array(pin["bones"][c]), arrays["A"]["rot"][7, i]
        assert min(np.abs(got - want).max(), np.abs(got + want).max()) < 1e-5, n
    np.testing.assert_allclose(pin["root"], arrays["A"]["pose_head"][7, names.index(f["hips"])], atol=1e-5)
    got, want = np.array(pin["root_rotation"]), arrays["A"]["rot"][7, names.index(root_bone)]
    assert min(np.abs(got - want).max(), np.abs(got + want).max()) < 1e-5  # the root the user saw


def test_pin_takes_the_tweak_out(rigs):
    """With tweak keys (on a slid strip), an unposed pin is the solve itself: after the re-solve,
    solved (x) tweak is again what was shown."""
    meta, arrays = tk.load_take_arrays(rigs["takes"]["generic_world_prefix"])
    f = meta["fighters"][0]
    names = [b["name"] for b in f["bones"]]
    t = rigs["res"]["generic_world_prefix"]["tweak"]
    fore = edits.canonical(t["fore"], "mixamorig:")
    got, want = np.array(t["pin"]["bones"][fore]), arrays["A"]["rot"][7, names.index(t["fore"])]
    assert min(np.abs(got - want).max(), np.abs(got + want).max()) < 1e-5
    hips = names.index(f["hips"])
    np.testing.assert_allclose(t["pin"]["root"], arrays["A"]["pose_head"][7, hips], atol=1e-5)
    shown = np.array(t["shown_head"])
    assert np.abs(shown - arrays["A"]["pose_head"][7, hips]).max() > 0.01  # the tweak did show
    arm = edits.canonical(t["arm"], "mixamorig:")
    got, want = np.array(t["pin"]["bones"][arm]), arrays["A"]["rot"][7, names.index(t["arm"])]
    assert min(np.abs(got - want).max(), np.abs(got + want).max()) < 1e-5  # the pushed-down layer too


def _qmul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def test_tweak_layers_match_blender(rigs):
    """tweak_basis vs Blender's own evaluation (solve^-1 x shown), with two layers (the tweak track
    and a pushed-down track at half influence holding a negative-w key: Blender's q^t takes the
    long way there, a short-path slerp would be off by tens of degrees); a non-Combine layer is
    refused."""
    t = rigs["res"]["generic_world_prefix"]["tweak"]
    for bone in (t["fore"], t["arm"]):
        s_inv = np.array(t["solved"][bone]) * np.array([1, -1, -1, -1])
        blender = _qmul_wxyz(s_inv, np.array(t["shown"][bone]))
        ours = np.array(t["ours"][bone])
        ang = np.degrees(2 * np.arccos(min(1.0, abs(float(blender @ ours)))))
        assert ang < 0.01, (bone, ang)
    assert t["refused"] and "Combine" in t["refused"] and "my layer" in t["refused"]


def test_paused_run_goes_on(result):
    assert result["review"]["kind"] == "run" and result["review"]["hints"] == [{"kind": "swap"}]
    assert result["go_on"]["op"] == ["FINISHED"] and not result["go_on"]["error"], result["go_on"]["status"]
    runs = [x["args"] for x in result["go_on"]["log"] if x["cmd"] == "run"]
    assert [r.get("yes") for r in runs] == [False, True]
    n_before, n_paused, n_after = result["go_on"]["history"]
    assert n_paused == n_before + 1 and n_after == n_paused  # recorded when it paused, once


def test_cancel_and_fallback(result):
    c = result["cancel"]
    assert not c["job"] and "cancelled" in c["status"] and c["log"][-1] == "cancel"
    f = result["fallback"]
    assert not f["error"], f["status"]
    assert "afv serve not available" in f["serve_off"] and "one-shot" in f["worker"]
    assert [c[0] for c in f["calls"]] == ["run", "take"]
    assert f["calls"][0][f["calls"][0].index("--frames") + 1] == "15:19"
    assert {"role": "A", "start": 15, "end": 19, "state": "free", "foot": "right"} in f["contacts"]["spans"]
    assert result["restart"] == {"serve_off": "", "client_gone": True, "had_client": True}
