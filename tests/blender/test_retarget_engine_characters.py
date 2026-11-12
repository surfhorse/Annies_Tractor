"""The retarget engine on the owner's three test characters (git-ignored ``Media_Samples/``) x the real
takes of ``projects/ufc_cross`` and ``projects/close_boxing`` (shot 001), both fighters, with and
without lock_feet. Each character is its own parametrized case and skips (saying why) when its file
or the projects are absent. Nothing is written into the projects: the takes are built into tmp, the
scene only lives in headless Blender. Everything is measured from the evaluated scene, penetration on
the evaluated foot mesh (plus toe tip and heel), independently of the module's own report.

Acceptance (track R1): guess_map maps every required bone (never the wrong side, never a bone twice);
lock_feet=False: bone directions within 1 deg of the source, twist within 2 deg (feet: their whole
rest-relative rotation within 1 deg); lock_feet=True: median planted-foot speed <= 3 cm/s,
penetration <= 1.5 cm on planted frames and never > 1.5 cm deeper than the source's own feet
elsewhere, peak ankle acceleration within 5 frames of a lock edge <= 1.5 x the free result's;
a tweak key moves the result; nothing else in the target changes; <= 10 s per fighter.
Set ``AFV_RETARGET_DUMP=<dir>`` to keep the per-run results as ``<dir>/characters_<name>.json``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from retarget_helpers import CHARACTERS, FACING, dump, edge_accel, penetration, planted, run_checks
from take_helpers import REPO

from afv import take as tk
from afv.core.project import Project
from afv.core.runner import Runner
from afv.export import blender as bl

pytestmark = [
    pytest.mark.blender,
    pytest.mark.slow,
    pytest.mark.skipif(bl.find_blender() is None, reason="Blender not installed"),
]

PROJECTS = Path(os.environ.get("AFV_REAL_PROJECTS") or REPO / "projects")
TAKES = ("ufc_cross", "close_boxing")
RUNS = [{"role": r, "lock_feet": lock} for r in ("A", "B") for lock in (False, True)]
SIDES = (("Left", "L"), ("Right", "R"))

EXPECTED = {
    "mixamo": {c: "mixamorig:" + c for c in (
        "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "HeadTop_End",
        *[f"{s}{b}" for s, _ in SIDES for b in ("Shoulder", "Arm", "ForeArm", "Hand", "UpLeg", "Leg", "Foot",
                                                 "ToeBase", "Toe_End")])},
    "unity": {"Hips": "Hips", "Spine": "Spine", "Spine1": "Chest", "Spine2": "UpperChest", "Neck": "Neck",
              "Head": "Head",
              **{f"{s}{c}": f"{s}_{u}" for s, _ in SIDES for c, u in (
                  ("Shoulder", "Shoulder"), ("Arm", "UpperArm"), ("ForeArm", "LowerArm"), ("Hand", "Hand"),
                  ("UpLeg", "UpperLeg"), ("Leg", "LowerLeg"), ("Foot", "Foot"), ("ToeBase", "Toes"),
                  ("Toe_End", "ToesEnd"))}},
    "girush": {"Hips": "Sacrum", "Spine": "Spine1", "Spine1": "Spine2", "Spine2": "Spine3",
               "Neck": "Spine3.002", "Head": "head",
               **{f"{s}{c}": f"{g}.{x}" for s, x in SIDES for c, g in (
                   ("Shoulder", "Clavicle"), ("Arm", "Humerus"), ("ForeArm", "LowerArm"), ("Hand", "hand001"),
                   ("UpLeg", "Femur"), ("Leg", "Tibia"), ("Foot", "Foot"))},
               **{f"{s}ToeBase": f"Foot.{x}.001" for s, x in SIDES}},
}
GIRUSH_FINGERS = {f"{s}Hand{f}{k}": f"hand{n:03d}.{x}" for s, x in SIDES
                  for f, ids in (("Thumb", (2, 3, 4)), ("Index", (5, 7, 8)), ("Middle", (9, 10, 11)),
                                 ("Ring", (12, 13, 14)), ("Pinky", (15, 16, 17)))
                  for k, n in enumerate(ids, start=1)}
FINGERS = [f"{s}Hand{f}{k}" for s, _ in SIDES for f in ("Thumb", "Index", "Middle", "Ring", "Pinky")
           for k in (1, 2, 3, 4)]


@pytest.fixture(scope="module")
def takes(tmp_path_factory):
    have = [t for t in TAKES if (PROJECTS / t / "shots" / "001" / "shot.json").is_file()]
    if len(have) != len(TAKES):
        pytest.skip(f"sample projects missing under {PROJECTS}: {sorted(set(TAKES) - set(have))}")
    root = tmp_path_factory.mktemp("retarget_takes")
    out = {}
    for t in TAKES:
        sh = Project(PROJECTS / t).shot("001")
        with Runner(sh).shot_lock():
            out[t] = tk.make_take(sh, root / t / "take")
    return out


@pytest.fixture(scope="module", params=sorted(CHARACTERS))
def char(request, takes, tmp_path_factory):
    name = request.param
    if not CHARACTERS[name].is_file():
        pytest.skip(f"test character {name} missing: {CHARACTERS[name]}")
    exe = bl.find_blender()
    root = tmp_path_factory.mktemp(f"retarget_{name}")
    res = {}
    for t, take in takes.items():
        job = {"take": bl.host_path(take, exe), "character": bl.host_path(CHARACTERS[name], exe),
               "facing": FACING[name], "runs": RUNS}
        res[t] = run_checks("character", job, root / t)
    dump(f"characters_{name}", res)
    return name, res


def _runs(res, lock=None):
    for t, v in res.items():
        for r in v["runs"]:
            if lock is None or r["lock_feet"] == lock:
                yield f"{t}/{r['role']}/{'locked' if r['lock_feet'] else 'free'}", r


def _free_of(res, take, role):
    return next(r for r in res[take]["runs"] if r["role"] == role and not r["lock_feet"])


def test_guess_map(char):
    """Acceptance 4: every required bone, the expected body bones, fingers on the right side only."""
    name, res = char
    for v in res.values():
        assert v["missing"] == [], v["missing"]
        assert v["sides_bad"] == [] and v["duplicates"] == 0
        body = {k: n for k, n in v["guess"].items() if k not in FINGERS}
        assert body == EXPECTED[name], body
        fingers = {k: n for k, n in v["guess"].items() if k in FINGERS}
        if name == "girush":
            assert fingers == GIRUSH_FINGERS
        else:
            assert len(fingers) == 40, sorted(set(FINGERS) - set(fingers))


def test_free_motion_follows_the_source(char):
    _, res = char
    for run_name, r in _runs(res, lock=False):
        m = r["measure"]
        print(run_name, {k: round(m[k], 4) for k in ("dir_err_deg", "twist_err_deg", "rot_err_deg",
                                                    "feet_rot_deg", "hips_rot_deg", "anchor_err_cm")})
        assert m["dir_err_deg"] <= 1.0, (run_name, m["worst"])
        assert m["twist_err_deg"] <= 2.0 and m["rot_err_deg"] <= 2.0, (run_name, m["worst"])
        assert m["feet_rot_deg"] <= 1.0, (run_name, m["worst"])
        assert m["hips_rot_deg"] <= 1.0 and m["anchor_err_cm"] <= 0.1, (run_name, m["anchor_err_cm"])


def test_facing_and_scale_are_what_blender_shows(char):
    _, res = char
    for run_name, r in _runs(res):
        rep, m = r["report"], r["measure"]
        assert abs(rep["facing_yaw_deg"] - m["yaw_deg"]) < 0.01, run_name
        assert abs(rep["hips_scale"] - m["k_auto"]) <= 1e-4 * m["k_auto"], (run_name, rep["hips_scale"],
                                                                            m["k_auto"])


def test_locked_feet(char):
    """Acceptance 2 on the evaluated scene (the character's own foot mesh)."""
    _, res = char
    for run_name, r in _runs(res, lock=True):
        m, rep = r["measure"], r["report"]
        take, role = run_name.split("/")[:2]
        pen, acc = penetration(m), edge_accel(m, _free_of(res, take, role)["measure"])
        print(run_name, "slide", m["slide_cm_s"], "pen", pen, "edges", acc, rep["notes"])
        assert m["mesh_feet"], run_name
        for side in ("left", "right"):
            assert m["slide_cm_s"][side + "_frames"] > 0, (run_name, side)  # real takes plant both feet
            assert m["slide_cm_s"][side] <= 3.0, (run_name, side, m["slide_cm_s"])
        for side, p in pen.items():
            assert p["planted_frames"] > 0, (run_name, side)
            assert p["planted_max"] <= 1.5 and p["other_excess_max"] <= 1.5, (run_name, side, p)
        for side, edges in acc.items():
            assert edges, (run_name, side)
            bad = [e for e in edges if not e["ok"]]
            assert not bad, (run_name, side, bad)
        # the whole mesh (every skinned mesh, Rest Position floor), not only the foot vertices
        whole = np.array(m["whole_mesh_pen"])
        src = np.max([np.array(sd["src_depth"]) for sd in m["sides"].values()], axis=0)
        assert (whole - src).max() <= 1.5, (run_name, float((whole - src).max()))
        both = np.all([planted(sd) for sd in m["sides"].values()], axis=0)
        assert whole[both].max(initial=0.0) <= 1.5, (run_name, float(whole[both].max(initial=0.0)))


def test_locked_report_leaves_out_the_ik_legs(char):
    """With lock_feet the IK'd legs (listed in ik_bones) are left out of the error maxima, so the
    report and Blender both show the rest of the body within 1 deg / 2 deg."""
    name, res = char
    for run_name, r in _runs(res, lock=True):
        rep, m = r["report"], r["measure"]
        legs = [EXPECTED[name][f"{s}{b}"] for s, _ in SIDES for b in ("UpLeg", "Leg", "Foot")]
        assert set(legs) <= set(rep["ik_bones"]), (run_name, rep["ik_bones"])
        assert rep["max_dir_err_deg"] <= 1.0 and rep["max_twist_err_deg"] <= 2.0, (run_name, rep)
        assert m["dir_err_deg"] <= 1.0 and m["twist_err_deg"] <= 2.0, (run_name, m["worst"])
    for _run_name, r in _runs(res, lock=False):
        assert r["report"]["ik_bones"] == []


def test_tweak_key_changes_the_result(char):
    """Acceptance 3 on the real character: +40 deg on LeftForeArm in fighter A's tweak track."""
    _, res = char
    for v in res.values():
        t = v["tweak"]
        assert min(t["changed_deg"]) > 15.0, t["changed_deg"]
        assert t["measure"]["dir_err_deg"] <= 1.0 and t["measure"]["twist_err_deg"] <= 2.0


def test_nothing_else_changes_and_speed(char):
    _, res = char
    for run_name, r in _runs(res):
        assert r["changed"] == [], (run_name, r["changed"])
        assert r["assigned"] and r["fake_user"]
        assert r["report"]["seconds"] <= 10.0, (run_name, r["report"]["seconds"])


def test_report_agrees_with_blender(char):
    _, res = char
    for run_name, r in _runs(res):
        rep, m = r["report"], r["measure"]
        assert abs(rep["penetration_cm"] - max(m["pen_cm"], 0.0)) <= 0.2, (run_name, rep["penetration_cm"],
                                                                           m["pen_cm"])
        for side in ("left", "right"):
            a, b = rep["foot_slide_cm_s"][side], m["slide_cm_s"][side]
            assert (a is None) == (b is None), run_name
            if a is not None:
                assert abs(a - b) <= 0.3, (run_name, side, a, b)
