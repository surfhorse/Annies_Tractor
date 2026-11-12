"""The retarget engine (``blender/afv_addon/retarget.py``) on synthetic rigs, in headless Blender.

Source: a synthetic 300-frame take (fighter A, plant masks on its left foot) and the same take on a
``unity_generic`` rig with the ``mixamorig:`` prefix and a root bone. Targets are built in Blender by
``retarget_checks.py`` (see there): Rigify-style, anatomical centimetre rig with a Copy Rotation knee,
Unity names under a 0.01-scaled object facing +Y, turned and parented copies, a 3ds Max Biped,
Blender's real Rigify metarig and a generated Rigify rig. Every number is measured in Blender from
the evaluated scene, independently of the module's maths and report (acceptance 1-5 of track R1).
Set ``AFV_RETARGET_DUMP=<dir>`` to keep the full results as ``<dir>/synthetic.json``.
"""

from __future__ import annotations

import json

import pytest
from retarget_helpers import dump, edge_correction_accel, penetration, run_checks
from take_helpers import synthetic_shot

from afv import take as tk
from afv.export import blender as bl

pytestmark = [
    pytest.mark.blender,
    pytest.mark.slow,
    pytest.mark.skipif(bl.find_blender() is None, reason="Blender not installed"),
]

N = 300
RIGS = ("syn_rigify", "syn_anatomical", "syn_unity_cm")
TURNED = {"syn_rot10": -10.0, "syn_rot90": -90.0, "syn_parented": -65.0}  # expected facing yaw (deg)


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    out = tmp_path_factory.mktemp("retarget_syn")
    exe = bl.find_blender()
    bm, cam = synthetic_shot(n_frames=N, seed=3)
    inp = tk.TakeInputs(motion=bm, camera=cam, shot_id="007", project_dir=str(out / "proj"))
    take = tk.write_take(out / "take", inp)
    rig = tk.RigSettings(preset="unity_generic", prefix="mixamorig:")
    prefixed = tk.write_take(out / "take_prefixed", inp, rig)
    job = {"take": bl.host_path(take, exe), "take_prefixed": bl.host_path(prefixed, exe)}
    r = run_checks("synthetic", job, out / "work")
    print("retarget synthetic result:", dump("synthetic", r))
    return r


def _free_ok(m, name):
    """Acceptance 1: direction <= 1 deg, twist <= 2 deg; bones without a mapped child rotate like the
    source <= 2 deg; the hips and the feet (facing alignment only) <= 1 deg; the pelvis follows the
    scaled source displacement."""
    assert m["dir_err_deg"] <= 1.0, (name, m["worst"])
    assert m["twist_err_deg"] <= 2.0, (name, m["worst"])
    assert m["rot_err_deg"] <= 2.0, (name, m["worst"])
    assert m["feet_rot_deg"] <= 1.0, (name, m["worst"])
    assert m["hips_rot_deg"] <= 1.0, name
    assert m["anchor_err_cm"] <= 0.1, (name, m["anchor_err_cm"])


@pytest.mark.parametrize("name", RIGS)
def test_guess_map_matches_the_rig(res, name):
    """Acceptance 4: every required bone mapped, the expected body and finger bones, never the wrong
    side, never one target bone twice; twist, knee-piece, face and helper bones are left unmapped."""
    r = res["rigs"][name]
    assert r["missing"] == []
    assert r["sides_bad"] == [] and r["duplicates"] == 0
    wrong = {c: (r["guess"].get(c), b) for c, b in r["expected"].items() if r["guess"].get(c) != b}
    extra = {c: b for c, b in r["guess"].items() if c not in r["expected"]}
    assert not wrong and not extra, (wrong, extra)


def test_guess_map_biped_and_rigify(res):
    """A 3ds Max Biped (clavicles under the neck), Blender's own Rigify metarig (head with a face
    subtree), and a generated Rigify rig (refused, by guess_map and by retarget)."""
    g = res["guess_extra"]
    b = g["biped"]
    assert b["missing"] == [] and b["sides_bad"] == [] and b["duplicates"] == 0
    assert {c: b["guess"].get(c) for c in b["expected"]} == b["expected"]
    m = g["metarig"]
    assert m["missing"] == [] and m["sides_bad"] == [] and m["duplicates"] == 0
    assert {c: m["guess"][c] for c in ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head")} == {
        "Hips": "spine", "Spine": "spine.001", "Spine1": "spine.002", "Spine2": "spine.003",
        "Neck": "spine.004", "Head": "spine.006"}
    assert m["guess"]["LeftUpLeg"] == "thigh.L" and m["guess"]["RightForeArm"] == "forearm.R"
    _free_ok(m["free"]["measure"], "metarig")
    assert m["free"]["changed"] == []
    gen = g["generated"]
    assert gen["bones"] > 300
    assert gen["guess"] and "generated Rigify rig" in gen["guess"] and "metarig" in gen["guess"]
    assert gen["retarget"] and "generated Rigify rig" in gen["retarget"]


@pytest.mark.parametrize("name", RIGS)
def test_free_motion_follows_the_source(res, name):
    m = res["rigs"][name]["free"]["measure"]
    print(name, json.dumps({k: v for k, v in m.items() if k != "worst"}))
    _free_ok(m, name)


@pytest.mark.parametrize("name", RIGS)
def test_facing_and_scale_are_what_blender_shows(res, name):
    """The report's facing yaw and hips scale equal the ones recomputed here (target forward from its
    armature axes, hips scale = pelvis heights above the floor)."""
    for lock in ("free", "locked"):
        rep, m = res["rigs"][name][lock]["report"], res["rigs"][name][lock]["measure"]
        assert abs(rep["facing_yaw_deg"] - m["yaw_deg"]) < 0.01, (rep["facing_yaw_deg"], m["yaw_deg"])
        assert abs(rep["hips_scale"] - m["k_auto"]) <= 1e-4 * m["k_auto"]
        assert rep["facing"] == "-Y"


@pytest.mark.parametrize("name", list(TURNED))
def test_turned_and_parented_targets(res, name):
    """Turned 10 / 90 deg about Z, and under an empty turned 45 deg (rig turned 20 deg in it): the
    facing comes from the armature's own axes, so everything still lines up."""
    r = res["turned"][name]
    m, rep = r["measure"], r["report"]
    assert abs(m["yaw_deg"] - TURNED[name]) < 1e-3 and abs(rep["facing_yaw_deg"] - TURNED[name]) < 1e-3
    _free_ok(m, name)
    assert r["changed"] == []


@pytest.mark.parametrize("name", RIGS)
def test_the_report_agrees_with_blender(res, name):
    for lock in ("free", "locked"):
        run = res["rigs"][name][lock]
        rep, m = run["report"], run["measure"]
        assert rep["frames"] == res["frames"] and rep["mapped"] == m["mapped"]
        assert abs(rep["max_dir_err_deg"] - m["dir_err_deg"]) <= 0.5
        assert abs(rep["penetration_cm"] - max(m["pen_cm"], 0.0)) <= 0.2, (rep["penetration_cm"], m["pen_cm"])
        for side in ("left", "right"):
            a, b = rep["foot_slide_cm_s"][side], m["slide_cm_s"][side]
            assert (a is None) == (b is None)
            if a is not None:
                assert abs(a - b) <= 0.3, (side, a, b)


@pytest.mark.parametrize("name", RIGS)
def test_locked_feet_stay_planted(res, name):
    """Acceptance 2 (lock_feet=True): median planted-foot speed <= 3 cm/s; penetration of the foot mesh,
    toe tip and heel <= 1.5 cm on planted frames and never > 1.5 cm deeper than the source's own feet
    elsewhere. Lock edges: the synthetic take's plant flags do not stop its foot (its free ankle barely
    accelerates at the edge), so the 1.5 x free ratio is checked on the real takes
    (test_retarget_engine_characters.py); here the lock's own correction (locked - free ankle, both
    evaluated) stays within its design limit, retarget._EDGE_ACCEL = 1200 fighter cm/s^2."""
    free, locked = res["rigs"][name]["free"], res["rigs"][name]["locked"]
    m = locked["measure"]
    pen = penetration(m)
    print(name, "slide", m["slide_cm_s"], "pen", pen, locked["report"]["notes"])
    assert m["mesh_feet"]
    assert m["slide_cm_s"]["left_frames"] > 100  # the synthetic take plants the left foot for half of it
    assert m["slide_cm_s"]["left"] <= 3.0
    assert free["measure"]["slide_cm_s"]["left"] > 3.0  # without the lock the synthetic foot slides
    for side, p in pen.items():
        assert p["planted_max"] <= 1.5 and p["other_excess_max"] <= 1.5, (side, p)
    corr = edge_correction_accel(m, free["measure"])
    assert corr["Left"] is not None and corr["Left"] <= 1200.0, corr  # 1.0 x retarget._EDGE_ACCEL


def test_unreachable_targets_are_reported(res):
    """With the hips far too high (1.6 x the hips scale) the hips come down as far as allowed (a
    quarter of the leg) and what still cannot be reached is listed with its frames."""
    notes = res["unreachable"]["notes"]
    assert any("hips lowered" in n for n in notes), notes
    assert any("out of reach" in n and "frames" in n for n in notes), notes


@pytest.mark.parametrize("name", RIGS)
def test_nothing_else_in_the_target_changes(res, name):
    """Acceptance 5: rest pose, object transform, meshes, constraints, other actions, NLA, animation
    settings, rotation modes, selection, active object, modes, the scene frame: all unchanged. The new
    action has a fake user, one key per frame, and is the target's active action."""
    for lock in ("free", "locked"):
        run = res["rigs"][name][lock]
        assert run["changed"] == [], run["changed"]
        assert run["assigned"] and run["fake_user"]
        assert N in run["key_counts"]
        assert set(run["key_counts"]) <= {1, N}  # rest-constant keys have one key


def test_previous_action_is_kept_and_unmapped_bones_rest(res):
    old = res["old_action"]
    assert old["exists"] and old["fake_user"]
    assert res["rigs"]["syn_anatomical"]["free"]["fake_users_set"] == ["syn_anatomical_old"]
    assert old["unmapped_at_rest_deg"] < 1e-3


def test_copy_rotation_knee(res):
    """The anatomical rig's shins copy their knee pieces' local rotation (like Girush): simulated, so
    the free legs still follow the source (acceptance 1 above) and locked feet stay put."""
    notes = res["rigs"]["syn_anatomical"]["free"]["report"]["notes"]
    assert not any("not simulated" in n for n in notes), notes
    _free_ok(res["rigs"]["syn_anatomical"]["free"]["measure"], "copy rotation")


def test_tweak_key_changes_the_result(res):
    """Acceptance 3: a key on fighter A's "AFV tweaks" track (LeftForeArm +40 deg) moves the
    retargeted forearm by about as much, and the result still follows the (tweaked) source."""
    t = res["tweak"]
    assert min(t["changed_deg"]) > 15.0, t["changed_deg"]
    assert t["measure"]["dir_err_deg"] <= 1.0 and t["measure"]["twist_err_deg"] <= 2.0


def test_speed(res):
    """One fighter, 300 frames, onto a ~65-bone rig: well under 10 s."""
    for name in RIGS:
        rep = res["rigs"][name]["locked"]["report"]
        print(name, res["rigs"][name]["bones"], "bones:", rep["seconds"], "s")
        assert rep["seconds"] <= 10.0


def test_frame_subset_and_scene_frame(res):
    s = res["subset"]
    assert s["frames"] == [20, 80] and s["changed"] == [] and s["frame_after"] == [7, 0.0]
    assert 61 in s["key_counts"]
    assert s["measure"]["pen_cm"] <= 1.5


def test_prefixed_source_with_root_bone(res):
    p = res["prefixed"]
    assert p["canonical"]["Hips"] == "mixamorig:Hips" and "root" not in p["canonical"].values()
    assert len(p["canonical"]) == 65
    m = p["run"]["measure"]
    assert p["run"]["changed"] == []
    assert m["slide_cm_s"]["left"] <= 3.0


def test_map_argument_and_errors(res):
    assert res["full_map"]["facing"] == "+X"
    k = res["rigs"]["syn_rigify"]["free"]["report"]["hips_scale"]
    assert abs(res["full_map"]["hips_scale"] - 0.9 * k) < 1e-5
    e = res["errors"]
    assert all(e.values()), e
    assert "Hips" in e["hips_missing"] and "mapped to both" in e["twice"]
    assert "hierarchy" in e["hierarchy"] and "shin.L" in e["hierarchy"]
    assert any("no plant masks" in n for n in res["no_plants"]["notes"])


def test_refusals(res):
    """Refuse, never guess silently: non-uniform scale (target and source), a target lying down, a
    source whose collection is excluded from the view layer."""
    r = res["refusals"]
    assert r["nonuniform_target"] and "non-uniform scale" in r["nonuniform_target"], r
    assert r["nonuniform_source"] and "source" in r["nonuniform_source"], r
    assert r["lying"] and "upright" in r["lying"], r
    assert r["excluded_source"] and "not evaluated" in r["excluded_source"], r


def test_swapped_map_is_noted(res):
    assert any("look like the character's right" in n for n in res["swapped_notes"]), res["swapped_notes"]


def test_error_while_writing_leaves_nothing(res):
    w = res["write_error"]
    assert w["raised"] and w["new_actions"] == [] and w["action_kept"], w
    t = res["write_error_tweak"]  # failing after NLA tweak mode was left: tweak mode is back
    assert t["raised"] and t["new_actions"] == [] and t["now"] == t["was"] and t["was"]["tweak"], t


def test_short_gap_fade_is_noted(res):
    """Two left-foot plants three frames apart: the fade across the gap cannot stay within the design
    limit; the note names those frames and its "up to" value is the lock's own correction there (the
    evaluated locked - free ankle)."""
    import re

    import numpy as np
    from retarget_helpers import FPS, _accel

    c = res["close_locks"]
    note = next((n for n in c["notes"]
                 if "fades in/out harder" in n and n.startswith("lock_feet: left")), None)
    assert note, c["notes"]
    frames = []
    for part in re.search(r"frames \(([^)]*)\)", note).group(1).split(", "):
        lo, _, hi = part.partition("-")
        frames += list(range(int(lo), int(hi or lo) + 1))
    up_to = float(re.search(r"up to (\d+) cm/s\^2", note).group(1))
    assert frames and all(127 <= f <= 135 for f in frames), (frames, note)  # around the 130-132 gap
    corr = np.array(c["locked"]["sides"]["Left"]["ankle"]) - np.array(c["free"]["sides"]["Left"]["ankle"])
    measured = float(_accel(corr)[frames].max())
    assert up_to > 1200.0 and abs(up_to - measured) <= 0.05 * measured + 5.0, (up_to, measured, FPS)


def test_failing_modifier_switch_restores_all(res):
    m = res["mute_error"]
    assert m["raised"] and m["calls"] == 2 and m["modifiers"] >= 2, m
    assert m["after"] == m["before"] and all(m["before"]) and m["new_actions"] == [], m


def test_y_up_target_facing_round_trip(res):
    """Fighter B's take armature (Y-up armature data) faces +Z in its own axes; that facing goes
    through a map file and back into retarget with the same keys."""
    y = res["yup_target"]
    assert y["facing"] == "+Z" and y["read_facing"] == "+Z" and y["second_facing"] == "+Z", y
    assert y["same_keys"] and abs(y["yaw"][0] - y["yaw"][1]) < 1e-9, y
    assert max(y["dir"]) <= 1.0, y


def test_map_files(res):
    m = res["mapio"]
    assert m["kept_unknown"] and m["roundtrip"] and m["no_tmp_left"] and m["write_rejects"]
    bad = {k: v for k, v in m["rejected"].items() if not v}
    assert not bad, bad
    assert "bad_facing.json" in m["rejected"]["facing"] and "facing" in m["rejected"]["facing"]
    assert m["z_roundtrip"]  # +Z / -Z are valid facings (Y-up armature data)
    assert m["from_empty"] is None and m["object_roundtrip"] and m["object_broken"]
    assert "Spine" in m["missing"] and "Hips" not in m["missing"]
    assert m["missing_full"] == [c for c in m["missing"] if c != "Spine"]
