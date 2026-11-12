"""Pinned poses (edits/pins.json) in the smooth stage: conversion from export-rig local rotations,
exactness at the pin frame through the real exporter, C1 blending, validation."""

from __future__ import annotations

import json

import numpy as np
import pytest
from real_metrics import load_clip

import afv.motion.stages  # noqa: F401  (registers the stages)
from afv.core.skeletons import get_skeleton
from afv.core.stage import Ctx, StageError, get_stage
from afv.core.types import BodyMotion, Take
from afv.export.exporter import take_motion
from afv.export.presets import get_preset
from afv.motion.pins import (
    PinError,
    canonical_bones,
    clusters,
    parse_pins,
    quat_angle_deg,
    spline_displacement,
)
from afv.motion.ranges import slice_motion
from afv.motion.rotations import from_rotvec, qmul
from afv.motion.smooth import SmoothParams, smooth_motion

SK = get_skeleton("soma77")
CANON = canonical_bones(SK)


@pytest.fixture(scope="module")
def clip() -> BodyMotion:
    bm, _ = load_clip("ufc_cross")  # real GEM-X motion (soma77), with GEM-X contact probabilities
    return slice_motion(bm, 0, 100)


def _export_rot(bm: BodyMotion, preset: str, role: str = "A"):
    """The export rig motion exactly as ``afv export`` / ``afv take`` build it."""
    pr = get_preset(preset)
    a = bm.actors[role]
    take = Take(
        skeleton=bm.skeleton, fps=bm.fps, speed_factor=1.0, rot=a.rot, root=a.root,
        root_motion=pr.root_motion, contact_channels=list(bm.contact_channels), contact=a.contact,
        role=role, rest_offsets=a.rest_offsets,
    )  # fmt: skip
    return take_motion(take, pr, skeleton=SK)


def _free_left_foot(bm: BodyMotion, role: str, i: int, half: int = 15) -> BodyMotion:
    """``bm`` with the left foot of ``role`` unplanted around frame ``i`` (as if painted free), so
    the leg re-pin leaves LeftUpLeg alone there (the fixture's boxers always have both feet down)."""
    out = slice_motion(bm, 0, bm.num_frames)
    c = out.actors[role].contact
    for ch in ("LeftFoot", "LeftToeBase"):
        c[i - half : i + half + 1, bm.contact_channels.index(ch)] = 0.0
    return out


BONES = ("Hips", "Spine1", "Neck", "Head", "LeftForeArm", "RightHand", "RightHandIndex1")


@pytest.mark.parametrize("preset", ["unity_humanoid", "unity_generic"])
def test_pinned_frame_reproduces_export_rotations(clip, preset):
    """Pins built from the export rig's own local rotations (+ random offsets of ~15 deg) hold at the
    pin frame within 0.1 deg in the exported rig, incl. Neck (re-aimed over the folded Neck2 joint).
    Hips under extracted root motion (unity_generic) holds as the world rotation the user saw: the
    pinned local rotation through the take's root bone at that frame."""
    role, other, i = "A", "B", 50
    clip = _free_left_foot(clip, role, i)
    base, _ = smooth_motion(clip)
    m0 = _export_rot(base, preset, role)
    names = m0.rig.canonical
    rng = np.random.default_rng(3)
    bones = {
        b: qmul(m0.rot[i, names.index(b)], from_rotvec(rng.normal(0, 0.15, 3))).tolist()
        for b in (*BONES, "LeftUpLeg")
    }
    edit = {
        "pins": [
            {"role": role, "frame": int(clip.frames[i]), "radius": 10, "preset": preset, "bones": bones},
            # a second pin inside the first one's radius (solved together, both exact)
            {"role": role, "frame": int(clip.frames[i + 5]), "radius": 6, "preset": preset,
             "bones": {"LeftForeArm": bones["LeftForeArm"], "Spine1": bones["Spine1"]}},
        ]
    }  # fmt: skip
    pins = parse_pins(edit, clip.frames, list(clip.actors), CANON)
    out, infos = smooth_motion(clip, pins=pins)
    m1 = _export_rot(out, preset, role)
    for p in edit["pins"]:
        f = p["frame"] - int(clip.frames[0])
        for b, q in p["bones"].items():
            if b == "Hips" and m0.rig.has_root:
                assert quat_angle_deg(_hips_world(out, role, f, m0.rig), qmul(m0.rot[f, 0], q)) < 0.1
            else:
                assert quat_angle_deg(m1.rot[f, names.index(b)], q) < 0.1, (preset, b, f)
    reports = [r.to_dict() for r in infos[role].pins]
    assert len(reports) == 2 and all(r["max_error_deg"] < 0.1 and not r["conflicts"] for r in reports)
    # the other fighter and frames outside every pin's reach are untouched
    assert np.array_equal(out.actors[other].rot, base.actors[other].rot)
    dif = np.abs(out.actors[role].rot - base.actors[role].rot).max(axis=(1, 2))
    changed = np.flatnonzero(dif > 1e-5)  # (the leg re-pin leaves float32 noise elsewhere)
    assert changed.min() >= i - 10 - 8 and changed.max() <= i + 5 + 6 + 8  # + leg re-pin blend frames


def _hips_world(bm: BodyMotion, role: str, f: int, rig) -> np.ndarray:
    from afv.export.retarget import retarget

    a = bm.actors[role]
    return retarget(rig, np.asarray(a.rot[f : f + 1], float), np.asarray(a.root[f : f + 1], float), 30.0).rot[
        0, rig.hips
    ]


@pytest.mark.parametrize("yaw", [-1.5, -0.45, 0.05, 0.45, 1.5])
def test_extracted_hips_pin_keeps_the_world_rotation(clip, yaw):
    """Large Hips yaws and pitches under extracted root motion, and two overlapping Hips pins: the
    Hips world rotation = take root bone (before the pins) x pinned local rotation, exactly."""
    base, _ = smooth_motion(clip)
    m0 = _export_rot(base, "unity_generic")
    h = m0.rig.hips
    pins = [
        {"role": "A", "frame": int(clip.frames[i]), "radius": 10, "preset": "unity_generic",
         "bones": {"Hips": qmul(m0.rot[i, h], from_rotvec(v)).tolist()}}
        for i, v in ((40, [0.0, yaw, 0.0]), (46, [0.3 * yaw, 0.0, 0.2]))
    ]  # fmt: skip
    out, infos = smooth_motion(clip, pins=parse_pins({"pins": pins}, clip.frames, ["A", "B"], CANON))
    for p, rep_ in zip(pins, infos["A"].pins, strict=True):
        f = p["frame"] - int(clip.frames[0])
        want = qmul(m0.rot[f, 0], p["bones"]["Hips"])
        assert quat_angle_deg(_hips_world(out, "A", f, m0.rig), want) < 0.1
        assert rep_.max_error_deg < 0.1
    # an explicit root_rotation (copied from the take) is used as given
    rr = from_rotvec([0.0, 0.7, 0.0]).tolist()
    p1 = {**pins[0], "root_rotation": rr}
    out, infos = smooth_motion(clip, pins=parse_pins({"pins": [p1]}, clip.frames, ["A", "B"], CANON))
    assert quat_angle_deg(_hips_world(out, "A", 40, m0.rig), qmul(rr, p1["bones"]["Hips"])) < 0.1


def test_overlapping_pins_do_not_leak_into_other_joints(clip):
    """A short LeftForeArm pin next to a long RightForeArm pin moves LeftForeArm only within its own
    radius; pins on one joint a frame apart are exact and warned about."""
    base, _ = smooth_motion(clip, SmoothParams(hold_contacts=False))
    m0 = _export_rot(base, "blender")
    n = m0.rig.canonical
    lf, rf = n.index("LeftForeArm"), n.index("RightForeArm")
    pins = [
        {"role": "A", "frame": int(clip.frames[50]), "radius": 4, "preset": "blender",
         "bones": {"LeftForeArm": qmul(m0.rot[50, lf], from_rotvec([0, 0.5, 0])).tolist()}},
        {"role": "A", "frame": int(clip.frames[54]), "radius": 30, "preset": "blender",
         "bones": {"RightForeArm": qmul(m0.rot[54, rf], from_rotvec([0.4, 0, 0])).tolist()}},
    ]  # fmt: skip
    out, infos = smooth_motion(
        clip,
        SmoothParams(hold_contacts=False),
        pins=parse_pins({"pins": pins}, clip.frames, ["A", "B"], CANON),
    )
    m1 = _export_rot(out, "blender")
    d = np.array([quat_angle_deg(m1.rot[f, lf], m0.rot[f, lf]) for f in range(clip.num_frames)])
    moved = np.flatnonzero(d > 1e-3)
    assert moved.min() >= 50 - 3 and moved.max() <= 50 + 3
    assert all(r.max_error_deg < 0.1 for r in infos["A"].pins)
    close = [
        {"role": "A", "frame": int(clip.frames[f]), "radius": 8, "preset": "blender",
         "bones": {"LeftArm": qmul(m0.rot[f, n.index("LeftArm")], from_rotvec([0, 0, z])).tolist()}}
        for f, z in ((60, 0.3), (61, -0.3))
    ]  # fmt: skip
    out, infos = smooth_motion(
        clip,
        SmoothParams(hold_contacts=False),
        pins=parse_pins({"pins": close}, clip.frames, ["A", "B"], CANON),
    )
    reps = infos["A"].pins
    assert all(r.max_error_deg < 0.1 for r in reps)
    assert reps[0].warnings and "1 frame(s) apart" in reps[0].warnings[0]


def test_unpinned_bones_keep_their_rig_local_rotation(clip):
    """Pinning a parent turns its children with it: their export-rig local rotations stay the solved ones."""
    base, _ = smooth_motion(clip, SmoothParams(hold_contacts=False))
    m0 = _export_rot(base, "blender")
    names = m0.rig.canonical
    i = 50
    q = qmul(m0.rot[i, names.index("Spine1")], from_rotvec([0.0, 0.0, 0.4])).tolist()
    pins = parse_pins(
        {"pins": [{"role": "A", "frame": int(clip.frames[i]), "preset": "blender", "bones": {"Spine1": q}}]},
        clip.frames, ["A", "B"], CANON,
    )  # fmt: skip
    out, _ = smooth_motion(clip, SmoothParams(hold_contacts=False), pins=pins)
    m1 = _export_rot(out, "blender")
    assert quat_angle_deg(m1.rot[i, names.index("Spine1")], q) < 0.1
    for b in ("Spine2", "Neck", "Head", "LeftArm", "RightShoulder", "LeftUpLeg"):
        assert quat_angle_deg(m1.rot[i, names.index(b)], m0.rot[i, names.index(b)]) < 1e-3, b
    assert np.linalg.norm(m1.loc[i, 0] - m0.loc[i, 0]) < 1e-6  # Hips stays where it was


def test_root_pin_moves_hips_exactly(clip):
    base, _ = smooth_motion(clip, SmoothParams(hold_contacts=False))
    i = 40
    m0 = _export_rot(base, "unity_humanoid")
    hips = m0.rig.hips
    target = m0.loc[i, hips] + np.array([0.05, -0.02, 0.04])
    edit = {"pins": [{"role": "B", "frame": int(clip.frames[i]), "radius": 8, "root": target.tolist()}]}
    pins = parse_pins(edit, clip.frames, ["A", "B"], CANON)
    out, infos = smooth_motion(clip, SmoothParams(hold_contacts=False), pins=pins)
    m1 = _export_rot(out, "unity_humanoid", "B")
    m0b = _export_rot(base, "unity_humanoid", "B")
    assert np.linalg.norm(m1.loc[i, hips] - target) < 1e-5
    assert infos["B"].pins[0].root_error_m < 1e-5
    d = np.linalg.norm(m1.loc[:, hips] - m0b.loc[:, hips], axis=-1)
    assert np.all(d[: i - 7] < 1e-6) and np.all(d[i + 8 :] < 1e-6)  # radius 8: |t - frame| >= 8 untouched


def test_displacement_is_c1_and_zero_outside():
    n, f, r = 60, 30, 10
    ((lo, hi, ids),) = clusters([(f, r)], n)
    assert (lo, hi, ids) == (f - r + 1, f + r - 1, [0])
    d = spline_displacement(n, lo, hi, {f: np.array([1.0, -2.0, 0.5])})
    full = np.zeros((n, 3))
    full[lo : hi + 1] = d
    assert np.allclose(full[f], [1.0, -2.0, 0.5])
    assert np.all(full[: f - r + 1] == 0) and np.all(full[f + r :] == 0)
    v = np.diff(full, axis=0)
    # velocity is continuous (no kink at the edges or at the pin): a linear hat blend would change it
    # by amplitude / radius at its edges and twice that at its peak; the spline stays well below
    amp = np.abs(full).max()
    assert np.abs(np.diff(v, axis=0)).max() < 0.15 * amp  # a hat: 0.2
    assert np.abs(v[f - r]).max() < 0.09 * amp  # leaves the untouched motion slowly (a hat: 0.1)
    # symmetric bump, peak at the pin
    assert np.allclose(full[f - 5], full[f + 5]) and np.argmax(np.abs(full[:, 1])) == f


def test_pin_near_clip_end_and_overlap():
    n = 20
    cl = clusters([(2, 8), (9, 8)], n)
    assert len(cl) == 1 and cl[0][0] == 0 and cl[0][2] == [0, 1]
    d = spline_displacement(n, cl[0][0], cl[0][1], {2: np.array([1.0]), 9: np.array([-1.0])})
    assert d[2, 0] == pytest.approx(1.0) and d[9, 0] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "pin, message",
    [
        ({"role": "C", "frame": 0, "bones": {"Hips": [0, 0, 0, 1]}}, "unknown fighter"),
        ({"role": "A", "frame": 999, "bones": {"Hips": [0, 0, 0, 1]}}, "outside the shot"),
        ({"role": "A", "frame": 2, "bones": {"Pelvis": [0, 0, 0, 1]}}, "unknown export-rig bone"),
        ({"role": "A", "frame": 2, "bones": {"root": [0, 0, 0, 1]}}, "pin Hips instead"),
        ({"role": "A", "frame": 2, "bones": {"Hips": [0, 0, 0]}}, "4 finite numbers"),
        ({"role": "A", "frame": 2, "bones": {"Hips": [0, 0, 0, 0]}}, "not all 0"),
        ({"role": "A", "frame": 2, "radius": 0, "bones": {"Hips": [0, 0, 0, 1]}}, "radius"),
        ({"role": "A", "frame": 2, "preset": "maya", "bones": {"Hips": [0, 0, 0, 1]}}, "unknown export"),
        ({"role": "A", "frame": 2, "root": [0, 1]}, "'root' must be"),
        ({"role": "A", "frame": 2}, "pins nothing"),
        ({"role": "A", "frame": 2, "preset": "blender", "root_motion": "extracted", "root": [0, 1, 0]},
         "does not support root motion"),
    ],
)  # fmt: skip
def test_invalid_pins_are_reported(pin, message):
    frames = np.arange(10)
    with pytest.raises(PinError, match=message):
        parse_pins({"pins": [pin]}, frames, ["A", "B"], CANON)


def test_duplicate_pin_and_prefixed_bone_names():
    frames = np.arange(10)
    q = [0, 0, 0, 1]
    with pytest.raises(PinError, match="same fighter and frame"):
        parse_pins({"pins": [{"role": "A", "frame": 3, "root": [0, 1, 0]}] * 2}, frames, ["A"], CANON)
    pins = parse_pins({"pins": [{"role": "A", "frame": 3, "bones": {"LeftArm": q}}]}, frames, ["A"], CANON)
    assert pins["A"][0].radius == 8 and pins["A"][0].preset == "unity_humanoid"
    pp = parse_pins(
        {"pins": [{"role": "A", "frame": 3, "prefix": "mixamorig:", "root_motion": "in_place",
                   "bones": {"mixamorig:LeftArm": q, "Head": q}}]}, frames, ["A"], CANON,
    )["A"][0]  # fmt: skip
    assert set(pp.bones) == {"LeftArm", "Head"} and pp.root_motion == "in_place"
    with pytest.raises(PinError, match="unknown export-rig bone"):
        parse_pins(
            {"pins": [{"role": "A", "frame": 3, "bones": {"mixamorig:LeftArm": q}}]}, frames, ["A"], CANON
        )
    assert pins["A"][0].root_motion == "world" and set(pins["A"][0].bones) == {"LeftArm"}


def test_stage_reads_pins_and_reports_them(clip, tmp_path):
    cls = get_stage("smooth", "default")
    assert "pins" in cls.edits and cls.supports_range
    i = 50
    m0 = _export_rot(smooth_motion(clip)[0], "unity_humanoid")
    q = qmul(m0.rot[i, m0.rig.index("RightForeArm")], from_rotvec([0.3, 0.0, 0.0])).tolist()
    edit = {"pins": [{"role": "A", "frame": int(clip.frames[i]), "bones": {"RightForeArm": q}, "x": 1}]}
    ctx = Ctx(shot_dir=tmp_path, work_dir=tmp_path / "w", edits={"pins": edit})
    out = cls().run(ctx, {"contacts": clip}, cls.Params())
    qc = json.loads((tmp_path / "w" / "qc.json").read_text())
    assert qc["pins"][0]["frame"] == int(clip.frames[i]) and qc["pins"][0]["max_error_deg"] < 0.1
    m1 = _export_rot(out, "unity_humanoid")
    assert quat_angle_deg(m1.rot[i, m1.rig.index("RightForeArm")], q) < 0.1
    bad = {"pins": [{"role": "A", "frame": 10**6, "bones": {"RightForeArm": q}}]}
    ctx = Ctx(shot_dir=tmp_path, work_dir=tmp_path / "w2", edits={"pins": bad})
    with pytest.raises(StageError, match="outside the shot"):
        cls().run(ctx, {"contacts": clip}, cls.Params())
