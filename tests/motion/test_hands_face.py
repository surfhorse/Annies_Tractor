"""hands_face stage (round 2, minimal): GEM-X pass-through, fist preset, neutral face, metacarpal QC."""

from __future__ import annotations

import json

import numpy as np
import pytest

from afv.core.skeletons import get_skeleton
from afv.core.stage import Ctx, Stage, StageError, get_stage
from afv.core.types import ActorMotion, BodyMotion
from afv.motion.fk import forward_kinematics
from afv.motion.hands_face import (
    HandsFaceParams,
    apply_hands_face,
    finger_joints,
    fist_pose,
    metacarpal_stats,
)
from afv.motion.rotations import from_rotvec, qconj, qrotate

SK = get_skeleton("soma77")


def _motion(n=5, skeleton=SK, seed=0):
    rng = np.random.default_rng(seed)
    rot = from_rotvec(rng.normal(0, 0.1, (n, skeleton.num_joints, 3))).astype(np.float32)
    root = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
    return BodyMotion(
        frames=np.arange(n), timestamps=np.arange(n) / 30.0, skeleton=skeleton.name, fps=30.0,
        contact_channels=[],
        actors={"A": ActorMotion(rot=rot, root=root), "B": ActorMotion(rot=rot, root=root)},
    )  # fmt: skip


def test_registered_and_pass_through_is_exact():
    cls = get_stage("hands_face", "default")
    assert isinstance(cls(), Stage) and cls.inputs == ("smooth",) and cls.env == "cpu"
    bm = _motion()
    out, qc = apply_hands_face(bm, HandsFaceParams())
    for r in bm.actors:
        assert np.array_equal(out.actors[r].rot, bm.actors[r].rot)
        assert np.array_equal(out.actors[r].root, bm.actors[r].root)
    assert qc["finger_joints"] == 38  # 2 hands x (thumb 3 + 4 fingers x 4) rotating finger joints


def test_fist_closes_fingers_toward_the_palm():
    names = finger_joints(SK)
    pose = fist_pose(SK)
    assert set(pose) == set(names)
    bm = _motion(n=2)
    out, _ = apply_hands_face(bm, HandsFaceParams(hands="fist"))
    rot = np.asarray(out.actors["A"].rot, np.float64)
    pos, wrot = forward_kinematics(SK, rot, np.asarray(out.actors["A"].root, np.float64))
    rest = forward_kinematics(SK, np.tile([0.0, 0, 0, 1], (1, SK.num_joints, 1)))[0][0]
    for side in ("Left", "Right"):
        hand = SK.index(f"{side}Hand")
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            tip = SK.index(f"{side}Hand{finger}End")
            # tip in the hand's frame: under the palm (-Y) and pulled back toward the wrist
            rel = qrotate(qconj(wrot[:, hand]), pos[:, tip] - pos[:, hand])
            extended = np.linalg.norm(rest[tip] - rest[hand])
            assert (rel[:, 1] < -0.005).all(), (side, finger)
            assert (np.linalg.norm(rel, axis=-1) < 0.7 * extended).all(), (side, finger)
    # the body is untouched
    body = [SK.index(n) for n in ("Hips", "LeftHand", "RightForeArm", "Head")]
    assert np.array_equal(rot[:, body], np.asarray(bm.actors["A"].rot)[:, body])


def test_fist_weight_blends_and_face_neutral():
    bm = _motion(n=3)
    half, _ = apply_hands_face(bm, HandsFaceParams(hands="fist", fist_weight=0.5, face="neutral"))
    full, _ = apply_hands_face(bm, HandsFaceParams(hands="fist"))
    j = SK.index("LeftHandIndex3")
    ang = lambda q: 2 * np.arccos(np.clip(np.abs(q[..., 3]), 0, 1))  # noqa: E731
    a_half, a_full = ang(half.actors["A"].rot[:, j]), ang(full.actors["A"].rot[:, j])
    assert (a_half < a_full).all()
    for n in ("Jaw", "LeftEye", "RightEye"):
        assert np.allclose(half.actors["A"].rot[:, SK.index(n)], [0, 0, 0, 1])


def test_metacarpal_stats_measure_rotation_and_knuckle_shift():
    bm = _motion(n=4)
    a = bm.actors["A"]
    rot = np.array(a.rot)
    rot[:] = [0, 0, 0, 1]
    j = SK.index("LeftHandIndex1")
    rot[:, j] = from_rotvec(np.array([0.0, 0.0, np.deg2rad(20.0)]))
    st = metacarpal_stats(SK, ActorMotion(rot=rot, root=a.root))
    assert st["joints"]["LeftHandIndex1"]["max_deg"] == pytest.approx(20.0, abs=0.01)
    assert st["joints"]["RightHandIndex1"]["max_deg"] == pytest.approx(0.0, abs=0.01)
    v = SK.rest_offsets[SK.index("LeftHandIndex2")]
    want = np.linalg.norm(qrotate(rot[0, j].astype(np.float64), v) - v) * 1000
    assert st["joints"]["LeftHandIndex1"]["knuckle_shift_max_mm"] == pytest.approx(want, abs=0.01)
    assert "LeftHandThumb1" not in st["joints"]


def test_skeleton_without_fingers_passes_through_with_note(tmp_path):
    sk = get_skeleton("soma_provisional")
    if finger_joints(sk):
        pytest.skip("provisional skeleton has soma77-style finger names")
    bm = _motion(n=2, skeleton=sk)
    out, qc = apply_hands_face(bm, HandsFaceParams(hands="fist"))
    assert np.array_equal(out.actors["A"].rot, bm.actors["A"].rot) and qc["notes"]


def test_stage_writes_qc_and_needs_motion(tmp_path):
    stage = get_stage("hands_face", "default")()
    ctx = Ctx(shot_dir=tmp_path / "s", work_dir=tmp_path / "w")
    out = stage.run(ctx, {"smooth": _motion()}, HandsFaceParams())
    assert isinstance(out, BodyMotion) and out.num_frames == 5
    qc = json.loads((ctx.work_dir / "qc.json").read_text())
    assert set(qc["summary"]["metacarpal_p95_deg"]) == {"A", "B"}
    with pytest.raises(StageError):
        stage.run(ctx, {}, HandsFaceParams())


def _fist_world(sk):
    """Rest-pose body with the fist on both hands: world joint positions [J,3]."""
    rot = np.tile([0.0, 0.0, 0.0, 1.0], (1, sk.num_joints, 1))
    for n, q in fist_pose(sk).items():
        rot[0, sk.index(n)] = q
    return forward_kinematics(sk, rot)[0][0]


def test_thumb_wraps_across_index_and_middle():
    pos = _fist_world(SK)
    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        names = ("", "Index3", "Middle3", "Ring3", "ThumbEnd", "Thumb1")
        p = {n: pos[SK.index(f"{side}Hand{n}")] for n in names}
        tip, pip = p["ThumbEnd"], 0.5 * (p["Index3"] + p["Middle3"])
        assert np.linalg.norm(tip - pip) < 0.025, side  # in front of the index/middle PIP joints
        assert tip[1] < p[""][1] - 0.02, side  # under the palm (palms face -Y at rest)
        # across the fingers: from the thumb side past the index, toward the middle finger
        assert sign * (tip[0] - p[""][0]) > 0.07, side
        assert p["Ring3"][2] < tip[2] < p["Thumb1"][2], side
        # not inside the curled fingers: every finger joint stays >= 8 mm from the thumb tip
        fingers = [SK.index(f"{side}Hand{f}{k}") for f in ("Index", "Middle", "Ring", "Pinky") for k in "234"]
        assert np.linalg.norm(pos[fingers] - tip, axis=-1).min() > 0.008, side


def test_fist_is_mirror_symmetric():
    rest = forward_kinematics(SK, np.tile([0.0, 0.0, 0.0, 1.0], (1, SK.num_joints, 1)))[0][0]
    m = np.array([-1.0, 1.0, 1.0])
    left = [n for n in SK.joint_names if n.startswith("LeftHand")]
    sym = max(np.abs(rest[SK.index(n)] * m - rest[SK.index("Right" + n[4:])]).max() for n in left)
    pos = _fist_world(SK)
    err = max(np.abs(pos[SK.index(n)] * m - pos[SK.index("Right" + n[4:])]).max() for n in left)
    assert err <= sym + 1e-4, (err, sym)  # as symmetric as the skeleton itself


def test_fist_is_correct_with_rest_rotations():
    """Same world rest pose, but LeftHand carries a 90 deg rest rotation (children compensate): the fist
    must put every joint at the same world position."""
    from afv.core.types import Skeleton

    j = SK.index("LeftHand")
    r = from_rotvec(np.array([0.0, 0.0, np.pi / 2]))  # 90 deg about z
    rr = np.tile([0.0, 0.0, 0.0, 1.0], (SK.num_joints, 1))
    off = SK.rest_offsets.copy()
    rr[j] = r
    for c in SK.children(j):
        off[c] = qrotate(qconj(r), off[c])  # offsets now in the rotated hand frame
        rr[c] = qconj(r)  # children's frames back to the original world orientation
    sk2 = Skeleton("soma77_rot_hand", SK.joint_names, SK.parents, off, rr)
    base = forward_kinematics(SK, np.tile([0.0, 0.0, 0.0, 1.0], (1, SK.num_joints, 1)))[0][0]
    assert np.allclose(forward_kinematics(sk2, np.tile([0.0, 0, 0, 1], (1, SK.num_joints, 1)))[0][0], base)
    assert np.abs(_fist_world(sk2) - _fist_world(SK)).max() < 1e-4


def test_hands_off_span_blends_to_relaxed_pose_with_eased_edges():
    from afv.motion.hands_face import relaxed_pose, span_weights

    n = 40
    bm = _motion(n)
    relaxed = relaxed_pose(SK)
    assert relaxed and all(n.startswith(("LeftHand", "RightHand")) for n in relaxed)
    edit = {"spans": [{"role": "A", "start": 15, "end": 25, "side": "left", "extra": 1}]}
    out, qc = apply_hands_face(bm, HandsFaceParams(hands_off_ease_frames=5), edit)
    a, a0 = out.actors["A"].rot, bm.actors["A"].rot
    li = [SK.index(n) for n in relaxed if n.startswith("LeftHand")]
    ri = [SK.index(n) for n in relaxed if n.startswith("RightHand")]
    for j, name in zip(li, [n for n in relaxed if n.startswith("LeftHand")], strict=True):
        assert np.allclose(np.abs(np.sum(a[15:25, j] * relaxed[name], -1)), 1.0, atol=1e-6)
    assert np.array_equal(a[:10], a0[:10]) and np.array_equal(a[30:], a0[30:])  # beyond the ease
    assert np.array_equal(a[:, ri], a0[:, ri])  # right hand untouched
    assert np.array_equal(out.actors["B"].rot, bm.actors["B"].rot)  # other fighter untouched
    assert qc["fighters"]["A"]["hands_off_frames"] == 10
    w = span_weights(n, [(15, 25)], 5)
    assert np.all(np.diff(w[9:16]) > 0) and np.all(np.diff(w[24:31]) < 0)
    assert w[9] == 0 < w[10] < 0.05 < 0.95 < w[14] < 1
    # the ease is smooth: the weight's slope changes gradually (smootherstep)
    assert np.abs(np.diff(w, 2)).max() < 0.35


@pytest.mark.parametrize(
    "span, message",
    [
        ({"role": "C", "start": 1, "end": 3}, "unknown fighter"),
        ({"role": "A", "start": 3, "end": 3}, "empty span"),
        ({"role": "A", "start": 100, "end": 120}, "outside the shot"),
        ({"role": "A", "start": 1, "end": 3, "side": "middle"}, "side must be"),
        ({"role": "A", "end": 3}, "'start' and 'end'"),
    ],
)
def test_hands_off_validation(span, message, tmp_path):
    with pytest.raises(ValueError, match=message):
        apply_hands_face(_motion(10), HandsFaceParams(), {"spans": [span]})
    cls = get_stage("hands_face", "default")
    ctx = Ctx(shot_dir=tmp_path, work_dir=tmp_path / "w", edits={"hands": {"spans": [span]}})
    with pytest.raises(StageError, match=message):
        cls().run(ctx, {"smooth": _motion(10)}, HandsFaceParams())


def test_hands_off_both_sides_on_a_fist():
    bm = _motion(20)
    edit = {"spans": [{"role": "B", "start": 0, "end": 20}]}
    out, _ = apply_hands_face(bm, HandsFaceParams(hands="fist"), edit)
    fist = fist_pose(SK)
    from afv.motion.hands_face import relaxed_pose

    relaxed = relaxed_pose(SK)
    j = SK.index("RightHandMiddle3")
    b, a = out.actors["B"].rot[:, j], out.actors["A"].rot[:, j]
    assert np.allclose(np.abs(np.sum(b * relaxed["RightHandMiddle3"], -1)), 1, atol=1e-6)  # relaxed
    assert np.allclose(np.abs(np.sum(a * fist["RightHandMiddle3"], -1)), 1, atol=1e-6)  # still a fist


def test_pinned_fingers_survive_the_fist_and_qc_reports_them(tmp_path):
    """hands_face keeps the smooth result on a pinned finger at the pin frame (the fist blends back
    in over the radius) and reports the final pin error; unpinned fingers stay fisted."""
    from afv.motion.pins import canonical_bones, parse_pins, quat_angle_deg
    from afv.motion.rotations import qmul

    bm = _motion(30)
    canon = canonical_bones(SK)
    q = from_rotvec([0.0, 0.0, -0.6]).tolist()
    pin = {"role": "A", "frame": 15, "radius": 6, "preset": "blender", "bones": {"RightHandIndex1": q}}
    edit = {"pins": [pin]}
    pins = parse_pins(edit, bm.frames, ["A", "B"], canon)
    out, _ = apply_hands_face(bm, HandsFaceParams(hands="fist"), None, pins)
    fist = fist_pose(SK)
    seg = [SK.index("RightHandIndex2"), SK.index("RightHandIndex1")]  # rig Index1: Index2 + folded metacarpal
    a, a0 = out.actors["A"].rot, bm.actors["A"].rot
    assert np.array_equal(a[15, seg], a0[15, seg])  # the smooth pose at the pin frame
    for j in seg:  # fist again from the radius on
        name = SK.joint_names[j]
        assert np.allclose(np.abs(np.sum(a[[0, 9, 21, 29], j] * fist[name], -1)), 1.0, atol=1e-6)
    j3 = SK.index("RightHandIndex3")
    assert np.allclose(np.abs(np.sum(a[15, j3] * fist["RightHandIndex3"], -1)), 1.0, atol=1e-6)
    b15 = out.actors["B"].rot[15, seg[0]]
    assert np.allclose(np.abs(np.sum(b15 * fist["RightHandIndex2"], -1)), 1.0, atol=1e-6)
    # through the stage: QC carries the final error of every pin
    cls = get_stage("hands_face", "default")
    ctx = Ctx(shot_dir=tmp_path, work_dir=tmp_path / "w", edits={"pins": edit})
    cls().run(ctx, {"smooth": bm}, HandsFaceParams(hands="fist"))
    qc = json.loads((tmp_path / "w" / "qc.json").read_text())
    assert len(qc["pins"]) == 1 and qc["pins"][0]["frame"] == 15
    assert isinstance(qc["summary"]["pins_max_error_deg"], float)
    del qmul, quat_angle_deg
