"""Joint roles and analytic two-bone IK."""

from __future__ import annotations

import numpy as np
import pytest
from motion_synth import ROLES, SK

from afv.core.types import Skeleton
from afv.motion.fk import forward_kinematics
from afv.motion.ik import (
    BodyRoles,
    LegTarget,
    _toe_lift,
    get_roles,
    guess_roles,
    quat_axis_angle,
    register_roles,
    solve_leg_ik,
)
from afv.motion.rotations import IDENTITY, qconj, qmul, qrotate


def _bent_pose(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rot = np.tile(IDENTITY, (n, SK.num_joints, 1))
    for side in "lr":
        rot[:, SK.index(f"{side}_thigh")] = quat_axis_angle(
            np.tile([1.0, 0, 0], (n, 1)), -rng.uniform(0.2, 0.9, n)
        )
        rot[:, SK.index(f"{side}_calf")] = quat_axis_angle(
            np.tile([1.0, 0, 0], (n, 1)), rng.uniform(0.3, 1.2, n)
        )
    root = np.tile([0.0, 0.9, 0.0], (n, 1)) + rng.normal(0, 0.02, (n, 3))
    return rot, root


def test_provisional_roles_valid_and_guessable():
    ROLES.validate(SK)
    g = guess_roles(SK)
    assert g.legs == ROLES.legs
    assert g.wrists == ROLES.wrists
    assert [c.channel for c in g.contacts] == ["l_heel", "l_toe", "r_heel", "r_toe"]
    assert get_roles(SK) is ROLES


def test_roles_dict_roundtrip_and_regions():
    d = ROLES.to_dict()
    back = BodyRoles.from_dict(d)
    assert back.to_dict() == d
    reg = ROLES.regions(SK)
    assert reg[SK.index("l_calf")] == "legs"
    assert reg[SK.index("r_forearm")] == "arms"
    assert reg[SK.index("r_hand")] == "arms"
    assert reg[SK.index("r_index2")] == "hands"
    assert reg[SK.index("head")] == "head"
    assert reg[SK.index("spine2")] == "torso"
    clear = ROLES.clearances(SK)
    assert clear["l_heel"] == pytest.approx(0.08)
    assert clear["l_toe"] == pytest.approx(0.02)


def test_guess_roles_mixamo_style_names():
    names = [
        "Hips",
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "Head",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "LeftToeBase",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "RightToeBase",
    ]
    par = {
        "Hips": None,
        "Spine": "Hips",
        "Spine1": "Spine",
        "Spine2": "Spine1",
        "Neck": "Spine2",
        "Head": "Neck",
        "LeftShoulder": "Spine2",
        "LeftArm": "LeftShoulder",
        "LeftForeArm": "LeftArm",
        "LeftHand": "LeftForeArm",
        "RightShoulder": "Spine2",
        "RightArm": "RightShoulder",
        "RightForeArm": "RightArm",
        "RightHand": "RightForeArm",
        "LeftUpLeg": "Hips",
        "LeftLeg": "LeftUpLeg",
        "LeftFoot": "LeftLeg",
        "LeftToeBase": "LeftFoot",
        "RightUpLeg": "Hips",
        "RightLeg": "RightUpLeg",
        "RightFoot": "RightLeg",
        "RightToeBase": "RightFoot",
    }
    parents = tuple(-1 if par[n] is None else names.index(par[n]) for n in names)
    sk = Skeleton("mixamo_test", tuple(names), parents, np.ones((len(names), 3)) * 0.1)
    r = guess_roles(sk)
    r.validate(sk)
    assert r.legs[0].thigh == "LeftUpLeg" and r.legs[0].calf == "LeftLeg" and r.legs[0].toe == "LeftToeBase"
    assert r.wrists == ("LeftHand", "RightHand")
    assert r.head == "Head"
    assert any(c.a == "Hips" and c.b == "Spine2" for c in r.capsules)
    assert r.channel("l_heel").sources[0] == "left_ankle"
    register_roles("mixamo_test", r)
    assert get_roles(sk) is r


def test_ik_reaches_targets_and_keeps_knee_forward():
    rng = np.random.default_rng(3)
    n = 50
    rot, root = _bent_pose(n, rng)
    # Reachable targets: the ankles of another random bent pose with the same root.
    other, _ = _bent_pose(n, rng)
    goal, _ = forward_kinematics(SK, other, root)
    targets = []
    want = {}
    for leg in ROLES.legs:
        t = goal[:, SK.index(leg.foot)]
        want[leg.foot] = t
        targets.append(LegTarget(leg, t, np.ones(n, bool)))
    new = solve_leg_ik(SK, rot, root, targets, forward=np.array([0, 0, 1.0]))
    p2, w2 = forward_kinematics(SK, new, root)
    for leg in ROLES.legs:
        assert np.abs(p2[:, SK.index(leg.foot)] - want[leg.foot]).max() < 1e-6
        # Knee still bends forward (+Z) relative to the hip-ankle line.
        h, k, a = (p2[:, SK.index(j)] for j in (leg.thigh, leg.calf, leg.foot))
        line = a - h
        line /= np.linalg.norm(line, axis=1, keepdims=True)
        off = (k - h) - np.sum((k - h) * line, axis=1, keepdims=True) * line
        assert np.all(off[:, 2] > 0)
    # Bone lengths are rotations only, so FK preserves them; foot world orientation is kept.
    _, w1 = forward_kinematics(SK, rot, root)
    for leg in ROLES.legs:
        j = SK.index(leg.foot)
        assert np.allclose(np.abs(np.sum(w1[:, j] * w2[:, j], -1)), 1.0, atol=1e-9)


def test_ik_inactive_frames_untouched_and_toe_target():
    rng = np.random.default_rng(4)
    n = 20
    rot, root = _bent_pose(n, rng)
    pos, _ = forward_kinematics(SK, rot, root)
    leg = ROLES.legs[0]
    active = np.zeros(n, bool)
    active[5:10] = True
    shift = np.array([0.0, 0.04, 0.03])  # up and forward: always reachable from a bent leg
    ankle = pos[:, SK.index(leg.foot)] + shift
    toe_goal = pos[:, SK.index(leg.toe)] + np.array([0.0, -0.04, 0.0]) + shift
    new = solve_leg_ik(SK, rot, root, [LegTarget(leg, ankle, active, toe_goal)])
    assert np.array_equal(new[~active], rot[~active])
    p2, _ = forward_kinematics(SK, new, root)
    assert np.abs(p2[active, SK.index(leg.foot)] - ankle[active]).max() < 1e-6
    # Toe lands on the target direction from the ankle (target at foot length).
    v_goal = toe_goal[active] - ankle[active]
    v_got = p2[active, SK.index(leg.toe)] - p2[active, SK.index(leg.foot)]
    cos = np.sum(v_goal * v_got, 1) / np.linalg.norm(v_goal, axis=1) / np.linalg.norm(v_got, axis=1)
    assert np.all(cos > 1 - 1e-9)


def test_ik_unreachable_target_straightens_leg():
    n = 3
    rot = np.tile(IDENTITY, (n, SK.num_joints, 1))
    rot[:, SK.index("l_calf")] = quat_axis_angle(np.tile([1.0, 0, 0], (n, 1)), np.full(n, 0.5))
    root = np.tile([0.0, 0.95, 0.0], (n, 1))
    leg = ROLES.legs[0]
    far = np.tile([0.09, -0.5, 0.0], (n, 1))
    new = solve_leg_ik(SK, rot, root, [LegTarget(leg, far, np.ones(n, bool))])
    p, _ = forward_kinematics(SK, new, root)
    hip, ank = p[:, SK.index(leg.thigh)], p[:, SK.index(leg.foot)]
    assert np.allclose(np.linalg.norm(ank - hip, axis=1), 0.82, atol=1e-4)
    assert np.all(np.isfinite(new))


@pytest.mark.parametrize("pitch", [0.6, 1.0, 1.3])
def test_toe_floor_lift_is_exact_for_a_steep_foot(pitch):
    """Round 2 fix: a foot pointed steeply down with its toe centimetres below the floor is lifted in
    one exact step (the former 3-step fixed-point iteration stopped up to ~2 cm short on real data,
    so the contacts output had toes below the floor and every later re-pin moved them)."""
    n = 1
    rot = np.tile(IDENTITY, (n, SK.num_joints, 1))
    rot[:, SK.index("l_calf")] = quat_axis_angle(np.tile([1.0, 0, 0], (n, 1)), np.full(n, 0.4))
    root = np.tile([0.0, 0.95, 0.0], (n, 1))
    leg = ROLES.legs[0]
    it, ia = SK.index(leg.toe), SK.index(leg.foot)
    foot = SK.index(leg.foot)
    best = None
    for sgn in (1.0, -1.0):  # whichever pitch sense points the toe down
        r = rot.copy()
        r[:, foot] = quat_axis_angle(np.tile([1.0, 0, 0], (n, 1)), np.full(n, sgn * pitch))
        p, _ = forward_kinematics(SK, r, root)
        if best is None or p[0, it, 1] < best[1][0, it, 1]:
            best = (r, p)
    rot, p = best
    assert p[0, it, 1] < p[0, ia, 1] - 0.03  # toe well below the ankle
    clear = 0.02
    off = clear - 0.05 - p[0, it, 1]  # floor puts the toe 5 cm below its clearance
    tg = LegTarget(leg, p[:, ia], np.ones(n, bool), floor=(np.array([0.0, 1.0, 0.0]), off, clear, 0.0, None))
    new = solve_leg_ik(SK, rot, root, [tg])
    q, wq = forward_kinematics(SK, new, root)
    _, wp = forward_kinematics(SK, rot, root)
    assert abs(q[0, it, 1] + off - clear) < 1e-6
    assert np.allclose(q[0, ia], p[0, ia], atol=1e-9)  # ankle untouched
    up = np.array([0.0, 1.0, 0.0])
    v0, v1 = p[0, it] - p[0, ia], q[0, it] - q[0, ia]
    assert np.linalg.norm(v1) == pytest.approx(np.linalg.norm(v0), abs=1e-9)  # foot length kept
    h0, h1 = v0 - (v0 @ up) * up, v1 - (v1 @ up) * up
    assert h1 @ h0 > 0  # not swung over to the heel side
    assert np.allclose(h1 / np.linalg.norm(h1), h0 / np.linalg.norm(h0), atol=1e-9)  # direction kept
    # Minimal pitch: the foot turned only about the horizontal axis across the toe direction, by the
    # angle between the two ankle->toe vectors.
    d = qmul(wq[0, ia], qconj(wp[0, ia]))
    d = d if d[3] >= 0 else -d
    angle = 2 * np.arctan2(np.linalg.norm(d[:3]), d[3])
    lateral = np.cross(up, h0 / np.linalg.norm(h0))
    assert abs(abs(d[:3] / np.linalg.norm(d[:3]) @ lateral) - 1.0) < 1e-9
    want = np.arccos(np.clip(v0 @ v1 / np.linalg.norm(v0) / np.linalg.norm(v1), -1, 1))
    assert angle == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("fallback", [None, np.array([[0.0, 0.0, 1.0]])])
def test_toe_lift_toe_straight_below_the_ankle(fallback):
    """Round 2: a toe exactly below the ankle has no horizontal direction of its own; it is still
    lifted exactly (the former code skipped it), towards the foot's up axis when given (forward for a
    toe pointing down), keeping the foot's length."""
    rel = np.array([[0.0, -0.12, 0.0]])
    q = _toe_lift(rel, np.array([0.05]), np.array([0.0, 1.0, 0.0]), fallback)
    v = qrotate(q, rel)[0]
    assert v[1] == pytest.approx(-0.07, abs=1e-12)
    assert np.linalg.norm(v) == pytest.approx(0.12, abs=1e-12)
    if fallback is not None:
        assert v[0] == pytest.approx(0.0, abs=1e-12) and v[2] > 0.09  # swung forward


def test_ik_bends_a_near_straight_knee_forward():
    """Round 3 review: a near-straight input knee that is slightly hyperextended (GEM-X: -0.1 to -5
    deg) kept its backward side whatever the target, so a pin pulling the ankle in bent it 50 deg
    backwards (ref walk A, ufc A). It now bends forward unless the target needs no more bend than its
    own hyperextension (+ ``KNEE_HYPER_TOL``); target == current stays an exact no-op."""
    n = 2
    rot = np.tile(IDENTITY, (n, SK.num_joints, 1))
    leg = ROLES.legs[0]
    rot[:, SK.index(leg.calf)] = quat_axis_angle(np.tile([1.0, 0, 0], (n, 1)), np.full(n, -0.03))
    root = np.tile([0.0, 0.95, 0.0], (n, 1))
    pos, _ = forward_kinematics(SK, rot, root)
    ankle = pos[:, SK.index(leg.foot)].copy()
    ankle[1] += np.array([0.0, 0.12, 0.0])  # frame 1: pulled 12 cm up (about 45 deg of bend)
    new = solve_leg_ik(
        SK, rot, root, [LegTarget(leg, ankle, np.ones(n, bool))], forward=np.array([0, 0, 1.0])
    )
    assert np.allclose(np.abs(np.sum(new[0] * rot[0], -1)), 1.0)  # unchanged target: exact no-op
    p2, _ = forward_kinematics(SK, new, root)
    assert np.abs(p2[1, SK.index(leg.foot)] - ankle[1]).max() < 1e-6
    h, k, a = (p2[1, SK.index(j)] for j in (leg.thigh, leg.calf, leg.foot))
    line = (a - h) / np.linalg.norm(a - h)
    off = (k - h) - np.dot(k - h, line) * line
    assert off[2] > 0.05  # the knee went forward (+Z), not 45 deg backwards
