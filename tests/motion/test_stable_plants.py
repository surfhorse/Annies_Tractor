"""Round 5, track M: plant decisions stable to GPU-level input noise.

* The root correction's rate limit is a convex one-sided penalty (one minimum, no path dependence):
  one candidate plant frame more or less moves the offset about as much as without the limit.
* The contact re-pin of later stages (smooth, retime) lowers the pelvis for the IK's real reach, so
  a nearly straight planted leg follows the reference exactly and no planted step grows.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from motion_synth import FOOT_CHANNELS, ROLES, SK, body

from afv.core.types import ActorMotion
from afv.motion.contacts import ContactParams, Floor, hold_contacts, solve_root_correction
from afv.motion.fk import forward_kinematics
from afv.motion.ik import LegTarget, knee_hinge_local, leg_max_reach, quat_axis_angle, solve_leg_ik
from afv.motion.rotations import qmul


def _standing(n: int = 40, knee: float | np.ndarray = 0.4, twist: float | np.ndarray = 0.0) -> ActorMotion:
    """A still fighter, knees bent by ``knee`` rad (about x, the knee hinge); ``twist`` rad turns each
    calf sideways (about z), out of the hinge plane. Both may vary per frame."""
    knee = np.broadcast_to(np.asarray(knee, np.float64), (n,))
    twist = np.broadcast_to(np.asarray(twist, np.float64), (n,))
    rot = np.tile([0.0, 0.0, 0.0, 1.0], (n, SK.num_joints, 1))
    for side in "lr":
        rot[:, SK.index(f"{side}_thigh")] = quat_axis_angle(np.tile([1.0, 0, 0], (n, 1)), -knee / 2)
        calf = quat_axis_angle(np.tile([1.0, 0, 0], (n, 1)), knee)
        rot[:, SK.index(f"{side}_calf")] = qmul(quat_axis_angle(np.tile([0.0, 0, 1.0], (n, 1)), twist), calf)
    root = np.tile([0.0, 1.0, 0.0], (n, 1))
    return ActorMotion(rot=rot, root=root)


def _bent_then_straight(n: int) -> ActorMotion:
    """Knees bent 0.8 rad on the first half (that bend sets the knee hinge the IK uses), nearly straight
    with the calves turned 0.25 rad out of the hinge plane on the second half."""
    h = n // 2
    return _standing(
        n, knee=np.r_[np.full(h, 0.8), np.full(n - h, 0.05)], twist=np.r_[np.zeros(h), np.full(n - h, 0.25)]
    )


def _sliding_stances(n: int = 60, right: bool = False):
    """Left-foot stances whose planted foot slides (GEM-X's root drift): the offset that holds them
    still must swing between them faster than the rate limit allows. ``right``: plus a right-foot
    plant sliding sideways over the first stance and the gap."""
    a = _standing(n)
    pos = forward_kinematics(SK, a.rot, a.root)[0]
    masks = {ch: np.zeros(n, dtype=bool) for ch in FOOT_CHANNELS}
    stances = ((3, 18, 0.0373), (23, 44, 0.005)) if right else ((2, 22, 0.02), (30, 52, -0.025))
    for s0, e0, v in stances:
        masks["l_heel"][s0:e0] = True
        pos[s0:e0, SK.index("l_foot"), 2] += v * np.arange(e0 - s0)
    if right:
        masks["r_heel"][2:27] = True
        pos[2:27, SK.index("r_foot"), 0] += 0.0233 * np.arange(25)
    return pos, masks


def test_root_correction_moves_little_when_one_plant_frame_flips():
    """close_boxing A (round 4 golden repeats 3.97 / 4.09 m smoothed root path): a candidate plant's
    first frame crossing its threshold on GPU-level noise moved the rate-limited root offset by 8 cm
    (1.1 cm without the limit), because the limit rows were switched on with the current rate as
    their target and kept on (path dependent). Now one frame changes the offset about as much as it
    does without the limit."""
    pos, masks = _sliding_stances(right=True)
    p = ContactParams()
    free = ContactParams(root_max_speed=0.0, root_speed_ratio=0.0)
    flip = {k: v.copy() for k, v in masks.items()}
    flip["l_heel"][22] = True  # the second stance starts one frame earlier
    fl = Floor.horizontal(0.0)
    change = {}
    for name, q in (("limited", p), ("free", free)):
        d0 = solve_root_correction(SK, ROLES, pos, masks, fl, q, None, 30.0)
        d1 = solve_root_correction(SK, ROLES, pos, flip, fl, q, None, 30.0)
        change[name] = float(np.linalg.norm(d0 - d1, axis=-1).max())
        if name == "limited":
            rate = np.linalg.norm(np.diff(d0, axis=0), axis=-1) * 30.0
            assert rate.max() <= p.root_max_speed * 1.05, rate.max()
    assert change["limited"] <= 1.5 * change["free"] + 0.001, change  # was 0.92 cm against 0.24 cm


def test_root_correction_is_continuous_in_its_input():
    """A sub-millimetre change of the input (GPU-level noise) moves the rate-limited offset by about
    as much, not by a jump: the penalty has one minimum and the iterations find it."""
    pos, masks = _sliding_stances()
    rng = np.random.default_rng(3)
    d0 = solve_root_correction(SK, ROLES, pos, masks, Floor.horizontal(0.0), ContactParams(), None, 30.0)
    for _ in range(5):
        noisy = pos + rng.normal(0.0, 2e-4, pos.shape)
        d1 = solve_root_correction(
            SK, ROLES, noisy, masks, Floor.horizontal(0.0), ContactParams(), None, 30.0
        )
        assert np.linalg.norm(d1 - d0, axis=-1).max() < 0.004


def test_leg_max_reach_is_what_the_ik_reaches():
    """The knee turns the calf about its hinge only: with the calf off the hinge plane the leg's
    reach is shorter than the bones' sum. A target at that reach is hit exactly; one at the bones'
    sum is not (that gap left a re-pinned straight planted leg ~0.1 mm off its pin)."""
    n = 8
    a = _bent_then_straight(n)
    pos, wrot = forward_kinematics(SK, a.rot, a.root)
    leg = ROLES.legs[0]
    ih, ik, ia = (SK.index(j) for j in (leg.thigh, leg.calf, leg.foot))
    hinge = knee_hinge_local(SK, wrot, pos, leg, np.asarray(ROLES.forward))
    reach = leg_max_reach(SK, wrot, pos, leg, hinge)
    bones = np.linalg.norm(pos[:, ik] - pos[:, ih], axis=-1) + np.linalg.norm(
        pos[:, ia] - pos[:, ik], axis=-1
    )
    straight = np.arange(n) >= n // 2
    assert np.all(reach[straight] < bones[straight] - 1e-4)
    down = pos[:, ia] - pos[:, ih]
    down /= np.linalg.norm(down, axis=-1, keepdims=True)
    for dist, exact in ((reach * (1 - 1e-6), True), (bones, False)):
        target = pos[:, ih] + dist[:, None] * down
        tg = LegTarget(leg, target, np.ones(n, bool), hinge_local=hinge)
        out = forward_kinematics(SK, solve_leg_ik(SK, a.rot, a.root, [tg]), a.root)[0]
        miss = np.linalg.norm(out[straight, ia] - target[straight], axis=-1)
        assert (miss.max() < 1e-6) == exact, miss


def test_repin_holds_a_nearly_straight_planted_leg_exactly():
    """smooth's re-pin left the planted joints of a nearly straight leg ~0.1 mm off the reference at
    plant edges, so planted steps grew: projects/ufc_cross B frame 249 2.29 cm/s in contacts, 2.70
    after smooth; the golden ufc_cross B frame 248 0.00 -> 0.53 cm/s. With the pelvis lowered for the
    IK's real reach the planted joints follow the reference exactly."""
    n = 30
    ref = _bent_then_straight(n)
    pos = forward_kinematics(SK, ref.rot, ref.root)[0]
    clear = ROLES.clearances(SK)
    sole = min(float(pos[-1, SK.index(c.joint), 1]) - clear[c.channel] for c in ROLES.contacts)
    ref.root[:, 1] -= sole  # straight-legged frames on the floor (bent ones above it)
    contact = np.zeros((n, len(FOOT_CHANNELS)), np.float32)
    contact[18:28] = 1.0  # all four channels planted on straight-legged frames
    ref.contact = contact
    bm = body({"A": ref}, channels=FOOT_CHANNELS)
    # "smoothed": hips 4 mm higher, so the straight legs cannot reach their pins without a pelvis drop
    root = np.asarray(ref.root, np.float64) + np.array([0.0, 0.004, 0.0])
    fix = hold_contacts(bm, {"A": ref.rot}, {"A": root}, ROLES, ContactParams())["A"]
    out = forward_kinematics(SK, fix.rot, fix.root)[0]
    ref_pos = forward_kinematics(SK, ref.rot, ref.root)[0]
    for c in ROLES.contacts:
        j = SK.index(c.joint)
        err = np.linalg.norm(out[18:28, j] - ref_pos[18:28, j], axis=-1)
        assert err.max() < 1e-5, (c.channel, err.max())
    assert fix.pelvis_drop.max() <= ContactParams().max_pelvis_drop + 1e-9


def test_root_solve_stops_close_to_its_converged_solution(monkeypatch):
    """The root solve converges linearly, and can do so fast at first and slowly after: on the first
    1.5 s of the close_boxing fixture fighter B's first solve takes a few fast steps, then needs ~250
    more. Stopping on a fixed step (round 5 pass 2: 0.05 mm, at most 40 iterations) left it 0.40 mm
    from the converged solve, and on the contraction rate of the fast steps alone (pass 3) 1.57 mm.
    With the rate floored at ``ROOT_SOLVE_RHO_MIN`` it ends within 2 x ``ROOT_SOLVE_TOL`` (0.09 mm)."""
    from real_metrics import FIXTURES, load_clip

    import afv.motion.contacts as C
    from afv.motion.pair import PairParams, pair_fighters

    if not (FIXTURES["close_boxing"] / "pose3d_close_boxing.npz").exists():
        pytest.skip("real GEM-X fixture missing")
    bm, cam = load_clip("close_boxing")
    n = 45  # the first 1.5 s
    bm = replace(
        bm,
        frames=bm.frames[:n],
        timestamps=bm.timestamps[:n],
        actors={
            r: replace(
                a,
                **{
                    f: getattr(a, f)[:n]
                    for f in ("rot", "root", "conf", "contact", "root_prior", "weak")
                    if getattr(a, f, None) is not None
                },
            )
            for r, a in bm.actors.items()
        },
    )
    pair, _ = pair_fighters(bm, cam, PairParams())
    err: list[float] = []
    solve = C.solve_root_correction

    def checked(*a, **k):
        got = solve(*a, **k)
        it, tol = C.ROOT_SOLVE_ITERS, C.ROOT_SOLVE_TOL
        C.ROOT_SOLVE_ITERS, C.ROOT_SOLVE_TOL = 6000, 1e-13  # the converged solve of the same inputs
        try:
            err.append(float(np.abs(got - solve(*a, **k)).max()))
        finally:
            C.ROOT_SOLVE_ITERS, C.ROOT_SOLVE_TOL = it, tol
        return got

    monkeypatch.setattr(C, "solve_root_correction", checked)
    C.clean_contacts(pair, C.ContactParams(), camera=cam)
    assert len(err) == 4
    assert max(err) <= 2 * 1e-4, err


def test_rest_position_cache_notices_in_place_changes():
    """``BodyRoles.clearances`` caches rest positions by the skeleton's contents: changing a fighter's
    bone offsets in place must give new clearances, not the cached ones."""

    sk = replace(SK, rest_offsets=np.array(SK.rest_offsets, np.float64, copy=True))
    before = ROLES.clearances(sk)
    sk.rest_offsets[SK.index("l_foot")] *= 1.5  # a longer shin
    after = ROLES.clearances(sk)
    assert after != before
    assert ROLES.clearances(
        replace(SK, rest_offsets=np.array(SK.rest_offsets, copy=True))
    ) == ROLES.clearances(SK)
