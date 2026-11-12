"""Pair: shared floor, depth smoothed along the camera rays (projection kept), push-apart."""

from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp
from motion_synth import (
    GEMX_CHANNELS,
    ROLES,
    SK,
    along_ray,
    body,
    look_at_camera,
    model_probs,
    telephoto_camera,
    walk,
)

from afv.core.types import ActorMotion, BodyMotion, CameraTrack
from afv.motion.fk import forward_kinematics
from afv.motion.ik import get_roles
from afv.motion.pair import (
    PairParams,
    _solve_bordered_banded,
    capsule_penetration,
    pair_fighters,
    segment_distance,
    solve_depth_and_floor,
)
from afv.qc.camera import project

N = 150


def _jittery_pair(seed: int = 1):
    rng = np.random.default_rng(seed)
    a, ma = walk(seconds=N / 30, speed=0.3, origin=(-0.7, 0.0), heading=np.pi / 2)
    b, mb = walk(seconds=N / 30, speed=0.3, origin=(0.9, 0.3), heading=-np.pi / 2)
    cam = telephoto_camera(N)
    cp = cam.cam_pos[0]
    dz_a = rng.normal(0, 0.25, N)
    dz_a[40:43] += 1.9  # P0: depth jumps up to 1.9 m per frame
    dz_b = rng.normal(0, 0.25, N) + 0.4
    dz_b[90] -= 1.9
    ab, bb = along_ray(a, cp, dz_a), along_ray(b, cp, dz_b)
    ab.contact, bb.contact = model_probs(ma, rng), model_probs(mb, rng)
    return {"A": a, "B": b}, body({"A": ab, "B": bb}, GEMX_CHANNELS), cam, {"A": ma, "B": mb}


def test_segment_distance():
    p0, p1 = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
    assert np.isclose(segment_distance(p0, p1, np.array([0.5, 1, 0]), np.array([0.5, 2, 0])), 1.0)
    assert np.isclose(segment_distance(p0, p1, np.array([2.0, 0, 0]), np.array([3.0, 0, 0])), 1.0)
    assert np.isclose(segment_distance(p0, p1, np.array([0.5, -1, 1]), np.array([0.5, 1, 1])), 1.0)


def test_depth_smoothed_projection_kept_shared_floor():
    clean, bm, cam, masks = _jittery_pair()
    cp = cam.cam_pos[0]
    out, rep = pair_fighters(bm, cam)
    for r in ("A", "B"):
        z_clean = np.linalg.norm(clean[r].root - cp, axis=-1)
        z_in = np.linalg.norm(bm.actors[r].root - cp, axis=-1)
        z_out = np.linalg.norm(out.actors[r].root.astype(np.float64) - cp, axis=-1)
        uv_in = project(bm.actors[r].root[:, None], cam, bm.frames)[:, 0]
        unshift = np.asarray(out.world_shift)  # the floor -> y = 0 translation, recorded
        assert np.allclose(unshift, [0.0, rep.floor_shift, 0.0])
        uv_out = project((out.actors[r].root.astype(np.float64) - unshift)[:, None], cam, bm.frames)[:, 0]
        jump_in, jump_out = np.abs(np.diff(z_in)).max(), np.abs(np.diff(z_out)).max()
        err_in = np.std(z_in - z_clean)
        err_out = np.std(z_out - z_clean)
        print(
            f"{r}: depth jump {jump_in:.2f} -> {jump_out:.3f} m/frame, depth error std {err_in:.2f} -> "
            f"{err_out:.3f} m, root reprojection change {np.nanmax(np.abs(uv_out - uv_in)):.4f} px"
        )
        assert jump_in > 1.5 and jump_out < 0.05
        assert err_out < 0.25 * err_in
        assert np.nanmax(np.abs(uv_out - uv_in)) < 0.01  # moved along the ray: same pixel
    # Both fighters stand on the same floor during their plants.
    heights = []
    for r in ("A", "B"):
        a = out.actors[r]
        pos = forward_kinematics(SK, a.rot, a.root)[0]
        m = masks[r]["l_heel"]
        heights.append(pos[m, SK.index("l_foot"), 1] - 0.08 - (-rep.floor.offset))
    assert all(np.abs(np.median(h)) < 0.02 for h in heights)
    assert abs(np.median(heights[0]) - np.median(heights[1])) < 0.02


def test_world_shift_respected_and_accumulated():
    _, bm, cam, _ = _jittery_pair()
    ws = np.array([0.0, 0.3, 0.0])  # e.g. pose3d already moved the world up 30 cm
    for a in bm.actors.values():
        a.root = a.root + ws
    bm.world_shift = list(ws)
    out, rep = pair_fighters(bm, cam)
    assert np.allclose(out.world_shift, ws + np.array([0.0, rep.floor_shift, 0.0]))
    for r in ("A", "B"):
        uv_in = project((bm.actors[r].root - ws)[:, None], cam, bm.frames)[:, 0]
        uv_out = project((out.actors[r].root.astype(np.float64) - out.world_shift)[:, None], cam, bm.frames)[
            :, 0
        ]
        assert np.nanmax(np.abs(uv_out - uv_in)) < 0.01


def test_camera_ground_prior_removes_common_offset():
    clean, bm, cam, _ = _jittery_pair()
    cp = cam.cam_pos[0]
    out, rep = pair_fighters(bm, cam, PairParams(camera_ground_sigma=0.001))
    assert abs(rep.floor_shift) < 0.01  # the shared floor came out at the camera's ground (y = 0)
    for r in ("A", "B"):
        z_clean = np.linalg.norm(clean[r].root - cp, axis=-1)
        z_out = np.linalg.norm(out.actors[r].root.astype(np.float64) - cp, axis=-1)
        assert abs(np.mean(z_out - z_clean)) < 0.1


def test_push_apart_overlapping_fighters():
    a, _ = walk(seconds=2.0, speed=0.0, origin=(0.0, 0.0))
    b, _ = walk(seconds=2.0, speed=0.0, origin=(0.05, -0.12), heading=np.pi)
    cam = telephoto_camera(60)
    bm = body({"A": a, "B": b})
    caps = tuple(c for c in ROLES.capsules if c.group != "arm")
    pa, pb = (forward_kinematics(SK, x.rot, x.root)[0] for x in (a, b))
    assert capsule_penetration(pa, pb, SK, SK, caps).max() > 0.1
    out, rep = pair_fighters(bm, cam)
    pa, pb = (forward_kinematics(SK, out.actors[r].rot, out.actors[r].root)[0] for r in ("A", "B"))
    assert capsule_penetration(pa, pb, SK, SK, caps).max() <= 0.005
    assert np.ptp(rep.push) < 0.02  # constant overlap -> constant push, no pops
    for r, x in (("A", a), ("B", b)):
        uv0 = project(x.root[:, None], cam, bm.frames)[:, 0]
        uv1 = project((out.actors[r].root.astype(np.float64) - out.world_shift)[:, None], cam, bm.frames)[
            :, 0
        ]
        assert np.nanmax(np.abs(uv1 - uv0)) < 0.01
    # A (nearer the camera at +Z) came forward, B went back.
    assert out.actors["A"].root[0, 2] > a.root[0, 2] and out.actors["B"].root[0, 2] < b.root[0, 2]


def test_without_camera_aligns_floors():
    a, _ = walk(seconds=1.0, origin=(-1.0, 0.0))
    b, _ = walk(seconds=1.0, origin=(1.0, 0.0), floor_y=0.3)
    out, rep = pair_fighters(body({"A": a, "B": b}), None)
    assert rep.floor.offset == 0.0
    ya = forward_kinematics(SK, out.actors["A"].rot, out.actors["A"].root)[0][:, SK.index("l_toe"), 1].min()
    yb = forward_kinematics(SK, out.actors["B"].rot, out.actors["B"].root)[0][:, SK.index("l_toe"), 1].min()
    assert abs(ya - yb) < 0.005
    assert abs(ya - 0.02) < 0.005  # toe clearance above the y = 0 floor


def test_root_prior_velocity_fixes_depth():
    """With GEM-X's rollout as root_prior (its velocity is right, its position drifts), depth
    follows the rollout's velocity and jitter is gone even on a telephoto shot."""
    clean, bm, cam, _ = _jittery_pair(seed=3)
    for r, a in bm.actors.items():
        a.root_prior = (clean[r].root + np.array([0.5, 0.0, -0.8]) + 0.002 * np.arange(N)[:, None]).astype(
            np.float32
        )
    out, _ = pair_fighters(bm, cam)
    cp = cam.cam_pos[0]
    for r in ("A", "B"):
        z_clean = np.linalg.norm(clean[r].root - cp, axis=-1)
        z_out = np.linalg.norm(out.actors[r].root.astype(np.float64) - cp, axis=-1)
        assert np.abs(np.diff(z_out - z_clean)).max() < 0.03  # frame-to-frame depth follows the rollout
        assert np.std(z_out - z_clean) < 0.1


def _sole(bm, r: str, mask: np.ndarray) -> float:
    a = bm.actors[r]
    pos = forward_kinematics(SK, a.rot, a.root)[0]
    return float(np.median(pos[mask, SK.index("l_foot"), 1]))


def _biased_pair(cam, bias_b: float, n: int = N, seed: int = 4):
    """Two fighters in GEM-X style: B's depth biased by ``bias_b`` m along its rays (a per-fighter scale
    error), per-frame depth noise, and GEM-X's smooth rollout as ``root_prior`` (so the per-frame floor
    rows are weak, divided by the ray pitch)."""
    rng = np.random.default_rng(seed)
    a, ma = walk(seconds=n / 30, speed=0.3, origin=(-0.7, 0.0), heading=np.pi / 2)
    b, mb = walk(seconds=n / 30, speed=0.3, origin=(0.9, 0.3), heading=-np.pi / 2)
    cp = cam.cam_pos[0]
    out = {}
    for r, x, m, bias in (("A", a, ma, 0.0), ("B", b, mb, bias_b)):
        y = along_ray(x, cp, bias + rng.normal(0, 0.1, n))
        y.contact = model_probs(m, rng)
        y.root_prior = (x.root + np.array([0.3, 0.0, -0.5])).astype(np.float32)  # drifted, smooth
        out[r] = y
    return body(out, GEMX_CHANNELS), {"A": ma, "B": mb}


def test_local_floor_rows_fix_a_per_fighter_depth_bias():
    """Round 2: on telephoto shots one fighter's soles sat 10-20 cm above the other's after pair (a
    per-fighter GEM-X depth/scale bias of ~3 %; the pitch-divided per-frame floor rows could not move
    the mean depth). The local-mean floor rows put both on one floor, still along the rays."""
    cam = telephoto_camera(N)
    bm, masks = _biased_pair(cam, 0.9)
    old, _ = pair_fighters(bm, cam, PairParams(floor_mean_sigma=None, depth_mean_sigma_rel=None))
    new, rep = pair_fighters(bm, cam)
    gap = {
        k: abs(_sole(x, "A", masks["A"]["l_heel"]) - _sole(x, "B", masks["B"]["l_heel"]))
        for k, x in (("old", old), ("new", new))
    }
    print(f"sole gap between fighters: {gap['old'] * 100:.1f} -> {gap['new'] * 100:.1f} cm")
    assert gap["old"] > 0.06 and gap["new"] < 0.02
    for r in ("A", "B"):
        uv_in = project(bm.actors[r].root[:, None], cam, bm.frames)[:, 0]
        uv_out = project((new.actors[r].root.astype(np.float64) - new.world_shift)[:, None], cam, bm.frames)
        assert np.nanmax(np.abs(uv_out[:, 0] - uv_in)) < 0.01  # still along the ray
    # B's bias relative to A undone (the common depth of both stays unobservable: module docstring)
    rel = np.mean(rep.depth_change["B"]) - np.mean(rep.depth_change["A"])
    print(f"relative depth change {rel:.2f} m (bias -0.9)")
    assert abs(rel + 0.9) < 0.2


def test_close_range_floor_disagreement_keeps_a_plausible_depth():
    """At close range the same floor gap would need a depth change of ~25 % (an implausible scale
    error that also changes the fighters' separation): the non-robust local-mean depth rows keep the
    mean depth change within a few percent of the distance."""
    n = 90
    cam = look_at_camera(np.array([0.0, 1.5, 4.0]), np.array([0.0, 0.9, 0.0]), 1200.0, n)
    bm, _ = _biased_pair(cam, 0.0, n=n)
    b = bm.actors["B"]
    b.root = b.root + np.array([0.0, 0.10, 0.0], dtype=np.float32)  # B 10 cm above A, not along a ray
    _, rep = pair_fighters(bm, cam)
    dist = np.linalg.norm(b.root - cam.cam_pos[0], axis=-1).mean()
    moved = {r: abs(np.mean(rep.depth_change[r])) for r in ("A", "B")}
    _, rep_free = pair_fighters(bm, cam, PairParams(depth_mean_sigma_rel=None))
    print(
        f"mean depth change A {moved['A']:.2f} B {moved['B']:.2f} m at {dist:.1f} m "
        f"(without the mean-depth rows B {abs(np.mean(rep_free.depth_change['B'])):.2f} m)"
    )
    assert max(moved.values()) <= 0.12  # measured 0.10 / 0.07 m; the gap alone would ask ~1 m
    assert abs(np.mean(rep_free.depth_change["B"])) > moved["B"]
    assert dist > 3.0


def test_push_order_does_not_flip_while_passing():
    """Round 2: two people passing each other on screen at the same GEM-X depth (ref walk) flipped the
    push direction when the smoothed depth difference changed sign: both roots jumped by the whole
    push (72 cm in one frame). One order per push episode."""
    n = 90
    a, _ = walk(seconds=n / 30, speed=0.0, origin=(0.0, 0.0))
    b, _ = walk(seconds=n / 30, speed=0.0, origin=(0.08, 0.0), heading=np.pi)
    cam = telephoto_camera(n)
    cp = cam.cam_pos[0]
    b = along_ray(b, cp, np.linspace(-0.15, 0.15, n))  # B's depth crosses A's mid-shot
    bm = body({"A": a, "B": b})
    out, rep = pair_fighters(bm, cam)
    caps = tuple(c for c in ROLES.capsules if c.group != "arm")
    pa, pb = (forward_kinematics(SK, out.actors[r].rot, out.actors[r].root)[0] for r in ("A", "B"))
    assert rep.push.max() > 0.2
    assert capsule_penetration(pa, pb, SK, SK, caps).max() <= 0.005
    for r in ("A", "B"):
        step = np.linalg.norm(np.diff(out.actors[r].root.astype(np.float64), axis=0), axis=-1)
        assert step.max() < 0.03, (r, step.max())


def test_push_never_sweeps_one_fighter_through_the_other():
    """Review: one order per episode, chosen without looking at the frames around it, pushed B through
    A when GEM-X has B pass from behind A to in front of it on screen: the push ramp before the overlap
    drove the bodies into each other (18 cm overlap, 6.6 cm/frame root steps). The order comes from the
    depth order just outside the episode, and every frame the push touches gets a separation from which
    the bodies stay clear in that order."""
    n = 60
    a, _ = walk(seconds=n / 30, speed=0.0, origin=(0.0, 0.0))
    b, _ = walk(seconds=n / 30, speed=0.0, origin=(0.08, 0.0), heading=np.pi)
    cam = telephoto_camera(n)
    b = along_ray(b, cam.cam_pos[0], np.linspace(0.6, -0.6, n))  # B: 0.6 m behind A -> 0.6 m in front
    bm = body({"A": a, "B": b})
    out, rep = pair_fighters(bm, cam, PairParams(floor_mean_sigma=None, depth_mean_sigma_rel=None))
    caps = tuple(c for c in ROLES.capsules if c.group != "arm")
    pa, pb = (forward_kinematics(SK, out.actors[r].rot, out.actors[r].root)[0] for r in ("A", "B"))
    over = capsule_penetration(pa, pb, SK, SK, caps)
    steps = {
        r: np.linalg.norm(np.diff(out.actors[r].root.astype(np.float64), axis=0), axis=-1).max() for r in "AB"
    }
    print(f"max overlap {over.max() * 100:.1f} cm, max push {rep.push.max():.2f} m, root steps {steps}")
    assert rep.penetration_before.max() > 0.1
    assert over.max() <= 0.005
    assert max(steps.values()) < 0.06


def test_bordered_banded_solver_matches_a_general_sparse_solve():
    rng = np.random.default_rng(0)
    n, band = 300, 7
    offs = list(range(-band, band + 1))
    k = sp.diags([rng.uniform(0.1, 1.0, n - abs(o)) * (4.0 if o == 0 else 0.2) for o in offs], offs)
    k = k @ k.T + sp.identity(n)  # SPD, bandwidth 2 x band
    border = rng.normal(0, 0.1, n)
    lhs = sp.bmat([[k, border[:, None]], [border[None, :], np.array([[10.0]])]]).tocsc()
    rhs = rng.normal(size=n + 1)
    x = _solve_bordered_banded(lhs, rhs)
    assert np.allclose(lhs @ x, rhs, atol=1e-9)


def _tiled(bm: BodyMotion, cam: CameraTrack, reps: int) -> tuple[BodyMotion, CameraTrack]:
    def t(x):
        return None if x is None else np.concatenate([x if i % 2 == 0 else x[::-1] for i in range(reps)])

    n = bm.num_frames * reps
    actors = {
        r: ActorMotion(
            t(a.rot), t(a.root), contact=t(a.contact), rest_offsets=a.rest_offsets, root_prior=t(a.root_prior)
        )
        for r, a in bm.actors.items()
    }
    b2 = body(actors, list(bm.contact_channels))
    c2 = telephoto_camera(n)
    assert b2.num_frames == n
    return b2, c2


def test_depth_solve_scales_about_linearly():
    """Review: the solve was superlinear in the shot length (3000 frames: 5.7 s vs 0.5 s before the
    window rows): a general sparse LU of the bordered banded system. Now banded Cholesky + Schur."""
    _, bm, cam, _ = _jittery_pair()
    roles = get_roles(SK)
    times = {}
    for reps in (2, 20):
        b2, c2 = _tiled(bm, cam, reps)
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            solve_depth_and_floor(b2, c2, roles, PairParams())
            best = min(best, time.perf_counter() - t0)
        times[b2.num_frames] = best
    (n0, t0_), (n1, t1_) = sorted(times.items())
    print(f"depth solve: {n0} frames {t0_ * 1000:.0f} ms, {n1} frames {t1_ * 1000:.0f} ms")
    assert t1_ <= 3.0 * (n1 / n0) * t0_ + 0.05  # linear within a generous factor


def _moved(actor: ActorMotion, x: np.ndarray, z: np.ndarray) -> ActorMotion:
    root = np.asarray(actor.root, np.float64).copy()
    root[:, 0] += x
    root[:, 2] += z
    return ActorMotion(rot=actor.rot, root=root)


def _push_result(a: ActorMotion, b: ActorMotion):
    """Pair two fighters on the telephoto camera; returns B-A depth along the rays before/after
    (> 0: B behind A), the largest root step, the remaining capsule overlap per frame and the report."""
    n = a.rot.shape[0]
    cam = telephoto_camera(n)
    cp = cam.cam_pos[0]
    out, rep = pair_fighters(body({"A": a, "B": b}), cam)
    roots = [np.asarray(out.actors[r].root, np.float64) for r in ("A", "B")]

    def depth(ra, rb):
        return np.linalg.norm(rb - cp, axis=-1) - np.linalg.norm(ra - cp, axis=-1)

    d_in = depth(np.asarray(a.root, np.float64), np.asarray(b.root, np.float64))
    d_out = depth(*roots)
    step = max(np.linalg.norm(np.diff(r, axis=0), axis=-1).max() for r in roots)
    caps = tuple(c for c in ROLES.capsules if c.group != "arm")
    pa, pb = (forward_kinematics(SK, out.actors[r].rot, out.actors[r].root)[0] for r in ("A", "B"))
    return d_in, d_out, step, capsule_penetration(pa, pb, SK, SK, caps), rep


def test_two_clinches_keep_their_own_depth_order():
    """Review: the order episodes grew to the shot end on cover_smooth's float residue, so a second
    clinch with the opposite GEM-X order took the first one's and was swapped (-0.22 -> +0.30 m)."""
    n = 240
    t = np.arange(n)
    a, _ = walk(seconds=n / 30, speed=0.0)
    b, _ = walk(seconds=n / 30, speed=0.0, heading=np.pi)
    x = np.interp(t, [0, 40, 80, 120, 160, 240], [0.08, 0.08, 1.5, 1.5, 0.08, 0.08])
    z = np.interp(t, [0, 40, 80, 120, 160, 240], [-0.22, -0.22, -0.22, 0.22, 0.22, 0.22])
    d_in, d_out, step, over, rep = _push_result(a, _moved(b, x, z))
    assert rep.push[:40].min() > 0.05 and rep.push[160:].min() > 0.05  # both clinches pushed
    assert np.all(rep.push[90:110] == 0.0)  # nothing between them, not even float residue
    for sl in (slice(0, 40), slice(160, n)):
        assert np.all(np.sign(d_out[sl]) == np.sign(d_in[sl]))
    assert over.max() <= 0.005 and step < 0.05


def test_episode_at_the_shot_end_keeps_gemx_inner_order():
    """Review: one frame outside the episode (13 frames before the overlap, where GEM-X still had B
    in front) decided the order, so B was pushed through A to the front (+0.20 -> -0.60 m, push 0.80)."""
    n = 150
    t = np.arange(n)
    a, _ = walk(seconds=n / 30, speed=0.0)
    b, _ = walk(seconds=n / 30, speed=0.0, heading=np.pi)
    x = np.interp(t, [0, 40, 70, 150], [0.9, 0.5, 0.08, 0.08])
    z = np.interp(t, [0, 40, 70, 150], [0.12, 0.12, -0.2, -0.2])
    d_in, d_out, step, over, rep = _push_result(a, _moved(b, x, z))
    assert d_in[100] > 0.15 and d_out[100] > d_in[100]  # B stays behind A, pushed a little further
    assert rep.push.max() < 0.3 and over.max() <= 0.005 and step < 0.05


@pytest.mark.parametrize(
    "case", ["walk 3 m in 10 s", "walk 3 m at 0.9 m/s", "walk 6 m at 0.9 m/s", "mirrored 3 m at 0.9 m/s"]
)
def test_long_crossing_has_no_root_jump(case):
    """Review: when GEM-X has B walk straight through A (on screen overlap all the way), some frames
    must overlap whatever the order; growing the episode over the whole crossing pushed the full
    max_push and jumped 0.9-1.3 m in one frame. The push eases out over the far side instead
    (round 2 checkpoint HEAD: 7-8 cm steps, 21-25 cm overlap while passing; same here)."""
    if case == "walk 3 m in 10 s":
        n = 300
        b, _ = walk(seconds=n / 30, speed=0.3, origin=(0.08, 1.5), heading=np.pi)
    else:
        d0 = 3.0 if "6 m" in case else 1.5
        n = round(2 * d0 / 0.9 * 30)
        b, _ = walk(seconds=n / 30, origin=(0.08, -d0), heading=0.0)
        if case.startswith("mirrored"):
            b = ActorMotion(rot=np.asarray(b.rot)[::-1].copy(), root=np.asarray(b.root)[::-1].copy())
    a, _ = walk(seconds=n / 30, speed=0.0)
    d_in, _, step, over, rep = _push_result(a, b)
    print(f"{case}: max root step {step * 100:.1f} cm, overlap {over.max() * 100:.1f} cm")
    assert step < 0.08
    assert rep.push.max() < 0.9
    far = np.abs(d_in) > 1.3  # where GEM-X has them well apart nothing is pushed
    assert np.all(rep.push[far] == 0.0)
