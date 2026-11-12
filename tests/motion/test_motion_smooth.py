"""Strike-aware smoothing: jitter removal, no pops, strike peaks kept (NFR-4), contacts held."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from motion_synth import GEMX_CHANNELS, SK, body, corrupt, model_probs, punch, walk

from afv.core.types import ActorMotion
from afv.motion.contacts import clean_contacts
from afv.motion.fk import forward_kinematics
from afv.motion.rotations import qmul
from afv.motion.smooth import (
    SmoothParams,
    detect_strikes,
    position_spikes,
    qexp,
    qlog,
    rotation_spikes,
    smooth_motion,
)
from afv.qc.metrics import POP_THRESHOLD, per_frame_metrics, summarise


def _noisy(
    actor: ActorMotion, rng: np.random.Generator, deg: float = 1.0, root: float = 0.003
) -> ActorMotion:
    noise = rng.normal(0, np.radians(deg), (*actor.rot.shape[:2], 3))
    return ActorMotion(
        rot=qmul(actor.rot, qexp(noise)), root=actor.root + rng.normal(0, root, actor.root.shape)
    )


def _pos(a: ActorMotion) -> np.ndarray:
    return forward_kinematics(SK, a.rot, a.root)[0]


def _speed(p: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.gradient(p, axis=0), axis=-1) * 30.0


def test_log_exp_roundtrip():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 1.0, (100, 3))
    r *= np.minimum(1.0, 3.0 / np.linalg.norm(r, axis=1, keepdims=True))
    assert np.allclose(qlog(qexp(r)), r, atol=1e-10)


@pytest.mark.parametrize("seed", range(10))
def test_punch_peak_speed_kept(seed):
    rng = np.random.default_rng(seed)
    clean = punch(ext_frames=3)
    noisy = _noisy(clean, rng)
    out, info = smooth_motion(body({"A": noisy}))
    hand = SK.index("r_hand")
    window = slice(25, 45)
    v_clean = _speed(_pos(clean)[:, hand])[window].max()
    v_noisy = _speed(_pos(noisy)[:, hand])[window].max()
    v_out = _speed(_pos(out.actors["A"])[:, hand])[window].max()
    print(
        f"peak wrist speed clean {v_clean:.2f}, noisy {v_noisy:.2f}, smoothed {v_out:.2f} m/s "
        f"({v_out / v_clean:.0%} of clean, {v_out / v_noisy:.0%} of unsmoothed)"
    )
    assert any(s.effector == "r_hand" and 29 <= s.frame <= 34 for s in info["A"].strikes)
    assert v_out >= 0.8 * v_noisy  # NFR-4: >= 80% of the unsmoothed peak
    assert v_out >= 0.8 * v_clean
    # Jitter away from the strike is reduced (the 3 frames at each clip end are held to the input).
    quiet = np.r_[3:20, 50:57]
    pc = _pos(clean)
    err_in = np.sqrt(np.mean(np.sum((_pos(noisy) - pc)[quiet] ** 2, -1)))
    err_out = np.sqrt(np.mean(np.sum((_pos(out.actors["A"]) - pc)[quiet] ** 2, -1)))
    assert err_out < 0.6 * err_in


def test_without_strike_easing_peak_would_be_lost():
    rng = np.random.default_rng(0)
    clean = punch(ext_frames=3)
    noisy = _noisy(clean, rng)
    hand = SK.index("r_hand")
    v_noisy = _speed(_pos(noisy)[:, hand])[25:45].max()
    out, _ = smooth_motion(body({"A": noisy}), SmoothParams(strike_ease=1.0))
    assert _speed(_pos(out.actors["A"])[:, hand])[25:45].max() < 0.8 * v_noisy


def test_single_frame_pops_removed_strikes_not_popped():
    rng = np.random.default_rng(1)
    clean = punch(ext_frames=3)
    noisy = _noisy(clean, rng, deg=0.5, root=0.001)
    q = noisy.rot.copy()
    for f, j in ((10, "spine2"), (50, "l_forearm"), (15, "r_calf"), (55, "head")):
        q[f, SK.index(j)] = qmul(q[f, SK.index(j)], qexp(np.radians(15.0) * np.array([0.6, 0.0, 0.8])))
    popped = ActorMotion(rot=q, root=noisy.root)
    spikes = rotation_spikes(q)
    assert {(10, "spine2"), (50, "l_forearm"), (15, "r_calf"), (55, "head")} <= {
        (int(f), SK.joint_names[j]) for f, j in zip(*np.nonzero(spikes), strict=True)
    }
    assert not spikes[28:40].any()  # the strike itself is never a pop
    before = position_spikes(_pos(popped), POP_THRESHOLD).any(axis=1).sum()
    out, info = smooth_motion(body({"A": popped}))
    after = summarise(per_frame_metrics(out))["A"]["pop_frames"]
    print(f"pop frames {before} -> {after}; strikes {[(s.effector, s.frame) for s in info['A'].strikes]}")
    assert before >= 3 and after == 0
    assert (
        _speed(_pos(out.actors["A"])[:, SK.index("r_hand")])[25:45].max()
        >= 0.8 * _speed(_pos(popped)[:, SK.index("r_hand")])[25:45].max()
    )


def test_strike_detector_rejects_out_and_back():
    p = np.zeros((40, 3))
    p[20, 0] = 0.3  # a 30 cm one-frame pop: 9 m/s "speed" but no net displacement
    assert detect_strikes({"r_hand": p}, 30.0) == []
    t = np.arange(40)
    p2 = np.zeros((40, 3))
    p2[:, 2] = 0.6 / (1 + np.exp(-(t - 20) / 1.2))  # a real 0.6 m, fast extension
    s = detect_strikes({"r_hand": p2}, 30.0)
    assert len(s) == 1 and s[0].frame == 20


def test_walk_smoothing_holds_contacts_and_removes_jitter():
    rng = np.random.default_rng(2)
    clean, masks = walk(seconds=4.0)
    bad = corrupt(clean, rng, floor_shift=0.2, jitter_deg=1.5)
    bad.contact = model_probs(masks, rng)
    cont, rep = clean_contacts(body({"A": bad}, GEMX_CHANNELS))
    out, _ = smooth_motion(cont)
    s = summarise(per_frame_metrics(out, floor=rep.floor))["A"]
    assert s["mean_contact_foot_speed_cm_s"] <= 3.0
    assert s["max_penetration_cm"] <= 0.5
    assert s["pop_frames"] == 0
    p_c, p_s, p_t = _pos(cont.actors["A"]), _pos(out.actors["A"]), _pos(clean)
    jit = lambda p: np.sqrt(np.mean(np.sum(np.diff(p, 2, axis=0) ** 2, -1)))  # noqa: E731
    print(
        f"2nd-diff RMS: clean {jit(p_t) * 1000:.1f} mm, contacts {jit(p_c) * 1000:.1f}, "
        f"smooth {jit(p_s) * 1000:.1f}"
    )
    assert jit(p_s) < 0.25 * jit(p_c)
    # Quaternion continuity: consecutive frames in the same hemisphere.
    q = out.actors["A"].rot.astype(np.float64)
    assert np.all(np.sum(q[1:] * q[:-1], axis=-1) > 0)


def test_strength_zero_is_identity_and_regions_scale():
    rng = np.random.default_rng(3)
    noisy = _noisy(punch(), rng)
    bm = body({"A": noisy})
    out, _ = smooth_motion(bm, SmoothParams(strength=0.0, hold_contacts=False))
    assert np.allclose(out.actors["A"].rot, noisy.rot, atol=1e-5)
    assert np.allclose(out.actors["A"].root, noisy.root, atol=1e-5)
    out, _ = smooth_motion(bm, SmoothParams(region_strength={"legs": 0.0}, hold_contacts=False))
    leg = SK.index("l_calf")
    arm = SK.index("l_forearm")
    assert np.allclose(out.actors["A"].rot[:, leg], noisy.rot[:, leg], atol=1e-5)
    assert not np.allclose(out.actors["A"].rot[:, arm], noisy.rot[:, arm], atol=1e-3)


def test_one_euro_fallback():
    rng = np.random.default_rng(4)
    clean = punch()
    noisy = _noisy(clean, rng)
    out, _ = smooth_motion(body({"A": noisy}), SmoothParams(method="one_euro"))
    quiet = np.r_[0:20, 50:60]
    pc = _pos(clean)
    e_in = np.sqrt(np.mean(np.sum((_pos(noisy) - pc)[quiet] ** 2, -1)))
    e_out = np.sqrt(np.mean(np.sum((_pos(out.actors["A"]) - pc)[quiet] ** 2, -1)))
    assert e_out < e_in


def test_punch_flurry_every_punch_kept():
    """Three straights 7 frames apart: the speed stays high between them, every peak must survive."""
    rng = np.random.default_rng(5)
    parts = [punch(seconds=0.25, t_punch=0.02, ext_frames=3, retract_frames=3) for _ in range(4)]
    clean = ActorMotion(
        rot=np.concatenate([p.rot for p in parts]), root=np.concatenate([p.root for p in parts])
    )
    noisy = _noisy(clean, rng, deg=0.7, root=0.002)
    out, _ = smooth_motion(body({"A": noisy}), SmoothParams(hold_contacts=False))
    hand, pel = SK.index("r_hand"), SK.index("pelvis")
    v0 = _speed(_pos(noisy)[:, hand] - _pos(noisy)[:, pel])
    v1 = _speed(_pos(out.actors["A"])[:, hand] - _pos(out.actors["A"])[:, pel])
    peaks = [t for t in range(4, len(v0) - 4) if v0[t] > 3.0 and v0[t] == v0[t - 2 : t + 3].max()]
    assert len(peaks) >= 3
    assert min(v1[t - 2 : t + 3].max() / v0[t] for t in peaks) >= 0.8


def test_fast_spin_tangent_unwrap():
    """A spin at 30 deg/frame turns 450 deg over a 15-frame window; the log map must not wrap."""
    from afv.motion.ik import quat_axis_angle
    from afv.motion.rotations import qconj
    from afv.motion.smooth import smooth_rotations

    n = 60
    q = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1, 1))
    q[:, 0] = quat_axis_angle(np.tile([0.0, 1.0, 0.0], (n, 1)), np.radians(30.0) * np.arange(n))
    out = smooth_rotations(q, np.ones((n, 1)), np.full((n, 1), 0.2), np.full((n, 1), 10.0), 7)
    err = np.degrees(np.linalg.norm(qlog(qmul(qconj(q[:, 0]), out[:, 0])), axis=-1))
    assert err[8:-8].max() < 1.0


def test_smooth_is_idempotent_with_hold():
    rng = np.random.default_rng(8)
    clean, masks = walk(seconds=3.0)
    bad = corrupt(clean, rng, floor_shift=0.1)
    bad.contact = model_probs(masks, rng)
    cont, _ = clean_contacts(body({"A": bad}, GEMX_CHANNELS))
    again, _ = smooth_motion(cont, SmoothParams(strength=0.0))  # hold_contacts on
    assert np.linalg.norm(_pos(again.actors["A"]) - _pos(cont.actors["A"]), axis=-1).max() < 1e-3


def test_clip_ends_are_held_to_the_input():
    """Review: a smoothing window at a clip end has neighbours on one side only, so the end frames were
    pulled along the trend (feet moved 6-9 cm on frame 0). They are held to the input."""
    from afv.motion.fk import forward_kinematics

    a, _ = walk(seconds=1.5)
    bm = body({"A": a})
    out, _ = smooth_motion(bm)
    p0 = forward_kinematics(SK, a.rot, a.root)[0]
    p1 = forward_kinematics(SK, out.actors["A"].rot, out.actors["A"].root)[0]
    for f in (0, -1):
        assert np.linalg.norm(p1[f] - p0[f], axis=-1).max() <= 0.002


def _roughness(q: np.ndarray) -> float:
    """Mean squared angular acceleration of rotations [F,K,4] (rad/frame^2)."""
    r = qlog(qmul(np.asarray(q[:-1], np.float64) * np.array([-1, -1, -1, 1.0]), q[1:]))
    return float(np.mean(np.sum(np.diff(r, axis=0) ** 2, axis=-1)))


@pytest.mark.parametrize("region", ["torso", "head", "arms", "hands", "legs", "root"])
def test_every_region_slider_works_and_is_monotone(region):
    """Each spec slider (smooth.region_strength.<region>) smooths only its region, more with a higher
    value (0 = that region passes through), on the real soma77 skeleton."""
    from real_metrics import load_clip

    from afv.core.skeletons import get_skeleton
    from afv.motion.ik import get_roles
    from afv.motion.ranges import slice_motion

    bm = slice_motion(load_clip("ufc_cross")[0], 0, 90)
    sk = get_skeleton(bm.skeleton)
    regions = np.array(get_roles(sk).regions(sk))
    a0 = bm.actors["A"]
    rough, outs = [], []
    for s in (0.0, 0.5, 1.0, 2.0, 4.0):
        p = SmoothParams(region_strength={region: s}, hold_contacts=False, use_confidence=False)
        out = smooth_motion(bm, p)[0].actors["A"]
        outs.append(out)
        if region == "root":
            rough.append(float(np.mean(np.sum(np.diff(out.root.astype(np.float64), 2, axis=0) ** 2, -1))))
        else:
            rough.append(_roughness(out.rot[:, regions == region]))
    assert all(x > y for x, y in itertools.pairwise(rough)), (region, rough)
    # strength 0 passes the region through
    if region == "root":
        assert np.allclose(outs[0].root, a0.root, atol=1e-5)
    else:
        sel = regions == region
        assert np.allclose(np.abs(np.sum(outs[0].rot[:, sel] * a0.rot[:, sel], -1)), 1.0, atol=1e-5)
    # ... and the other regions do not depend on it
    other = regions != region
    assert np.array_equal(outs[0].rot[:, other], outs[-1].rot[:, other])
    if region != "root":
        assert np.array_equal(outs[0].root, outs[-1].root)


def test_unknown_region_rejected():
    with pytest.raises(ValueError, match="unknown smoothing region"):
        SmoothParams(region_strength={"arm": 2.0})
    with pytest.raises(ValueError, match=">= 0"):
        SmoothParams(region_strength={"arms": -1.0})
