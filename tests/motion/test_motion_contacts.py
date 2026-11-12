"""Contacts: floor fit, detection, skate removal and penetration against the NFR targets."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from motion_synth import FOOT_CHANNELS, GEMX_CHANNELS, ROLES, SK, body, corrupt, model_probs, walk

from afv.core.types import ActorMotion
from afv.motion.contacts import (
    HOVER_TOL,
    ContactParams,
    Floor,
    FootFix,
    LegPins,
    _fill_held_holes,
    _knee_angle,
    _knee_space_blend,
    _LegPlan,
    _pivot_ankle,
    _reachable_pin,
    _smooth_leg_correction,
    _smooth_lift,
    _trim_sliding_edges,
    apply_foot_targets,
    channel_offsets,
    clean_contacts,
    cover_smooth,
    fit_floor,
    foot_sole_heights,
    hysteresis_mask,
    painted_spans,
    segments,
    solve_root_correction,
    split_pivot_toe_plants,
)
from afv.motion.fk import forward_kinematics
from afv.motion.ik import LegTarget, quat_axis_angle
from afv.motion.rotations import qmul
from afv.motion.smooth import position_spikes
from afv.qc.labels import Labels, PlantLabel, contact_recall, ground_penetration, slide_within_plants
from afv.qc.metrics import POP_THRESHOLD, per_frame_metrics, summarise

FLOOR_Y = 0.2  # the corrupted world's floor is not at 0: the fit has to find it


def _labels(masks: dict[str, np.ndarray], role: str = "A") -> Labels:
    return Labels(
        [PlantLabel(role, s, e, channels=[ch]) for ch in FOOT_CHANNELS for s, e in segments(masks[ch])]
    )


def _scene(seed: int = 0, with_model: bool = True, **kw):
    rng = np.random.default_rng(seed)
    clean, masks = walk(seconds=4.0)
    bad = corrupt(clean, rng, floor_shift=FLOOR_Y, **kw)
    if with_model:
        bad.contact = model_probs(masks, rng)
    return body({"A": bad}, GEMX_CHANNELS if with_model else []), masks


def test_fit_floor_robust_to_outliers():
    rng = np.random.default_rng(0)
    pts = np.column_stack(
        [rng.uniform(-2, 2, 400), 0.31 + rng.normal(0, 0.005, 400), rng.uniform(-2, 2, 400)]
    )
    pts[:60, 1] += rng.uniform(0.1, 0.5, 60)  # feet in the air
    pts[60:70, 1] -= 0.3  # wild penetrations
    assert fit_floor(pts).height(np.array([0.0, 0.31, 0.0])) == pytest.approx(0.0, abs=0.003)
    tilted = pts.copy()
    tilted[:, 1] += 0.05 * tilted[:, 0]  # ~2.9 degrees
    f = fit_floor(tilted, tilt=True)
    assert np.degrees(np.arccos(f.normal[1])) == pytest.approx(np.degrees(np.arctan(0.05)), abs=0.3)


def test_hysteresis_fills_gaps_and_drops_blips():
    p = ContactParams()
    prob = np.zeros(40)
    prob[5:20] = 0.9
    prob[11] = 0.45  # dip between thresholds: stays on
    prob[14:16] = 0.1  # 2-frame gap: filled
    prob[30:32] = 0.95  # 2-frame blip: dropped
    m = hysteresis_mask(prob, p)
    assert segments(m) == [(5, 20)]


def test_cover_smooth_never_undercuts():
    rng = np.random.default_rng(1)
    x = np.maximum(0, rng.normal(0, 1, 200)) * (rng.random(200) < 0.1)
    y = cover_smooth(x, 3)
    assert np.all(y >= x - 1e-12)
    assert np.abs(np.diff(y)).max() < np.abs(np.diff(x)).max()


@pytest.mark.parametrize("with_model", [True, False])
def test_walk_skate_removed_to_spec(with_model):
    bm, masks = _scene(with_model=with_model)
    out, rep = clean_contacts(bm)
    # Floor found (0.2 m) and moved to y = 0 by translating the shot (project convention).
    assert rep.floor_shift == pytest.approx(-FLOOR_Y, abs=0.01)
    assert rep.floor.offset == pytest.approx(0.0, abs=1e-9) and rep.floor.normal[1] == pytest.approx(1.0)
    assert out.world_shift == pytest.approx([0.0, rep.floor_shift, 0.0])
    before = summarise(
        per_frame_metrics(
            type(bm)(
                frames=bm.frames,
                timestamps=bm.timestamps,
                skeleton=bm.skeleton,
                fps=bm.fps,
                contact_channels=out.contact_channels,
                actors={
                    "A": type(bm.actors["A"])(
                        bm.actors["A"].rot, bm.actors["A"].root, contact=out.actors["A"].contact
                    )
                },
            ),
            floor=Floor.horizontal(-rep.floor_shift),
        )
    )["A"]
    after = summarise(per_frame_metrics(out, floor=rep.floor))["A"]
    labels = _labels(masks)
    gt = Floor.horizontal(FLOOR_Y + rep.floor_shift)  # the true floor, after the shift
    recall = contact_recall(out, labels)
    slide = slide_within_plants(out, labels, gt).max()
    pen = ground_penetration(out, labels, gt)
    print(
        f"model={with_model}: contact foot speed {before['mean_contact_foot_speed_cm_s']:.1f} -> "
        f"{after['mean_contact_foot_speed_cm_s']:.3f} cm/s, recall {recall:.2f}, slide {slide * 100:.2f} cm, "
        f"penetration {pen * 100:.3f} cm, max penetration {after['max_penetration_cm']:.3f} cm"
    )
    assert before["mean_contact_foot_speed_cm_s"] > 10.0  # the corruption is GEM-X-sized skate
    assert after["mean_contact_foot_speed_cm_s"] <= 3.0  # NFR-3
    assert recall >= 0.9  # metrics table
    assert slide <= 0.02  # slide within a plant <= 2 cm
    assert pen <= 0.015  # ground penetration <= 1.5 cm
    assert after["max_penetration_cm"] <= 0.5  # no penetration anywhere, not just in plants
    # Output channels: the four foot channels (binary) then GEM-X wrists passed through.
    assert out.contact_channels[:4] == FOOT_CHANNELS
    c = out.actors["A"].contact
    assert set(np.unique(c[:, :4])) <= {0.0, 1.0}
    if with_model:
        assert out.contact_channels[4:] == ["left_wrist", "right_wrist"]
        assert np.allclose(c[:, 4:], bm.actors["A"].contact[:, 4:])


@pytest.mark.parametrize("with_model", [True, False])
def test_no_leg_pops_at_plant_boundaries(with_model):
    bm, _ = _scene(with_model=with_model, jitter_deg=0.0, root_noise=0.0)
    out, _ = clean_contacts(bm)
    a = out.actors["A"]
    pos = forward_kinematics(SK, a.rot, a.root)[0]
    pin = forward_kinematics(SK, bm.actors["A"].rot, bm.actors["A"].root)[0]
    legs = [SK.index(n) for leg in ROLES.legs for n in (leg.calf, leg.foot, leg.toe)]
    # No single-frame pops on the legs, and accelerations of the same order as the input's (a
    # foot released after being held against 14 cm of root drift accelerates a little harder).
    assert not position_spikes(pos[:, legs], POP_THRESHOLD).any()
    acc_out = np.linalg.norm(np.diff(pos[:, legs], 2, axis=0), axis=-1).max()
    acc_in = np.linalg.norm(np.diff(pin[:, legs], 2, axis=0), axis=-1).max()
    print(f"model={with_model}: max leg acceleration {acc_in * 100:.1f} -> {acc_out * 100:.1f} cm/frame^2")
    assert acc_out <= 2.5 * acc_in


def test_floating_body_absorbed_by_root_correction():
    rng = np.random.default_rng(5)
    clean, masks = walk(seconds=3.0)
    bad = corrupt(clean, rng, skate=0.0, jitter_deg=0.0, root_noise=0.0, penetration=0.0)
    n = bad.root.shape[0]
    bad.root[:, 1] += 0.08 * np.sin(np.pi * np.clip((np.arange(n) - 30) / 30, 0, 1))  # body floats up
    bad.contact = model_probs(masks, rng, flicker=0.0)
    out, rep = clean_contacts(body({"A": bad}, GEMX_CHANNELS))
    drop = rep.actors["A"].fix.pelvis_drop
    assert np.abs(drop).max() <= ContactParams().max_pelvis_drop + 1e-9  # residual only, capped
    assert np.abs(np.diff(drop, 2)).max() < 0.005
    a = out.actors["A"]
    pos = forward_kinematics(SK, a.rot, a.root)[0]
    for c in ROLES.contacts:
        m = rep.actors["A"].masks[c.channel]
        assert m.sum() > 0.8 * masks[c.channel].sum()
        both = m[1:] & m[:-1]
        sp = np.linalg.norm(np.diff(pos[:, SK.index(c.joint)], axis=0), axis=-1)[both] * 30
        assert sp.mean() <= 0.03
        assert rep.actors["A"].fix.miss[c.channel].max() <= ContactParams().max_pin_miss + 1e-6


def test_swing_leg_follows_pelvis_under_drop():
    """With the pelvis lowered, an unplanted leg keeps its pose relative to the pelvis instead of
    reaching for its old world position (which snapped knees on real data)."""
    rng = np.random.default_rng(6)
    clean, masks = walk(seconds=3.0)
    bad = corrupt(clean, rng, skate=0.12, jitter_deg=0.0, root_noise=0.0, penetration=0.0)
    bad.contact = model_probs(masks, rng, flicker=0.0)
    params = ContactParams(root_correction=False, max_pelvis_drop=0.1, ik_smooth=0.0)
    out, rep = clean_contacts(body({"A": bad}, GEMX_CHANNELS), params)
    fix = rep.actors["A"].fix
    assert fix.pelvis_drop.max() > 0.008  # round 4: edge-pop/step rules trim a few plant frames
    pos_in = forward_kinematics(SK, bad.rot, bad.root)[0]
    clear = ROLES.clearances(SK)
    for li, leg in enumerate(ROLES.legs):
        w = np.maximum.reduce([fix.weights[c.channel] for c in ROLES.contacts if c.leg == li])
        # Frames this leg swings while the pelvis is lowered (only planted frames ask for a drop, so on
        # this gait the left leg swings under a drop of only a few millimetres), clear of the floor clamp.
        high = np.min(
            [pos_in[:, SK.index(c.joint), 1] - clear[c.channel] for c in ROLES.contacts if c.leg == li],
            axis=0,
        )
        free = (w == 0) & (fix.pelvis_drop > 0.001) & (high > fix.pelvis_drop + params.floor_soft_width)
        assert free.any()
        for j in (leg.thigh, leg.calf):
            q0, q1 = bad.rot[free, SK.index(j)], out.actors["A"].rot[free, SK.index(j)]
            assert np.allclose(np.abs(np.sum(q0 * q1, axis=-1)), 1.0, atol=1e-5)


def test_painted_contacts_override():
    bm, masks = _scene()
    s0, e0 = segments(masks["l_heel"])[1]
    # Free the second left plant; plant a span of the right foot that is really swing.
    rs, _ = segments(~masks["r_heel"])[1]
    edits = {
        "spans": [
            {"role": "A", "foot": "left", "start": s0, "end": e0, "state": "free"},
            {"role": "A", "channels": ["r_heel"], "start": rs + 1, "end": rs + 3, "state": "plant"},
        ]
    }
    spans = painted_spans(edits, "A", ROLES, bm.frames)
    assert spans["l_heel"] == [(s0, e0, False)] and spans["l_toe"] == [(s0, e0, False)]
    _, rep = clean_contacts(bm, painted=edits)
    m = rep.actors["A"].masks
    assert not m["l_heel"][s0:e0].any() and not m["l_toe"][s0:e0].any()
    assert m["r_heel"][rs + 1 : rs + 3].all()  # kept although shorter than min_plant_frames
    with pytest.raises(ValueError):
        painted_spans(
            {"spans": [{"role": "A", "foot": "l", "start": 0, "end": 2, "state": "maybe"}]},
            "A",
            ROLES,
            bm.frames,
        )
    with pytest.raises(ValueError):  # a side that resolves to no channel is an error, not a no-op
        painted_spans({"A": [{"foot": "middle", "start": 0, "end": 2}]}, "A", ROLES, bm.frames)


def test_two_fighters_share_one_floor():
    rng = np.random.default_rng(7)
    a, ma = walk(seconds=3.0, origin=(-1.0, 0.0))
    b, mb = walk(seconds=3.0, origin=(1.0, 0.5), heading=np.pi)
    a = corrupt(a, rng, floor_shift=FLOOR_Y)
    b = corrupt(b, rng, floor_shift=FLOOR_Y)
    a.contact, b.contact = model_probs(ma, rng), model_probs(mb, rng)
    out, rep = clean_contacts(body({"A": a, "B": b}, GEMX_CHANNELS))
    assert rep.floor_shift == pytest.approx(-FLOOR_Y, abs=0.01)
    for r, m in (("A", ma), ("B", mb)):
        pos = forward_kinematics(SK, out.actors[r].rot, out.actors[r].root)[0]
        sole = pos[m["l_toe"], SK.index("l_toe"), 1] - ROLES.clearances(SK)["l_toe"]
        assert np.abs(sole).max() < 0.005  # planted soles on y = 0
    for r, m in (("A", ma), ("B", mb)):
        s = summarise(per_frame_metrics(out, floor=rep.floor))[r]
        assert s["mean_contact_foot_speed_cm_s"] <= 3.0
        assert contact_recall(out, _labels(m, r)) >= 0.9


def test_realistic_skate_and_rigid_foot_pins():
    """~0.6 m/s root drift plus non-rigid heel/toe wobble: plants still, foot orientation fixed."""
    rng = np.random.default_rng(11)
    clean, masks = walk(seconds=4.0)
    bad = corrupt(clean, rng, skate=0.12, foot_wobble_deg=12.0, floor_shift=FLOOR_Y)
    bad.contact = model_probs(masks, rng)
    bm = body({"A": bad}, GEMX_CHANNELS)
    before = np.concatenate(
        [
            np.linalg.norm(
                np.diff(forward_kinematics(SK, bad.rot, bad.root)[0][:, SK.index(c.joint)], axis=0), axis=-1
            )[masks[c.channel][1:] & masks[c.channel][:-1]]
            * 30
            for c in ROLES.contacts
        ]
    )
    assert before.mean() >= 0.5  # >= 50 cm/s of skate on the true plants
    out, rep = clean_contacts(bm)
    s = summarise(per_frame_metrics(out, floor=rep.floor))["A"]
    assert s["mean_contact_foot_speed_cm_s"] <= 3.0
    assert contact_recall(out, _labels(masks)) >= 0.9
    _, wrot = forward_kinematics(SK, out.actors["A"].rot, out.actors["A"].root)
    for leg, side in zip(ROLES.legs, "lr", strict=True):
        flat = rep.actors["A"].masks[f"{side}_heel"] & rep.actors["A"].masks[f"{side}_toe"]
        for s0, e0 in segments(flat):
            q = wrot[s0:e0, SK.index(leg.foot)]
            ang = 2 * np.degrees(np.arccos(np.clip(np.abs(np.sum(q * q[:1], axis=-1)), 0, 1)))
            assert ang.max() <= 1.0  # one rigid foot transform per plant: no yaw/pitch drift


def test_swing_leg_correction_smoothed_while_other_foot_planted():
    """Round 2 fix: the IK-correction smoothing holds (and re-solves exactly) each leg only where its
    *own* foot is planted. With the union of both legs, a swinging leg was re-solved exactly on every
    frame the other foot was down, so a jump in its correction stayed a knee snap."""
    n = 40
    rot = np.tile([0.0, 0.0, 0.0, 1.0], (n, SK.num_joints, 1))
    for side in "lr":
        rot[:, SK.index(f"{side}_thigh")] = quat_axis_angle(np.tile([1.0, 0, 0], (n, 1)), np.full(n, -0.5))
        rot[:, SK.index(f"{side}_calf")] = quat_axis_angle(np.tile([1.0, 0, 0], (n, 1)), np.full(n, 0.8))
    root = np.tile([0.0, 0.9, 0.0], (n, 1))
    new = rot.copy()
    calf = SK.index("r_calf")
    new[20:, calf] = quat_axis_angle(np.tile([1.0, 0, 0], (n - 20, 1)), np.full(n - 20, 0.8 + 0.5))
    pos_new = forward_kinematics(SK, new, root)[0]
    pos_old = forward_kinematics(SK, rot, root)[0]
    l_leg, r_leg = ROLES.legs
    targets = [
        LegTarget(l_leg, pos_old[:, SK.index(l_leg.foot)], np.ones(n, bool)),
        LegTarget(r_leg, pos_new[:, SK.index(r_leg.foot)], np.ones(n, bool)),
    ]
    ones, zeros = np.ones(n), np.zeros(n)
    plans = [
        _LegPlan(l_leg, None, None, None, None, ones, ones, ones),
        _LegPlan(r_leg, None, None, None, None, zeros, zeros, zeros),
    ]
    out = _smooth_leg_correction(
        SK, ROLES, rot, new, root, targets, plans, ContactParams(), np.array(ROLES.forward)
    )
    pos = forward_kinematics(SK, out, root)[0]

    def knee(p: np.ndarray, leg) -> np.ndarray:
        t = p[:, SK.index(leg.calf)] - p[:, SK.index(leg.thigh)]
        c = p[:, SK.index(leg.foot)] - p[:, SK.index(leg.calf)]
        return np.arccos(
            np.clip(np.sum(t * c, -1) / np.linalg.norm(t, axis=-1) / np.linalg.norm(c, axis=-1), -1, 1)
        )

    step = np.abs(np.diff(knee(pos_new, r_leg))).max()
    assert np.abs(np.diff(knee(pos, r_leg))).max() < 0.5 * step  # the swing leg's jump is spread
    # The planted leg is held exactly.
    assert np.allclose(pos[:, SK.index(l_leg.foot)], pos_old[:, SK.index(l_leg.foot)], atol=1e-6)


def _needed_raise(pos, leg, pin, floor, reach, drop):
    """Per-frame raise a heel pin needs to be reached (independent of the implementation)."""
    hip = pos[:, SK.index(leg.thigh)]
    knee, ank = pos[:, SK.index(leg.calf)], pos[:, SK.index(leg.foot)]
    length = np.linalg.norm(knee - hip, axis=1) + np.linalg.norm(ank - knee, axis=1)
    r = np.maximum(reach * length, np.minimum(np.linalg.norm(hip - ank, axis=1), 0.995 * length))
    rel = hip - drop * floor.normal - pin
    up = rel @ floor.normal
    horiz = np.linalg.norm(rel - up[:, None] * floor.normal, axis=1)
    return np.maximum(up - np.sqrt(r**2 - horiz**2), 0.0)


def test_reachable_pin_raised_exactly_as_far_as_needed():
    n = 11
    rot = np.tile([0.0, 0.0, 0.0, 1.0], (n, SK.num_joints, 1))
    rot[:, SK.index("l_calf")] = quat_axis_angle(np.tile([1.0, 0, 0], (n, 1)), np.full(n, 0.2))
    root = np.stack([np.zeros(n), np.linspace(0.93, 0.97, n), np.zeros(n)], axis=1)  # frames differ
    pos = forward_kinematics(SK, rot, root)[0]
    leg = ROLES.legs[0]
    ank = pos[:, SK.index(leg.foot)]
    med = np.median(ank, axis=0)
    floor = Floor.horizontal(float(med[1]) - 0.08)  # the input foot hovers ~8 cm above the floor
    pin = floor.project(med, 0.0)
    top = 1.0  # no cap
    raises = {}
    for drop, q in ((0.0, 0.9), (0.02, 0.9), (0.0, 0.5), (0.0, 1.0), (0.0, 0.0)):
        need = _needed_raise(pos, leg, pin, floor, 0.96, drop)
        assert np.ptp(need) > 0.01  # a mixed-frame case: the quantile matters
        out, up_by, capped = _reachable_pin(pin, top, floor, pos, SK, leg, 0.96, drop, q)
        assert up_by == pytest.approx(float(np.quantile(need, q)), abs=1e-9)
        assert np.allclose(out, pin + up_by * floor.normal) and not capped
        raises[(drop, q)] = up_by
    assert 0.0 < raises[(0.02, 0.9)] < raises[(0.0, 0.9)]  # a pelvis drop allowance needs less raise
    assert raises[(0.0, 0.0)] < raises[(0.0, 0.5)] < raises[(0.0, 0.9)] < raises[(0.0, 1.0)]
    # Capped: raised exactly to the cap and reported, so the frames it cannot reach get unplanted.
    out, up_by, capped = _reachable_pin(pin, floor.height(pin) + 0.01, floor, pos, SK, leg, 0.96, 0.0, 0.9)
    assert up_by == pytest.approx(0.01) and capped
    # A pin at the input's own ankle needs no raise (beyond the 0.995 L singularity margin).
    out, up_by, capped = _reachable_pin(ank[0], top, floor, pos[:1], SK, leg, 0.96, 0.0, 0.9)
    assert up_by < 1e-4 and np.allclose(out, ank[0], atol=1e-4) and not capped


def _stance(hover: float, n: int = 40, knee: float = 0.0) -> ActorMotion:
    """A still standing fighter (legs straight, or bent by ``knee`` rad) whose soles sit ``hover``
    above y = 0, with GEM-X-like contact probabilities saying both feet are down."""
    rot = np.tile([0.0, 0.0, 0.0, 1.0], (n, SK.num_joints, 1))
    for side in "lr":
        rot[:, SK.index(f"{side}_thigh")] = quat_axis_angle(
            np.tile([1.0, 0, 0], (n, 1)), np.full(n, -knee / 2)
        )
        rot[:, SK.index(f"{side}_calf")] = quat_axis_angle(np.tile([1.0, 0, 0], (n, 1)), np.full(n, knee))
    root = np.tile([0.0, 1.0, 0.0], (n, 1))
    pos = forward_kinematics(SK, rot, root)[0]
    clear = ROLES.clearances(SK)
    sole = min(float(pos[0, SK.index(c.joint), 1]) - clear[c.channel] for c in ROLES.contacts)
    root[:, 1] += hover - sole
    contact = np.tile(np.array([0.95, 0.95, 0.95, 0.95, 0.02, 0.02], np.float32), (n, 1))
    return ActorMotion(rot=rot, root=root, contact=contact)


@pytest.mark.parametrize(("hover", "reach_drop"), [(0.028, 0.0), (0.08, 0.0125)])
def test_hovering_straight_legs_are_unplanted_not_pinned_in_the_air(hover, reach_drop):
    """A planted contact must not float (a game's foot IK keyed on contact flags would snap it down).
    Straight legs whose soles hover above the floor: a pin is raised to stay reachable only up to
    ``max_pin_raise`` above the clearance; where the leg cannot reach that, the frames are unplanted
    (counted in the report) instead of pinning the foot in the air."""
    p = ContactParams(pin_reach_drop=reach_drop)
    bm = body({"A": _stance(hover)}, GEMX_CHANNELS)
    out, rep = clean_contacts(bm, p, floor=Floor.horizontal(0.0))
    info = rep.actors["A"]
    s = summarise(per_frame_metrics(out))["A"]
    assert s["max_hover_in_contact_cm"] <= 100 * (p.max_pin_raise + p.max_pin_miss) + 1e-6
    if hover <= p.max_pin_raise:  # reachable by a raise within the cap: kept, still, raised
        assert info.raised_plants > 0 and info.capped_plants == 0
        assert s["contact_frames"] > 0 and s["mean_contact_foot_speed_cm_s"] <= 1.0
    else:  # would need a floating pin: capped, and the unreachable plants are dropped
        assert info.capped_plants > 0
        assert info.dropped_plants > 0 and info.miss_unplanted_frames > 0
        assert s["contact_frames"] == 0
    for c in ROLES.contacts:  # heel and toe pins both bounded by the cap
        pos = forward_kinematics(SK, out.actors["A"].rot, out.actors["A"].root)[0]
        h = pos[info.masks[c.channel], SK.index(c.joint), 1] - ROLES.clearances(SK)[c.channel]
        assert h.size == 0 or h.max() <= p.max_pin_raise + p.max_pin_miss + 1e-6


def test_qc_floor_is_y0_and_hover_reported():
    """QC measures against the contacts floor (y = 0), not a floor re-fitted from the planted soles:
    a plant left 2 cm above the floor shows as 2 cm hover, and the other plants show no penetration."""
    a = _stance(0.0, n=10)
    b = _stance(0.02, n=10)
    chans = [c.channel for c in ROLES.contacts]
    for x in (a, b):
        x.contact = np.ones((10, len(chans)), np.float32)
    bm = body({"A": a, "B": b}, chans)
    s = summarise(per_frame_metrics(bm))
    assert s["A"]["max_penetration_cm"] == pytest.approx(0.0, abs=1e-6)
    assert s["A"]["mean_hover_in_contact_cm"] == pytest.approx(0.0, abs=1e-6)
    assert s["B"]["mean_hover_in_contact_cm"] == pytest.approx(2.0, abs=0.3)  # toe vs heel clearance
    assert s["B"]["max_hover_in_contact_cm"] >= 2.0 - 1e-6


def test_pin_miss_loop_counts_and_painted_plants(monkeypatch):
    """The pin-miss loop judges only non-painted frames, re-solves at most three times and unplants
    frames still missing after the last solve without another solve (counted); painted frames stay."""
    import afv.motion.contacts as C

    real = C.apply_foot_targets
    calls = []

    def always_miss(sk, roles, rot, root, masks, floor, params, **kw):
        fix = real(sk, roles, rot, root, masks, floor, params, **kw)
        calls.append(1)
        for ch, m in masks.items():
            d = fix.miss.setdefault(ch, np.zeros(len(m)))
            d[m] = 0.05  # every planted frame "misses" by 5 cm, every solve
        return fix

    monkeypatch.setattr(C, "apply_foot_targets", always_miss)
    bm = body({"A": _stance(0.0, knee=0.4)}, GEMX_CHANNELS)
    painted = {"spans": [{"role": "A", "start": 5, "end": 15, "foot": "left", "state": "plant"}]}
    _, rep = clean_contacts(bm, ContactParams(), floor=Floor.horizontal(0.0), painted=painted)
    info = rep.actors["A"]
    # First solve, one re-solve without the missing unpainted frames, then the loop stops: painted
    # frames that "miss" do not count (judging them too spun the loop for all three re-solves).
    assert len(calls) == 2
    for ch in ("l_heel", "l_toe"):  # painted: kept although they "miss"
        assert info.masks[ch][5:15].all() and not info.masks[ch][15:].any()
    assert not info.masks["r_heel"].any() and not info.masks["r_toe"].any()
    assert info.miss_unplanted_frames > 0
    # Only painted frames were left after the first round, so nothing is left to unplant at the end.
    assert info.unsolved_unplanted_frames == 0
    calls.clear()
    # Without paint: the first round unplants everything and the loop exits at once (no frame left).
    _, rep = clean_contacts(bm, ContactParams(), floor=Floor.horizontal(0.0))
    assert len(calls) == 2 and not any(m.any() for m in rep.actors["A"].masks.values())


def test_pin_miss_frames_left_after_the_last_solve_are_unplanted(monkeypatch):
    import afv.motion.contacts as C

    real = C.apply_foot_targets
    n_call = [0]

    def late_miss(sk, roles, rot, root, masks, floor, params, **kw):
        fix = real(sk, roles, rot, root, masks, floor, params, **kw)
        n_call[0] += 1
        for ch, m in masks.items():  # each solve misses on its first 3 planted frames of r_heel
            d = fix.miss.setdefault(ch, np.zeros(len(m)))
            d[:] = 0.0
            if ch == "r_heel":
                d[np.flatnonzero(m)[:3]] = 0.05
        return fix

    monkeypatch.setattr(C, "apply_foot_targets", late_miss)
    bm = body({"A": _stance(0.0, knee=0.4)}, GEMX_CHANNELS)
    out, rep = clean_contacts(bm, ContactParams(), floor=Floor.horizontal(0.0))
    info = rep.actors["A"]
    assert n_call[0] == 4
    assert info.miss_unplanted_frames == 9 and info.unsolved_unplanted_frames == 3
    m = info.masks["r_heel"]
    assert not m[:12].any() and m[12:].all()  # 4 x 3 frames gone, the rest still planted
    col = out.contact_channels.index("r_heel")
    assert np.array_equal(out.actors["A"].contact[:, col] > 0.5, m)


def test_smooth_lift_never_undercuts_and_has_no_kink():
    """Round 2: a swinging foot pushed below the floor for two frames by the IK-correction smoothing is
    lifted by a smoothed, never-undercutting lift (an exact per-frame clamp put a velocity kink in the
    swing), and nothing leaks into the neighbouring plant."""
    need = np.zeros(30)
    need[14:16] = 0.01
    free = np.ones(30, dtype=bool)
    free[25:] = False  # a plant starts at frame 25
    need[24] = 0.004  # a small requirement right next to it
    lift = _smooth_lift(need, free)
    assert np.all(lift >= need - 1e-12) and not lift[25:].any()
    exact_kink = np.abs(np.diff(need[:24], 2)).max()
    assert np.abs(np.diff(lift[:24], 2)).max() < 0.25 * exact_kink


def test_pelvis_drop_follows_a_gait():
    """Round 2: with the default smoothing radius a drop needed once per walking step (every 15
    frames at 30 fps) comes back up between steps instead of becoming a constant lowering."""
    need = np.zeros(120)
    need[7::15] = 0.025
    drop = np.minimum(
        cover_smooth(need, ContactParams().pelvis_smooth_frames), ContactParams().max_pelvis_drop
    )
    assert np.all(drop >= need - 1e-12)
    assert drop[15:105].min() < 0.5 * 0.025


@pytest.mark.parametrize(
    "bad",
    [
        {"pin_reach_quantile": 1.5},
        {"pin_reach_quantile": -0.1},
        {"pin_reach_drop": -0.01},
        {"max_pin_raise": -0.01},
        {"pelvis_smooth_frames": -1},
    ],
)
def test_contact_params_are_bounded(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ContactParams(**bad)


def _leg_geometry(pos: np.ndarray, ankle: np.ndarray, leg) -> tuple[np.ndarray, np.ndarray]:
    hip, knee, ank = (pos[:, SK.index(j)] for j in (leg.thigh, leg.calf, leg.foot))
    l1, l2 = np.linalg.norm(knee - hip, axis=-1), np.linalg.norm(ank - knee, axis=-1)
    return _knee_angle(np.linalg.norm(ank - hip, axis=-1), l1, l2), _knee_angle(
        np.linalg.norm(ankle - hip, axis=-1), l1, l2
    )


def _leg_frames(theta_in: np.ndarray, theta_tg: np.ndarray, leg, l1: float = 0.42, l2: float = 0.4):
    """Hip/knee/ankle positions [F,J,3] of a two-bone leg bent by ``theta_in`` (rad) and ankle targets
    in the same direction at the distance of a ``theta_tg`` bend."""
    n = len(theta_in)
    pos = np.zeros((n, len(SK.joint_names), 3))
    hip = np.array([0.0, 1.0, 0.0])
    knee = hip + np.array([0.0, -l1, 0.0])
    ankle = knee + l2 * np.stack([np.sin(theta_in), -np.cos(theta_in), np.zeros(n)], axis=1)
    pos[:, SK.index(leg.thigh)] = hip
    pos[:, SK.index(leg.calf)] = knee
    pos[:, SK.index(leg.foot)] = ankle
    d_tg = np.sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * np.cos(theta_tg))
    rel = ankle - hip
    target = hip + rel / np.linalg.norm(rel, axis=-1, keepdims=True) * d_tg[:, None]
    return pos, target


def _blend_weights(blend: int = ContactParams().blend_frames) -> np.ndarray:
    x = (blend + 1 - np.arange(1, blend + 1)) / (blend + 1)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x * x)


def test_knee_space_blend_continues_the_edge_correction_with_its_rate():
    """Round 2: around a plant the knee gets the input's knee plus the plant's edge knee correction
    continued with its edge rate and eased (as the position offsets are), not the eased position
    offset: near a straight leg a few centimetres of eased offset in one frame bent the knee by
    ~25 deg (ref walk). Input knee and correction both move, so the rate term's sign matters."""
    leg = ROLES.legs[0]
    n, s, e = 24, 8, 15  # plant on frames 8..14
    t = np.arange(n)
    th_in = np.radians(20.0 + 2.0 * t)
    corr = np.radians(15.0 + 1.5 * (t - s))  # inside the plant; rate 1.5 deg/frame
    planted = (t >= s) & (t < e)
    pos, target = _leg_frames(th_in, np.where(planted, th_in + corr, th_in), leg)
    out = _knee_space_blend(SK, leg, pos, target, planted, ContactParams().blend_frames)
    _, th_out = _leg_geometry(pos, out, leg)
    w = _blend_weights()
    k = np.arange(1, len(w) + 1)
    want = th_in.copy()
    rate = np.radians(1.5)
    want[s - k] = th_in[s - k] + w * (corr[s] - k * rate)
    want[e - 1 + k] = th_in[e - 1 + k] + w * (corr[e - 1] + k * rate)
    want[planted] = th_in[planted] + corr[planted]
    assert np.allclose(th_out, want, atol=1e-6)


def test_knee_space_blend_between_two_close_plants():
    """Review: with a 5-frame gap the blend zones of two plants overlap; taking only the nearest
    plant's correction switched plants mid-gap (a 17 deg knee step). Both contribute,
    sum(w_k c_k) / max(1, sum(w_k)), so the correction moves smoothly from one plant's to the other's."""
    leg = ROLES.legs[0]
    n = 25
    t = np.arange(n)
    planted = (t < 10) | (t >= 15)  # gap = frames 10..14
    th_in = np.full(n, np.radians(10.0))
    c1 = np.radians(40.0)  # the first plant pinned higher: a much more bent knee; the second as input
    pos, target = _leg_frames(th_in, np.where(t < 10, th_in + c1, th_in), leg)
    out = _knee_space_blend(SK, leg, pos, target, planted, ContactParams().blend_frames)
    _, th_out = _leg_geometry(pos, out, leg)
    w = _blend_weights()
    w1 = np.zeros(n)
    w2 = np.zeros(n)
    w1[10:14] = w  # after plant 1
    w2[11:15] = w[::-1]  # before plant 2
    want = np.where(planted, th_in + np.where(t < 10, c1, 0.0), th_in + w1 * c1 / np.maximum(1.0, w1 + w2))
    assert np.allclose(th_out, want, atol=1e-6)
    assert np.degrees(np.abs(np.diff(th_out))).max() <= 0.4 * 40.0


@pytest.mark.parametrize("period", [0.4, 0.45, 0.5])
def test_knee_correction_smooth_through_quick_footwork(period):
    """Review: a quick gait (boxing footwork: 0.4-0.5 s steps) whose every other left stance is 5 cm
    low (pinned 5 cm up) has short swings with overlapping blend zones; the knee correction must ramp
    through each swing instead of jumping (nearest-plant: 0, 0, 0, 0, 0, 11.9, 14.6 deg)."""
    a, masks = walk(seconds=3.0, period=period, speed=0.6)
    pos = forward_kinematics(SK, a.rot, a.root)[0]
    leg = ROLES.legs[0]
    planted = masks["l_heel"] | masks["l_toe"]
    cur = pos[:, SK.index(leg.foot)]
    pins = []
    for i, (s, e) in enumerate(segments(planted)):
        pins.append(np.median(cur[s:e], axis=0) + (np.array([0.0, 0.05, 0.0]) if i % 2 else 0.0))
    blend = ContactParams().blend_frames
    off, _, _ = channel_offsets(cur, None, planted, Floor.horizontal(0.0), 0.0, blend, pins)
    target = cur + off
    out = _knee_space_blend(SK, leg, pos, target, planted, blend)
    th_in, th_out = _leg_geometry(pos, out, leg)
    _, th_tg = _leg_geometry(pos, target, leg)
    corr = np.degrees(th_out - th_in)
    big = np.degrees(np.abs(th_tg - th_in)[planted]).max()
    step = np.abs(np.diff(corr)).max()
    print(f"period {period}: max plant correction {big:.1f} deg, max correction change {step:.1f} deg/frame")
    assert big > 10.0
    assert step <= 7.0  # measured 5.5; the nearest-plant blend gave 9-12 deg/frame


def test_sliding_plant_edges_are_unflagged():
    """Round 2: frames at a plant's end where the leg stops reaching its pin (within max_pin_miss) and
    the foot slides are flagged unplanted (the motion is kept); interior and painted frames stay."""
    a, masks = walk(seconds=2.0)
    s, e = segments(masks["l_heel"])[1]
    root = np.asarray(a.root, dtype=np.float64).copy()
    root[e - 2 :] += np.array([0.004, 0.0, 0.0])  # the last two plant frames slide 4 + 4 mm/frame
    root[e - 1 :] += np.array([0.004, 0.0, 0.0])
    fix = FootFix(np.asarray(a.rot, np.float64), root, np.zeros(len(root)), {}, {})
    m = {ch: masks[ch].copy() for ch in FOOT_CHANNELS}
    forced = {ch: np.zeros(len(root), dtype=bool) for ch in FOOT_CHANNELS}
    n = _trim_sliding_edges(SK, ROLES, fix, m, forced, 30.0, ContactParams())
    assert n >= 2
    assert not m["l_heel"][e - 2 : e].any() and m["l_heel"][s : e - 2].all()
    m = {ch: masks[ch].copy() for ch in FOOT_CHANNELS}
    forced["l_heel"][s:e] = True  # painted: kept as painted
    _trim_sliding_edges(SK, ROLES, fix, m, forced, 30.0, ContactParams())
    assert m["l_heel"][s:e].all()


def test_only_plant_frames_request_a_pelvis_drop():
    """Review: frames of a 3-frame gap between two plants have summed blend weights >= 1; counting
    them as planted (w >= 1) made them request a pelvis drop. The plant masks decide."""
    a, _ = walk(seconds=1.0, speed=0.0)
    n = 25
    rot = np.repeat(np.asarray(a.rot[:1], np.float64), n, axis=0)
    root0 = np.repeat(np.asarray(a.root[:1], np.float64), n, axis=0)
    masks = {ch: np.zeros(n, dtype=bool) for ch in FOOT_CHANNELS}
    masks["l_heel"][:10] = True
    masks["l_heel"][13:] = True  # gap: frames 10..12
    params = ContactParams(pelvis_smooth_frames=0, pin_reach=False, ik_smooth=0.0)
    lifted = root0 + np.array([0.0, 0.10, 0.0])  # 10 cm above the pose the pins come from
    fix = apply_foot_targets(SK, ROLES, rot, lifted, masks, Floor.horizontal(0.0), params, ref=(rot, root0))
    assert fix.weights["l_heel"][10:13].min() >= 1.0  # the gap frames' summed weights
    assert fix.pelvis_drop[:10].min() > 0.01 and fix.pelvis_drop[13:].min() > 0.01
    assert np.all(fix.pelvis_drop[10:13] == 0.0)


# ---------------------------------------------------------------------------
# Round 3: one sole reference for flat, pivoting and floating plants
# ---------------------------------------------------------------------------

_X = np.array([1.0, 0.0, 0.0])


def _pitched_stance(pitch: float, n: int = 30, knee: float = 0.4) -> ActorMotion:
    """Bent-knee stance whose feet are pitched by ``pitch`` rad (> 0: toe down) and whose heels sit
    exactly at their clearance; GEM-X-like contact probabilities say both feet are down."""
    a = _stance(0.0, n, knee)
    for side in "lr":
        # _stance's thigh/calf leave the foot pitched by knee / 2: undo that, then pitch.
        a.rot[:, SK.index(f"{side}_foot")] = quat_axis_angle(
            np.tile(_X, (n, 1)), np.full(n, pitch - knee / 2)
        )
    pos = forward_kinematics(SK, a.rot, a.root)[0]
    clear = ROLES.clearances(SK)
    a.root[:, 1] -= min(float(pos[0, SK.index(f"{s}_foot"), 1]) - clear[f"{s}_heel"] for s in "lr")
    return a


@pytest.mark.parametrize("pitch", [0.2, -0.2])
def test_flat_plant_puts_the_toe_on_the_floor(pitch):
    """Round 3: a flat (heel + toe) plant derives its toe pin from the heel pin and the mean planted
    foot orientation. GEM-X's planted foot is rarely flat, so the toe pin landed below the floor (toe
    down: the floor clamp lifted the toe off its pin, the pin-miss rule unplanted it: ref walk B kept
    38 of 137 toe frames) or above it (toe up: a floating toe contact, QC hover 2.9 cm). The pinned foot
    is now pitched so the toe sits between its clearance and ``max_plant_hover`` above it."""
    p = ContactParams()
    bm = body({"A": _pitched_stance(pitch)}, GEMX_CHANNELS)
    out, rep = clean_contacts(bm, p, floor=Floor.horizontal(0.0))
    info = rep.actors["A"]
    pos = forward_kinematics(SK, out.actors["A"].rot, out.actors["A"].root)[0]
    clear = ROLES.clearances(SK)
    for side in "lr":
        toe = info.masks[f"{side}_toe"]
        assert toe.mean() >= 0.8, (side, toe.mean())  # kept, not unplanted by the pin-miss rule
        h = pos[toe, SK.index(f"{side}_toe"), 1] - clear[f"{side}_toe"]
        assert h.min() >= -1e-4 and h.max() <= p.max_plant_hover + 1e-3, (side, h.min(), h.max())
        assert info.fix.miss[f"{side}_toe"][toe].max() <= 1e-3
    s = summarise(per_frame_metrics(out))["A"]
    assert s["max_hover_in_contact_cm"] <= 100 * (p.max_plant_hover + HOVER_TOL)


def test_pivot_toe_plants_are_split_between_heel_plants():
    """Round 3: a toe plant spanning two heel plants (heel lifted and set down again over the ball)
    took its pin from one heel plant, so every frame missed the other's foot pin by 2-4 cm and was
    unplanted (Who Am I A: 23 of 23 frames). It is split at the toe's fastest frame between them."""
    n = 30
    masks = {ch: np.zeros(n, dtype=bool) for ch in FOOT_CHANNELS}
    masks["l_heel"][5:12] = True
    masks["l_heel"][16:25] = True
    masks["l_toe"][3:28] = True
    pos = np.zeros((n, SK.num_joints, 3))
    pos[14:, SK.index("l_toe"), 0] = 0.01  # the toe steps 1 cm between frames 13 and 14
    removed = split_pivot_toe_plants(SK, ROLES, pos, masks, np.array([0.0, 1.0, 0.0]), ContactParams())
    assert removed == 1
    assert segments(masks["l_toe"]) == [(3, 14), (15, 28)]
    for s, e in segments(masks["l_toe"]):
        assert len(segments(masks["l_heel"][s:e])) == 1


def test_toe_only_frames_pivot_on_their_pin():
    """Round 3: frames where the toe is planted but the heel is not (push-off: the heel rises while
    the ball stays down) missed the toe pin by 1.4-5 cm on almost every such frame (ref walk A: 44 of
    46) and were unplanted: the ankle followed the heel plant's eased-out offset, which does not fit
    the pin. The ankle is now put on the sphere of foot length around the toe pin."""
    n, lift = 30, 20
    a = _pitched_stance(0.0, n)
    t = np.arange(n)
    pitch = np.where(t >= lift, np.minimum(t - lift + 1, 8) * 0.05, 0.0)
    a.rot[:, SK.index("l_foot")] = quat_axis_angle(np.tile(_X, (n, 1)), pitch - 0.2)
    pos = forward_kinematics(SK, a.rot, a.root)[0]
    it = SK.index("l_toe")
    a.root += pos[0, it] - pos[:, it]  # the ball stays put while the heel rises ...
    a.root[:, 0] += np.where(t >= lift, 0.002 * (t - lift), 0.0)  # ... give or take GEM-X noise
    contact = np.full((n, len(GEMX_CHANNELS)), 0.02, np.float32)
    contact[:lift, 0] = 0.95  # left heel down until the push-off
    contact[:, 1] = 0.95  # left ball down throughout
    a.contact = contact
    out, rep = clean_contacts(body({"A": a}, GEMX_CHANNELS), ContactParams(), floor=Floor.horizontal(0.0))
    info = rep.actors["A"]
    toe_only = info.masks["l_toe"] & ~info.masks["l_heel"]
    assert toe_only.sum() >= 6, toe_only.sum()  # the push-off frames stay planted
    assert info.fix.miss["l_toe"][info.masks["l_toe"]].max() <= 1e-3
    p_out = forward_kinematics(SK, out.actors["A"].rot, out.actors["A"].root)[0]
    m = info.masks["l_toe"]
    step = np.linalg.norm(np.diff(p_out[:, it], axis=0), axis=-1)[m[1:] & m[:-1]]
    assert step.max() <= 1e-3  # the ball does not slide


def test_heel_only_plant_floating_above_max_hover_is_unplanted():
    """Round 3: a planted foot touches the floor with its lowest sole point. A heel pin raised to
    stay reachable (up to ``max_pin_raise``) is a heel lift only while the ball is down; a heel-only
    plant whose whole foot stays higher than ``max_plant_hover`` floats and is unplanted."""
    p = ContactParams(pin_reach_drop=0.0)
    a = _pitched_stance(-0.45, knee=0.0)  # straight legs, toes up (well off the floor) ...
    a.root[:, 1] += 0.028  # ... heels 2.8 cm up: reachable only by a raised pin
    a.contact[:, [1, 3]] = 0.02  # GEM-X: heels down, balls not
    out, rep = clean_contacts(body({"A": a}, GEMX_CHANNELS), p, floor=Floor.horizontal(0.0))
    s = summarise(per_frame_metrics(out))["A"]
    assert s["max_hover_in_contact_cm"] <= 100 * (p.max_plant_hover + HOVER_TOL)
    assert not any(m.any() for m in rep.actors["A"].masks.values())
    assert rep.actors["A"].miss_unplanted_frames > 0  # unplanted by the float rule, not dropped silently
    # Heels 1.5 cm up: a pin within ``max_plant_hover``, no float: kept.
    a.root[:, 1] -= 0.013
    out, rep = clean_contacts(body({"A": a}, GEMX_CHANNELS), p, floor=Floor.horizontal(0.0))
    assert all(rep.actors["A"].masks[ch].mean() >= 0.8 for ch in ("l_heel", "r_heel"))
    assert not rep.actors["A"].masks["l_toe"].any()


def test_foot_orientation_of_a_plant_is_not_overwritten_by_the_next_plant():
    """Round 3: the blend frames of a heel plant overwrote the pinned foot orientation of the last
    frames of the previous plant when the two were closer than ``blend_frames``, so the toe left its
    pin by up to 0.8 cm for 2-3 frames."""
    n = 26
    a = _pitched_stance(0.0, n)
    yaw = np.where(np.arange(n) >= 12, 0.35, 0.0)  # the foot turns between the plants
    a.rot[:, SK.index("l_foot")] = qmul(
        quat_axis_angle(np.tile([0.0, 1.0, 0.0], (n, 1)), yaw), np.asarray(a.rot[:, SK.index("l_foot")])
    )
    masks = {ch: np.zeros(n, dtype=bool) for ch in FOOT_CHANNELS}
    for ch in ("l_heel", "l_toe"):
        masks[ch][:10] = True
        masks[ch][13:] = True
    fix = apply_foot_targets(
        SK, ROLES, a.rot, a.root, masks, Floor.horizontal(0.0), ContactParams(ik_smooth=0.0)
    )
    assert fix.miss["l_toe"][:10].max() <= 1e-4 and fix.miss["l_toe"][13:].max() <= 1e-4


# ---------------------------------------------------------------------------
# Round 3 review: no pops, slides or snaps from the round 3 plant rules
# ---------------------------------------------------------------------------


def test_pivot_correction_eases_out_without_reaiming_at_another_pin():
    """Review item 1: around toe-only frames the ankle correction was a fraction of the way to the
    *nearest* toe plant's pin, which switched mid-gap (a 22 cm ankle step on ref walk A). It is now
    computed on toe-only frames only and eased out like a plant offset."""
    n = 18
    a = _pitched_stance(0.0, n)
    pos = forward_kinematics(SK, a.rot, a.root)[0]
    leg = ROLES.legs[0]
    heel, toe = ROLES.channel("l_heel"), ROLES.channel("l_toe")
    it = SK.index(leg.toe)
    masks = {ch: np.zeros(n, dtype=bool) for ch in FOOT_CHANNELS}
    masks["l_toe"][:6] = True
    masks["l_toe"][11:] = True  # two toe-only plants, a 5-frame gap between them
    # pins 6 cm ahead of / behind the input ball, along the foot: the ankle has to follow them
    pins = LegPins([], [], [pos[3, it] + [0.0, 0.0, 0.06], pos[13, it] - [0.0, 0.0, 0.06]])
    off = _pivot_ankle(
        SK, leg, pos, np.zeros((n, 3)), masks, heel, toe, pins, Floor.horizontal(0.0), ContactParams()
    )
    ankle = pos[:, SK.index(leg.foot)] + off
    for (s, e), pin in zip(segments(masks["l_toe"]), pins.toe, strict=True):  # on the sphere around its pin
        ell = np.linalg.norm(pos[s:e, it] - pos[s:e, SK.index(leg.foot)], axis=-1)
        assert np.allclose(np.linalg.norm(ankle[s:e] - pin, axis=-1), ell, atol=1e-9)
    steps = np.linalg.norm(np.diff(off, axis=0), axis=-1)
    # eased through the gap: 9 cm of change over 5 frames, at most 3.5 cm a frame (was a 7 cm jump)
    assert steps[5:11].max() <= 0.035, steps[5:11]


def test_planted_joints_never_step_inside_a_plant():
    """Review item 3: a leg that misses its pin by less than ``max_pin_miss`` (1 cm) on one frame and
    reaches it on the next made a planted foot jump up to 30 cm/s. Frames whose planted joint steps
    faster than ``max_plant_step`` are unplanted and solved again."""
    a = _stance(0.0, n=30)
    a.root[14:16, 1] += 0.008  # straight legs pulled up 8 mm for two frames: pins out of reach
    bm = body({"A": a}, GEMX_CHANNELS)
    base = dict(root_correction=False, max_pelvis_drop=0.0, pin_reach=False, ik_smooth=0.0)
    for p, ok in ((ContactParams(**base), True), (ContactParams(**base, max_plant_step=10.0), False)):
        out, rep = clean_contacts(bm, p, floor=Floor.horizontal(0.0))
        pos = forward_kinematics(SK, out.actors["A"].rot, out.actors["A"].root)[0]
        worst = 0.0
        for c in ROLES.contacts:
            m = rep.actors["A"].masks[c.channel]
            both = m[1:] & m[:-1]
            st = np.linalg.norm(np.diff(pos[:, SK.index(c.joint)], axis=0), axis=-1)[both] * 30.0
            worst = max(worst, float(st.max()))
        limit = ContactParams().max_plant_step  # the default rule (p's own limit in the ok case)
        assert (worst <= limit + 1e-6) == ok, worst  # without the rule: a 24 cm/s jump


def test_foot_orientation_turns_smoothly_across_a_short_gap():
    """Review item 4: frames of a short gap between two pinned flat plants took the later plant's
    orientation at full weight, so the foot snapped by the whole difference in one frame (22 deg on
    ufc B). They now turn from one pinned orientation to the next."""
    n = 26
    a = _pitched_stance(0.0, n)
    yaw = np.where(np.arange(n) >= 12, 0.35, 0.0)
    a.rot[:, SK.index("l_foot")] = qmul(
        quat_axis_angle(np.tile([0.0, 1.0, 0.0], (n, 1)), yaw), np.asarray(a.rot[:, SK.index("l_foot")])
    )
    masks = {ch: np.zeros(n, dtype=bool) for ch in FOOT_CHANNELS}
    for ch in ("l_heel", "l_toe"):
        masks[ch][:10] = True
        masks[ch][13:] = True
    fix = apply_foot_targets(
        SK, ROLES, a.rot, a.root, masks, Floor.horizontal(0.0), ContactParams(ik_smooth=0.0)
    )
    w = forward_kinematics(SK, fix.rot, fix.root)[1][:, SK.index("l_foot")]
    step = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(w[1:] * w[:-1], axis=-1)), 0.0, 1.0)))
    assert step[8:14].max() <= 0.4 * np.degrees(0.35), step[8:14]  # 20 deg spread over the gap


def test_widened_cut_when_the_ball_moved_between_heel_plants():
    """Review item 5: a pivot split whose two pieces' balls are far apart (a step, not a pivot) is
    widened to the frames the toe needs at ``speed_threshold`` instead of one frame."""
    n = 30
    masks = {ch: np.zeros(n, dtype=bool) for ch in FOOT_CHANNELS}
    masks["l_heel"][5:12] = True
    masks["l_heel"][16:25] = True
    masks["l_toe"][3:28] = True
    pos = np.zeros((n, SK.num_joints, 3))
    pos[14:, SK.index("l_toe"), 0] = 0.07  # the ball moved 7 cm between the heel plants
    p = ContactParams()
    removed = split_pivot_toe_plants(SK, ROLES, pos, masks, np.array([0.0, 1.0, 0.0]), p, 30.0)
    need = min(int(np.ceil(0.07 * 30.0 / p.speed_threshold)), 16 - 12)  # bounded by the heel-plant gap
    assert removed >= need
    gaps = [s1 - e0 for (_, e0), (s1, _) in itertools.pairwise(segments(masks["l_toe"]))]
    assert gaps and min(gaps) >= need


def test_raised_heel_levels_the_ball_onto_the_floor():
    """Review item 6: a flat plant's toe used to be clamped to the nearest edge of its band, so a heel
    pinned 1.5 cm up with the toe higher left the whole foot floating at 1.5 cm. The band's top is
    lowered by the heel's own lift: the ball goes down to the floor."""
    a = _pitched_stance(-0.2, knee=0.4)  # toes up
    a.root[:, 1] += 0.015  # heels 1.5 cm up (within ``max_plant_hover``: the pin keeps it)
    out, rep = clean_contacts(body({"A": a}, GEMX_CHANNELS), ContactParams(), floor=Floor.horizontal(0.0))
    pos = forward_kinematics(SK, out.actors["A"].rot, out.actors["A"].root)[0]
    low = foot_sole_heights(SK, ROLES, pos, Floor.horizontal(0.0))
    for li, side in enumerate("lr"):
        m = rep.actors["A"].masks[f"{side}_heel"] & rep.actors["A"].masks[f"{side}_toe"]
        assert m.mean() >= 0.8
        assert low[li][m].max() <= 0.005, low[li][m].max()  # the ball is down (was 1.5 cm up)


def test_root_offset_rate_is_limited_between_stances():
    """Review: with GEM-X's feet drifting through each stance, the root offset that holds them still
    ramped one way in every stance and swung back in the plant-free frames between, up to 50 cm in 8
    frames (hips 11 cm/frame against the input's 3). Its rate is now limited to ``root_max_speed`` +
    ``root_speed_ratio`` x the input root speed; plants it can no longer hold are left to detection."""
    n = 40
    a = _stance(0.0, n=n, knee=0.4)
    pos = forward_kinematics(SK, a.rot, a.root)[0]  # the body stands still ...
    masks = {ch: np.zeros(n, dtype=bool) for ch in FOOT_CHANNELS}
    for s0, e0 in ((0, 15), (21, 36)):
        masks["l_heel"][s0:e0] = True
        pos[s0:e0, SK.index("l_foot"), 2] += 0.02 * np.arange(e0 - s0)  # ... while the planted foot slides
    params = ContactParams()
    d = solve_root_correction(SK, ROLES, pos, masks, Floor.horizontal(0.0), params, None, 30.0)
    rate = np.linalg.norm(np.diff(d, axis=0), axis=-1) * 30.0
    assert rate.max() <= params.root_max_speed * 1.01, rate.max()
    free = solve_root_correction(
        SK,
        ROLES,
        pos,
        masks,
        Floor.horizontal(0.0),
        ContactParams(root_max_speed=0.0, root_speed_ratio=0.0),
        None,
        30.0,
    )
    assert (np.linalg.norm(np.diff(free, axis=0), axis=-1) * 30.0).max() > 1.5 * params.root_max_speed


def _sliding_stances(n: int = 40):
    a = _stance(0.0, n=n, knee=0.4)
    pos = forward_kinematics(SK, a.rot, a.root)[0]
    masks = {ch: np.zeros(n, dtype=bool) for ch in FOOT_CHANNELS}
    for s0, e0 in ((0, 15), (21, 36)):
        masks["l_heel"][s0:e0] = True
        pos[s0:e0, SK.index("l_foot"), 2] += 0.02 * np.arange(e0 - s0)
    return pos, masks


@pytest.mark.parametrize(
    "kw",
    [{"root_max_speed": 0.0, "root_speed_ratio": 0.5}, {"root_max_speed": 1e-4, "root_speed_ratio": 0.0}],
)
def test_root_rate_limit_with_a_zero_limit_stays_finite(kw):
    """Review: with ``root_max_speed`` = 0 (allowed) the stiffness of the rate rows was 0 x the limit:
    NaN offsets. It is floored."""
    pos, masks = _sliding_stances()
    d = solve_root_correction(SK, ROLES, pos, masks, Floor.horizontal(0.0), ContactParams(**kw), None, 30.0)
    assert np.isfinite(d).all()


def test_root_rate_limit_exits_early(monkeypatch):
    """Review: the limiter ran all 12 iterations (its soft rows settle ~1 % above the limit, the exit
    test allowed 0.1 %). It exits once every rate is within 3 % (+ the row stiffness)."""
    import afv.motion.contacts as C

    calls = [0]
    real = C.spsolve

    def counted(*a, **k):
        calls[0] += 1
        return real(*a, **k)

    monkeypatch.setattr(C, "spsolve", counted)
    pos, masks = _sliding_stances()
    p = ContactParams()
    d = C.solve_root_correction(SK, ROLES, pos, masks, Floor.horizontal(0.0), p, None, 30.0)
    assert calls[0] < 12
    assert (np.linalg.norm(np.diff(d, axis=0), axis=-1) * 30.0).max() <= p.root_max_speed * 1.05


def test_held_one_frame_holes_are_re_merged():
    """Review: the unplant rules could leave a one-frame hole in a flag while the solve still held the
    foot (moved 0.0 cm): a game's foot IK would let go of it for a frame. Holes of up to 2 frames where
    the joint stays put are flagged again; a moving joint or a painted "free" frame is not."""
    a = _stance(0.0, n=20)
    fix = FootFix(np.asarray(a.rot, np.float64), np.asarray(a.root, np.float64), np.zeros(20), {}, {})
    free = {ch: np.zeros(20, dtype=bool) for ch in FOOT_CHANNELS}

    def masks():
        m = {ch: np.zeros(20, dtype=bool) for ch in FOOT_CHANNELS}
        m["l_heel"][2:9] = True
        m["l_heel"][10:16] = True  # a one-frame hole at 9
        return m

    m = masks()
    assert _fill_held_holes(SK, ROLES, fix, m, free, 30.0, ContactParams()) == 1 and m["l_heel"][2:16].all()
    m = masks()
    free["l_heel"][9] = True
    assert _fill_held_holes(SK, ROLES, fix, m, free, 30.0, ContactParams()) == 0 and not m["l_heel"][9]
    free["l_heel"][9] = False
    moved = np.asarray(a.root, np.float64).copy()
    moved[9:, 0] += 0.01  # the foot moves 1 cm across the hole
    fix2 = FootFix(np.asarray(a.rot, np.float64), moved, np.zeros(20), {}, {})
    m = masks()
    assert _fill_held_holes(SK, ROLES, fix2, m, free, 30.0, ContactParams()) == 0 and not m["l_heel"][9]


def test_edge_rules_trim_but_never_delete_a_plant(monkeypatch):
    """Review: the edge-pop and knee-snap rules unplanted both edge frames of a 3-frame plant, which
    then fell under ``min_plant_frames`` and vanished (ref walk/002 B lost its only plant). They only
    trim now: a plant keeps at least ``min_plant_frames``."""
    import afv.motion.contacts as C

    n = 30
    a = _stance(0.0, n=n, knee=0.4)
    masks = {ch: np.zeros(n, dtype=bool) for ch in FOOT_CHANNELS}
    masks["l_heel"][14:17] = True
    calls = [0]

    def snapping(sk, leg, pos):  # the output knee snaps into and out of the plant, the input's does not
        calls[0] += 1
        k = np.zeros(len(pos))
        if calls[0] % 2 == 1 and leg is ROLES.legs[0]:
            k[14:17] = 0.6
        return k

    monkeypatch.setattr(C, "_knee_angles", snapping)
    base = dict(root_correction=False, ik_smooth=0.0, pin_reach=False)
    fl = Floor.horizontal(0.0)
    trims = C.apply_foot_targets(
        SK, ROLES, a.rot, a.root, masks, fl, ContactParams(**base, min_plant_frames=1)
    )
    assert trims.stepping["l_heel"][14:17].any()  # the snap marks the plant's edge frames ...
    kept = C.apply_foot_targets(SK, ROLES, a.rot, a.root, masks, fl, ContactParams(**base))
    assert not kept.stepping["l_heel"][14:17].any()  # ... but not when that would delete the plant


def test_widened_cut_stays_between_the_heel_plants():
    """Review: the widened pivot cut could unflag toe frames under a heel plant; it stays in the gap."""
    n = 30
    masks = {ch: np.zeros(n, dtype=bool) for ch in FOOT_CHANNELS}
    masks["l_heel"][5:13] = True
    masks["l_heel"][15:25] = True  # a 2-frame gap, narrower than the 7 frames a 7 cm step needs
    masks["l_toe"][3:28] = True
    pos = np.zeros((n, SK.num_joints, 3))
    pos[14:, SK.index("l_toe"), 0] = 0.07
    split_pivot_toe_plants(SK, ROLES, pos, masks, np.array([0.0, 1.0, 0.0]), ContactParams(), 30.0)
    assert masks["l_toe"][5:13].all() and masks["l_toe"][15:25].all()
    assert not masks["l_toe"][13:15].any()
