"""Module E on REAL GEM-X data (binding for the fix round): pair -> contacts -> smooth on full-rate
fixtures (both fighters, ``root_prior`` set; motion data only).

Fixtures: ``tests/motion/fixtures/pose3d_{ufc_cross,close_boxing,ref_walk}.npz`` (+ camera) and
``tests/models/fixtures/pose3d_who_am_i.npz`` (+ camera), all cut from the pipeline's own pose3d and
camera stage results (round 2, stage version 2) by ``fixtures/make_real_fixtures.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from real_metrics import (
    FIXTURES,
    hips_px,
    knee_change,
    load_clip,
    plant_speed,
    positions,
    signed_knee,
    strike_ratios,
)

from afv.motion.contacts import (
    HOVER_TOL,
    ContactParams,
    Floor,
    actor_skeleton,
    clean_contacts,
    contact_masks,
    foot_lurches,
    foot_sole_heights,
    model_probabilities,
)
from afv.motion.ik import get_roles
from afv.motion.pair import PairParams, pair_fighters
from afv.motion.retime import retime
from afv.motion.smooth import SmoothParams, smooth_motion
from afv.qc.camera import camera_centres
from afv.qc.metrics import per_frame_metrics, summarise

CLIPS = [n for n, d in FIXTURES.items() if (d / f"pose3d_{n}.npz").exists()]
pytestmark = pytest.mark.skipif(not CLIPS, reason="real GEM-X fixtures missing")

# Round 2 re-baseline: the fixtures now come from the pipeline's own camera/pose3d v2 results (correct
# telephoto camera rotations, more plants kept). Cases that fail on the new data with the thresholds
# unchanged are listed here as strict xfails (an unexpected pass is reported too). The five found at the
# re-baseline (ref walk pelvis drop, close boxing / ref walk knee snaps, smooth(0) not a no-op on UFC and
# Who Am I) are fixed in module E (round 2 motion fixes) and have been removed.
# The two found after the gravity re-level pass and fixture rebuild (ref walk A knee change 26.5 deg
# per frame, Who Am I A plant recall 0.14) are fixed too (round 2 motion fixes 2: knee-space plant
# blends, push-apart order per episode, the pair stage's local-mean floor rows).
# The close boxing separation xfail added after the pose3d v3 fixture rebuild (1.58 -> 1.27 m vs a 0.30 m
# limit against pose3d) is resolved: pose3d's median is inflated by GEM-X depth excursions (10-frame
# medians 2.78 m at the start and 2.78 / 3.17 m during B's knockdown vs 1.3-1.7 m in between); the depth
# solve with every floor row off already gives 1.36 m. The test now checks what the floor rows add.
KNOWN_FAILS: dict[tuple[str, str], str] = {}

# Plant recall against GEM-X's own contacts (frames GEM-X calls a contact that the result plants), a
# conservative regression floor per fighter, kept well below the measured values so a regeneration of
# the fixtures does not trip it. Measured after round 3 (A / B): ufc 0.94 / 0.90, close boxing
# 0.70 / 0.86, ref walk 0.95 / 0.94, Who Am I 0.56 / 0.74 (round 2: 0.95 / 0.19, 0.58 / 0.62, 0.72 / 0.72,
# 0.25 / 0.52). Round 3 kept the toe channels: flat plants are levelled so the derived toe pin is on
# the floor (it was up to 3.4 cm below it, so the floor clamp pulled the toe off its pin and the
# pin-miss rule unplanted it), toe-only frames pivot about their pin, and toe plants spanning two heel
# plants are split. Who Am I A stays lowest: GEM-X itself calls a contact on only 36 frames of 122 (a
# jumping, kicking fighter; median max contact probability 0.26) and several of those are landings at
# 1-2 m/s. B used to lose almost every plant (ufc 0.07, close boxing 0.12, ref walk 0): its soles sat 8-12
# cm above A's after pair (per-fighter GEM-X depth bias), fixed by the pair stage's local-mean floor rows.
MIN_RECALL = {
    ("ufc_cross", "A"): 0.8,
    ("ufc_cross", "B"): 0.75,
    ("close_boxing", "A"): 0.55,
    ("close_boxing", "B"): 0.7,
    ("ref_walk", "A"): 0.8,
    ("ref_walk", "B"): 0.8,
    ("who_am_i", "A"): 0.45,
    ("who_am_i", "B"): 0.6,
}


def clips(test: str) -> list:
    return [
        pytest.param(n, marks=pytest.mark.xfail(strict=True, reason=KNOWN_FAILS[(test, n)]))
        if (test, n) in KNOWN_FAILS
        else n
        for n in CLIPS
    ]


@pytest.fixture(scope="module")
def chain() -> dict:
    out = {}
    for name in CLIPS:
        bm, cam = load_clip(name)
        p, _ = pair_fighters(bm, cam)
        c, crep = clean_contacts(p, camera=cam)
        s, _ = smooth_motion(c)
        out[name] = {
            "pose3d": bm, "cam": cam, "pair": p, "contacts": c, "crep": crep, "smooth": s,
            "retime60": retime(c, 60.0, 1.0), "retime24": retime(c, 24.0, 1.0),
        }  # fmt: skip
    return out


@pytest.mark.parametrize("name", CLIPS)
def test_feet_still_inside_plants(chain, name):
    d = chain[name]
    for r in ("A", "B"):
        for stage in ("contacts", "smooth", "retime60", "retime24"):
            sp = plant_speed(d[stage], r)  # per playback second at the result's own fps
            if sp.size >= 20:  # NFR-3, on enough planted frame pairs to mean something
                assert sp.mean() <= 0.03, (name, r, stage, sp.mean())


@pytest.mark.parametrize("name", clips("pins"))
def test_pins_reached_and_pelvis_residual(chain, name):
    d = chain[name]
    cap = ContactParams().max_pelvis_drop
    for r in ("A", "B"):
        fix = d["crep"].actors[r].fix
        miss = np.concatenate([m[m > 0] for m in fix.miss.values()] + [np.zeros(1)])
        assert np.percentile(miss, 95) <= 0.01 + 1e-6  # plant miss distance
        assert miss.max() <= ContactParams().max_pin_miss + 1e-6  # unreachable frames are unplanted
        assert np.abs(fix.pelvis_drop).max() <= cap + 1e-9
        assert np.abs(fix.pelvis_drop).mean() <= 0.02


@pytest.mark.parametrize("name", clips("knee"))
def test_no_knee_snaps(chain, name):
    """Knee changes per frame stay near the input's, measured signed about the input's knee hinge:
    a near-straight knee flipping from +47 to -50 deg (round 3 review, ref walk A) is a 97 deg snap
    that the unsigned angle reports as 3 deg. No knee bends backwards beyond the input's own."""
    d = chain[name]
    for r in ("A", "B"):
        k_in = knee_change(d["pose3d"], r)
        b_in = signed_knee(d["pose3d"], r, d["pose3d"])
        for stage in ("contacts", "smooth"):
            assert knee_change(d[stage], r) <= max(k_in + 10.0, 25.0), (name, r, stage)
            for bo, bi in zip(signed_knee(d[stage], r, d["pose3d"]), b_in, strict=True):
                step_in = np.abs(np.diff(bi)).max()
                assert np.abs(np.diff(bo)).max() <= max(step_in + 10.0, 25.0), (name, r, stage)
                assert bo.min() >= min(bi.min(), 0.0) - 5.0, (name, r, stage, bo.min(), bi.min())


@pytest.mark.parametrize("name", CLIPS)
def test_hips_stay_on_the_image(chain, name):
    d = chain[name]
    for r in ("A", "B"):
        h0 = hips_px(d["pose3d"], d["cam"], r)
        for stage in ("pair", "contacts", "smooth"):
            dh = np.linalg.norm(hips_px(d[stage], d["cam"], r) - h0, axis=-1)
            assert np.nanmean(dh) <= 10.0, (name, r, stage, np.nanmean(dh))
            assert np.nanpercentile(dh, 95) <= 20.0, (name, r, stage, np.nanpercentile(dh, 95))
        assert np.nanmax(np.linalg.norm(hips_px(d["pair"], d["cam"], r) - h0, axis=-1)) < 0.5


@pytest.mark.parametrize("name", clips("idempotent"))
def test_stages_are_idempotent(chain, name):
    d = chain[name]
    c, cam = d["contacts"], d["cam"]
    s0, _ = smooth_motion(c, SmoothParams(strength=0.0))  # re-pin onto its own pins: a no-op
    r30 = retime(c, 30.0, 1.0)
    c2, rep2 = clean_contacts(c, camera=cam)
    for r in ("A", "B"):
        pc = positions(c, r)[1]
        assert np.linalg.norm(positions(s0, r)[1] - pc, axis=-1).max() <= 0.01
        assert np.linalg.norm(positions(r30, r)[1] - pc, axis=-1).max() <= 0.01
        # Contacts on its own output: no new root correction, the floor kept, pins kept.
        assert np.abs(rep2.actors[r].root_offset).max() <= 0.005
        assert abs(rep2.floor_shift) <= 0.02
        root_move = np.linalg.norm(np.asarray(c2.actors[r].root) - np.asarray(c.actors[r].root), axis=-1)
        assert root_move.max() <= 0.04
        assert plant_speed(c2, r).mean() <= 0.03 if plant_speed(c2, r).size >= 20 else True


def _frames(bm, cut: slice):
    return type(bm)(
        frames=bm.frames[cut],
        timestamps=bm.timestamps[cut],
        skeleton=bm.skeleton,
        fps=bm.fps,
        contact_channels=bm.contact_channels,
        world_shift=bm.world_shift,
        actors={
            k: type(a)(a.rot[cut], a.root[cut], rest_offsets=a.rest_offsets) for k, a in bm.actors.items()
        },
    )


@pytest.mark.skipif("who_am_i" not in CLIPS, reason="module C's Who Am I fixture missing")
def test_strike_retention_on_real_flurries(chain):
    d = chain["who_am_i"]
    edge = SmoothParams().strike_edge_frames
    for r in ("A", "B"):
        c, s = d["contacts"], d["smooth"]
        cut = slice(edge, c.num_frames - edge)  # peaks at the clip ends are not judged
        ratios = strike_ratios(_frames(c, cut), _frames(s, cut), r, ("LeftHand", "RightHand"))
        print(
            f"who_am_i {r}: {len(ratios)} wrist peaks > 3 m/s, kept min {ratios.min():.2f}, "
            f"median {np.median(ratios):.2f}"
        )
        assert len(ratios) >= 10
        assert ratios.min() >= 0.8  # NFR-4


@pytest.mark.parametrize("name", CLIPS)
def test_one_bad_frame_does_not_spread(name):
    bm, cam = load_clip(name)
    bm.actors["B"].root = np.asarray(bm.actors["B"].root).copy()
    bm.actors["B"].root[10] = np.nan
    bm.actors["B"].rot = np.asarray(bm.actors["B"].rot).copy()
    bm.actors["B"].rot[11, 5] = np.nan
    p, _ = pair_fighters(bm, cam)
    c, rep = clean_contacts(p, camera=cam)
    s, _ = smooth_motion(c)
    assert rep.filled_frames["B"] == 0  # pair already filled them
    for res in (p, c, s):
        for a in res.actors.values():
            assert np.isfinite(a.root).all() and np.isfinite(a.rot).all()


@pytest.mark.parametrize("name", CLIPS)
def test_retimed_plants_do_not_jump(chain, name):
    """Round 2 fix: at 60 fps a sample between a planted and a lifted source frame is not a plant, so
    planted feet do not jump at plant edges (was 2 m/s jumps, 44 cm/s p95 on close boxing)."""
    for r in ("A", "B"):
        src = plant_speed(chain[name]["contacts"], r)
        for stage in ("retime60", "retime24"):
            sp = plant_speed(chain[name][stage], r)
            if sp.size and src.size:  # resampled plants are as still as the source plants
                assert np.percentile(sp, 95) <= np.percentile(src, 95) + 0.01, (name, r, stage)
                assert sp.max() <= src.max() + 0.01, (name, r, stage, sp.max(), src.max())


@pytest.mark.parametrize("name", CLIPS)
def test_feet_never_below_the_floor(chain, name):
    """Round 2 fix: after contacts no foot joint is below its clearance (exact toe lift, per-leg floor
    clamp after the IK smoothing), so the exact clamp of later re-pins has nothing to move."""
    for r in ("A", "B"):
        sk, pos = positions(chain[name]["contacts"], r)
        roles = get_roles(sk)
        clear = roles.clearances(sk)
        for ch, h0 in clear.items():
            assert pos[:, sk.index(roles.channel(ch).joint), 1].min() >= h0 - 2e-4, (name, r, ch)


@pytest.mark.parametrize("name", CLIPS)
def test_smooth_zero_is_a_no_op(chain, name):
    s0, _ = smooth_motion(chain[name]["contacts"], SmoothParams(strength=0.0))
    for r in ("A", "B"):
        dev = np.linalg.norm(positions(s0, r)[1] - positions(chain[name]["contacts"], r)[1], axis=-1)
        assert dev.max() <= 0.002, (name, r, dev.max())


@pytest.mark.parametrize("name", clips("recall"))
def test_plant_recall_vs_gemx(chain, name):
    d = chain[name]
    for r in ("A", "B"):
        if (name, r) not in MIN_RECALL:
            continue
        c = d["contacts"]
        roles = get_roles(actor_skeleton(c, c.actors[r]))
        gem = model_probabilities(roles, d["pose3d"].contact_channels, d["pose3d"].actors[r].contact)
        masks = contact_masks(c, c.actors[r], roles)
        g = np.stack([gem[ch] > 0.5 for ch in masks])
        m = np.stack([masks[ch] for ch in masks])
        recall = (g & m).sum() / max(g.sum(), 1)
        print(f"{name} {r}: recall {recall:.2f}")
        assert recall >= MIN_RECALL[(name, r)], (name, r, recall)


@pytest.mark.parametrize("name", CLIPS)
def test_planted_contacts_do_not_float(chain, name):
    """A planted contact must not float (a game's foot IK keyed on contact flags would snap it down).
    Round 3: a planted foot touches the floor with its lowest sole point (heel or ball), at most
    ``max_plant_hover`` up, which QC reports as ``max_hover_in_contact_cm`` (round 2 measured each
    joint against ``max_pin_raise`` + miss: whole feet floated 3-4 cm); a single planted joint may still
    sit up to ``max_pin_raise`` (+ miss) above its clearance (a heel lifted off a planted ball)."""
    p = ContactParams()
    crep = chain[name]["crep"]
    for r in ("A", "B"):
        sk, pos = positions(chain[name]["contacts"], r)
        roles = get_roles(sk)
        clear = roles.clearances(sk)
        lowest = foot_sole_heights(sk, roles, pos, Floor.horizontal(0.0))
        for ch, m in crep.actors[r].masks.items():
            h = pos[m, sk.index(roles.channel(ch).joint), 1] - clear[ch]
            assert h.size == 0 or h.max() <= p.max_pin_raise + p.max_pin_miss + 1e-6, (name, r, ch, h.max())
            foot = lowest[roles.channel(ch).leg][m]
            assert foot.size == 0 or foot.max() <= p.max_plant_hover + HOVER_TOL, (name, r, ch, foot.max())
        s = summarise(per_frame_metrics(chain[name]["contacts"]))[r]
        assert s["max_hover_in_contact_cm"] <= 100 * (p.max_plant_hover + HOVER_TOL), (name, r)
        info = crep.actors[r]
        assert min(info.raised_plants, info.capped_plants, info.dropped_plants) >= 0
        assert info.unsolved_unplanted_frames >= 0 and info.miss_unplanted_frames >= 0


def _planted_sole(bm, gem, role: str) -> float:
    """Median height above its clearance of each foot joint on GEM-X's contact frames (prob > 0.5)."""
    sk, pos = positions(bm, role)
    roles = get_roles(sk)
    clear = roles.clearances(sk)
    probs = model_probabilities(roles, gem.contact_channels, gem.actors[role].contact)
    h = [
        pos[probs[c.channel] > 0.5, sk.index(c.joint), 1] - clear[c.channel]
        for c in roles.contacts
        if probs.get(c.channel) is not None
    ]
    return float(np.median(np.concatenate(h)))


def _pelvis_separation(bm) -> float:
    (ska, pa), (skb, pb) = positions(bm, "A"), positions(bm, "B")
    pel = get_roles(ska).pelvis
    d = pa[:, ska.index(pel)] - pb[:, skb.index(pel)]
    return float(np.median(np.linalg.norm(d[:, [0, 2]], axis=-1)))


@pytest.mark.parametrize("name", clips("floor"))
def test_fighters_share_the_floor_after_pair(chain, name):
    """Round 2: after pair both fighters' planted soles are on one floor (B floated 8-12 cm above A on
    the telephoto clips: a per-fighter GEM-X depth bias of ~3 %), through depth along the rays only,
    by a plausible amount (a few % of the distance), without changing the fighters' separation much.
    Where the gap would need more (ufc: close range; Who Am I: nearly horizontal rays) it is left.

    The separation is compared with the same depth solve without the local-mean floor/depth rows, not
    with pose3d: pose3d's own median includes GEM-X depth excursions the depth smoothing removes anyway
    (close boxing: 1.58 m in pose3d, 1.36 m with every floor row off, 1.27 m with the rows)."""
    d = chain[name]
    bm, p, cam = d["pose3d"], d["pair"], d["cam"]
    p_off, _ = pair_fighters(bm, cam, PairParams(floor_mean_sigma=None, depth_mean_sigma_rel=None))
    both = (("in", bm), ("pair", p))
    gap = {k: abs(_planted_sole(x, bm, "A") - _planted_sole(x, bm, "B")) for k, x in both}
    sep = {k: _pelvis_separation(x) for k, x in (*both, ("rows off", p_off))}
    centres = {k: camera_centres(cam, x.frames) + np.asarray(x.world_shift) for k, x in both}
    print(
        f"{name}: sole gap {gap['in'] * 100:.1f} -> {gap['pair'] * 100:.1f} cm, "
        f"separation {sep['in']:.2f} -> {sep['pair']:.2f} m ({sep['rows off']:.2f} m without the rows)"
    )
    assert gap["pair"] <= gap["in"] + 0.01
    need = []
    for r in ("A", "B"):
        ray = np.asarray(bm.actors[r].root, np.float64) - centres["in"]
        z_in = np.linalg.norm(ray, axis=-1)
        z_out = np.linalg.norm(np.asarray(p.actors[r].root, np.float64) - centres["pair"], axis=-1)
        assert abs(np.mean(z_out - z_in)) <= 0.05 * np.mean(z_in), (name, r)
        # depth change (fraction of the distance) that closes half the gap along this fighter's ray
        need.append(0.5 * gap["in"] / max(abs(float(np.median(ray[:, 1] / z_in))), 1e-3) / np.median(z_in))
    if max(need) <= 0.045:  # a plausible per-fighter depth (scale) error explains the gap: it is closed
        assert gap["pair"] <= max(0.05, 0.5 * gap["in"]), (name, gap, need)
    assert abs(sep["pair"] - sep["rows off"]) <= 0.3  # measured 0.08-0.22 m


@pytest.mark.parametrize("name", CLIPS)
def test_feet_do_not_lurch(chain, name):
    """Review: 3-step foot lurches against each stage's input (``foot_lurches``). Contacts: the root
    correction's rate limit keeps catch-ups to a few frames (fixtures: 0-9 %, close boxing A highest;
    without the limit the depth offset swung 50 cm in 8 frames). Smooth adds none (it used to fill
    GEM-X's foot stalls and pull the clip ends along)."""
    d = chain[name]
    for r in ("A", "B"):
        sk, p_pair = positions(d["pair"], r)
        p_c, p_s = positions(d["contacts"], r)[1], positions(d["smooth"], r)[1]
        roles = get_roles(sk)
        fc = np.zeros(len(p_c), dtype=bool)
        fs = np.zeros(len(p_c), dtype=bool)
        for leg in roles.legs:
            for jn in (leg.foot, leg.toe):
                j = sk.index(jn)
                fc |= foot_lurches(p_c[:, j], p_pair[:, j])
                fs |= foot_lurches(p_s[:, j], p_c[:, j])
        assert fc.mean() <= 0.12, (name, r, fc.mean())
        assert fs.sum() <= 2, (name, r, int(fs.sum()))
