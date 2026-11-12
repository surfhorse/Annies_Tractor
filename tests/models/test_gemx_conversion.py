"""SOMA -> soma77 conversion on real GEM-X P0 data (fixtures: motion data only, no footage)."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

from afv.core.skeletons import get_skeleton, load_skeleton_json
from afv.core.types import BodyMotion, load_result, save_result
from afv.models.camera import place_on_ground, shifted_camera, world_to_camera_cv
from afv.models.gemx import (
    FLOOR_JOINTS,
    GEMX_CONTACT_CHANNELS,
    GEMX_CONTACT_LOGITS,
    SomaFK,
    body_motion_from_fk,
    camera_from_gemx,
    contact_probabilities,
    fighter_rest_offsets,
    fit_floor,
    gemx_world_to_world,
    sole_heights,
    unobserved_spans,
)
from afv.motion.fk import forward_kinematics, rest_positions

FIX = Path(__file__).parent / "fixtures"
FIXTURES = sorted(FIX.glob("gemx_*.npz"))
SOMA77_JSON = Path(__file__).parents[2] / "src/afv/core/data/skeletons/soma77.json"
P0_FK = sorted(glob.glob(str(Path.home() / "afv/runs/afv_c_fk/*_rank*.npz")))
P0_RUNS = Path.home() / "afv/runs"


@pytest.fixture(params=FIXTURES, ids=lambda p: p.stem)
def fk(request) -> SomaFK:
    return SomaFK.load(request.param)


def _fk_positions(motion: BodyMotion, role: str = "A") -> np.ndarray:
    sk = get_skeleton(motion.skeleton)
    a = motion.actors[role]
    pos, _ = forward_kinematics(sk.with_offsets(a.rest_offsets.astype(np.float64)), a.rot, a.root)
    return pos


def _in_camera(motion: BodyMotion, cam, pos: np.ndarray) -> np.ndarray:
    """World positions of ``motion`` -> OpenCV camera frame (camera moved by ``world_shift``)."""
    r, t = world_to_camera_cv(shifted_camera(cam, motion.world_shift))
    return np.einsum("fij,fnj->fni", r, pos) + t[:, None]


# --- skeleton ---------------------------------------------------------------------------------


def test_soma77_skeleton_registered():
    sk = get_skeleton("soma77")
    assert sk.num_joints == 77 and sk.joint_names[0] == "Hips" and sk.parents[0] == -1
    assert sk.rest_rotations is None
    assert sk.parents[sk.index("LeftForeArm")] == sk.index("LeftArm")
    assert sk.parents[sk.index("RightLeg")] == sk.index("Hips")
    for name in GEMX_CONTACT_CHANNELS:
        sk.index(name)
    assert load_skeleton_json(SOMA77_JSON).joint_names == sk.joint_names


def test_soma77_rest_pose_is_y_up_t_pose_facing_plus_z():
    sk = get_skeleton("soma77")
    p = rest_positions(sk)
    j = sk.index
    assert p[j("LeftHand"), 0] > 0.6 and p[j("RightHand"), 0] < -0.6  # arms out, left = +X
    assert abs(p[j("LeftHand"), 1] - p[j("LeftArm"), 1]) < 0.01  # horizontal arms
    assert p[j("HeadEnd"), 1] > 1.6  # ~1.7 m tall, Y up
    assert 0.0 < p[j("LeftToeEnd"), 1] < 0.03 and 0.05 < p[j("LeftFoot"), 1] < 0.12  # sole at y = 0
    assert p[j("LeftToeEnd"), 2] > p[j("LeftFoot"), 2] + 0.1  # toes point +Z (forward)


def test_soma77_regions_cover_all_joints():
    import json

    d = json.loads(SOMA77_JSON.read_text())
    names = set(d["joint_names"])
    regions = d["regions"]
    assert set(regions["body"]) | set(regions["hands"]) | set(regions["face"]) == names
    assert len(regions["hands"]) == 48 and regions["face"] == ["Jaw", "LeftEye", "RightEye"]
    assert list(d["contact_channels"]) == list(GEMX_CONTACT_CHANNELS)
    assert list(d["contact_channels"].values()) == list(GEMX_CONTACT_LOGITS)


# --- conversion ---------------------------------------------------------------------------------


def test_fk_reproduces_soma_joints_in_camera_frame(fk):
    """afv FK on the converted motion = SOMA's own joints (GEM-X incam), <= 1 mm."""
    cam = camera_from_gemx(fk)
    for refit in (False, True):
        motion, _ = body_motion_from_fk({"A": fk}, cam, refit_floor=refit)
        err = np.linalg.norm(_in_camera(motion, cam, _fk_positions(motion)) - fk.incam_pos, axis=-1)
        assert err.max() < 1e-3, f"max FK error {err.max() * 1000:.3f} mm"


def test_fk_reproduces_soma_joints_gemx_world(fk):
    cam = camera_from_gemx(fk)
    motion, qc = body_motion_from_fk({"A": fk}, cam, root_mode="gemx_world")
    _, ref, _ = gemx_world_to_world(fk, cam)
    err = np.linalg.norm(_fk_positions(motion) - (ref + motion.world_shift), axis=-1)
    assert err.max() < 1e-3
    assert abs(qc["fighters"]["A"]["alignment"]["yaw_deg"]) <= 180.0


def test_rest_offsets_are_this_fighters_bones(fk):
    sk = get_skeleton("soma77")
    off = fighter_rest_offsets(fk)
    assert off.shape == (77, 3)
    neutral = np.linalg.norm(sk.rest_offsets[1:], axis=1)
    mine = np.linalg.norm(off[1:], axis=1)
    ratio = mine[neutral > 0.02] / neutral[neutral > 0.02]
    assert 0.8 < ratio.min() and ratio.max() < 1.25  # same skeleton, the fighter's own lengths
    assert not np.allclose(mine, neutral, atol=1e-4)  # identity/scale actually used
    # bone lengths are locked: the posed SOMA joints keep exactly these lengths on every frame
    for j, p in enumerate(fk.parents):
        if p >= 0:
            d = np.linalg.norm(fk.incam_pos[:, j] - fk.incam_pos[:, p], axis=-1)
            assert np.allclose(d, mine[j - 1], atol=1e-4)  # 0.1 mm (float32 joints ~20 m away)


def test_local_rotations_unit_and_continuous(fk):
    motion, _ = body_motion_from_fk({"A": fk}, camera_from_gemx(fk))
    q = motion.actors["A"].rot.astype(np.float64)
    assert q.shape == (fk.num_frames, 77, 4)
    assert np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-5)
    assert (np.sum(q[1:] * q[:-1], axis=-1) >= 0).all()


def test_contacts_and_confidence(fk):
    motion, _ = body_motion_from_fk({"A": fk}, camera_from_gemx(fk))
    a = motion.actors["A"]
    assert motion.contact_channels == list(GEMX_CONTACT_CHANNELS)
    assert a.contact.shape == (fk.num_frames, 6)
    assert np.allclose(a.contact, 1.0 / (1.0 + np.exp(-fk.contact_logits)), atol=1e-6)
    assert a.conf.shape == (fk.num_frames, 77) and 0.0 <= a.conf.min() and a.conf.max() <= 1.0


def test_confidence_is_zero_on_unobserved_box_frames(fk):
    fk.observed = fk.observed.copy()
    fk.observed[3:7] = False
    fk.observed[-2:] = False
    motion, qc = body_motion_from_fk({"A": fk}, camera_from_gemx(fk))
    conf = motion.actors["A"].conf
    assert (conf[3:7] == 0).all() and (conf[-2:] == 0).all()
    assert np.allclose(conf[0], np.clip(fk.kp2d[0, :, 2], 0, 1))
    n = fk.num_frames
    assert qc["fighters"]["A"]["unobserved_spans"] == [[3, 6], [n - 2, n - 1]]


def test_unobserved_spans():
    assert unobserved_spans(np.array([1, 1, 1], bool)) == []
    assert unobserved_spans(np.array([0, 0, 1, 0, 1, 0], bool)) == [[0, 1], [3, 3], [5, 5]]


def test_floor_refit_puts_planted_feet_on_y0(fk):
    """pose3d's floor re-fit: contact-weighted median sole height = 0; world_shift records the move."""
    cam = camera_from_gemx(fk)  # camera at the origin: the floor starts ~1-20 m off
    sk = get_skeleton("soma77")
    m0, _ = body_motion_from_fk({"A": fk, "B": fk}, cam, refit_floor=False)
    m1, qc = body_motion_from_fk({"A": fk, "B": fk}, cam)
    assert m0.world_shift == [0.0, 0.0, 0.0]
    assert m1.world_shift[0] == 0.0 and m1.world_shift[2] == 0.0
    assert m1.world_shift[1] == pytest.approx(-qc["floor"]["floor_y_in_camera_world_m"], abs=1e-4)
    a = m1.actors["A"]
    assert np.allclose(a.root, m0.actors["A"].root + np.asarray(m1.world_shift), atol=1e-5)
    sole = sole_heights(a, sk).ravel()
    w = (a.contact[:, : len(FLOOR_JOINTS)] * fk.observed[:, None]).ravel()
    order = np.argsort(sole)
    med = sole[order][np.searchsorted(np.cumsum(w[order]), 0.5 * w.sum())]
    assert abs(med) < 1e-4
    assert qc["floor"]["method"] == "contact_weighted_median"
    # reprojection uses cam_pos + world_shift: unchanged by the move
    _, q0 = body_motion_from_fk({"A": fk}, cam, refit_floor=False)
    rep0, rep1 = q0["fighters"]["A"]["reprojection_px_median"], qc["fighters"]["A"]["reprojection_px_median"]
    assert rep1 == pytest.approx(rep0, abs=1e-4)  # float32 roots


def test_floor_falls_back_without_contacts(fk):
    sk = get_skeleton("soma77")
    motion, _ = body_motion_from_fk({"A": fk}, camera_from_gemx(fk), refit_floor=False)
    a = motion.actors["A"]
    a.contact = np.zeros_like(a.contact)
    y, info = fit_floor({"A": a}, {"A": fk.observed}, sk)
    assert info["method"] == "lowest_sole_percentile"
    assert y == pytest.approx(np.percentile(sole_heights(a, sk).min(axis=1), 10.0))


def test_root_prior_is_gemx_world_root_in_the_same_world(fk):
    """root_prior = GEM-X's world-rollout hips, one rigid yaw + translation into root's world."""
    cam = camera_from_gemx(fk)
    motion, qc = body_motion_from_fk({"A": fk}, cam)
    a = motion.actors["A"]
    assert a.root_prior.shape == a.root.shape and a.root_prior.dtype == np.float32
    _, g_pos, align = gemx_world_to_world(fk, cam)
    assert np.allclose(a.root_prior, g_pos[:, 0] + motion.world_shift, atol=1e-4)
    assert qc["fighters"]["A"]["root_prior_alignment"] == align
    # rigid: its speeds are exactly GEM-X's own world speeds (frame spacing does not matter)
    d_prior = np.linalg.norm(np.diff(a.root_prior.astype(np.float64), axis=0), axis=1)
    d_gemx = np.linalg.norm(np.diff(fk.global_pos[:, 0], axis=0), axis=1)
    assert np.allclose(d_prior, d_gemx, atol=1e-4)
    # same world as root: the median translation offset puts them together
    assert np.linalg.norm(np.median(a.root_prior - a.root, axis=0)) < 1e-3
    mg, _ = body_motion_from_fk({"A": fk}, cam, root_mode="gemx_world")
    assert np.allclose(mg.actors["A"].root_prior, mg.actors["A"].root, atol=1e-5)


def test_pose3d_fixture_for_other_modules():
    """The consecutive-frame telephoto pose3d fixture (for pair/contacts tests) is self-consistent."""
    from afv.core.types import CameraTrack

    motion = load_result(FIX / "pose3d_who_am_i.npz")
    cam = load_result(FIX / "camera_who_am_i.npz")
    assert isinstance(motion, BodyMotion) and isinstance(cam, CameraTrack)
    assert motion.skeleton == "soma77" and set(motion.actors) == {"A", "B"}
    assert np.array_equal(motion.frames, np.arange(motion.num_frames)) and motion.num_frames == cam.num_frames
    assert motion.world_shift[1] != 0.0
    sk = get_skeleton("soma77")
    for a in motion.actors.values():
        speed = np.median(np.linalg.norm(np.diff(a.root, axis=0), axis=1)) * 30
        prior = np.median(np.linalg.norm(np.diff(a.root_prior, axis=0), axis=1)) * 30
        assert prior < speed  # GEM-X's world root is smoother than the camera-ray root (telephoto)
        assert abs(np.median(sole_heights(a, sk).min(axis=1))) < 0.25  # floor re-fitted to y = 0


def test_contact_probabilities_saturate_safely():
    p = contact_probabilities(np.array([-1000.0, 0.0, 1000.0]))
    assert np.allclose(p, [0.0, 0.5, 1.0])


def test_body_motion_round_trip(tmp_path, fk):
    motion, _ = body_motion_from_fk({"A": fk, "B": fk}, camera_from_gemx(fk))
    save_result(motion, tmp_path / "m.npz")
    back = load_result(tmp_path / "m.npz")
    assert isinstance(back, BodyMotion) and back.skeleton == "soma77" and back.fps == 30.0
    assert set(back.actors) == {"A", "B"}
    for f in ("rot", "root", "conf", "contact", "rest_offsets"):
        assert np.array_equal(getattr(back.actors["B"], f), getattr(motion.actors["B"], f))


@pytest.mark.slow
@pytest.mark.skipif(not P0_FK, reason="P0 FK NPZs not present (~/afv/runs/afv_c_fk)")
@pytest.mark.parametrize("path", P0_FK, ids=lambda p: Path(p).stem)
def test_fk_reproduces_soma_on_full_p0_runs(path):
    fk = SomaFK.load(path)
    cam = camera_from_gemx(fk)
    motion, _ = body_motion_from_fk({"A": fk}, cam)
    in_cam = _in_camera(motion, cam, _fk_positions(motion))
    assert np.linalg.norm(in_cam - fk.incam_pos, axis=-1).max() < 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("clip", ["Close_boxing_short", "UFC_cross", "Who_Am_I", "ref_walk"])
def test_floor_refit_grounds_p0_fighters(clip):
    """Regression (review): with the box-height ground alone, solved feet floated +0.24..+0.48 m.
    After the camera prior + pose3d's floor re-fit, planted feet sit on y = 0."""
    tracks = P0_RUNS / "track" / clip / "tracks.npz"
    runs = [P0_RUNS / "afv_c_fk" / f"{clip}_rank{k}.npz" for k in (0, 1)]
    if not tracks.exists() or not all(r.exists() for r in runs):
        pytest.skip("P0 tracks/FK not present")
    z = np.load(tracks)
    boxes = z["boxes"].astype(np.float64)[:, list(z["fighter_slots"])]
    fks = {r: SomaFK.load(p) for r, p in zip("AB", runs, strict=True)}
    cam, ground = place_on_ground(camera_from_gemx(fks["A"]), boxes)
    if clip == "Close_boxing_short":  # most boxes are cut by the bottom border in this close shot
        assert ground.get("flag") == "few_usable_boxes"
    _, qc = body_motion_from_fk(fks, cam)
    soles = [q["sole_min_height_m_median"] for q in qc["fighters"].values()]
    assert all(abs(v) < 0.25 for v in soles), soles
    assert abs(np.mean(soles)) < 0.12, soles
