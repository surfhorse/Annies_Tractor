"""The planted feet as a check on the gravity re-level (``afv.models.gemx.feet_tilt``,
``afv.models.camera.feet_refute_relevel``): a camera pitch error tilts the solved floor about the
camera, so the soles of a fighter walking away rise or sink with distance (CPU, synthetic)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from test_camera_flow import _standing_fighter, _summary
from test_camera_stages import _rx

from afv.models.camera import (
    FEET_TILT_BIAS_DEG,
    camera_to_world_cv,
    feet_data_malformed,
    feet_refute_relevel,
    gravity_disagrees,
    make_camera_track,
    relevel_fraction,
    relevel_override,
)
from afv.models.gemx import body_motion_from_fk, camera_from_gemx


def _walking_away(n: int = 90, metres: float = 6.0):
    """A planted fighter sliding straight away from the camera over the floor (soles level in the true
    world), and that true camera."""
    fk = _standing_fighter(n)
    cam = camera_from_gemx(fk)
    r, _ = camera_to_world_cv(cam)
    fwd = r[0][:, 2].copy()
    fwd[1] = 0.0
    fwd /= np.linalg.norm(fwd)
    step = np.linspace(0.0, metres, n)[:, None] * fwd  # world
    incam = fk.incam_pos + np.einsum("tba,tb->ta", r, step)[:, None]  # into each camera frame
    logits = np.array(fk.contact_logits)
    logits[:, :4] = 6.0  # planted
    return dataclasses.replace(fk, incam_pos=incam, contact_logits=logits), cam


def _pitched(cam, deg: float):
    r, c = camera_to_world_cv(cam)
    return make_camera_track(
        r @ _rx(deg), c, cam.focal_px, width=cam.width, height=cam.height, cx=cam.cx, cy=cam.cy
    )


@pytest.mark.parametrize("err", [2.0, -2.0])
def test_feet_tilt_measures_the_pitch_error_with_gemx_gravity_sign(err):
    fk, cam = _walking_away()
    _, qc = body_motion_from_fk({"A": fk}, cam)
    assert abs(qc["gravity"]["feet_tilt_deg"]) < 0.2 and qc["gravity"]["feet_distance_spread_m"] > 4
    _, qc = body_motion_from_fk({"A": fk}, _pitched(cam, err))
    g = qc["gravity"]
    assert abs(g["feet_tilt_deg"]) == pytest.approx(2.0, abs=0.3)
    assert abs(g["pitch_difference_deg"]) == pytest.approx(2.0, abs=0.3)
    assert np.sign(g["feet_tilt_deg"]) == np.sign(g["pitch_difference_deg"])  # one sign convention


def test_feet_tilt_needs_planted_feet_over_a_range_of_distances():
    fk, cam = _walking_away(metres=0.3)
    _, qc = body_motion_from_fk({"A": fk}, cam)
    assert qc["gravity"]["feet_tilt_deg"] is None and "feet_tilt_note" in qc["gravity"]
    still = dataclasses.replace(fk, contact_logits=np.full_like(fk.contact_logits, -6.0))
    _, qc = body_motion_from_fk({"A": still}, cam)
    assert qc["gravity"]["feet_tilt_deg"] is None and qc["gravity"]["feet_tilt_frames"] == 0


def _gravity(pitch, height, ci, per=None):
    return {
        "pitch_difference_deg": pitch,
        "height_change_if_gemx_m": height,
        "feet_tilt_deg": None if ci is None else 0.5 * (ci[0] + ci[1]),
        "feet_tilt_ci_deg": ci,
        "feet_tilt_per_fighter_deg": per or {},
        "gemx_up_cam0": [0.0, -0.99, -0.14],
    }


def test_feet_limit_the_relevel_to_what_they_do_not_contradict():
    """Media Samples (round 3) before any re-level: GEM-X's pitch difference p clipped to the feet's
    interval widened by FEET_TILT_BIAS_DEG."""
    wai = _gravity(4.78, 1.8, [1.13, 2.32], {"A": 2.05, "B": 1.56})  # camera 0.42 m below the floor
    assert gravity_disagrees(wai) and relevel_fraction(wai) == pytest.approx(
        (2.32 + FEET_TILT_BIAS_DEG) / 4.78, abs=1e-3
    )
    rw = _gravity(3.58, 1.6, [-1.89, 1.34], {"A": -4.02, "B": 1.69})
    assert gravity_disagrees(rw) and 0.5 < relevel_fraction(rw) < 1.0
    cb = _gravity(4.26, 1.2, [-2.33, -1.64], {"A": -1.06, "B": -2.19})  # both lean the other way
    assert relevel_fraction(cb) == 0.0 and feet_refute_relevel(cb) and not gravity_disagrees(cb)
    assert relevel_override({"gravity": cb}) is None
    assert relevel_fraction(_gravity(4.0, 1.2, [-3.0, 6.0])) == 1.0  # wide: GEM-X's in full
    assert relevel_fraction(_gravity(4.3, 1.2, None)) == 1.0  # no feet measurement: GEM-X alone
    assert relevel_fraction(_gravity(4.3, 1.2, [-2.0, -1.0], {"A": -1.5, "B": 4.5})) == 1.0  # B supports it
    assert relevel_fraction(_gravity(-4.0, -1.0, [-5.0, -2.0])) == 1.0  # the other direction, agreeing
    assert relevel_fraction(_gravity(-4.0, -1.0, [2.0, 3.0], {"A": 2.5})) == 0.0  # ... and refuting
    assert relevel_fraction(_gravity(1.0, 0.2, [-3.0, -2.0])) == 0.0  # GEM-X asks for nothing


def test_the_limited_relevel_is_idempotent_and_goes_part_of_the_way():
    p, ci = 10.0, [1.0, 3.0]
    f = relevel_fraction(_gravity(p, 3.0, ci))
    assert f == pytest.approx((3.0 + FEET_TILT_BIAS_DEG) / p)
    after = _gravity(p * (1 - f), 3.0, [c - f * p for c in ci])  # the feet and GEM-X after the re-level
    assert relevel_fraction(after) == 0.0  # nothing more to do (and not a reversal)
    g = _gravity(p, 3.0, ci)
    cam_up = np.array([0.0, -1.0, 0.0])
    gemx_up = np.array([0.0, -np.cos(np.radians(p)), -np.sin(np.radians(p))])
    g |= {"camera_up_cam0": cam_up.tolist(), "gemx_up_cam0": gemx_up.tolist()}
    up = np.asarray(relevel_override({"gravity": g})["up_cam0"])
    assert np.degrees(np.arccos(up @ cam_up)) == pytest.approx(f * p, abs=1e-3)


def test_summary_flags_a_refuted_relevel():
    fk, cam = _walking_away()
    bad = _pitched(cam, 3.6)  # GEM-X (the true up) says re-level; so do the feet
    motion, qc = body_motion_from_fk({"A": fk}, bad)
    g = qc["gravity"]
    lo, hi = g["feet_tilt_ci_deg"]
    assert np.sign(lo) == np.sign(hi) == np.sign(g["pitch_difference_deg"])  # significant agreement
    assert "gravity_pitch_disagrees" in _summary(motion, qc)["flags"]
    g["feet_tilt_ci_deg"] = [-hi, -lo]  # feet significantly leaning the other way
    g["feet_tilt_per_fighter_deg"] = {"A": -g["feet_tilt_per_fighter_deg"]["A"]}
    flags = _summary(motion, qc)["flags"]
    assert "gravity_relevel_refuted_by_feet" in flags and "gravity_pitch_disagrees" not in flags


def test_feet_are_weighted_by_their_keypoint_confidence():
    fk, cam = _walking_away()
    _, qc = body_motion_from_fk({"A": fk}, _pitched(cam, 2.0))
    kp = np.array(fk.kp2d)
    kp[:, :, 2] = 0.0  # feet not seen: no planted weight
    _, qc0 = body_motion_from_fk({"A": dataclasses.replace(fk, kp2d=kp)}, _pitched(cam, 2.0))
    assert qc["gravity"]["feet_tilt_deg"] is not None and qc0["gravity"]["feet_tilt_deg"] is None
    assert qc0["gravity"]["feet_tilt_frames"] == 0


def test_unreadable_feet_and_an_unknown_camera_up_never_give_a_full_relevel():
    bad = _gravity(4.3, 1.2, [float("nan"), 1.0])
    assert (
        feet_data_malformed(bad)
        and relevel_fraction(bad) == 0.0
        and relevel_override({"gravity": bad}) is None
    )
    bad2 = _gravity(4.3, 1.2, [0.5, 1.0], {"A": "x"})
    assert feet_data_malformed(bad2) and relevel_fraction(bad2) == 0.0
    limited = _gravity(4.78, 1.8, [1.13, 2.32])  # feet limit it to 80 %, but no camera_up_cam0 (old QC)
    assert 0.0 < relevel_fraction(limited) < 1.0 and relevel_override({"gravity": limited}) is None


def test_feet_still_disagreeing_after_a_relevel_is_flagged():
    fk, cam = _walking_away()
    motion, qc = body_motion_from_fk({"A": fk}, cam)
    qc["gravity"] |= {"relevelled": True, "feet_tilt_ci_deg": [-4.45, -1.2]}
    assert "gravity_relevel_feet_still_disagree" in _summary(motion, qc)["flags"]
    qc["gravity"]["feet_tilt_ci_deg"] = [-2.0, 0.5]
    assert "gravity_relevel_feet_still_disagree" not in _summary(motion, qc)["flags"]
    qc["gravity"] |= {"relevelled": False, "feet_tilt_ci_deg": [-4.45, -1.2]}
    assert "gravity_relevel_feet_still_disagree" not in _summary(motion, qc)["flags"]
