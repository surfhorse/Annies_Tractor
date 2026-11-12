"""pose3d: frames where a fighter's 2D evidence is weak (truncated or occluded body, interpolated
box) are published as conf 0 on every joint and listed in qc.json; pose3d itself does not move the
body there (the pair stage's depth solve handles them, tests/motion/test_pair_weak_2d.py)."""

from __future__ import annotations

import dataclasses

import numpy as np
from test_camera_flow import _standing_fighter, _summary

from afv.models.gemx import DEPTH_BODY_JOINTS, body_motion_from_fk, camera_from_gemx, soma77, weak_2d_frames


def _weakened(fk, frames: slice, joints=None, depth_factor: float = 1.4):
    """``joints`` (default: all) unseen on ``frames``, and GEM-X guessing the body further away."""
    sk = soma77()
    kp = np.array(fk.kp2d)
    cols = list(range(kp.shape[1])) if joints is None else [sk.index(n) for n in joints]
    for f in range(*frames.indices(fk.num_frames)):
        kp[f, cols, 2] = 0.1  # not seen
    pos = np.array(fk.incam_pos)
    pos[frames] *= depth_factor  # and GEM-X guesses it further away (same image)
    return dataclasses.replace(fk, kp2d=kp, incam_pos=pos)


def test_weak_frames_are_torso_and_legs_unseen_or_unobserved():
    fk = _standing_fighter(60)
    assert not weak_2d_frames(fk).any()
    assert np.flatnonzero(weak_2d_frames(_weakened(fk, slice(20, 30)))).tolist() == list(range(20, 30))
    # hands and forearms unseen (a guard, a clinch): the body's size is still seen
    arms = ("LeftForeArm", "RightForeArm", "LeftHand", "RightHand", "LeftHandIndex1", "RightHandIndex1")
    assert not weak_2d_frames(_weakened(fk, slice(20, 30), arms)).any()
    # a box cut by the image border is not weak by itself (Close boxing: the bottom border, body seen)
    boxes = np.tile([100.0, 50.0, 400.0, 719.0], (60, 1))
    assert not weak_2d_frames(dataclasses.replace(fk, boxes_xyxy=boxes)).any()
    observed = np.ones(60, bool)
    observed[40:50] = False
    assert weak_2d_frames(dataclasses.replace(fk, observed=observed)).sum() == 10


def test_pose3d_publishes_the_weak_mask_and_leaves_the_body():
    fk = _standing_fighter(90)
    cam = camera_from_gemx(fk)
    weak = _weakened(fk, slice(60, 80))
    motion, qc = body_motion_from_fk({"A": weak}, cam, refit_floor=False)
    ref, _ = body_motion_from_fk({"A": dataclasses.replace(weak, kp2d=fk.kp2d)}, cam, refit_floor=False)
    a = motion.actors["A"]
    assert np.array_equal(a.root, ref.actors["A"].root)  # no depth hold in pose3d
    assert np.flatnonzero(a.weak).tolist() == list(range(60, 80))  # the published mask
    assert ref.actors["A"].weak is not None and not ref.actors["A"].weak.any()
    kp_conf = np.clip(weak.kp2d[..., 2], 0.0, 1.0)
    assert np.allclose(a.conf, kp_conf, atol=1e-6)  # conf stays GEM-X's (visible limbs keep weight)
    assert qc["fighters"]["A"]["weak_2d_spans"] == [[60, 79]]
    summary = _summary(motion, qc)
    assert "A_weak_2d_evidence" in summary["flags"] and summary["A"]["weak_2d_frames"] == 20


def test_a_well_fitted_partial_body_is_not_weak():
    """UFC cross B 202-206, partly hidden: 11 joints seen, GEM-X fits them at its usual error. Its body
    size is fixed per shot, so that fit pins the depth: not weak."""
    fk = _standing_fighter(60)
    sk = soma77()
    kp = np.array(fk.kp2d)
    hidden = [sk.index(n) for n in DEPTH_BODY_JOINTS[5:]]  # arms and legs unseen, torso and head seen
    kp[20:30, hidden, 2] = 0.1
    seen_left = (kp[25, :, 2] > 0.5).sum()
    assert seen_left >= 8
    assert not weak_2d_frames(dataclasses.replace(fk, kp2d=kp)).any()
    off = np.array(kp)
    off[20:30, :, :2] += 60.0  # the same seen joints, but GEM-X's body does not fit them
    assert weak_2d_frames(dataclasses.replace(fk, kp2d=off)).sum() == 10


def test_the_weak_mask_round_trips_and_names_its_strip_spans(tmp_path):
    from afv.core.types import load_result, save_result
    from afv.take import qc_curves, qc_spans

    fk = _standing_fighter(90)
    motion, _ = body_motion_from_fk({"A": _weakened(fk, slice(60, 80))}, camera_from_gemx(fk))
    save_result(motion, tmp_path / "m.npz")
    back = load_result(tmp_path / "m.npz")
    assert back.actors["A"].weak.dtype == bool and np.array_equal(
        back.actors["A"].weak, motion.actors["A"].weak
    )
    curves, _ = qc_curves(back)
    spans = [s for s in qc_spans(curves, back.frames) if s["kind"] == "weak_2d"]
    assert [(s["start"], s["end"], s["label"]) for s in spans] == [(60, 79, "A weak 2D")]
