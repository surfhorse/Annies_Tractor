"""Retiming: speed factor, resampling, slerp, contacts by threshold."""

from __future__ import annotations

import numpy as np
import pytest
from motion_synth import GEMX_CHANNELS, SK, body, corrupt, model_probs, walk

from afv.motion.contacts import clean_contacts
from afv.motion.fk import forward_kinematics
from afv.motion.retime import resample_plants, retime, sample_times
from afv.motion.rotations import slerp
from afv.qc.metrics import per_frame_metrics, summarise


def test_sample_times():
    assert np.allclose(sample_times(31, 30, 30, 1.0), np.arange(31))
    assert len(sample_times(31, 30, 60, 1.0)) == 61
    t = sample_times(31, 30, 30, 0.5)  # half speed: twice as many frames
    assert len(t) == 61 and np.allclose(t[1], 0.5) and t[-1] == 30
    assert len(sample_times(31, 30, 24, 1.0)) == 25
    with pytest.raises(ValueError):
        sample_times(10, 30, 30, 0.0)


def test_identity_and_upsampled_midpoints():
    clean, _ = walk(seconds=1.0)
    bm = body({"A": clean})
    same = retime(bm, 30.0, 1.0, hold_contacts=False)
    assert same.num_frames == bm.num_frames
    assert np.allclose(np.abs(np.sum(same.actors["A"].rot * clean.rot, -1)), 1.0, atol=1e-6)
    up = retime(bm, 60.0, 1.0, hold_contacts=False)
    assert up.fps == 60.0 and up.num_frames == 2 * bm.num_frames - 1
    a = up.actors["A"]
    mid = slerp(clean.rot[3], clean.rot[4], 0.5)
    assert np.allclose(np.abs(np.sum(a.rot[7] * mid, -1)), 1.0, atol=1e-6)
    assert np.allclose(a.root[7], 0.5 * (clean.root[3] + clean.root[4]), atol=1e-6)
    assert np.allclose(up.timestamps[7], 3.5 / 30)
    assert np.array_equal(up.frames, np.arange(up.num_frames))


def test_slow_motion_keeps_contacts_binary_and_planted():
    rng = np.random.default_rng(0)
    clean, masks = walk(seconds=3.0)
    bad = corrupt(clean, rng, floor_shift=0.1)
    bad.contact = model_probs(masks, rng)
    cont, rep = clean_contacts(body({"A": bad}, GEMX_CHANNELS))
    slow = retime(cont, 60.0, 0.5)  # default: only the foot plant channels are binary
    assert slow.num_frames == 4 * (cont.num_frames - 1) + 1
    c = slow.actors["A"].contact
    assert set(np.unique(c[:, :4])) <= {0.0, 1.0}
    assert np.allclose(c[:, 4:], 0.02, atol=1e-3)  # wrist probabilities stay probabilities
    s = summarise(per_frame_metrics(slow, floor=rep.floor))["A"]
    # Speeds are per playback second: 3 cm/s at half speed is still the NFR-3 bar.
    assert s["mean_contact_foot_speed_cm_s"] <= 3.0
    a = slow.actors["A"]
    assert np.isfinite(forward_kinematics(SK, a.rot, a.root)[0]).all()


def test_retime_same_rate_is_identity_with_hold():
    rng = np.random.default_rng(1)
    clean, masks = walk(seconds=2.0)
    bad = corrupt(clean, rng, floor_shift=0.1)
    bad.contact = model_probs(masks, rng)
    cont, _ = clean_contacts(body({"A": bad}, GEMX_CHANNELS))
    same = retime(cont, 30.0, 1.0)
    p0 = forward_kinematics(SK, cont.actors["A"].rot, cont.actors["A"].root)[0]
    p1 = forward_kinematics(SK, same.actors["A"].rot, same.actors["A"].root)[0]
    assert np.linalg.norm(p1 - p0, axis=-1).max() < 1e-3


def test_plants_resampled_only_between_two_planted_frames():
    """Round 2 fix: a sample half-way between a planted and a lifted source frame is swing, not a
    plant (thresholding the interpolated value at 0.5 made planted feet jump at every plant edge and
    across one-frame gaps at 60 fps)."""
    src = np.array([1, 1, 0, 1, 1, 1, 0], dtype=np.float64)[:, None]  # one-frame gap at 2, lifted at 6
    at60 = resample_plants(src, sample_times(7, 30, 60))[:, 0]
    #               0  .5  1  1.5  2  2.5  3  3.5  4  4.5  5  5.5  6
    want = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    assert np.array_equal(at60, want)
    at24 = resample_plants(src, sample_times(7, 30, 24))[:, 0]  # 0, 1.25, 2.5, 3.75, 5
    assert np.array_equal(at24, np.array([1, 0, 0, 1, 1], dtype=bool))
    slow = resample_plants(src, np.array([3.0, 3.1, 4.9, 5.0, 5.0001, 5.9]))[:, 0]
    assert np.array_equal(slow, np.array([1, 1, 1, 1, 0, 0], dtype=bool))


def test_retime_upsampled_plants_stay_still():
    rng = np.random.default_rng(2)
    clean, masks = walk(seconds=3.0)
    bad = corrupt(clean, rng, floor_shift=0.1)
    bad.contact = model_probs(masks, rng)
    cont, rep = clean_contacts(body({"A": bad}, GEMX_CHANNELS))
    for fps in (60.0, 24.0, 50.0):
        out = retime(cont, fps, 1.0)
        s = summarise(per_frame_metrics(out, floor=rep.floor))["A"]
        assert s["mean_contact_foot_speed_cm_s"] <= 1.0, (fps, s["mean_contact_foot_speed_cm_s"])


def test_resample_plants_guards_non_finite_input():
    src = np.array([1.0, np.nan, 1.0, np.inf])[:, None]
    m = resample_plants(src, np.array([0.0, 0.5, 1.0, 2.0, 2.5, 3.0]))[:, 0]
    assert np.array_equal(m, np.array([1, 0, 0, 1, 0, 0], dtype=bool))  # non-finite = not planted
    for bad in (np.array([0.0, np.nan]), np.array([-0.5]), np.array([3.5])):
        with pytest.raises(ValueError):
            resample_plants(src, bad)
