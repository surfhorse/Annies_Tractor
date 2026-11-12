"""Smoke tests of the CPU noise harness (``tests/motion/noise_harness.py``, round 5 track M): the
noise model and the real-repeat mixes are identities at zero, measure what they are given, and the
whole chain runs on synthetic fighters."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from motion_synth import FOOT_CHANNELS, body, model_probs, telephoto_camera, walk
from noise_harness import _legendre, _rays, _trend, combine, measure_pair, modes, perturb, run_chain


def _scene():
    rng = np.random.default_rng(0)
    a, masks = walk(seconds=2.0, origin=(-0.6, 0.0))
    b, masks_b = walk(seconds=2.0, origin=(0.6, 0.3), heading=np.pi)
    a = replace(a, contact=model_probs(masks, rng), root_prior=np.asarray(a.root, np.float32))
    b = replace(b, contact=model_probs(masks_b, rng), root_prior=np.asarray(b.root, np.float32))
    bm = body({"A": a, "B": b}, channels=[*FOOT_CHANNELS, "l_wrist", "r_wrist"])
    return bm, telephoto_camera(bm.num_frames)


def test_trend_recovers_a_drift():
    n = 90
    d = 0.01 + 0.02 * np.linspace(-1, 1, n)
    coef, resid, _ = _trend(d)
    assert np.allclose(coef[:2], [0.01, 0.02], atol=1e-9) and resid < 1e-9
    assert _legendre(n).shape == (n, 4)


def test_modes_split_on_gaps_not_on_continuous_values():
    assert [len(g) for g in modes([1.0, 1.001, 1.002, 1.003, 1.2, 1.201, 1.202], 0.05)] == [4, 3]
    assert len(modes(list(np.linspace(1.0, 1.2, 21)), 0.005)) == 1  # evenly spread: one mode


def test_zero_noise_and_zero_mix_are_identities():
    bm, cam = _scene()
    other_bm = replace(
        bm, actors={r: replace(a, root=np.asarray(a.root) + 0.01) for r, a in bm.actors.items()}
    )
    same, _ = combine((bm, cam), [(other_bm, cam)], [0.0])
    half, _ = combine((bm, cam), [(other_bm, cam)], [0.5])
    for r, a in bm.actors.items():
        assert np.allclose(same.actors[r].root, a.root, atol=1e-6)
        assert np.allclose(half.actors[r].root, np.asarray(a.root) + 0.005, atol=1e-6)
        assert np.allclose(same.actors[r].contact, a.contact, atol=1e-5)
    model = measure_pair((bm, cam), (other_bm, cam))["model"]
    still, cam2 = perturb(bm, cam, model, np.random.default_rng(1), scale=0.0)
    for r, a in bm.actors.items():
        assert np.allclose(still.actors[r].root, a.root, atol=1e-6)
    assert np.allclose(cam2.cam_pos, cam.cam_pos, atol=1e-6)


def test_measured_depth_noise_is_along_the_ray():
    bm, cam = _scene()
    moved = {}
    for r, a in bm.actors.items():
        u = _rays(bm, cam, a.root)
        moved[r] = replace(a, root=(np.asarray(a.root) + 0.02 * u).astype(np.float32))
    m = measure_pair((bm, cam), (replace(bm, actors=moved), cam))["model"]["actors"]["A"]
    assert abs(m["root_along_trend_m"][0] - 0.02) < 1e-4
    assert max(m["root_across_rms_m"]) < 1e-4


def test_chain_runs_and_reports_the_golden_keys():
    bm, cam = _scene()
    s = run_chain(bm, cam)
    for k in (
        "pair.distance_median_m",
        "contacts.A.contact_frames",
        "smooth.B.root_path_m",
        "smooth.A.strikes",
    ):
        assert k in s and np.isfinite(float(s[k]))
