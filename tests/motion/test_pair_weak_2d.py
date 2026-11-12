"""[Z, round 3] pair: frames published as weak 2D evidence (pose3d's ``ActorMotion.weak``) keep
only an image-error budget in the depth solve: depth there follows GEM-X's rollout and the
neighbouring frames, continuously at the span edges."""

from __future__ import annotations

import numpy as np
from motion_synth import GEMX_CHANNELS, along_ray, body, model_probs, telephoto_camera, walk

from afv.motion.pair import pair_fighters

N = 150
SPAN = slice(60, 80)


def _scene(mark_weak: bool):
    rng = np.random.default_rng(7)
    a, ma = walk(seconds=N / 30, speed=0.3, origin=(-0.7, 0.0), heading=np.pi / 2)
    b, mb = walk(seconds=N / 30, speed=0.3, origin=(0.9, 0.3), heading=-np.pi / 2)
    cam = telephoto_camera(N)
    cp = cam.cam_pos[0]
    dz_a = rng.normal(0, 0.05, N)
    dz_a[SPAN] += np.r_[np.linspace(1.0, 4.0, 10), np.full(10, 4.0)]  # GEM-X guesses on a hidden body
    ab, bb = along_ray(a, cp, dz_a), along_ray(b, cp, rng.normal(0, 0.05, N))
    ab.contact, bb.contact = model_probs(ma, rng), model_probs(mb, rng)
    n_j = a.rot.shape[1]
    for act, clean in ((ab, a), (bb, b)):
        act.conf = np.ones((N, n_j), np.float32)
        act.root_prior = (clean.root + np.array([0.5, 0.0, -0.8])).astype(np.float32)
    if mark_weak:
        ab.weak = np.zeros(N, bool)
        ab.weak[SPAN] = True  # pose3d's ActorMotion.weak
    return a, body({"A": ab, "B": bb}, GEMX_CHANNELS), cam


def _depth(root, cp):
    return np.linalg.norm(np.asarray(root, dtype=np.float64) - cp, axis=-1)


def test_weak_frames_follow_the_rollout_without_edge_pops():
    clean, bm, cam = _scene(mark_weak=True)
    cp = cam.cam_pos[0]
    out, _ = pair_fighters(bm, cam)
    err = _depth(out.actors["A"].root, cp) - _depth(clean.root, cp)
    assert np.abs(err[SPAN]).max() < 0.1  # GEM-X's +4 m guess is not followed (measured 0.04 m)
    _, bm_unmarked, _ = _scene(mark_weak=False)
    out_u, _ = pair_fighters(bm_unmarked, cam)
    # unmarked, the robust rows reject most of the guess, but not all of it (measured 0.2 m)
    assert np.abs((_depth(out_u.actors["A"].root, cp) - _depth(clean.root, cp))[SPAN]).max() > 0.15
    # span edges: root speed and acceleration within 1.5x of the neighbouring frames'
    root = np.asarray(out.actors["A"].root, dtype=np.float64)
    speed = np.linalg.norm(np.diff(root, axis=0), axis=1) * 30.0
    acc = np.linalg.norm(np.diff(root, 2, axis=0), axis=1) * 900.0
    for edge in (SPAN.start, SPAN.stop):
        near = np.r_[edge - 12 : edge - 3, edge + 3 : edge + 12]
        assert speed[edge - 2 : edge + 2].max() <= 1.5 * speed[near].max() + 0.05
        assert acc[edge - 2 : edge + 1].max() <= 1.5 * acc[near - 1].max() + 0.5
