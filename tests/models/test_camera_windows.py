"""Round 3: the zoom reaches the CameraTrack anchored at ViPE's calibration frames, and the auto
decisions are windowed: ViPE's rotations are replaced by flow's only where they fail
(``rotation_source`` "vipe+flow"), and the camera root follows GEM-X's rollout only over the frames
where it disagrees with it (CPU, synthetic)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from test_camera_flow import (
    CX,
    CY,
    F,
    N,
    Q,
    _flow_handler,
    _pitch_yaw,
    _r_w_cv,
    _run,
    _standing_fighter,
    _summary,
    _tracks_for,
    _vipe_handler,
)
from test_camera_stages import _ry

from afv.models.camera import (
    camera_to_world_cv,
    clean_mask,
    flow_rotations,
    make_camera_track,
    relative_rotations,
    splice_rotations,
    vipe_focal_frames,
)
from afv.models.gemx import (
    body_motion_from_fk,
    camera_from_gemx,
    decide_root_mode,
    prior_root_in_spans,
    shared_excess,
)


def test_vipe_focal_frames_follow_vipe():
    assert vipe_focal_frames(92) == [0, 30, 60]
    assert vipe_focal_frames(8) == [0, 3, 6]
    assert vipe_focal_frames(1) == [0, 0, 0]


def test_flow_zoom_is_anchored_at_the_reference_frames():
    focal_true = F * np.exp(np.linspace(0.0, np.log(1.5), N))
    rel = np.einsum("ab,nbc->nac", _r_w_cv(0.1)[0].T, _r_w_cv(0.1))
    p0, p1, count = _tracks_for(rel, focal_true, m=80)
    ref = vipe_focal_frames(N)
    f_ref = float(np.exp(np.mean(np.log(focal_true[ref]))))
    fc = flow_rotations(p0, p1, count, f_ref, CX, CY, apply_zoom=True, zoom_reference=ref)
    assert fc.zoom_applied and np.exp(np.mean(np.log(fc.zoom[ref]))) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(fc.focal_px / focal_true, 1.0, atol=3e-3)
    median_anchored = flow_rotations(p0, p1, count, f_ref, CX, CY, apply_zoom=True)
    assert np.median(median_anchored.zoom) == pytest.approx(1.0)  # the default: shot median


def test_splice_takes_the_second_camera_only_on_marked_pairs():
    n = 12
    a = np.stack([_ry(1.0 * t) for t in range(n)]) @ Q  # its own world
    b = np.stack([_ry(-2.0 * t) for t in range(n)])
    use = np.zeros(n - 1, bool)
    use[4:7] = True
    s = splice_rotations(a, b, use)
    assert np.allclose(s[0], a[0])
    rel_s, rel_a, rel_b = relative_rotations(s), relative_rotations(a), relative_rotations(b)
    assert np.allclose(rel_s[use], rel_b[use]) and np.allclose(rel_s[~use], rel_a[~use])


def test_clean_mask_merges_gaps_and_drops_blips():
    m = np.zeros(40, bool)
    m[5:12] = m[14:20] = True  # gap of 2: merged
    m[30:32] = True  # 2 long: dropped
    out = clean_mask(m, merge_gap=3, min_len=4)
    assert np.flatnonzero(out).tolist() == list(range(5, 20))


def _vipe_wrong_on(pairs: slice) -> np.ndarray:
    """ViPE's rotations: the true pan, but panning the wrong way on ``pairs`` (camera-to-V, ViPE world)."""
    true = _r_w_cv(0.1)
    rel = relative_rotations(true)
    wrong = relative_rotations(_r_w_cv(-0.1))
    use = np.zeros(N - 1, bool)
    use[pairs] = True
    r = np.empty_like(true)
    r[0] = true[0]
    for t in range(N - 1):
        r[t + 1] = r[t] @ (wrong[t] if use[t] else rel[t]).T
    return np.einsum("ab,nbc->nac", Q, r)


def test_vipe_stage_replaces_vipe_only_where_it_fails(tmp_path, monkeypatch):
    handlers = {"vipe": _vipe_handler(_vipe_wrong_on(slice(15, 25))), "flow": _flow_handler(np.full(N, F))}
    cam, qc, _ = _run(tmp_path, monkeypatch, handlers, rotation_window=8, min_splice_pairs=4)
    assert qc["rotation_source"] == qc["summary"]["rotation_source"] == "vipe+flow"
    ((s, e),) = qc["flow"]["vipe_disagrees_pairs"]
    assert 11 <= s <= 15 and 24 <= e <= 28
    assert qc["flow"]["spliced_drift_m"] < 0.05  # the shot-level drift came from that window
    pitch, yaw = _pitch_yaw(cam)  # the true camera throughout
    assert np.allclose(yaw, 0.1 * np.arange(N), atol=0.02) and np.allclose(pitch, -8.0, atol=0.05)
    assert qc["flow"]["camera_residual_px"]["p90"] < 0.5


def test_vipe_stage_uses_flow_throughout_when_vipe_fails_on_most_of_the_shot(tmp_path, monkeypatch):
    handlers = {"vipe": _vipe_handler(_vipe_wrong_on(slice(5, 32))), "flow": _flow_handler(np.full(N, F))}
    cam, qc, _ = _run(tmp_path, monkeypatch, handlers, rotation_window=8, min_splice_pairs=4)
    assert qc["rotation_source"] == "flow" and qc["flow"]["vipe_disagrees_fraction"] > 0.5
    _, yaw = _pitch_yaw(cam)
    assert np.allclose(yaw, 0.1 * np.arange(N), atol=0.02)


def test_vipe_stage_splices_nothing_when_vipe_is_right(tmp_path, monkeypatch):
    good = np.einsum("ab,nbc->nac", Q, _r_w_cv(0.1))
    handlers = {"vipe": _vipe_handler(good), "flow": _flow_handler(np.full(N, F))}
    _, qc, _ = _run(tmp_path, monkeypatch, handlers, rotation_window=8, min_splice_pairs=4)
    assert qc["rotation_source"] == "vipe" and qc["flow"]["vipe_disagrees_pairs"] == []


# ---------------------------------------------------------------------------------------------
# pose3d: root_mode="auto" per window
# ---------------------------------------------------------------------------------------------


def test_prior_root_in_spans_is_pinned_where_the_span_meets_the_camera_root():
    n = 50
    t = np.arange(n, dtype=np.float64)
    root = np.c_[0.05 * t + 0.3 * np.sin(t), np.ones(n), -20.0 + 0.01 * t]  # jittery camera root
    prior = np.c_[0.02 * t, np.ones(n), np.full(n, -19.0)]  # smooth rollout, its own offset
    out = prior_root_in_spans(root, prior, [[10, 19], [0, 4], [45, 49]])
    keep = np.r_[5:10, 20:45]
    assert np.array_equal(out[keep], root[keep])
    v = np.diff(out[9:21], axis=0)  # inside: the rollout's velocity plus one constant correction
    assert np.allclose(v, v[0])
    assert np.allclose(np.diff(out[:6], axis=0), np.diff(prior[:6], axis=0))  # start: anchored at 5
    assert np.allclose(np.diff(out[44:], axis=0), np.diff(prior[44:], axis=0))  # end: anchored at 44


def _spinning_between(cam, deg_per_frame: float, start: int, stop: int):
    """``cam`` plus a pan the image does not show, only on frames start..stop (a local camera failure)."""
    r, c = camera_to_world_cv(cam)
    ang = deg_per_frame * np.clip(np.arange(cam.num_frames) - start, 0, stop - start)
    spin = np.stack([_ry(a) for a in ang])
    return make_camera_track(
        np.einsum("tab,tbc->tac", spin, r),
        c,
        cam.focal_px,
        width=cam.width,
        height=cam.height,
        cx=cam.cx,
        cy=cam.cy,
    )


def test_auto_root_mode_falls_back_locally_where_the_camera_fails():
    n = 150
    a = _standing_fighter(n)  # ~22 m away, GEM-X's rollout stands still
    fks = {"A": a, "B": dataclasses.replace(a, incam_pos=a.incam_pos * 0.9)}
    good = camera_from_gemx(a)
    bad = _spinning_between(good, 0.12, 60, 100)  # ~1.4 m/s of false slide over 40 frames
    used, ev = decide_root_mode(fks, bad)
    assert used == "camera" and len(ev["local_fallback_spans"]) == 1
    ((s, e),) = ev["local_fallback_spans"]
    assert 45 <= s <= 62 and 98 <= e <= 118
    motion, qc = body_motion_from_fk(fks, bad, root_mode="auto", refit_floor=False)
    ref, _ = body_motion_from_fk(fks, bad, root_mode="camera", refit_floor=False)
    for role, act in motion.actors.items():
        root, cam_root, prior = act.root, ref.actors[role].root, act.root_prior
        outside = np.r_[0:s, e + 1 : n]
        assert np.allclose(root[outside], cam_root[outside], atol=1e-5)
        inside = slice(s, e + 1)
        slide_cam = np.linalg.norm(np.diff(cam_root[inside], axis=0), axis=1).sum()
        slide = np.linalg.norm(np.diff(root[inside] - prior[inside], axis=0), axis=1)
        assert slide.max() < 0.05 * slide_cam  # only the constant end correction remains
        assert qc["fighters"][role]["local_root_fallback_spans"] == [[s, e]]
    assert qc["root_mode"]["used"] == "camera" and qc["root_mode"]["local_fallback_spans"] == [[s, e]]
    assert "root_local_fallback_gemx_world" in _summary(motion, qc)["flags"]


def test_auto_root_mode_keeps_the_camera_root_without_a_failure():
    a = _standing_fighter(150)
    used, ev = decide_root_mode({"A": a}, camera_from_gemx(a))
    assert used == "camera" and ev["local_fallback_spans"] == [] and ev["disagreeing_fraction"] == 0.0


def test_shared_excess_needs_every_fighter_to_slide_by_the_same_angle():
    n = 40
    one = lambda v: np.full(n, float(v))  # noqa: E731
    # ref walk 001, frames 140-170: A out-runs its rollout, B does not (GEM-X under-moves A)
    assert not shared_excess([one(1.4), one(0.25)], [one(0.4), one(0.2)], [one(27), one(26)]).any()
    # a camera error: the same angular excess at both distances
    assert shared_excess([one(1.4), one(0.7)], [one(0.2), one(0.1)], [one(24), one(12)]).all()
    assert shared_excess([one(1.4)], [one(0.2)], [one(24)]).all()  # one fighter: no cross-check


def _crossing(fk, metres_per_s: float):
    """``fk`` walking across the image (camera x) while GEM-X's rollout stands still."""
    n = fk.num_frames
    shift = np.zeros((n, 1, 3))
    shift[:, 0, 0] = metres_per_s * np.arange(n) / 30.0
    return dataclasses.replace(fk, incam_pos=fk.incam_pos + shift)


def test_a_rollout_that_under_moves_is_not_blamed_on_the_camera():
    a = _standing_fighter(150)
    cam = camera_from_gemx(a)  # still camera: its rotation cannot slide anyone
    fks = {"A": _crossing(a, 1.4), "B": dataclasses.replace(a, incam_pos=a.incam_pos * 0.9)}
    used, ev = decide_root_mode(fks, cam)
    assert used == "camera" and ev["local_fallback_spans"] == [] and ev["not_the_camera"]
    assert ev["fighters"]["A"]["camera_m_s"] > 1.2
    # the same with a camera that pans: the excess is A's alone
    used, ev = decide_root_mode(fks, _spinning_between(cam, 0.03, 0, 150))
    assert used == "camera" and ev["local_fallback_spans"] == []


def _panned(cam, deg_per_frame: np.ndarray):
    """``cam`` turned about world y by the cumulative per-frame angles (an estimated pan)."""
    r, c = camera_to_world_cv(cam)
    ang = np.r_[0.0, np.cumsum(deg_per_frame)[:-1]]
    spin = np.stack([_ry(a) for a in ang])
    return make_camera_track(
        np.einsum("tab,tbc->tac", spin, r),
        c,
        cam.focal_px,
        width=cam.width,
        height=cam.height,
        cx=cam.cx,
        cy=cam.cy,
    )


def _seen_from_a_panning_camera(fk, deg_per_frame: float):
    """A world-static fighter in the camera frames of a camera truly panning ``deg_per_frame``."""
    inv = np.stack([_ry(-deg_per_frame * t) for t in range(fk.num_frames)])
    return dataclasses.replace(fk, incam_pos=np.einsum("tab,tjb->tja", inv, fk.incam_pos))


@pytest.mark.parametrize("speed", [1.0, 2.0])
def test_a_camera_panning_the_wrong_way_falls_back(speed):
    """Round 2's ViPE on Who Am I: the estimate pans the other way. The slide is |w_est| + |w_true|,
    beyond the estimate's own rotation, so no bound on it may veto the fallback."""
    n, rate = 150, 0.12
    a = _standing_fighter(n)
    fks = {
        r: _seen_from_a_panning_camera(f, rate)
        for r, f in (("A", a), ("B", dataclasses.replace(a, incam_pos=a.incam_pos * 0.9)))
    }
    good = camera_from_gemx(a)
    used, ev = decide_root_mode(fks, _panned(good, np.full(n, -rate)))  # the right estimate
    assert used == "camera" and ev["camera_m_s"] < 0.2 and ev["local_fallback_spans"] == []
    used, ev = decide_root_mode(fks, _panned(good, np.full(n, speed * rate)))  # the wrong way
    assert used == "gemx_world" and ev["camera_m_s"] > 2.0 and "not_the_camera" not in ev


def test_a_camera_panning_the_wrong_way_for_a_while_falls_back_there():
    n, rate = 150, 0.12
    a = _standing_fighter(n)
    fks = {
        r: _seen_from_a_panning_camera(f, rate)
        for r, f in (("A", a), ("B", dataclasses.replace(a, incam_pos=a.incam_pos * 0.9)))
    }
    est = np.full(n, -rate)
    est[60:100] = rate  # wrong way on frames 60-100 only
    used, ev = decide_root_mode(fks, _panned(camera_from_gemx(a), est))
    assert used == "camera" and len(ev["local_fallback_spans"]) == 1
    ((s, e),) = ev["local_fallback_spans"]
    assert 45 <= s <= 62 and 98 <= e <= 118


def test_vipe_translation_is_dropped_when_the_zoom_is_applied(tmp_path, monkeypatch):
    """ViPE reads a zoom as a dolly: keeping its translations with the zoom would count it twice."""
    focal_true = F * np.exp(np.linspace(0.0, np.log(1.5), N))
    f_vipe = float(np.exp(np.mean(np.log(focal_true[vipe_focal_frames(N)]))))
    good = np.einsum("ab,nbc->nac", Q, _r_w_cv(0.1))
    handlers = {"vipe": _vipe_handler(good, focal=f_vipe), "flow": _flow_handler(focal_true)}
    cam, qc, _ = _run(tmp_path, monkeypatch, handlers, use_translation=True)
    assert qc["rotation_source"] == "vipe" and qc["flow"]["zoom_applied"]
    assert "translation_dropped" in qc["summary"]["flags"] and np.allclose(cam.cam_pos, cam.cam_pos[0])
