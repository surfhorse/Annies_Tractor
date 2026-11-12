"""Camera rotations (+ zoom) from background flow, the vipe stage's choice between flow and ViPE, and
pose3d's zoom depth correction and root/prior consistency QC (CPU, synthetic; no model environments).

Background: on the telephoto Media Samples ViPE's rotations did not explain the background's image
motion (ref walk 3.6x too fast, Who Am I mirrored, Close boxing ~0 while panning and zooming 1.9x),
which made the camera-based root slide across the ray (telephoto root investigation, round 2).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_camera_stages import CV_TO_GL, FIX, FakeWorkers, _ctx, _pitch_yaw, _rx, _ry, _tracks

import afv.models.gemx as gemx_mod
from afv.core.stage import Ctx
from afv.core.types import ActorMotion, BodyMotion, Tracks
from afv.models import camera as camera_mod
from afv.models.camera import (
    CameraParams,
    CameraStatic,
    CameraVipe,
    camera_to_world_cv,
    fit_rotation_zoom,
    flow_rotations,
    gemx_provided_camera,
    make_camera_track,
    project,
    rotation_residual_px,
    unproject,
    vipe_disagreement,
    vipe_focal_frames,
)
from afv.models.gemx import (
    Pose3dGemx,
    Pose3dParams,
    SomaFK,
    actor_from_fk,
    body_motion_from_fk,
    camera_from_gemx,
    camera_root_lateral,
    incam_to_world,
    lateral_disagrees,
    lateral_speed,
    pose3d_summary,
    root_lateral_speeds,
    zoom_corrected_incam,
)

W, H, CX, CY, F = 1280, 720, 640.0, 360.0, 6000.0


def _project(rays: np.ndarray, f: float) -> np.ndarray:
    return np.c_[CX + f * rays[:, 0] / rays[:, 2], CY + f * rays[:, 1] / rays[:, 2]]


def _tracks_for(r_c2v: np.ndarray, focal: np.ndarray, m: int = 150, noise: float = 0.1, seed: int = 0):
    """Background point pairs [T-1,M,2] seen by a rotating (zooming) camera, with pixel noise."""
    rng = np.random.default_rng(seed)
    n = len(r_c2v)
    p0s = np.full((n - 1, m, 2), np.nan)
    p1s = np.full_like(p0s, np.nan)
    for t in range(n - 1):
        px = np.c_[rng.uniform(50, W - 50, m), rng.uniform(50, H - 50, m)]
        rays = unproject(px, focal[t], CX, CY)
        rel = r_c2v[t + 1].T @ r_c2v[t]  # camera t -> camera t+1
        p0s[t] = px + rng.normal(0, noise, px.shape)
        p1s[t] = _project(rays @ rel.T, focal[t + 1]) + rng.normal(0, noise, px.shape)
    return p0s, p1s, np.full(n - 1, m)


def _pan_tilt(n: int, pan_deg: float = 0.1, tilt_deg: float = 0.02) -> np.ndarray:
    """OpenCV camera-to-V rotations of a telephoto pan (+ slow tilt); camera 0 = identity."""
    return np.stack([_ry(pan_deg * t) @ _rx(tilt_deg * t) for t in range(n)])


def _rel_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rel = np.einsum("tba,tbc->tac", a, b)
    return np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))


def test_fit_rotation_zoom_recovers_pan_and_zoom():
    r_true = _ry(0.12) @ _rx(-0.05)
    rng = np.random.default_rng(1)
    p0 = np.c_[rng.uniform(0, W, 200), rng.uniform(0, H, 200)]
    p1 = _project(unproject(p0, F, CX, CY) @ r_true.T, F * 1.01)
    p1[:20] += rng.uniform(-30, 30, (20, 2))  # a few outliers (a moving person RANSAC missed)
    r, s, res = fit_rotation_zoom(p0, p1, F, CX, CY)
    assert _rel_angle_deg(r[None], r_true[None])[0] < 2e-4
    assert s == pytest.approx(1.01, abs=1e-5) and res < 1e-3
    r2, s2, _ = fit_rotation_zoom(p0[20:], p1[20:], F, CX, CY, zoom=False)
    assert s2 == 1.0 and _rel_angle_deg(r2[None], r_true[None])[0] < 0.05  # zoom read as a little rotation


def test_flow_rotations_chain_a_telephoto_pan_without_zoom():
    n = 60
    r_true = _pan_tilt(n)
    p0, p1, count = _tracks_for(r_true, np.full(n, F))
    count[[10, 11, 40]] = 3  # pairs without enough background: angular velocity interpolated
    fc = flow_rotations(p0, p1, count, F, CX, CY)
    assert not fc.zoom_applied and np.allclose(fc.focal_px, F)
    assert fc.valid.sum() == n - 1 - 3 and np.isnan(fc.residual_px[10])
    assert _rel_angle_deg(fc.r_v_cv, r_true).max() < 0.03  # degrees after 2 s of chaining (5 mm at 10 m)
    assert np.nanmedian(fc.residual_px) < 0.2


def test_flow_rotations_measure_a_zoom_and_keep_the_pan():
    n = 60
    r_true = _pan_tilt(n, pan_deg=0.08)
    focal_true = F * np.exp(np.linspace(0.0, np.log(1.9), n))  # a 1.9x zoom-in (Close boxing)
    p0, p1, count = _tracks_for(r_true, focal_true)
    flagged = flow_rotations(p0, p1, count, float(np.median(focal_true)), CX, CY)  # default: measure only
    assert not flagged.zoom_applied and np.allclose(flagged.focal_px, np.median(focal_true))
    assert np.allclose(flagged.zoom, focal_true / np.median(focal_true), atol=0.02)
    # one focal: angles = image motion / median focal (consistent with GEM-X's K and depth), and
    # the zoom is not read as rotation
    rel = np.einsum("tba,tbc->tac", flagged.r_v_cv[:-1], flagged.r_v_cv[1:])
    pan = np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))
    expect = 0.08 * focal_true[:-1] / np.median(focal_true)
    assert np.allclose(pan, expect, atol=0.004)
    fc = flow_rotations(p0, p1, count, float(np.median(focal_true)), CX, CY, apply_zoom=True)
    assert fc.zoom_applied
    assert np.allclose(fc.focal_px / focal_true, 1.0, atol=2e-3)
    assert np.allclose(fc.zoom, focal_true / np.median(focal_true), atol=2e-3)
    assert _rel_angle_deg(fc.r_v_cv, r_true).max() < 0.03
    off = flow_rotations(p0, p1, count, float(np.median(focal_true)), CX, CY, zoom=False, apply_zoom=True)
    assert not off.zoom_applied and np.allclose(off.focal_px, np.median(focal_true))


def test_rotation_residual_tells_right_from_mirrored_and_too_fast():
    n = 30
    r_true = _pan_tilt(n)
    p0, p1, count = _tracks_for(r_true, np.full(n, F), noise=0.05)
    good = rotation_residual_px(p0, p1, count, r_true, F, CX, CY)
    mirrored = rotation_residual_px(p0, p1, count, np.swapaxes(r_true, 1, 2), F, CX, CY)
    fast = rotation_residual_px(p0, p1, count, _pan_tilt(n, 0.36, 0.072), F, CX, CY)  # ref walk: 3.6x
    assert np.median(good) < 0.15
    assert np.median(mirrored) > 15 and np.median(fast) > 20  # 0.1 deg/frame at 6000 px ~ 10 px


# ---------------------------------------------------------------------------------------------
# The vipe stage chooses its rotation source
# ---------------------------------------------------------------------------------------------

N = 40
PITCH = -8.0
Q = _ry(55) @ _rx(30)  # ViPE's arbitrary world V = Q W


def _r_w_cv(pan_deg: float) -> np.ndarray:
    return np.stack([_ry(pan_deg * t) @ _rx(PITCH) @ CV_TO_GL for t in range(N)])


def _vipe_handler(r_v_cv: np.ndarray, focal: float = F):
    """ViPE output with the given rotations (camera-to-V) and gravity from the TRUE camera."""
    frames = np.array([0, 13, 27, 39])
    r_true_v = np.einsum("ab,nbc->nac", Q, _r_w_cv(0.1))

    def vipe(req):
        c2w = np.tile(np.eye(4), (N, 1, 1))
        c2w[:, :3, :3] = r_v_cv
        intr = np.tile([focal, focal, CX, CY], (N, 1))
        up_cam = np.einsum("nba,b->na", r_true_v[frames], Q @ [0, 1.0, 0])
        np.savez(
            req["out_npz"],
            c2w=c2w,
            intrinsics=intr,
            up_cam=up_cam,
            up_frames=frames,
            up_weight=np.ones(len(frames)),
            up_focal=np.full(len(frames), focal),
        )
        return {"out_npz": req["out_npz"], "frames": N, "gravity": "geocalib"}

    return vipe


def _flow_handler(focal_true: np.ndarray, *, valid_every: int = 1):
    r_true = _r_w_cv(0.1)
    rel = np.einsum("ab,nbc->nac", r_true[0].T, r_true)  # camera t -> camera 0

    def flow(req):
        assert Path(req["boxes_npz"]).exists()  # people are masked
        p0, p1, count = _tracks_for(rel, focal_true, m=80)
        count[(np.arange(N - 1) + 1) % valid_every != 0] = 0  # pairs without background points
        np.savez(req["out_npz"], p0=p0, p1=p1, count=count, static_fraction=np.float32(0.05))
        return {"out_npz": req["out_npz"], "pairs": N - 1}

    return flow


def _run(tmp_path, monkeypatch, handlers, **params):
    fake = FakeWorkers(tmp_path, handlers)

    def call(env, script, request, work_dir, **kw):
        fake.kwargs = getattr(fake, "kwargs", {}) | {request["command"]: kw}
        out = fake(env, script, request, work_dir, **kw)
        return {**out, "frame_count": N} if request["command"] == "normalize" else out  # N frames here

    monkeypatch.setattr(camera_mod, "run_env_worker", call)
    ctx = _ctx(tmp_path)
    cam = CameraVipe().run(ctx, {"detect_track": _tracks(n=N)}, CameraParams(video="/v.mp4", **params))
    return cam, json.loads((ctx.work_dir / "qc.json").read_text()), fake


def test_vipe_stage_replaces_mirrored_vipe_rotations_with_flow(tmp_path, monkeypatch):
    mirrored = np.einsum("ab,nbc->nac", Q, _r_w_cv(-0.1))  # Who Am I: ViPE pans the wrong way
    cam, qc, fake = _run(
        tmp_path, monkeypatch, {"vipe": _vipe_handler(mirrored), "flow": _flow_handler(np.full(N, F))}
    )
    pitch, yaw = _pitch_yaw(cam)
    assert np.allclose(pitch, PITCH, atol=0.01)  # gravity re-expressed in the flow camera's world
    assert np.allclose(yaw, 0.1 * np.arange(N), atol=0.005)
    assert qc["rotation_source"] == "flow" and qc["summary"]["rotation_source"] == "flow"
    assert "vipe_rotation_disagrees" in qc["summary"]["flags"]
    assert qc["flow"]["vipe_residual_px"]["median"] > 10 > 1 > qc["flow"]["residual_px"]["median"]
    assert not qc["flow"]["zoom_applied"] and np.allclose(cam.focal_px, F)
    assert fake.request("flow")["video"] == fake.request("vipe")["video"]


@pytest.mark.parametrize("source", ["auto", "flow"])
def test_vipe_stage_keeps_vipe_rotations_that_explain_the_background(tmp_path, monkeypatch, source):
    good = np.einsum("ab,nbc->nac", Q, _r_w_cv(0.1))  # wide lens: ViPE right
    handlers = {"vipe": _vipe_handler(good), "flow": _flow_handler(np.full(N, F))}
    cam, qc, _ = _run(tmp_path, monkeypatch, handlers, rotation_source=source)
    assert qc["rotation_source"] == ("vipe" if source == "auto" else "flow")
    assert "vipe_rotation_disagrees" not in qc["summary"]["flags"] and qc["flow"]["vipe_explains_background"]
    assert qc["flow"]["vipe_residual_px"]["median"] < 0.5
    pitch, yaw = _pitch_yaw(cam)
    assert np.allclose(pitch, PITCH, atol=0.01) and np.allclose(yaw, 0.1 * np.arange(N), atol=0.005)


@pytest.mark.parametrize(
    ("valid_every", "flag"), [(4, "camera_rotation_unreliable"), (N, "camera_rotation_unverified")]
)
def test_vipe_stage_falls_back_to_vipe_when_flow_lacks_background(tmp_path, monkeypatch, valid_every, flag):
    mirrored = np.einsum("ab,nbc->nac", Q, _r_w_cv(-0.1))
    handlers = {
        "vipe": _vipe_handler(mirrored),
        "flow": _flow_handler(np.full(N, F), valid_every=valid_every),
    }
    cam, qc, _ = _run(tmp_path, monkeypatch, handlers)
    assert qc["rotation_source"] == "vipe" and not qc["flow"]["usable"]
    assert flag in qc["summary"]["flags"]  # 1 pair in 4 shows ViPE is wrong; none cannot say
    _, yaw = _pitch_yaw(cam)
    assert np.allclose(yaw, -0.1 * np.arange(N), atol=0.005)  # ViPE's (wrong) pan, flagged


def test_vipe_stage_survives_a_failing_flow_worker(tmp_path, monkeypatch):
    def broken(req):
        raise RuntimeError("cv2 missing")

    good = np.einsum("ab,nbc->nac", Q, _r_w_cv(0.1))
    cam, qc, _ = _run(tmp_path, monkeypatch, {"vipe": _vipe_handler(good), "flow": broken})
    assert qc["rotation_source"] == "vipe" and "flow_rotation_failed" in qc["summary"]["flags"]
    assert cam.num_frames == N


def test_vipe_stage_rotation_source_vipe_skips_flow(tmp_path, monkeypatch):
    good = np.einsum("ab,nbc->nac", Q, _r_w_cv(0.1))
    _, qc, fake = _run(tmp_path, monkeypatch, {"vipe": _vipe_handler(good)}, rotation_source="vipe")
    assert [r["command"] for _, _, r in fake.calls] == ["normalize", "vipe"]
    assert qc["summary"]["rotation_source"] == "vipe"


def _vipe_focal(focal_true: np.ndarray) -> float:
    """The focal ViPE reports for a zooming shot: GeoCalib's on its calibration frames (frozen)."""
    return float(np.exp(np.mean(np.log(focal_true[vipe_focal_frames(len(focal_true))]))))


@pytest.mark.parametrize("zoom", ["flag", "apply"])
def test_vipe_stage_applies_a_zoom_unless_asked_to_only_flag_it(tmp_path, monkeypatch, zoom):
    focal_true = F * np.exp(np.linspace(0.0, np.log(1.5), N))
    f_vipe = _vipe_focal(focal_true)
    stuck = np.einsum("ab,nbc->nac", Q, _r_w_cv(0.0))  # Close boxing: ViPE hardly rotates
    handlers = {"vipe": _vipe_handler(stuck, focal=f_vipe), "flow": _flow_handler(focal_true)}
    cam, qc, _ = _run(tmp_path, monkeypatch, handlers, zoom=zoom)
    assert ("zoom" in qc["summary"]["flags"]) == (zoom == "flag")
    assert qc["flow"]["zoom_applied"] == qc["summary"]["zoom_applied"] == (zoom == "apply")
    assert qc["summary"]["zoom_range"] == pytest.approx(1.5, abs=0.01)
    if zoom == "apply":  # anchored at ViPE's calibration frames: the true focal per frame
        assert np.allclose(cam.focal_px / focal_true, 1.0, atol=3e-3)
        assert qc["summary"]["focal_px_range"] == pytest.approx([F, 1.5 * F], rel=3e-3)
    else:  # one focal per shot
        assert np.allclose(cam.focal_px, f_vipe)
    _, k = gemx_provided_camera(cam)  # GEM-X gets the camera's K per frame
    assert np.allclose(k[:, 0, 0], cam.focal_px) and np.allclose(k[:, 1, 1], cam.focal_px)
    _, yaw = _pitch_yaw(cam)
    scale = np.ones(N) if zoom == "apply" else focal_true / f_vipe  # one focal: angles in that K
    expect = np.r_[0.0, np.cumsum((0.1 * scale)[:-1])]
    assert np.allclose(yaw, expect, atol=0.02)


# ---------------------------------------------------------------------------------------------
# pose3d: zoom depth correction and the root/prior consistency check
# ---------------------------------------------------------------------------------------------


def test_zoom_correction_keeps_the_image_and_scales_depth():
    fk = SomaFK.load(FIX)
    cam = camera_from_gemx(fk)
    f0 = 0.5 * (fk.K[:, 0, 0] + fk.K[:, 1, 1])
    assert np.array_equal(zoom_corrected_incam(fk, cam), fk.incam_pos)  # GEM-X's own focal: identity
    ratio = np.linspace(0.7, 1.4, fk.num_frames)
    zoomed = make_camera_track(
        *camera_to_world_cv(cam), f0 * ratio, width=cam.width, height=cam.height, cx=cam.cx, cy=cam.cy
    )
    pos = zoom_corrected_incam(fk, zoomed)
    assert np.allclose(pos[:, 0, 2], fk.incam_pos[:, 0, 2] * ratio)
    assert np.allclose(pos[..., :2], fk.incam_pos[..., :2])  # moved along the optical axis only
    _, wpos = incam_to_world(fk, zoomed)
    uv, _ = project(zoomed, wpos[:, :1])
    gemx_uv = fk.incam_pos[:, 0] @ fk.K[0].T
    assert np.allclose(
        uv[:, 0], gemx_uv[:, :2] / gemx_uv[:, 2:], atol=0.1
    )  # root on GEM-X's pixel (its fx != fy by 0.15 px)
    body = np.linalg.norm(wpos[:, 0] - wpos[:, 10], axis=-1)
    assert np.allclose(body, np.linalg.norm(fk.incam_pos[:, 0] - fk.incam_pos[:, 10], axis=-1))  # rigid


def test_lateral_speed_ignores_motion_along_the_ray():
    t = np.arange(61)[:, None]
    centre = np.zeros((61, 3))
    along = np.c_[np.zeros(61), np.zeros(61), -20.0 - 0.05 * t[:, 0]]  # walks straight away from camera
    across = np.c_[0.02 * t[:, 0], np.zeros(61), np.full(61, -20.0)]  # 0.6 m/s across
    assert np.allclose(lateral_speed(along, centre), 0.0, atol=1e-9)
    assert np.allclose(lateral_speed(across, centre), 0.6, atol=1e-3)


def test_root_disagreeing_with_prior_is_flagged():
    n = 90
    t = np.arange(n)[:, None] / 30.0
    prior = np.c_[0.4 * t[:, 0], np.full(n, 1.0), np.full(n, -20.0)]
    cam = make_camera_track(
        np.tile(np.eye(3), (n, 1, 1)), np.zeros(3), np.full(n, F), width=W, height=H, cx=CX, cy=CY
    )
    q = {}
    for name, speed in (("A", 0.45), ("B", 1.4)):  # B: the ViPE ref walk case (1.36 vs 0.43 m/s)
        root = np.c_[speed * t[:, 0], np.full(n, 1.0), np.full(n, -20.0)]
        rot = np.tile([0.0, 0.0, 0.0, 1.0], (n, 77, 1))
        q[name] = root_lateral_speeds(ActorMotion(rot, root, root_prior=prior), cam)
    assert q["A"]["root_lateral_ratio"] == pytest.approx(0.45 / 0.4, abs=0.02)
    assert q["B"]["root_lateral_m_s"] == pytest.approx(1.4, abs=0.01)
    fighters = {
        r: {
            "reprojection_px_median": 10.0,
            "gravity_disagreement_deg_median": 1.0,
            "observed_fraction": 1.0,
            "unobserved_spans": [],
            **q[r],
        }
        for r in q
    }
    motion = BodyMotion(frames=np.arange(n), timestamps=t[:, 0])
    summary = pose3d_summary(motion, {"floor": {"method": "contact_weighted_median"}, "fighters": fighters})
    assert summary["flags"] == ["B_root_disagrees_with_prior"]
    assert summary["B"]["root_prior_lateral_m_s"] == pytest.approx(0.4, abs=0.01)


def test_pose3d_qc_reports_lateral_speeds(tmp_path, monkeypatch):
    fk = SomaFK.load(FIX)
    n = fk.num_frames
    cam = camera_from_gemx(fk)
    boxes = np.stack([np.load(FIX)["boxes_xyxy"]] * 2, axis=1)
    tracks = Tracks(
        frames=np.arange(n),
        timestamps=np.arange(n) / 30.0,
        track_ids=np.array([7, 9]),
        boxes=boxes,
        scores=np.ones((n, 2), np.float32),
        fighters={"A": 7, "B": 9},
    )

    def pose3d(req):
        out = {}
        for f in req["fighters"]:
            d = Path(req["out_dir"]) / f["role"]
            d.mkdir(parents=True)
            shutil.copy(FIX, d / "soma_fk.npz")
            out[f["role"]] = {"soma_fk": str(d / "soma_fk.npz"), "motion_pt": str(d / "motion.pt")}
        return {"frames": n, "fighters": out}

    monkeypatch.setattr(gemx_mod, "run_env_worker", FakeWorkers(tmp_path, {"pose3d": pose3d}))
    ctx = _ctx(tmp_path)
    Pose3dGemx().run(ctx, {"detect_track": tracks, "camera": cam}, Pose3dParams(video="/v.mp4"))
    qc = json.loads((ctx.work_dir / "qc.json").read_text())
    a = qc["fighters"]["A"]
    assert a["root_lateral_m_s"] > 0 and a["root_prior_lateral_m_s"] > 0
    assert a["root_lateral_ratio"] == pytest.approx(
        a["root_lateral_m_s"] / a["root_prior_lateral_m_s"], rel=0.02
    )
    assert "root_lateral_m_s" in qc["summary"]["A"]


def _standing_fighter(n: int = 60) -> SomaFK:
    """The fixture's first frame held for ``n`` frames at 30 Hz (the fixtures are subsampled)."""
    fk = SomaFK.load(FIX.with_name("gemx_who_am_i_rank1.npz"))  # telephoto, ~22 m
    held = {
        k: np.repeat(getattr(fk, k)[:1], n, axis=0)
        for k in (
            "incam_rot",
            "incam_pos",
            "global_rot",
            "global_pos",
            "contact_logits",
            "kp2d",
            "K",
            "R_w2c",
            "observed",
            "boxes_xyxy",
        )
    }
    return dataclasses.replace(fk, **held)


def _rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _spinning(cam, deg_per_frame: float):
    """``cam`` plus a pan that is not in the image (like P0's ViPE on ref walk)."""
    r, c = camera_to_world_cv(cam)
    spin = np.stack([_ry(deg_per_frame * t) for t in range(cam.num_frames)])
    return make_camera_track(
        np.einsum("tab,tbc->tac", spin, r),
        c,
        cam.focal_px,
        width=cam.width,
        height=cam.height,
        cx=cam.cx,
        cy=cam.cy,
    )


def _closer(fk: SomaFK, factor: float) -> SomaFK:
    return dataclasses.replace(fk, incam_pos=fk.incam_pos * factor)


def _summary(motion, qc):
    fighters = {
        r: {
            "reprojection_px_median": 10.0,
            "gravity_disagreement_deg_median": 1.0,
            "observed_fraction": 1.0,
            "unobserved_spans": [],
            **q,
        }
        for r, q in qc["fighters"].items()
    }
    return pose3d_summary(motion, {**qc, "fighters": fighters})


def test_auto_root_mode_is_one_decision_for_the_shot():
    a = _standing_fighter()  # ~22 m away, GEM-X's rollout stands still
    fks = {"A": a, "B": _closer(a, 0.25)}  # B ~5.5 m away: the same camera error slides it 4x less
    good = camera_from_gemx(a)
    motion, qc = body_motion_from_fk(fks, good, root_mode="auto")
    assert qc["root_mode"]["used"] == "camera"
    assert {q["root_mode_used"] for q in qc["fighters"].values()} == {"camera"}
    # 0.09 deg/frame: A slides ~0.9 m/s (alone it would fall back), B ~0.22 m/s (alone it would not)
    bad = _spinning(good, 0.09)
    per = {r: camera_root_lateral(fk, bad) for r, fk in fks.items()}
    assert lateral_disagrees(per["A"]["camera_m_s"], per["A"]["prior_m_s"])
    assert not lateral_disagrees(per["B"]["camera_m_s"], per["B"]["prior_m_s"])
    motion, qc = body_motion_from_fk(fks, bad, root_mode="auto")
    assert qc["root_mode"]["used"] == "gemx_world"
    assert {q["root_mode_used"] for q in qc["fighters"].values()} == {"gemx_world"}  # one world
    for act in motion.actors.values():
        assert np.array_equal(act.root, act.root_prior)
    summary = _summary(motion, qc)
    # the evidence survives the fallback (the result's own root is now the rollout)
    assert summary["A"]["root_lateral_m_s"] == pytest.approx(per["A"]["camera_m_s"])
    root_flags = [f for f in summary["flags"] if "root" in f]  # (B is shrunk: its floor disagrees)
    assert root_flags == ["A_root_disagrees_with_prior", "root_fallback_gemx_world"]
    # a smaller error: A alone would still be flagged, but the shot keeps the camera
    motion, qc = body_motion_from_fk(fks, _spinning(good, 0.06), root_mode="auto")
    assert qc["root_mode"]["used"] == "camera"
    assert [f for f in _summary(motion, qc)["flags"] if "root" in f] == ["A_root_disagrees_with_prior"]


def test_single_fighter_auto_matches_the_shot_decision():
    fk = _standing_fighter()
    bad = _spinning(camera_from_gemx(fk), 0.1)
    actor, info = actor_from_fk(fk, bad, root_mode="auto")
    assert info["root_mode"] == "auto" and info["root_mode_used"] == "gemx_world"
    assert info["lateral_check"]["camera_m_s"] > 0.8 and np.array_equal(actor.root, actor.root_prior)


def test_vipe_disagreement_is_relative_and_sees_drift():
    # review repro: ViPE 15% too fast on telephoto (1.565 px vs flow 0.161 px) was kept
    assert vipe_disagreement(1.565, 0.161, 0.05, 0.02) == ["residual"]
    # UFC cross, wide lens: ViPE 0.52 px vs flow 0.18 px, 0.14 m / 0.055 m/s of drift at 3.5 m: kept
    assert vipe_disagreement(0.52, 0.18, 0.14, 0.055) == []
    assert vipe_disagreement(0.4, 0.2, 0.5, 0.05) == ["drift"]
    assert vipe_disagreement(0.4, 0.2, 0.1, 0.15) == ["drift"]
    assert vipe_disagreement(None, None, 0.0, 0.0) == []


def test_vipe_stage_replaces_slightly_too_fast_vipe(tmp_path, monkeypatch):
    fast = np.einsum("ab,nbc->nac", Q, _r_w_cv(0.115))  # 15% too fast: ~1.6 px per frame at 6000 px
    cam, qc, _ = _run(
        tmp_path, monkeypatch, {"vipe": _vipe_handler(fast), "flow": _flow_handler(np.full(N, F))}
    )
    assert qc["flow"]["vipe_residual_px"]["median"] < 2.0  # the old absolute 2 px rule kept it
    assert qc["rotation_source"] == "flow" and "residual" in qc["flow"]["vipe_disagrees"]
    assert "vipe_rotation_disagrees" in qc["summary"]["flags"]
    _, yaw = _pitch_yaw(cam)
    assert np.allclose(yaw, 0.1 * np.arange(N), atol=0.01)


def test_vipe_stage_catches_slow_drift_far_away(tmp_path, monkeypatch):
    slow = np.einsum("ab,nbc->nac", Q, _r_w_cv(0.103))  # 3% too fast: 0.3 px/frame, under any residual rule
    boxes = np.tile(np.array([[[600, 300, 650, 400]]], np.float32), (N, 1, 1))  # 100 px tall: ~85 m away

    def run(boxes_):
        handlers = {"vipe": _vipe_handler(slow), "flow": _flow_handler(np.full(N, F))}
        fake = FakeWorkers(tmp_path, handlers)

        def call(env, script, request, work_dir, **kw):
            out = fake(env, script, request, work_dir, **kw)
            return {**out, "frame_count": N} if request["command"] == "normalize" else out

        monkeypatch.setattr(camera_mod, "run_env_worker", call)
        work = tmp_path / f"w{len(list(tmp_path.iterdir()))}"
        work.mkdir()
        ctx = Ctx(shot_dir=tmp_path / "shot", work_dir=work)
        CameraVipe().run(ctx, {"detect_track": _tracks(n=N, boxes=boxes_)}, CameraParams(video="/v.mp4"))
        return json.loads((work / "qc.json").read_text())

    far = run(boxes)
    assert far["flow"]["fighter_distance_m"] == pytest.approx(F * 1.42 / 100, rel=0.01)
    assert far["flow"]["vipe_disagrees"] == ["drift"] and far["rotation_source"] == "flow"
    near = run(None)  # no usable boxes: 10 m default, the same drift is ~2 cm: ViPE kept
    assert near["flow"]["vipe_disagrees"] == [] and near["rotation_source"] == "vipe"


def test_zoom_is_applied_when_vipe_rotations_are_kept(tmp_path, monkeypatch):
    """UFC cross: ViPE's rotations are kept (they explain the background) and the zoom the flow
    measures still makes the focal per frame."""
    focal_true = F * np.exp(np.linspace(0.0, np.log(1.5), N))
    f_vipe = _vipe_focal(focal_true)
    good = np.einsum("ab,nbc->nac", Q, _r_w_cv(0.1))
    handlers = {"vipe": _vipe_handler(good, focal=f_vipe), "flow": _flow_handler(focal_true)}
    cam, qc, _ = _run(tmp_path, monkeypatch, handlers)  # zoom="apply" is the default
    assert qc["rotation_source"] == "vipe" and qc["flow"]["vipe_disagrees_pairs"] == []
    assert qc["flow"]["zoom_applied"] is True and "zoom" not in qc["summary"]["flags"]
    assert np.allclose(cam.focal_px / focal_true, 1.0, atol=3e-3)
    assert qc["flow"]["camera_residual_px"]["median"] < 0.3  # ViPE's rotations + the zoom fit


def test_qc_json_has_no_nan_and_flow_gets_the_timeout(tmp_path, monkeypatch):
    mirrored = np.einsum("ab,nbc->nac", Q, _r_w_cv(-0.1))
    handlers = {"vipe": _vipe_handler(mirrored), "flow": _flow_handler(np.full(N, F), valid_every=N)}
    _, qc, fake = _run(tmp_path, monkeypatch, handlers, timeout_s=123.0)
    text = (tmp_path / "work" / "qc.json").read_text()
    assert "NaN" not in text and qc["flow"]["vipe_residual_px"]["median"] is None
    assert "camera_rotation_unverified" in qc["summary"]["flags"]
    assert fake.kwargs["flow"]["timeout"] == 123.0


def test_dropping_vipe_translation_is_flagged(tmp_path, monkeypatch):
    mirrored = np.einsum("ab,nbc->nac", Q, _r_w_cv(-0.1))
    handlers = {"vipe": _vipe_handler(mirrored), "flow": _flow_handler(np.full(N, F))}
    cam, qc, _ = _run(tmp_path, monkeypatch, handlers, use_translation=True)
    assert qc["rotation_source"] == "flow" and "translation_dropped" in qc["summary"]["flags"]
    assert np.allclose(cam.cam_pos, cam.cam_pos[0])


def test_chaining_order_with_large_non_commuting_rotations():
    n, f = 30, 1500.0
    r_true = np.stack([_ry(3.0 * t) @ _rx(1.5 * t) @ _rz(1.0 * t) for t in range(n)])  # pan, tilt, roll
    p0, p1, count = _tracks_for(r_true, np.full(n, f), noise=0.02)
    fc = flow_rotations(p0, p1, count, f, CX, CY, zoom=False)
    assert _rel_angle_deg(fc.r_v_cv, r_true).max() < 0.05
    # the other composition order really is wrong here (so this test pins it)
    rel = np.einsum("tba,tbc->tac", fc.r_v_cv[1:], fc.r_v_cv[:-1])  # x_{t+1} = rel[t] x_t
    wrong = [np.eye(3)]
    for t in range(n - 1):
        wrong.append(rel[t].T @ wrong[-1])
    assert _rel_angle_deg(np.stack(wrong), r_true).max() > 1.0


@pytest.fixture(scope="module")
def vw():
    """The vipe worker module, loaded by path (it never imports afv; cv2 is imported lazily)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("vipe_worker", camera_mod.WORKERS / "vipe_worker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_overlay_mask_takes_bars_and_logos_but_not_a_still_background(vw):
    h, w = 180, 320
    static = np.zeros((h, w), bool)
    static[:, :40] = static[:, -40:] = True  # pillarbox bars
    static[140:160, 220:270] = True  # a burned-in scoreboard
    m = vw.overlay_mask(static, grow=0)
    assert m[:, :40].all() and m[:, -40:].all() and m[140:160, 220:270].all()
    assert not m[:, 60:200].any()
    # locked-off camera, 45% of the frame moving (crowd, fighters): the still 55% is the background
    locked = np.ones((h, w), bool)
    locked[:, : int(0.45 * w)] = False
    assert not vw.overlay_mask(locked, grow=0).any()
    assert not vw.overlay_mask(np.ones((h, w), bool), grow=0).any()  # all still: nothing masked


def test_ransac_prefers_the_motion_covering_the_frame_over_the_densest():
    rng = np.random.default_rng(3)
    still = np.c_[rng.uniform(0.45 * W, W, 60), rng.uniform(0, H, 60)]  # 55% of the frame, few corners
    crowd = np.c_[rng.uniform(0, 0.45 * W, 240), rng.uniform(0, H, 240)]  # 45%, 4x the corners, moving
    p0 = np.r_[still, crowd]
    p1 = np.r_[still, crowd + np.array([4.0, 0.0])]
    r, s, _ = fit_rotation_zoom(p0, p1, F, CX, CY)
    assert _rel_angle_deg(r[None], np.eye(3)[None])[0] < 1e-3 and s == pytest.approx(1.0, abs=1e-4)


def _tilted(cam, deg: float):
    """``cam`` with its world pitched by ``deg`` about the world x axis (a gravity pitch error)."""
    r, c = camera_to_world_cv(cam)
    return make_camera_track(
        np.einsum("ab,tbc->tac", _rx(deg), r),
        c,
        cam.focal_px,
        width=cam.width,
        height=cam.height,
        cx=cam.cx,
        cy=cam.cy,
    )


def test_gravity_check_reports_pitch_error_and_camera_below_floor():
    fk = _standing_fighter()  # ~22 m away
    fks = {"A": fk}
    level = camera_from_gemx(fk)  # world up = GEM-X's: no disagreement
    _, qc = body_motion_from_fk(fks, level)
    assert abs(qc["gravity"]["pitch_difference_deg"]) < 0.3
    heights = {}
    for deg in (-8.0, 8.0):
        motion, qc = body_motion_from_fk(fks, _tilted(level, deg))
        g = qc["gravity"]
        assert abs(g["pitch_difference_deg"]) == pytest.approx(8.0, abs=0.5)
        assert g["height_change_if_gemx_m"] == pytest.approx(
            g["fighter_distance_m"] * np.tan(np.radians(g["pitch_difference_deg"])), rel=1e-3
        )
        heights[deg] = g["camera_height_m"]
        flags = _summary(motion, qc)["flags"]
        assert "gravity_pitch_disagrees" in flags
        assert ("camera_below_floor" in flags) == (g["camera_height_m"] < 0)
        # GEM-X's up, used as the camera stage's up_cam0, re-levels the world: back to agreement
        r, _ = camera_to_world_cv(_tilted(level, deg))
        up_w = r[0] @ np.asarray(g["gemx_up_cam0"])
        assert np.degrees(np.arccos(np.clip(up_w[1], -1, 1))) == pytest.approx(abs(deg), abs=0.5)
    # 16 deg of pitch at ~22 m moves the camera by ~6 m relative to the floor; one side is below it
    assert abs(heights[8.0] - heights[-8.0]) == pytest.approx(2 * 22 * np.tan(np.radians(8.0)), rel=0.25)
    assert min(heights.values()) < 0


def test_camera_up_override_replaces_geocalib(tmp_path, monkeypatch):
    r_cv = np.stack([_rx(-8.0) @ CV_TO_GL] * 12)  # locked-off camera pitched 8 deg down
    up_cam0 = r_cv[0].T @ np.array([0.0, 1.0, 0.0])
    fake = FakeWorkers(tmp_path, {})  # no "gravity" handler: GeoCalib must not run
    monkeypatch.setattr(camera_mod, "run_env_worker", fake)
    ctx = _ctx(tmp_path)
    cam = CameraStatic().run(
        ctx,
        {"detect_track": _tracks()},
        CameraParams(video="/v.mp4", focal_px=1000.0, up_cam0=list(up_cam0), up_override="full"),
    )
    pitch, _ = _pitch_yaw(cam)
    assert np.allclose(pitch, -8.0, atol=1e-6)  # (static mode's own rotations are identity: level camera 0)
    qc = json.loads((ctx.work_dir / "qc.json").read_text())
    assert qc["gravity"] == "override" and "gravity_not_estimated" not in qc["summary"]["flags"]
    assert [r[2]["command"] for r in fake.calls] == ["normalize"]
