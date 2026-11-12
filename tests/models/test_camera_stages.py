"""Camera and pose3d stage solve paths with a fake worker bridge (CPU; no model environments)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import afv.models.gemx as gemx_mod
from afv.core.stage import Ctx, StageError
from afv.core.types import BodyMotion, Tracks
from afv.models import camera as camera_mod
from afv.models.camera import (
    CV_TO_GL,
    CameraBackground,
    CameraParams,
    CameraStatic,
    CameraVipe,
    camera_to_world_cv,
    gemx_network_rotations,
    gemx_provided_camera,
)
from afv.models.gemx import Pose3dGemx, Pose3dParams, SomaFK, camera_from_gemx

FIX = Path(__file__).parent / "fixtures" / "gemx_ufc_cross_rank0.npz"
N, W, H = 12, 1280, 720


def _rx(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _ry(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


# true world: a camera pitched 8 deg down, panning 2 deg per frame
R_W_CV = np.stack([_ry(2.0 * t) @ _rx(-8.0) @ CV_TO_GL for t in range(N)])
Q = _ry(55) @ _rx(30)  # the solver's arbitrary world V = Q W


class FakeWorkers:
    """Stands in for run_env_worker: records requests, writes the files a real worker would."""

    def __init__(self, tmp: Path, handlers: dict):
        self.tmp, self.handlers, self.calls = tmp, handlers, []

    def __call__(self, env, script, request, work_dir, **kw):
        self.calls.append((env, script, dict(request)))
        cmd = request["command"]
        if cmd == "normalize":
            Path(request["out_mp4"]).write_bytes(b"")
            return {
                "out_mp4": request["out_mp4"],
                "frame_count": N,
                "width": W,
                "height": H,
                "fps": 30.0,
                "ffmpeg": "/home/x/.local/bin/ffmpeg",
            }
        return self.handlers[cmd](request)

    def request(self, cmd):
        return next(r for _, _, r in self.calls if r["command"] == cmd)


def _ctx(tmp_path: Path) -> Ctx:
    work = tmp_path / "work"
    work.mkdir()
    return Ctx(shot_dir=tmp_path / "shot", work_dir=work)


def _tracks(n=N, boxes=None) -> Tracks:
    b = np.full((n, 1, 4), np.nan, np.float32) if boxes is None else boxes
    return Tracks(
        frames=np.arange(n),
        timestamps=np.arange(n) / 30.0,
        track_ids=np.arange(b.shape[1]),
        boxes=b,
        scores=np.ones(b.shape[:2], np.float32),
        fighters={"A": 0},
    )


def _pitch_yaw(cam):
    r, _ = camera_to_world_cv(cam)
    view = r[:, :, 2]  # OpenCV +z = viewing direction
    pitch = np.degrees(np.arcsin(view[:, 1]))
    yaw = np.degrees(np.arctan2(-view[:, 0], -view[:, 2]))
    return pitch, yaw


def _up_cam(r_v_cv, frames):
    return np.einsum("nba,b->na", r_v_cv[frames], Q @ [0, 1.0, 0])


def test_vipe_stage_from_c2w(tmp_path, monkeypatch):
    fx = 1500 + 10 * np.sin(np.arange(N))
    intr = np.stack([fx, fx + 0.3, np.full(N, 641.0), np.full(N, 359.0)], axis=1)
    r_v_cv = np.einsum("ab,nbc->nac", Q, R_W_CV)
    c2w = np.tile(np.eye(4), (N, 1, 1))
    c2w[:, :3, :3] = r_v_cv
    c2w[:, :3, 3] = [1.0, 2.0, 3.0]
    frames = np.array([0, 5, 11])

    def vipe(req):
        np.savez(
            req["out_npz"],
            c2w=c2w,
            intrinsics=intr,
            up_cam=_up_cam(r_v_cv, frames),
            up_frames=frames,
            up_weight=np.ones(3),
            up_focal=np.full(3, 1490.0),
        )
        return {"out_npz": req["out_npz"], "frames": N, "gravity": "geocalib"}

    fake = FakeWorkers(tmp_path, {"vipe": vipe})
    monkeypatch.setattr(camera_mod, "run_env_worker", fake)
    ctx = _ctx(tmp_path)
    cam = CameraVipe().run(
        ctx,
        {"detect_track": _tracks()},
        CameraParams(video="/src/clip.mp4", start=2.0, rotation_source="vipe"),
    )
    pitch, yaw = _pitch_yaw(cam)
    assert np.allclose(pitch, -8.0, atol=1e-6) and np.allclose(yaw, 2.0 * np.arange(N), atol=1e-6)
    # one focal per shot: the K GEM-X will actually use
    assert np.allclose(cam.focal_px, np.median(intr[:, :2].mean(1)))
    assert (cam.cx, cam.cy) == (641.0, 359.0)
    assert np.allclose(cam.timestamps, np.arange(N) / 30)  # the shot timeline comes from the tracks
    assert cam.cam_pos[0, 1] == pytest.approx(1.5)  # no usable boxes: default camera height
    qc = json.loads((ctx.work_dir / "qc.json").read_text())
    assert qc["summary"]["mode"] == "vipe" and "few_usable_boxes" in qc["summary"]["flags"]
    assert qc["vipe_focal_px"]["min"] == pytest.approx(intr[:, :2].mean(1).min())
    assert fake.request("normalize")["video"] == "/src/clip.mp4" and fake.request("normalize")["start"] == 2.0
    assert fake.request("vipe")["gravity"] is True
    assert [r["command"] for _, _, r in fake.calls] == ["normalize", "vipe"]  # no second GeoCalib run


@pytest.mark.parametrize("focal", [None, 1000.0])
def test_static_stage_gives_geocalib_a_focal_only_when_known(tmp_path, monkeypatch, focal):
    r_cv = np.stack([_rx(-8.0) @ CV_TO_GL] * N)  # locked-off camera pitched 8 deg down

    def gravity(req):
        np.savez(
            req["out_npz"],
            up_cam=np.einsum("nba,b->na", r_cv[:3], [0, 1.0, 0]),
            up_frames=np.arange(3),
            up_weight=np.ones(3),
            up_focal=np.full(3, 1400.0),
        )
        return {
            "out_npz": req["out_npz"],
            "frames": N,
            "gravity": "geocalib",
            "focal_prior_px": req["focal_px"],
            "geocalib_focal_px": 1400.0,
        }

    fake = FakeWorkers(tmp_path, {"gravity": gravity})
    monkeypatch.setattr(camera_mod, "run_env_worker", fake)
    ctx = _ctx(tmp_path)
    cam = CameraStatic().run(ctx, {"detect_track": _tracks()}, CameraParams(video="/v.mp4", focal_px=focal))
    assert fake.request("gravity")["focal_px"] == focal  # never GEM-X's diagonal guess
    assert np.allclose(cam.focal_px, focal or np.hypot(W, H))
    pitch, yaw = _pitch_yaw(cam)
    assert np.allclose(pitch, -8.0, atol=1e-6) and np.allclose(yaw, 0.0, atol=1e-6)
    qc = json.loads((ctx.work_dir / "qc.json").read_text())
    assert qc["geocalib"]["geocalib_focal_px"] == 1400.0 and qc["summary"]["gravity"] == "geocalib"


def test_background_stage_transposes_r_w2c(tmp_path, monkeypatch):
    r_c0_ct = np.stack([_ry(-3.0 * t) for t in range(N)])  # OpenCV camera t -> camera 0 (pan)
    boxes = np.tile(np.array([[[600, 200, 700, 600]]], np.float32), (N, 1, 1))

    def background(req):
        np.savez(req["out_npz"], R_w2c=np.swapaxes(r_c0_ct, 1, 2), K=np.eye(3))
        return {"out_npz": req["out_npz"], "diagnostics": {"method": "background"}}

    fake = FakeWorkers(tmp_path, {"background_camera": background})
    monkeypatch.setattr(camera_mod, "run_env_worker", fake)
    cam = CameraBackground().run(
        _ctx(tmp_path), {"detect_track": _tracks(boxes=boxes)}, CameraParams(video="/v.mp4", gravity="level")
    )
    r, _ = camera_to_world_cv(cam)
    rel = np.einsum("ba,nbc->nac", r[0], r)  # camera t -> camera 0, from the CameraTrack
    assert np.allclose(rel, r_c0_ct, atol=1e-9)
    assert fake.request("background_camera")["slot"] == 0
    assert [r2["command"] for _, _, r2 in fake.calls] == ["normalize", "background_camera"]


def test_camera_stage_needs_a_source(tmp_path):
    with pytest.raises(StageError, match="no shot context"):
        CameraStatic().run(_ctx(tmp_path), {"detect_track": _tracks()}, CameraParams())


def test_pose3d_stage_with_fake_gemx(tmp_path, monkeypatch):
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
        return {"frames": n, "fighters": out, "timings": {"motion_s": 1.0}, "peak_vram_mib": 1, "ffmpeg": "x"}

    fake = FakeWorkers(tmp_path, {"pose3d": pose3d})
    monkeypatch.setattr(gemx_mod, "run_env_worker", fake)
    ctx = _ctx(tmp_path)
    motion = Pose3dGemx().run(
        ctx, {"detect_track": tracks, "camera": cam}, Pose3dParams(video="/src/clip.mp4", start=1.0)
    )
    assert isinstance(motion, BodyMotion) and set(motion.actors) == {"A", "B"}
    assert motion.world_shift[1] != 0.0 and motion.world_shift[0] == 0.0
    req = fake.request("pose3d")
    assert req["video"] == "/src/clip.mp4" and req["start"] == 1.0 and req["camera"]["mode"] == "provided"
    assert [f["slot"] for f in req["fighters"]] == [0, 1]
    k = req["camera"]["intrinsics"]
    assert k[0][0] == k[1][1] == pytest.approx(float(np.median(cam.focal_px)))
    r_w2c, k_frames = gemx_provided_camera(cam)
    rot = np.load(req["camera"]["rotations_npy"])  # relative to camera 0: same cam_angvel as r_w2c
    assert np.allclose(rot, gemx_network_rotations(r_w2c)) and np.allclose(rot[0], np.eye(3), atol=1e-6)
    rel = np.einsum("tab,tcb->tac", r_w2c[1:].astype(np.float64), r_w2c[:-1].astype(np.float64))
    assert np.allclose(np.einsum("tab,tcb->tac", rot[1:], rot[:-1]), rel, atol=1e-6)
    assert np.array_equal(np.load(req["camera"]["intrinsics_npy"]), k_frames)
    gemx_cache = json.loads((ctx.work_dir / "qc.json").read_text())["gemx_cache"]
    assert gemx_cache["status"].startswith("uncached")  # no source file to identify
    qc = json.loads((ctx.work_dir / "qc.json").read_text())
    assert qc["summary"]["world_shift"] == motion.world_shift
    assert qc["summary"]["floor_method"] == "contact_weighted_median"
    assert set(qc["fighters"]) == {"A", "B"} and qc["fighters"]["A"]["reprojection_px_median"] < 25
