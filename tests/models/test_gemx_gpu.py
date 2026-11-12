"""GPU end-to-end checks of the pose3d and camera stages against the P0 runs (opt-in, few and short).

Run with ``AFV_GPU_TESTS=1 ~/afv/afv-env/bin/python -m pytest tests/models -m gpu``. They need the
model environments, the Media Samples clips and the P0 outputs in ``~/afv/runs``; the GPU lock
(``~/afv/.gpu.lock``) serialises them with other GPU jobs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from afv.core.stage import Ctx
from afv.core.types import Tracks
from afv.models.camera import CameraParams, CameraVipe, camera_to_world_cv, place_on_ground
from afv.models.gemx import Pose3dGemx, Pose3dParams, SomaFK, camera_from_gemx

REPO = Path(__file__).parents[2]
RUNS = Path.home() / "afv/runs"
MEDIA = Path(os.environ.get("AFV_MEDIA_SAMPLES") or next(
    (p for p in (REPO / "Media_Samples", REPO / "Media Samples") if p.is_dir()), REPO / "Media_Samples"))

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("AFV_GPU_TESTS") != "1", reason="set AFV_GPU_TESTS=1 to run GPU tests"),
]


def _need(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            pytest.skip(f"missing {p}")


def _p0_tracks(clip: str) -> Tracks:
    z = np.load(RUNS / "track" / clip / "tracks.npz")
    n = z["boxes"].shape[0]
    people = z["people"].astype(np.int64)
    slots = z["fighter_slots"]
    return Tracks(
        frames=np.arange(n),
        timestamps=np.arange(n) / 30.0,
        track_ids=people,
        boxes=z["boxes"],
        scores=np.isfinite(z["boxes"][..., 0]).astype(np.float32),
        fighters={"A": int(people[slots[0]]), "B": int(people[slots[1]])},
    )


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _ctx(tmp_path: Path, log: list[str]) -> Ctx:
    return Ctx(shot_dir=tmp_path / "shot", work_dir=tmp_path / "work", progress=lambda f, m: log.append(m))


def test_pose3d_stage_reproduces_p0(tmp_path):
    """pose3d on Close boxing with the P0 boxes and camera: same solve as P0, FK exact, QC sane."""
    clip, video = "Close_boxing_short", MEDIA / "Close boxing short.mp4"
    _need(video, RUNS / "track" / clip / "tracks.npz", RUNS / "afv_c_fk" / f"{clip}_rank0.npz")
    p0 = {r: SomaFK.load(RUNS / "afv_c_fk" / f"{clip}_rank{k}.npz") for k, r in enumerate("AB")}
    tracks = _p0_tracks(clip)
    cam = camera_from_gemx(p0["A"])
    cam, _ = place_on_ground(cam, tracks.boxes[:, :2].astype(np.float64))
    log: list[str] = []
    ctx = _ctx(tmp_path, log)
    ctx.work_dir.mkdir(parents=True)
    motion = Pose3dGemx().run(ctx, {"detect_track": tracks, "camera": cam}, Pose3dParams(video=str(video)))
    assert motion.skeleton == "soma77" and set(motion.actors) == {"A", "B"} and motion.num_frames == 92
    qc = json.loads((ctx.work_dir / "qc.json").read_text())
    for role in "AB":
        new = SomaFK.load(ctx.work_dir / "gemx" / role / "soma_fk.npz")
        # Same boxes and camera rotations as P0 (K now square): the solve matches P0 closely.
        diff = np.linalg.norm(new.incam_pos - p0[role].incam_pos, axis=-1)
        assert np.median(diff) < 0.05, f"{role}: median joint change vs P0 {np.median(diff):.3f} m"
        assert qc["fighters"][role]["reprojection_px_median"] < 25.0
        assert qc["fighters"][role]["gravity_disagreement_deg_median"] < 10.0
        r, c = camera_to_world_cv(cam)
        s = qc["floor"]["scale"][role]["scale"]  # floor scale about the camera centre (image kept)
        assert qc["root_mode"]["local_fallback_spans"] == []
        hips_world = c + s * np.einsum("fij,fj->fi", r, new.incam_pos[:, 0]) + motion.world_shift
        assert abs(qc["fighters"][role]["sole_min_height_m_median"]) < 0.25  # floor re-fit
        root = np.asarray(motion.actors[role].root, dtype=np.float64)
        weak = np.zeros(len(root), bool)
        for a, b in qc["fighters"][role]["weak_2d_spans"]:  # depth held there (truncated/occluded body)
            weak[a : b + 1] = True
        assert np.linalg.norm(root - hips_world, axis=-1)[~weak].max() < 1e-3

        eye = c + np.asarray(motion.world_shift)
        assert np.abs(_unit(root - eye) - _unit(hips_world - eye))[weak].max(initial=0.0) < 1e-4  # same ray
    assert any("pose3d A" in m for m in log)


def test_camera_vipe_stage_matches_p0_and_gemx_gravity(tmp_path):
    """ViPE once per shot on UFC cross: P0's rotations again, GeoCalib up close to GEM-X's up."""
    clip, video = "UFC_cross", MEDIA / "UFC cross.mp4"
    _need(video, RUNS / "track" / clip / "tracks.npz", RUNS / "afv_c_fk" / f"{clip}_rank0.npz")
    p0 = SomaFK.load(RUNS / "afv_c_fk" / f"{clip}_rank0.npz")
    tracks = _p0_tracks(clip)
    log: list[str] = []
    ctx = _ctx(tmp_path, log)
    ctx.work_dir.mkdir(parents=True)
    # ViPE's own rotations (auto would splice flow's in over UFC's zoom, where ViPE reads a dolly)
    params = CameraParams(video=str(video), rotation_source="vipe")
    cam = CameraVipe().run(ctx, {"detect_track": tracks}, params)
    assert cam.num_frames == 302 and cam.width == 1280 and cam.height == 720
    info = json.loads((ctx.work_dir / "qc.json").read_text())
    assert info["gravity"] == "geocalib", info
    # relative camera rotations agree with P0's ViPE run (ViPE is not bit-deterministic)
    r_new, _ = camera_to_world_cv(cam)
    rel_new = np.einsum("tba,bc->tac", r_new, r_new[0])  # R_c0<-ct
    r_p0 = np.swapaxes(p0.R_w2c, -1, -2)
    rel_p0 = np.einsum("tba,bc->tac", r_p0, r_p0[0])
    cos = (np.trace(np.einsum("tab,tcb->tac", rel_new, rel_p0), axis1=1, axis2=2) - 1) / 2
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))).max() < 2.0
    assert abs(np.median(cam.focal_px) - p0.K[0, 0, 0]) / p0.K[0, 0, 0] < 0.05
    # GeoCalib gravity vs GEM-X's own gravity (both fighters' P0 solves), camera 0 frame
    up_gemx = np.einsum("tab,tcb,c->ta", p0.incam_rot[:, 0], p0.global_rot[:, 0], [0, 1.0, 0]).mean(0)
    up_new = r_new[0].T @ np.array([0, 1.0, 0])
    ang = np.degrees(np.arccos(np.clip(up_new @ up_gemx / np.linalg.norm(up_gemx), -1, 1)))
    assert ang < 8.0, f"GeoCalib vs GEM-X gravity {ang:.1f} deg"
    assert 0.5 < info["ground"]["camera_height_m"] < 5.0


@pytest.fixture(scope="module")
def ufc_auto(tmp_path_factory):
    """The default camera (rotation_source auto, zoom applied) on UFC cross, P0's tracks."""
    clip, video = "UFC_cross", MEDIA / "UFC cross.mp4"
    _need(video, RUNS / "track" / clip / "tracks.npz")
    tmp = tmp_path_factory.mktemp("ufc_auto")
    tracks = _p0_tracks(clip)
    ctx = _ctx(tmp, [])
    ctx.work_dir.mkdir(parents=True)
    cam = CameraVipe().run(ctx, {"detect_track": tracks}, CameraParams(video=str(video)))
    return cam, json.loads((ctx.work_dir / "qc.json").read_text()), tracks, video, tmp


def test_default_camera_splices_flow_and_applies_the_zoom_on_ufc(ufc_auto):
    """UFC cross zooms ~1.2x over frames ~180-250; ViPE reads it as a dolly there."""
    cam, qc, *_ = ufc_auto
    assert qc["rotation_source"] == "vipe+flow" and qc["flow"]["zoom_applied"]
    (s, e), *_ = qc["flow"]["vipe_disagrees_pairs"]
    assert 160 <= s <= 200 and 215 <= e <= 250
    f = np.asarray(cam.focal_px, dtype=np.float64)
    assert f[-30:].mean() / f[:30].mean() == pytest.approx(1.2, abs=0.04)
    assert qc["flow"]["camera_residual_px"]["median"] < 0.5 and "zoom" not in qc["summary"]["flags"]


def test_gemx_depth_follows_the_focal_it_is_given(ufc_auto):
    """Real data: GEM-X's depth over the focal it was given, d / f, is the same with the zoomed K and
    with one K per shot (it reads image size), so the zoom moves depth by exactly the focal ratio."""
    import copy

    cam, _, tracks, video, tmp = ufc_auto
    fixed = copy.copy(cam)
    fixed.focal_px = np.full(cam.num_frames, float(cam.focal_px[0]), dtype=np.float32)
    depth = {}
    for name, c in (("zoom", cam), ("fixed", fixed)):
        ctx = Ctx(shot_dir=tmp / "shot", work_dir=tmp / f"pose_{name}")
        ctx.work_dir.mkdir(parents=True)
        Pose3dGemx().run(ctx, {"detect_track": tracks, "camera": c}, Pose3dParams(video=str(video)))
        for role in "AB":
            fk = SomaFK.load(ctx.work_dir / "gemx" / role / "soma_fk.npz")
            depth[name, role] = fk.incam_pos[:, 0, 2] / fk.K[:, 0, 0]
    zoom = np.asarray(cam.focal_px, dtype=np.float64) / float(cam.focal_px[0])
    zoomed = zoom > 1.1
    assert zoomed.sum() > 30
    for role in "AB":
        ratio = depth["zoom", role] / depth["fixed", role]
        assert np.median(np.abs(ratio - 1.0)) < 0.03 and np.median(np.abs(ratio[zoomed] - 1.0)) < 0.05
