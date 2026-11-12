"""The vipe worker's ``flow`` command in its model environment (opt-in, like the other GPU tests:
``AFV_GPU_TESTS=1 ~/afv/afv-env/bin/python -m pytest tests/models -m gpu``; it runs on the CPU but
needs the vipe env). A synthetic panning + zooming video with a moving "person" gives the truth; on
ref walk (telephoto, P0 outputs in ``~/afv/runs``) the flow camera must explain the background where
P0's ViPE rotations did not (the telephoto root bug)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, map_coordinates

from afv.models.camera import flow_rotations, rotation_residual_px, run_env_worker

REPO = Path(__file__).parents[2]
RUNS = Path.home() / "afv/runs"
MEDIA = Path(os.environ.get("AFV_MEDIA_SAMPLES") or next(
    (p for p in (REPO / "Media_Samples", REPO / "Media Samples") if p.is_dir()), REPO / "Media_Samples"))

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("AFV_GPU_TESTS") != "1", reason="set AFV_GPU_TESTS=1 to run GPU tests"),
]

W, H, N, F = 640, 360, 45, 1500.0


def _ry(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    import av

    with av.open(str(path), "w") as out:
        st = out.add_stream("libx264", rate=30)
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.options = {"crf": "8"}
        for img in frames:
            rgb = np.repeat(img[..., None], 3, axis=2)
            for pkt in st.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)


def test_flow_worker_recovers_a_synthetic_pan_and_zoom(tmp_path):
    rng = np.random.default_rng(0)
    # a far textured scene on a "panorama" indexed by viewing angle (pure rotation: no parallax)
    pano = gaussian_filter(rng.uniform(0, 255, (900, 2400)), 2.0)
    pano = (pano - pano.min()) / np.ptp(pano) * 255
    pan = 0.15  # deg/frame
    focal = F * np.exp(np.linspace(0.0, np.log(1.2), N))  # 20% zoom-in
    u, v = np.meshgrid(np.arange(W) - W / 2, np.arange(H) - H / 2)
    frames, boxes = [], np.full((N, 1, 4), np.nan, np.float32)
    for t in range(N):
        rays = np.stack([u / focal[t], v / focal[t], np.ones_like(u)], -1) @ _ry(pan * t).T
        az, el = (
            np.arctan2(rays[..., 0], rays[..., 2]),
            np.arctan2(rays[..., 1], np.hypot(rays[..., 0], rays[..., 2])),
        )
        img = map_coordinates(pano, [450 + el * 1500, 1200 + az * 1500], order=1)
        x0 = int(100 + 8 * t)  # a "person" walking across, faster than the pan, masked by its box
        img[100:330, x0 : x0 + 90] = 255 * (rng.uniform(size=(230, 90)) > 0.5)
        boxes[t, 0] = (x0, 100, x0 + 90, 330)
        frames.append(img.astype(np.uint8))
    video = tmp_path / "pan.mp4"
    _write_video(video, frames)
    np.savez(tmp_path / "boxes.npz", boxes=boxes)
    resp = run_env_worker(
        "vipe",
        "vipe_worker.py",
        {
            "command": "flow",
            "video": str(video),
            "out_npz": str(tmp_path / "flow.npz"),
            "boxes_npz": str(tmp_path / "boxes.npz"),
        },
        tmp_path,
        gpu=False,
    )
    z = np.load(resp["out_npz"])
    assert z["count"].shape == (N - 1,) and (z["count"] >= 50).all()
    fc = flow_rotations(z["p0"], z["p1"], z["count"], float(np.median(focal)), W / 2, H / 2, apply_zoom=True)
    rel = np.einsum("tba,tbc->tac", fc.r_v_cv[:-1], fc.r_v_cv[1:])
    pan_deg = np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))
    assert np.allclose(pan_deg, pan, atol=0.01)  # the person's motion did not leak in
    assert fc.zoom_applied and np.allclose(fc.focal_px / focal, 1.0, atol=0.01)
    assert np.nanmedian(fc.residual_px) < 0.3


def test_flow_worker_keeps_a_locked_off_camera_still_when_much_of_the_frame_moves(tmp_path):
    """Review repro: 45% of the frame translating 4 px/frame on a locked-off camera used to lose the
    still background to the static mask, giving a 0.15 deg/frame pan."""
    rng = np.random.default_rng(1)
    scene = gaussian_filter(rng.uniform(0, 255, (H, W)), 1.5)
    crowd = gaussian_filter(rng.uniform(0, 255, (H, 2 * W)), 1.5)
    split = int(0.45 * W)
    frames = []
    for t in range(N):
        img = scene.copy()
        img[:, :split] = crowd[:, 4 * t : 4 * t + split]
        frames.append(np.clip((img - img.min()) / np.ptp(img) * 255, 0, 255).astype(np.uint8))
    video = tmp_path / "locked.mp4"
    _write_video(video, frames)
    resp = run_env_worker(
        "vipe",
        "vipe_worker.py",
        {"command": "flow", "video": str(video), "out_npz": str(tmp_path / "f.npz")},
        tmp_path,
        gpu=False,
    )
    z = np.load(resp["out_npz"])
    fc = flow_rotations(z["p0"], z["p1"], z["count"], F, W / 2, H / 2)
    rel = np.einsum("tba,tbc->tac", fc.r_v_cv[:-1], fc.r_v_cv[1:])
    pan = np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))
    assert np.abs(pan).max() < 0.01 and float(z["static_fraction"]) < 0.05


def test_flow_explains_ref_walk_background_where_p0_vipe_did_not(tmp_path):
    clip = "ref_walk"
    names = (MEDIA / "ref_walk.mp4", MEDIA / "ref walk.mp4")  # round 5 name, older name
    video = next((p for p in names if p.is_file()), names[0])
    fk_path, tr_path = RUNS / "afv_c_fk" / f"{clip}_rank0.npz", RUNS / "track" / clip / "tracks.npz"
    for p in (video, fk_path, tr_path):
        if not p.exists():
            pytest.skip(f"missing {p}")
    norm = run_env_worker(
        "gemx",
        "gemx_worker.py",
        {
            "command": "normalize",
            "video": str(video),
            "start": 0.0,
            "duration": 8.782,
            "out_mp4": str(tmp_path / "s.mp4"),
        },
        tmp_path,
        gpu=False,
    )
    n = int(norm["frame_count"])
    assert n == 264  # shot 0, before the hard cut
    np.savez(tmp_path / "boxes.npz", boxes=np.load(tr_path)["boxes"][:n])
    resp = run_env_worker(
        "vipe",
        "vipe_worker.py",
        {
            "command": "flow",
            "video": norm["out_mp4"],
            "out_npz": str(tmp_path / "f.npz"),
            "boxes_npz": str(tmp_path / "boxes.npz"),
        },
        tmp_path,
        gpu=False,
    )
    z = np.load(resp["out_npz"])
    fk = np.load(fk_path)
    k = fk["K"][0]
    f, cx, cy = float(k[0, 0]), float(k[0, 2]), float(k[1, 2])
    fc = flow_rotations(z["p0"], z["p1"], z["count"], f, cx, cy)
    vipe = rotation_residual_px(z["p0"], z["p1"], z["count"], np.swapaxes(fk["R_w2c"][:n], 1, 2), f, cx, cy)
    assert fc.valid.mean() > 0.95 and np.nanmedian(fc.residual_px) < 0.5
    assert np.nanmedian(vipe) > 5.0  # P0's ViPE rotations: ~11 px per frame on this shot
    (tmp_path / "summary.json").write_text(json.dumps({"flow": float(np.nanmedian(fc.residual_px))}))
