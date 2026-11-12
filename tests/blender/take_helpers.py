"""Synthetic shots for the take / Blender add-on tests (no footage, no models)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from afv.core.skeletons import get_skeleton
from afv.core.types import ActorMotion, BodyMotion, CameraTrack
from afv.motion.rotations import from_rotvec, qmul

REPO = Path(__file__).resolve().parents[2]
ADDON_DIR = REPO / "blender" / "afv_addon"
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "tests" / "export"))
from export_helpers import fighter_offsets, synthetic_motion  # noqa: E402

CHANNELS = ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase", "LeftHand", "RightHand"]


def addon_module(name: str):
    """A bpy-free module of the add-on (takeio, wsl), loaded from its file."""
    key = f"_afv_addon_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, ADDON_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def synthetic_shot(n_frames: int = 24, seed: int = 0, scale: float = 1.0, world_shift=(0.0, -0.1, 0.0),
                   width: int = 640, height: int = 360) -> tuple[BodyMotion, CameraTrack]:
    """Two soma77 fighters and a moving, zooming camera with an off-centre principal point.

    The camera stage's world differs from the bodies' by ``world_shift`` (as after pose3d/contacts)."""
    sk = get_skeleton("soma77")
    actors = {}
    rng = np.random.default_rng(seed)
    for role, s, start in (("A", seed + 1, (0.0, 0.0)), ("B", seed + 2, (1.4, 0.6))):
        rot, root = synthetic_motion(sk, n_frames, s, amp=0.3, start=start)
        conf = np.clip(rng.uniform(0.5, 1.0, size=(n_frames, sk.num_joints)), 0, 1).astype(np.float32)
        conf[n_frames // 3 : n_frames // 3 + 3] = 0.1  # a low-confidence span
        contact = np.zeros((n_frames, len(CHANNELS)), np.float32)
        contact[: n_frames // 2, 0] = 1.0
        actors[role] = ActorMotion(rot=rot, root=root, conf=conf, contact=contact,
                                   rest_offsets=fighter_offsets(sk, s) * scale)
    frames = np.arange(n_frames)
    bm = BodyMotion(frames=frames, timestamps=2.0 + frames / 30.0, skeleton="soma77", fps=30.0,
                    contact_channels=list(CHANNELS), actors=actors, world_shift=list(world_shift))
    t = np.linspace(0.0, 1.0, n_frames)
    pos = np.stack([0.6 + 0.5 * t, 1.6 + 0.1 * t, 7.5 - 0.5 * t], axis=-1) - np.asarray(world_shift)
    pitch = from_rotvec(np.stack([-0.08 + 0.02 * t, np.zeros_like(t), np.zeros_like(t)], -1))
    yaw = from_rotvec(np.stack([np.zeros_like(t), 0.12 * np.sin(3 * t), np.zeros_like(t)], -1))
    roll = from_rotvec(np.stack([np.zeros_like(t), np.zeros_like(t), 0.03 * t], -1))
    rot = qmul(qmul(yaw, pitch), roll)
    cam = CameraTrack(frames=frames, timestamps=bm.timestamps, cam_rot=rot, cam_pos=pos,
                      focal_px=600.0 + 250.0 * t, width=width, height=height,
                      cx=0.53 * width, cy=0.46 * height)
    return bm, cam


def write_video(path: Path, n_frames: int, width: int = 320, height: int = 180, fps: int = 30) -> Path:
    """A tiny synthetic H.264 video (frame number as a moving bar), like the ingest proxy."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = width, height, "yuv420p"
        for i in range(n_frames):
            img = np.zeros((height, width, 3), np.uint8)
            img[:, (i * 7) % width : (i * 7) % width + 6] = 255
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return path
