"""Camera helpers on ``CameraTrack`` (camera-to-world, looks down -Z, +Y up, OpenGL convention)."""

from __future__ import annotations

import numpy as np

from afv.core.stage import StageError
from afv.core.types import CameraTrack
from afv.motion.rotations import qconj, qrotate


def camera_index(cam: CameraTrack, frames: np.ndarray) -> np.ndarray:
    """Index into the camera's per-frame arrays for each motion frame number in ``frames``.

    A one-sample (or static) camera broadcasts. Raises if a motion frame has no camera sample.
    """
    frames = np.asarray(frames, dtype=np.int64)
    n = len(cam.cam_pos)
    if n == 1:
        return np.zeros(len(frames), dtype=np.int64)
    cf = np.asarray(cam.frames, dtype=np.int64)
    if len(cf) != n:
        cf = np.arange(n)
    idx = np.clip(np.searchsorted(cf, frames), 0, n - 1)
    bad = cf[idx] != frames
    if np.any(bad):
        raise StageError(f"camera has no sample for frames {frames[bad][:5].tolist()}...")
    return idx


def camera_centres(cam: CameraTrack, frames: np.ndarray) -> np.ndarray:
    return np.asarray(cam.cam_pos, dtype=np.float64)[camera_index(cam, frames)]


def to_camera(points: np.ndarray, cam: CameraTrack, frames: np.ndarray) -> np.ndarray:
    """World points [F,...,3] to camera coordinates (x right, y up, looking down -z)."""
    idx = camera_index(cam, frames)
    rot = np.asarray(cam.cam_rot, dtype=np.float64)[idx]
    pos = np.asarray(cam.cam_pos, dtype=np.float64)[idx]
    p = np.asarray(points, dtype=np.float64)
    extra = p.ndim - 2
    rot = rot.reshape(rot.shape[0], *([1] * extra), 4)
    pos = pos.reshape(pos.shape[0], *([1] * extra), 3)
    return qrotate(qconj(rot), p - pos)


def project(points: np.ndarray, cam: CameraTrack, frames: np.ndarray) -> np.ndarray:
    """Pixel coordinates [F,...,2] (x right, y down) of world points [F,...,3]; NaN behind camera."""
    pc = to_camera(points, cam, frames)
    idx = camera_index(cam, frames)
    f = np.asarray(cam.focal_px, dtype=np.float64)
    f = f[idx] if f.size > 1 else np.full(len(idx), float(f.reshape(-1)[0]) if f.size else 1.0)
    f = f.reshape(f.shape[0], *([1] * (pc.ndim - 2)))
    depth = -pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.cx + f * pc[..., 0] / depth
        v = cam.cy - f * pc[..., 1] / depth
    out = np.stack([u, v], axis=-1)
    out[depth <= 1e-6] = np.nan
    return out
