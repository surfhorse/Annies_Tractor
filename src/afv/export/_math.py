"""Small vector/quaternion helpers for export (xyzw, broadcasting), on top of afv.motion.rotations."""

from __future__ import annotations

import numpy as np

from afv.motion.rotations import normalize, qconj, qmul


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal (swing) rotation taking direction a onto direction b, xyzw, broadcasting over [...,3]."""
    a = unit(a)
    b = unit(b)
    d = np.sum(a * b, axis=-1, keepdims=True)
    c = np.cross(a, b)
    q = np.concatenate([c, 1.0 + d], axis=-1)
    # Antiparallel: rotate 180 degrees about any axis perpendicular to a.
    anti = (1.0 + d)[..., 0] < 1e-9
    if np.any(anti):
        ref = np.where(np.abs(a[..., :1]) < 0.9, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        axis = unit(np.cross(a, ref))
        q = np.where(anti[..., None], np.concatenate([axis, np.zeros_like(d)], axis=-1), q)
    return normalize(q)


def qinv_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a^-1 * b for unit quaternions."""
    return qmul(qconj(a), b)


def yaw_quat(yaw: np.ndarray) -> np.ndarray:
    """Rotation by ``yaw`` radians about +Y (xyzw)."""
    yaw = np.asarray(yaw, dtype=np.float64)
    z = np.zeros_like(yaw)
    return np.stack([z, np.sin(yaw / 2.0), z, np.cos(yaw / 2.0)], axis=-1)
