"""Quaternion and basis helpers. All quaternions are xyzw (scalar last), unit, float64 in maths.

Arrays broadcast over leading axes: shape [..., 4] for quaternions, [..., 3] for vectors.
Composition order: ``qmul(a, b)`` applies ``b`` first, then ``a`` (as matrices: A @ B).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("zero-length quaternion")
    return q / n


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b for xyzw quaternions (broadcasting)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def qconj(q: np.ndarray) -> np.ndarray:
    """Conjugate = inverse for unit quaternions."""
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def qrotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v [...,3] by unit quaternions q [...,4] (broadcasting)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def make_continuous(q: np.ndarray, axis: int = 0) -> np.ndarray:
    """Flip signs along ``axis`` (time) so consecutive quaternions lie in the same hemisphere."""
    q = np.array(q, dtype=np.float64, copy=True)
    q = np.moveaxis(q, axis, 0)
    for i in range(1, q.shape[0]):
        d = np.sum(q[i] * q[i - 1], axis=-1, keepdims=True)
        q[i] = np.where(d < 0.0, -q[i], q[i])
    return np.moveaxis(q, 0, axis)


def slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Spherical interpolation, shortest path, broadcasting; t in [0,1]."""
    a = normalize(a)
    b = normalize(b)
    t = np.asarray(t, dtype=np.float64)
    d = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(d < 0.0, -b, b)
    d = np.abs(d)
    t = t[..., None] if t.ndim == a.ndim - 1 or (t.ndim > 0 and t.shape[-1] != 1) else t
    theta = np.arccos(np.clip(d, -1.0, 1.0))
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-8
    safe = np.where(small, 1.0, sin_theta)
    wa = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    wb = np.where(small, t, np.sin(t * theta) / safe)
    return normalize(wa * a + wb * b)


def to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(*q.shape[:-1], 3, 3)


def from_matrix(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return Rotation.from_matrix(m.reshape(-1, 3, 3)).as_quat().reshape(*m.shape[:-2], 4)


def to_rotvec(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return Rotation.from_quat(q.reshape(-1, 4)).as_rotvec().reshape(*q.shape[:-1], 3)


def from_rotvec(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    return Rotation.from_rotvec(r.reshape(-1, 3)).as_quat().reshape(*r.shape[:-1], 4)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """The one place that converts to Blender's scalar-first order."""
    q = np.asarray(q)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


# --- change of basis: internal Y-up  ->  Blender Z-up -------------------------------------------
# Internal: right-handed, +Y up, character forward +Z.  Blender: right-handed, +Z up, forward -Y.
# Mapping (x, y, z)_yup -> (x, -z, y)_zup is a +90 degree rotation about X. It is a proper
# rotation (det +1), so handedness is preserved and quaternions convert by conjugation.
YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_Q_YUP_TO_ZUP = from_matrix(YUP_TO_ZUP)


def yup_to_zup_vec(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) @ YUP_TO_ZUP.T


def zup_to_yup_vec(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) @ YUP_TO_ZUP


def yup_to_zup_quat(q: np.ndarray) -> np.ndarray:
    """Re-express a world-space rotation in the Z-up basis: C q C^-1."""
    return qmul(qmul(_Q_YUP_TO_ZUP, q), qconj(_Q_YUP_TO_ZUP))


def zup_to_yup_quat(q: np.ndarray) -> np.ndarray:
    return qmul(qmul(qconj(_Q_YUP_TO_ZUP), q), _Q_YUP_TO_ZUP)
