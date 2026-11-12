"""Forward kinematics on a ``Skeleton``.

World transform of joint j at frame f:
    R_w[j] = R_w[parent] * R_rest[j] * R_local[j]
    p_w[j] = p_w[parent] + R_w[parent] (rest_offsets[j])          (non-root)
    p_w[root] = root[f];  R_w[root] = R_rest[root] * R_local[root]

``rest_offsets[root]`` is the rest position used only when no root translation is given.
"""

from __future__ import annotations

import numpy as np

from afv.core.types import Skeleton
from afv.motion.rotations import IDENTITY, qmul, qrotate


def forward_kinematics(
    sk: Skeleton, rot: np.ndarray, root: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (world_pos [F,J,3], world_rot [F,J,4] xyzw) for local rotations ``rot`` [F,J,4].

    ``root`` [F,3] is the root joint's world translation; None uses the rest root position.
    """
    rot = np.asarray(rot, dtype=np.float64)
    if rot.ndim != 3 or rot.shape[1:] != (sk.num_joints, 4):
        raise ValueError(f"rot shape {rot.shape}, expected [F,{sk.num_joints},4]")
    n_frames = rot.shape[0]
    if root is None:
        root = np.broadcast_to(sk.rest_offsets[0], (n_frames, 3))
    root = np.asarray(root, dtype=np.float64)
    if root.shape != (n_frames, 3):
        raise ValueError(f"root shape {root.shape}, expected [{n_frames},3]")
    rest_rot = sk.rest_rotations if sk.rest_rotations is not None else np.tile(IDENTITY, (sk.num_joints, 1))
    pos = np.empty((n_frames, sk.num_joints, 3))
    wrot = np.empty((n_frames, sk.num_joints, 4))
    for j, p in enumerate(sk.parents):
        local = qmul(rest_rot[j], rot[:, j])
        if p < 0:
            wrot[:, j] = local
            pos[:, j] = root
        else:
            wrot[:, j] = qmul(wrot[:, p], local)
            pos[:, j] = pos[:, p] + qrotate(wrot[:, p], sk.rest_offsets[j])
    return pos, wrot


def rest_positions(sk: Skeleton) -> np.ndarray:
    """World joint positions [J,3] in the rest pose."""
    rot = np.tile(IDENTITY, (1, sk.num_joints, 1))
    pos, _ = forward_kinematics(sk, rot)
    return pos[0]
