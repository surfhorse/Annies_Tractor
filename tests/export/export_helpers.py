"""Synthetic fighters and takes for the export tests (no footage, no models)."""

from __future__ import annotations

import numpy as np

from afv.core.skeletons import SOMA_PROVISIONAL
from afv.core.types import Skeleton, Take
from afv.motion.rotations import from_rotvec, qmul


def fighter_offsets(sk: Skeleton, seed: int = 0) -> np.ndarray:
    """A fighter's own bone offsets: every bone scaled by its own factor in [0.9, 1.15]."""
    rng = np.random.default_rng(seed)
    off = sk.rest_offsets * rng.uniform(0.9, 1.15, size=(sk.num_joints, 1))
    off[0] = sk.rest_offsets[0] * 1.05
    return off


def synthetic_motion(sk: Skeleton, n_frames: int = 48, seed: int = 0, amp: float = 0.45,
                     start=(0.0, 0.0), turn: float = np.pi / 2) -> tuple[np.ndarray, np.ndarray]:
    """Smooth random local rotations [F,J,4] and a curved root path [F,3] with a turn of ``turn``."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_frames)[:, None]
    axes = rng.normal(size=(sk.num_joints, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    freq = rng.uniform(0.5, 2.0, size=sk.num_joints)
    phase = rng.uniform(0, 2 * np.pi, size=sk.num_joints)
    a = amp * np.sin(2 * np.pi * freq * t + phase)  # [F,J]
    rot = from_rotvec(a[..., None] * axes[None])
    yaw = from_rotvec(np.stack([np.zeros(n_frames), turn * t[:, 0], np.zeros(n_frames)], axis=-1))
    rot[:, 0] = qmul(yaw, rot[:, 0])
    s = t[:, 0]
    root = np.stack([start[0] + 0.4 * np.sin(2 * np.pi * s),
                     sk.rest_offsets[0, 1] + 0.05 * np.sin(4 * np.pi * s) - 0.05,
                     start[1] + 1.2 * s], axis=-1)
    return rot, root


def make_take(sk: Skeleton = SOMA_PROVISIONAL, n_frames: int = 48, seed: int = 0, fps: float = 30.0,
              role: str = "A", rest_offsets: np.ndarray | None = None, **kw) -> Take:
    rot, root = synthetic_motion(sk, n_frames, seed, **kw)
    return Take(skeleton=sk.name, fps=fps, speed_factor=1.0, rot=rot, root=root, role=role,
                provenance={"source": "synthetic", "seed": seed}, rest_offsets=rest_offsets)


def single_joint_motion(sk: Skeleton, joint_rot: dict[str, np.ndarray], root: np.ndarray) -> np.ndarray:
    """Local rotations [F,J,4]: identity except the given joints (each [F,4])."""
    n_frames = root.shape[0]
    rot = np.tile([0.0, 0.0, 0.0, 1.0], (n_frames, sk.num_joints, 1))
    for name, q in joint_rot.items():
        rot[:, sk.index(name)] = q
    return rot


def build_skeleton(name: str, joints: list[tuple[str, str | None, tuple[float, float, float]]],
                   rest_rotations: dict[str, np.ndarray] | None = None) -> Skeleton:
    names = tuple(j[0] for j in joints)
    idx = {n: i for i, n in enumerate(names)}
    parents = tuple(-1 if p is None else idx[p] for _, p, _ in joints)
    off = np.array([o for _, _, o in joints], dtype=np.float64)
    rr = None
    if rest_rotations:
        rr = np.tile([0.0, 0.0, 0.0, 1.0], (len(names), 1))
        for n, q in rest_rotations.items():
            rr[idx[n]] = q
    return Skeleton(name, names, parents, off, rr)


def provisional_joints() -> list[tuple[str, str | None, tuple[float, float, float]]]:
    sk = SOMA_PROVISIONAL
    return [
        (n, sk.joint_names[p] if p >= 0 else None, tuple(sk.rest_offsets[i]))
        for i, (n, p) in enumerate(zip(sk.joint_names, sk.parents, strict=True))
    ]
