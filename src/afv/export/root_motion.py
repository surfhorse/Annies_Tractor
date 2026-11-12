"""Root-motion modes for an exported take (spec: FR export, engine presets).

- ``world``: the whole world trajectory stays on Hips; ``root`` (if present) is still at the origin.
- ``in_place``: Hips' horizontal translation is frozen at its frame-0 value (so paired fighters
  still start where they stood); vertical motion and all rotations are kept.
- ``extracted``: needs a ``root`` bone. ``root`` takes the hips' ground projection (on the floor,
  y = ``floor_y``) and the body heading (yaw about +Y); Hips keeps the remainder in root space,
  so every joint's world position is unchanged. Vertical motion stays on Hips in every mode.

Floor: the contacts stage guarantees the floor is y = 0 (module E), so ``floor_y`` defaults to 0.

Heading (``robust_heading``) comes from the hip line (RightUpLeg -> LeftUpLeg), turned 90 degrees
about +Y: it stays horizontal when the pelvis pitches (bending over, lying on the back, a flip),
where the pelvis forward axis would point up or down and its yaw would jump by 180 degrees. When the
hip line itself tilts towards vertical (lying on a side, cartwheels) the pelvis forward axis takes
over. Each frame's heading is a 2-D vector weighted by how horizontal its source axis is; the
vectors are low-pass filtered with those weights (Gaussian, ``yaw_smooth_s``; long enough that a
punch's hip twist barely moves the root), and frames with no reliable heading hold the previous
one. Filtering only moves motion between root and Hips: it changes how engines turn the capsule,
not the pose. The ground track is filtered the same way with ``smooth_s``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d

from afv.export._math import qinv_mul, yaw_quat
from afv.export.retarget import RigMotion, rig_fk
from afv.motion.rotations import IDENTITY, make_continuous, qconj, qrotate

RootMotionMode = Literal["in_place", "world", "extracted"]
ROOT_MOTION_MODES: tuple[str, ...] = ("in_place", "world", "extracted")

FORWARD = np.array([0.0, 0.0, 1.0])  # character forward at rest (internal frame)
_LATERAL_FULL = 0.5  # hip line horizontal fraction from which it alone gives the heading
_LATERAL_NONE = 0.25  # ... below which only the pelvis forward axis is used
_MIN_WEIGHT = 0.05  # weaker heading evidence than this: hold the previous heading


def _smooth(x: np.ndarray, sigma_frames: float) -> np.ndarray:
    if sigma_frames <= 0.0 or x.shape[0] < 3:
        return x
    return gaussian_filter1d(x, sigma_frames, axis=0, mode="nearest")


def heading_vectors(left: np.ndarray, hips_rot: np.ndarray) -> np.ndarray:
    """Per-frame heading evidence [F,2] as (x, z); its length is the weight (0..1).

    ``left`` [F,3]: the hip line pointing to the character's left (RightUpLeg -> LeftUpLeg);
    ``hips_rot`` [F,4]: Hips world rotation (rest = facing +Z).
    """
    left = np.asarray(left, dtype=np.float64)
    left = left / np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-12)
    from_lateral = np.stack([-left[:, 2], left[:, 0]], axis=-1)  # left x up = forward
    fwd = qrotate(hips_rot, FORWARD)
    from_forward = np.stack([fwd[:, 0], fwd[:, 2]], axis=-1)
    lat_len = np.linalg.norm(from_lateral, axis=-1, keepdims=True)
    a = np.clip((lat_len - _LATERAL_NONE) / (_LATERAL_FULL - _LATERAL_NONE), 0.0, 1.0)
    return a * from_lateral + (1.0 - a) * from_forward


def robust_heading(vectors: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Unwrapped yaw [F] (radians) from weighted heading vectors [F,2] (see ``heading_vectors``)."""
    v = _smooth(np.asarray(vectors, dtype=np.float64), sigma_frames)
    w = np.linalg.norm(v, axis=-1)
    yaw = np.arctan2(v[:, 0], v[:, 1])
    good = w >= _MIN_WEIGHT
    if not good.any():
        return np.zeros(len(v))
    first = int(np.argmax(good))
    yaw[:first] = yaw[first]
    for i in range(first + 1, len(v)):
        if not good[i]:
            yaw[i] = yaw[i - 1]
    return np.unwrap(yaw)


def heading(m: RigMotion, pos: np.ndarray, wrot: np.ndarray, yaw_smooth_s: float) -> np.ndarray:
    """Body heading [F] (radians, unwrapped) of a rig motion, from its FK ``pos``/``wrot``."""
    rig = m.rig
    lu, ru = rig.index("LeftUpLeg"), rig.index("RightUpLeg")
    rest = rig.rest_positions()
    rest_v = heading_vectors((rest[lu] - rest[ru])[None], IDENTITY[None])[0]
    rest_yaw = float(np.arctan2(rest_v[0], rest_v[1]))  # hip line not exactly along +X at rest
    vec = heading_vectors(pos[:, lu] - pos[:, ru], wrot[:, rig.hips])
    return robust_heading(vec, yaw_smooth_s * m.fps) - rest_yaw


def apply_root_motion(
    m: RigMotion,
    mode: str,
    *,
    smooth_s: float = 0.2,
    yaw_smooth_s: float = 0.5,
    yaw: bool = True,
    floor_y: float = 0.0,
) -> RigMotion:
    """Return a copy of ``m`` (retargeted in ``world`` mode) with the requested root-motion mode."""
    if mode not in ROOT_MOTION_MODES:
        raise ValueError(f"root motion {mode!r}; expected one of {ROOT_MOTION_MODES}")
    if m.root_motion != "world":
        raise ValueError("apply_root_motion expects a 'world' motion (straight from retarget)")
    out = m.copy(root_motion=mode)
    rig = m.rig
    h = rig.hips
    if mode == "world":
        return out
    if mode == "in_place":
        # With a root bone the root is identity at the origin, so Hips' loc is its world position.
        out.loc[:, h, 0] = m.loc[0, h, 0]
        out.loc[:, h, 2] = m.loc[0, h, 2]
        return out
    if not rig.has_root:
        raise ValueError("root motion 'extracted' needs a rig with a root bone (choose a preset with one)")
    pos, wrot = rig_fk(m)
    track = np.stack([pos[:, h, 0], np.full(m.num_frames, floor_y), pos[:, h, 2]], axis=-1)
    track[:, [0, 2]] = _smooth(track[:, [0, 2]], smooth_s * m.fps)
    if yaw:
        root_rot = yaw_quat(heading(m, pos, wrot, yaw_smooth_s))
    else:
        root_rot = np.tile(IDENTITY, (m.num_frames, 1))
    out.loc[:, 0] = track
    out.rot[:, 0] = root_rot
    out.loc[:, h] = qrotate(qconj(root_rot), pos[:, h] - track)
    out.rot[:, h] = make_continuous(qinv_mul(root_rot, wrot[:, h]), axis=0)
    return out
