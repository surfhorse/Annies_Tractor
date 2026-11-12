"""Rest-relative rotation transfer from a source skeleton onto the export rig.

For every rig bone b mapped to source joint s (world rotations from ``afv.motion.fk``)::

    Q_rig[f, b] = Q_src[f, s] * Q_src_tpose[s]^-1

i.e. the bone turns in world space exactly as its source joint turns away from the source's
T-pose. Because the rig's rest offsets are the source's T-pose offsets, every rig head whose source
joint is a *direct* child of the parent bone's joint lands exactly on the source joint, for any
pose. Unmapped source joints are folded: their rotation reaches every mapped descendant through
``Q_src`` (world space), so each child bone's orientation stays exact (the folded rotation ends up
in the child's local rotation).

Where a rig link spans folded joints:

- the parent has **one** rig child (a chain such as Neck -> [Neck2] -> Head, or a spine with an
  extra joint): the parent is re-aimed at the child's actual position (minimal swing), so the bend
  of the folded joint is spread over the rig chain; the child head is then exact up to the change
  of chord length (millimetres for ordinary bends);
- the parent has **several** children (Hand with folded metacarpals, a chest): the parent keeps its
  exact orientation; a folded child's head moves by (folded rotation x offset), e.g. knuckles when
  a metacarpal rotates. Re-aiming here would tilt the parent (the wrist) for every child.

Translations: the top non-root bone (Hips) carries the source FK position of its mapped joint; with
a root bone, ``root`` sits at the origin (see ``afv.export.root_motion`` for the other modes).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from afv.export._math import qinv_mul, rot_between
from afv.export.rig import ExportRig
from afv.motion.fk import forward_kinematics
from afv.motion.rotations import IDENTITY, make_continuous, qmul, qrotate


@dataclass
class RigMotion:
    """Animation on an ``ExportRig``: local rotations and local translations per bone and frame.

    ``loc[f, b]`` is bone b's head relative to its parent's head in the parent's frame (equal to
    ``rig.offsets[b]`` unless b is in ``translating``); for the top bone it is the world position.
    """

    rig: ExportRig
    fps: float
    rot: np.ndarray  # [F,B,4] xyzw, local (rest rotations are identity)
    loc: np.ndarray  # [F,B,3]
    translating: tuple[int, ...]
    root_motion: str = "world"

    @property
    def num_frames(self) -> int:
        return int(self.rot.shape[0])

    def copy(self, **changes) -> RigMotion:
        base = replace(self, rot=self.rot.copy(), loc=self.loc.copy())
        return replace(base, **changes) if changes else base


def rig_fk(m: RigMotion) -> tuple[np.ndarray, np.ndarray]:
    """World head positions [F,B,3] and world rotations [F,B,4] of the rig."""
    n_frames, n_bones = m.rot.shape[:2]
    pos = np.empty((n_frames, n_bones, 3))
    wrot = np.empty((n_frames, n_bones, 4))
    for b, p in enumerate(m.rig.parents):
        if p < 0:
            wrot[:, b] = m.rot[:, b]
            pos[:, b] = m.loc[:, b]
        else:
            wrot[:, b] = qmul(wrot[:, p], m.rot[:, b])
            pos[:, b] = pos[:, p] + qrotate(wrot[:, p], m.loc[:, b])
    return pos, wrot


def source_fk(rig: ExportRig, rot: np.ndarray, root: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return forward_kinematics(rig.source, rot, root)


def retarget(rig: ExportRig, rot: np.ndarray, root: np.ndarray, fps: float) -> RigMotion:
    """Transfer source local rotations ``rot`` [F,J,4] + root translation ``root`` [F,3] to the rig."""
    spos, srot = source_fk(rig, rot, root)
    n_frames = spos.shape[0]
    n_bones = rig.num_bones
    sidx = [rig.source.index(s) if s is not None else -1 for s in rig.source_joint]

    wq = np.empty((n_frames, n_bones, 4))
    for b in range(n_bones):
        if sidx[b] < 0:
            wq[:, b] = IDENTITY
        else:
            wq[:, b] = qmul(srot[:, sidx[b]], rig.source_tpose_rot[b] * np.array([-1.0, -1.0, -1.0, 1.0]))

    # Single-child parents of folded links are re-aimed so the rig chain follows the source chain;
    # multi-child parents stay exact (see module docstring).
    for b in reaimed_bones(rig):
        k = rig.children(b)[0]
        now = qrotate(wq[:, b], rig.offsets[k])
        wq[:, b] = qmul(rot_between(now, spos[:, sidx[k]] - spos[:, sidx[b]]), wq[:, b])

    wq = make_continuous(wq, axis=0)
    local = np.empty_like(wq)
    for b, p in enumerate(rig.parents):
        local[:, b] = wq[:, b] if p < 0 else qinv_mul(wq[:, p], wq[:, b])
    local = make_continuous(local, axis=0)

    loc = np.broadcast_to(rig.offsets, (n_frames, n_bones, 3)).copy()
    hips = rig.hips
    loc[:, hips] = spos[:, sidx[hips]]  # root (if any) is identity at the origin
    translating = (0, hips) if rig.has_root else (hips,)
    return RigMotion(rig, float(fps), local, loc, translating, "world")


def reaimed_bones(rig: ExportRig) -> list[int]:
    """Bones whose orientation is re-aimed (single rig child across folded source joints)."""
    return [
        b for b in range(rig.num_bones)
        if rig.source_joint[b] is not None and len(kids := rig.children(b)) == 1 and rig.compressed[kids[0]]
    ]


def mapped_source_positions(rig: ExportRig, rot: np.ndarray, root: np.ndarray) -> np.ndarray:
    """Source FK positions [F,B,3] at each rig bone's mapped joint (root bone: NaN)."""
    spos, _ = source_fk(rig, rot, root)
    out = np.full((spos.shape[0], rig.num_bones, 3), np.nan)
    for b, s in enumerate(rig.source_joint):
        if s is not None:
            out[:, b] = spos[:, rig.source.index(s)]
    return out
