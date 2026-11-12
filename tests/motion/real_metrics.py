"""Metrics for module E on real GEM-X data (shared by the real-data tests and the review table)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from afv.core.types import BodyMotion, CameraTrack, load_result
from afv.motion.contacts import actor_skeleton, contact_masks
from afv.motion.fk import forward_kinematics
from afv.motion.ik import get_roles, knee_hinge_local
from afv.motion.rotations import qrotate
from afv.motion.smooth import effector_speed
from afv.qc.camera import project

REPO = Path(__file__).resolve().parents[2]
FIXTURES = {
    "ufc_cross": REPO / "tests/motion/fixtures",
    "close_boxing": REPO / "tests/motion/fixtures",
    "ref_walk": REPO / "tests/motion/fixtures",
    "who_am_i": REPO / "tests/models/fixtures",  # module C's full-rate fixture
}


def load_clip(name: str) -> tuple[BodyMotion, CameraTrack]:
    d = FIXTURES[name]
    bm = load_result(d / f"pose3d_{name}.npz")
    cam = load_result(d / f"camera_{name}.npz")
    assert isinstance(bm, BodyMotion) and isinstance(cam, CameraTrack)
    return bm, cam


def positions(bm: BodyMotion, role: str):
    a = bm.actors[role]
    sk = actor_skeleton(bm, a)
    return sk, forward_kinematics(sk, a.rot, a.root)[0]


def plant_speed(bm: BodyMotion, role: str) -> np.ndarray:
    """Foot speeds (m/s) of frame pairs inside the result's own plants (NFR-3 measure)."""
    roles = get_roles(actor_skeleton(bm, bm.actors[role]))
    sk, pos = positions(bm, role)
    masks = contact_masks(bm, bm.actors[role], roles)
    out = []
    for c in roles.contacts:
        m = masks[c.channel]
        both = m[1:] & m[:-1]
        out.append(np.linalg.norm(np.diff(pos[:, sk.index(c.joint)], axis=0), axis=-1)[both] * bm.fps)
    return np.concatenate(out)


def knee_change(bm: BodyMotion, role: str) -> float:
    """Largest knee angle change between consecutive frames (degrees)."""
    sk, pos = positions(bm, role)
    roles = get_roles(sk)
    worst = 0.0
    for leg in roles.legs:
        h, k, a = (pos[:, sk.index(j)] for j in (leg.thigh, leg.calf, leg.foot))
        t, c = k - h, a - k
        cos = np.sum(t * c, -1) / np.linalg.norm(t, axis=-1) / np.linalg.norm(c, axis=-1)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        worst = max(worst, float(np.abs(np.diff(ang)).max()))
    return worst


def hips_px(bm: BodyMotion, cam: CameraTrack, role: str) -> np.ndarray:
    """Hips pixel position per frame, reprojected with cam_pos + world_shift."""
    sk, pos = positions(bm, role)
    return project(pos[:, [sk.index(get_roles(sk).pelvis)]] - np.asarray(bm.world_shift), cam, bm.frames)[
        :, 0
    ]


def strike_ratios(
    before: BodyMotion, after: BodyMotion, role: str, effectors: tuple[str, ...], min_speed: float = 3.0
) -> np.ndarray:
    """For every peak (+-2 frames) of pelvis-relative effector speed above ``min_speed`` in
    ``before``: the peak speed after (max within +-2 frames) / before."""
    sk, p0 = positions(before, role)
    _, p1 = positions(after, role)
    pel = sk.index(get_roles(sk).pelvis)
    out = []
    for e in effectors:
        j = sk.index(e)
        v0 = effector_speed(p0[:, j] - p0[:, pel], before.fps)
        v1 = effector_speed(p1[:, j] - p1[:, pel], after.fps)
        for t in range(2, len(v0) - 2):
            if v0[t] > min_speed and v0[t] == v0[t - 2 : t + 3].max():
                out.append(v1[t - 2 : t + 3].max() / v0[t])
    return np.asarray(out)


def signed_knee(bm: BodyMotion, role: str, ref: BodyMotion) -> list[np.ndarray]:
    """Signed knee bend per leg [F] (degrees; > 0 = the input's usual bend side, < 0 = backwards),
    about the knee hinge estimated from ``ref`` (the input motion)."""
    sk, pos = positions(bm, role)
    roles = get_roles(sk)
    a, ar = bm.actors[role], ref.actors[role]
    wrot = forward_kinematics(sk, a.rot, a.root)[1]
    p_in, w_in = forward_kinematics(sk, ar.rot, ar.root)
    out = []
    for leg in roles.legs:
        hinge = knee_hinge_local(sk, w_in, p_in, leg, np.asarray(roles.forward))
        h, k, an = (pos[:, sk.index(j)] for j in (leg.thigh, leg.calf, leg.foot))
        t, c = k - h, an - k
        ax = qrotate(wrot[:, sk.index(leg.thigh)], np.broadcast_to(hinge, (len(pos), 3)))
        s = np.sum(np.cross(t, c) * ax, -1) / np.linalg.norm(t, axis=-1) / np.linalg.norm(c, axis=-1)
        co = np.sum(t * c, -1) / np.linalg.norm(t, axis=-1) / np.linalg.norm(c, axis=-1)
        out.append(np.degrees(np.arctan2(s, co)))
    return out
