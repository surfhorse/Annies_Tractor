"""Per-frame QC metrics on a ``BodyMotion`` (feeds the Blender confidence strip and stage meta).

All metrics are per fighter and per frame on the motion's own timeline:

``contact_<ch>``        1 where the channel is planted (from the result's contact channels)
``foot_speed_<ch>``     m/s of the channel joint between this frame and the previous one
``sole_height_<ch>``    m of the channel's sole point above the floor (y = 0 unless given)
``foot_contact_<foot>`` 1 where any channel of that foot (the leg's foot joint, e.g. LeftFoot) is planted
``foot_sole_height_<foot>`` m of the foot's lowest sole point (heel or ball) above the floor: what
                        touches it (``afv.motion.contacts.foot_sole_heights``, the solver's measure)
``skate``               max foot speed over planted channels (only frame pairs both planted), m/s
``penetration``         max depth of any sole point below the floor, m (>= 0)
``accel``               mean joint acceleration magnitude, m/s^2
``pops``                number of joints with a single-frame pop: > 3 cm off the neighbours'
                        midpoint, out of line with both neighbours' motion and isolated
                        (``afv.motion.smooth.position_spikes``)
``speed_<effector>``    end-effector (wrist/ankle) speed, m/s
``reproj_px``           mean reprojection error over visible matched keypoints (camera + 2D given)
``foot_pops``           (``ref`` given: the stage's input motion) number of foot/toe joints lurching
                        here against the input (``afv.motion.contacts.foot_lurches``: 3-step
                        displacement > ``LURCH_M`` longer than the input's and ``LURCH_RATIO`` x it)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from afv.core.skeletons import get_skeleton
from afv.core.types import BodyMotion, CameraTrack, Keypoints2D
from afv.motion.contacts import (
    Floor,
    actor_skeleton,
    channel_sole_heights,
    contact_masks,
    foot_lurches,
    foot_sole_heights,
)
from afv.motion.fk import forward_kinematics
from afv.motion.ik import BodyRoles, get_roles
from afv.motion.smooth import position_spikes
from afv.qc.camera import project

# A "visible" single-frame pop: > 3 cm out of line for one frame. Ordinary swing-leg acceleration
# already deviates ~2 cm from the neighbours' midpoint at 30 Hz (a 4 m/s foot), so a lower bar would
# count normal motion; injected 15-degree pops on a spine or limb joint deviate 5-14 cm.
POP_THRESHOLD = 0.03


def step_speed(pos: np.ndarray, fps: float) -> np.ndarray:
    """Speed [F] from frame t-1 to t (0 at frame 0)."""
    sp = np.zeros(len(pos))
    if len(pos) > 1:
        sp[1:] = np.linalg.norm(np.diff(np.asarray(pos, dtype=np.float64), axis=0), axis=-1) * fps
    return sp


def contact_speed(pos: np.ndarray, mask: np.ndarray, fps: float) -> np.ndarray:
    """Speeds of frame pairs (t-1, t) that are both in contact (the NFR-3 measure)."""
    mask = np.asarray(mask, dtype=bool)
    both = np.zeros_like(mask)
    both[1:] = mask[1:] & mask[:-1]
    return step_speed(pos, fps)[both]


def reprojection_error(
    joints: np.ndarray,
    joint_names: tuple[str, ...] | list[str],
    cam: CameraTrack,
    frames: np.ndarray,
    kp: np.ndarray,
    kp_conf: np.ndarray,
    kp_names: list[str],
    min_conf: float = 0.3,
    name_map: dict[str, str] | None = None,
    world_shift: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Mean pixel error per frame [F] between projected joints and 2D keypoints (NaN if none).

    Keypoints are matched to joints by name (or ``name_map`` keypoint -> joint).
    ``world_shift`` (``BodyMotion.world_shift``) is the translation from the camera's world to the
    joints' world; projecting ``joints - world_shift`` equals using ``cam_pos + world_shift``.
    """
    name_map = name_map or {}
    pairs = [
        (ki, list(joint_names).index(name_map.get(k, k)))
        for ki, k in enumerate(kp_names)
        if name_map.get(k, k) in joint_names
    ]
    if not pairs:
        return np.full(len(frames), np.nan)
    ki, ji = np.array(pairs).T
    ws = np.zeros(3) if world_shift is None else np.asarray(world_shift, dtype=np.float64)
    uv = project(joints[:, ji] - ws, cam, frames)
    err = np.linalg.norm(uv - kp[:, ki], axis=-1)
    vis = (kp_conf[:, ki] >= min_conf) & np.isfinite(err)
    with np.errstate(invalid="ignore"):
        return np.where(vis.any(1), np.nansum(np.where(vis, err, 0.0), 1) / np.maximum(vis.sum(1), 1), np.nan)


def per_frame_metrics(
    bm: BodyMotion,
    roles: BodyRoles | None = None,
    floor: Floor | None = None,
    camera: CameraTrack | None = None,
    kp2d: Keypoints2D | None = None,
    kp_name_map: dict[str, str] | None = None,
    ref: BodyMotion | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-frame metrics per fighter (see module docstring). ``ref``: the stage's input motion (same
    frames), for ``foot_pops``.

    ``floor`` defaults to y = 0, the project convention after the pair and contacts stages. (It used
    to be re-fitted from the planted soles, which a plant pinned above the floor biased upwards and
    turned into false penetration of the other plants: 0.44 cm on close boxing A.)"""
    roles = roles or get_roles(get_skeleton(bm.skeleton))
    floor = floor or Floor.horizontal(0.0)
    fps = bm.fps or 30.0
    out: dict[str, dict[str, np.ndarray]] = {}
    for r, a in bm.actors.items():
        sk = actor_skeleton(bm, a)
        pos, _ = forward_kinematics(sk, a.rot, a.root)
        sole = channel_sole_heights(sk, roles, pos, floor)
        masks = contact_masks(bm, a, roles)
        m: dict[str, np.ndarray] = {}
        skate = np.zeros(bm.num_frames)
        pen = np.zeros(bm.num_frames)
        for c in roles.contacts:
            p = pos[:, sk.index(c.joint)]
            mk = masks[c.channel]
            sp = step_speed(p, fps)
            h = sole[c.channel]
            m[f"contact_{c.channel}"] = mk.astype(np.float32)
            m[f"foot_speed_{c.channel}"] = sp
            m[f"sole_height_{c.channel}"] = h
            both = np.zeros_like(mk)
            both[1:] = mk[1:] & mk[:-1]
            skate = np.maximum(skate, np.where(both, sp, 0.0))
            pen = np.maximum(pen, np.maximum(0.0, -h))
        lowest = foot_sole_heights(sk, roles, pos, floor)
        for li, leg in enumerate(roles.legs):
            if not any(c.leg == li for c in roles.contacts):
                continue  # no sole point on this leg
            planted = np.zeros(bm.num_frames, dtype=bool)
            for c in roles.contacts:
                if c.leg == li:
                    planted |= masks[c.channel]
            m[f"foot_contact_{leg.foot}"] = planted.astype(np.float32)
            m[f"foot_sole_height_{leg.foot}"] = lowest[li]
        if ref is not None and r in ref.actors and ref.num_frames == bm.num_frames:
            ra = ref.actors[r]
            p_in = forward_kinematics(actor_skeleton(ref, ra), ra.rot, ra.root)[0]
            fp = np.zeros(bm.num_frames)
            for leg in roles.legs:
                for jn in (leg.foot, leg.toe):
                    if jn is None:
                        continue
                    fp += foot_lurches(pos[:, sk.index(jn)], p_in[:, sk.index(jn)])
            m["foot_pops"] = fp
        m["skate"] = skate
        m["penetration"] = pen
        acc = np.zeros(bm.num_frames)
        if bm.num_frames >= 3:
            acc[1:-1] = np.mean(np.linalg.norm(pos[2:] - 2 * pos[1:-1] + pos[:-2], axis=-1), axis=1) * fps**2
        m["accel"] = acc
        m["pops"] = position_spikes(pos, POP_THRESHOLD).sum(axis=1).astype(np.float64)
        for e in dict.fromkeys((*roles.wrists, *roles.ankles)):
            m[f"speed_{e}"] = step_speed(pos[:, sk.index(e)], fps)
        if camera is not None and kp2d is not None and r in kp2d.actors:
            ak = kp2d.actors[r]
            m["reproj_px"] = reprojection_error(
                pos,
                sk.joint_names,
                camera,
                bm.frames,
                ak.kp,
                ak.conf,
                kp2d.keypoint_names,
                name_map=kp_name_map,
                world_shift=bm.world_shift,
            )
        out[r] = m
    return out


def summarise(metrics: dict[str, dict[str, np.ndarray]], roles: BodyRoles | None = None) -> dict[str, Any]:
    """JSON-safe summary per fighter: the NFR-3/NFR-4/penetration/reprojection numbers."""
    out: dict[str, Any] = {}
    for r, m in metrics.items():
        speeds = []
        pen_in = []
        joint_hover_in = []
        hover_in = []
        for k in m:
            if k.startswith("contact_"):
                ch = k[len("contact_") :]
                mk = m[k] > 0.5
                both = np.zeros_like(mk)
                both[1:] = mk[1:] & mk[:-1]
                speeds.append(m[f"foot_speed_{ch}"][both])
                pen_in.append(np.maximum(0.0, -m[f"sole_height_{ch}"][mk]))
                joint_hover_in.append(np.maximum(0.0, m[f"sole_height_{ch}"][mk]))
            elif k.startswith("foot_contact_"):
                foot = k[len("foot_contact_") :]
                hover_in.append(np.maximum(0.0, m[f"foot_sole_height_{foot}"][m[k] > 0.5]))
        sp = np.concatenate(speeds) if speeds else np.zeros(0)
        pi = np.concatenate(pen_in) if pen_in else np.zeros(0)
        hv = np.concatenate(hover_in) if hover_in else np.zeros(0)
        jh = np.concatenate(joint_hover_in) if joint_hover_in else np.zeros(0)
        has_foot = any(k.startswith("foot_contact_") for k in m) or not any(
            k.startswith("contact_") for k in m
        )
        s: dict[str, Any] = {
            "contact_frames": int(sum(int((m[k] > 0.5).sum()) for k in m if k.startswith("contact_"))),
            "mean_contact_foot_speed_cm_s": float(sp.mean() * 100) if sp.size else 0.0,
            "p95_contact_foot_speed_cm_s": float(np.percentile(sp, 95) * 100) if sp.size else 0.0,
            "mean_penetration_in_contact_cm": float(pi.mean() * 100) if pi.size else 0.0,
            "max_penetration_cm": float(m["penetration"].max() * 100) if m["penetration"].size else 0.0,
            # Planted feet must not float: height of the foot's lowest sole point (heel or ball, what
            # touches the floor) on frames where any of its channels is planted. The contacts stage
            # keeps it <= ``max_plant_hover`` (2 cm, + 1 mm rounding).
            # None when the metrics predate the foot-level measure (no ``foot_*`` keys).
            "mean_hover_in_contact_cm": (float(hv.mean() * 100) if hv.size else 0.0) if has_foot else None,
            "p95_hover_in_contact_cm": (
                (float(np.percentile(hv, 95) * 100) if hv.size else 0.0) if has_foot else None
            ),
            "max_hover_in_contact_cm": (float(hv.max() * 100) if hv.size else 0.0) if has_foot else None,
            # Per planted channel (its own sole point): a heel lifted off a planted ball counts here,
            # up to ``max_pin_raise`` (+ ``max_pin_miss``).
            "max_joint_hover_in_contact_cm": float(jh.max() * 100) if jh.size else 0.0,
            "pop_frames": int((m["pops"] > 0).sum()),
            "mean_accel_m_s2": float(m["accel"].mean()) if m["accel"].size else 0.0,
            "peak_effector_speed_m_s": {
                k[len("speed_") :]: float(m[k].max()) if m[k].size else 0.0
                for k in m
                if k.startswith("speed_")
            },
        }
        if "foot_pops" in m:
            s["foot_pop_frames"] = int((m["foot_pops"] > 0).sum())
        if "reproj_px" in m:
            s["mean_reproj_px"] = (
                float(np.nanmean(m["reproj_px"])) if np.isfinite(m["reproj_px"]).any() else None
            )
        out[r] = s
    return out
