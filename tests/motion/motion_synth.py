"""Synthetic motion with known ground truth for module E tests (no real footage).

* ``walk``: a walk on the provisional skeleton whose ankles follow a designed footstep plan
  (planted spans exactly still on the floor, swing arcs between), legs solved with IK.
* ``corrupt``: GEM-X-like damage: root skate drift, rotation jitter, root noise, a penetration dip
  and a floor offset; ``model_probs`` makes flickery contact probabilities from the truth.
* ``punch``: a standing jab/cross whose wrist speed peak is known from the clean motion.
* ``telephoto_camera`` / ``along_ray``: a broadcast-like camera and depth corruption along rays.
"""

from __future__ import annotations

import numpy as np

from afv.core.skeletons import SOMA_PROVISIONAL
from afv.core.types import ActorMotion, BodyMotion, CameraTrack
from afv.motion.fk import forward_kinematics
from afv.motion.ik import PROVISIONAL_ROLES, LegTarget, quat_axis_angle, quat_from_to, solve_leg_ik
from afv.motion.rotations import IDENTITY, from_matrix, qmul, slerp
from afv.motion.smooth import qexp

SK = SOMA_PROVISIONAL
ROLES = PROVISIONAL_ROLES
FPS = 30.0
GEMX_CHANNELS = ["left_ankle", "left_foot", "right_ankle", "right_foot", "left_wrist", "right_wrist"]
FOOT_CHANNELS = ["l_heel", "l_toe", "r_heel", "r_toe"]


def _yaw(angle: np.ndarray) -> np.ndarray:
    return quat_axis_angle(np.broadcast_to([0.0, 1.0, 0.0], (*np.shape(angle), 3)), angle)


def _arms_down(rot: np.ndarray, swing: np.ndarray | None = None) -> None:
    n = rot.shape[0]
    sw = np.zeros(n) if swing is None else swing
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        down = quat_axis_angle(np.tile([0.0, 0.0, 1.0], (n, 1)), np.full(n, -sgn * np.radians(75)))
        fwd = quat_axis_angle(np.tile([1.0, 0.0, 0.0], (n, 1)), sgn * sw)
        rot[:, SK.index(f"{side}_upperarm")] = qmul(fwd, down)
        rot[:, SK.index(f"{side}_forearm")] = quat_axis_angle(
            np.tile([0.0, 1.0, 0.0], (n, 1)), np.full(n, sgn * np.radians(20))
        )


def walk(
    seconds: float = 4.0,
    speed: float = 0.9,
    period: float = 1.0,
    stance: float = 0.6,
    origin: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    floor_y: float = 0.0,
    fps: float = FPS,
) -> tuple[ActorMotion, dict[str, np.ndarray]]:
    """Clean walk + ground-truth plant masks per foot channel (heel and toe share the stance)."""
    n = round(seconds * fps)
    t = np.arange(n) / fps
    c, s = np.cos(heading), np.sin(heading)
    rot_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def world(local: np.ndarray) -> np.ndarray:
        return local @ rot_y.T + np.array([origin[0], floor_y, origin[1]])

    pel = np.stack(
        [0.02 * np.sin(2 * np.pi * t / period), 0.86 + 0.01 * np.cos(4 * np.pi * t / period), speed * t],
        axis=1,
    )
    masks: dict[str, np.ndarray] = {}
    ankles = {}
    for side, x, off in (("l", 0.1, 0.0), ("r", -0.1, 0.5)):
        u = t / period + off
        cyc = np.floor(u)
        ph = u - cyc

        def fall(k: np.ndarray, off: float = off) -> np.ndarray:
            return speed * ((k - off + stance / 2) * period)

        st = ph < stance
        sw = np.clip((ph - stance) / (1 - stance), 0.0, 1.0)
        e = sw * sw * (3 - 2 * sw)
        z = np.where(st, fall(cyc), (1 - e) * fall(cyc) + e * fall(cyc + 1))
        y = 0.08 + np.where(st, 0.0, 0.08 * np.sin(np.pi * sw))
        ankles[side] = world(np.stack([np.full(n, x), y, z], axis=1))
        masks[f"{side}_heel"] = st.copy()
        masks[f"{side}_toe"] = st.copy()
    root = world(pel)
    rot = np.tile(IDENTITY, (n, SK.num_joints, 1))
    rot[:, 0] = qmul(_yaw(np.full(n, heading)), _yaw(0.08 * np.sin(2 * np.pi * t / period)))
    rot[:, SK.index("spine2")] = _yaw(-0.08 * np.sin(2 * np.pi * t / period))
    _arms_down(rot, 0.3 * np.sin(2 * np.pi * t / period))
    # Legs by IK onto the designed footstep plan.
    targets = [LegTarget(leg, ankles[s], np.ones(n, bool)) for s, leg in zip("lr", ROLES.legs, strict=True)]
    rot = solve_leg_ik(SK, rot, root, targets, forward=np.array(ROLES.forward))
    # Feet keep the heading only: set foot world rotation = heading yaw.
    _, wrot = forward_kinematics(SK, rot, root)
    for leg in ROLES.legs:
        ia, ip = SK.index(leg.foot), SK.parents[SK.index(leg.foot)]
        want = _yaw(np.full(n, heading))
        rot[:, ia] = qmul(np.array([-1, -1, -1, 1.0]) * wrot[:, ip], want)
    return ActorMotion(rot=rot, root=root), masks


def corrupt(
    actor: ActorMotion,
    rng: np.random.Generator,
    skate: float = 0.05,
    jitter_deg: float = 0.5,
    root_noise: float = 0.004,
    penetration: float = 0.03,
    pen_span: tuple[float, float] = (1.0, 2.0),
    floor_shift: float = 0.0,
    fps: float = FPS,
    foot_wobble_deg: float = 0.0,
) -> ActorMotion:
    """GEM-X-like damage. ``skate`` 0.12 gives ~0.6 m/s of root drift against planted feet (P0 saw
    12-65 cm/s after GEM-X's own IK); ``foot_wobble_deg`` adds a slow independent foot rotation, so
    heel and toe skate non-rigidly."""
    n = actor.rot.shape[0]
    t = np.arange(n) / fps
    root = np.array(actor.root, dtype=np.float64)
    root[:, 0] += skate * np.sin(2 * np.pi * t / 1.3)
    root[:, 2] += skate * np.cos(2 * np.pi * t / 1.7)
    w = np.clip((t - pen_span[0]) / (pen_span[1] - pen_span[0]), 0, 1)
    root[:, 1] -= penetration * np.sin(np.pi * w) ** 2
    root[:, 1] += floor_shift
    root += rng.normal(0, root_noise, root.shape)
    noise = rng.normal(0, np.radians(jitter_deg), (*actor.rot.shape[:2], 3))
    if foot_wobble_deg:
        for k, side in enumerate("lr"):
            ph = 2 * np.pi * t / (0.9 + 0.3 * k)
            noise[:, SK.index(f"{side}_foot")] += np.radians(foot_wobble_deg) * np.stack(
                [np.sin(ph), np.cos(1.3 * ph), 0.5 * np.sin(0.7 * ph)], axis=1
            )
    rot = qmul(actor.rot, qexp(noise))
    return ActorMotion(rot=rot, root=root, rest_offsets=actor.rest_offsets)


def model_probs(masks: dict[str, np.ndarray], rng: np.random.Generator, flicker: float = 0.1) -> np.ndarray:
    """GEM-X-style probabilities [F,6] in GEMX_CHANNELS order from ground-truth foot masks."""
    order = {"left_ankle": "l_heel", "left_foot": "l_toe", "right_ankle": "r_heel", "right_foot": "r_toe"}
    n = len(next(iter(masks.values())))
    cols = []
    for ch in GEMX_CHANNELS:
        if ch in order:
            m = masks[order[ch]]
            p = np.where(m, 0.85, 0.08)
            flip = rng.random(n) < flicker
            p = np.where(flip, np.where(m, 0.3, 0.6), p)
        else:
            p = np.full(n, 0.02)
        cols.append(p)
    return np.stack(cols, axis=1).astype(np.float32)


def body(actors: dict[str, ActorMotion], channels: list[str] | None = None, fps: float = FPS) -> BodyMotion:
    n = next(iter(actors.values())).rot.shape[0]
    return BodyMotion(
        frames=np.arange(n),
        timestamps=np.arange(n) / fps,
        skeleton=SK.name,
        fps=fps,
        contact_channels=list(channels or []),
        actors=actors,
    )


def punch(
    seconds: float = 2.0, t_punch: float = 1.0, ext_frames: int = 4, retract_frames: int = 7, fps: float = FPS
) -> ActorMotion:
    """Standing fighter throwing a right straight: guard -> full extension in ``ext_frames``."""
    n = round(seconds * fps)
    f = np.arange(n, dtype=np.float64)
    f0 = t_punch * fps
    a = np.clip((f - f0) / ext_frames, 0, 1)
    b = np.clip((f - (f0 + ext_frames + 2)) / retract_frames, 0, 1)
    mj = lambda x: x**3 * (10 - 15 * x + 6 * x * x)  # noqa: E731  minimum-jerk profile
    s = mj(a) * (1 - mj(b))
    rot = np.tile(IDENTITY, (n, SK.num_joints, 1))
    _arms_down(rot)
    guard_dir = np.array([-0.35, -0.55, 0.55])
    ua_guard = quat_from_to(np.array([-1.0, 0, 0]), guard_dir)
    ua_ext = quat_from_to(np.array([-1.0, 0, 0]), np.array([0.05, 0.05, 1.0]))
    fa_guard = quat_axis_angle(np.array([0.0, 1.0, 0.0]), np.radians(115.0))
    rot[:, SK.index("r_upperarm")] = slerp(
        np.broadcast_to(ua_guard, (n, 4)), np.broadcast_to(ua_ext, (n, 4)), s
    )
    rot[:, SK.index("r_forearm")] = slerp(
        np.broadcast_to(fa_guard, (n, 4)), np.broadcast_to(IDENTITY, (n, 4)), s
    )
    rot[:, SK.index("spine2")] = _yaw(np.radians(25.0) * s)
    # Slightly bent knees, standing on the floor.
    root = np.tile([0.0, 0.93, 0.0], (n, 1))
    ankles = {side: np.tile([x, 0.08, 0.0], (n, 1)) for side, x in (("l", 0.12), ("r", -0.12))}
    targets = [LegTarget(leg, ankles[sd], np.ones(n, bool)) for sd, leg in zip("lr", ROLES.legs, strict=True)]
    rot = solve_leg_ik(SK, rot, root, targets, forward=np.array(ROLES.forward))
    return ActorMotion(rot=rot, root=root)


def look_at_camera(
    pos: np.ndarray, target: np.ndarray, focal: float, n: int, w: int = 1920, h: int = 1080
) -> CameraTrack:
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    z = -fwd
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = from_matrix(np.stack([x, y, z], axis=1))
    return CameraTrack(
        frames=np.arange(n),
        timestamps=np.arange(n) / FPS,
        cam_rot=np.tile(q, (n, 1)),
        cam_pos=np.tile(pos, (n, 1)),
        focal_px=np.full(n, focal),
        width=w,
        height=h,
        cx=w / 2,
        cy=h / 2,
    )


def telephoto_camera(n: int) -> CameraTrack:
    """Broadcast-like: 22 m away, 3.5 m up, focal 7000 px at 1080p (P0's telephoto clips)."""
    return look_at_camera(np.array([0.0, 3.5, 22.0]), np.array([0.0, 1.0, 0.0]), 7000.0, n)


def along_ray(actor: ActorMotion, cam_pos: np.ndarray, dz: np.ndarray) -> ActorMotion:
    """Translate the body by ``dz`` [F] metres along the camera ray through its root."""
    root = np.asarray(actor.root, dtype=np.float64)
    u = root - cam_pos
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    return ActorMotion(
        rot=actor.rot, root=root + dz[:, None] * u, contact=actor.contact, rest_offsets=actor.rest_offsets
    )
