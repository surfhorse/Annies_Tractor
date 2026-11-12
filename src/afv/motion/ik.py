"""Joint-role config and analytic two-bone leg IK.

Joint roles
-----------
Motion processing never uses joint indices. Everything it needs to know about a skeleton is in a
small :class:`BodyRoles` object that names the joints by role: pelvis, the leg chains (thigh,
calf, foot, toe), strike effectors (wrists, ankles), contact channels (which joint each named
contact channel pins and which input channels, e.g. GEM-X's, feed it) and body capsules for the
two-fighter push-apart. Roles are registered per skeleton name (:func:`register_roles`); for an
unregistered skeleton :func:`get_roles` guesses them from common naming schemes (afv
``l_thigh``/``l_calf``, Mixamo/SOMA ``LeftUpLeg``/``LeftLeg``/``LeftFoot``/``LeftToeBase``), so the
real SOMA skeleton can be plugged in with ``register_roles("soma77", BodyRoles.from_dict(...))``
or, if its names follow the Mixamo pattern, with nothing at all.

Two-bone IK
-----------
:func:`solve_leg_ik` places the foot joint (ankle) of a thigh/calf/foot chain exactly on a world
target, frame by frame, fully vectorised over frames:

1. **Knee hinge.** The knee bends about a hinge axis fixed in the thigh's frame. It is estimated
   from the data itself (mean of ``cross(thigh, calf)`` expressed in the thigh frame, weighted by
   bend) so the bend direction is whatever the input motion uses; a straight-legged input falls
   back to "knee points forward" from the skeleton's rest forward axis.
2. **Knee angle.** Rotating the calf about that axis changes the hip-ankle distance as
   ``d^2 = l1^2 + l2^2 + 2 l1 |c_perp| cos(phi)``; solving for ``phi`` gives the exact knee angle for
   the target distance (law of cosines, generalised to an axis not exactly perpendicular to the
   calf). The distance is clamped to the reachable range.
3. **Swing.** The thigh is rotated by the *minimal* rotation taking the new hip->ankle vector onto
   hip->target. A minimal swing keeps the knee's pole direction (the plane the leg bends in) as close
   as possible to the input, so knees do not flip or twist.
4. **Foot.** The foot keeps its input *world* orientation (no wobble from the leg change), then is
   optionally rotated about the ankle by the minimal rotation that brings the toe joint onto a toe
   target (toe pin) and/or out of the floor.

Local rotations are rebuilt along the actual parent chain from thigh to foot, so intermediate
joints (e.g. twist joints) between thigh, calf and foot are carried along unchanged in local
space.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from afv.core.types import Skeleton
from afv.motion.fk import forward_kinematics
from afv.motion.rotations import IDENTITY, qconj, qmul, qrotate

# ---------------------------------------------------------------------------
# Joint roles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegRoles:
    """One leg chain, by joint name. ``toe`` is optional (no toe pin without it)."""

    thigh: str
    calf: str
    foot: str
    toe: str | None = None


@dataclass(frozen=True)
class ContactRole:
    """A named contact channel: the joint it pins, its leg, and which input channels feed it.

    ``kind`` "heel" pins the leg's foot (ankle) joint, "toe" pins the leg's toe joint.
    ``sources`` lists input contact-channel names (e.g. GEM-X's ``left_ankle``) whose probability
    is used as model evidence for this channel; the channel's own name always counts too.
    """

    channel: str
    joint: str
    leg: int
    kind: Literal["heel", "toe"]
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class Capsule:
    """A body capsule between two joints for inter-fighter push-apart. ``group``: core/arm/leg."""

    a: str
    b: str
    radius: float
    group: str = "core"


REGIONS = ("torso", "head", "arms", "hands", "legs")

_REST_CACHE: dict[tuple, np.ndarray] = {}


def _rest_positions(sk: Skeleton) -> np.ndarray:
    """``afv.motion.fk.rest_positions`` cached per skeleton (contents): the contacts stage asks for the
    clearances some 50 times per fighter, each a full FK pass (0.6 s of 2.7 s on ufc_cross)."""
    from afv.motion.fk import rest_positions

    rr = sk.rest_rotations
    key = (
        sk.name,
        tuple(sk.parents),
        np.asarray(sk.rest_offsets, np.float64).tobytes(),
        None if rr is None else np.asarray(rr, np.float64).tobytes(),
    )
    rp = _REST_CACHE.get(key)
    if rp is None:
        if len(_REST_CACHE) >= 64:
            _REST_CACHE.clear()
        rp = rest_positions(sk)
        rp.setflags(write=False)
        _REST_CACHE[key] = rp
    return rp


@dataclass(frozen=True)
class BodyRoles:
    """Everything motion processing needs to know about a skeleton, by joint name.

    ``forward``/``up`` are the character's forward and up axes in the rest pose (afv: +Z, +Y).
    ``rest_floor`` is the floor height in the rest pose; None means "the lowest contact joint minus
    2 cm" (joints sit inside the foot, above the sole). ``clearance`` optionally overrides the
    height of a contact channel's joint above the floor when the foot is flat (metres).
    """

    pelvis: str
    legs: tuple[LegRoles, ...]
    contacts: tuple[ContactRole, ...]
    wrists: tuple[str, ...] = ()
    ankles: tuple[str, ...] = ()
    arm_roots: tuple[str, ...] = ()
    head: str | None = None
    capsules: tuple[Capsule, ...] = ()
    forward: tuple[float, float, float] = (0.0, 0.0, 1.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    rest_floor: float | None = None
    clearance: dict[str, float] = field(default_factory=dict, hash=False, compare=False)

    # -- validation and derived data -------------------------------------------------------
    def joints(self) -> list[str]:
        names = [self.pelvis, *self.wrists, *self.ankles, *self.arm_roots]
        for leg in self.legs:
            names += [leg.thigh, leg.calf, leg.foot] + ([leg.toe] if leg.toe else [])
        names += [c.joint for c in self.contacts]
        names += [n for cap in self.capsules for n in (cap.a, cap.b)]
        if self.head:
            names.append(self.head)
        return names

    def validate(self, sk: Skeleton) -> None:
        missing = sorted({n for n in self.joints() if n not in sk.joint_names})
        if missing:
            raise KeyError(f"roles reference joints missing from skeleton {sk.name!r}: {missing}")
        for c in self.contacts:
            leg = self.legs[c.leg]
            want = leg.foot if c.kind == "heel" else leg.toe
            if c.joint != want:
                raise ValueError(
                    f"contact {c.channel}: joint {c.joint} must be leg {c.leg}'s {c.kind} {want}"
                )
        for leg in self.legs:
            for child, anc in ((leg.calf, leg.thigh), (leg.foot, leg.calf)):
                if anc not in ancestors(sk, child):
                    raise ValueError(f"{anc} is not an ancestor of {child} in {sk.name}")

    def floor_height_rest(self, sk: Skeleton) -> float:
        if self.rest_floor is not None:
            return float(self.rest_floor)
        rp = _rest_positions(sk)
        up = np.asarray(self.up, dtype=np.float64)
        return float(min(rp[sk.index(c.joint)] @ up for c in self.contacts) - 0.02)

    def clearances(self, sk: Skeleton) -> dict[str, float]:
        """Height of each contact channel's joint above the floor with the foot flat (rest pose)."""
        rp = _rest_positions(sk)
        up = np.asarray(self.up, dtype=np.float64)
        floor = self.floor_height_rest(sk)
        out = {c.channel: float(rp[sk.index(c.joint)] @ up - floor) for c in self.contacts}
        out.update({k: float(v) for k, v in self.clearance.items()})
        return out

    def side_channels(self, side: str) -> list[str]:
        """Contact channels of one foot by side ("left"/"l"/"right"/"r"/leg index), via ``legs``
        order (legs[0] = left). Raises if the side names no channel."""
        s = str(side).strip().lower()
        leg = 0 if s in ("left", "l", "0") else 1 if s in ("right", "r", "1") else None
        if leg is None or leg >= len(self.legs):
            raise ValueError(f"unknown foot {side!r}: use 'left' or 'right'")
        out = [c.channel for c in self.contacts if c.leg == leg]
        if not out:
            raise ValueError(f"foot {side!r} has no contact channels")
        return out

    def channel(self, name: str) -> ContactRole:
        for c in self.contacts:
            if c.channel == name:
                return c
        raise KeyError(f"no contact channel {name!r}")

    def regions(self, sk: Skeleton) -> list[str]:
        """Region name per joint: legs (thigh subtrees), hands (below wrists), arms, head, torso."""
        region = ["torso"] * sk.num_joints

        def paint(root: str, name: str, include_root: bool = True) -> None:
            r = sk.index(root)
            for j in subtree(sk, r):
                if j != r or include_root:
                    region[j] = name

        if self.head:
            paint(self.head, "head")
        for a in self.arm_roots:
            paint(a, "arms")
        for w in self.wrists:
            paint(w, "hands", include_root=False)
        for leg in self.legs:
            paint(leg.thigh, "legs")
        return region

    # -- (de)serialisation -----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "pelvis": self.pelvis,
            "legs": [vars(leg).copy() for leg in self.legs],
            "contacts": [{**vars(c), "sources": list(c.sources)} for c in self.contacts],
            "wrists": list(self.wrists),
            "ankles": list(self.ankles),
            "arm_roots": list(self.arm_roots),
            "head": self.head,
            "capsules": [vars(c).copy() for c in self.capsules],
            "forward": list(self.forward),
            "up": list(self.up),
            "rest_floor": self.rest_floor,
            "clearance": dict(self.clearance),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BodyRoles:
        return cls(
            pelvis=d["pelvis"],
            legs=tuple(LegRoles(**leg) for leg in d["legs"]),
            contacts=tuple(
                ContactRole(**{**c, "sources": tuple(c.get("sources", ()))}) for c in d["contacts"]
            ),
            wrists=tuple(d.get("wrists", ())),
            ankles=tuple(d.get("ankles", ())),
            arm_roots=tuple(d.get("arm_roots", ())),
            head=d.get("head"),
            capsules=tuple(Capsule(**c) for c in d.get("capsules", ())),
            forward=tuple(d.get("forward", (0.0, 0.0, 1.0))),
            up=tuple(d.get("up", (0.0, 1.0, 0.0))),
            rest_floor=d.get("rest_floor"),
            clearance=dict(d.get("clearance", {})),
        )


# GEM-X contact channel names (soma_worker.CONTACT_ORDER) feeding afv's four foot channels.
GEMX_SOURCES = {
    "l_heel": ("left_ankle",),
    "l_toe": ("left_foot",),
    "r_heel": ("right_ankle",),
    "r_toe": ("right_foot",),
}


def _standard_roles(
    pelvis: str,
    legs: tuple[LegRoles, LegRoles],
    wrists: tuple[str, str],
    arm_roots: tuple[str, ...],
    upper: tuple[str, str] | None,
    forearms: tuple[str, str] | None,
    head: str | None,
    chest: str | None,
    channel_names: dict[str, str] | None = None,
    rest_floor: float | None = None,
) -> BodyRoles:
    """Roles for a two-legged skeleton. ``channel_names`` renames afv's l_heel/l_toe/r_heel/r_toe
    output channels (e.g. to the skeleton's joint names, as soma77 contact channels are)."""
    names = channel_names or {}
    contacts: list[ContactRole] = []
    for li, (side, leg) in enumerate(zip(("l", "r"), legs, strict=True)):
        for kind, joint in (("heel", leg.foot), ("toe", leg.toe)):
            if joint is None:
                continue
            key = f"{side}_{kind}"
            name = names.get(key, key)
            srcs = tuple(dict.fromkeys((key, *GEMX_SOURCES[key], joint)))
            contacts.append(ContactRole(name, joint, li, kind, tuple(x for x in srcs if x != name)))
    caps: list[Capsule] = []
    if chest:
        caps.append(Capsule(pelvis, chest, 0.15, "core"))
    if chest and head:
        caps.append(Capsule(chest, head, 0.11, "core"))
    for leg in legs:
        caps.append(Capsule(leg.thigh, leg.calf, 0.08, "leg"))
        caps.append(Capsule(leg.calf, leg.foot, 0.06, "leg"))
    if upper and forearms:
        for u, f, w in zip(upper, forearms, wrists, strict=True):
            caps.append(Capsule(u, f, 0.05, "arm"))
            caps.append(Capsule(f, w, 0.045, "arm"))
    return BodyRoles(
        pelvis=pelvis,
        legs=legs,
        contacts=tuple(contacts),
        wrists=wrists,
        ankles=(legs[0].foot, legs[1].foot),
        arm_roots=arm_roots,
        head=head,
        capsules=tuple(caps),
        rest_floor=rest_floor,
    )


PROVISIONAL_ROLES = _standard_roles(
    pelvis="pelvis",
    legs=(
        LegRoles("l_thigh", "l_calf", "l_foot", "l_toe"),
        LegRoles("r_thigh", "r_calf", "r_foot", "r_toe"),
    ),
    wrists=("l_hand", "r_hand"),
    arm_roots=("l_clavicle", "r_clavicle"),
    upper=("l_upperarm", "r_upperarm"),
    forearms=("l_forearm", "r_forearm"),
    head="head",
    chest="spine3",
)

# Real SOMA skeleton (module C, src/afv/core/data/skeletons/soma77.json). Its thigh joint is called
# "LeftLeg" and its calf "LeftShin", so name guessing would mislabel it: roles are explicit. Contact
# channels are named by joint (LeftFoot, LeftToeBase, ...) like module C's BodyMotion channels; GEM-X's
# left_ankle/left_foot names are accepted as sources too. The rest root offset is the hip height
# above the T-pose sole, so the rest floor is y = 0.
SOMA77_ROLES = _standard_roles(
    pelvis="Hips",
    legs=(
        LegRoles("LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase"),
        LegRoles("RightLeg", "RightShin", "RightFoot", "RightToeBase"),
    ),
    wrists=("LeftHand", "RightHand"),
    arm_roots=("LeftShoulder", "RightShoulder"),
    upper=("LeftArm", "RightArm"),
    forearms=("LeftForeArm", "RightForeArm"),
    head="Head",
    chest="Chest",
    channel_names={
        "l_heel": "LeftFoot",
        "l_toe": "LeftToeBase",
        "r_heel": "RightFoot",
        "r_toe": "RightToeBase",
    },
    rest_floor=0.0,
)

_ROLES: dict[str, BodyRoles] = {"soma_provisional": PROVISIONAL_ROLES, "soma77": SOMA77_ROLES}


def register_roles(skeleton_name: str, roles: BodyRoles) -> None:
    _ROLES[skeleton_name] = roles


def get_roles(sk: Skeleton) -> BodyRoles:
    """Registered roles for ``sk.name``, else roles guessed from joint names (validated)."""
    roles = _ROLES.get(sk.name) or guess_roles(sk)
    roles.validate(sk)
    return roles


def _tokens(name: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name.split(":")[-1])
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def guess_roles(sk: Skeleton) -> BodyRoles:
    """Guess roles from names: afv-style (l_thigh) or Mixamo/SOMA-style (LeftUpLeg, LeftLeg)."""
    parts: dict[str, set[str]] = {
        "thigh": {"thigh", "upleg", "upperleg", "femur"},
        "calf": {"calf", "knee", "shin", "leg", "lowerleg"},
        "foot": {"foot", "ankle"},
        "toe": {"toe", "toes", "toebase", "ball"},
        "wrist": {"hand", "wrist"},
        "clavicle": {"clavicle", "shoulder", "collar"},
        "upper": {"upperarm", "arm"},
        "forearm": {"forearm", "lowerarm", "elbow"},
    }
    found: dict[tuple[str, str], str] = {}
    head = chest = None
    for name in sk.joint_names:
        tok = _tokens(name)
        if not tok or {"end", "site", "twist", "roll"} & set(tok):
            continue
        side = "l" if tok[0] in ("l", "left") else "r" if tok[0] in ("r", "right") else ""
        if not side and tok[-1] in ("l", "left", "r", "right"):
            side = tok[-1][0]
        key = "".join(t for t in tok if t not in ("l", "left", "r", "right", "mixamorig"))
        if side:
            for part, words in parts.items():
                if key in words and (side, part) not in found:
                    found[(side, part)] = name
        elif key == "head" and head is None:
            head = name
        elif key.startswith(("spine", "chest", "upperchest")):
            chest = name  # the last (highest) spine joint in parent-before-child order
    pelvis = sk.joint_names[0]
    need = [(s, p) for s in "lr" for p in ("thigh", "calf", "foot", "wrist")]
    miss = [f"{s}_{p}" for s, p in need if (s, p) not in found]
    if miss:
        raise KeyError(
            f"cannot guess joint roles for skeleton {sk.name!r}; missing {miss}. "
            "Register them with afv.motion.ik.register_roles()."
        )
    legs = tuple(
        LegRoles(found[(s, "thigh")], found[(s, "calf")], found[(s, "foot")], found.get((s, "toe")))
        for s in "lr"
    )
    arm_roots = tuple(found.get((s, "clavicle")) or found.get((s, "upper")) or "" for s in "lr")
    arm_roots = tuple(a for a in arm_roots if a)
    upper = (
        (found[("l", "upper")], found[("r", "upper")])
        if ("l", "upper") in found and ("r", "upper") in found
        else None
    )
    fore = (
        (found[("l", "forearm")], found[("r", "forearm")])
        if ("l", "forearm") in found and ("r", "forearm") in found
        else None
    )
    return _standard_roles(
        pelvis, legs, (found[("l", "wrist")], found[("r", "wrist")]), arm_roots, upper, fore, head, chest
    )


# ---------------------------------------------------------------------------
# Hierarchy helpers
# ---------------------------------------------------------------------------


def ancestors(sk: Skeleton, joint: str | int) -> list[str]:
    j = sk.index(joint) if isinstance(joint, str) else joint
    out = []
    while sk.parents[j] >= 0:
        j = sk.parents[j]
        out.append(sk.joint_names[j])
    return out


def subtree(sk: Skeleton, j: int) -> list[int]:
    """``j`` and all its descendants (parents precede children, so one forward pass suffices)."""
    inside = np.zeros(sk.num_joints, dtype=bool)
    inside[j] = True
    for k in range(j + 1, sk.num_joints):
        if sk.parents[k] >= 0 and inside[sk.parents[k]]:
            inside[k] = True
    return [int(k) for k in np.flatnonzero(inside)]


def chain(sk: Skeleton, top: str, bottom: str) -> list[int]:
    """Joint indices from ``top`` down to ``bottom`` (inclusive), in parent-to-child order."""
    t, j = sk.index(top), sk.index(bottom)
    path = [j]
    while j != t:
        j = sk.parents[j]
        if j < 0:
            raise ValueError(f"{top} is not an ancestor of {bottom}")
        path.append(j)
    return path[::-1]


# ---------------------------------------------------------------------------
# Quaternion helpers (vectorised, xyzw)
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def quat_from_to(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Minimal rotation taking direction ``u`` onto direction ``v`` (broadcasting)."""
    u = _unit(np.asarray(u, dtype=np.float64))
    v = _unit(np.asarray(v, dtype=np.float64))
    d = np.sum(u * v, axis=-1, keepdims=True)
    c = np.cross(u, v)
    q = np.concatenate([c, 1.0 + d], axis=-1)
    # Opposite vectors: rotate 180 degrees about any axis perpendicular to u.
    opp = (1.0 + d[..., 0]) < 1e-9
    if np.any(opp):
        alt = np.cross(u, np.array([1.0, 0.0, 0.0]))
        alt = np.where(np.linalg.norm(alt, axis=-1, keepdims=True) < 1e-6, np.cross(u, [0.0, 1.0, 0.0]), alt)
        q[opp] = np.concatenate([_unit(alt), np.zeros_like(d)], axis=-1)[opp]
    return _unit(q)


def quat_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    axis = _unit(np.asarray(axis, dtype=np.float64))
    half = 0.5 * np.asarray(angle, dtype=np.float64)[..., None]
    return np.concatenate([axis * np.sin(half), np.cos(half)], axis=-1)


def quat_power(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """q^w for unit q and weights w in [0,1] (slerp from identity)."""
    q = np.where(q[..., 3:4] < 0, -q, q)
    s = np.linalg.norm(q[..., :3], axis=-1)
    ang = 2.0 * np.arctan2(s, q[..., 3])
    axis = q[..., :3] / np.maximum(s, 1e-12)[..., None]
    return quat_axis_angle(axis, ang * np.asarray(w, dtype=np.float64))


def floor_lift(h: np.ndarray, clear: float, width: float, hard: np.ndarray | float = 0.0) -> np.ndarray:
    """Height a point at height ``h`` should be raised to so it stays at or above ``clear``.

    ``hard`` = 1 is the exact clamp ``max(h, clear)``; ``hard`` = 0 the C1 smooth maximum with
    half-width ``width`` (equal to ``h`` above ``clear + width``, to ``clear`` below
    ``clear - width``, a parabola between), which avoids the velocity kink a hard clamp puts into a
    swinging foot that grazes the floor. Values in between blend the two.
    """
    h = np.asarray(h, dtype=np.float64)
    hard_v = np.maximum(h, clear)
    if width <= 0:
        return hard_v
    x = h - clear
    soft = np.where(x >= width, h, np.where(x <= -width, clear, clear + (x + width) ** 2 / (4.0 * width)))
    w = np.clip(np.asarray(hard, dtype=np.float64), 0.0, 1.0)
    return w * hard_v + (1.0 - w) * soft


def _horizontal(v: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Part of ``v`` [N,3] perpendicular to ``n`` and its length [N]."""
    h = v - (v @ n)[:, None] * n
    return h, np.linalg.norm(h, axis=-1)


def _toe_lift(
    rel: np.ndarray, rise: np.ndarray, n: np.ndarray, fallback: np.ndarray | None = None
) -> np.ndarray:
    """Rotation about the ankle that raises the ankle->toe vectors ``rel`` [N,3] by ``rise`` [N]
    along the floor normal ``n``, exactly: the toe moves on its circle around the ankle in the
    vertical plane through ``rel`` (the minimal pitch of the foot), keeping its horizontal direction
    and the foot's length. Rises beyond the foot's length are clamped (toe straight up). A toe
    straight below the ankle has no horizontal direction of its own: it swings towards the
    horizontal part of ``fallback`` [N,3] (the caller passes the foot's up axis, which points forward
    when the toe points down), else towards any horizontal direction. (A fixed-point iteration of
    "rotate onto the lifted point", the former approach, stopped centimetres short on steep feet.)"""
    rel = np.asarray(rel, dtype=np.float64)
    rise = np.asarray(rise, dtype=np.float64)
    length = np.linalg.norm(rel, axis=-1)
    up = rel @ n
    hr = np.clip(up + rise, -length, length)
    perp, pn = _horizontal(rel, n)
    degenerate = pn <= 1e-9 * np.maximum(length, 1e-12)
    if np.any(degenerate):
        alt = np.cross(n, [1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.cross(n, [0.0, 0.0, 1.0])
        alt = np.broadcast_to(alt, rel.shape)
        if fallback is not None:
            fh, fn = _horizontal(np.asarray(fallback, dtype=np.float64), n)
            alt = np.where((fn > 1e-6)[:, None], fh, alt)
        perp = np.where(degenerate[:, None], alt, perp)
        pn = np.linalg.norm(perp, axis=-1)
    flat = np.sqrt(np.maximum(length**2 - hr**2, 0.0))
    goal = perp / np.maximum(pn, 1e-12)[:, None] * flat[:, None] + hr[:, None] * n
    return quat_from_to(rel, goal)


def rest_rotations(sk: Skeleton) -> np.ndarray:
    return sk.rest_rotations if sk.rest_rotations is not None else np.tile(IDENTITY, (sk.num_joints, 1))


def set_chain_world_rotations(
    sk: Skeleton,
    rot: np.ndarray,
    wrot: np.ndarray,
    path: list[int],
    new_world: dict[int, np.ndarray],
    frames: np.ndarray,
) -> None:
    """Rewrite local rotations (in place) on ``frames`` so the joints in ``new_world`` get the
    given world rotations; other joints on ``path`` keep their local rotations. ``path`` is in
    parent-to-child order and its first joint's parent world rotation must be unchanged."""
    rest = rest_rotations(sk)
    w_new: dict[int, np.ndarray] = {}
    for j in path:
        p = sk.parents[j]
        par = w_new[p] if p in w_new else wrot[frames, p]
        if j in new_world:
            w = new_world[j]
            rot[frames, j] = qmul(qconj(rest[j]), qmul(qconj(par), w))
        else:
            w = qmul(par, qmul(rest[j], rot[frames, j]))
        w_new[j] = w


# ---------------------------------------------------------------------------
# Two-bone leg IK
# ---------------------------------------------------------------------------

# rad; how far past its own hyperextension an input knee may stay bent backwards (see solve_leg_ik)
KNEE_HYPER_TOL = np.radians(2.0)


def knee_hinge_local(
    sk: Skeleton, wrot: np.ndarray, pos: np.ndarray, leg: LegRoles, forward: np.ndarray
) -> np.ndarray:
    """Knee hinge axis in the thigh's frame, estimated from the motion (bend-weighted mean).

    Falls back to "knee points forward" (axis = thigh x -forward in the pelvis frame) when the
    input never bends the knee by more than ~6 degrees.
    """
    ih, ik, ia = sk.index(leg.thigh), sk.index(leg.calf), sk.index(leg.foot)
    t = _unit(pos[:, ik] - pos[:, ih])
    c = _unit(pos[:, ia] - pos[:, ik])
    cr = np.cross(t, c)
    s = np.linalg.norm(cr, axis=-1)
    good = s > 0.1
    if np.count_nonzero(good) >= 3:
        local = qrotate(qconj(wrot[good, ih]), cr[good] / s[good, None])
        axis = np.sum(local * s[good, None], axis=0)
        if np.linalg.norm(axis) > 1e-6:
            return axis / np.linalg.norm(axis)
    # Fallback: in the first frame, the axis that bends the ankle backwards (knee forward).
    fwd_world = qrotate(wrot[0, sk.parents[ih]] if sk.parents[ih] >= 0 else IDENTITY, forward)
    ax = np.cross(t[0], -fwd_world)
    if np.linalg.norm(ax) < 1e-6:
        ax = np.cross(t[0], np.array([0.0, 0.0, -1.0]))
    return qrotate(qconj(wrot[0, ih]), _unit(ax))


def leg_max_reach(
    sk: Skeleton, wrot: np.ndarray, pos: np.ndarray, leg: LegRoles, hinge_local: np.ndarray
) -> np.ndarray:
    """Largest hip-ankle distance [F] :func:`solve_leg_ik` can reach with this knee hinge: the knee
    turns the calf about the hinge only, so its component along the hinge stays and the leg is
    shorter than the bones' sum when the calf leaves the hinge plane."""
    ih, ik, ia = sk.index(leg.thigh), sk.index(leg.calf), sk.index(leg.foot)
    t = pos[:, ik] - pos[:, ih]
    c = pos[:, ia] - pos[:, ik]
    l1 = np.linalg.norm(t, axis=-1)
    th = t / np.maximum(l1, 1e-12)[:, None]
    a = qrotate(wrot[:, ih], hinge_local)
    a = _unit(a - np.sum(a * th, axis=-1, keepdims=True) * th)
    lp = np.linalg.norm(c - np.sum(c * a, axis=-1, keepdims=True) * a, axis=-1)
    return np.sqrt(l1**2 + np.sum(c * c, axis=-1) + 2 * l1 * lp)


@dataclass
class LegTarget:
    """IK goals for one leg. ``ankle`` [F,3] world target of the foot joint (rows where
    ``active`` is False are left untouched). ``toe`` [F,3] optional toe-joint target, used with
    weight ``toe_weight`` [F] in [0,1]; it is faded out where the ankle could not reach its target (a toe
    correction about a misplaced ankle would twist the foot). ``foot_rot`` [F,4] optional world
    orientation for the foot (a rigid foot pin), blended in with ``foot_rot_weight`` [F] before
    the toe correction. ``floor`` (normal [3], offset, toe clearance, soft width, hard weight [F]
    or None) lifts the toe out of the floor by rotating the foot about the ankle
    (see :func:`floor_lift`)."""

    leg: LegRoles
    ankle: np.ndarray
    active: np.ndarray
    toe: np.ndarray | None = None
    toe_weight: np.ndarray | None = None
    floor: tuple[np.ndarray, float, float, float, np.ndarray | None] | None = None
    hinge_local: np.ndarray | None = None
    foot_rot: np.ndarray | None = None
    foot_rot_weight: np.ndarray | None = None


def solve_leg_ik(
    sk: Skeleton,
    rot: np.ndarray,
    root: np.ndarray,
    targets: list[LegTarget],
    forward: np.ndarray | None = None,
) -> np.ndarray:
    """Return new local rotations [F,J,4] with each leg's ankle on its target (see module doc).

    Legs are solved independently from one FK pass; they share only ancestors above the thighs,
    which IK never changes.
    """
    rot = np.array(rot, dtype=np.float64, copy=True)
    pos, wrot = forward_kinematics(sk, rot, root)
    fwd = np.asarray(forward if forward is not None else (0.0, 0.0, 1.0), dtype=np.float64)
    rest_w: np.ndarray | None = None  # world rotations of the rest pose (for the toe-lift fallback)
    for tg in targets:
        leg = tg.leg
        ih, ik, ia = sk.index(leg.thigh), sk.index(leg.calf), sk.index(leg.foot)
        frames = np.flatnonzero(np.asarray(tg.active, dtype=bool))
        if frames.size == 0:
            continue
        hinge = tg.hinge_local if tg.hinge_local is not None else knee_hinge_local(sk, wrot, pos, leg, fwd)
        H, K, A = pos[frames, ih], pos[frames, ik], pos[frames, ia]
        T = np.asarray(tg.ankle, dtype=np.float64)[frames]
        t = K - H
        c = A - K
        l1 = np.linalg.norm(t, axis=-1)
        a = qrotate(wrot[frames, ih], hinge)
        th = t / np.maximum(l1, 1e-12)[:, None]
        a = _unit(a - np.sum(a * th, axis=-1, keepdims=True) * th)  # hinge perpendicular to thigh
        c_par = np.sum(c * a, axis=-1, keepdims=True) * a
        c_perp = c - c_par
        lp = np.linalg.norm(c_perp, axis=-1)
        l2sq = np.sum(c * c, axis=-1)
        # Current angle from thigh direction to calf (perpendicular part) about the hinge.
        phi0 = np.arctan2(np.sum(np.cross(th, c_perp) * a, axis=-1), np.sum(th * c_perp, axis=-1))
        d_min = np.sqrt(np.maximum(l1**2 + l2sq - 2 * l1 * lp, 0.0))
        d_max = np.sqrt(l1**2 + l2sq + 2 * l1 * lp)
        HT = T - H
        d = np.linalg.norm(HT, axis=-1)
        d_c = np.clip(d, d_min + 1e-6, d_max * (1.0 - 1e-7))
        kappa = np.clip((d_c**2 - l1**2 - l2sq) / np.maximum(2 * l1 * lp, 1e-12), -1.0, 1.0)
        # Bend forward (the anatomical side, phi > 0), except that a slightly hyperextended input knee
        # stays hyperextended while the target needs no more bend than the input's own hyperextension
        # (+ KNEE_HYPER_TOL): it does not flip when its target barely moves, and target == current is
        # an exact no-op. Following the input's side whatever the bend bent near-straight GEM-X knees
        # 50 degrees backwards whenever a pin pulled the ankle in (ref walk A, ufc A).
        bend = np.arccos(kappa)
        back = (phi0 < -1e-3) & (bend <= -phi0 + KNEE_HYPER_TOL)
        phi = bend * np.where(back, -1.0, 1.0)
        rk = quat_axis_angle(a, phi - phi0)
        A1 = K + qrotate(rk, c)
        T_c = H + _unit(HT) * d_c[:, None]
        sw = quat_from_to(A1 - H, T_c - H)
        new_world = {
            ih: qmul(sw, wrot[frames, ih]),
            ik: qmul(sw, qmul(rk, wrot[frames, ik])),
        }
        # Foot keeps its world orientation (or blends to a pinned one), then an optional toe
        # correction about the ankle.
        foot_w = wrot[frames, ia]
        rigid = np.tile(IDENTITY, (frames.size, 1))
        if tg.foot_rot is not None:
            fw = (
                np.zeros(frames.size)
                if tg.foot_rot_weight is None
                else np.asarray(tg.foot_rot_weight)[frames]
            )
            fr = np.asarray(tg.foot_rot, dtype=np.float64)[frames]
            fr = np.where(np.sum(fr * foot_w, axis=-1, keepdims=True) < 0, -fr, fr)
            rigid = quat_power(qmul(fr, qconj(foot_w)), np.clip(fw, 0.0, 1.0))
            foot_w = qmul(rigid, foot_w)
        if leg.toe is not None and (tg.toe is not None or tg.floor is not None):
            it = sk.index(leg.toe)
            v = qrotate(rigid, pos[frames, it] - A)  # ankle->toe, rigid with the foot's world rotation
            corr = np.tile(IDENTITY, (frames.size, 1))
            if tg.toe is not None:
                w = np.ones(frames.size) if tg.toe_weight is None else np.asarray(tg.toe_weight)[frames]
                miss = np.linalg.norm(T_c - T, axis=-1)
                w = np.clip(w, 0.0, 1.0) * np.clip(1.0 - miss / 0.02, 0.0, 1.0)
                corr = quat_power(quat_from_to(v, np.asarray(tg.toe)[frames] - T_c), w)
            if tg.floor is not None:
                n, off, clear, width, hard = tg.floor
                hard_f = 0.0 if hard is None else np.asarray(hard)[frames]
                toe_p = T_c + qrotate(corr, v)
                h = toe_p @ n + off
                want = floor_lift(h, clear, width, hard_f)
                low = want > h + 1e-5
                if np.any(low):
                    if rest_w is None:
                        rest_w = forward_kinematics(
                            sk, np.tile(IDENTITY, (1, sk.num_joints, 1)), np.zeros((1, 3))
                        )[1][0]
                    # The foot's up axis (rest up in the foot's frame, now): forward when the toe points down.
                    foot_up = qrotate(
                        qmul(qmul(corr, foot_w), qconj(rest_w[ia])), np.broadcast_to(n, v.shape)
                    )
                    lift = _toe_lift(toe_p - T_c, want - h, n, foot_up)
                    corr = np.where(low[:, None], qmul(lift, corr), corr)
            foot_w = qmul(corr, foot_w)
        new_world[ia] = foot_w
        set_chain_world_rotations(sk, rot, wrot, chain(sk, leg.thigh, leg.foot), new_world, frames)
    return rot
