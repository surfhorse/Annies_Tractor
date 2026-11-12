"""Canonical humanoid export rig (Mixamo-compatible names), built per fighter.

The rig is defined in afv's internal frame: Y-up, right-handed, metres, character faces +Z, its left
is +X. Its rest pose is a **T-pose** with identity rest rotations, so a bone's world rotation in an
animation frame *is* its rotation away from rest, and bone ``offsets`` are plain parent-to-child
vectors in the rest pose.

The rig is built from the fighter's own skeleton (``Skeleton.with_offsets(rest_offsets)``):

1. Put the source skeleton into a T-pose: starting from its rest pose, swing the upper arms and
   forearms onto +/-X and the thighs and shins onto -Y (``tpose`` directions below). Other bones
   keep the source rest shape (spine curve, feet, fingers).
2. Each rig bone's head is the T-posed world position of its mapped source joint (mapping table,
   ``afv.export.mapping``), so bone lengths are the fighter's own.

``root`` (optional, unprefixed like Unreal/Godot expect) sits on the floor under the hips at the
world origin and becomes the parent of ``Hips``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from afv.core.types import Skeleton
from afv.export._math import qinv_mul, rot_between
from afv.export.mapping import Mapping, get_mapping
from afv.motion.fk import forward_kinematics
from afv.motion.rotations import IDENTITY, qconj, qmul

ROOT_BONE = "root"


@dataclass(frozen=True)
class RigBoneDef:
    name: str  # canonical (unprefixed) Mixamo name
    parent: str | None
    tail: str | None = None  # child bone the bone points at (Blender tail, T-pose aim)
    tpose: tuple[float, float, float] | None = None  # T-pose direction head -> tail (world, Y-up)
    required: bool = False


def _humanoid() -> tuple[RigBoneDef, ...]:
    b: list[RigBoneDef] = [
        RigBoneDef("Hips", None, "Spine", required=True),
        RigBoneDef("Spine", "Hips", "Spine1", required=True),
        RigBoneDef("Spine1", "Spine", "Spine2"),
        RigBoneDef("Spine2", "Spine1", "Neck"),
        RigBoneDef("Neck", "Spine2", "Head"),
        RigBoneDef("Head", "Neck", "HeadTop_End", required=True),
        RigBoneDef("HeadTop_End", "Head"),
    ]
    for side, sx in (("Left", 1.0), ("Right", -1.0)):
        b += [
            RigBoneDef(f"{side}Shoulder", "Spine2", f"{side}Arm"),
            RigBoneDef(f"{side}Arm", f"{side}Shoulder", f"{side}ForeArm", (sx, 0.0, 0.0), True),
            RigBoneDef(f"{side}ForeArm", f"{side}Arm", f"{side}Hand", (sx, 0.0, 0.0), True),
            RigBoneDef(f"{side}Hand", f"{side}ForeArm", f"{side}HandMiddle1", required=True),
        ]
        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            for k in (1, 2, 3, 4):  # 4 = Mixamo's fingertip end bone (optional)
                parent = f"{side}Hand" if k == 1 else f"{side}Hand{finger}{k - 1}"
                tail = f"{side}Hand{finger}{k + 1}" if k < 4 else None
                b.append(RigBoneDef(f"{side}Hand{finger}{k}", parent, tail))
    for side in ("Left", "Right"):
        b += [
            RigBoneDef(f"{side}UpLeg", "Hips", f"{side}Leg", (0.0, -1.0, 0.0), True),
            RigBoneDef(f"{side}Leg", f"{side}UpLeg", f"{side}Foot", (0.0, -1.0, 0.0), True),
            RigBoneDef(f"{side}Foot", f"{side}Leg", f"{side}ToeBase", required=True),
            RigBoneDef(f"{side}ToeBase", f"{side}Foot", f"{side}Toe_End"),
            RigBoneDef(f"{side}Toe_End", f"{side}ToeBase"),
        ]
    return tuple(b)


HUMANOID: tuple[RigBoneDef, ...] = _humanoid()
HUMANOID_BY_NAME: dict[str, RigBoneDef] = {d.name: d for d in HUMANOID}
FINGER_BONES: frozenset[str] = frozenset(
    d.name for d in HUMANOID if "Hand" in d.name and d.name[-1].isdigit()
)


@dataclass(frozen=True)
class ExportRig:
    """A built export rig. Arrays are indexed by bone, parents before children.

    ``offsets[b]``: rest translation from the parent's head (for the top bone: its rest world
    position). ``tails``: rest world tail positions (bone display direction / BVH end sites).
    ``source_joint[b]``: mapped source joint name (None for ``root``). ``source_tpose_rot[b]``: that
    joint's world rotation in the source T-pose. ``compressed[b]``: the source chain from the
    parent's joint to this bone's joint has unmapped joints in between (folded joints).
    """

    names: tuple[str, ...]
    canonical: tuple[str, ...]
    parents: tuple[int, ...]
    offsets: np.ndarray  # [B,3]
    tails: np.ndarray  # [B,3]
    source_joint: tuple[str | None, ...]
    source_tpose_rot: np.ndarray  # [B,4]
    compressed: tuple[bool, ...]
    source: Skeleton  # the fighter's source skeleton (own bone offsets)
    source_tpose_local: np.ndarray  # [J,4] local rotations that put the source into the T-pose
    prefix: str = ""
    has_root: bool = False

    @property
    def num_bones(self) -> int:
        return len(self.names)

    @property
    def hips(self) -> int:
        return self.canonical.index("Hips")

    @property
    def root(self) -> int | None:
        return 0 if self.has_root else None

    def index(self, name: str) -> int:
        if name in self.canonical:
            return self.canonical.index(name)
        if name in self.names:
            return self.names.index(name)
        raise KeyError(f"export rig has no bone {name!r}")

    def children(self, b: int) -> list[int]:
        return [k for k, p in enumerate(self.parents) if p == b]

    def rest_positions(self) -> np.ndarray:
        """Rest world head positions [B,3] (rest rotations are identity, so offsets just add up)."""
        pos = np.zeros((self.num_bones, 3))
        for b, p in enumerate(self.parents):
            pos[b] = self.offsets[b] + (pos[p] if p >= 0 else 0.0)
        return pos

    def as_skeleton(self, name: str = "afv_humanoid") -> Skeleton:
        return Skeleton(name, self.names, self.parents, self.offsets.copy())

    def describe(self) -> list[dict]:
        rest = self.rest_positions()
        return [
            {
                "name": n,
                "canonical": c,
                "parent": self.names[p] if p >= 0 else None,
                "source": s,
                "head": [round(float(x), 6) for x in rest[i]],
            }
            for i, (n, c, p, s) in enumerate(
                zip(self.names, self.canonical, self.parents, self.source_joint, strict=True)
            )
        ]


def _ancestors(sk: Skeleton, j: int) -> list[int]:
    out = []
    p = sk.parents[j]
    while p >= 0:
        out.append(p)
        p = sk.parents[p]
    return out


def source_tpose(sk: Skeleton, mapping: Mapping, bones: list[RigBoneDef]) -> np.ndarray:
    """Local rotations [J,4] that swing the source's limb bones onto the T-pose directions."""
    local = np.tile(IDENTITY, (sk.num_joints, 1))
    rest_rot = sk.rest_rotations if sk.rest_rotations is not None else np.tile(IDENTITY, (sk.num_joints, 1))
    present = {d.name for d in bones}
    for d in bones:
        if d.tpose is None or d.tail not in present:
            continue
        s = sk.index(mapping.bones[d.name])
        c = sk.index(mapping.bones[d.tail])
        pos, wrot = forward_kinematics(sk, local[None])
        v = pos[0, c] - pos[0, s]
        if np.linalg.norm(v) < 1e-9:
            continue
        swing = rot_between(v, np.asarray(d.tpose))
        p = sk.parents[s]
        parent_w = wrot[0, p] if p >= 0 else IDENTITY
        # new world(s) = swing * world(s);  world(s) = parent_w * rest_rot[s] * local[s]
        local[s] = qmul(qconj(rest_rot[s]), qinv_mul(parent_w, qmul(swing, wrot[0, s])))
    return local


def build_rig(
    source: Skeleton,
    mapping: Mapping | None = None,
    *,
    rest_offsets: np.ndarray | None = None,
    prefix: str = "",
    root_bone: bool = False,
    fingers: bool = True,
) -> ExportRig:
    """Build the export rig for one fighter from its source skeleton and bone offsets."""
    if rest_offsets is not None:
        source = source.with_offsets(np.asarray(rest_offsets, dtype=np.float64))
    mapping = mapping if mapping is not None else get_mapping(source.name)
    for name, joint in mapping.bones.items():
        if name not in HUMANOID_BY_NAME:
            raise ValueError(f"mapping {mapping.source_skeleton}: unknown rig bone {name!r}")
        source.index(joint)  # raises KeyError for unknown joints
    if len(set(mapping.bones.values())) != len(mapping.bones):
        raise ValueError(f"mapping {mapping.source_skeleton}: a source joint is mapped twice")

    bones = [d for d in HUMANOID if d.name in mapping.bones and (fingers or d.name not in FINGER_BONES)]
    missing = [d.name for d in HUMANOID if d.required and d.name not in mapping.bones]
    if missing:
        raise ValueError(f"mapping {mapping.source_skeleton}: required rig bones missing: {missing}")
    present = {d.name for d in bones}

    def rig_parent(d: RigBoneDef) -> str | None:
        p = d.parent
        while p is not None and p not in present:
            p = HUMANOID_BY_NAME[p].parent
        return p

    tlocal = source_tpose(source, mapping, bones)
    tpos, trot = forward_kinematics(source, tlocal[None])
    tpos, trot = tpos[0], trot[0]

    canonical: list[str] = []
    parents: list[int] = []
    src: list[str | None] = []
    heads: list[np.ndarray] = []
    srot: list[np.ndarray] = []
    compressed: list[bool] = []
    if root_bone:
        canonical.append(ROOT_BONE)
        parents.append(-1)
        src.append(None)
        heads.append(np.zeros(3))
        srot.append(IDENTITY.copy())
        compressed.append(False)
    for d in bones:
        j = source.index(mapping.bones[d.name])
        rp = rig_parent(d)
        if rp is None:
            parents.append(0 if root_bone else -1)
            compressed.append(False)
        else:
            pi = canonical.index(rp)
            pj = source.index(src[pi])  # type: ignore[arg-type]
            anc = _ancestors(source, j)
            if pj not in anc:
                raise ValueError(
                    f"mapping {mapping.source_skeleton}: {d.name}->{source.joint_names[j]} is not below "
                    f"{rp}->{source.joint_names[pj]} in the source hierarchy"
                )
            parents.append(pi)
            compressed.append(anc[0] != pj)
        canonical.append(d.name)
        src.append(source.joint_names[j])
        heads.append(tpos[j])
        srot.append(trot[j])

    heads_a = np.array(heads)
    offsets = np.array([heads_a[b] - (heads_a[p] if p >= 0 else 0.0) for b, p in enumerate(parents)])
    tails = _tails(canonical, parents, heads_a, present)
    names = tuple(n if n == ROOT_BONE else prefix + n for n in canonical)
    return ExportRig(
        names=names,
        canonical=tuple(canonical),
        parents=tuple(parents),
        offsets=offsets,
        tails=tails,
        source_joint=tuple(src),
        source_tpose_rot=np.array(srot),
        compressed=tuple(compressed),
        source=source,
        source_tpose_local=tlocal,
        prefix=prefix,
        has_root=root_bone,
    )


_MIN_BONE = 0.01


def _tails(canonical: list[str], parents: list[int], heads: np.ndarray, present: set[str]) -> np.ndarray:
    """Rest tail per bone: its tail bone's head, else its only child's head, else continue the parent."""
    tails = np.zeros_like(heads)
    for b, name in enumerate(canonical):
        kids = [k for k, p in enumerate(parents) if p == b]
        if name == ROOT_BONE:
            tails[b] = heads[b] + np.array([0.0, 0.0, 0.2])  # on the floor, pointing forward (+Z)
            continue
        d = HUMANOID_BY_NAME[name]
        target = None
        if d.tail in present:
            target = heads[canonical.index(d.tail)]
        elif len(kids) == 1:
            target = heads[kids[0]]
        if target is not None and np.linalg.norm(target - heads[b]) >= _MIN_BONE:
            tails[b] = target
            continue
        p = parents[b]
        if p >= 0 and canonical[p] != ROOT_BONE:
            v = heads[b] - heads[p]
            n = float(np.linalg.norm(v))
            if n > 1e-9:
                tails[b] = heads[b] + v / n * max(0.5 * n, _MIN_BONE * 2)
                continue
        tails[b] = heads[b] + np.array([0.0, 0.1, 0.0])
    return tails
