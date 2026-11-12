"""Skeleton registry.

``soma_provisional`` is a STAND-IN for NVIDIA's SOMA skeleton (77 body joints + 25 per hand),
whose exact joint list must be verified against GEM-X output in P0 (spec: open items). Code must
address joints by name, never by index, so the real list can replace this one.

Rest pose: T-pose, Y-up, character faces +Z, character's left is +X, metres, ~1.75 m tall.
Root joint is the pelvis; its rest offset is its world position at rest (hip height).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from afv.core.types import Skeleton

_BODY: list[tuple[str, str | None, tuple[float, float, float]]] = [
    ("pelvis", None, (0.0, 0.95, 0.0)),
    ("spine1", "pelvis", (0.0, 0.10, 0.0)),
    ("spine2", "spine1", (0.0, 0.12, 0.0)),
    ("spine3", "spine2", (0.0, 0.12, 0.0)),
    ("neck", "spine3", (0.0, 0.14, 0.0)),
    ("head", "neck", (0.0, 0.10, 0.0)),
    ("head_end", "head", (0.0, 0.18, 0.0)),
    ("l_clavicle", "spine3", (0.04, 0.10, 0.0)),
    ("l_upperarm", "l_clavicle", (0.14, 0.0, 0.0)),
    ("l_forearm", "l_upperarm", (0.28, 0.0, 0.0)),
    ("l_hand", "l_forearm", (0.25, 0.0, 0.0)),
    ("r_clavicle", "spine3", (-0.04, 0.10, 0.0)),
    ("r_upperarm", "r_clavicle", (-0.14, 0.0, 0.0)),
    ("r_forearm", "r_upperarm", (-0.28, 0.0, 0.0)),
    ("r_hand", "r_forearm", (-0.25, 0.0, 0.0)),
    ("l_thigh", "pelvis", (0.09, -0.05, 0.0)),
    ("l_calf", "l_thigh", (0.0, -0.42, 0.0)),
    ("l_foot", "l_calf", (0.0, -0.40, 0.0)),
    ("l_toe", "l_foot", (0.0, -0.06, 0.13)),
    ("l_toe_end", "l_toe", (0.0, 0.0, 0.05)),
    ("r_thigh", "pelvis", (-0.09, -0.05, 0.0)),
    ("r_calf", "r_thigh", (0.0, -0.42, 0.0)),
    ("r_foot", "r_calf", (0.0, -0.40, 0.0)),
    ("r_toe", "r_foot", (0.0, -0.06, 0.13)),
    ("r_toe_end", "r_toe", (0.0, 0.0, 0.05)),
]

_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# Base offset of each finger's first joint from the hand joint, for the LEFT hand (+X outward).
_FINGER_BASE = {
    "thumb": (0.03, -0.01, 0.03),
    "index": (0.09, 0.0, 0.03),
    "middle": (0.095, 0.0, 0.01),
    "ring": (0.09, 0.0, -0.01),
    "pinky": (0.08, 0.0, -0.03),
}
_SEG = 0.03  # each finger segment length


def _build_soma_provisional() -> Skeleton:
    joints = list(_BODY)
    for side, sx in (("l", 1.0), ("r", -1.0)):
        for f in _FINGERS:
            bx, by, bz = _FINGER_BASE[f]
            parent = f"{side}_hand"
            for k in (1, 2, 3):
                name = f"{side}_{f}{k}"
                off = (sx * bx, by, bz) if k == 1 else (sx * _SEG, 0.0, 0.0)
                joints.append((name, parent, off))
                parent = name
    names = tuple(j[0] for j in joints)
    idx = {n: i for i, n in enumerate(names)}
    parents = tuple(-1 if p is None else idx[p] for _, p, _ in joints)
    offsets = np.array([o for _, _, o in joints], dtype=np.float64)
    return Skeleton("soma_provisional", names, parents, offsets)


SOMA_PROVISIONAL = _build_soma_provisional()

# Named contact channels produced by the contacts stage for SOMA-family skeletons.
CONTACT_CHANNELS: tuple[str, ...] = ("l_heel", "l_toe", "r_heel", "r_toe")
# Joint whose world position represents each contact channel.
CONTACT_JOINTS: dict[str, str] = {"l_heel": "l_foot", "l_toe": "l_toe", "r_heel": "r_foot", "r_toe": "r_toe"}
# soma77 (GEM-X contact logits: left/right ankle, foot, wrist): the channel name is the joint it refers to.
SOMA77_CONTACT_CHANNELS: tuple[str, ...] = (
    "LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase", "LeftHand", "RightHand",
)
CONTACT_JOINTS.update({c: c for c in SOMA77_CONTACT_CHANNELS})

SKELETONS: dict[str, Skeleton] = {SOMA_PROVISIONAL.name: SOMA_PROVISIONAL}

_DATA = Path(__file__).parent / "data" / "skeletons"


def load_skeleton_json(path: Path) -> Skeleton:
    """Load a skeleton data file: {"name", "joint_names", "parents", "rest_offsets", "rest_rotations"?}."""
    d = json.loads(Path(path).read_text())
    rr = d.get("rest_rotations")
    return Skeleton(
        d["name"],
        tuple(d["joint_names"]),
        tuple(int(p) for p in d["parents"]),
        np.asarray(d["rest_offsets"], dtype=np.float64),
        None if rr is None else np.asarray(rr, dtype=np.float64),
    )


# Skeletons shipped as data (e.g. the real SOMA 77-joint skeleton extracted from py-soma-x).
for _f in sorted(_DATA.glob("*.json")) if _DATA.is_dir() else []:
    _sk = load_skeleton_json(_f)
    SKELETONS[_sk.name] = _sk


def get_skeleton(name: str) -> Skeleton:
    try:
        return SKELETONS[name]
    except KeyError:
        raise KeyError(f"unknown skeleton {name!r}; known: {sorted(SKELETONS)}") from None


def register_skeleton(sk: Skeleton) -> None:
    SKELETONS[sk.name] = sk
