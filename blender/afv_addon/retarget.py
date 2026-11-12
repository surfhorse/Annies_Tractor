"""Retarget an afv fighter's evaluated motion onto any humanoid armature (bpy + numpy only).

No afv imports, no operators or UI here: the add-on's Retarget panel and ``afv retarget`` build on
this module. Bones are addressed by name only.

Public API
----------
``CANONICAL`` (canonical afv bone names, parents first), ``REQUIRED`` (bones every map needs), and:

- ``canonical_bones(source_obj) -> dict[str, str]``: canonical name -> bone name on an afv armature
  (a take armature or an imported afv export, any bone prefix such as ``mixamorig:``).
- ``guess_map(target_obj) -> dict[str, str]``: canonical -> target bone, from bone names and the
  hierarchy (Mixamo, Unity Humanoid, Unreal, 3ds Max Biped, DAZ, Rigify metarigs ``.L/.R`` and
  anatomical names such as ``Femur.L``/``Humerus.L``). A limb bone is only taken from a bone whose
  name says its side (``Left``/``Right``, ``.L``/``_R``, ``L_``, ``lShldr``...); twist/roll/helper
  bones are never mapped; the spine, neck and head come from the hierarchy between the hips (common
  ancestor of the thighs and the head) and the head (a named neck wins; an unnamed head ends the
  chain where the name stem changes or the chain branches, so a metarig's ``face`` is not the head);
  unnamed fingers are told apart by position (best effort). A pick that is not below its canonical
  parent's bone is dropped. Raises ValueError for a generated Rigify rig (see ``generated_rigify``).
- ``missing_required(bone_map) -> list[str]``: required canonical bones not mapped.
- ``generated_rigify(obj) -> bool``; ``new_map(bones, *, facing, hips_scale, target) -> dict``.
- ``read_map(path) -> dict``, ``write_map(path, data) -> None``, ``map_from_object(obj) -> dict | None``
  (None when the object has no map; ValueError when it has a broken one), ``map_to_object(obj, data)
  -> None``: the ``*.afvmap.json`` format ``{"format": "afv_retarget_map", "version": 1, "bones":
  {canonical: target bone}, "facing": null | "-Y" | "+Y" | "-X" | "+X", "hips_scale": null | number,
  "target": armature name}``; kept on the object as the string property ``afv_retarget_map``.
  Readers validate strictly (ValueError naming the file and field); writers write atomically (file:
  temp file, fsync, rename) and keep keys they do not know from the file/property they replace.
- ``retarget(scene, source_obj, target_obj, bone_map, *, frames=None, lock_feet=True,
  hips_scale=None, facing=None, action_name=None) -> dict``.

``retarget``
~~~~~~~~~~~~
``bone_map``: canonical -> target bone (as from ``guess_map``), or a whole map dict (its
``bones``; its ``facing``/``hips_scale`` apply when the keyword is None). ``frames``: None (the
scene's frame range) or ``(first, last)``, both included. ``facing``: the target's forward in **its
own armature axes** (so a stored map stays valid when the object is rotated); None = detect.
``hips_scale``: None = the ratio of the pelvis anchors' rest heights above the floor (target /
source). The source's **evaluated** pose is sampled on every frame (``scene.frame_set`` + depsgraph),
so the NLA tweak layers and an active action are included. A new action (fake user, one key per frame
for every mapped bone's rotation, in the bone's own rotation mode, plus the hips location; frame
range set) is assigned to the target. Unmapped bones keep their rest pose (a constant rest key is
added for a channel that other animation of the target drives, or that is not at rest now). Existing
actions are never edited or deleted (the target's previous active action gets a fake user if it
would otherwise lose its last user); the rest pose, meshes and object transform are not touched; the
scene frame is restored; no mode is changed. To make the new action play it may leave NLA tweak mode
and set the action blend to Replace at influence 1 (both reported in ``notes``). On an error no
partial action is left behind and the target's animation is as before.

Refuse, never guess silently -- ``ValueError`` (the message says what to fix) for: non-armatures; a
bad map or frame range (Hips unmapped, a target bone mapped twice); a non-uniformly scaled, sheared or
mirrored source or target; a target not standing upright in its rest pose (hips -> head more than
30 deg from world Z: wrong import axes); a source not evaluated in the scene's view layer (e.g. its
collection is excluded); a map that contradicts the target's hierarchy (a mapped bone not below its
mapped canonical parent's bone); a generated Rigify rig. A suspected left/right swap is noted.

Method (all rotations in Blender world space, so object transforms and unit scale do not matter):

- Joint = bone head; a bone's direction = its joint -> its *mapped child joint* (the next mapped
  bone down the canonical chain: Spine -> Spine1 -> Spine2 -> Neck -> Head, Arm -> ForeArm -> Hand ->
  HandMiddle1, UpLeg -> Leg -> Foot ...), not Blender's head -> tail (Girush's ``Sacrum`` points
  forward).
- Facing: each rig's forward = (left - right joint pairs) x its up axis **in its own armature axes**,
  snapped to the nearest armature axis (within 15 deg; otherwise kept and noted), then taken to the
  world. ``F`` = the yaw (about world Z) turning the target's forward onto the source's.
- Rest alignment per bone ``A = arc(F d_t, d_s) F`` (``arc``: shortest arc from the target's rest
  direction onto the source's, so the target keeps its own roll); the hips and the feet (``Foot``,
  ``ToeBase``, ``Toe_End``) get only ``A = F``, so a foot keeps its own rest pitch to the floor; a bone
  with no mapped child inherits its canonical parent's ``A``.
- Per frame ``W_t = C (W_s W_s_rest^-1) A W_t_rest`` with ``C`` = the shortest arc from the
  rest-rigid prediction of the source direction onto the actual source joint -> mapped-child
  direction (identity on a take; it matters when the target skips a canonical bone, e.g. no Spine1).
- World rotations become pose-bone ``matrix_basis`` top-down through the actual posed parent chain,
  so unmapped bones in between (twist bones, Girush's ``Spine1.001``) stay at rest relative to their
  posed parent. Rest matrices with a slight scale (Girush: 0.25 %) are orthonormalized for the
  rotations; the hips location is inverted with the real matrix, as Blender applies it.
- Hips position: the *pelvis anchor* (midpoint of the two thigh joints; the hips joint if a thigh is
  not mapped) follows ``anchor_t_rest + hips_scale * (anchor_s - anchor_s_rest)`` in world space and
  the hips bone is placed so its anchor lands there (pivot-independent: Girush's ``Sacrum`` head sits
  8 cm behind its spine).
- The floor: the source's is world z = 0 (afv's floor on take armatures); the target's is the height
  of its feet's lowest rest point. A foot's floor points are its bones' heads and tails (toe tip), its
  heel (the ankle's point on the rest floor) and the low rest vertices of skinned meshes weighted at
  least 25 % to the foot's bones, taken from the meshes evaluated with the armature in its rest
  position (so a Mirror or Displace before the Armature modifier counts) and moved by linear blend
  skinning of up to 4 bones, like Blender's Armature modifier.
- ``lock_feet`` (needs the source's plant masks ``afv_plant_<channel>``, read evaluated per frame):
  per foot and planted run, a planted ``Foot`` channel holds the target's ankle, a planted ``ToeBase``
  channel alone (heel up) its toe joint, handing over in place (heel-off pivots on the toe) and eased
  over a few frames centred on the handover; the first hold of a run sits where the retargeted joint is
  (median), at its rest height + hips_scale x the source joint's height above its own rest height, and
  never sinks the sole below the floor. Into and out of each run the correction fades with matched
  position and velocity (cubic Hermite, long enough to stay within ``_EDGE_ACCEL``, up to 30 frames;
  a capped or squeezed fade is noted).
  On every frame the lowest floor point is lifted back toward the floor by a one-sided C1 ramp (a foot
  at or above the floor is untouched; at most 0.75 fighter-cm is left at 1.5 cm depth, less deeper).
  Where a locked foot cannot be reached (the target's legs are short for the scaled hips), the hips
  come down just enough
  (at most ``HIPS_DROP_MAX`` of the leg length), widened and smoothed over ``HIPS_DROP_FRAMES``
  (reported in ``notes``). Two-bone IK on the mapped thigh/shin (knee in the plane of its retargeted
  bend, foot world rotation kept), then the solved pose is checked and re-aimed (up to 3 passes) for
  what is left: a planted run whose sole still sinks (vertices weighted to the IK'd shin) comes up as a
  whole, so its sole touches the floor without sliding; off the plants the floor ramp lifts what is
  left; simulated constraints re-aim their ankle error. Goals still out of reach are reported in
  ``notes`` with their frames.
- Constraints on the target: a bone with Copy Rotation (local -> local, Replace, all axes, full
  influence) of another bone is simulated (Girush's shin copies its knee piece, so that knee bends
  over two joints): the driving bone's rotation is fitted per frame (damped Gauss-Newton) so that,
  with the copies, the child joint is in the intended direction and the bone keeps its intended
  twist. Any other enabled constraint on a mapped bone or above one is listed in ``notes``.

Report (``retarget``'s return value): ``{"action", "frames": [first, last], "mapped": n, "unmapped":
[canonical bones the source has but the map does not], "hips_scale", "facing": the target's forward as
an axis of its own armature ("-Y"...), "facing_yaw_deg": the yaw of F, "max_dir_err_deg" (worst angle
between a target bone's direction and the source's, any frame), "max_twist_err_deg" (worst twist
about the bone direction relative to the aligned rest; for the hips, the feet and bones without a
mapped child, their whole rotation error), "ik_bones" (the target bones the foot lock's IK solved: the
mapped thigh, shin and foot of each locked leg and the bones between them; both error maxima leave
them out, since the lock moves them away from the source on purpose; [] without lock_feet or plant
masks), "foot_slide_cm_s":
{"left", "right"} (median speed of the held joint over planted frames, None when the foot is never
planted), "penetration_cm" (deepest point of the feet below the floor, any frame: toe tip, heel and
foot-weighted mesh vertices), "notes": [...], "seconds"}``. Report lengths are **centimetres at the
fighter's scale** (target length / ``hips_scale``), so a 176-unit Girush and a 1.1 m mech report
comparable numbers.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import re
import time

import bpy
import numpy as np
from bpy_extras import anim_utils
from mathutils import Quaternion

MAP_FORMAT = "afv_retarget_map"
MAP_VERSION = 1
MAP_PROP = "afv_retarget_map"
MAP_KEYS = ("format", "version", "bones", "facing", "hips_scale", "target")
FACINGS = ("-Y", "+Y", "-X", "+X", "-Z", "+Z")  # the forward as an axis of the target armature
PLANT_PROP = "afv_plant_"
BLEND_FRAMES = 4  # lock fades last at least this many frames; handovers are eased over this many
HIPS_DROP_FRAMES = 6  # a hips drop (locked foot out of reach) is widened and smoothed over this many frames
HIPS_DROP_MAX = 0.25  # ... and at most this fraction of the leg length (more means a wrong hips_scale)
_REACH = 0.999  # a lock may straighten the leg to this fraction of thigh + shin
_SNAP_DEG = 15.0  # a detected facing this close to an armature axis is snapped onto it
_LINEAR = 1  # Keyframe.interpolation enum value of 'LINEAR' (foreach_set takes the int)


# --- canonical skeleton (the export rig's bone names, afv.export.rig.HUMANOID) ---------------------
def _canonical_defs() -> list[tuple[str, str | None, str | None, bool]]:
    """(name, parent, tail = the child it points at, required), parents first."""
    out: list[tuple[str, str | None, str | None, bool]] = [
        ("Hips", None, "Spine", True), ("Spine", "Hips", "Spine1", True),
        ("Spine1", "Spine", "Spine2", False), ("Spine2", "Spine1", "Neck", False),
        ("Neck", "Spine2", "Head", False), ("Head", "Neck", "HeadTop_End", True),
        ("HeadTop_End", "Head", None, False),
    ]
    for s in ("Left", "Right"):
        out += [(f"{s}Shoulder", "Spine2", f"{s}Arm", False),
                (f"{s}Arm", f"{s}Shoulder", f"{s}ForeArm", True),
                (f"{s}ForeArm", f"{s}Arm", f"{s}Hand", True),
                (f"{s}Hand", f"{s}ForeArm", f"{s}HandMiddle1", True)]
        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            for k in (1, 2, 3, 4):
                parent = f"{s}Hand" if k == 1 else f"{s}Hand{finger}{k - 1}"
                tail = f"{s}Hand{finger}{k + 1}" if k < 4 else None
                out.append((f"{s}Hand{finger}{k}", parent, tail, False))
    for s in ("Left", "Right"):
        out += [(f"{s}UpLeg", "Hips", f"{s}Leg", True), (f"{s}Leg", f"{s}UpLeg", f"{s}Foot", True),
                (f"{s}Foot", f"{s}Leg", f"{s}ToeBase", True),
                (f"{s}ToeBase", f"{s}Foot", f"{s}Toe_End", False),
                (f"{s}Toe_End", f"{s}ToeBase", None, False)]
    return out


_DEFS = _canonical_defs()
CANONICAL: tuple[str, ...] = tuple(d[0] for d in _DEFS)
REQUIRED: tuple[str, ...] = tuple(d[0] for d in _DEFS if d[3])
_PARENT: dict[str, str | None] = {d[0]: d[1] for d in _DEFS}
_TAIL: dict[str, str | None] = {d[0]: d[2] for d in _DEFS}
_FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
_SIDES = (("Left", "L"), ("Right", "R"))


# --- map files -----------------------------------------------------------------------------------
def _validate_map(data, where: str) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"{where}: a retarget map must be a JSON object, got {type(data).__name__}")
    if data.get("format") != MAP_FORMAT:
        raise ValueError(f"{where}: format: expected {MAP_FORMAT!r}, got {data.get('format')!r}")
    v = data.get("version")
    if type(v) is not int or v != MAP_VERSION:
        raise ValueError(f"{where}: version: expected {MAP_VERSION}, got {v!r}")
    bones = data.get("bones")
    if not isinstance(bones, dict):
        raise ValueError(f"{where}: bones: expected an object, got {bones!r}")
    seen: dict[str, str] = {}
    for c, b in bones.items():
        if c not in _PARENT:
            raise ValueError(f"{where}: bones.{c}: not a canonical afv bone")
        if not isinstance(b, str) or not b:
            raise ValueError(f"{where}: bones.{c}: expected a bone name, got {b!r}")
        if b in seen:
            raise ValueError(f"{where}: bones.{c}: target bone {b!r} is already mapped to {seen[b]}")
        seen[b] = c
    f = data.get("facing")
    if f is not None and f not in FACINGS:
        raise ValueError(f"{where}: facing: expected null or one of {', '.join(FACINGS)}, got {f!r}")
    h = data.get("hips_scale")
    if h is not None and (type(h) not in (int, float) or not math.isfinite(h) or h <= 0):
        raise ValueError(f"{where}: hips_scale: expected null or a positive number, got {h!r}")
    t = data.get("target")
    if t is not None and not isinstance(t, str):
        raise ValueError(f"{where}: target: expected a string, got {t!r}")
    return {**data, "bones": dict(bones)}


def new_map(bones: dict[str, str], *, facing: str | None = None, hips_scale: float | None = None,
            target: str | None = None) -> dict:
    """A complete map dict (validated) from a canonical -> target bone dict."""
    data = {"format": MAP_FORMAT, "version": MAP_VERSION, "bones": dict(bones), "facing": facing,
            "hips_scale": hips_scale, "target": target}
    return _validate_map(data, "map")


def read_map(path) -> dict:
    """The validated map in the ``*.afvmap.json`` file ``path`` (ValueError names the file and field)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except ValueError as e:
        raise ValueError(f"{path}: not valid JSON: {e}") from e
    return _validate_map(data, str(path))


def _merged(old_raw, data: dict) -> dict:
    """``data`` on top of the unknown keys of the map it replaces (``old_raw``: parsed JSON or None)."""
    keep = {k: v for k, v in old_raw.items() if k not in MAP_KEYS} if isinstance(old_raw, dict) else {}
    return {**keep, **data}


def write_map(path, data: dict) -> None:
    """Write the map ``data`` (validated) to ``path`` atomically, keeping keys of the file it replaces
    that this format does not know."""
    data = _validate_map(data, str(path))
    old = None
    try:
        with open(path, encoding="utf-8") as fh:
            old = json.load(fh)
    except (OSError, ValueError):
        pass
    out = _merged(old, data)
    path = os.fspath(path)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def map_from_object(obj) -> dict | None:
    """The map stored on ``obj`` (property ``afv_retarget_map``), None if it has none; ValueError for
    a broken one."""
    raw = obj.get(MAP_PROP) if obj is not None else None
    if raw is None:
        return None
    where = f"{obj.name}[{MAP_PROP!r}]"
    if not isinstance(raw, str):
        raise ValueError(f"{where}: expected a JSON string")
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise ValueError(f"{where}: not valid JSON: {e}") from e
    return _validate_map(data, where)


def map_to_object(obj, data: dict) -> None:
    """Store the map ``data`` (validated) on ``obj`` as a JSON string, keeping unknown keys of the map
    it replaces."""
    data = _validate_map(data, f"{obj.name}[{MAP_PROP!r}]")
    old = None
    raw = obj.get(MAP_PROP)
    if isinstance(raw, str):
        try:
            old = json.loads(raw)
        except ValueError:
            old = None
    obj[MAP_PROP] = json.dumps(_merged(old, data), ensure_ascii=False)


def missing_required(bone_map: dict) -> list[str]:
    """Required canonical bones not mapped in ``bone_map`` (a bones dict or a whole map dict)."""
    bones = bone_map.get("bones", bone_map) if isinstance(bone_map.get("bones"), dict) else bone_map
    return [c for c in REQUIRED if not bones.get(c)]


# --- source armatures ------------------------------------------------------------------------------
def canonical_bones(source_obj) -> dict[str, str]:
    """Canonical name -> bone name on an afv armature (prefix found from its hips bone)."""
    names = set(source_obj.data.bones.keys())
    cands = []
    hint = source_obj.get("afv_hips")  # take armatures record their hips bone (builder.OBJ_HIPS)
    if isinstance(hint, str) and hint.endswith("Hips") and hint in names:
        cands.append(hint[:-4])
    cands += [n[:-4] for n in sorted(names) if n.endswith("Hips")]
    best, best_n = None, 0
    for p in dict.fromkeys(cands):
        n = sum(1 for c in CANONICAL if p + c in names)
        if n > best_n:
            best, best_n = p, n
    if best is None:
        return {}
    return {c: best + c for c in CANONICAL if best + c in names}


# --- guess_map -------------------------------------------------------------------------------------
_PREFIX_WORDS = {"def", "mixamorig", "bip", "biped", "cc", "base", "rig", "jnt", "joint", "bn", "bone", "sk",
                 "skel"}
_EXCLUDE_WORDS = {"twist", "roll", "tw", "share", "helper", "adjust", "corrective", "fix", "mch", "org", "ik",
                  "fk", "pole", "target", "ctrl", "ctl", "control", "wgt", "vis", "prop", "mount", "weapon",
                  "socket", "attach", "jiggle", "physics", "dyn", "heel", "pelvis"}
_STEMS = {
    "Shoulder": {"shoulder", "clavicle", "collar", "collarbone", "clav"},
    "Arm": {"arm", "upperarm", "uparm", "humerus", "shldr", "shldrbend"},
    "ForeArm": {"forearm", "lowerarm", "lowarm", "elbow", "radius", "forearmbend"},
    "Hand": {"hand", "wrist"},
    "UpLeg": {"upleg", "upperleg", "thigh", "femur", "thighbend"},
    "Leg": {"leg", "lowerleg", "lowleg", "shin", "calf", "tibia", "knee", "crus"},
    "Foot": {"foot", "ankle"},
    "ToeBase": {"toe", "toes", "toebase", "ball"},
    "Toe_End": {"toeend", "toesend", "toetip", "toenub", "toebaseend"},
    "Head": {"head"},
    "HeadTop_End": {"headtop", "headend", "headtopend", "headnub", "headtip"},
    "Neck": {"neck"},
}
_FINGER_WORDS = {"thumb": "Thumb", "index": "Index", "pointer": "Index", "middle": "Middle", "mid": "Middle",
                 "ring": "Ring", "pinky": "Pinky", "pinkie": "Pinky", "little": "Pinky", "small": "Pinky"}


def _parse_name(name: str) -> tuple[str | None, list[str]]:
    """(side 'L'|'R'|None, lowercase words without side, digits or a leading rig prefix)."""
    base = re.split(r"[:|]", name)[-1]
    toks: list[str] = []
    for part in re.split(r"[._\-\s]+", base):
        toks += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", part)
    sides, words = set(), []
    for t in toks:
        tl = t.lower()
        if tl in ("l", "left"):
            sides.add("L")
        elif tl in ("r", "right"):
            sides.add("R")
        elif not t.isdigit():
            words.append(tl)
    while words and words[0] in _PREFIX_WORDS:
        words = words[1:]
    return (sides.pop() if len(sides) == 1 else None), words


class _Rig:
    """Hierarchy and rest data of a target armature for ``guess_map``."""

    def __init__(self, obj):
        self.bones = list(obj.data.bones)
        self.names = [b.name for b in self.bones]
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in self.bones}
        self.children: dict[str, list[str]] = {n: [] for n in self.names}
        for n, p in self.parent.items():
            if p is not None:
                self.children[p].append(n)
        self.depth = {n: self._depth(n) for n in self.names}
        self.head = {b.name: np.array(b.head_local, dtype=np.float64) for b in self.bones}
        self.deform = {b.name: bool(b.use_deform) for b in self.bones}
        self.info = {n: _parse_name(n) for n in self.names}

    def _depth(self, n):
        d = 0
        while self.parent[n] is not None:
            n = self.parent[n]
            d += 1
        return d

    def ancestors(self, n):
        out = []
        while self.parent[n] is not None:
            n = self.parent[n]
            out.append(n)
        return out

    def is_under(self, n, top):
        return top is None or top in self.ancestors(n)

    def excluded(self, n):
        return any(w in _EXCLUDE_WORDS for w in self.info[n][1])

    def stem(self, n):
        return "".join(self.info[n][1])

    def side(self, n):
        return self.info[n][0]

    def top(self, cands):
        cands = list(cands)
        if not cands:
            return None
        return min(cands, key=lambda n: (self.depth[n], not self.deform[n], n))

    def find(self, slot, side, under=None, used=()):
        return self.top(n for n in self.names if n not in used and self.side(n) == side
                        and not self.excluded(n) and self.stem(n) in _STEMS[slot] and n != under
                        and self.is_under(n, under))

    def lca(self, names):
        names = [n for n in names if n is not None]
        if not names:
            return None
        common = None
        for n in names:
            chain = [n, *self.ancestors(n)]
            common = chain if common is None else [a for a in common if a in chain]
        return common[0] if common else None

    def path(self, top, bottom):
        """Bones strictly between ``top`` and ``bottom`` (top-down); None if not an ancestor."""
        anc = self.ancestors(bottom)
        if top not in anc:
            return None
        return list(reversed(anc[: anc.index(top)]))

    def chain_from(self, n, limit):
        """``n`` and its single-child continuation (the child with most descendants), up to ``limit``."""
        out = [n]
        while len(out) < limit:
            kids = [k for k in self.children[out[-1]] if not self.excluded(k)]
            if not kids:
                break
            out.append(max(kids, key=lambda k: (self._count(k), k)))
        return out

    def _count(self, n):
        return 1 + sum(self._count(k) for k in self.children[n])


_RIGIFY_MSG = ("{name} is a generated Rigify rig (control and ORG-/MCH-/DEF- bones): retarget onto its "
               "metarig or onto a deform-only game rig instead")


def generated_rigify(obj) -> bool:
    """True for a rig generated by Rigify (its ``rig_id``, or two of the ORG-/MCH-/DEF- bone families)."""
    if obj.data.get("rig_id") is not None:
        return True
    names = obj.data.bones.keys()
    return sum(1 for pre in ("ORG-", "MCH-", "DEF-") if any(n.startswith(pre) for n in names)) >= 2


def _hierarchy_breaks(bone_map: dict[str, str], is_under) -> list[tuple[str, str]]:
    """(canonical bone, its nearest mapped canonical ancestor) pairs whose target bones are not one
    below the other, in canonical order (``is_under(bone, top)``)."""
    bad = []
    for c in CANONICAL:
        if c not in bone_map:
            continue
        p = _PARENT[c]
        while p is not None and p not in bone_map:
            p = _PARENT[p]
        if p is not None and not is_under(bone_map[c], bone_map[p]):
            bad.append((c, p))
    return bad


def guess_map(target_obj) -> dict[str, str]:
    """Canonical afv bone -> target bone, from the target's bone names and hierarchy. Raises
    ValueError for a generated Rigify rig. A pick that is not below its canonical parent's bone is
    dropped (``missing_required`` then shows it)."""
    if generated_rigify(target_obj):
        raise ValueError(_RIGIFY_MSG.format(name=target_obj.name))
    rig = _Rig(target_obj)
    out: dict[str, str] = {}
    used: set[str] = set()

    def put(c, n):
        if n is not None and n not in used and c not in out:
            out[c] = n
            used.add(n)

    for side, s in _SIDES:  # legs
        up = rig.find("UpLeg", s, used=used)
        put(f"{side}UpLeg", up)
        leg = rig.find("Leg", s, up, used) if up else None
        put(f"{side}Leg", leg)
        foot = rig.find("Foot", s, leg or up, used) if up else None
        put(f"{side}Foot", foot)
        if foot:
            toe = rig.find("ToeBase", s, foot, used)
            if toe is None:
                kids = [k for k in rig.children[foot] if not rig.excluded(k) and rig.side(k) in (s, None)]
                toe = kids[0] if len(kids) == 1 else None
            put(f"{side}ToeBase", toe)
            if toe:
                put(f"{side}Toe_End", rig.find("Toe_End", s, toe, used))
    for side, s in _SIDES:  # arms
        sh = rig.find("Shoulder", s, used=used)
        arm = rig.find("Arm", s, sh, used) if sh else None
        if arm is None:
            arm = rig.find("Arm", s, used=used)
            if sh is not None and arm is not None and not rig.is_under(arm, sh):
                sh = None
        put(f"{side}Shoulder", sh)
        put(f"{side}Arm", arm)
        fore = rig.find("ForeArm", s, arm, used) if arm else None
        put(f"{side}ForeArm", fore)
        hand = rig.find("Hand", s, fore or arm, used) if arm else None
        put(f"{side}Hand", hand)
    roots = [out.get(f"{side}Shoulder") or out.get(f"{side}Arm") for side, _ in _SIDES]
    chest = rig.lca(roots) if all(roots) else None
    head = rig.top(n for n in rig.names if n not in used and rig.side(n) is None and not rig.excluded(n)
                   and rig.stem(n) in _STEMS["Head"])
    if head is None and chest is not None:  # Rigify-style: the end of the chain above the chest
        head = _chain_end(rig, chest, roots)
    legs = [out.get("LeftUpLeg"), out.get("RightUpLeg")]
    hips = rig.lca([*legs, head or chest]) if all(legs) else None
    if hips is not None:
        put("Hips", hips)
    if hips is not None and head is not None:
        _spine(rig, hips, head, chest, put)
        put("Head", head)
        put("HeadTop_End", rig.top(n for n in rig.names if n not in used and rig.is_under(n, head)
                                   and rig.stem(n) in _STEMS["HeadTop_End"]))
    for side, s in _SIDES:
        if out.get(f"{side}Hand"):
            _fingers(rig, out[f"{side}Hand"], side, s, put, used)
    while True:  # drop picks that contradict the hierarchy (then their children are checked again)
        bad = _hierarchy_breaks(out, lambda n, top: rig.is_under(n, top))
        if not bad:
            break
        out.pop(bad[0][0])
    return {c: out[c] for c in CANONICAL if c in out}


def _chain_end(rig: _Rig, chest: str, arm_roots) -> str | None:
    kids = [k for k in rig.children[chest] if not rig.excluded(k)
            and not any(r == k or rig.is_under(r, k) for r in arm_roots) and rig.side(k) is None]
    if not kids:
        return None
    n = max(kids, key=lambda k: (rig._count(k), k))
    while True:  # follow the chain while it keeps its name stem (spine.004 -> spine.006, not 'face')
        same = [k for k in rig.children[n] if not rig.excluded(k) and rig.side(k) is None
                and rig.stem(k) == rig.stem(n)]
        if len(same) != 1:
            return n
        n = same[0]


def _spine(rig: _Rig, hips: str, head: str, chest: str | None, put) -> None:
    path = rig.path(hips, head) or []
    cand = [n for n in path if not rig.excluded(n)]
    if not cand:
        return
    spine = cand[0]
    put("Spine", spine)
    necks = [n for n in cand[1:] if rig.stem(n) in _STEMS["Neck"]]
    if necks:  # a named neck wins, even when the arms hang from it (3ds Max Biped: clavicles under Neck)
        neck = necks[0]
        before = cand[1: cand.index(neck)]
        spine2 = chest if chest in before else (before[-1] if before else None)
    else:
        spine2 = chest if chest in cand and chest != spine else None
        after = cand[cand.index(spine2) + 1:] if spine2 else cand[1:]
        neck = after[0] if after and (spine2 or len(after) > 1) else None
        if spine2 is None and neck is not None:
            before = cand[1: cand.index(neck)]
            spine2 = before[-1] if before else None
    if spine2:
        put("Spine2", spine2)
        mid = 0.5 * (rig.head[spine] + rig.head[spine2])
        between = cand[1: cand.index(spine2)]
        if between:
            put("Spine1", min(between, key=lambda n: float(np.linalg.norm(rig.head[n] - mid))))
    if neck:
        put("Neck", neck)


def _fingers(rig: _Rig, hand: str, side: str, s: str, put, used) -> None:
    named = False
    for word_set_name in _FINGERS:
        bones = [n for n in rig.names if n not in used and rig.is_under(n, hand)
                 and rig.side(n) in (s, None) and not rig.excluded(n)
                 and any(_FINGER_WORDS.get(w) == word_set_name for w in rig.info[n][1])]
        if not bones:
            continue
        named = True
        chain = [rig.top(bones)]
        while len(chain) < 4:
            nxt = [n for n in bones if rig.parent[n] == chain[-1]]
            if not nxt:
                break
            chain.append(rig.top(nxt))
        for k, n in enumerate(chain, start=1):
            put(f"{side}Hand{word_set_name}{k}", n)
    if named:
        return
    roots = [k for k in rig.children[hand]
             if k not in used and not rig.excluded(k) and rig.side(k) in (s, None)]
    if len(roots) != 5:
        return
    chains = {r: rig.chain_from(r, 4) for r in roots}
    wrist = rig.head[hand]
    thumb = min(roots, key=lambda r: float(np.linalg.norm(rig.head[r] - wrist)))
    rest = [r for r in roots if r != thumb]
    pts = np.array([rig.head[r] for r in rest])
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre)
    axis = vt[0]
    order = sorted(rest, key=lambda r: float(np.dot(rig.head[r] - centre, axis)))
    thumb_mean = np.mean([rig.head[n] for n in chains[thumb]], axis=0)
    if np.linalg.norm(rig.head[order[-1]] - thumb_mean) < np.linalg.norm(rig.head[order[0]] - thumb_mean):
        order.reverse()
    for name, r in zip(_FINGERS, [thumb, *order], strict=True):
        for k, n in enumerate(chains[r], start=1):
            put(f"{side}Hand{name}{k}", n)


# --- rotation maths (numpy, matrices act on column vectors) ------------------------------------------
def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def _skew(k):
    z = np.zeros(k.shape[:-1])
    x, y, w = k[..., 0], k[..., 1], k[..., 2]
    return np.stack([np.stack([z, -w, y], -1), np.stack([w, z, -x], -1), np.stack([-y, x, z], -1)], -2)


def _arc(u, v):
    """Shortest-arc rotations [...,3,3] taking direction(s) ``u`` onto ``v``."""
    u, v = np.broadcast_arrays(_unit(u), _unit(v))
    k = np.cross(u, v)
    c = np.sum(u * v, axis=-1)
    anti = c < -1.0 + 1e-10
    kk = _skew(k)
    r = np.eye(3) + kk + kk @ kk / np.where(anti, 1.0, 1.0 + c)[..., None, None]
    if np.any(anti):
        a = np.where((np.abs(u[..., :1]) < 0.9), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        n = _unit(np.cross(u, a))
        r180 = 2.0 * n[..., :, None] * n[..., None, :] - np.eye(3)
        r = np.where(anti[..., None, None], r180, r)
    return r


def _rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ortho(m):
    """Nearest rotations (polar decomposition) of [...,3,3] matrices that carry scale or shear (object
    scale; some files store bone rest matrices with a slight scale)."""
    u, _, vt = np.linalg.svd(np.asarray(m, dtype=np.float64))
    r = u @ vt
    flip = np.linalg.det(r) < 0
    if np.any(flip):
        u = u.copy()
        u[..., :, 2] = np.where(flip[..., None], -u[..., :, 2], u[..., :, 2])
        r = u @ vt
    return r


def _T(m):
    return np.swapaxes(m, -1, -2)


def _mv(m, v):
    return np.einsum("...ij,...j->...i", m, v)


def _quat(m):
    """Rotation matrices [...,3,3] -> unit quaternions [...,4] (w, x, y, z), Shepperd's method."""
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    cand = np.stack([
        np.stack([1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01], -1),
        np.stack([m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20], -1),
        np.stack([m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21], -1),
        np.stack([m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22], -1),
    ], 1)  # [N, case, 4]
    pick = np.argmax(np.stack([cand[:, i, i] for i in range(4)], -1), axis=-1)
    q = cand[np.arange(len(m)), pick]
    q = _unit(q)
    q = np.where(q[:, :1] < 0, -q, q)
    return q.reshape(*shape, 4)


def _angle(m):
    """Rotation angle (radians) of rotation matrices."""
    c = (np.trace(m, axis1=-2, axis2=-1) - 1.0) * 0.5
    return np.arccos(np.clip(c, -1.0, 1.0))


def _twist(m, axis):
    """Twist angle (radians, signed) of rotations ``m`` about unit ``axis``."""
    q = _quat(m)
    return 2.0 * np.arctan2(np.sum(q[..., 1:] * axis, axis=-1), q[..., 0])


def _vec_angle(a, b):
    a, b = _unit(a), _unit(b)
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))


def _continuous(q):
    q = np.array(q)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0.0:
            q[i] = -q[i]
    return q


# --- retarget --------------------------------------------------------------------------------------
_ARM_AXES = {"+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0), "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
             "+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0)}
_FOOT_FAMILY = frozenset(f"{s}{b}" for s in ("Left", "Right") for b in ("Foot", "ToeBase", "Toe_End"))
_YAW_ONLY = _FOOT_FAMILY | {"Hips"}  # rest alignment: facing only (they keep their own pitch/roll)
_UPRIGHT_DEG = 30.0  # a target whose rest hips -> head leans more than this from world Z is refused
_EDGE_ACCEL = 1200.0  # fighter cm/s^2: the lock blends in/out no harder than this (longer when needed)
_EDGE_MAX_FRAMES = 30  # frames: longest lock fade (a longer one is capped and noted)
_FLOOR_SOFT_CM = 1.5  # fighter cm: the floor lift eases in over this depth (no kink at contact; leaves
#                        at most half of it, 0.75 cm, at that depth and less below)
_MESH_BAND = 0.35  # foot mesh points: rest vertices up to this fraction of the ankle height above the floor
_MESH_POINTS = 4000  # ... at most this many per foot
_GROUP_READS = 12000  # vertex-group reads (Python) for the sole band, at most this many vertices
_LBS_BONES = 4  # bone influences kept per vertex


def _frame_range(scene, frames) -> tuple[int, int]:
    if frames is None:
        f0, f1 = scene.frame_start, scene.frame_end
    elif isinstance(frames, range):
        if len(frames) == 0 or frames.step != 1:
            raise ValueError(f"frames: expected consecutive frames, got {frames!r}")
        f0, f1 = frames.start, frames.stop - 1
    else:
        try:
            f0, f1 = frames
        except (TypeError, ValueError) as e:
            raise ValueError(f"frames: expected None or (first, last), got {frames!r}") from e
    for f in (f0, f1):
        if isinstance(f, bool) or not float(f).is_integer():
            raise ValueError(f"frames: expected whole frame numbers, got {frames!r}")
    f0, f1 = int(f0), int(f1)
    if f1 < f0:
        raise ValueError(f"frames: last frame {f1} is before the first {f0}")
    return f0, f1


def _view_layer(scene):
    ctx = bpy.context
    return ctx.view_layer if ctx.scene == scene and ctx.view_layer is not None else scene.view_layers[0]


def _depsgraph(scene):
    ctx = bpy.context
    if ctx.scene == scene and ctx.view_layer is not None:
        return ctx.evaluated_depsgraph_get()
    return scene.view_layers[0].depsgraph


def _runs(mask) -> list[tuple[int, int]]:
    """[start, end) runs of True in a 1-D bool array."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[0::2].tolist(), d[1::2].tolist(), strict=True))


def _ranges_text(frames_idx, f0, limit=8) -> str:
    runs = _runs(np.isin(np.arange(max(frames_idx) + 1), frames_idx))
    txt = [f"{a + f0}" if b - a == 1 else f"{a + f0}-{b - 1 + f0}" for a, b in runs]
    return ", ".join(txt[:limit]) + (f", ... ({len(txt)} spans)" if len(txt) > limit else "")


def _smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _soft_ramp(x, w):
    """One-sided C1 lift for a depth ``x`` below the floor: 0 for x <= 0 (a foot at or above the floor is
    left alone), x^3 / (x^2 + w^2) above (slope 0 at 0, tends to x; it leaves at most w/2 at x = w)."""
    xp = np.maximum(x, 0.0)
    return xp ** 3 / (xp * xp + w * w)


def _widen_smooth(x, r):
    """``x`` [n] dilated by ``r`` frames, then smoothed over ``r`` frames: C1 and >= ``x`` everywhere."""
    n = len(x)
    if r <= 0 or not np.any(x > 0):
        return x
    pad = np.pad(x, r, mode="edge")
    wide = np.max(np.stack([pad[i:i + n] for i in range(2 * r + 1)]), axis=0)
    kern = np.hanning(2 * r + 3)[1:-1]
    kern /= kern.sum()
    return np.convolve(np.pad(wide, r, mode="edge"), kern, mode="valid")


def _up_axis(robj) -> np.ndarray:
    """World up (Z) in an armature's own axes, snapped to the nearest axis."""
    up = robj.T @ np.array([0.0, 0.0, 1.0])
    i = int(np.argmax(np.abs(up)))
    out = np.zeros(3)
    out[i] = np.sign(up[i])
    return out


def _facing(heads_arm: dict, robj, override, what: str, notes) -> tuple[np.ndarray, str]:
    """A rig's forward: detected from its left/right joint pairs in its **own armature axes** and
    snapped to the nearest armature axis (within ``_SNAP_DEG``), or ``override`` (an armature axis);
    returned in world (horizontal, unit) with its armature-axis name."""
    up = _up_axis(robj)
    if override is not None:
        f = np.array(_ARM_AXES[override])
        if abs(float(np.dot(f, up))) > 0.5:
            raise ValueError(f"facing {override}: that is {what}'s up axis, not a forward direction")
        name = override
    else:
        lat = np.zeros(3)
        for a, b in _LR_PAIRS:
            if a in heads_arm and b in heads_arm:
                lat += heads_arm[a] - heads_arm[b]
        lat -= np.dot(lat, up) * up
        if np.linalg.norm(lat) < 1e-12:
            raise ValueError(f"{what}: no mapped left/right bone pairs to tell which way it faces; "
                             "set facing")
        f = _unit(np.cross(_unit(lat), up))
        name = max((k for k, v in _ARM_AXES.items() if abs(np.dot(v, up)) < 0.5),
                   key=lambda k: float(np.dot(f, _ARM_AXES[k])))
        off = math.degrees(_vec_angle(f, _ARM_AXES[name]))
        if off <= _SNAP_DEG:
            f = np.array(_ARM_AXES[name])
        else:
            notes.append(f"facing: {what} faces {off:.0f} deg off its armature's {name} axis; using the "
                         "detected direction (set facing to override)")
    fw = robj @ f
    fw[2] = 0.0
    return _unit(fw), name


def _check_transform(obj, what: str) -> None:
    """Refuse a non-uniform scale, shear or mirror in ``obj``'s world transform."""
    m = np.array(obj.matrix_world, dtype=np.float64)[:3, :3]
    sv = np.linalg.svd(m, compute_uv=False)
    if sv.min() <= 1e-12 or sv.max() / sv.min() - 1.0 > 1e-3:
        sc = tuple(round(float(x), 4) for x in obj.matrix_world.to_scale())
        raise ValueError(f"{what} {obj.name!r} has a non-uniform scale or shear in its transform (scale "
                         f"{sc}, parents included): apply the scale (Ctrl+A > Scale) or make it uniform")
    if np.linalg.det(m) < 0:
        raise ValueError(f"{what} {obj.name!r} is mirrored (negative scale): apply the scale first")


class _Target:
    """Rest data of the target armature, all bones, parents before children."""

    def __init__(self, obj):
        bones = list(obj.data.bones)
        depth = {}
        for b in bones:
            d, p = 0, b.parent
            while p is not None:
                d, p = d + 1, p.parent
            depth[b.name] = d
        bones.sort(key=lambda b: depth[b.name])
        self.obj = obj
        self.names = [b.name for b in bones]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.parent = np.array([self.index[b.parent.name] if b.parent else -1 for b in bones])
        self.bones = bones
        mloc = np.array([np.array(b.matrix_local, dtype=np.float64) for b in bones])
        self.loc_r = _ortho(mloc[:, :3, :3])
        self.loc_t = mloc[:, :3, 3]
        # some files store rest matrices with a slight scale (Girush: 0.25 %); Blender applies it to
        # a bone's location channel, so the hips location is inverted with it
        self.scale_inv = np.linalg.inv(_T(self.loc_r) @ mloc[:, :3, :3])
        self.rel_r = np.empty_like(self.loc_r)
        self.rel_t = np.empty_like(self.loc_t)
        for i, p in enumerate(self.parent):  # rest relative to the parent, exact rest heads
            if p < 0:
                self.rel_r[i], self.rel_t[i] = self.loc_r[i], self.loc_t[i]
            else:
                self.rel_r[i] = self.loc_r[p].T @ self.loc_r[i]
                self.rel_t[i] = self.loc_r[p].T @ (self.loc_t[i] - self.loc_t[p])
        self.world = np.array(obj.matrix_world, dtype=np.float64)
        self.world_inv = np.linalg.inv(self.world)
        self.robj = _ortho(self.world[:3, :3])
        self.rest_w_rot = self.robj @ self.loc_r  # [B,3,3]
        self.rest_w_head = self.to_world(self.loc_t)
        # bone constraints: Copy Rotation local -> local (Replace, all axes, full influence) is simulated
        # (Girush's shin copies its knee piece); anything else is only reported
        self.copy_rot: dict[int, int] = {}
        self.other_constraints: dict[int, list[str]] = {}
        for i, n in enumerate(self.names):
            for con in obj.pose.bones[n].constraints:
                if not con.enabled or con.influence <= 0.0:
                    continue
                if (con.type == "COPY_ROTATION" and con.target == obj and con.subtarget in self.index
                        and self.index[con.subtarget] < i and con.owner_space == "LOCAL"
                        and con.target_space == "LOCAL" and con.mix_mode == "REPLACE" and con.influence >= 1.0
                        and con.use_x and con.use_y and con.use_z
                        and not (con.invert_x or con.invert_y or con.invert_z) and i not in self.copy_rot):
                    self.copy_rot[i] = self.index[con.subtarget]
                else:
                    self.other_constraints.setdefault(i, []).append(f"{con.type} '{con.name}'")

    def to_world(self, p):
        return _mv(self.world[:3, :3], p) + self.world[:3, 3]

    def to_arm(self, p):
        return _mv(self.world_inv[:3, :3], p) + self.world_inv[:3, 3]

    def descends(self, child, top) -> bool:
        p = self.parent[child]
        while p >= 0:
            if p == top:
                return True
            p = self.parent[p]
        return False

    def subtree(self, b) -> list[int]:
        return [b, *[i for i in range(len(self.names)) if self.descends(i, b)]]

    def basis(self, r_all, b):
        """Pose-basis rotations [n,3,3] of bone ``b`` given the armature-space rotations."""
        p = self.parent[b]
        base = r_all[:, p] @ self.rel_r[b] if p >= 0 else self.loc_r[b]
        return _T(base) @ r_all[:, b]

    def fk(self, rot: dict[int, np.ndarray], hips: int, hips_head: np.ndarray, n: int, constraints=True):
        """Armature-space rotations [n,B,3,3] and heads [n,B,3]: mapped bones take ``rot``, unmapped
        ones stay at rest relative to their posed parent (or copy a bone's local rotation, see
        ``copy_rot``, when ``constraints``), the hips head is ``hips_head``."""
        nb = len(self.names)
        r_all = np.empty((n, nb, 3, 3))
        h_all = np.empty((n, nb, 3))
        for b in range(nb):
            p = self.parent[b]
            if b in rot:
                r_all[:, b] = rot[b]
            elif p >= 0:
                r_all[:, b] = r_all[:, p] @ self.rel_r[b]
                if constraints and b in self.copy_rot:
                    r_all[:, b] = r_all[:, b] @ self.basis(r_all, self.copy_rot[b])
            else:
                r_all[:, b] = self.loc_r[b]
            if b == hips:
                h_all[:, b] = hips_head
            elif p >= 0:
                h_all[:, b] = h_all[:, p] + _mv(r_all[:, p], self.rel_t[b])
            else:
                h_all[:, b] = self.loc_t[b]
        return r_all, h_all


def _map_args(bone_map, facing, hips_scale):
    if not isinstance(bone_map, dict):
        raise ValueError(f"bone_map: expected a dict, got {type(bone_map).__name__}")
    if "format" in bone_map or isinstance(bone_map.get("bones"), dict):
        data = _validate_map(bone_map, "bone_map")
        facing = facing if facing is not None else data.get("facing")
        hips_scale = hips_scale if hips_scale is not None else data.get("hips_scale")
        bone_map = data["bones"]
    if facing is not None and facing not in FACINGS:
        raise ValueError(f"facing: expected None or one of {', '.join(FACINGS)}, got {facing!r}")
    if hips_scale is not None and (isinstance(hips_scale, bool) or not math.isfinite(float(hips_scale))
                                   or float(hips_scale) <= 0):
        raise ValueError(f"hips_scale: expected None or a positive number, got {hips_scale!r}")
    return bone_map, facing, (None if hips_scale is None else float(hips_scale))


def _pairs(bone_map, src_bones, tgt: _Target, source_obj, target_obj, notes) -> dict[str, int]:
    """canonical -> target bone index for the map entries both armatures have (others noted)."""
    pairs: dict[str, int] = {}
    seen: dict[str, str] = {}
    unknown, no_target, no_source = [], [], []
    for c, tb in bone_map.items():
        if not tb:
            continue
        if c not in _PARENT:
            unknown.append(repr(c))
            continue
        if tb in seen:
            raise ValueError(f"bone_map: target bone {tb!r} is mapped to both {seen[tb]} and {c}")
        seen[tb] = c
        if tb not in tgt.index:
            no_target.append(f"{tb!r} ({c})")
        elif c not in src_bones:
            no_source.append(c)
        else:
            pairs[c] = tgt.index[tb]
    if unknown:
        notes.append(f"map: not canonical afv bones (ignored): {', '.join(unknown)}")
    if no_target:
        notes.append(f"map: {target_obj.name} has no bone {', '.join(no_target)} (ignored)")
    if no_source:
        notes.append(f"map: {source_obj.name} has no {', '.join(no_source)} (ignored)")
    if "Hips" not in pairs:
        raise ValueError("bone_map: Hips is not mapped (or not on both armatures)")
    pairs = {c: pairs[c] for c in CANONICAL if c in pairs}
    bad = _hierarchy_breaks({c: b for c, b in pairs.items()}, lambda b, top: tgt.descends(b, top))
    if bad:
        txt = "; ".join(f"{tgt.names[pairs[c]]} ({c}) is not below {tgt.names[pairs[p]]} ({p})"
                        for c, p in bad)
        raise ValueError(f"bone_map contradicts {target_obj.name}'s hierarchy: {txt}")
    return pairs


def retarget(scene, source_obj, target_obj, bone_map, *, frames=None, lock_feet=True, hips_scale=None,
             facing=None, action_name=None) -> dict:
    """Bake ``source_obj``'s evaluated motion onto ``target_obj`` as a new action (see module doc)."""
    t_start = time.perf_counter()
    for what, o in (("source", source_obj), ("target", target_obj)):
        if o is None or o.type != "ARMATURE":
            raise ValueError(f"{what}: expected an armature object, got {getattr(o, 'name', o)!r}")
    if source_obj == target_obj:
        raise ValueError("source and target are the same armature")
    if generated_rigify(target_obj):
        raise ValueError(_RIGIFY_MSG.format(name=target_obj.name))
    vl = _view_layer(scene)
    vl.update()  # objects linked or collections excluded since the last update show only after it
    if vl.objects.get(source_obj.name) != source_obj:
        raise ValueError(f"source {source_obj.name!r} is not evaluated in view layer {vl.name!r} of scene "
                         f"{scene.name!r} (not in the scene, or its collection is excluded): include it")
    _check_transform(source_obj, "source")
    _check_transform(target_obj, "target")
    bone_map, facing, hips_scale = _map_args(bone_map, facing, hips_scale)
    f0, f1 = _frame_range(scene, frames)
    notes: list[str] = []
    src_bones = canonical_bones(source_obj)
    tgt = _Target(target_obj)
    pairs = _pairs(bone_map, src_bones, tgt, source_obj, target_obj, notes)
    _check_upright(tgt, pairs, target_obj)
    miss = [c for c in REQUIRED if c not in pairs]
    if miss:
        notes.append(f"required bones not mapped: {', '.join(miss)}")
    mapped_idx = set(pairs.values())
    for b in range(len(tgt.names)):
        name = tgt.names[b]
        if b in mapped_idx and b in tgt.copy_rot:
            notes.append(f"{name}: its Copy Rotation constraint overrides the retargeted keys")
        cons = tgt.other_constraints.get(b)
        if cons and (b in mapped_idx or any(tgt.descends(m, b) for m in mapped_idx)):
            notes.append(f"{name}: constraint {', '.join(cons)} is not simulated and acts on top of the "
                         "retargeted keys; the result may be off")
        if b in mapped_idx and not tgt.bones[b].use_inherit_rotation:
            notes.append(f"{name} does not inherit its parent's rotation; the result may be off")
    fps = scene.render.fps / scene.render.fps_base
    orig_frame, orig_sub = scene.frame_current, scene.frame_subframe
    muted: list = []  # the target's meshes need not deform while the source plays
    try:
        _mute_skin(scene, target_obj, muted)
        src = _sample(scene, source_obj, src_bones, pairs, f0, f1)
    finally:
        for md in muted:
            md.show_viewport = True
        scene.frame_set(orig_frame, subframe=orig_sub)
    feet = _feet_model(tgt, pairs, scene, target_obj)
    solve = _solve(src, tgt, pairs, feet, facing, hips_scale, lock_feet, f0, fps, notes)
    rep = _report(src, tgt, pairs, feet, solve, fps)
    if rep["ik_bones"]:
        notes.append(f"lock_feet: max_dir_err_deg / max_twist_err_deg leave out the IK'd legs "
                     f"({', '.join(rep['ik_bones'])})")
    name = action_name or f"{target_obj.name}_from_{source_obj.name}"
    action = _write(target_obj, tgt, pairs, solve, f0, f1, name, notes)
    scene.frame_set(orig_frame, subframe=orig_sub)  # the target shows the new action right away
    return {"action": action.name, "frames": [f0, f1], "mapped": len(pairs),
            "unmapped": [c for c in src_bones if c not in pairs], **rep, "notes": notes,
            "seconds": round(time.perf_counter() - t_start, 3)}


def _set_viewport(md, value: bool) -> None:
    md.show_viewport = value


def _mute_skin(scene, target_obj, muted: list) -> None:
    """Turn off (viewport) the enabled Armature modifiers that deform with ``target_obj``, adding each
    one to ``muted`` as soon as it is off, so sampling the source does not re-skin the target's meshes
    on every frame (a target that already has an action, e.g. from an earlier retarget, is animated).
    The caller turns exactly those back on, also when this fails part way."""
    for mobj in _skinned_meshes(scene, target_obj):
        for md in mobj.modifiers:
            if md.type == "ARMATURE" and md.object == target_obj and md.show_viewport:
                _set_viewport(md, False)
                muted.append(md)


def _hull_ids(points: np.ndarray) -> np.ndarray:
    """Indices of the points on their convex hull (bmesh); all of them when there are few. The lowest
    point of a rigidly moved point set is always one of these."""
    if len(points) <= 32:
        return np.arange(len(points))
    import bmesh

    bm = bmesh.new()
    try:
        vs = [bm.verts.new(p) for p in points]
        bm.verts.index_update()
        res = bmesh.ops.convex_hull(bm, input=vs, use_existing_faces=False)
        drop = {v.index for v in res.get("geom_interior", []) if isinstance(v, bmesh.types.BMVert)}
        drop |= {v.index for v in res.get("geom_unused", []) if isinstance(v, bmesh.types.BMVert)}
    finally:
        bm.free()
    keep = [i for i in range(len(points)) if i not in drop]
    return np.array(keep if keep else range(len(points)))


def _check_upright(tgt: _Target, pairs, target_obj) -> None:
    top = next((c for c in ("Head", "Neck", "Spine2", "Spine1", "Spine") if c in pairs), None)
    if top is None:
        return
    v = tgt.rest_w_head[pairs[top]] - tgt.rest_w_head[pairs["Hips"]]
    lean = math.degrees(_vec_angle(v, [0.0, 0.0, 1.0]))
    if lean > _UPRIGHT_DEG:
        raise ValueError(f"target {target_obj.name!r} does not stand upright in its rest pose (hips -> "
                         f"{top.lower()} is {lean:.0f} deg from world Z): re-import it with the right axes "
                         "(FBX: Forward -Z, Up Y) or rotate the object upright")


def _sample(scene, source_obj, src_bones, pairs, f0, f1) -> dict:
    """The source's evaluated world rotations/heads of the mapped canonical bones and its plant masks."""
    names = [pb.name for pb in source_obj.pose.bones]
    want = list(pairs)
    idx = np.array([names.index(src_bones[c]) for c in want])
    n = f1 - f0 + 1
    nb = len(names)
    buf = np.empty(nb * 16, dtype=np.float32)
    mats = np.empty((n, len(idx), 4, 4))
    world = np.empty((n, 4, 4))
    props = [k for k in source_obj.keys() if k.startswith(PLANT_PROP)]
    plants = {k[len(PLANT_PROP):]: np.zeros(n) for k in props}
    for i, f in enumerate(range(f0, f1 + 1)):
        scene.frame_set(f)
        ev = source_obj.evaluated_get(_depsgraph(scene))
        ev.pose.bones.foreach_get("matrix", buf)
        mats[i] = buf.reshape(nb, 4, 4).transpose(0, 2, 1)[idx]
        world[i] = np.array(ev.matrix_world)
        for k in props:
            try:
                plants[k[len(PLANT_PROP):]][i] = float(ev.get(k, 0.0))
            except (TypeError, ValueError):
                pass
    rest = np.array([np.array(source_obj.data.bones[src_bones[c]].matrix_local) for c in want])
    wr = _ortho(world[:, :3, :3])
    rot = wr[:, None] @ _ortho(mats[:, :, :3, :3])  # [n,M,3,3]
    head = np.einsum("nij,nmj->nmi", world[:, :3, :3], mats[:, :, :3, 3]) + world[:, None, :3, 3]
    rest_rot = wr[0][None] @ _ortho(rest[:, :3, :3])
    rest_head = _mv(world[0][:3, :3], rest[:, :3, 3]) + world[0][:3, 3]
    canon_plants = {}
    prefix = src_bones["Hips"][:-4]
    for ch, v in plants.items():
        c = ch if ch in _PARENT else (ch[len(prefix):] if prefix and ch.startswith(prefix) else None)
        if c in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"):
            canon_plants[c] = v > 0.5
    return {"order": want, "rot": rot, "head": head, "rest_rot": rest_rot, "rest_head": rest_head,
            "rest_head_arm": {c: rest[i, :3, 3] for i, c in enumerate(want)}, "robj": wr[0],
            "plants": canon_plants, "has_plants": bool(props), "n": n, "name": source_obj.name}


def _mapped_child(c, pairs, tgt) -> str | None:
    nxt = _TAIL[c]
    while nxt is not None and nxt not in pairs:
        nxt = _TAIL[nxt]
    if nxt is None or not tgt.descends(pairs[nxt], pairs[c]):
        return None
    return nxt


_LR_PAIRS = (("LeftUpLeg", "RightUpLeg"), ("LeftArm", "RightArm"), ("LeftShoulder", "RightShoulder"),
             ("LeftFoot", "RightFoot"))


# --- the feet: points that touch the floor ------------------------------------------------------------
def _skinned_meshes(scene, target_obj):
    """Meshes of the scene's view layer with an enabled Armature modifier on ``target_obj``."""
    for o in _view_layer(scene).objects:
        if o.type == "MESH" and any(md.type == "ARMATURE" and md.object == target_obj and md.show_viewport
                                    for md in o.modifiers):
            yield o


def _rest_low_vertices(scene, target_obj, fam_names, ankle_top, joint_floor):
    """The sole vertices of the target's skinned meshes that have a vertex group of ``fam_names``, as
    Blender evaluates them with the armature in its rest position (so modifiers before the Armature
    one, like Mirror or Displace, are in): world rest positions [N,3] and per vertex [(group name,
    weight)]. Positions are read in bulk; vertex groups only for the vertices of the low band (below
    ``_MESH_BAND`` of the ankle height above the provisional floor), the lowest ones always and the rest
    sampled down to ``_GROUP_READS``. The armature's pose position is set back afterwards."""
    meshes = [m for m in _skinned_meshes(scene, target_obj)
              if any(g.name in fam_names for g in m.vertex_groups)]
    if not meshes:
        return np.zeros((0, 3)), []
    arm, vl = target_obj.data, _view_layer(scene)
    old = arm.pose_position
    try:
        arm.pose_position = "REST"
        vl.update()
        dg = _depsgraph(scene)
        low = []  # (mesh, evaluated vertices, indices, world positions) below the ankle
        for mobj in meshes:
            ev = mobj.evaluated_get(dg)
            mv = ev.data.vertices
            co = np.empty(3 * len(mv))
            mv.foreach_get("co", co)
            mw = np.array(ev.matrix_world, dtype=np.float64)
            co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
            idx = np.flatnonzero(co[:, 2] <= ankle_top)
            if len(idx):
                low.append((mobj, mv, idx, co[idx]))
        if not low:
            return np.zeros((0, 3)), []
        z = np.concatenate([c[:, 2] for *_, c in low])
        floor_est = min(float(z.min()), joint_floor)
        top = max(floor_est + _MESH_BAND * (ankle_top - floor_est),
                  joint_floor + _MESH_BAND * (ankle_top - joint_floor))
        owner = np.concatenate([np.full(len(i), k) for k, (_, _, i, _) in enumerate(low)])
        vidx = np.concatenate([i for _, _, i, _ in low])
        pos = np.concatenate([c for *_, c in low])
        band = np.flatnonzero(z <= top)
        if len(band) > _GROUP_READS:  # the lowest always, the rest evenly sampled
            order = band[np.argsort(z[band], kind="stable")]
            spread = order[np.linspace(0, len(order) - 1, _GROUP_READS).astype(int)]
            keep = np.unique(np.concatenate([order[:_GROUP_READS // 8], spread]))
            band = keep
        names = [{g.index: g.name for g in m.vertex_groups} for m, *_ in low]
        groups = []
        for j in band.tolist():
            nm, mv = names[owner[j]], low[owner[j]][1]
            groups.append([(nm[g.group], g.weight) for g in mv[int(vidx[j])].groups if g.group in nm])
        return pos[band], groups
    finally:
        arm.pose_position = old
        vl.update()


def _feet_model(tgt: _Target, pairs, scene, target_obj) -> dict:
    """Per foot the points that can touch the floor: the heads and tails of the foot's bones (the
    foot's subtree: toe tips), its heel (the ankle's point on the floor) and the low rest vertices of
    skinned meshes weighted at least 25 % to the foot's bones, each deformed like Blender does (linear
    blend of up to ``_LBS_BONES`` bone influences); plus the rest floor height (world z of the lowest)."""
    sides = {s: tgt.subtree(pairs[f"{s}Foot"]) for s in ("Left", "Right") if f"{s}Foot" in pairs}
    if not sides:
        return {"sides": {}, "floor": None, "mesh": False}
    jb = {s: np.array([b for b in fam for _ in (0, 1)]) for s, fam in sides.items()}
    jp = {s: np.array([p for b in fam for p in (tgt.bones[b].head_local, tgt.bones[b].tail_local)])
          for s, fam in sides.items()}
    owner = {b: s for s, fam in sides.items() for b in fam}
    fam_names = {tgt.names[b] for b in owner}
    ankle_top = max(float(tgt.rest_w_head[fam[0]][2]) for fam in sides.values())
    joint_floor = float(min(tgt.to_world(p)[:, 2].min() for p in jp.values()))
    pos_w, groups = _rest_low_vertices(scene, target_obj, fam_names, ankle_top, joint_floor)
    side_of, infl_of = [], []
    for vg in groups:  # the band's vertices only (a few thousand at most)
        infl = [(tgt.index[nm], w) for nm, w in vg if nm in tgt.index and w > 0.0]
        total = sum(w for _, w in infl)
        foot: dict[str, float] = {}
        for bi, w in infl:
            if bi in owner:
                foot[owner[bi]] = foot.get(owner[bi], 0.0) + w / total
        best = max(foot.items(), key=lambda kv: kv[1]) if foot else (None, 0.0)
        if best[1] >= 0.25:
            infl = sorted(infl, key=lambda bw: -bw[1])[:_LBS_BONES]
            t = sum(w for _, w in infl)
            side_of.append(best[0])
            infl_of.append([(bi, w / t) for bi, w in infl])
        else:
            side_of.append(None)
            infl_of.append(None)
    side_of = np.array(side_of, dtype=object)
    kept = np.array([sd is not None for sd in side_of], dtype=bool)
    floor = min(joint_floor, float(pos_w[kept, 2].min()) if kept.any() else joint_floor)
    out = {}
    for side, fam in sides.items():
        ankle_b = fam[0]
        heel = tgt.rest_w_head[ankle_b].copy()
        heel[2] = floor
        band = floor + _MESH_BAND * max(tgt.rest_w_head[ankle_b][2] - floor, 1e-9)
        sel = np.flatnonzero(kept & (side_of == side) & (pos_w[:, 2] <= band))
        if len(sel) > 32:  # per main bone, the outline of its sole vertices (what can be lowest)
            main = np.array([infl_of[j][0][0] for j in sel.tolist()])
            sel = np.concatenate([sel[main == b][_hull_ids(pos_w[sel[main == b]])] for b in np.unique(main)])
        if len(sel) > _MESH_POINTS:
            order = sel[np.argsort(pos_w[sel, 2], kind="stable")]
            sel = np.unique(np.concatenate([order[:_MESH_POINTS // 8],
                                            order[np.linspace(0, len(order) - 1, _MESH_POINTS).astype(int)]]))
        nj = len(jb[side]) + 1
        m = nj + len(sel)
        idx = np.full((m, _LBS_BONES), ankle_b)
        wts = np.zeros((m, _LBS_BONES))
        arm_pts = np.empty((m, 3))
        idx[:nj - 1, 0], wts[:nj, 0] = jb[side], 1.0
        arm_pts[:nj - 1] = jp[side]
        arm_pts[nj - 1] = tgt.to_arm(heel)
        if len(sel):
            arm_pts[nj:] = tgt.to_arm(pos_w[sel])
            for i, j in enumerate(sel.tolist()):
                for kk, (bi, w) in enumerate(infl_of[j]):
                    idx[nj + i, kk], wts[nj + i, kk] = bi, w
        # each point in the frame of each of its bones (rest): loc = R_b^T (p - t_b)
        loc = np.einsum("mkji,mkj->mki", tgt.loc_r[idx], arm_pts[:, None, :] - tgt.loc_t[idx])
        out[side] = {"idx": idx, "w": wts, "local": loc}
    return {"sides": out, "floor": floor, "mesh": bool(kept.any())}


def _foot_points_world(tgt: _Target, r_all, h_all, side_model) -> np.ndarray:
    """World positions [n,M,3] of one foot's floor points in a pose (linear blend skinning), one
    transform per bone that influences them."""
    idx, w, local = side_model["idx"], side_model["w"], side_model["local"]
    p = np.zeros((r_all.shape[0], len(idx), 3))
    for b in np.unique(idx[w > 0]).tolist():
        rows, slots = np.nonzero((idx == b) & (w > 0))
        wl = w[rows, slots, None] * local[rows, slots]  # [M_b,3], weighted offsets in bone b's frame
        contrib = np.einsum("nij,mj->nmi", r_all[:, b], wl)
        contrib += h_all[:, b, None, :] * w[rows, slots][None, :, None]
        p[:, rows] += contrib  # a point has each bone at most once
    return tgt.to_world(p)


def _solve(src, tgt: _Target, pairs, feet, facing, hips_scale, lock_feet, f0, fps, notes) -> dict:
    order = src["order"]
    si = {c: i for i, c in enumerate(order)}
    n = src["n"]
    s_rot, s_head = src["rot"], src["head"]
    s_rest_rot, s_rest_head = src["rest_rot"], src["rest_head"]
    t_rest_rot = {c: tgt.rest_w_rot[b] for c, b in pairs.items()}
    t_rest_head = {c: tgt.rest_w_head[b] for c, b in pairs.items()}
    # facing, each rig in its own armature axes
    s_heads = {c: src["rest_head_arm"][c] for c in pairs}
    t_heads = {c: tgt.loc_t[b] for c, b in pairs.items()}
    fs, _ = _facing(s_heads, src["robj"], None, f"source {src['name']!r}", notes)
    ft, ft_name = _facing(t_heads, tgt.robj, facing, f"target {tgt.obj.name!r}", notes)
    _swap_note(tgt, pairs, ft_name, notes)
    yaw = math.atan2(float(np.cross(ft, fs)[2]), float(np.dot(ft, fs)))
    big_f = _rot_z(yaw)
    # rest alignment
    child = {c: _mapped_child(c, pairs, tgt) for c in pairs}
    d_s, d_t = {}, {}
    for c, k in child.items():
        if k is None:
            continue
        ds = s_rest_head[si[k]] - s_rest_head[si[c]]
        dt = t_rest_head[k] - t_rest_head[c]
        if np.linalg.norm(ds) < 1e-9 or np.linalg.norm(dt) < 1e-9:
            child[c] = None
            continue
        d_s[c], d_t[c] = _unit(ds), _unit(dt)
    align: dict[str, np.ndarray] = {}
    for c in pairs:
        if c in _YAW_ONLY:
            align[c] = big_f
        elif child[c] is not None:
            align[c] = _arc(big_f @ d_t[c], d_s[c]) @ big_f
        else:
            p = _PARENT[c]
            while p is not None and p not in align:
                p = _PARENT[p]
            align[c] = align[p] if p is not None else big_f
    # world rotations of the mapped target bones
    delta = {c: s_rot[:, si[c]] @ _T(s_rest_rot[si[c]]) for c in pairs}
    w_t = {}
    for c in pairs:
        w = delta[c] @ (align[c] @ t_rest_rot[c])
        if c not in _YAW_ONLY and child[c] is not None:
            pred = _mv(delta[c], d_s[c])
            actual = s_head[:, si[child[c]]] - s_head[:, si[c]]
            w = _arc(pred, actual) @ w
        w_t[c] = w
    # scale and hips placement
    legs = all(x in pairs for x in ("LeftUpLeg", "RightUpLeg")) and all(
        tgt.descends(pairs[x], pairs["Hips"]) for x in ("LeftUpLeg", "RightUpLeg"))
    anchor_c = ("LeftUpLeg", "RightUpLeg") if legs else ("Hips",)
    if not legs:
        notes.append("hips: the thighs are not both mapped below the hips; the hips joint is the anchor")
    a_s = np.mean([s_head[:, si[c]] for c in anchor_c], axis=0)
    a_s_rest = np.mean([s_rest_head[si[c]] for c in anchor_c], axis=0)
    a_t_rest = np.mean([t_rest_head[c] for c in anchor_c], axis=0)
    floor = feet["floor"]
    if floor is None:
        floor = float(min(tgt.rest_w_head[:, 2]))
        notes.append("feet: no foot is mapped; the floor is the target's lowest joint")
    k_auto = None
    hs, ht = a_s_rest[2], a_t_rest[2] - floor  # pelvis anchors above the floor (the source's is z = 0)
    if hs > 1e-9 and ht > 1e-9:
        k_auto = float(ht / hs)
    else:
        notes.append("hips_scale: cannot measure the rest hip heights above the floor; using 1")
    k = hips_scale if hips_scale is not None else (k_auto or 1.0)
    goal = a_t_rest + k * (a_s - a_s_rest)
    hb = pairs["Hips"]
    d_hips = w_t["Hips"] @ _T(t_rest_rot["Hips"])
    hips_head_w = goal - _mv(d_hips, a_t_rest - t_rest_head["Hips"])
    rot_arm = {pairs[c]: _T(tgt.robj) @ w_t[c] for c in pairs}
    hips_head = tgt.to_arm(hips_head_w)
    r_all, h_all = tgt.fk(rot_arm, hb, hips_head, n, constraints=False)  # the intended (rigid) pose
    solve = {"yaw": yaw, "ft": ft, "ft_name": ft_name, "k": k, "align": align, "child": child, "d_s": d_s,
             "delta": delta, "rot_arm": rot_arm, "hips_head": hips_head, "r_all": r_all, "h_all": h_all,
             "floor": floor}
    if lock_feet:
        if not src["has_plants"]:
            notes.append(f"lock_feet: {src['name']} has no plant masks ({PLANT_PROP}*): feet not locked")
        else:
            goals = [_foot_goal(side, src, tgt, pairs, feet, solve, fps, f0, notes)
                     for side in ("Left", "Right")]
            goals = [g for g in goals if g is not None]
            drop = _hips_drop(goals, tgt, pairs, solve)
            if np.any(drop > 0):
                hips_head = tgt.to_arm(hips_head_w - drop[:, None] * np.array([0.0, 0.0, 1.0]))
                solve["hips_head"] = hips_head
                solve["r_all"], solve["h_all"] = tgt.fk(rot_arm, hb, hips_head, n, constraints=False)
                moved = np.flatnonzero(drop > 1e-9)
                notes.append(f"lock_feet: hips lowered by up to {drop.max() * 100.0 / k:.1f} cm "
                             f"(fighter scale) on {len(moved)} frames ({_ranges_text(moved, f0)}) so the "
                             "planted feet reach")
            legs_rot = {pairs[c]: rot_arm[pairs[c]].copy() for g in goals for c in g["bones"][:2]}
            solve["ik_sides"] = [g["side"] for g in goals]
            ik_notes = [m for g in goals for m in _leg_ik(g, tgt, pairs, solve, f0)]
            for g in goals:
                g["lift"], g["aim"], g["want"] = np.zeros(n), np.zeros((n, 3)), g["goal"].copy()
            tol = 1e-4 * k  # 0.01 fighter cm
            for _ in range(3):
                # check the solved pose: simulated constraints change a leg's length as it bends (Girush's
                # two-joint knee), and sole vertices weighted to the shin move with the IK'd shin. A
                # planted run that still sinks comes up as a whole (its lock, so it neither slides nor
                # hovers, the fades rebuilt C1); elsewhere the floor ramp lifts what is left; constraint
                # ankle errors re-aim the IK. Then IK again from the same start.
                pose = _fit_constraints(tgt, pairs, solve, n) if tgt.copy_rot else rot_arm
                r_now, h_now = tgt.fk(pose, hb, hips_head, n)
                change = 0.0
                for g in goals:
                    ank = tgt.to_world(h_now[:, pairs[g["bones"][2]]])
                    model = feet["sides"].get(g["side"])
                    if model is not None:
                        lowest = _foot_points_world(tgt, r_now, h_now, model)[:, :, 2].min(axis=1)
                        depth = solve["floor"] - lowest
                        for a, b in _runs(g["planted"]):
                            sink = float(depth[a:b].max())
                            if sink > tol:
                                g["lock_goal"][a:b, 2] += sink
                                change = max(change, sink)
                        lift = np.where(g["planted"], 0.0, _soft_ramp(depth, _FLOOR_SOFT_CM * k / 100.0))
                        g["lift"] += lift
                        change = max(change, float(lift.max()))
                    if tgt.copy_rot:
                        err = g["want"] - ank
                        g["aim"] += err
                        change = max(change, float(np.abs(err).max()))
                    want = _goal_from(g, solve["floor"], k)[0]
                    want[:, 2] += g["lift"]
                    g["want"], g["goal"] = want, want + g["aim"]
                if change < tol:
                    break
                rot_arm.update({b: r.copy() for b, r in legs_rot.items()})
                ik_notes = [m for g in goals for m in _leg_ik(g, tgt, pairs, solve, f0)]
            notes.extend(ik_notes)
    rot_fit = _fit_constraints(tgt, pairs, solve, n, notes) if tgt.copy_rot else solve["rot_arm"]
    solve["r_all"], solve["h_all"] = tgt.fk(rot_fit, hb, hips_head, n)
    return solve


def _swap_note(tgt: _Target, pairs, ft_name, notes) -> None:
    """Note a map whose Left/Right bones look swapped: the toes point backwards for the forward the
    left/right pairs give."""
    f = np.array(_ARM_AXES[ft_name])
    up = _up_axis(tgt.robj)
    toes = np.zeros(3)
    for side in ("Left", "Right"):
        if f"{side}Foot" in pairs and f"{side}ToeBase" in pairs:
            toes += tgt.loc_t[pairs[f"{side}ToeBase"]] - tgt.loc_t[pairs[f"{side}Foot"]]
    toes -= np.dot(toes, up) * up
    if np.linalg.norm(toes) > 1e-12 and float(np.dot(_unit(toes), f)) < -0.3:
        notes.append("map: the bones mapped as Left look like the character's right (its toes point away "
                     "from the forward its left/right bones give): check the Left/Right mapping or set "
                     "facing")


def _expmap(w):
    """Rotation vectors [n,3] -> rotation matrices [n,3,3]."""
    th = np.linalg.norm(w, axis=-1)
    k = _skew(_unit(np.where(th[:, None] > 1e-12, w, [1.0, 0.0, 0.0])))
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def _fit_constraints(tgt: _Target, pairs, solve, n, notes=None, iters=15, tol=1e-6) -> dict:
    """The mapped rotations (a copy of ``solve["rot_arm"]``, the intended pose of the rigid model)
    fitted for the simulated constraints: a mapped bone whose way down to its mapped child passes bones
    that copy its local rotation (Girush's knee piece ``Tibia`` and its ``Tibia.001``) gets, per frame,
    the rotation (damped Gauss-Newton on 3 parameters) that with the copies puts the child joint in
    the intended direction and keeps the bone's intended twist about it."""
    rot = dict(solve["rot_arm"])
    hb, hh = pairs["Hips"], solve["hips_head"]
    _, h_want = tgt.fk(rot, hb, hh, n, constraints=False)
    todo = []
    for c, k in solve["child"].items():
        if k is None or c == "Hips":
            continue
        b, kb = pairs[c], pairs[k]
        chain, x = [], tgt.parent[kb]
        while x >= 0 and x != b:
            chain.append(x)
            x = tgt.parent[x]
        if any(tgt.copy_rot.get(x) == b for x in chain):
            todo.append((b, kb))
    worst = 0.0
    for b, kb in todo:  # parents first: a fitted bone's children are fitted after it
        want_dir = _unit(h_want[:, kb] - h_want[:, b])
        want_rot = rot[b]
        side = np.where(np.abs(want_dir[:, :1]) < 0.9, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        u1 = _unit(np.cross(want_dir, side))
        u2 = np.cross(want_dir, u1)

        def resid(r, b=b, kb=kb, want_dir=want_dir, want_rot=want_rot, u1=u1, u2=u2):
            r_all, h = tgt.fk({**rot, b: r}, hb, hh, n)
            got = _unit(h[:, kb] - h[:, b])
            q = _quat(r_all[:, b] @ _T(want_rot))
            twist = 2.0 * np.arctan2(np.sum(q[:, 1:] * want_dir, -1), q[:, 0])
            return np.stack([np.sum(got * u1, -1), np.sum(got * u2, -1), twist], -1)

        cur = rot[b]
        r0 = resid(cur)
        lam = np.full(n, 1e-3)
        eps = 1e-5
        for _ in range(iters):
            if float(np.abs(r0).max()) < tol:
                break
            jac = np.stack([(resid(_expmap(np.tile(eps * np.eye(3)[i], (n, 1))) @ cur) - r0) / eps
                            for i in range(3)], -1)  # [n, 3 residuals, 3 params]
            jt = _T(jac)
            step = -np.linalg.solve(jt @ jac + lam[:, None, None] * np.eye(3), (jt @ r0[..., None]))[..., 0]
            trial = _expmap(step) @ cur
            r1 = resid(trial)
            better = np.sum(r1 * r1, -1) < np.sum(r0 * r0, -1)
            cur = np.where(better[:, None, None], trial, cur)
            r0 = np.where(better[:, None], r1, r0)
            lam = np.where(better, lam * 0.3, lam * 10.0)
        rot[b] = cur
        worst = max(worst, float(np.degrees(np.abs(r0)).max()))
    if worst > 0.05 and notes is not None:
        notes.append(f"constraints: copied rotations could not be fitted exactly ({worst:.2f} deg left)")
    return rot


# --- foot locking ---------------------------------------------------------------------------------------
def _hermite(c0, m0, c1, m1, length):
    """Cubic Hermite [length+1, 3] from (c0, slope m0 per frame) to (c1, m1) over ``length`` frames."""
    t = np.linspace(0.0, 1.0, length + 1)[:, None]
    h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
    h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * c0 + h10 * length * m0 + h01 * c1 + h11 * length * m1


def _blend_frames(c, m, accel) -> int:
    """Frames L to fade a correction ``c`` (moving ``m`` per frame) to rest with a cubic Hermite whose
    acceleration (largest at its ends: 6|c|/L^2 + 4|m|/L) stays within ``accel`` per frame^2: the
    smallest such L, at least BLEND_FRAMES, at most _EDGE_MAX_FRAMES."""
    c, m = float(np.linalg.norm(c)), float(np.linalg.norm(m))
    need = (4.0 * m + math.sqrt(16.0 * m * m + 24.0 * accel * c)) / (2.0 * accel)
    return int(min(max(math.ceil(need - 1e-9), BLEND_FRAMES), _EDGE_MAX_FRAMES))


def _corrections(locked, lock_goal, ank_w, accel) -> np.ndarray:
    """The ankle correction [n,3] (goal - retargeted ankle): exact on locked frames, and between them a
    C1 fade (value and velocity matched at each run's edge, Hermite, long enough for ``accel``)."""
    n = len(ank_w)
    c = np.zeros((n, 3))
    runs = _runs(locked)
    if not runs:
        return c
    for a, b in runs:
        c[a:b] = lock_goal[a:b] - ank_w[a:b]

    def end_state(run):  # correction and its slope (per frame) at the run's last frame
        a, b = run
        return b - 1, c[b - 1].copy(), (c[b - 1] - c[b - 2]) if b - a >= 2 else np.zeros(3)

    def start_state(run):
        a, b = run
        return a, c[a].copy(), (c[a + 1] - c[a]) if b - a >= 2 else np.zeros(3)

    zero = np.zeros(3)
    edges = [(None, runs[0]), *itertools.pairwise(runs), (runs[-1], None)]
    for prev, nxt in edges:
        out = end_state(prev) if prev is not None else None
        inn = start_state(nxt) if nxt is not None else None
        n1 = _blend_frames(out[1], out[2], accel) if out else 0
        n2 = _blend_frames(inn[1], inn[2], accel) if inn else 0
        if out and inn and inn[0] - out[0] < n1 + n2:  # one fade across a short gap
            c[out[0]:inn[0] + 1] = _hermite(out[1], out[2], inn[1], inn[2], inn[0] - out[0])
            continue
        if out:
            e = out[0]
            seg = _hermite(out[1], out[2], zero, zero, n1)
            stop = min(n, e + n1 + 1)
            c[e:stop] = seg[: stop - e]
        if inn:
            s = inn[0]
            seg = _hermite(zero, zero, inn[1], inn[2], n2)
            start = max(0, s - n2)
            c[start:s + 1] = seg[start - (s - n2):]
    return c


def _foot_goal(side, src, tgt: _Target, pairs, feet, solve, fps, f0, notes) -> dict | None:
    """The ankle goal per frame (world) for one foot: each planted run locked (a planted Foot channel
    holds the ankle, a planted ToeBase channel alone the toe joint; the lock moves over from one to the
    other where they hand over, so heel-off never jumps), C1 fades in and out, and the foot's floor
    points (toe tip, heel, sole vertices) kept above the floor with a smoothed lift."""
    th, sh, ft = (f"{side}UpLeg", f"{side}Leg", f"{side}Foot")
    toe_c = f"{side}ToeBase"
    if not all(c in pairs for c in (th, sh, ft)):
        notes.append(f"lock_feet: {side.lower()} leg not fully mapped (thigh, shin, foot); not locked")
        return None
    n = src["n"]
    si = {c: i for i, c in enumerate(src["order"])}
    k = solve["k"]
    plants = src["plants"]
    has_toe = toe_c in pairs
    p_foot = plants.get(ft, np.zeros(n, bool))
    p_toe = plants.get(toe_c, np.zeros(n, bool))
    if not has_toe:
        p_foot, p_toe = p_foot | p_toe, np.zeros(n, bool)
    r_all, h_all = solve["r_all"], solve["h_all"]
    ank_w = tgt.to_world(h_all[:, pairs[ft]])
    toe_w = tgt.to_world(h_all[:, pairs[toe_c]]) if has_toe else ank_w
    off = toe_w - ank_w
    zr = {"foot": tgt.rest_w_head[pairs[ft]][2], "toe": tgt.rest_w_head[pairs[toe_c]][2] if has_toe else 0.0}
    s_z = {"foot": src["head"][:, si[ft], 2], "toe": src["head"][:, si[toe_c], 2] if has_toe else None}
    s_rz = {"foot": src["rest_head"][si[ft]][2], "toe": src["rest_head"][si[toe_c]][2] if has_toe else 0.0}
    planted = p_foot | p_toe
    mode = np.where(p_foot, 0, 1)  # 0: the ankle holds, 1: the toe joint holds
    model = feet["sides"].get(side)
    if model is not None:  # the foot's lowest floor point below the ankle, per frame (rigid with it)
        low = _foot_points_world(tgt, r_all, h_all, model)[:, :, 2].min(axis=1) - ank_w[:, 2]
    else:
        low = np.zeros(n)
    floor = solve["floor"]
    lock_goal = ank_w.copy()
    for a, b in _runs(planted):
        t, prev = a, None  # the previous hold's last ankle goal
        holds = []  # (start, end, ankle-goal path over all frames)
        while t < b:
            e = t
            while e < b and mode[e] == mode[t]:
                e += 1
            ch = "foot" if mode[t] == 0 else "toe"
            if prev is None:  # the run's first hold: where the retargeted joint is, at the source's height
                held = np.median((ank_w if ch == "foot" else toe_w)[t:e], axis=0)
                held[2] = zr[ch] + k * (float(np.median(s_z[ch][t:e])) - s_rz[ch])
            else:  # handing over: the joint that holds now stays where it is (heel-off pivots on the toe)
                held = prev if ch == "foot" else prev + off[t - 1]
            path = np.repeat(held[None], n, 0) if ch == "foot" else held - off
            sink = float(np.max(floor - (path[t:e, 2] + low[t:e])))
            if sink > 0.0:  # a hold never sinks the sole into the floor: the whole hold comes up
                path = path + np.array([0.0, 0.0, sink])
            lock_goal[t:e] = path[t:e]
            holds.append((t, e, path))
            prev = path[e - 1].copy()
            t = e
        for (_, e0, p0), (s1, e1, p1) in itertools.pairwise(holds):  # ease each handover, centred on it
            h = max(1, min(BLEND_FRAMES // 2, e0 - a, e1 - s1))
            lo, hi = max(a, s1 - h), min(b, s1 + h)
            for t2 in range(lo, hi):
                w = _smoothstep((t2 - lo + 0.5) / (hi - lo))
                lock_goal[t2] = (1.0 - w) * p0[t2] + w * p1[t2]
    accel = _EDGE_ACCEL * k / 100.0 / (fps * fps)  # world units per frame^2
    g = {"side": side, "weight": planted.astype(float), "bones": (th, sh, ft), "planted": planted,
         "lock_goal": lock_goal, "ank_w": ank_w, "low": low, "accel": accel, "model": model is not None}
    corr = _goal_from(g, floor, k)[1]
    acc = np.zeros(n)
    acc[1:-1] = np.linalg.norm(corr[2:] - 2.0 * corr[1:-1] + corr[:-2], axis=-1)
    over = np.flatnonzero(~planted & (acc > accel * 1.001))
    if len(over):  # a fade capped at _EDGE_MAX_FRAMES, or squeezed into a short gap between plants
        notes.append(f"lock_feet: {side.lower()} foot: the lock fades in/out harder than "
                     f"{_EDGE_ACCEL:.0f} cm/s^2 on {len(over)} frames ({_ranges_text(over, f0)}): up to "
                     f"{acc[over].max() * fps * fps * 100.0 / k:.0f} cm/s^2 (fighter scale; a short gap or "
                     "a long fade)")
    g["goal"] = _goal_from(g, floor, k)[0]
    return g


def _goal_from(g, floor, k):
    """(ankle goal [n,3], the lock correction [n,3]) of a foot: its locks with C1 fades, and the floor
    ramp under the foot's lowest point."""
    corr = _corrections(g["planted"], g["lock_goal"], g["ank_w"], g["accel"])
    goal = g["ank_w"] + corr
    if g["model"]:
        goal[:, 2] += _soft_ramp(floor - (goal[:, 2] + g["low"]), _FLOOR_SOFT_CM * k / 100.0)
    return goal, corr


def _leg_points(g, tgt: _Target, pairs, solve):
    """World hip, knee and ankle joints [n,3] of the current (rigid) pose and the two bone lengths."""
    h_all = solve["h_all"]
    hip_w, knee_w, ank_w = (tgt.to_world(h_all[:, pairs[c]]) for c in g["bones"])
    l1 = np.linalg.norm(knee_w - hip_w, axis=-1)
    return hip_w, knee_w, ank_w, l1, np.linalg.norm(ank_w - knee_w, axis=-1)


def _hips_drop(goals, tgt: _Target, pairs, solve) -> np.ndarray:
    """How far (world z, per frame) the hips must come down so every locked foot can reach its goal:
    the needed drop where a lock holds, at most HIPS_DROP_MAX of the leg, widened and smoothed over
    HIPS_DROP_FRAMES so it never pops; 0 where the feet reach anyway."""
    n = solve["h_all"].shape[0]
    need = np.zeros(n)
    for g in goals:
        hip_w, _, _, l1, l2 = _leg_points(g, tgt, pairs, solve)
        d = g["goal"] - hip_w
        reach = (l1 + l2) * _REACH
        dz = d[:, 2]
        dist2 = np.sum(d * d, axis=-1)
        disc = dz * dz - (dist2 - reach * reach)  # |d + drop * z| = reach, smallest drop
        drop = np.where(disc >= 0.0, -dz - np.sqrt(np.maximum(disc, 0.0)), np.maximum(-dz, 0.0))
        drop = np.where((dist2 > reach * reach) & (g["weight"] > 0), np.maximum(drop, 0.0), 0.0)
        need = np.maximum(need, np.minimum(drop, HIPS_DROP_MAX * (l1 + l2)))
    return _widen_smooth(need, HIPS_DROP_FRAMES)


def _leg_ik(g, tgt: _Target, pairs, solve, f0) -> list[str]:
    """Two-bone IK of the mapped thigh/shin onto the ankle goal (foot world rotation kept), in
    ``solve["rot_arm"]``; returns the notes (goals out of reach)."""
    th, sh, _ = g["bones"]
    side, goal = g["side"], g["goal"]
    hip_w, knee_w, ank_w, l1, l2 = _leg_points(g, tgt, pairs, solve)
    active = np.linalg.norm(goal - ank_w, axis=-1) > 1e-7 * max(1.0, float(l1[0]))
    if not np.any(active):
        return []
    d = goal - hip_w
    dist = np.linalg.norm(d, axis=-1)
    hi = (l1 + l2) * (1.0 - 1e-6)
    lo = np.abs(l1 - l2) * (1.0 + 1e-6) + 1e-9
    far = active & (dist > hi)
    near = active & (dist < lo)
    dist_c = np.clip(dist, lo, hi)
    u = _unit(d)
    kv = knee_w - hip_w
    p = kv - np.sum(kv * u, -1, keepdims=True) * u  # the knee's retargeted bend direction
    # a nearly straight leg has no bend plane: lean on the thigh's forward (its rest facing, posed),
    # fading out as the knee bends
    fwd = _mv(tgt.robj @ solve["rot_arm"][pairs[th]] @ _T(tgt.rest_w_rot[pairs[th]]), solve["ft"])
    straight = np.clip(1.0 - np.linalg.norm(p, axis=-1) / (0.02 * l1), 0.0, 1.0)
    p = p + (0.02 * l1 * straight)[:, None] * _unit(fwd - np.sum(fwd * u, -1, keepdims=True) * u)
    v = _unit(p - np.sum(p * u, -1, keepdims=True) * u)
    a_ = (l1 ** 2 - l2 ** 2 + dist_c ** 2) / (2.0 * dist_c)
    b_ = np.sqrt(np.maximum(l1 ** 2 - a_ ** 2, 0.0))
    knee_new = hip_w + a_[:, None] * u + b_[:, None] * v
    goal_c = hip_w + dist_c[:, None] * u
    r1 = _arc(kv, knee_new - hip_w)
    ank_r1 = knee_new + _mv(r1, ank_w - knee_w)
    r2 = _arc(ank_r1 - knee_new, goal_c - knee_new)
    eye = np.eye(3)
    r1 = np.where(active[:, None, None], r1, eye)
    r2 = np.where(active[:, None, None], r2, eye)
    ro = tgt.robj
    r1_arm = _T(ro) @ r1 @ ro
    r2_arm = _T(ro) @ r2 @ ro
    rot = solve["rot_arm"]
    rot[pairs[th]] = r1_arm @ rot[pairs[th]]
    rot[pairs[sh]] = r2_arm @ r1_arm @ rot[pairs[sh]]
    bad = np.flatnonzero(far | near)
    if not len(bad):
        return []
    short = (dist - (l1 + l2))[far]
    what = f"up to {short.max() * 100.0 / solve['k']:.1f} cm short" if far.any() else "too close to the hip"
    return [f"lock_feet: {side.lower()} foot IK target out of reach on {len(bad)} frames "
            f"({_ranges_text(bad, f0)}): {what} (fighter scale)"]


# --- writing the action -------------------------------------------------------------------------------
def _rest_value(prop):
    if prop == "location":
        return [0.0, 0.0, 0.0]
    if prop == "scale":
        return [1.0, 1.0, 1.0]
    if prop == "rotation_quaternion":
        return [1.0, 0.0, 0.0, 0.0]
    if prop == "rotation_axis_angle":
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0]


def _new_fcurve(bag, path, index, frames, values, group):
    fc = bag.fcurves.new(path, index=index, group_name=group)
    n = len(frames)
    fc.keyframe_points.add(n)
    co = np.empty(2 * n)
    co[0::2] = frames
    co[1::2] = values
    fc.keyframe_points.foreach_set("co", co)
    fc.keyframe_points.foreach_set("interpolation", [_LINEAR] * n)
    fc.update()
    return fc


def _other_paths(ad) -> set[str]:
    """Data paths that the target's existing animation (active action, NLA strips) drives."""
    out: set[str] = set()
    if ad is None:
        return out
    pairs = []
    if ad.action is not None:
        pairs.append((ad.action, ad.action_slot))
    for t in ad.nla_tracks:
        for s in t.strips:
            if s.action is not None and not t.mute and not s.mute:
                pairs.append((s.action, s.action_slot))
    for action, slot in pairs:
        bag = anim_utils.action_get_channelbag_for_slot(action, slot) if slot is not None else None
        if bag is not None:
            out.update(fc.data_path for fc in bag.fcurves)
    return out


_ROT_PROPS = {"QUATERNION": "rotation_quaternion", "AXIS_ANGLE": "rotation_axis_angle"}


def _rotation_values(q, pb):
    """Quaternions [n,4] (continuous) as the pose bone's own rotation channel values."""
    mode = pb.rotation_mode
    if mode == "QUATERNION":
        return "rotation_quaternion", q
    if mode == "AXIS_ANGLE":
        ang = 2.0 * np.arccos(np.clip(q[:, 0], -1.0, 1.0))
        ax = _unit(np.where(np.linalg.norm(q[:, 1:], axis=-1, keepdims=True) > 1e-12, q[:, 1:],
                            [0.0, 1.0, 0.0]))
        return "rotation_axis_angle", np.concatenate([ang[:, None], ax], -1)
    vals, prev = [], None
    for qi in q:
        e = Quaternion(qi).to_euler(mode, prev) if prev is not None else Quaternion(qi).to_euler(mode)
        vals.append(tuple(e))
        prev = e
    return "rotation_euler", np.array(vals)


def _hips_location(tgt: _Target, solve, b):
    r_all, h_all = solve["r_all"], solve["h_all"]
    p = tgt.parent[b]
    bone = tgt.bones[b]
    if p >= 0:
        rel = _mv(_T(r_all[:, p]), h_all[:, b] - h_all[:, p]) - tgt.rel_t[b]
        if not bone.use_local_location:
            return rel
        return _mv((tgt.scale_inv[b] @ _T(tgt.rel_r[b]))[None], rel)
    rel = h_all[:, b] - tgt.loc_t[b]
    if not bone.use_local_location:
        return rel
    return _mv((tgt.scale_inv[b] @ _T(tgt.loc_r[b]))[None], rel)


def _write(target_obj, tgt: _Target, pairs, solve, f0, f1, name, notes):
    """The new action, assigned to the target; on any error it is removed again and the target's
    animation state is put back."""
    r_all = solve["r_all"]
    frames = np.arange(f0, f1 + 1, dtype=np.float64)
    ad = target_obj.animation_data
    other = _other_paths(ad)
    mapped = set(pairs.values())
    hb = pairs["Hips"]
    action = bpy.data.actions.new(name)
    undo = []  # (callable) to put the target back if anything below fails
    try:
        action.use_fake_user = True
        slot = action.slots.new(id_type="OBJECT", name=target_obj.name)
        bag = anim_utils.action_ensure_channelbag_for_slot(action, slot)
        for b, bname in enumerate(tgt.names):
            pb = target_obj.pose.bones[bname]
            dpath = f'pose.bones["{bpy.utils.escape_identifier(bname)}"]'
            rprop = _ROT_PROPS.get(pb.rotation_mode, "rotation_euler")
            keyed = set()
            if b in mapped:
                rprop, vals = _rotation_values(_continuous(_quat(tgt.basis(r_all, b))), pb)
                for c in range(vals.shape[1]):
                    _new_fcurve(bag, f"{dpath}.{rprop}", c, frames, vals[:, c], bname)
                keyed.add(rprop)
            if b == hb:
                if tgt.bones[b].use_connect:
                    notes.append(f"{bname} is connected to its parent: its location keys have no effect")
                loc = _hips_location(tgt, solve, b)
                for c in range(3):
                    _new_fcurve(bag, f"{dpath}.location", c, frames, loc[:, c], bname)
                keyed.add("location")
            for prop in ("location", rprop, "scale"):
                if prop in keyed:
                    continue
                rest = _rest_value(prop)
                cur = list(getattr(pb, prop))
                off_rest = any(abs(a - r) > 1e-6 for a, r in zip(cur, rest, strict=True))
                if f"{dpath}.{prop}" in other or off_rest:
                    for c, v in enumerate(rest):
                        _new_fcurve(bag, f"{dpath}.{prop}", c, frames[:1], [v], bname)
        action.use_frame_range = True
        action.frame_start = float(f0)
        action.frame_end = float(max(f1, f0 + 1))
        if ad is None:
            ad = target_obj.animation_data_create()
            undo.append(target_obj.animation_data_clear)
        if ad.use_tweak_mode:
            ad.use_tweak_mode = False
            undo.append(lambda: setattr(ad, "use_tweak_mode", True))
            notes.append(f"{target_obj.name} was in NLA tweak mode: left it")
        old, old_slot, old_fake = ad.action, ad.action_slot, None
        if old is not None and old.users <= 1 and not old.use_fake_user:
            old_fake = old
            old.use_fake_user = True
            notes.append(f"the previous action {old.name!r} was given a fake user so it is kept")
        undo.append(lambda: _restore_action(ad, old, old_slot, old_fake))
        _assign(ad, action, slot)
        if ad.action_blend_type != "REPLACE" or ad.action_influence != 1.0:
            blend, infl = ad.action_blend_type, ad.action_influence
            undo.append(lambda: (setattr(ad, "action_blend_type", blend),
                                 setattr(ad, "action_influence", infl)))
            ad.action_blend_type = "REPLACE"
            ad.action_influence = 1.0
            notes.append(f"{target_obj.name}: active action blending set to Replace at influence 1")
        if any(not t.mute for t in ad.nla_tracks):
            notes.append(f"{target_obj.name} has NLA tracks: they still play under the new action")
    except Exception:
        for fn in reversed(undo):
            try:
                fn()
            except Exception:  # best effort while unwinding
                pass
        bpy.data.actions.remove(action)
        raise
    return action


def _assign(ad, action, slot) -> None:
    ad.action = action
    ad.action_slot = slot


def _restore_action(ad, old, old_slot, old_fake) -> None:
    ad.action = old
    if old is not None and old_slot is not None:
        ad.action_slot = old_slot
    if old_fake is not None:
        old_fake.use_fake_user = False


# --- the report -----------------------------------------------------------------------------------
def _report(src, tgt: _Target, pairs, feet, solve, fps) -> dict:
    si = {c: i for i, c in enumerate(src["order"])}
    r_all, h_all = solve["r_all"], solve["h_all"]
    hw = {c: tgt.to_world(h_all[:, b]) for c, b in pairs.items()}
    size = 100.0 / solve["k"]  # target world length -> fighter cm
    dir_err, twist_err = 0.0, 0.0
    ik_canon = {f"{s}{b}" for s in solve.get("ik_sides", []) for b in ("UpLeg", "Leg", "Foot")}
    ik_bones = []
    for s in solve.get("ik_sides", []):  # the IK'd chain: mapped thigh, shin, foot and the bones between
        top, foot = pairs[f"{s}UpLeg"], pairs[f"{s}Foot"]
        chain, x = [foot], tgt.parent[foot]
        while x >= 0 and x != top:
            chain.append(x)
            x = tgt.parent[x]
        ik_bones += [tgt.names[b] for b in [top, *reversed(chain)]]
    for c, k in solve["child"].items():
        if c in ik_canon:  # the foot lock moves these on purpose: not measured against the source
            continue
        w = tgt.robj @ r_all[:, pairs[c]]
        rel = w @ _T(solve["align"][c] @ tgt.rest_w_rot[pairs[c]])
        e = _T(solve["delta"][c]) @ rel
        if k is None or c in _YAW_ONLY:  # no direction alignment: the whole rotation
            twist_err = max(twist_err, float(np.degrees(_angle(e)).max()))
            continue
        a = _vec_angle(hw[k] - hw[c], src["head"][:, si[k]] - src["head"][:, si[c]])
        dir_err = max(dir_err, float(np.degrees(a).max()))
        twist_err = max(twist_err, float(np.degrees(np.abs(_twist(e, solve["d_s"][c]))).max()))
    pen = 0.0
    for model in feet["sides"].values():
        z = _foot_points_world(tgt, r_all, h_all, model)[:, :, 2].min(axis=1)
        pen = max(pen, float(np.max(solve["floor"] - z)))
    slide: dict[str, float | None] = {}
    for side in ("Left", "Right"):
        ft, toe = f"{side}Foot", f"{side}ToeBase"
        p_foot = src["plants"].get(ft)
        p_toe = src["plants"].get(toe)
        if ft not in pairs or (p_foot is None and p_toe is None):
            slide[side.lower()] = None
            continue
        n = src["n"]
        p_foot = p_foot if p_foot is not None else np.zeros(n, bool)
        p_toe = p_toe if p_toe is not None else np.zeros(n, bool)
        if toe not in pairs:
            p_foot, p_toe = p_foot | p_toe, np.zeros(n, bool)
        pos = np.where(p_foot[:, None], hw[ft], hw[toe] if toe in pairs else hw[ft])
        both = (p_foot[:-1] & p_foot[1:]) | (p_toe[:-1] & p_toe[1:] & ~p_foot[:-1] & ~p_foot[1:])
        if not both.any():
            slide[side.lower()] = None
            continue
        speed = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)[both] * fps * size
        slide[side.lower()] = round(float(np.median(speed)), 4)
    return {"hips_scale": round(float(solve["k"]), 6), "facing": solve["ft_name"],
            "facing_yaw_deg": round(math.degrees(solve["yaw"]), 4), "max_dir_err_deg": round(dir_err, 5),
            "max_twist_err_deg": round(twist_err, 5), "ik_bones": ik_bones, "foot_slide_cm_s": slide,
            "penetration_cm": round(max(pen, 0.0) * size, 4)}
