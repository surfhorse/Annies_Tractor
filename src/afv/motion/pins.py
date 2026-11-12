"""Pinned poses (``edits/pins.json``, read by the smooth stage; hands_face keeps pinned fingers).

Edit format (PM contract, round 4)::

    {"pins": [{"role": "A", "frame": 120, "radius": 8, "preset": "unity_humanoid",
               "root_motion": "world", "prefix": "",            # optional, from take.json's rig
               "bones": {"LeftForeArm": [x, y, z, w], ...}, "root": [x, y, z] | null}]}

``bones`` are **export-rig local rotations** (afv convention: Y-up, xyzw; the take package's ``rot``
arrays) on canonical rig bones, or prefixed with ``prefix`` (default: the preset's); ``root`` is the
Hips world position in afv world metres. ``preset`` names the export preset of the take the pose was
made on: it decides whether the rig has a ``root`` bone, and with ``root_motion`` (default: the
preset's) how the take's Hips rotation is meant (relative to the root bone under ``extracted``).
An optional ``root_rotation`` ([x, y, z, w], the take's root-bone rotation at that frame) fixes that
root exactly. Unknown keys are ignored.

Semantics
---------
**Target pose.** At the pin frame the take's rig pose is the solved pose with the pinned bones'
local rotations replaced (so an unpinned child keeps its rig-local rotation and turns with its
pinned parent, exactly as when posing in Blender). It is converted to source-skeleton (soma77)
local rotations by inverting ``afv.export.retarget`` for the pinned bones only: a pinned bone b
mapped to source joint s gets the source world rotation ``Q[s] = A_b T_b`` (``T_b`` = s's T-pose
world rotation, ``A_b`` = the bone's target world rotation; for a re-aimed bone, e.g. Neck whose
folded Neck2 joint bends the chain, ``A_b = W_b R0^-1`` with ``R0`` the re-aim of the current chain,
so the folded bend is kept). Only the source joints mapped to pinned bones change; folded joints
keep their local rotations, so every other bone's rig-local rotation stays the solved one. A rig
bone's local rotation depends only on the source locals between its parent's joint and its own,
which is why the conversion is exact whatever the rest of the pose does.

**Hips under extracted root motion.** A pin means the world rotation the user saw: the pinned Hips
local rotation is taken relative to the take's root bone at that frame (``root_rotation`` if given,
else the root-bone rotation of the solved motion before the pins), converted once through that fixed
root, without iterating. **What is kept is the Hips world rotation**: the exported root yaw is a
filtered heading of the result, which the pin itself shifts (0.8-24 deg for a 0.05-1.5 rad Hips yaw
on ufc_cross), so the take's Hips *local* rotation can differ from the pinned one by that shift.
QC measures Hips in the conversion frame. The root heading is computed with the export's default
root filters (``root_smooth_s`` 0.2 s, ``root_yaw_smooth_s`` 0.5 s); a shot exporting with other
filters gets a slightly different reference root unless the pin carries ``root_rotation``.

**Blending.** Each touched column (a joint's local tangent space, ``q'(t) = q(t) exp(d(t))``, or the
root translation) gets a displacement ``d(t)`` that is exactly the pin's correction at the pin frame,
exactly 0 wherever no pin *constraining that column* reaches (``|t - frame| >= radius``), and
otherwise minimises the summed squared second differences (a discrete cubic spline, velocity
continuous, meeting the untouched motion with ~zero velocity: C1); where pins of one column are
closer than half their radius a tension term (first differences, ``TENSION``) limits the swing
between them. Pins of one column whose radii overlap are solved together, so
every pin holds exactly; pins on the same column closer than ``CLOSE_FRAMES`` get a QC warning
(the motion between them swings hard). The displacement is applied after smoothing and before the
contacts are re-pinned: planted legs are re-pinned onto their plants afterwards (contacts win on
planted frames; such conflicts are reported per pin in ``qc.json``).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping as MappingT
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from afv.core.types import ActorMotion, Skeleton
from afv.export._math import rot_between
from afv.export.mapping import get_mapping
from afv.export.presets import PRESETS, get_preset
from afv.export.retarget import reaimed_bones, retarget
from afv.export.rig import ROOT_BONE, ExportRig, build_rig
from afv.export.root_motion import ROOT_MOTION_MODES, apply_root_motion
from afv.motion.fk import forward_kinematics
from afv.motion.rotations import IDENTITY, make_continuous, normalize, qconj, qmul, qrotate

DEFAULT_RADIUS = 8
# First-difference weight of the displacement spline, only where pins of one joint are close (closer
# than half their radius): there it cuts the overshoot of opposite pins a frame apart from 2.7x to 1.9x
# their correction; a lone pin keeps the pure cubic (step at the radius edge 0.073 x its correction,
# 0.087 with tension: +0.4 deg on a 28.6 deg pin of radius 8).
TENSION = 0.5
CLOSE_FRAMES = 4  # pins on the same joint closer than this get a QC warning
DEFAULT_PRESET = "unity_humanoid"  # ExportParams' default preset
PIN_TOL_DEG = 0.1  # QC: a pinned bone further than this from its pin at the pin frame is reported


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (degrees) between two rotations, accurate near 0 (atan2, not arccos of a dot)."""
    r = qmul(qconj(normalize(np.asarray(a, np.float64))), normalize(np.asarray(b, np.float64)))
    return float(np.degrees(2.0 * np.arctan2(np.linalg.norm(r[..., :3]), abs(float(r[..., 3])))))


class PinError(ValueError):
    """An invalid pin (the smooth stage turns it into a StageError)."""


@dataclass
class Pin:
    role: str
    frame: int  # shot-timeline frame number
    index: int  # index into the motion's frames
    radius: int
    preset: str
    root_motion: str
    bones: dict[str, np.ndarray]  # canonical rig bone -> local rotation xyzw
    root: np.ndarray | None = None  # Hips world position
    prefix: str = ""
    root_rotation: np.ndarray | None = None  # the take's root-bone rotation at the frame (optional)
    hips_parent: np.ndarray | None = None  # the rotation Hips was converted through (set by apply_pins)

    def describe(self) -> str:
        return f"pin {self.role}@{self.frame}"


def _quat(v: Any, where: str) -> np.ndarray:
    try:
        q = np.asarray(v, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise PinError(f"{where}: rotation must be 4 numbers [x, y, z, w], got {v!r}") from None
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-6:
        raise PinError(f"{where}: rotation must be 4 finite numbers [x, y, z, w] (not all 0), got {v!r}")
    return q / np.linalg.norm(q)


def _resolve_bone(name: str, canonical: set[str], prefix: str) -> str | None:
    if name in canonical:
        return name
    if prefix and name.startswith(prefix) and name[len(prefix) :] in canonical:
        return name[len(prefix) :]
    return None


def pin_list(edits: Any) -> list[dict]:
    if not edits:
        return []
    if isinstance(edits, MappingT):
        pins = edits.get("pins", [])
    else:
        pins = edits
    if not isinstance(pins, list) or not all(isinstance(p, MappingT) for p in pins):
        raise PinError('pins.json must be {"pins": [{"role", "frame", "bones", ...}, ...]}')
    return list(pins)


def parse_pins(edits: Any, frames: np.ndarray, roles: list[str], canonical: set[str]) -> dict[str, list[Pin]]:
    """Validate ``edits/pins.json`` against a motion (its shot-timeline ``frames``, fighters
    ``roles``) and the export rig's canonical bone names; {role: pins sorted by frame}."""
    frames = np.asarray(frames)
    out: dict[str, list[Pin]] = {}
    seen: set[tuple[str, int]] = set()
    for i, p in enumerate(pin_list(edits)):
        where = f"pins.json pin {i + 1}"
        role = p.get("role")
        if role not in roles:
            raise PinError(f"{where}: unknown fighter {role!r}; the shot has {sorted(roles)}")
        try:
            frame = int(p["frame"])
        except (KeyError, TypeError, ValueError):
            raise PinError(f"{where}: 'frame' must be a shot frame number, got {p.get('frame')!r}") from None
        idx = int(np.searchsorted(frames, frame))
        if len(frames) == 0 or idx >= len(frames) or int(frames[idx]) != frame:
            lo, hi = (int(frames[0]), int(frames[-1])) if len(frames) else (0, -1)
            raise PinError(f"{where}: frame {frame} is outside the shot (frames {lo}..{hi})")
        where = f"pins.json pin {i + 1} ({role}@{frame})"
        if (role, frame) in seen:
            raise PinError(f"{where}: two pins on the same fighter and frame; merge them into one")
        seen.add((role, frame))
        radius = p.get("radius", DEFAULT_RADIUS)
        if radius is None:
            radius = DEFAULT_RADIUS
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or int(radius) != radius
            or radius < 1
        ):
            raise PinError(f"{where}: 'radius' must be a whole number of frames >= 1, got {radius!r}")
        preset = p.get("preset") or DEFAULT_PRESET
        if preset not in PRESETS:
            raise PinError(f"{where}: unknown export preset {preset!r}; choose one of {sorted(PRESETS)}")
        pr = get_preset(preset)
        mode = p.get("root_motion") or pr.root_motion
        if mode not in ROOT_MOTION_MODES or mode not in pr.allowed_root_motion:
            raise PinError(f"{where}: preset {preset} does not support root motion {mode!r}")
        prefix = p.get("prefix")
        if prefix is None:
            prefix = pr.prefix
        if not isinstance(prefix, str):
            raise PinError(f"{where}: 'prefix' must be a string (the take's bone-name prefix)")
        root_rot = None
        if p.get("root_rotation") is not None:
            root_rot = _quat(p["root_rotation"], f"{where} root_rotation")
        bones_in = p.get("bones") or {}
        if not isinstance(bones_in, MappingT):
            raise PinError(f"{where}: 'bones' must be an object {{bone name: [x, y, z, w]}}")
        bones: dict[str, np.ndarray] = {}
        unknown = []
        for name, q in bones_in.items():
            if name == ROOT_BONE:
                raise PinError(
                    f"{where}: the '{ROOT_BONE}' bone is derived from Hips (root motion); pin Hips instead"
                )
            c = _resolve_bone(str(name), canonical, prefix)
            if c is None:
                unknown.append(str(name))
                continue
            if c in bones:
                raise PinError(f"{where}: bone {c} given twice")
            bones[c] = _quat(q, f"{where} bone {name}")
        if unknown:
            raise PinError(f"{where}: unknown export-rig bone(s) {unknown}; known: {sorted(canonical)}")
        root = p.get("root")
        root_a = None
        if root is not None:
            try:
                root_a = np.asarray(root, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                root_a = np.zeros(0)
            if root_a.shape != (3,) or not np.all(np.isfinite(root_a)):
                raise PinError(f"{where}: 'root' must be [x, y, z] (Hips world position, metres) or null")
        if not bones and root_a is None:
            raise PinError(f"{where}: pins nothing (no bones and no root)")
        out.setdefault(role, []).append(
            Pin(role, frame, idx, int(radius), preset, mode, bones, root_a, prefix, root_rot)
        )
    for r in out:
        out[r].sort(key=lambda q: q.index)
    return out


# ---------------------------------------------------------------------------
# Rig and conversion
# ---------------------------------------------------------------------------


def actor_rig(sk: Skeleton, actor: ActorMotion, preset: str, cache: dict | None = None) -> ExportRig:
    """The export rig the take was built on (rotations do not depend on the prefix or on whether
    finger bones are exported, so the rig is built with fingers and without a prefix)."""
    pr = get_preset(preset)
    key = (id(actor), pr.root_bone)
    if cache is not None and key in cache:
        return cache[key]
    rig = build_rig(
        sk, get_mapping(sk.name), rest_offsets=actor.rest_offsets, root_bone=pr.root_bone, fingers=True
    )
    if cache is not None:
        cache[key] = rig
    return rig


def canonical_bones(sk: Skeleton) -> set[str]:
    rig = build_rig(sk, get_mapping(sk.name), root_bone=False, fingers=True)
    return set(rig.canonical)


def rig_local(rig: ExportRig, rot: np.ndarray, root: np.ndarray, fps: float, mode: str) -> np.ndarray:
    """Export-rig local rotations [F,B,4] of source motion ``rot``/``root`` as the take/export
    would write them in root-motion ``mode`` (default root filters)."""
    m = retarget(rig, np.asarray(rot, np.float64), np.asarray(root, np.float64), fps)
    if mode != "world":
        m = apply_root_motion(m, mode)
    return m.rot


def pin_target(
    rig: ExportRig,
    q_f: np.ndarray,
    root_f: np.ndarray,
    pin: Pin,
    fps: float,
    hips_parent: np.ndarray | None = None,
) -> tuple[dict[int, np.ndarray], np.ndarray | None]:
    """Source local rotations {joint: xyzw} for the joints the pin touches, and the target root
    translation (None = not pinned), given the current source pose ``q_f`` [J,4] / ``root_f`` [3]
    at the pin frame. ``hips_parent``: world rotation of Hips' rig parent (the root bone's yaw in
    ``extracted`` mode; None = identity)."""
    src = rig.source
    cur = retarget(rig, q_f[None], root_f[None], fps).rot[0]  # world mode: root bone identity
    hips = rig.hips
    target_local = cur.copy()
    for name, q in pin.bones.items():
        b = rig.canonical.index(name)
        if b == hips and hips_parent is not None:
            q = qmul(hips_parent, q)  # take Hips is relative to the root bone; world mode: to the world
        target_local[b] = q
    w_t = np.empty_like(target_local)  # target rig world rotations
    for b, p in enumerate(rig.parents):
        w_t[b] = target_local[b] if p < 0 else qmul(w_t[p], target_local[b])

    pos, wrot = forward_kinematics(src, q_f[None], root_f[None])
    pos, wrot = pos[0], wrot[0]
    reaimed = set(reaimed_bones(rig))
    want: dict[int, np.ndarray] = {}  # source joint -> target world rotation
    for name in pin.bones:
        b = rig.canonical.index(name)
        s = src.index(rig.source_joint[b])  # type: ignore[arg-type]
        a = w_t[b]
        if b in reaimed:
            k = rig.children(b)[0]
            sk_ = src.index(rig.source_joint[k])  # type: ignore[arg-type]
            c_local = qrotate(qconj(wrot[s]), pos[sk_] - pos[s])
            u = qrotate(rig.source_tpose_rot[b], c_local)
            r0 = rot_between(rig.offsets[k], u)
            a = qmul(a, qconj(r0))
        want[s] = qmul(a, rig.source_tpose_rot[b])

    rest = src.rest_rotations if src.rest_rotations is not None else np.tile(IDENTITY, (src.num_joints, 1))
    loc = np.array(q_f, dtype=np.float64, copy=True)
    wnew = np.empty_like(wrot)
    out: dict[int, np.ndarray] = {}
    for j, p in enumerate(src.parents):
        pw = IDENTITY if p < 0 else wnew[p]
        if j in want:
            loc[j] = qmul(qconj(rest[j]), qmul(qconj(pw), want[j]))
            out[j] = normalize(loc[j])
        wnew[j] = qmul(pw, qmul(rest[j], loc[j]))
    root_t = None
    if pin.root is not None:
        s_h = src.index(rig.source_joint[hips])  # type: ignore[arg-type]
        p_new = forward_kinematics(src, loc[None], root_f[None])[0][0, s_h]
        root_t = np.asarray(root_f, np.float64) + (pin.root - p_new)
    return out, root_t


# ---------------------------------------------------------------------------
# Displacement
# ---------------------------------------------------------------------------


def _qlog(q: np.ndarray) -> np.ndarray:
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    v = q[..., :3]
    s = np.linalg.norm(v, axis=-1)
    ang = 2.0 * np.arctan2(s, q[..., 3])
    scale = np.where(s > 1e-12, ang / np.maximum(s, 1e-12), 2.0)
    return v * scale[..., None]


def _qexp(r: np.ndarray) -> np.ndarray:
    ang = np.linalg.norm(r, axis=-1)
    scale = np.where(ang > 1e-12, np.sin(0.5 * ang) / np.maximum(ang, 1e-12), 0.5)
    return np.concatenate([r * scale[..., None], np.cos(0.5 * ang)[..., None]], axis=-1)


def clusters(windows: list[tuple[int, int]], n_frames: int) -> list[tuple[int, int, list[int]]]:
    """Groups of pin windows ``(frame index, radius)`` whose open windows overlap or touch: (first,
    last free frame, window indices). Free frames are those strictly inside some window."""
    out: list[tuple[int, int, list[int]]] = []
    for i, (f, r) in sorted(enumerate(windows), key=lambda x: x[1][0]):
        lo = max(0, f - r + 1)
        hi = min(n_frames - 1, f + r - 1)
        if out and lo <= out[-1][1] + 1:
            a, b, ids = out[-1]
            out[-1] = (a, max(b, hi), [*ids, i])
        else:
            out.append((lo, hi, [i]))
    return out


def spline_displacement(
    n_frames: int, lo: int, hi: int, fixed: dict[int, np.ndarray], tension: float = TENSION
) -> np.ndarray:
    """Displacement on frames [lo, hi] (0 just outside, within the clip) equal to ``fixed[t]`` [D]
    at the given frames, minimising squared second differences + ``tension`` x squared first
    differences. Returns [hi - lo + 1, D]."""
    m = hi - lo + 1
    dim = len(next(iter(fixed.values())))
    rows = []
    for c in range(lo - 1, hi + 2):  # second differences centred lo-1 .. hi+1 (3 frames in the clip)
        if c - 1 < 0 or c + 1 > n_frames - 1:
            continue
        r = np.zeros(m)
        for t, w in ((c - 1, 1.0), (c, -2.0), (c + 1, 1.0)):
            if lo <= t <= hi:
                r[t - lo] = w
        if np.any(r):
            rows.append(r)
    if tension > 0:
        st = np.sqrt(tension)
        for c in range(lo - 1, hi + 1):  # first differences d(c+1) - d(c)
            if c < 0 or c + 1 > n_frames - 1:
                continue
            r = np.zeros(m)
            for t, w in ((c, -st), (c + 1, st)):
                if lo <= t <= hi:
                    r[t - lo] = w
            if np.any(r):
                rows.append(r)
    d2 = np.array(rows) if rows else np.zeros((0, m))
    fix_idx = np.array(sorted(fixed)) - lo
    free = np.setdiff1d(np.arange(m), fix_idx)
    x = np.zeros((m, dim))
    x[fix_idx] = np.array([fixed[t + lo] for t in fix_idx])
    if len(free):
        a = d2[:, free]
        rhs = -d2[:, fix_idx] @ x[fix_idx]
        h = a.T @ a + 1e-9 * np.eye(len(free))
        x[free] = np.linalg.solve(h, a.T @ rhs)
    return x


@dataclass
class PinReport:
    role: str
    frame: int
    bones: list[str]
    root: bool
    radius: int
    max_error_deg: float = 0.0  # after the whole stage (set by the caller)
    errors_deg: dict[str, float] = field(default_factory=dict)
    root_error_m: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        bad = sorted(b for b, e in self.errors_deg.items() if e > PIN_TOL_DEG)
        return {
            "role": self.role,
            "frame": self.frame,
            "radius": self.radius,
            "bones": self.bones,
            "root": self.root,
            "max_error_deg": round(self.max_error_deg, 4),
            "root_error_m": None if self.root_error_m is None else round(self.root_error_m, 5),
            "conflicts": bad,
            "warnings": list(self.warnings),
        }


def root_bone_rotation(
    sk: Skeleton, actor: ActorMotion, pin: Pin, rot: np.ndarray, root: np.ndarray, fps: float, cache: dict
) -> np.ndarray:
    """The take's root-bone rotation at the pin frame for motion ``rot``/``root`` (extracted mode)."""
    rig = actor_rig(sk, actor, pin.preset, cache)
    m = apply_root_motion(retarget(rig, rot, root, fps), "extracted")
    return np.asarray(m.rot[pin.index, rig.root], dtype=np.float64)  # type: ignore[index]


def hips_parent(
    sk: Skeleton, actor: ActorMotion, pin: Pin, rot: np.ndarray, root: np.ndarray, fps: float, cache: dict
) -> np.ndarray | None:
    """The rotation a pinned Hips is relative to: None (the world) unless extracted root motion;
    then the pin's ``root_rotation`` or the root bone of ``rot``/``root`` (the motion before pins)."""
    if pin.root_motion != "extracted" or "Hips" not in pin.bones:
        return None
    if pin.root_rotation is not None:
        return pin.root_rotation
    return root_bone_rotation(sk, actor, pin, rot, root, fps, cache)


def apply_pins(
    sk: Skeleton,
    actor: ActorMotion,
    rot: np.ndarray,
    root: np.ndarray,
    pins: list[Pin],
    fps: float,
    rig_cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[PinReport]]:
    """Displace ``rot`` [F,J,4] / ``root`` [F,3] (the smoothed motion) so it passes through every
    pin (module docstring). Returns the new rotations, root and one report per pin; sets each
    pin's ``hips_parent``."""
    rot = make_continuous(np.asarray(rot, np.float64))
    root = np.asarray(root, np.float64)
    n = rot.shape[0]
    if not pins:
        return rot, root, []
    rig_cache = {} if rig_cache is None else rig_cache
    reports = [PinReport(p.role, p.frame, sorted(p.bones), p.root is not None, p.radius) for p in pins]
    targets = []
    for p in pins:
        p.hips_parent = hips_parent(sk, actor, p, rot, root, fps, rig_cache)
        rig = actor_rig(sk, actor, p.preset, rig_cache)
        targets.append(pin_target(rig, rot[p.index], root[p.index], p, fps, p.hips_parent))
    out_rot, out_root, close = _displace(rot, root, pins, targets, n)
    names = {j: sk.joint_names[j] for j in range(sk.num_joints)}
    for i, k, col in close:
        what = "the root" if col < 0 else names[col]
        gap = abs(pins[k].index - pins[i].index)
        msg = f"pins at frames {pins[i].frame} and {pins[k].frame} both move {what}, {gap} frame(s) apart"
        reports[i].warnings.append(msg)
        reports[k].warnings.append(msg)
    return make_continuous(out_rot), out_root, reports


def _displace(
    rot: np.ndarray,
    root: np.ndarray,
    pins: list[Pin],
    targets: list[tuple[dict[int, np.ndarray], np.ndarray | None]],
    n: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    """Apply the per-column displacement splines; also returns (pin, pin, column) for pins on the
    same column closer than ``CLOSE_FRAMES`` (column -1 = root)."""
    rot = rot.copy()
    root = root.copy()
    n_j = rot.shape[1]
    # per column (joint, or n_j = root): {pin index: correction at its frame}
    cols: dict[int, dict[int, np.ndarray]] = {}
    for i, p in enumerate(pins):
        joints, root_t = targets[i]
        f = p.index
        for j, q in joints.items():
            cols.setdefault(j, {})[i] = _qlog(qmul(qconj(rot[f, j]), q))
        if root_t is not None:
            cols.setdefault(n_j, {})[i] = root_t - root[f]
    close: list[tuple[int, int, int]] = []
    # columns constrained by the same pins share their solves
    groups: dict[tuple[int, ...], list[int]] = {}
    for c, per_pin in cols.items():
        groups.setdefault(tuple(sorted(per_pin)), []).append(c)
    for ids, cs in groups.items():
        ordered = sorted(ids, key=lambda i: pins[i].index)
        for x, y in itertools.pairwise(ordered):
            if pins[y].index - pins[x].index < CLOSE_FRAMES:
                close += [(x, y, -1 if c == n_j else c) for c in cs]
        windows = [(pins[i].index, pins[i].radius) for i in ids]
        for lo, hi, members in clusters(windows, n):
            pin_ids = [ids[m] for m in members]
            fixed = {pins[i].index: np.concatenate([cols[c][i] for c in cs]) for i in pin_ids}
            d = spline_displacement(n, lo, hi, fixed, _cluster_tension([pins[i] for i in pin_ids]))
            for k, c in enumerate(cs):
                dk = d[:, 3 * k : 3 * k + 3]
                if c == n_j:
                    root[lo : hi + 1] += dk
                else:
                    rot[lo : hi + 1, c] = qmul(rot[lo : hi + 1, c], _qexp(dk))
                    for i in pin_ids:  # exact at the pin frame (no exp/log round-off)
                        rot[pins[i].index, c] = normalize(targets[i][0][c])
    return rot, root, close


def _cluster_tension(cluster: list[Pin]) -> float:
    """``TENSION`` for a cluster with two pins closer than half the smaller radius, else 0."""
    ps = sorted(cluster, key=lambda q: q.index)
    for x, y in itertools.pairwise(ps):
        if y.index - x.index < 0.5 * min(x.radius, y.radius):
            return TENSION
    return 0.0


def segment_joints(rig: ExportRig, bone: str) -> list[int]:
    """Source joints whose local rotations make up ``bone``'s rig-local rotation: its mapped joint
    and the folded joints between it and its rig parent's joint."""
    src = rig.source
    b = rig.canonical.index(bone)
    j = src.index(rig.source_joint[b])  # type: ignore[arg-type]
    p = rig.parents[b]
    stop = src.index(rig.source_joint[p]) if p >= 0 and rig.source_joint[p] is not None else -1
    out = []
    while j >= 0 and j != stop:
        out.append(j)
        j = src.parents[j]
    return out


def pin_errors(
    sk: Skeleton,
    actor: ActorMotion,
    rot: np.ndarray,
    root: np.ndarray,
    pins: list[Pin],
    reports: list[PinReport],
    fps: float,
    rig_cache: dict | None = None,
) -> None:
    """Fill each report with the pinned bones' angle error (degrees) in the export rig at the pin
    frame, and the Hips position error, for the final motion ``rot``/``root``. Hips under extracted
    root motion is measured in the frame it was converted through (``pin.hips_parent``: the world
    rotation the user saw); without one (not converted here) it is left out (``skip_hips``)."""
    rig_cache = {} if rig_cache is None else rig_cache
    for p, rep in zip(pins, reports, strict=True):
        rig = actor_rig(sk, actor, p.preset, rig_cache)
        cur = retarget(rig, rot[p.index : p.index + 1], root[p.index : p.index + 1], fps).rot[0]
        errs = {}
        for name, q in p.bones.items():
            c = cur[rig.canonical.index(name)]  # world mode: Hips local = its world rotation
            if name == "Hips" and p.root_motion == "extracted":
                if p.hips_parent is None:
                    continue
                q = qmul(p.hips_parent, q)
            errs[name] = quat_angle_deg(c, q)
        rep.errors_deg = errs
        rep.max_error_deg = max(errs.values(), default=0.0)
        if p.root is not None:
            s_h = rig.source.index(rig.source_joint[rig.hips])  # type: ignore[arg-type]
            ph = forward_kinematics(rig.source, rot[p.index][None], root[p.index][None])[0][0, s_h]
            rep.root_error_m = float(np.linalg.norm(ph - p.root))


def stage_pins(ctx: Any, bm: Any, stage: str) -> dict[str, list[Pin]]:
    """Validated pins of ``edits/pins.json`` for a stage ({role: [Pin]}, frame indices into ``bm``);
    invalid pins raise ``StageError``."""
    from afv.core.skeletons import get_skeleton
    from afv.core.stage import StageError

    edit = ctx.edits.get("pins", ctx.edits.get("pins.json"))
    if not edit:
        return {}
    try:
        canonical = canonical_bones(get_skeleton(bm.skeleton))
    except (KeyError, ValueError) as e:
        raise StageError(f"{stage}: pins need the export rig of skeleton {bm.skeleton!r}: {e}") from e
    try:
        return parse_pins(edit, bm.frames, list(bm.actors), canonical)
    except PinError as e:
        raise StageError(f"{stage}: {e}") from e
