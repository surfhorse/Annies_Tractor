"""``hands_face`` stage (minimal, round 2): pass GEM-X's hands and face through, or pose a fist preset.

There is no dedicated hand/face model yet (spec: fingers and face are later work). This stage keeps
the pipeline order intact (``... smooth, hands_face, export``) and offers the spec's fist preset:

- ``hands="gemx"`` (default): GEM-X's own finger rotations, unchanged;
- ``hands="fist"``: a closed-fist pose on every finger joint, blended over GEM-X's with ``fist_weight``
  (1 = pure preset). Gloved boxers are always fisted; GEM-X's fingers on broadcast footage are guesses;
- ``face="neutral"`` resets the jaw and eyes to rest (``"gemx"`` keeps them).

**Hands off** (``edits/hands.json``, ``{"spans": [{"role", "start", "end", "side": "left"|"right"|
"both"}]}``, shot-timeline frames, ``end`` exclusive): where the user marked the hand data as bad,
that side's finger joints are blended to a *relaxed* hand pose (a light natural curl, the same
curl axes as the fist, :data:`RELAXED_DEG`): full weight on ``[start, end)``, easing in and out over
``hands_off_ease_frames`` frames outside the span (smootherstep, so the fingers' motion stays C1).
It is applied after the fist preset, so a marked span is relaxed even on a fisted fighter.

**Pins** (``edits/pins.json``, see ``afv.motion.pins``): the joints that make up a pinned bone
(its mapped joint and the folded joints before it, e.g. a metacarpal) keep the smooth result at the
pin frame, where smooth made the motion pass through the pin; the fist / hands-off / neutral-face
change is blended back in over the pin's radius (smootherstep of the distance). The final pin
error is reported in ``qc.json`` (``pins``).

Finger joints are found by name (``<Left|Right>Hand<Thumb|Index|Middle|Ring|Pinky><n>``, the soma77
layout); a skeleton without them passes through with a QC note. The fist curls each finger segment
toward the palm about the axis ``segment x palm_normal`` (afv rest poses are T-poses with palms down,
so ``palm_normal = -Y`` in the rest-pose world; it is brought into each joint's accumulated rest frame,
so skeletons with rest rotations work too). The thumb is then wrapped across the index and middle
fingers: a small, deterministic IK (thumb metacarpal 3 dof + two flexions, regularised) puts its tip in
front of their middle-phalanx (PIP) joints. Angles are a plausible preset, not a measurement.

QC (``qc.json``): per fighter, the magnitude of GEM-X's metacarpal rotations (``<Finger>1`` joints,
which the export rig folds into the fingers) and the knuckle shift that folding causes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import least_squares

from afv.core.skeletons import get_skeleton
from afv.core.stage import Ctx, StageError, register
from afv.core.types import ActorMotion, BodyMotion, LicenceRecord, Result, Skeleton
from afv.motion.fk import forward_kinematics
from afv.motion.pins import Pin, PinReport, actor_rig, pin_errors, segment_joints, stage_pins
from afv.motion.ranges import HANDS_BLEND, HANDS_MARGIN, blend_into_previous, range_plan, range_summary
from afv.motion.rotations import IDENTITY, from_rotvec, make_continuous, qconj, qrotate, slerp, to_rotvec

FINGER_RE = re.compile(r"^(Left|Right)Hand(Thumb|Index|Middle|Ring|Pinky)(\d)$")
FACE_JOINTS: tuple[str, ...] = ("Jaw", "LeftEye", "RightEye")
PALM_NORMAL = np.array([0.0, -1.0, 0.0])  # rest-pose world: T-pose, palms down
# Fist curl per finger segment (degrees), keyed by the joint number in the finger chain. The thumb
# angles are only the fallback when the thumb IK cannot run (missing joints).
FIST_DEG: dict[str, dict[int, float]] = {
    "Thumb": {1: 15.0, 2: 35.0, 3: 45.0},
    "Index": {1: 3.0, 2: 85.0, 3: 100.0, 4: 60.0},
    "Middle": {1: 3.0, 2: 88.0, 3: 100.0, 4: 60.0},
    "Ring": {1: 5.0, 2: 90.0, 3: 100.0, 4: 60.0},
    "Pinky": {1: 8.0, 2: 92.0, 3: 100.0, 4: 60.0},
}


# Relaxed hand (hands off): a light curl per finger segment (degrees), same numbering as FIST_DEG.
RELAXED_DEG: dict[str, dict[int, float]] = {
    "Thumb": {1: 8.0, 2: 10.0, 3: 12.0},
    "Index": {1: 2.0, 2: 12.0, 3: 20.0, 4: 10.0},
    "Middle": {1: 2.0, 2: 16.0, 3: 24.0, 4: 12.0},
    "Ring": {1: 3.0, 2: 20.0, 3: 28.0, 4: 14.0},
    "Pinky": {1: 4.0, 2: 24.0, 3: 32.0, 4: 16.0},
}
SIDES = {"left": ("Left",), "right": ("Right",), "both": ("Left", "Right")}


class HandsFaceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hands: Literal["gemx", "fist"] = "gemx"
    fist_weight: float = Field(1.0, ge=0.0, le=1.0)
    face: Literal["gemx", "neutral"] = "gemx"
    hands_off_ease_frames: int = Field(6, ge=0, le=60)  # hands-off spans ease in/out over this many frames


def finger_joints(sk: Skeleton) -> dict[str, tuple[str, str, int]]:
    """{joint name: (side, finger, number)} for every finger joint that rotates a segment."""
    out = {}
    for i, n in enumerate(sk.joint_names):
        m = FINGER_RE.match(n)
        if m and sk.children(i):
            out[n] = (m.group(1), m.group(2), int(m.group(3)))
    return out


THUMB_TIP_AHEAD_M = 0.012  # thumb tip this far in front of the index/middle PIP joints
THUMB_REG = 0.05  # regularisation of the thumb IK (per radian, vs 10 per metre of tip miss)


def _rest_frames(sk: Skeleton) -> tuple[np.ndarray, np.ndarray]:
    """Rest-pose world positions [J,3] and the frame each joint's LOCAL rotation acts in [J,4]
    (parent world rotation x the joint's rest rotation, i.e. its world rotation at rest)."""
    pos, wrot = forward_kinematics(sk, np.tile(IDENTITY, (1, sk.num_joints, 1)))
    return pos[0], wrot[0]


def _curl_axis_local(sk: Skeleton, j: int, frame: np.ndarray) -> np.ndarray | None:
    """Unit curl axis of joint ``j``'s segment in its local frame: segment x palm normal."""
    d = sk.rest_offsets[sk.children(j)[0]]  # the segment, in j's own (local) rest frame
    axis = np.cross(d, qrotate(qconj(frame), PALM_NORMAL))
    norm = np.linalg.norm(axis)
    return None if norm < 1e-9 else axis / norm


_FIST_CACHE: dict[tuple, dict[str, np.ndarray]] = {}


def fist_pose(sk: Skeleton) -> dict[str, np.ndarray]:
    """Local rotation (xyzw) per finger joint for a closed fist on ``sk``'s rest pose (cached per
    skeleton: the thumb IK takes about a second)."""
    rr = b"" if sk.rest_rotations is None else np.asarray(sk.rest_rotations).tobytes()
    key = (sk.name, sk.joint_names, sk.parents, np.asarray(sk.rest_offsets).tobytes(), rr)
    if key not in _FIST_CACHE:
        _FIST_CACHE[key] = _fist_pose(sk)
    return {k: v.copy() for k, v in _FIST_CACHE[key].items()}


def _fist_pose(sk: Skeleton) -> dict[str, np.ndarray]:
    _, frames = _rest_frames(sk)
    pose: dict[str, np.ndarray] = {}
    for name, (_side, finger, num) in finger_joints(sk).items():
        deg = FIST_DEG.get(finger, {}).get(num)
        j = sk.index(name)
        axis = _curl_axis_local(sk, j, frames[j])
        if deg is None or axis is None:
            continue
        pose[name] = from_rotvec(axis * np.deg2rad(deg))
    for side in ("Left", "Right"):
        pose.update(_thumb_wrap(sk, pose, side, frames))
    return pose


def relaxed_pose(sk: Skeleton) -> dict[str, np.ndarray]:
    """Local rotation (xyzw) per finger joint for a relaxed hand on ``sk``'s rest pose."""
    _, frames = _rest_frames(sk)
    pose: dict[str, np.ndarray] = {}
    for name, (_side, finger, num) in finger_joints(sk).items():
        deg = RELAXED_DEG.get(finger, {}).get(num)
        j = sk.index(name)
        axis = _curl_axis_local(sk, j, frames[j])
        if deg is None or axis is None:
            continue
        pose[name] = from_rotvec(axis * np.deg2rad(deg))
    return pose


def hands_off_spans(
    edits: object, frames: np.ndarray, roles: list[str]
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Validate ``edits/hands.json``: {(role, "Left"|"Right"): [(i0, i1), ...]} as frame indices
    (spans clipped to the shot; a span entirely outside it is an error)."""
    if not edits:
        return {}
    spans = edits.get("spans", []) if isinstance(edits, dict) else edits
    if not isinstance(spans, list) or not all(isinstance(x, dict) for x in spans):
        raise ValueError('hands.json must be {"spans": [{"role", "start", "end", "side"}, ...]}')
    frames = np.asarray(frames)
    out: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for k, sp in enumerate(spans):
        where = f"hands.json span {k + 1}"
        role = sp.get("role")
        if role not in roles:
            raise ValueError(f"{where}: unknown fighter {role!r}; the shot has {sorted(roles)}")
        side = str(sp.get("side", "both")).lower()
        if side not in SIDES:
            raise ValueError(f"{where}: side must be 'left', 'right' or 'both', got {sp.get('side')!r}")
        try:
            start, end = int(sp["start"]), int(sp["end"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"{where}: 'start' and 'end' must be shot frame numbers (end exclusive)"
            ) from None
        if end <= start:
            raise ValueError(f"{where}: empty span {start}:{end} (end is exclusive)")
        i0 = int(np.searchsorted(frames, start, side="left"))
        i1 = int(np.searchsorted(frames, end, side="left"))
        if i1 <= i0:
            lo, hi = (int(frames[0]), int(frames[-1])) if len(frames) else (0, -1)
            raise ValueError(f"{where}: span {start}:{end} is outside the shot (frames {lo}..{hi})")
        for s_ in SIDES[side]:
            out.setdefault((role, s_), []).append((i0, i1))
    return out


def span_weights(n_frames: int, spans: list[tuple[int, int]], ease: int) -> np.ndarray:
    """[F] blend weight: 1 inside every span, smootherstep ease over ``ease`` frames outside."""
    w = np.zeros(n_frames)
    t = np.arange(n_frames)
    for i0, i1 in spans:
        dist = np.where(t < i0, i0 - t, np.where(t >= i1, t - i1 + 1, 0)).astype(np.float64)
        u = np.clip(1.0 - dist / (ease + 1.0), 0.0, 1.0)
        w = np.maximum(w, u * u * u * (u * (6.0 * u - 15.0) + 10.0))
    return w


def _thumb_wrap(sk: Skeleton, pose: dict[str, np.ndarray], side: str, frames: np.ndarray) -> dict:
    """Thumb rotations that put its tip across the curled index and middle fingers (in front of their
    PIP joints), or {} (keep the fallback curls) when the joints are missing."""
    names = [f"{side}Hand{n}" for n in ("Thumb1", "Thumb2", "Thumb3", "ThumbEnd", "Index3", "Middle3")]
    if not all(n in sk.joint_names for n in names) or f"{side}Hand" not in sk.joint_names:
        return {}
    t1, t2, t3, tip, i3, m3 = (sk.index(n) for n in names)
    hand = sk.index(f"{side}Hand")
    rot = np.tile(IDENTITY, (1, sk.num_joints, 1))
    for n, q in pose.items():
        if "Thumb" not in n:
            rot[0, sk.index(n)] = q
    pos = forward_kinematics(sk, rot)[0][0]
    ahead = (
        pos[sk.index(f"{side}HandMiddle2")] - pos[hand]
        if f"{side}HandMiddle2" in sk.joint_names
        else (pos[i3] - pos[hand])
    )
    ahead = ahead - (ahead @ PALM_NORMAL) * PALM_NORMAL
    ahead /= max(np.linalg.norm(ahead), 1e-9)
    target = 0.5 * (pos[i3] + pos[m3]) + THUMB_TIP_AHEAD_M * ahead
    axes = [_curl_axis_local(sk, j, frames[j]) for j in (t2, t3)]
    if any(a is None for a in axes):
        return {}
    # Thumb1's rotation is solved as a WORLD-axis rotation vector (frame independent, mirrors cleanly)
    f1 = frames[t1]

    def local(x: np.ndarray) -> dict[int, np.ndarray]:
        return {
            t1: from_rotvec(qrotate(qconj(f1), x[:3])),
            t2: from_rotvec(axes[0] * x[3]),  # type: ignore[operator]
            t3: from_rotvec(axes[1] * x[4]),  # type: ignore[operator]
        }

    def residual(x: np.ndarray) -> np.ndarray:
        r = rot.copy()
        for j, q in local(x).items():
            r[0, j] = q
        p = forward_kinematics(sk, r)[0][0]
        return np.r_[(p[tip] - target) * 10.0, THUMB_REG * x]

    lateral = 1.0 if side == "Left" else -1.0  # mirror the start (rotvec of a reflection in x: x, -y, -z)
    x0 = np.array([0.1, 0.1 * lateral, 0.1 * lateral, 0.5, 0.5])
    sol = least_squares(residual, x0, bounds=([-2, -2, -2, 0, 0], [2, 2, 2, 1.6, 1.6]), xtol=1e-10)
    return {sk.joint_names[j]: q for j, q in local(sol.x).items()}


def _angle_deg(q: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.linalg.norm(to_rotvec(q), axis=-1))


def metacarpal_stats(sk: Skeleton, actor: ActorMotion) -> dict[str, object]:
    """GEM-X's metacarpal (``<Finger>1``, not the thumb) rotation magnitudes and the knuckle shift
    if the export folds them (|R v - v| for the knuckle offset v)."""
    joints = [n for n, (_s, f, k) in finger_joints(sk).items() if k == 1 and f != "Thumb"]
    if not joints:
        return {}
    offsets = actor.rest_offsets if actor.rest_offsets is not None else sk.rest_offsets
    per: dict[str, dict[str, float]] = {}
    all_deg, all_shift = [], []
    for n in joints:
        j = sk.index(n)
        q = np.asarray(actor.rot[:, j], np.float64)
        deg = _angle_deg(q)
        v = offsets[sk.children(j)[0]]
        shift = np.linalg.norm(qrotate(q, v) - v, axis=-1)
        all_deg.append(deg)
        all_shift.append(shift)
        per[n] = {
            "median_deg": round(float(np.median(deg)), 2),
            "p95_deg": round(float(np.percentile(deg, 95)), 2),
            "max_deg": round(float(deg.max()), 2),
            "knuckle_shift_max_mm": round(float(shift.max() * 1000), 2),
        }
    d, s = np.concatenate(all_deg), np.concatenate(all_shift)
    return {
        "median_deg": round(float(np.median(d)), 2),
        "p95_deg": round(float(np.percentile(d, 95)), 2),
        "max_deg": round(float(d.max()), 2),
        "knuckle_shift_p95_mm": round(float(np.percentile(s, 95) * 1000), 2),
        "knuckle_shift_max_mm": round(float(s.max() * 1000), 2),
        "joints": per,
    }


def pin_keep_weights(bm: BodyMotion, pins: dict[str, list[Pin]]) -> dict[str, np.ndarray]:
    """{role: [F,J] weight of the input (smooth) pose}: 1 at a pin frame on the joints of its pinned
    bones, easing to 0 at ``radius`` frames (smootherstep), 0 elsewhere."""
    sk = get_skeleton(bm.skeleton)
    out: dict[str, np.ndarray] = {}
    cache: dict = {}
    t = np.arange(bm.num_frames)
    for role, ps in pins.items():
        a = bm.actors[role]
        keep = np.zeros((bm.num_frames, sk.num_joints))
        for pin in ps:
            rig = actor_rig(sk, a, pin.preset, cache)
            u = np.clip(1.0 - np.abs(t - pin.index) / float(pin.radius), 0.0, 1.0)
            w = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
            for b in pin.bones:
                for j in segment_joints(rig, b):
                    keep[:, j] = np.maximum(keep[:, j], w)
        out[role] = keep
    return out


def apply_hands_face(
    bm: BodyMotion,
    p: HandsFaceParams,
    hands_edit: object = None,
    pins: dict[str, list[Pin]] | None = None,
) -> tuple[BodyMotion, dict[str, object]]:
    sk = get_skeleton(bm.skeleton)
    keep = pin_keep_weights(bm, pins) if pins else {}
    fingers = finger_joints(sk)
    notes: list[str] = []
    off = hands_off_spans(hands_edit, bm.frames, list(bm.actors))
    relaxed = relaxed_pose(sk) if off else {}
    if off and not relaxed:
        notes.append(f"skeleton {sk.name} has no finger joints: hands-off spans ignored")
    fist = fist_pose(sk) if p.hands == "fist" else {}
    if p.hands == "fist" and not fist:
        notes.append(f"skeleton {sk.name} has no finger joints: hands passed through")
    face = [sk.index(n) for n in FACE_JOINTS if n in sk.joint_names] if p.face == "neutral" else []
    actors: dict[str, ActorMotion] = {}
    qc: dict[str, object] = {"hands": p.hands, "face": p.face, "fighters": {}}
    for role, a in bm.actors.items():
        qc["fighters"][role] = {"gemx_metacarpals": metacarpal_stats(sk, a)}  # type: ignore[index]
        rot = np.array(a.rot, dtype=np.float64)
        for name, q in fist.items():
            j = sk.index(name)
            target = np.broadcast_to(q, rot[:, j].shape)
            rot[:, j] = slerp(rot[:, j], target, p.fist_weight) if p.fist_weight < 1.0 else target
        for j in face:
            rot[:, j] = IDENTITY
        n_off = 0
        for side in ("Left", "Right"):
            spans = off.get((role, side))
            if not spans or not relaxed:
                continue
            w = span_weights(rot.shape[0], spans, p.hands_off_ease_frames)
            n_off += int((w >= 1.0).sum())
            on = w > 0.0  # frames outside the eased spans are left exactly as they are
            for name, q in relaxed.items():
                if fingers[name][0] != side:
                    continue
                j = sk.index(name)
                rot[on, j] = slerp(rot[on, j], np.broadcast_to(q, rot[on, j].shape), w[on, None])
        qc["fighters"][role]["hands_off_frames"] = n_off  # type: ignore[index]
        kw = keep.get(role)
        if kw is not None and (fist or face or n_off):  # pinned joints keep smooth's pose at their pins
            src = np.asarray(a.rot, dtype=np.float64)
            on = (kw > 0.0) & np.any(rot != src, axis=-1)
            rot[on] = slerp(rot[on], src[on], kw[on][:, None])
            full = (kw >= 1.0) & on
            rot[full] = src[full]
        if fist or face or n_off:
            rot = make_continuous(rot)
        actors[role] = ActorMotion(
            rot=rot.astype(np.float32), root=a.root, conf=a.conf, contact=a.contact,
            rest_offsets=a.rest_offsets, root_prior=a.root_prior,
        )  # fmt: skip
    qc["finger_joints"] = len(fingers)
    qc["notes"] = notes
    out = BodyMotion(
        frames=bm.frames, timestamps=bm.timestamps, skeleton=bm.skeleton, fps=bm.fps,
        contact_channels=list(bm.contact_channels), actors=actors, world_shift=list(bm.world_shift),
    )  # fmt: skip
    return out, qc


@register("default")
class HandsFaceStage:
    """Pass-through / fist preset for hands, pass-through / neutral face (no model yet)."""

    name: ClassVar[str] = "hands_face"
    # 2: hands-off spans (edits/hands.json), pinned fingers kept (edits/pins.json); unchanged without them
    version: ClassVar[str] = "2"
    env: ClassVar[str] = "cpu"
    inputs: ClassVar[tuple[str, ...]] = ("smooth",)
    edits: ClassVar[tuple[str, ...]] = ("hands", "pins")
    licences: ClassVar[tuple[LicenceRecord, ...]] = ()
    Params: ClassVar[type[BaseModel]] = HandsFaceParams
    weight_hash: ClassVar[str] = ""
    supports_range: ClassVar[bool] = True

    def run(self, ctx: Ctx, inputs: dict[str, Result], params: BaseModel) -> Result:
        bm = inputs.get("smooth")
        if not isinstance(bm, BodyMotion) or not bm.actors:
            raise StageError("hands_face needs the smooth stage's BodyMotion with at least one fighter")
        p = params if isinstance(params, HandsFaceParams) else HandsFaceParams.model_validate(params)
        edit = ctx.edits.get("hands", ctx.edits.get("hands.json"))
        pins = stage_pins(ctx, bm, self.name)
        try:
            # a per-frame stage: the whole shot again, blended into the previous result around a range
            out, qc = apply_hands_face(bm, p, edit, pins)
            plan = range_plan(ctx, self.name, bm, HANDS_MARGIN, HANDS_BLEND)
            prev = ctx.previous if plan is not None else None
            if plan is not None and isinstance(prev, BodyMotion):
                out = blend_into_previous(prev, out, plan)
                qc.update(range_summary(ctx, bm, plan))
        except ValueError as e:
            raise StageError(f"hands_face: {e}") from e
        sk = get_skeleton(bm.skeleton)
        reports = []
        for role, ps in pins.items():  # the final pin error (smooth's QC is before this stage)
            reps = [PinReport(q.role, q.frame, sorted(q.bones), q.root is not None, q.radius) for q in ps]
            a = out.actors[role]
            pin_errors(sk, bm.actors[role], a.rot, a.root, ps, reps, bm.fps)
            reports += [r.to_dict() for r in reps]
        qc["pins"] = reports
        fighters: dict[str, dict] = qc["fighters"]  # type: ignore[assignment]
        summary = {
            "hands": p.hands,
            "face": p.face,
            "metacarpal_p95_deg": {r: f["gemx_metacarpals"].get("p95_deg") for r, f in fighters.items()},
            "hands_off_frames": {r: f.get("hands_off_frames", 0) for r, f in fighters.items()},
            "pins_max_error_deg": max((r["max_error_deg"] for r in reports), default=None),
            **({"range": qc["range"]} if "range" in qc else {}),
        }
        work = Path(ctx.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        (work / "qc.json").write_text(json.dumps({"summary": summary, **qc}, indent=1))
        ctx.progress(1.0, f"hands {p.hands}, face {p.face}")
        return out
