"""The cleanup tools (slice 2): contact painting, pins, smoothing, range re-runs, hands off and ID
swaps. Each one writes an edit file and/or stage settings, re-solves through ``afv`` (the serve
worker, or one-shot commands) and re-imports the result:

    edit (edits/<name>.json) -> run --from <stage> [--frames a:b] --to hands_face -> take -> re-import

Re-solves of the motion stages (contacts, smooth, hands_face) re-import the solved motion only
(``builder.reimport_take(motion_only=True)``: camera, video, the current frame, Pose Mode and NLA
tweak mode stay); re-solves from earlier stages (ID swaps, re-runs from pose3d etc.) re-import
everything. The tweak track is never touched. Frame ranges: the panel's Start/End (inclusive), or the
timeline's preview range when it is on, clipped to the take; edits store them end-exclusive.

Blender's undo does not reach the shot's files: every fix is recorded (``edits.push_history``) and
"Revert last fix" puts the edit file (or the smoothing settings) back and re-solves.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import bpy
from bpy.props import EnumProperty, IntProperty, StringProperty

from . import builder, edits, ops, overlay, serve, takeio

MOTION_TARGET = "hands_face"  # the take's motion stage: every re-solve runs up to it
RANGE_STAGES = ("contacts", "smooth", "hands_face")  # range re-runs (others run in full)
STAGES = ("ingest", "detect_track", "camera", "pose3d", "pair", "contacts", "smooth", "hands_face")
_ERRORS = (builder.BuildError, takeio.TakeError, edits.EditError, serve.ServeError, ValueError, RuntimeError)


# --- context helpers -------------------------------------------------------------------------------
def fighter(context):
    """(armature object, record, role) of the active AFV armature, else (None, None, None)."""
    obj = context.active_object
    if obj is None or obj.get(builder.OBJ_KIND) != "armature":
        return None, None, None
    rec = builder.record_for(context.scene, obj)
    if rec is None:
        return None, None, None
    return obj, rec, obj.get(builder.OBJ_ROLE)


def active_bone(context):
    obj = context.active_object
    if obj is None or obj.type != "ARMATURE" or obj.data.bones.active is None:
        return None
    return obj.data.bones.active.name


def edit_range(scene, rec=None):
    """(start, end exclusive, where from) of the edit range, clipped to the take's frames (``rec``).
    An empty range after clipping raises EditError."""
    if scene.use_preview_range:
        a, b, src = scene.frame_preview_start, scene.frame_preview_end, "preview range"
    else:
        a, b, src = scene.afv.range_start, scene.afv.range_end, "panel"
    a, b = min(a, b), max(a, b)
    if rec is not None:
        lo, hi = int(rec["frame_start"]), int(rec["frame_end"])
        if b < lo or a > hi:
            raise edits.EditError(f"the edit range {a}-{b} ({src}) is outside the take ({lo}-{hi})")
        a, b = max(a, lo), min(b, hi)
    return a, b + 1, src


def _base(rec):
    t = ops.target(rec)
    if t["shot_dir"] is None:
        raise builder.BuildError(f"cannot find the shot folder of take {rec.get('take_id')}")
    return t, {"project": t["project"], "shot": t["shot_id"]}


_EDIT_CACHE: dict = {}


def read_edit_cached(shot_dir, name):
    """An edit file for the panel (re-read when it changes); {} when missing or unreadable."""
    p = edits.edit_path(shot_dir, name)
    try:
        key = os.stat(p).st_mtime_ns
    except OSError:
        return {}
    hit = _EDIT_CACHE.get(str(p))
    if hit is None or hit[0] != key:
        try:
            hit = (key, edits.read(shot_dir, name))
        except edits.EditError:
            hit = (key, {})
        _EDIT_CACHE[str(p)] = hit
    return hit[1]


def _current(shot_dir, name):
    """The edit file's content now (None when there is none), for the history."""
    return edits.read(shot_dir, name) if edits.edit_path(shot_dir, name).is_file() else None


class Fix:
    """One cleanup fix as the history will remember it (``edits.push_history``). Recorded once:
    when the edit was written, or, for settings, when the run ended in any way (settings are saved
    to the shot before the stages run: a paused or failed run changed them too)."""

    def __init__(self, shot_dir, label, resolve_args, edit=None, sets=()):
        self.shot_dir, self.recorded = shot_dir, False
        self.entry = {"id": uuid.uuid4().hex[:12], "label": label, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "resolve": resolve_args}
        if edit is not None:
            self.entry.update(name=edit[0], before=_current(shot_dir, edit[0]))
        elif sets:
            self.entry["settings_before"] = edits.shot_smoothing(shot_dir)

    def record(self, *_args):
        if self.recorded:
            return
        self.recorded = True
        try:
            self.entry["after_hash"] = edits.state_hash(edits.current_state(self.shot_dir, self.entry))
            edits.push_history(self.shot_dir, self.entry)
        except (edits.EditError, OSError) as e:
            ops.set_status(f"could not record '{self.entry['label']}' for Revert: {e}", warning=True)


def resolve(rec, scene, label, kind, *, stage, frames=None, sets=(), edit=None, full=False, yes=False,
            history=True, after_first=None, fix=None):
    """Start: [edit] -> run --from ``stage`` (to hands_face) -> take -> re-import of ``rec``'s copy.

    ``edit`` = (name, data or None to delete). With ``history`` the fix is recorded for Revert (see
    ``Fix``); "Run anyway" re-runs with the same ``fix`` and records it if that never happened.
    ``after_first(job)`` runs once the first step succeeded (Revert drops its entry there)."""
    t, base = _base(rec)
    shot_dir = t["shot_dir"]
    resolve_args = {"stage": stage, "frames": list(frames) if frames else None, "full": bool(full)}
    if fix is None and history and (edit is not None or sets):
        fix = Fix(shot_dir, label, resolve_args, edit, sets)

    def first_done(job):
        if fix is not None and (edit is not None or sets):
            fix.record()
        if after_first is not None:
            after_first(job)

    def run_ended(job, msg=None):  # a paused or failed run saved its settings all the same
        if fix is not None and sets and edit is None:
            fix.record()

    steps = []
    if edit is not None:
        name, data = edit
        args = {**base, "name": name, "data": data, "shot_dir": str(shot_dir)}
        steps.append((f"edit {name}", "edit", args, first_done))
    run = {**base, "from": stage, "to": MOTION_TARGET, "sets": list(sets), "yes": bool(yes)}
    if frames is not None:
        run["frames"] = serve.frames_arg(*frames)
    what = stage if frames is None else f"{stage} {frames[0]}-{frames[1] - 1}"
    run_done = first_done if edit is None else (lambda job: fix.record() if fix is not None else None)
    steps.append((f"run {what}", "run", run, run_done, run_ended))
    steps.append(("take", "take", ops.take_request(t, rec.get("preset", ""), check_plan=False)))
    scene_name, instance = scene.name, rec.get("instance")

    def finish(job):
        sc = bpy.data.scenes.get(scene_name)
        if sc is None:
            raise builder.BuildError(f"scene {scene_name!r} is gone; nothing re-imported")
        r = builder.record_by_instance(sc, instance) if instance else builder.scene_take(sc)
        if r is None:
            raise builder.BuildError("the take copy is no longer in the scene; nothing re-imported")
        builder.reimport_take(sc, rec=r, motion_only=not full)
        overlay.redraw_all()
        return f"{label}: re-solved and re-imported"

    def rerun(yes=True):
        resolve(rec, bpy.data.scenes.get(scene_name) or bpy.context.scene, label, kind, stage=stage,
                frames=frames, sets=sets, edit=None, full=full, yes=yes, history=False, fix=fix)

    ops.run_chain(label, steps, finish, kind, rerun=rerun)


class _Base:
    bl_options = {"REGISTER"}  # noqa: RUF012 (Blender's API); results are undo steps of their own

    def fail(self, e):
        self.report({"ERROR"}, str(e))
        return {"CANCELLED"}


# --- frame range -----------------------------------------------------------------------------------
class AFV_OT_set_range(bpy.types.Operator):
    """Set the edit range's start or end to the current frame"""

    bl_idname = "afv.set_range"
    bl_label = "Set range"

    edge: EnumProperty(items=[("start", "Start", ""), ("end", "End", "")])

    def execute(self, context):
        props, f = context.scene.afv, context.scene.frame_current
        if self.edge == "start":
            props.range_start = f
            props.range_end = max(props.range_end, f)
        else:
            props.range_end = f
            props.range_start = min(props.range_start, f)
        return {"FINISHED"}


# --- contacts --------------------------------------------------------------------------------------
class AFV_OT_paint_contact(_Base, bpy.types.Operator):
    """Mark the selected foot as planted / free over the edit range (edits/contacts.json; Clear hands
    the range back to the solver), then re-solve contacts and smoothing there and re-import"""

    bl_idname = "afv.paint_contact"
    bl_label = "Paint contact"

    state: EnumProperty(items=[("plant", "Plant", "Foot on the floor"), ("free", "Free", "Foot in the air"),
                               ("clear", "Clear", "Remove painted spans: the solver decides")])

    @classmethod
    def poll(cls, context):
        obj, rec, _ = fighter(context)
        bone = active_bone(context)
        return (not ops.busy() and obj is not None and bone is not None
                and edits.foot_side(bone, rec.get("prefix", "")) is not None)

    def execute(self, context):
        _obj, rec, role = fighter(context)
        foot = edits.foot_side(active_bone(context), rec.get("prefix", ""))
        try:
            a, b, _ = edit_range(context.scene, rec)
            t, _base_args = _base(rec)
            data = edits.paint_contact(edits.read(t["shot_dir"], "contacts"), role, foot, a, b, self.state)
            resolve(rec, context.scene, f"{self.state} {role} {foot} foot {a}-{b - 1}", "contacts",
                    stage="contacts", frames=(a, b), edit=("contacts", data))
        except _ERRORS as e:
            return self.fail(e)
        self.report({"INFO"}, f"{role} {foot} foot: {self.state} {a}-{b - 1}; re-solving in the background")
        return {"FINISHED"}


class AFV_OT_span_delete(_Base, bpy.types.Operator):
    """Remove this painted span (contacts) or hands-off span and re-solve its frames"""

    bl_idname = "afv.span_delete"
    bl_label = "Remove span"

    name: StringProperty(options={"HIDDEN"})
    index: IntProperty(options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return not ops.busy() and builder.record_for(context.scene, context.active_object) is not None

    def execute(self, context):
        rec = builder.record_for(context.scene, context.active_object)
        stage = {"contacts": "contacts", "hands": "hands_face"}.get(self.name)
        try:
            if stage is None:
                raise edits.EditError(f"no spans in edits/{self.name}.json")
            t, _b = _base(rec)
            data = edits.read(t["shot_dir"], self.name)
            spans = list(data.get("spans") or [])
            if not 0 <= self.index < len(spans):
                raise edits.EditError(f"no span {self.index} in edits/{self.name}.json")
            gone = spans.pop(self.index)
            lo, hi = int(rec["frame_start"]), int(rec["frame_end"]) + 1
            a, b = max(lo, int(gone["start"])), min(hi, int(gone["end"]))
            frames = (a, b) if b > a else None
            resolve(rec, context.scene, f"remove {self.name} span {gone.get('role')} {gone['start']}-"
                    f"{int(gone['end']) - 1}", self.name, stage=stage, frames=frames,
                    edit=(self.name, {**data, "spans": spans}))
        except _ERRORS as e:
            return self.fail(e)
        return {"FINISHED"}


# --- hands -----------------------------------------------------------------------------------------
class AFV_OT_hands(_Base, bpy.types.Operator):
    """Hands off over the edit range (blend to a relaxed hand: edits/hands.json), or back on;
    re-solves hands_face there and re-imports"""

    bl_idname = "afv.hands"
    bl_label = "Hands off"

    off: bpy.props.BoolProperty(default=True)

    @classmethod
    def poll(cls, context):
        return not ops.busy() and fighter(context)[0] is not None

    def execute(self, context):
        _obj, rec, role = fighter(context)
        side = context.scene.afv.hands_side
        try:
            a, b, _ = edit_range(context.scene, rec)
            t, _b = _base(rec)
            data = edits.mark_hands(edits.read(t["shot_dir"], "hands"), role, side, a, b, off=self.off)
            what = "hands off" if self.off else "hands on"
            resolve(rec, context.scene, f"{what} {role} {side} {a}-{b - 1}", "hands", stage="hands_face",
                    frames=(a, b), edit=("hands", data))
        except _ERRORS as e:
            return self.fail(e)
        return {"FINISHED"}


# --- pins ------------------------------------------------------------------------------------------
def pin_bones(obj, bones):
    """``bones`` without a separate root bone (above the hips; smooth rejects it: pin the hips)."""
    hips = obj.get(builder.OBJ_HIPS)
    roots = {b.name for b in obj.data.bones if b.parent is None and b.name != hips} if hips else set()
    keep = [b for b in bones if b not in roots]
    if not keep:
        raise edits.EditError("the root bone cannot be pinned: pin the hips instead")
    return keep


def _strip_time(strip, frame):
    """(action frame, influence) of an NLA strip at scene ``frame``, or None where it has no effect
    (muted, outside with extrapolation NOTHING)."""
    if strip.mute or strip.action is None:
        return None
    f0, f1 = strip.frame_start, strip.frame_end
    ext = strip.extrapolation
    if frame < f0:
        if ext != "HOLD":
            return None
        frame = f0
    elif frame > f1:
        if ext == "NOTHING":
            return None
        frame = f1
    a0, a1 = strip.action_frame_start, strip.action_frame_end
    length = max(a1 - a0, 1e-9)
    t = (frame - f0) / max(strip.scale, 1e-9)
    if strip.repeat > 1.0 and t > length:
        t = length if frame >= f1 else t % length
    t = min(t, length * max(strip.repeat, 1.0))
    t = a1 - t if strip.use_reverse else a0 + t
    if strip.use_animated_influence:
        infl = strip.influence
    else:
        infl = 1.0
        if strip.blend_in > 0 and frame < f0 + strip.blend_in:
            infl = min(infl, (frame - f0) / strip.blend_in)
        if strip.blend_out > 0 and frame > f1 - strip.blend_out:
            infl = min(infl, (f1 - frame) / strip.blend_out)
    return t, max(0.0, min(1.0, infl))


def _channel(action, slot, path, n, t, default):
    from bpy_extras import anim_utils

    bag = anim_utils.action_get_channelbag_for_slot(action, slot) if slot is not None else None
    vals = list(default)
    if bag is None:
        return vals, False
    found = False
    for fc in bag.fcurves:
        if fc.data_path == path and 0 <= fc.array_index < n and not fc.mute:
            vals[fc.array_index] = fc.evaluate(t)
            found = True
    return vals, found


def qpow(q, t):
    """q^t as Blender's NLA does it (``pow_qt_fl_normalized``): the angle from the stored w, so a key
    with negative w turns the long way, like the evaluated pose."""
    import math

    from mathutils import Quaternion, Vector

    q = q.normalized()
    angle = t * math.acos(max(-1.0, min(1.0, q.w)))
    v = Vector((q.x, q.y, q.z))
    v = v.normalized() * math.sin(angle) if v.length > 1e-12 else Vector((0.0, 0.0, 0.0))
    return Quaternion((math.cos(angle), v.x, v.y, v.z))


def tweak_sources(obj, frame):
    """The layers on top of the solve at ``frame``, bottom to top: [(action, slot, action frame,
    influence)] of every unmuted NLA track above "AFV solved" (the "AFV tweaks" track, pushed-down
    actions, ...) and the active action outside tweak mode. All must blend with Combine (the solve
    is then multiplied by them); another blend type raises EditError."""
    ad = obj.animation_data
    if ad is None:
        return []
    tracks = list(ad.nla_tracks)
    first = next((i + 1 for i, t in enumerate(tracks) if t.name == builder.TRACK_SOLVED), 0)
    upper = getattr(ad, "use_upper_stack_evaluation", True)
    tweaked = ad.nla_tracks.active if ad.use_tweak_mode else None
    out = []
    for track in tracks[first:]:
        if track.mute:
            continue
        for strip in track.strips:
            st = _strip_time(strip, frame)
            if st is None:
                continue
            if strip.blend_type != "COMBINE":
                raise edits.EditError(
                    f"the NLA strip '{strip.name}' on track '{track.name}' blends with "
                    f"{strip.blend_type.title()}: a pin needs every layer above the solve set to "
                    "Combine (or muted)")
            out.append((strip.action, strip.action_slot, st[0], st[1]))
        if tweaked is not None and track == tweaked and not upper:
            break  # tweak mode without upper-stack evaluation: tracks above are not shown
    if ad.action is not None and not ad.use_tweak_mode:
        if ad.action_blend_type != "COMBINE":
            raise edits.EditError(f"the active action '{ad.action.name}' blends with "
                                  f"{ad.action_blend_type.title()}: a pin needs Combine")
        out.append((ad.action, ad.action_slot, frame, ad.action_influence))
    return out


def tweak_basis(obj, bone, frame, sources=None):
    """The layers' contribution to ``bone`` at ``frame`` (``tweak_sources``): (rotation quaternion
    wxyz, location); the final basis is solved (x) this. Strip time mapping, repeat, reverse,
    extrapolation, blend in/out and influence are followed as Blender evaluates them."""
    from mathutils import Quaternion, Vector

    q, loc = Quaternion(), Vector((0.0, 0.0, 0.0))
    rpath, lpath = f'pose.bones["{bone}"].rotation_quaternion', f'pose.bones["{bone}"].location'
    for action, slot, t, infl in (tweak_sources(obj, frame) if sources is None else sources):
        r, has_r = _channel(action, slot, rpath, 4, t, (1.0, 0.0, 0.0, 0.0))
        lv, has_l = _channel(action, slot, lpath, 3, t, (0.0, 0.0, 0.0))
        if has_r:
            q = q @ qpow(Quaternion(r), infl)
        if has_l:
            loc = loc + Vector(lv) * infl
    return q, loc


def pose_pin(obj, rec, role, frame, radius, bones):
    """A pins.json entry from the armature's displayed pose at ``frame``, so that after the re-solve
    the final pose (solved (x) tweak) looks like the displayed one: per bone the local rotation (the
    take's ``rot`` convention, canonical bone names) of displayed (x) tweak^-1, and, with the hips
    among ``bones``, the hips head (armature space = afv world) minus the tweak's offset. The
    displayed pose is the pose bones' own transforms (constraints are not part of a pin)."""
    import numpy as np

    lo, hi = int(rec["frame_start"]), int(rec["frame_end"])
    if not lo <= int(frame) <= hi:
        raise edits.EditError(f"frame {frame} is outside the take ({lo}-{hi})")
    bones = pin_bones(obj, bones)
    prefix = rec.get("prefix", "") or ""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    sources = tweak_sources(obj, frame)

    def solved_local(name):  # displayed (x) layers^-1, as the take's local rotation (xyzw)
        tq, _tl = tweak_basis(obj, name, frame, sources)
        basis = ev.pose.bones[name].matrix_basis.to_quaternion() @ tq.inverted()
        rest = list(obj.data.bones[name].matrix_local.to_quaternion())
        return [round(float(v), 7) for v in edits.basis_to_local(np.array(rest), np.array(list(basis)))]

    out = {edits.canonical(name, prefix): solved_local(name) for name in bones}
    hips = obj.get(builder.OBJ_HIPS)
    hips = hips or next((b.name for b in obj.data.bones if b.name.endswith("Hips")), None)
    root = None
    if hips in bones:
        pb = ev.pose.bones[hips]
        _tq, tl = tweak_basis(obj, hips, frame, sources)
        lin = (pb.matrix @ pb.matrix_basis.inverted()).to_3x3()  # head = lin @ basis location + c
        root = [round(float(v), 6) for v in (pb.head - lin @ tl)]
    pin = {"role": role, "frame": int(frame), "radius": int(radius), "preset": rec.get("preset", ""),
           "bones": out, "root": root}
    if rec.get("root_motion"):
        pin["root_motion"] = rec["root_motion"]
    pin["prefix"] = prefix
    root_bone = next((b.name for b in obj.data.bones if b.parent is None and b.name != hips), None)
    if root_bone is not None:  # the root the user saw (extracted mode: the hips are relative to it)
        pin["root_rotation"] = solved_local(root_bone)
    return pin


def pin_qc(shot_dir):
    """smooth's per-pin QC of its current result (``qc.pins``: role, frame, max_error_deg,
    root_error_m, conflicts), keyed by (role, frame)."""
    cur = takeio._current_meta(Path(shot_dir) / "stages" / "smooth")
    pins = (((cur[1] if cur else {}) or {}).get("qc") or {}).get("pins") or []
    return {(p.get("role"), int(p.get("frame", -1))): p for p in pins if isinstance(p, dict)}


def conflict_text(c) -> str:
    """One pin conflict as a hint (smooth: leg pins on planted frames lose to contacts)."""
    if isinstance(c, dict):
        text = c.get("message") or c.get("kind") or json.dumps(c)
    else:
        text = str(c)
    if "plant" in text.lower() or "contact" in text.lower():
        text += ": paint the foot Free there first"
    return text


class AFV_OT_pin(_Base, bpy.types.Operator):
    """Pin the pose of the selected bones at the current frame (edits/pins.json): smoothing (the
    whole shot) then passes through it, blending over +-radius frames. Pose without keying, then
    Pin; the result (with your tweak keys) looks like the pose you pinned. Leg pins on planted frames
    lose to the contacts (paint the foot Free first)"""

    bl_idname = "afv.pin"
    bl_label = "Pin pose"

    @classmethod
    def poll(cls, context):
        obj = fighter(context)[0]
        return not ops.busy() and obj is not None and obj.mode == "POSE" and active_bone(context) is not None

    def execute(self, context):
        obj, rec, role = fighter(context)
        bones = [pb.name for pb in (context.selected_pose_bones or []) if pb.id_data == obj]
        if not bones:
            bones = [active_bone(context)]
        scene = context.scene
        try:
            pin = pose_pin(obj, rec, role, scene.frame_current, scene.afv.pin_radius, bones)
            t, _b = _base(rec)
            data = edits.add_pin(edits.read(t["shot_dir"], "pins"), pin)
            # smooth in full: a range re-run applies only the pins inside it (and weakens those in
            # its blend ramp); the whole shot is ~0.5 s
            resolve(rec, scene, f"pin {role} at {pin['frame']} ({len(pin['bones'])} bones)", "pins",
                    stage="smooth", edit=("pins", data))
        except _ERRORS as e:
            return self.fail(e)
        n = len(pin["bones"])
        self.report({"INFO"}, f"pinned {n} bones of {role} at frame {pin['frame']}; re-solving")
        return {"FINISHED"}


class AFV_OT_pin_delete(_Base, bpy.types.Operator):
    """Delete this pin and re-solve smoothing"""

    bl_idname = "afv.pin_delete"
    bl_label = "Delete pin"

    index: IntProperty()

    @classmethod
    def poll(cls, context):
        return not ops.busy() and builder.record_for(context.scene, context.active_object) is not None

    def execute(self, context):
        rec = builder.record_for(context.scene, context.active_object)
        try:
            t, _b = _base(rec)
            data, gone = edits.delete_pin(edits.read(t["shot_dir"], "pins"), self.index)
            resolve(rec, context.scene, f"unpin {gone.get('role')} at {gone.get('frame')}", "pins",
                    stage="smooth", edit=("pins", data))
        except _ERRORS as e:
            return self.fail(e)
        return {"FINISHED"}


class AFV_OT_goto(bpy.types.Operator):
    """Go to this frame"""

    bl_idname = "afv.goto"
    bl_label = "Go to frame"

    frame: IntProperty()

    def execute(self, context):
        context.scene.frame_set(self.frame)
        return {"FINISHED"}


# --- revert --------------------------------------------------------------------------------------
def revertable(shot_dir):
    """The newest recorded fix whose state is still what it left, or None."""
    return next((e for e in reversed(edits.history(shot_dir)) if edits.entry_matches(shot_dir, e)), None)


class AFV_OT_revert(_Base, bpy.types.Operator):
    """Undo a cleanup fix in the shot: put the edit file (or the shot's smoothing settings) back as
    it was before that fix and re-solve. Default: the newest fix that nothing changed since. Ctrl+Z
    only undoes Blender, not the shot's files"""

    bl_idname = "afv.revert"
    bl_label = "Revert last fix"

    entry_id: StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return not ops.busy() and builder.record_for(context.scene, context.active_object) is not None

    def execute(self, context):
        rec = builder.record_for(context.scene, context.active_object)
        try:
            t, _b = _base(rec)
            shot_dir = t["shot_dir"]
            if self.entry_id:
                entry = next((e for e in edits.history(shot_dir) if e.get("id") == self.entry_id), None)
                if entry is None:
                    raise edits.EditError("that fix is no longer in the history")
                edits.revert_check(shot_dir, entry)
            else:
                entry = revertable(shot_dir)
                if entry is None:
                    raise edits.EditError("no cleanup fix in this shot can be reverted (Forget changed ones)")
            r = entry.get("resolve") or {}
            frames = tuple(r["frames"]) if r.get("frames") else None
            edit = (entry["name"], entry.get("before")) if entry.get("name") else None
            sets = edits.revert_sets(entry["settings_before"]) if entry.get("settings_before") else ()
            if edit is None and not sets:
                raise edits.EditError(f"'{entry.get('label')}' changed nothing that can be put back")
            eid = entry.get("id")
            resolve(rec, context.scene, f"revert: {entry.get('label')}", "revert",
                    stage=r.get("stage", "smooth"), frames=frames, sets=sets, edit=edit,
                    full=bool(r.get("full")), history=False,
                    after_first=lambda job: edits.drop_history(shot_dir, eid))
        except _ERRORS as e:
            return self.fail(e)
        if sets:
            load_smoothing(context.scene, rec, settings=entry["settings_before"])
        return {"FINISHED"}


class AFV_OT_forget(bpy.types.Operator):
    """Forget this fix (its file or settings changed since, so it cannot be reverted); the shot is
    not changed"""

    bl_idname = "afv.forget"
    bl_label = "Forget"

    entry_id: StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return builder.record_for(context.scene, context.active_object) is not None

    def execute(self, context):
        rec = builder.record_for(context.scene, context.active_object)
        try:
            gone = edits.drop_history(ops.target(rec)["shot_dir"], self.entry_id)
        except (builder.BuildError, takeio.TakeError, OSError) as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        ops.set_status(f"forgot '{gone.get('label')}'" if gone else "already forgotten")
        return {"FINISHED"}


# --- smoothing -------------------------------------------------------------------------------------
def load_smoothing(scene, rec, settings=None):
    """The panel's sliders from the shot's saved settings (or from ``settings``, shot-level values
    by ``--set`` key, None = default)."""
    props = scene.afv
    if settings is not None:
        for key, val in settings.items():
            name = key.split(".")[-1]
            attr = "smooth_strength" if name == "strength" else f"smooth_{name}"
            if hasattr(props, attr):
                setattr(props, attr, 1.0 if val is None else float(val))
        return
    try:
        t = ops.target(rec)
    except (builder.BuildError, takeio.TakeError):
        return
    if t["shot_dir"] is None:
        return
    s = edits.smooth_settings(t["shot_dir"])
    props.smooth_strength = s["strength"]
    for r in edits.REGIONS:
        setattr(props, f"smooth_{r}", s["region_strength"][r])


class AFV_OT_smooth_apply(_Base, bpy.types.Operator):
    """Re-run smoothing with the sliders (saved to the shot as smooth.strength /
    smooth.region_strength.<region>) and re-import; the elapsed time is shown below"""

    bl_idname = "afv.smooth_apply"
    bl_label = "Apply"

    @classmethod
    def poll(cls, context):
        return not ops.busy() and builder.record_for(context.scene, context.active_object) is not None

    def execute(self, context):
        rec = builder.record_for(context.scene, context.active_object)
        props = context.scene.afv
        regions = {r: getattr(props, f"smooth_{r}") for r in edits.REGIONS}
        sets = edits.smooth_sets(props.smooth_strength, regions)
        try:
            resolve(rec, context.scene, "smoothing applied", "smooth", stage="smooth", sets=sets)
        except _ERRORS as e:
            return self.fail(e)
        return {"FINISHED"}


class AFV_OT_smooth_reset(bpy.types.Operator):
    """Set the sliders back to 1 (press Apply to re-solve)"""

    bl_idname = "afv.smooth_reset"
    bl_label = "Reset"

    def execute(self, context):
        props = context.scene.afv
        props.smooth_strength = 1.0
        for r in edits.REGIONS:
            setattr(props, f"smooth_{r}", 1.0)
        return {"FINISHED"}


class AFV_OT_smooth_load(bpy.types.Operator):
    """Load the sliders from the shot's saved settings"""

    bl_idname = "afv.smooth_load"
    bl_label = "From shot"

    @classmethod
    def poll(cls, context):
        return builder.record_for(context.scene, context.active_object) is not None

    def execute(self, context):
        load_smoothing(context.scene, builder.record_for(context.scene, context.active_object))
        return {"FINISHED"}


# --- re-run range ----------------------------------------------------------------------------------
class AFV_OT_rerun(_Base, bpy.types.Operator):
    """Re-run the chosen stage (and the motion stages after it) on the edit range, blended into the
    previous result at the edges (stages without range support run in full), then re-import"""

    bl_idname = "afv.rerun"
    bl_label = "Re-run range"

    whole: bpy.props.BoolProperty(name="Whole shot", default=False, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return not ops.busy() and builder.record_for(context.scene, context.active_object) is not None

    def execute(self, context):
        rec = builder.record_for(context.scene, context.active_object)
        stage = context.scene.afv.rerun_stage
        try:
            a, b, _ = edit_range(context.scene, rec)
            frames = None if self.whole else (a, b)
            what = "whole shot" if frames is None else f"{a}-{b - 1}"
            resolve(rec, context.scene, f"re-run {stage} {what}", "rerun", stage=stage, frames=frames,
                    full=stage not in RANGE_STAGES, history=False)
        except _ERRORS as e:
            return self.fail(e)
        return {"FINISHED"}


# --- tracking boxes and ID swaps -------------------------------------------------------------------
_OVERLAPPING = ("UI", "TOOLS", "HEADER", "TOOL_HEADER", "ASSET_SHELF", "ASSET_SHELF_HEADER")


def _inside(region, x, y):
    return region.x <= x < region.x + region.width and region.y <= y < region.y + region.height


def _view3d_under(context, x, y):
    """(region, its RegionView3D) of the 3D viewport's main region under window coordinates (x, y);
    (None, None) over a sidebar, toolbar or header drawn on top of it, or elsewhere. Quad view: the
    quadrant's own ``region.data``."""
    for area in context.window.screen.areas:
        if area.type != "VIEW_3D":
            continue
        if any(r.type in _OVERLAPPING and r.width > 1 and _inside(r, x, y) for r in area.regions):
            return None, None
        for region in area.regions:
            if region.type == "WINDOW" and _inside(region, x, y):
                return region, region.data
    return None, None


def swap_ids(context, rec, objects):
    scene = context.scene
    t, _b = _base(rec)
    lo, hi = int(rec["frame_start"]), int(rec["frame_end"])
    if not lo <= scene.frame_current <= hi:
        raise edits.EditError(f"frame {scene.frame_current} is outside the take ({lo}-{hi})")
    data = edits.add_swap(edits.read(t["shot_dir"], "ids"), scene.frame_current, objects)
    resolve(rec, scene, f"swap #{objects[0]} and #{objects[1]} from {scene.frame_current}", "ids",
            stage="detect_track", edit=("ids", data), full=True)


class AFV_OT_pick_boxes(_Base, bpy.types.Operator):
    """Click two tracking boxes in the camera view to swap their IDs from the current frame on
    (edits/ids.json), then re-solve from detect_track (pose3d runs again: minutes) and re-import"""

    bl_idname = "afv.pick_boxes"
    bl_label = "Swap IDs"

    @classmethod
    def poll(cls, context):
        return (not ops.busy() and not overlay.PICK.active
                and builder.record_for(context.scene, context.active_object) is not None)

    def invoke(self, context, event):
        self.rec = builder.record_for(context.scene, context.active_object)
        if overlay.tracks_for(context.scene, self.rec) is None:
            return self.fail("this take has no tracking boxes (re-import it after 'afv take' of round 4)")
        cam = overlay.take_camera(context.scene, self.rec)
        self.scene_name, self.old_camera = context.scene.name, None
        if cam is not None and context.scene.camera != cam:
            self.old_camera = context.scene.camera.name if context.scene.camera else ""
            context.scene.camera = cam  # the boxes are this take's: look through its camera (restored after)
        overlay.PICK.active, overlay.PICK.selected, overlay.PICK.rec = True, [], self.rec
        context.window_manager.modal_handler_add(self)
        overlay.redraw_all()
        return {"RUNNING_MODAL"}

    def _end(self):
        overlay.PICK.active, overlay.PICK.selected, overlay.PICK.rec = False, [], None
        scene = bpy.data.scenes.get(getattr(self, "scene_name", ""))
        if scene is not None and getattr(self, "old_camera", None) is not None:
            scene.camera = bpy.data.objects.get(self.old_camera) if self.old_camera else None
            self.old_camera = None
        overlay.redraw_all()

    def modal(self, context, event):
        if event.type == "ESC" and event.value == "PRESS":
            self._end()
            ops.set_status("swap IDs: cancelled")
            return {"CANCELLED"}
        if event.type not in ("LEFTMOUSE", "RIGHTMOUSE") or event.value != "PRESS":
            return {"PASS_THROUGH"}
        region, rv3d = _view3d_under(context, event.mouse_x, event.mouse_y)
        if region is None:
            return {"PASS_THROUGH"}  # another editor, or a sidebar / header over the viewport
        boxes, _tracks, corners = overlay.region_boxes(context.scene, region, rv3d, self.rec)
        x, y = event.mouse_x - region.x, event.mouse_y - region.y
        if corners is None or not takeio.inside_frame(corners, x, y):
            return {"PASS_THROUGH"}  # outside the camera frame: the viewport works as usual
        if event.type == "RIGHTMOUSE":
            self._end()
            ops.set_status("swap IDs: cancelled")
            return {"CANCELLED"}
        obj = takeio.box_at(boxes, x, y)
        if obj is None:
            return {"RUNNING_MODAL"}  # a click in the camera frame outside every box: ignored
        sel = overlay.PICK.selected
        if obj in sel:
            sel.remove(obj)
        else:
            sel.append(obj)
        overlay.redraw_all()
        if len(sel) < 2:
            return {"RUNNING_MODAL"}
        pair = list(sel)
        self._end()
        try:
            swap_ids(context, self.rec, pair)
        except _ERRORS as e:
            ops.set_status(f"swap IDs: {e}", error=True)
            return {"CANCELLED"}
        return {"FINISHED"}

    def cancel(self, context):
        self._end()


class AFV_OT_swap_delete(_Base, bpy.types.Operator):
    """Delete this ID swap and re-solve from detect_track"""

    bl_idname = "afv.swap_delete"
    bl_label = "Delete swap"

    index: IntProperty()

    @classmethod
    def poll(cls, context):
        return not ops.busy() and builder.record_for(context.scene, context.active_object) is not None

    def execute(self, context):
        rec = builder.record_for(context.scene, context.active_object)
        try:
            t, _b = _base(rec)
            data, gone = edits.delete_swap(edits.read(t["shot_dir"], "ids"), self.index)
            resolve(rec, context.scene, f"undo swap #{gone['objects'][0]}/#{gone['objects'][1]}", "ids",
                    stage="detect_track", edit=("ids", data), full=True)
        except _ERRORS as e:
            return self.fail(e)
        return {"FINISHED"}


CLASSES = (AFV_OT_set_range, AFV_OT_paint_contact, AFV_OT_span_delete, AFV_OT_hands, AFV_OT_pin,
           AFV_OT_pin_delete, AFV_OT_goto, AFV_OT_revert, AFV_OT_forget, AFV_OT_smooth_apply,
           AFV_OT_smooth_reset, AFV_OT_smooth_load, AFV_OT_rerun, AFV_OT_pick_boxes, AFV_OT_swap_delete)
