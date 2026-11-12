"""The "Retarget" sub-panel: a take fighter's motion onto your own character (``retarget.py``).

Pick the target armature (or **Import character**: FBX / GLB / an armature with its meshes from a
.blend), **Guess map** or **Load map** (``*.afvmap.json``), fix the bone list by hand, choose fighter
A/B and **Lock feet**, press **Retarget** (one undo step; the report is shown), then **Export
character** (FBX of the armature, its meshes and the retargeted action with an engine preset's FBX
settings). The map is also kept on the armature (custom property ``afv_retarget_map``), so a saved
.blend remembers it.

The presets' FBX settings come from afv (``afv.export.presets``) through the ``afv serve`` worker's
``presets`` request (or the one-shot ``afv retarget --list-presets --json``), once per session; they
are never imported here. Import and export are in ``retarget_io.py`` (shared with headless
``afv retarget``).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import bpy
import numpy as np
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from . import builder, ops, retarget_io, takeio, ui
from . import retarget as rt

PRESETS: dict[str, dict] = {}  # name -> afv preset dict (with "fbx"), fetched once per session
_SYNC = {"off": False}  # bulk edits of the bone list: don't store the map on every row change
_FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
AUTO = "AUTO"


def _is_finger(name: str) -> bool:
    return any(f"Hand{f}" in name for f in _FINGERS)


# --- properties ------------------------------------------------------------------------------------
def _target_poll(self, obj):
    return obj.type == "ARMATURE" and obj.get(builder.OBJ_KIND) is None  # not a take's own fighter


def _target_changed(self, context):
    self.report = ""
    self.error = ""
    tgt = self.target
    try:
        data = rt.map_from_object(tgt) if tgt is not None else None
    except ValueError as e:
        data = None
        self.error = f"the map stored on {tgt.name} is broken ({e}): Guess or Load a map"
    fill(self, data)


def _bone_changed(self, context):
    if not _SYNC["off"]:
        store(context.scene.afv_retarget, quiet=True)


def _facing_items(self, context):
    return [(AUTO, "Detect", "Detect which way the character faces")] + [
        (f, f, f"The character faces its armature's {f} axis") for f in rt.FACINGS]


class AFVRetargetBone(bpy.types.PropertyGroup):
    name: StringProperty(name="Canonical bone", description="afv's bone name (the export rig's)")
    bone: StringProperty(name="Target bone", description="The character's bone that follows it (empty: "
                         "not mapped, keeps its rest pose)", update=_bone_changed)


class AFVRetargetProps(bpy.types.PropertyGroup):
    target: PointerProperty(type=bpy.types.Object, name="Target", poll=_target_poll, update=_target_changed,
                            description="The character's armature (not a take's own fighter)")
    role: EnumProperty(name="Fighter", items=[("A", "A", "Fighter A"), ("B", "B", "Fighter B")])
    lock_feet: BoolProperty(name="Lock feet", default=True,
                            description="Hold planted feet in place (foot IK on the character's legs)")
    facing: EnumProperty(name="Facing", items=_facing_items,
                         description="Which way the character faces in its own armature axes (Detect: "
                                     "from its left/right bones)")
    hips_scale: FloatProperty(name="Hips scale", default=0.0, min=0.0, soft_max=200.0, precision=4,
                              description="Character size / fighter size for the hips motion (0 = from the "
                                          "rest heights of the pelvis)")
    bones: CollectionProperty(type=AFVRetargetBone)
    bone_index: IntProperty(name="Bone")
    show_fingers: BoolProperty(name="Fingers", default=False, description="List the finger bones too")
    preset: EnumProperty(name="Preset", items=ui._preset_items,
                         description="Engine preset whose FBX settings Export character uses")
    report: StringProperty()  # the last retarget's report (JSON)
    error: StringProperty()  # the last problem (import, map, retarget, export)


def props_of(context):
    return context.scene.afv_retarget


def fill(props, data) -> None:
    """The bone list (every canonical bone), facing and hips scale from a map dict (None: empty,
    detect)."""
    _SYNC["off"] = True
    try:
        if [r.name for r in props.bones] != list(rt.CANONICAL):
            props.bones.clear()
            for c in rt.CANONICAL:
                props.bones.add().name = c
        bones = (data or {}).get("bones") or {}
        for r in props.bones:
            r.bone = bones.get(r.name, "")
        f = (data or {}).get("facing")
        props.facing = f if f in rt.FACINGS else AUTO
        props.hips_scale = float((data or {}).get("hips_scale") or 0.0)
    finally:
        _SYNC["off"] = False


def current_map(props) -> dict:
    """The panel's map as a map dict (ValueError for a target bone used twice)."""
    bones = {r.name: r.bone for r in props.bones if r.bone}
    tgt = props.target
    hips = round(float(props.hips_scale), 6) or None  # a float32 property: no 0.6100000143051147 in files
    return rt.new_map(bones, facing=None if props.facing == AUTO else props.facing,
                      hips_scale=hips, target=tgt.name if tgt is not None else None)


def store(props, quiet=False) -> bool:
    """Keep the panel's map on the target armature (``map_to_object``). ``quiet``: a map that is not
    valid yet (a bone used twice while editing; the panel lists it) is not stored, no error."""
    tgt = props.target
    if tgt is None:
        return False
    try:
        rt.map_to_object(tgt, current_map(props))
    except ValueError:
        if not quiet:
            raise
        return False
    return True


def problems(props) -> dict:
    """What is wrong with the panel's map: missing required bones, names the target does not have,
    target bones used twice."""
    tgt = props.target
    bones = {r.name: r.bone for r in props.bones if r.bone}
    names = tgt.data.bones if tgt is not None else {}
    count = Counter(bones.values())
    return {"missing": rt.missing_required(bones), "absent": [c for c, b in bones.items() if b not in names],
            "twice": sorted(b for b, n in count.items() if n > 1), "mapped": len(bones)}


# --- the source fighter ----------------------------------------------------------------------------
def source(context):
    """(take record, the chosen fighter's armature or None) of the take the panel acts on."""
    scene = context.scene
    rec = builder.record_for(scene, context.active_object)
    if rec is None:
        return None, None
    return rec, builder.fighters(scene, rec).get(props_of(context).role)


# --- presets (from afv) ----------------------------------------------------------------------------
def presets_from(job) -> list[dict]:
    """The preset list in a ``presets`` reply (serve result, or the one-shot command's event)."""
    res = job.result() or {}
    if isinstance(res.get("presets"), list):
        return res["presets"]
    for ev in reversed(getattr(job, "events", [])):
        if ev.get("event") == "presets" and isinstance(ev.get("presets"), list):
            return ev["presets"]
    raise ValueError("afv sent no presets (update afv, then Restart worker in the add-on's preferences)")


def remember_presets(presets: list[dict]) -> None:
    for p in presets:
        if isinstance(p, dict) and isinstance(p.get("name"), str) and isinstance(p.get("fbx"), dict):
            PRESETS[p["name"]] = p


# --- the panel's own actions ---------------------------------------------------------------------------
ACTION_TAG = "afv_retarget"  # on actions Retarget made: JSON, see ``tag``
_PLAIN = {"BOOLEAN", "INT", "FLOAT", "ENUM", "STRING"}


def _rna_values(struct) -> list:
    """The plain (non-pointer) property values of an RNA struct, e.g. an F-curve modifier's settings."""
    out = []
    for prop in struct.bl_rna.properties:
        if prop.identifier == "rna_type" or prop.type not in _PLAIN:
            continue
        v = getattr(struct, prop.identifier, None)
        if prop.type == "ENUM" and prop.is_enum_flag:
            v = sorted(v)
        elif hasattr(v, "__len__") and not isinstance(v, str):
            v = list(v)
        out.append((prop.identifier, v))
    return out


def action_fingerprint(act) -> str:
    """A hash of everything about ``act`` a user may change: its slots, the F-curve set, every key's
    position, handles, handle types, interpolation and easing, and the F-curve modifiers with their
    settings."""
    h = hashlib.sha1()
    h.update(repr(sorted(sl.identifier for sl in act.slots)).encode())
    for layer in act.layers:
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", ()):
                h.update(repr(getattr(getattr(bag, "slot", None), "identifier", "")).encode())
                for fc in bag.fcurves:
                    kps = fc.keyframe_points
                    head = f"{fc.data_path}[{fc.array_index}] {len(kps)} {fc.extrapolation} {fc.mute}"
                    h.update(head.encode())
                    for attr in ("co", "handle_left", "handle_right"):
                        buf = np.empty(2 * len(kps), np.float32)
                        kps.foreach_get(attr, buf)
                        h.update(buf.tobytes())
                    for attr in ("interpolation", "easing", "handle_left_type", "handle_right_type"):
                        h.update(repr([getattr(k, attr) for k in kps]).encode())
                    h.update(repr([(m.type, _rna_values(m)) for m in fc.modifiers]).encode())
    return h.hexdigest()


def _take_key(rec) -> str:
    return f"{(rec or {}).get('take_id', '')}[{(rec or {}).get('instance', '')}]"


def tag(act, tgt, rec, role) -> None:
    """Remember on ``act`` that this panel made it: for which character object and take copy, under
    which name, as which datablock, and its fingerprint."""
    act[ACTION_TAG] = json.dumps({"target": tgt.name, "target_uid": tgt.session_uid, "take": _take_key(rec),
                                  "role": role, "name": act.name, "uid": act.session_uid,
                                  "fingerprint": action_fingerprint(act)})


def own_previous(tgt, rec, role):
    """The action this panel may replace on the next Retarget of ``tgt`` / this take copy / fighter:
    only the target's **active** action, and only when it is exactly as the panel made it: same
    datablock (not a copy), same character object, same name, same fingerprint
    (``action_fingerprint``). Otherwise None (a new action is made and nothing is deleted)."""
    ad = tgt.animation_data
    act = ad.action if ad is not None else None
    raw = act.get(ACTION_TAG) if act is not None else None
    if act is None or act.library is not None or not isinstance(raw, str):
        return None
    try:
        info = json.loads(raw)
    except ValueError:
        return None
    want = {"target_uid": tgt.session_uid, "take": _take_key(rec), "role": role, "name": act.name,
            "uid": act.session_uid}
    if any(info.get(k) != v for k, v in want.items()) or info.get("fingerprint") != action_fingerprint(act):
        return None
    return act


# --- operators ---------------------------------------------------------------------------------------
def short_error(e) -> str:
    """The last meaningful line of an error (Blender's operators raise RuntimeError with the whole
    traceback, then "Location: file:line"; bare "Error:" prefixes are dropped); the full text goes to
    the console."""
    text = str(e).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        print(f"AFV: {text}")
    lines = [ln[len("Error:"):].strip() if ln.startswith("Error:") else ln for ln in lines]
    useful = [ln for ln in lines if ln and not ln.startswith(("Location:", "Traceback", "File \"", "^", "~"))]
    return useful[-1] if useful else text


class _Op:
    def fail(self, context, message):
        msg = short_error(message)
        props_of(context).error = msg
        self.report({"ERROR"}, msg)
        return {"CANCELLED"}


class AFV_OT_retarget_import(_Op, bpy.types.Operator):
    """Import a character (FBX, GLB/glTF, or the armature and its meshes from a .blend; the .blend's
    scripts are not run) and make it the retarget target; the scene's fps and frame range stay"""

    bl_idname = "afv.retarget_import"
    bl_label = "Import character"
    bl_options = {"REGISTER", "UNDO"}  # noqa: RUF012 (Blender's API)

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.fbx;*.glb;*.gltf;*.blend", options={"HIDDEN"})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = props_of(context)
        try:
            res = retarget_io.import_character(context, bpy.path.abspath(self.filepath))
        except (retarget_io.CharacterError, RuntimeError) as e:
            return self.fail(context, e)
        arm = res["armature"]
        props.target = arm
        notes = list(res["notes"])
        try:
            stored = rt.map_from_object(arm)
        except ValueError:
            stored = None  # a broken stored map: guessed anew (the panel said why)
        if stored is None:
            try:
                fill(props, rt.new_map(rt.guess_map(arm), target=arm.name))
                store(props)
                props.error = ""
                miss = problems(props)["missing"]
                notes.append("map guessed" + (f"; map by hand: {', '.join(miss)}" if miss else ""))
            except ValueError as e:
                notes.append(str(e))
        msg = f"imported {arm.name} ({len(arm.data.bones)} bones, {len(res['objects'])} objects)"
        ops.set_status("; ".join([msg, *notes]), warning=bool(notes))
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class AFV_OT_retarget_guess(_Op, bpy.types.Operator):
    """Guess the bone map from the target's bone names and hierarchy (replaces the list)"""

    bl_idname = "afv.retarget_guess"
    bl_label = "Guess map"
    bl_options = {"REGISTER", "UNDO"}  # noqa: RUF012 (Blender's API)

    @classmethod
    def poll(cls, context):
        return props_of(context).target is not None

    def execute(self, context):
        props = props_of(context)
        tgt = props.target
        try:
            guess = rt.guess_map(tgt)
        except ValueError as e:
            return self.fail(context, e)
        data = {"bones": guess, "facing": None if props.facing == AUTO else props.facing,
                "hips_scale": props.hips_scale or None}
        fill(props, data)
        store(props, quiet=True)
        miss = problems(props)["missing"]
        props.error = ""
        text = f"guessed {len(guess)} bones on {tgt.name}"
        if miss:
            text += f"; map by hand: {', '.join(miss)}"
        self.report({"WARNING" if miss else "INFO"}, text)
        return {"FINISHED"}


class AFV_OT_retarget_load_map(_Op, bpy.types.Operator):
    """Load a bone map (*.afvmap.json) into the list"""

    bl_idname = "afv.retarget_load_map"
    bl_label = "Load map"
    bl_options = {"REGISTER", "UNDO"}  # noqa: RUF012 (Blender's API)

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.afvmap.json;*.json", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return props_of(context).target is not None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = props_of(context)
        try:
            data = rt.read_map(bpy.path.abspath(self.filepath))
        except (ValueError, OSError) as e:
            return self.fail(context, e)
        fill(props, data)
        store(props, quiet=True)
        p = problems(props)
        props.error = ""
        if p["absent"]:
            props.error = f"{props.target.name} has no bone named in the map for: {', '.join(p['absent'])}"
        self.report({"WARNING" if p["absent"] or p["missing"] else "INFO"},
                    f"loaded {len(data['bones'])} bones from {Path(self.filepath).name}")
        return {"FINISHED"}


class AFV_OT_retarget_save_map(_Op, bpy.types.Operator):
    """Save the bone map to a *.afvmap.json file (for afv retarget --map, or another .blend)"""

    bl_idname = "afv.retarget_save_map"
    bl_label = "Save map"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.afvmap.json;*.json", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return props_of(context).target is not None

    def invoke(self, context, event):
        if not self.filepath:
            base = Path(bpy.path.abspath("//")) if bpy.data.filepath else Path.home()
            self.filepath = str(base / f"{retarget_io.safe_name(props_of(context).target.name)}.afvmap.json")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = props_of(context)
        path = bpy.path.abspath(self.filepath)
        if not path.lower().endswith(".json"):
            path += ".afvmap.json"
        try:
            data = current_map(props)
            rt.write_map(path, data)
            rt.map_to_object(props.target, data)
        except (ValueError, OSError) as e:
            return self.fail(context, e)
        self.report({"INFO"}, f"saved the map ({len(data['bones'])} bones) to {path}")
        return {"FINISHED"}


class AFV_OT_retarget(_Op, bpy.types.Operator):
    """Retarget the chosen fighter's motion (its evaluated pose, tweak keys included) onto the target
    as a new action (one undo step; existing actions are kept)"""

    bl_idname = "afv.retarget"
    bl_label = "Retarget"
    bl_options = {"REGISTER", "UNDO"}  # noqa: RUF012 (Blender's API)

    @classmethod
    def poll(cls, context):
        return props_of(context).target is not None and source(context)[1] is not None

    def execute(self, context):
        props = props_of(context)
        tgt = props.target
        rec, src = source(context)
        props.report = ""
        if not _target_poll(None, tgt):
            return self.fail(context, f"{tgt.name} is a take's own fighter or not an armature: "
                                      "pick your character")
        if tgt.mode == "EDIT" or context.mode.startswith("EDIT"):
            return self.fail(context, "leave Edit Mode first (the retarget reads the armature's saved bones)")
        prev = own_previous(tgt, rec, props.role)
        try:
            data = current_map(props)
            p = problems(props)
            if p["missing"]:
                raise ValueError(f"map these required bones first: {', '.join(p['missing'])}")
            if p["absent"]:
                raise ValueError(f"{tgt.name} has no bone named in the map for: {', '.join(p['absent'])}")
            name = f"{rec.get('shot_id', 'take')}_{props.role}_{tgt.name}"
            rep = rt.retarget(context.scene, src, tgt, data, lock_feet=props.lock_feet, action_name=name)
            rt.map_to_object(tgt, data)
        except ValueError as e:  # the engine's refusals say what to fix
            return self.fail(context, e)
        act = bpy.data.actions[rep["action"]]
        if prev is not None and prev != act and prev.users <= 1:  # only its fake user: nothing else uses it
            old = prev.name
            bpy.data.actions.remove(prev)
            act.name = old
            rep["action"] = act.name
            rep["notes"].append(f"replaced this panel's previous unedited action {old!r}")
        tag(act, tgt, rec, props.role)
        drift = retarget_io.drift_note(tgt)
        if drift:
            rep["notes"].append(drift)
        props.error = ""
        props.report = json.dumps({**rep, "role": props.role, "source": src.name, "target": tgt.name})
        text = (f"retargeted fighter {props.role} onto {tgt.name}: {rep['mapped']} bones, frames "
                f"{rep['frames'][0]}-{rep['frames'][1]}, {rep['seconds']:.1f} s")
        ops.set_status(text + (f"; {drift}" if drift else ""), warning=bool(rep["notes"]))
        self.report({"INFO"}, text)
        return {"FINISHED"}


class AFV_OT_retarget_fix_rest(_Op, bpy.types.Operator):
    """Re-derive the character's stored bone rest matrices from their head, tail and roll (what Edit
    Mode does; the bones do not move), so its FBX re-imports exactly. Existing poses and actions then
    play slightly differently (about 2 mm at fighter scale for Girush): retarget again afterwards.
    One undo step"""

    bl_idname = "afv.retarget_fix_rest"
    bl_label = "Fix rest matrices"
    bl_options = {"REGISTER", "UNDO"}  # noqa: RUF012 (Blender's API)

    @classmethod
    def poll(cls, context):
        tgt = props_of(context).target
        return tgt is not None and not context.mode.startswith("EDIT")

    def execute(self, context):
        tgt = props_of(context).target
        try:
            n, worst = retarget_io.normalize_rest(context, tgt)
        except (retarget_io.CharacterError, RuntimeError) as e:
            return self.fail(context, e)
        if n:
            text = (f"{tgt.name}: re-derived {n} bones' rest matrices (they carried up to "
                    f"{100 * worst:.2f} % scale); existing actions play slightly differently now: "
                    "retarget again")
        else:
            text = f"{tgt.name}: the rest matrices are already clean"
        self.report({"INFO"}, text)
        return {"FINISHED"}


def _default_export_path(context) -> str:
    props = props_of(context)
    rec, _src = source(context)
    shot_id = rec.get("shot_id", "take") if rec else "take"
    name = f"{shot_id}_{props.role}_{retarget_io.safe_name(props.target.name)}"
    folder = None
    if rec is not None:
        try:
            shot = ops.target(rec)["shot_dir"]
            folder = Path(shot) / "exports" / "retarget" if shot is not None else None
        except (builder.BuildError, takeio.TakeError, ValueError, OSError):  # the .blend's folder then
            folder = None
    if folder is None:
        folder = Path(bpy.path.abspath("//")) if bpy.data.filepath else Path.home()
    return str(folder / f"{name}.fbx")


def export_character(context, target_name: str, preset: str, path: str) -> dict:
    """Write the target's FBX with ``preset``'s settings (``PRESETS`` must hold it)."""
    tgt = bpy.data.objects.get(target_name)
    if tgt is None:
        raise retarget_io.CharacterError(f"{target_name} is gone; nothing exported")
    p = PRESETS.get(preset)
    if p is None:
        known = ", ".join(sorted(PRESETS))
        raise retarget_io.CharacterError(f"afv has no preset {preset!r} (known: {known})")
    return retarget_io.export_fbx(context, tgt, path, p["fbx"])


class AFV_OT_retarget_export(_Op, bpy.types.Operator):
    """Export the target armature, its meshes and its retargeted action as FBX with the chosen engine
    preset's FBX settings (taken from afv)"""

    bl_idname = "afv.retarget_export"
    bl_label = "Export character"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.fbx", options={"HIDDEN"})
    check_existing: BoolProperty(default=True, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        tgt = props_of(context).target
        return tgt is not None and tgt.animation_data is not None and tgt.animation_data.action is not None

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = _default_export_path(context)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        props = props_of(context)
        path = bpy.path.abspath(self.filepath or _default_export_path(context))
        if not path.lower().endswith(".fbx"):
            path += ".fbx"
        tgt = props.target
        try:  # everything that can be checked now, before the presets are fetched
            if context.mode.startswith("EDIT") or tgt.mode == "EDIT":
                raise retarget_io.CharacterError("leave Edit Mode before exporting")
            retarget_io.action_frames(tgt)
        except retarget_io.CharacterError as e:
            return self.fail(context, e)
        name, preset, scene_name = tgt.name, props.preset, context.scene.name

        def write(ctx):
            res = export_character(ctx, name, preset, path)
            notes = list(res["notes"])
            drift = retarget_io.drift_note(bpy.data.objects[name])
            if drift:
                notes.append(drift)
            msg = (f"exported {name} ({res['meshes']} meshes, frames {res['frames'][0]}-{res['frames'][1]}, "
                   f"clip {res['take']!r}) with {preset} to {path}")
            ops.set_status("; ".join([msg, *notes]), warning=bool(notes))
            return msg

        if preset in PRESETS:
            try:
                self.report({"INFO"}, write(context))
            except (retarget_io.CharacterError, RuntimeError, OSError) as e:
                return self.fail(context, e)
            props.error = ""
            return {"FINISHED"}

        def show(msg):
            msg = short_error(msg)
            scene = bpy.data.scenes.get(scene_name)
            if scene is not None:
                scene.afv_retarget.error = msg
            ops.set_status(msg, error=True)

        def done(job):
            try:
                remember_presets(presets_from(job))
                write(bpy.context)
            except (retarget_io.CharacterError, ValueError, RuntimeError, OSError, KeyError) as e:
                show(f"Export character failed: {short_error(e)}")

        def failed(job, msg):
            show(f"Export character: cannot get the presets from afv: {msg}")

        try:
            ops.submit("presets", {}, "afv presets", done, context, on_fail=failed, undo=False)
        except (ValueError, RuntimeError) as e:
            return self.fail(context, e)
        props.error = ""
        self.report({"INFO"}, "getting the presets' FBX settings from afv; exports when done")
        return {"FINISHED"}


# --- panel -------------------------------------------------------------------------------------------
class AFV_UL_retarget_bones(bpy.types.UIList):
    """Canonical afv bone -> the character's bone (search over its bones)."""

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        tgt = data.target
        required = item.name in rt.REQUIRED
        absent = bool(item.bone) and tgt is not None and item.bone not in tgt.data.bones
        bad = (required and not item.bone) or absent
        row = layout.row(align=True)
        row.alert = bool(bad)
        split = row.split(factor=0.42, align=True)
        split.label(text=item.name + (" *" if required else ""),
                    icon="ERROR" if bad else ("BONE_DATA" if item.bone else "DOT"))
        if tgt is not None:
            split.prop_search(item, "bone", tgt.data, "bones", text="")
        else:
            split.prop(item, "bone", text="")

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        flags = bpy.types.UI_UL_list.filter_items_by_name(self.filter_name, self.bitflag_filter_item, items,
                                                          "name")
        if not flags:
            flags = [self.bitflag_filter_item] * len(items)
        if not data.show_fingers:
            flags = [0 if _is_finger(it.name) else f for f, it in zip(flags, items, strict=True)]
        return flags, []


def _fmt(v, unit=""):
    return "-" if v is None else f"{v:.2f}{unit}"


def report_lines(rep: dict) -> list[str]:
    slide = rep.get("foot_slide_cm_s") or {}
    f0, f1 = rep["frames"]
    lines = [f"{rep['source']} -> {rep['target']}: action {rep['action']}",
             f"frames {f0}-{f1}, {rep['mapped']} bones mapped, {rep['seconds']:.1f} s",
             f"facing {rep['facing']} (yaw {rep['facing_yaw_deg']:.1f} deg), "
             f"hips scale {rep['hips_scale']:.4f}",
             f"foot slide L {_fmt(slide.get('left'))} / R {_fmt(slide.get('right'))} cm/s",
             f"penetration {rep['penetration_cm']:.2f} cm; max error: direction "
             f"{rep['max_dir_err_deg']:.2f} deg, twist {rep['max_twist_err_deg']:.2f} deg"
             + (" (IK legs left out)" if rep.get("ik_bones") else "")]
    un = rep.get("unmapped") or []
    if un:
        lines.append(f"unmapped ({len(un)}): {', '.join(un[:6])}{' ...' if len(un) > 6 else ''}")
    return lines


def _wrap(layout, text, width=70, **kw):
    words, line = str(text).split(), ""
    for w in words:
        if line and len(line) + len(w) + 1 > width:
            layout.label(text=line, **kw)
            line, kw = w, {}
        else:
            line = f"{line} {w}".strip()
    if line:
        layout.label(text=line, **kw)


class AFV_PT_retarget(ui._Sub, bpy.types.Panel):
    bl_label = "Retarget"
    bl_idname = "AFV_PT_retarget"

    def draw(self, context):
        layout = self.layout
        props = props_of(context)
        rec, src = source(context)
        row = layout.row(align=True)
        row.label(text=f"Take {rec.get('shot_id', '?')} [{rec.get('instance', '-')}]", icon="ARMATURE_DATA")
        row.prop(props, "role", expand=True)
        if src is None:
            layout.label(text=f"This take has no fighter {props.role}", icon="ERROR")
        layout.prop(props, "target")
        layout.operator("afv.retarget_import", icon="IMPORT")
        tgt = props.target
        if tgt is None:
            layout.label(text="Pick the character's armature, or Import character", icon="INFO")
            return
        drift = retarget_io.drift_note(tgt)
        if drift:
            col = layout.column()
            col.alert = True
            _wrap(col, drift, icon="ERROR")
            col.operator("afv.retarget_fix_rest", icon="BONE_DATA")
        row = layout.row(align=True)
        row.operator("afv.retarget_guess", icon="VIEWZOOM")
        row.operator("afv.retarget_load_map", text="Load", icon="FILEBROWSER")
        row.operator("afv.retarget_save_map", text="Save", icon="FILE_TICK")
        row = layout.row(align=True)
        row.prop(props, "facing")
        row.prop(props, "hips_scale")
        p = problems(props)
        box = layout.box()
        box.label(text=f"{p['mapped']} of {len(rt.CANONICAL)} bones mapped (* = required)")
        for key, what in (("missing", "Required, not mapped"), ("absent", "Not in the armature"),
                          ("twice", "Used twice")):
            if p[key]:
                col = box.column()
                col.alert = True
                _wrap(col, f"{what}: {', '.join(p[key])}", icon="ERROR")
        row = box.row()
        row.prop(props, "show_fingers")
        box.template_list("AFV_UL_retarget_bones", "", props, "bones", props, "bone_index", rows=8)
        layout.prop(props, "lock_feet")
        row = layout.row()
        row.scale_y = 1.4
        row.operator("afv.retarget", icon="ARMATURE_DATA")
        if props.error:
            col = layout.column()
            col.alert = True
            _wrap(col, props.error, icon="ERROR")
        if props.report:
            try:
                rep = json.loads(props.report)
            except ValueError:
                rep = None
            if rep is not None:
                box = layout.box()
                box.label(text=f"Fighter {rep.get('role')}: last retarget", icon="CHECKMARK")
                for line in report_lines(rep):
                    box.label(text=line)
                for n in rep.get("notes") or []:
                    _wrap(box, f"note: {n}", icon="INFO")
        box = layout.box()
        box.label(text="Export character", icon="EXPORT")
        box.prop(props, "preset")
        box.operator("afv.retarget_export", icon="EXPORT")
        if props.preset not in PRESETS:
            box.label(text="(its FBX settings are fetched from afv on first use)")


PROP_CLASSES = (AFVRetargetBone, AFVRetargetProps)
CLASSES = (AFV_OT_retarget_import, AFV_OT_retarget_guess, AFV_OT_retarget_load_map, AFV_OT_retarget_save_map,
           AFV_OT_retarget, AFV_OT_retarget_fix_rest, AFV_OT_retarget_export, AFV_UL_retarget_bones,
           AFV_PT_retarget)


def register():
    for cls in PROP_CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.afv_retarget = PointerProperty(type=AFVRetargetProps)
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.afv_retarget
    for cls in reversed(PROP_CLASSES):
        bpy.utils.unregister_class(cls)
