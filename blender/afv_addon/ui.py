"""The "AFV" sidebar panel (3D viewport, N panel), its scene settings and the add-on preferences.

Sub-panels (slice 2): Contacts, Pins, Smoothing, Re-run range, Hands, Tracking."""

from __future__ import annotations

from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty

from . import builder, cleanup, edits, ops, overlay, takeio, wsl

FALLBACK_PRESETS = [
    ("unity_humanoid", "Unity, Humanoid", ""),
    ("unity_generic", "Unity, Generic", ""),
    ("unreal", "Unreal", ""),
    ("godot", "Godot", ""),
    ("blender", "Blender", ""),
]
_ITEMS_CACHE: list = []  # Blender needs enum item strings kept alive


def _preset_items(self, context):
    rec = builder.scene_take(context.scene) if context and context.scene else None
    presets = sorted((rec or {}).get("presets", []), key=lambda p: p["name"] != (rec or {}).get("preset"))
    items = [(p["name"], p.get("label", p["name"]), ", ".join(p.get("formats", [])))
             for p in presets] or FALLBACK_PRESETS  # the take's own preset first (the default)
    _ITEMS_CACHE[:] = items
    return _ITEMS_CACHE


def _redraw_overlays(self, context):
    overlay.redraw_all()


def _region(name):
    return FloatProperty(name=name.capitalize(), default=1.0, min=0.0, soft_max=4.0, max=10.0,
                         description=f"Smoothing strength of the {name} (x the global strength; 1 = default)")


class AFVSceneProps(bpy.types.PropertyGroup):
    export_preset: EnumProperty(name="Preset", description="Engine preset for Export", items=_preset_items)
    export_copy_dir: StringProperty(
        name="Copy to", subtype="DIR_PATH",
        description="After Export, copy the FBX/GLB/BVH files here (e.g. your game's import folder)",
    )
    range_start: IntProperty(name="Start", default=0, description="First frame of the edit range")
    range_end: IntProperty(name="End", default=0, description="Last frame of the edit range (included)")
    pin_radius: IntProperty(name="Radius", default=edits.DEFAULT_PIN_RADIUS, min=1, soft_max=30,
                            description="Frames on each side over which smoothing blends into the pin")
    smooth_strength: FloatProperty(name="Strength", default=1.0, min=0.0, soft_max=4.0, max=10.0,
                                   description="Global smoothing strength (smooth.strength; 1 = default)")
    smooth_torso: _region("torso")
    smooth_head: _region("head")
    smooth_arms: _region("arms")
    smooth_hands: _region("hands")
    smooth_legs: _region("legs")
    smooth_root: _region("root")
    hands_side: EnumProperty(name="Hand",
                             items=[("both", "Both", ""), ("left", "Left", ""), ("right", "Right", "")])
    rerun_stage: EnumProperty(
        name="Stage", default="contacts",
        items=[(s, s + ("" if s in cleanup.RANGE_STAGES else " (whole shot)"), "") for s in cleanup.STAGES],
        description="Re-run from this stage (and the motion stages after it); contacts, smooth and "
                    "hands_face re-solve only the range, earlier stages the whole shot",
    )
    show_boxes: BoolProperty(name="Show boxes", default=False,
                             description="Draw detect_track's boxes and ids over the camera view",
                             update=_redraw_overlays)


class AFVPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    wsl_distro: StringProperty(name="WSL distribution", default=wsl.DEFAULT_DISTRO)
    afv_command: StringProperty(
        name="afv command (in WSL)", default=wsl.DEFAULT_AFV,
        description="The afv executable inside WSL, e.g. ~/afv/afv-env/bin/afv",
    )
    refresh_on_reimport: BoolProperty(
        name="Re-import refreshes the take", default=True,
        description="Re-import first runs 'afv take' in WSL so the take matches the latest results",
    )
    use_serve: BoolProperty(
        name="Use the afv serve worker", default=True,
        description="Keep one 'afv serve' running in WSL for this Blender session (fast re-solves); "
                    "off: every request is a one-shot afv command",
    )
    projects_dir: StringProperty(
        name="Projects folder", subtype="DIR_PATH",
        description="Shots > New project from video makes the project here, in a folder named after the "
                    "video (empty: next to the video)",
    )
    draw_strip: BoolProperty(
        name="Draw the confidence strip", default=True,
        description="Coloured QC bars in the Timeline / Dope Sheet / NLA / Graph editors "
                    "(the markers stay either way)",
    )

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "wsl_distro")
        col.prop(self, "afv_command")
        col.prop(self, "refresh_on_reimport")
        col.prop(self, "use_serve")
        row = col.row()
        row.label(text=ops.worker_status())
        row.operator("afv.restart_worker", icon="FILE_REFRESH")
        col.prop(self, "draw_strip")
        col.prop(self, "projects_dir")


class AFV_PT_panel(bpy.types.Panel):
    bl_label = "AFV"
    bl_idname = "AFV_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AFV"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        rec = builder.record_for(scene, context.active_object)
        layout.operator("afv.import_take", icon="IMPORT")
        if rec is not None:
            box = layout.box()
            box.label(text=f"Take {rec['take_id']}", icon="ARMATURE_DATA")
            box.label(text=f"Copy [{rec.get('instance', '-')}], "
                           f"frames {rec['frame_start']}-{rec['frame_end']} "
                           f"at {rec['fps']:g} fps")
            n = len(builder.instance_records(scene))
            if n > 1:
                box.label(text=f"{n} takes in this scene: select one to act on it", icon="INFO")
            box.operator("afv.reimport", icon="FILE_REFRESH")

            spans = builder.scene_spans(scene)
            box = layout.box()
            box.label(text=f"Confidence strip: {len(spans)} issue{'s' if len(spans) != 1 else ''}",
                      icon="ERROR" if spans else "CHECKMARK")
            row = box.row(align=True)
            row.operator("afv.issue", text="Previous", icon="TRIA_LEFT").direction = -1
            row.operator("afv.issue", text="Next issue", icon="TRIA_RIGHT").direction = 1
            for s in takeio.spans_at(spans, scene.frame_current)[:6]:
                box.label(text=f"{s['label']}  ({s['start']}-{s['end']})",
                          icon="CANCEL" if s["severity"] == "bad" else "INFO")

            box = layout.box()
            box.label(text="Export", icon="EXPORT")
            box.prop(scene.afv, "export_preset")
            box.prop(scene.afv, "export_copy_dir")
            box.operator("afv.export", icon="EXPORT")
            review = ops.STATE.review
            if review is not None and review.get("kind") != "run":
                rbox = box.box()
                rbox.label(text=f"Shot {review.get('shot_id', '')}: tracking QC hints", icon="ERROR")
                for line in str(review.get("message", "")).splitlines()[:10]:
                    if line.strip():
                        rbox.label(text=line.strip())
                if review.get("review_video"):
                    rbox.label(text=f"Review video: {review['review_video']}")
                row = rbox.row(align=True)
                op = row.operator("afv.export", text="Export anyway", icon="CHECKMARK")
                op.yes, op.instance = True, review.get("instance", "")
                op.reviewed_preset = review.get("preset", "")
                rbox.label(text=f"Exports copy [{review.get('instance', '-')}] with {review.get('preset')}")
                row.operator("afv.dismiss_review", icon="X")
        review = ops.STATE.review
        if review is not None and review.get("kind") == "run":
            rbox = layout.box()
            rbox.label(text=f"{review.get('label', 're-solve')}: tracking QC hints", icon="ERROR")
            for line in str(review.get("message", "")).splitlines()[:10]:
                if line.strip():
                    rbox.label(text=line.strip())
            row = rbox.row(align=True)
            row.operator("afv.go_on", icon="CHECKMARK")
            row.operator("afv.dismiss_review", icon="X")
        if rec is not None:
            fix_box(layout, context)
        status_box(layout)


_MATCH_CACHE: dict = {}


def _matches(shot, entry):
    """``edits.entry_matches``, cached while the file it looks at is unchanged (panel redraws)."""
    import os

    f = edits.edit_path(shot, entry["name"]) if entry.get("name") else Path(shot) / "shot.json"
    try:
        st = os.stat(f)
        key = (entry.get("id"), st.st_mtime_ns, st.st_size)
    except OSError:
        key = (entry.get("id"), None, None)
    if key not in _MATCH_CACHE:
        if len(_MATCH_CACHE) > 256:
            _MATCH_CACHE.clear()
        _MATCH_CACHE[key] = edits.entry_matches(shot, entry)
    return _MATCH_CACHE[key]


def fix_box(layout, context, shown=3):
    """The last fix that failed (its edit may be the cause), and the newest recorded fixes, each
    with Revert (nothing changed it since) or Forget (it was changed since: cannot be reverted)."""
    shot = _shot_dir(context)
    fixes = edits.history(shot) if shot is not None else []
    failed = ops.STATE.failed if ops.STATE.job is None else None
    if failed is None and not fixes:
        return
    box = layout.box()
    if failed is not None:
        col = box.column()
        col.alert = True
        col.label(text=f"{failed['label']}: {failed['step']} failed", icon="ERROR")
        for line in str(failed.get("message", "")).splitlines()[:4]:
            col.label(text=line[:160])
        box.label(text="If an edit causes it, revert that fix or remove the pin / span in its list")
    if fixes:
        box.label(text="Fixes in this shot (newest first):", icon="LOOP_BACK")
    for entry in list(reversed(fixes))[:shown]:
        row = box.row(align=True)
        if _matches(shot, entry):
            row.label(text=entry.get("label", ""))
            row.operator("afv.revert", text="Revert", icon="LOOP_BACK").entry_id = entry.get("id", "")
        else:
            sub = row.row()
            sub.alert = True
            sub.label(text=f"{entry.get('label', '')} (changed since)")
            row.operator("afv.forget", text="Forget", icon="X").entry_id = entry.get("id", "")
    if len(fixes) > shown:
        box.label(text=f"... and {len(fixes) - shown} older")
    box.label(text="Ctrl+Z undoes Blender only, not the shot's edits", icon="INFO")


def status_box(layout):
    if ops.STATE.job is not None:
        row = layout.row()
        row.label(text=ops.STATE.status or "working...", icon="SORTTIME")
        row.operator("afv.cancel", text="", icon="X")
    elif ops.STATE.status:
        col = layout.column()
        col.alert = ops.STATE.level == "error"
        icon = {"error": "CANCEL", "warning": "ERROR"}.get(ops.STATE.level, "INFO")
        col.label(text=ops.STATE.status, icon=icon)
    layout.label(text=ops.worker_status(), icon="CONSOLE")


class _Sub:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AFV"
    bl_parent_id = "AFV_PT_panel"
    bl_options = {"DEFAULT_CLOSED"}  # noqa: RUF012 (Blender's API)

    @classmethod
    def poll(cls, context):
        return builder.record_for(context.scene, context.active_object) is not None


def range_row(layout, scene):
    a, b, src = cleanup.edit_range(scene)  # (not clipped here: the tools clip to the take)
    if src == "preview range":
        layout.label(text=f"Frames {a}-{b - 1} (timeline preview range)", icon="PREVIEW_RANGE")
        return
    row = layout.row(align=True)
    row.prop(scene.afv, "range_start")
    row.operator("afv.set_range", text="", icon="TRIA_LEFT_BAR").edge = "start"
    row.prop(scene.afv, "range_end")
    row.operator("afv.set_range", text="", icon="TRIA_RIGHT_BAR").edge = "end"


def _shot_dir(context):
    rec = builder.record_for(context.scene, context.active_object)
    try:
        return ops.target(rec)["shot_dir"] if rec is not None else None
    except (builder.BuildError, takeio.TakeError):
        return None


def _timing(layout, kind):
    t = ops.STATE.timings.get(kind)
    if t:
        steps = ", ".join(f"{n} {s:.1f}" for n, s in t["steps"])
        layout.label(text=f"Last: {t['total']:.1f} s ({steps})", icon="TIME")


class AFV_PT_contacts(_Sub, bpy.types.Panel):
    bl_label = "Contacts"
    bl_idname = "AFV_PT_contacts"

    def draw(self, context):
        layout = self.layout
        obj, rec, role = cleanup.fighter(context)
        bone = cleanup.active_bone(context)
        foot = edits.foot_side(bone, rec.get("prefix", "")) if obj is not None and bone else None
        if foot is None:
            layout.label(text="Select a fighter's foot or leg bone", icon="INFO")
        else:
            layout.label(text=f"Fighter {role}, {foot} foot", icon="BONE_DATA")
        range_row(layout, context.scene)
        row = layout.row(align=True)
        row.operator("afv.paint_contact", text="Plant").state = "plant"
        row.operator("afv.paint_contact", text="Free").state = "free"
        row.operator("afv.paint_contact", text="Clear").state = "clear"
        shot = _shot_dir(context)
        if shot is not None:
            span_list(layout, cleanup.read_edit_cached(shot, "contacts"), "contacts", role,
                      lambda sp: f"{sp.get('foot') or ','.join(sp.get('channels') or [])} {sp.get('state')}")
        _timing(layout, "contacts")


def span_list(layout, data, name, role, what, limit=12):
    """The spans of an edit file (the selected fighter's first), each with jump and remove."""
    spans = list(enumerate(data.get("spans") or []))
    spans.sort(key=lambda e: (e[1].get("role") != role, int(e[1].get("start", 0))))
    if not spans:
        layout.label(text=f"no spans in edits/{name}.json")
        return
    for i, sp in spans[:limit]:
        row = layout.row(align=True)
        last = int(sp.get("end", 0)) - 1
        op = row.operator("afv.goto", text=f"{sp.get('role')} {what(sp)} {sp.get('start')}-{last}")
        op.frame = int(sp.get("start", 0))
        op = row.operator("afv.span_delete", text="", icon="X")
        op.name, op.index = name, i
    if len(spans) > limit:
        layout.label(text=f"... and {len(spans) - limit} more")


class AFV_PT_pins(_Sub, bpy.types.Panel):
    bl_label = "Pins"
    bl_idname = "AFV_PT_pins"

    def draw(self, context):
        layout = self.layout
        obj = cleanup.fighter(context)[0]
        if obj is None or obj.mode != "POSE":
            layout.label(text="Pose bones of a fighter (Pose Mode), then Pin", icon="INFO")
        row = layout.row(align=True)
        row.prop(context.scene.afv, "pin_radius")
        row.operator("afv.pin", icon="PINNED")
        shot = _shot_dir(context)
        pins = cleanup.read_edit_cached(shot, "pins").get("pins", []) if shot is not None else []
        qc = cleanup.pin_qc(shot) if pins else {}
        for i, pin in enumerate(pins):
            row = layout.row(align=True)
            op = row.operator("afv.goto", text=f"{pin.get('role')} frame {pin.get('frame')}", icon="PINNED")
            op.frame = int(pin.get("frame", 0))
            radius = pin.get("radius", edits.DEFAULT_PIN_RADIUS)
            q = qc.get((pin.get("role"), int(pin.get("frame", -1))))
            err = f", off {q['max_error_deg']:.1f} deg" if q and q.get("max_error_deg") is not None else ""
            row.label(text=f"{len(pin.get('bones') or {})} bones, +-{radius}{err}")
            row.operator("afv.pin_delete", text="", icon="X").index = i
            for c in (q or {}).get("conflicts") or []:
                sub = layout.row()
                sub.alert = True
                sub.label(text=cleanup.conflict_text(c), icon="ERROR")
        _timing(layout, "pins")


class AFV_PT_smoothing(_Sub, bpy.types.Panel):
    bl_label = "Smoothing"
    bl_idname = "AFV_PT_smoothing"

    def draw(self, context):
        layout = self.layout
        props = context.scene.afv
        layout.prop(props, "smooth_strength", slider=True)
        col = layout.column(align=True)
        for r in edits.REGIONS:
            col.prop(props, f"smooth_{r}", slider=True)
        row = layout.row(align=True)
        row.operator("afv.smooth_apply", icon="CHECKMARK")
        row.operator("afv.smooth_reset", icon="LOOP_BACK")
        row.operator("afv.smooth_load", icon="IMPORT")
        _timing(layout, "smooth")


class AFV_PT_rerun(_Sub, bpy.types.Panel):
    bl_label = "Re-run range"
    bl_idname = "AFV_PT_rerun"

    def draw(self, context):
        layout = self.layout
        range_row(layout, context.scene)
        layout.prop(context.scene.afv, "rerun_stage")
        row = layout.row(align=True)
        row.operator("afv.rerun", icon="FILE_REFRESH")
        row.operator("afv.rerun", text="Whole shot").whole = True
        _timing(layout, "rerun")


class AFV_PT_hands(_Sub, bpy.types.Panel):
    bl_label = "Hands"
    bl_idname = "AFV_PT_hands"

    def draw(self, context):
        layout = self.layout
        obj, _rec, role = cleanup.fighter(context)
        layout.label(text=f"Fighter {role}" if obj is not None else "Select a fighter", icon="VIEW_PAN")
        range_row(layout, context.scene)
        layout.prop(context.scene.afv, "hands_side", expand=True)
        row = layout.row(align=True)
        row.operator("afv.hands", text="Hands off").off = True
        row.operator("afv.hands", text="Hands on").off = False
        shot = _shot_dir(context)
        if shot is not None:
            span_list(layout, cleanup.read_edit_cached(shot, "hands"), "hands", role,
                      lambda sp: f"{sp.get('side')} hands off")
        _timing(layout, "hands")


class AFV_PT_tracking(_Sub, bpy.types.Panel):
    bl_label = "Tracking"
    bl_idname = "AFV_PT_tracking"

    def draw(self, context):
        layout = self.layout
        tracks = overlay.tracks_for(context.scene, builder.record_for(context.scene, context.active_object))
        if tracks is None:
            layout.label(text="No tracking boxes in this take (refresh it)", icon="INFO")
            return
        layout.prop(context.scene.afv, "show_boxes")
        fighters = ", ".join(f"{r} = #{o}" for r, o in sorted(tracks.fighters.items()))
        layout.label(text=f"Objects {', '.join(f'#{i}' for i in tracks.ids)}; {fighters}")
        layout.operator("afv.pick_boxes", icon="UV_SYNC_SELECT")
        layout.label(text="Swaps run from detect_track: pose3d re-solves (minutes)", icon="INFO")
        shot = _shot_dir(context)
        swaps = cleanup.read_edit_cached(shot, "ids").get("swaps", []) if shot is not None else []
        for i, sw in enumerate(swaps):
            row = layout.row(align=True)
            o = sw.get("objects") or ["?", "?"]
            end = sw.get("end")
            op = row.operator("afv.goto", text=f"#{o[0]} <> #{o[1]} from {sw.get('start')}"
                              + (f" to {end - 1}" if end is not None else ""), icon="UV_SYNC_SELECT")
            op.frame = int(sw.get("start", 0))
            row.operator("afv.swap_delete", text="", icon="X").index = i
        _timing(layout, "ids")


CLASSES = (AFVSceneProps, AFVPreferences, AFV_PT_panel, AFV_PT_contacts, AFV_PT_pins, AFV_PT_smoothing,
           AFV_PT_rerun, AFV_PT_hands, AFV_PT_tracking)
