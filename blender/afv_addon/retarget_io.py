"""Characters for the retarget: import, export and the FBX round-trip check (bpy + numpy only, no afv
imports, no UI). Shared by the add-on's Retarget panel (``retarget_ui.py``) and headless
``afv retarget`` (``retarget_headless.py``); the retarget itself is ``retarget.py``.

- ``CHARACTER_EXTS``: the file types; ``afv.retarget`` keeps an equal copy for its early check (a
  test compares them).
- ``import_character(context, path)``: an FBX / glTF (GLB) file through Blender's importers, or the
  armature with the most bones of a .blend (appended with its child objects and the meshes it
  deforms). For a .blend, ``bpy.data.libraries.load`` reads the objects and what they use, never the
  file's text blocks, registered scripts or handlers; every Python-expression driver in what it read
  is muted before anything is linked to the scene (so it cannot run even with Auto Run Python
  Scripts on; the mute is saved), and a file that links data with such drivers from a third file is
  refused (linked data cannot be changed, so its drivers would run). A hidden armature is shown. The data the
  kept objects use comes along (meshes, materials, the character's own actions); data only the
  dropped objects used stays behind as orphan data (Blender does not save it unless it has a fake
  user). The new objects go into their own collection ``Character <file stem>``; the scene's
  fps, frame range and current frame are put back afterwards (the FBX importer sets the scene to the
  file's fps); stored rest matrices with a scale are re-derived (``normalize_rest``).
- ``export_fbx(context, armature, path, fbx_args, frames)``: the armature, its meshes and its active
  action as FBX with an engine preset's exporter arguments (``afv.export.presets``, handed over as
  JSON); only those objects are written (``use_selection``), the scene's frame range is the action's
  and the scene is named like the action for the export (Blender's FBX exporter names the animation
  stack, the clip engines show, after the scene); selection, visibility, frame range, current frame
  and the scene name are restored.
- ``check_fbx(context, armature, path, frames)``: re-import the file and compare every common bone's
  evaluated world head with the armature's on every frame; the imported objects, data and actions
  are removed again.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import bpy
import numpy as np

CHARACTER_EXTS = (".fbx", ".glb", ".gltf", ".blend")
SET_ARGS = {"object_types"}  # exporter arguments that Blender wants as sets
COLLECTION_PREFIX = "Character "


class CharacterError(ValueError):
    """The character cannot be imported or exported; the message is fit to show."""


def safe_name(text: str) -> str:
    """A file-name-safe version of an object / file name."""
    out = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))).strip("._")
    return out or "character"


# --- import --------------------------------------------------------------------------------------
def _timing(scene):
    return (scene.render.fps, scene.render.fps_base, scene.frame_start, scene.frame_end, scene.frame_current)


def _restore_timing(scene, keep):
    scene.render.fps, scene.render.fps_base, scene.frame_start, scene.frame_end, frame = keep
    scene.frame_set(frame)


def _object_mode(context):
    if context.mode != "OBJECT" and context.view_layer.objects.active is not None:
        bpy.ops.object.mode_set(mode="OBJECT")


def deformed_by(obj, arm) -> bool:
    return any(m.type == "ARMATURE" and m.object == arm for m in getattr(obj, "modifiers", ()))


_SKIP_COLLECTIONS = {"window_managers", "screens", "workspaces", "libraries"}


def _id_collections():
    for p in bpy.data.bl_rna.properties:
        if p.type == "COLLECTION" and p.identifier not in _SKIP_COLLECTIONS:
            coll = getattr(bpy.data, p.identifier, None)
            if isinstance(coll, bpy.types.bpy_prop_collection):
                yield coll


def _id_pointers() -> set[int]:
    return {idb.as_pointer() for coll in _id_collections() for idb in coll if isinstance(idb, bpy.types.ID)}


def purge_new(before: set[int]) -> int:
    """Remove the local data blocks that are not in ``before`` (``_id_pointers``) and have no users
    (what a refused import left: armature, mesh, material data ...), repeating while removing one frees
    another. Returns how many were removed."""
    n = 0
    while True:
        dead = [idb for coll in _id_collections() for idb in coll
                if isinstance(idb, bpy.types.ID) and idb.as_pointer() not in before and idb.library is None
                and idb.users == 0]
        if not dead:
            return n
        bpy.data.batch_remove(dead)
        n += len(dead)


def _python_drivers(idb):
    """The drivers on ``idb`` (and its embedded node tree) whose expression needs Python."""
    for owner in (idb, getattr(idb, "node_tree", None)):
        ad = getattr(owner, "animation_data", None)
        for fc in (ad.drivers if ad is not None else ()):
            if fc.driver.type == "SCRIPTED" and not fc.driver.is_simple_expression:
                yield fc


def _new_ids(before: set[int]):
    for coll in _id_collections():
        for idb in coll:
            if isinstance(idb, bpy.types.ID) and idb.as_pointer() not in before:
                yield idb


def linked_python_drivers(before: set[int]) -> list[str]:
    """New data blocks linked from a library (not part of the appended copy, so not editable: muting
    their drivers would not be saved) that carry Python drivers: ``name (library path)``."""
    out = []
    for idb in _new_ids(before):
        if idb.library is not None and any(True for _ in _python_drivers(idb)):
            out.append(f"{idb.name} (from {bpy.path.abspath(idb.library.filepath)})")
    return sorted(set(out))


def mute_python_drivers(before: set[int]) -> int:
    """Mute every driver whose expression needs Python (type Scripted, not a simple expression) on the
    local data blocks that are not in ``before`` (``_id_pointers``): objects, their data, shape keys,
    materials and their node trees, node groups, ... Muted drivers never run, whatever Blender's Auto
    Run Python Scripts setting, and the mute is saved with the file; the user can unmute them in the
    Drivers editor. Linked data is not touched here (see ``linked_python_drivers``). Returns how many."""
    n = 0
    for idb in list(_new_ids(before)):
        if idb.library is not None:
            continue
        for fc in _python_drivers(idb):
            if not fc.mute:
                fc.mute = True
                n += 1
    return n


def _append_blend(path: str, collection, notes: list[str]) -> list:
    """The armature with the most bones in ``path``, its child objects, the meshes it deforms and
    their parents, linked to ``collection``; the file's other objects are not kept. Python drivers in
    everything the file brought are muted before anything is linked to the scene; data linked from a
    third file that carries Python drivers is refused (its drivers cannot be switched off)."""
    before = _id_pointers()
    libs_before = set(bpy.data.libraries)
    try:
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = list(src.objects)
    except (OSError, RuntimeError) as e:  # not a .blend, damaged, or from a newer Blender
        raise CharacterError(f"{path}: cannot read this .blend ({str(e).strip()})") from None
    linked = linked_python_drivers(before)
    if linked:
        for o in [o for o in dst.objects if o is not None]:
            bpy.data.objects.remove(o)
        for lib in [lib for lib in bpy.data.libraries if lib not in libs_before]:
            bpy.data.libraries.remove(lib)  # with the data it linked
        raise CharacterError(f"{path}: not imported: it links data with Python drivers from another file, "
                             f"which cannot be switched off: {', '.join(linked)}. Make that data local in "
                             "the character's file (Object > Relations > Make Local) or remove those drivers")
    muted = mute_python_drivers(before)
    if muted:
        notes.append(f"{muted} Python-expression drivers in the file were muted so they cannot run "
                     "(unmute them in the Drivers editor only if you trust the file)")
    loaded = [o for o in dst.objects if o is not None]
    arms = [o for o in loaded if o.type == "ARMATURE"]
    if not arms:
        for o in loaded:
            bpy.data.objects.remove(o)
        raise CharacterError(f"{path}: the file has no armature")
    arm = max(arms, key=lambda o: (len(o.data.bones), o.name))
    keep = {arm, *arm.children_recursive, *(o for o in loaded if deformed_by(o, arm))}
    for o in list(keep):
        p = o.parent
        while p is not None:
            keep.add(p)
            p = p.parent
    for o in loaded:
        if o in keep:
            collection.objects.link(o)
        else:
            bpy.data.objects.remove(o)
    return [o for o in loaded if o in keep]


def import_character(context, path) -> dict:
    """Import the character file ``path`` (.fbx, .glb/.gltf, .blend) into ``context.scene``.

    Returns ``{"armature": the armature with the most bones, "objects": [new objects], "collection",
    "seconds", "notes"}``. Raises CharacterError for an unknown type, a missing or unreadable file, a
    file without an armature, or a failure afterwards; anything imported is removed again then."""
    path = os.fspath(path)
    ext = Path(path).suffix.lower()
    if ext not in CHARACTER_EXTS:
        raise CharacterError(f"{path}: expected a character file ({', '.join(CHARACTER_EXTS)})")
    if not os.path.isfile(path):
        raise CharacterError(f"{path}: no such file")
    scene = context.scene
    t0 = time.perf_counter()
    keep = _timing(scene)
    _object_mode(context)
    before = set(bpy.data.objects)
    ids_before = _id_pointers()
    coll = bpy.data.collections.new(COLLECTION_PREFIX + Path(path).stem)
    scene.collection.children.link(coll)
    notes: list[str] = []
    try:
        if ext == ".blend":
            new = _append_blend(path, coll, notes)
        else:
            try:
                if ext == ".fbx":
                    res = bpy.ops.import_scene.fbx(filepath=path)
                else:
                    res = bpy.ops.import_scene.gltf(filepath=path)
            except RuntimeError as e:  # the importer's own error report
                raise CharacterError(f"{path}: the importer failed: {e}") from None
            if "FINISHED" not in res:
                raise CharacterError(f"{path}: the importer returned {sorted(res)}")
            new = [o for o in bpy.data.objects if o not in before]
            for o in new:  # into the character's own collection
                for c in list(o.users_collection):
                    c.objects.unlink(o)
                coll.objects.link(o)
        arms = [o for o in new if o.type == "ARMATURE"]
        if not arms:
            raise CharacterError(f"{path}: no armature in the file")
        arm = max(arms, key=lambda o: (len(o.data.bones), o.name))
        if arm.hide_viewport or arm.hide_select or arm.hide_get():
            arm.hide_viewport = arm.hide_select = False
            arm.hide_set(False)
            notes.append(f"the armature {arm.name!r} was hidden in the file; shown")
        view_layer = context.view_layer
        for o in view_layer.objects:
            o.select_set(o == arm)
        view_layer.objects.active = arm
        fixed, worst = normalize_rest(context, arm)
    except Exception as e:
        for o in [o for o in bpy.data.objects if o not in before]:
            bpy.data.objects.remove(o)
        bpy.data.collections.remove(coll)
        purge_new(ids_before)  # the 0-user data the import brought (meshes, materials, armatures ...)
        _restore_timing(scene, keep)
        if isinstance(e, CharacterError):
            raise
        raise CharacterError(f"{path}: import failed: {type(e).__name__}: {e}") from e
    fps_file = scene.render.fps / scene.render.fps_base
    _restore_timing(scene, keep)
    fps = scene.render.fps / scene.render.fps_base
    if abs(fps_file - fps) > 1e-6:
        notes.append(f"the importer set the scene to {fps_file:g} fps: put back to {fps:g} fps")
    if len(arms) > 1:
        notes.append(f"{len(arms)} armatures in the file: using {arm.name!r} ({len(arm.data.bones)} bones)")
    if fixed:
        notes.append(f"{fixed} bones' stored rest matrices carried up to {100 * worst:.2f} % scale "
                     "(file drift): re-derived from their head, tail and roll, as Edit Mode does")
    return {"armature": arm, "objects": new, "collection": coll, "notes": notes,
            "seconds": round(time.perf_counter() - t0, 3)}


REST_SCALE_TOL = 1e-5


def rest_drift(arm) -> tuple[int, float]:
    """(bones whose stored rest matrix is not a pure rotation, the worst scale deviation)."""
    n, worst = 0, 0.0
    for b in arm.data.bones:
        dev = float(np.abs(np.array(b.matrix_local.to_3x3().to_scale()) - 1.0).max())
        if dev > REST_SCALE_TOL:
            n, worst = n + 1, max(worst, dev)
    return n, worst


def drift_note(arm) -> str | None:
    """A warning when ``arm``'s stored rest matrices carry a scale (its FBX would not re-import exactly)."""
    n, worst = rest_drift(arm)
    if not n:
        return None
    return (f"{arm.name}: {n} bones' stored rest matrices carry up to {100 * worst:.2f} % scale (file "
            "drift), so its FBX re-imports up to that fraction off (about 1 mm at fighter scale for "
            "Girush): press Fix rest matrices, or bring the character in with Import character")


def normalize_rest(context, arm) -> tuple[int, float]:
    """Re-derive the bones' rest matrices of ``arm`` from their head, tail and roll (an Edit Mode round
    trip; head, tail and roll stay exactly as they are) when they carry a scale (Girush: up to
    0.12 %). Blender's FBX importer drops that scale, so the exported file would not come back as it
    was. The armature is made visible and selectable for it and put back, in the mode it was in. The
    bones' rest matrices change slightly, so existing poses and actions play slightly differently
    (Girush: posed heads up to about 2 mm at fighter scale): retarget again afterwards. Returns
    ``rest_drift`` from before (0 bones: nothing was done)."""
    n, worst = rest_drift(arm)
    if not n:
        return n, worst
    if arm.library is not None or arm.data.library is not None:
        raise CharacterError(f"{arm.name} is linked from a library: its rest matrices cannot be fixed")
    view_layer = context.view_layer
    if view_layer.objects.get(arm.name) != arm:
        raise CharacterError(f"{arm.name} is not in the view layer {view_layer.name!r}")
    active = view_layer.objects.active
    vis = (arm.hide_viewport, arm.hide_select, arm.hide_get())
    mode = arm.mode  # e.g. POSE: given back afterwards
    try:
        arm.hide_viewport = False
        arm.hide_select = False
        arm.hide_set(False)
        view_layer.objects.active = arm
        with context.temp_override(active_object=arm, object=arm):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.object.mode_set(mode=mode if mode in ("POSE", "OBJECT") else "OBJECT")
    finally:
        arm.hide_viewport, arm.hide_select = vis[0], vis[1]
        arm.hide_set(vis[2])
        view_layer.objects.active = active
    return n, worst


# --- export --------------------------------------------------------------------------------------
def character_meshes(scene, arm) -> list:
    """The meshes of ``arm`` in ``scene``: its descendants and the meshes it deforms."""
    out = []
    kids = set(arm.children_recursive)
    for o in scene.objects:
        if o.type == "MESH" and (o in kids or deformed_by(o, arm)):
            out.append(o)
    return out


def fbx_kwargs(fbx_args: dict) -> dict:
    """The preset's exporter arguments for one character: only the selected objects are written."""
    if not isinstance(fbx_args, dict) or not fbx_args:
        raise CharacterError("no FBX exporter arguments for this preset")
    args = {k: (set(v) if k in SET_ARGS else v) for k, v in fbx_args.items()}
    args.update(use_selection=True, use_visible=False, use_active_collection=False)
    return args


def action_frames(arm) -> tuple[int, int]:
    ad = arm.animation_data
    act = ad.action if ad is not None else None
    if act is None:
        raise CharacterError(f"{arm.name} has no action: retarget first")
    a, b = act.frame_range
    return round(a), round(b)


def export_fbx(context, arm, path, fbx_args: dict, frames: tuple[int, int] | None = None) -> dict:
    """Write ``arm``, its meshes and its active action to ``path`` (FBX) with ``fbx_args``.
    ``frames`` (first, last; default the action's range) is the exported range."""
    scene, view_layer = context.scene, context.view_layer
    if context.mode.startswith("EDIT"):
        raise CharacterError("leave Edit Mode before exporting")
    f0, f1 = frames if frames is not None else action_frames(arm)
    meshes = character_meshes(scene, arm)
    objs = [arm, *meshes]
    missing = [o.name for o in objs if view_layer.objects.get(o.name) != o]
    if arm.name in missing:
        raise CharacterError(f"{arm.name} is not in the view layer {view_layer.name!r} "
                             "(an excluded collection?)")
    objs = [o for o in objs if o.name not in missing]
    args = fbx_kwargs(fbx_args)
    path = os.fspath(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sel = [o for o in view_layer.objects if o.select_get()]
    active = view_layer.objects.active
    hidden = [o for o in objs if o.hide_get() or o.hide_viewport]
    vis = {o: (o.hide_get(), o.hide_viewport) for o in hidden}
    keep = _timing(scene)
    act = arm.animation_data.action if arm.animation_data is not None else None
    take = act.name if act is not None else scene.name
    scene_name = scene.name
    t0 = time.perf_counter()
    try:
        # the FBX exporter names the animation stack (the clip engines show) after the scene
        scene.name = take
        for o in hidden:
            o.hide_viewport = False
            o.hide_set(False)
        for o in view_layer.objects:
            o.select_set(False)
        for o in objs:
            o.select_set(True)
        view_layer.objects.active = arm
        scene.frame_start, scene.frame_end = int(f0), int(f1)
        res = bpy.ops.export_scene.fbx(filepath=path, **args)
        if "FINISHED" not in res:
            raise CharacterError(f"the FBX exporter returned {sorted(res)}")
    finally:
        written_take = scene.name
        scene.name = scene_name
        for o, (h, hv) in vis.items():
            o.hide_viewport = hv
            o.hide_set(h)
        for o in view_layer.objects:
            o.select_set(o in sel)
        view_layer.objects.active = active
        _restore_timing(scene, keep)
    notes = [f"{n} is not in the view layer: not exported" for n in missing]
    n_mesh = len([o for o in objs if o.type == "MESH"])
    if written_take != take:
        notes.append(f"another scene is called {take!r}: the clip in the file is {written_take!r}")
    return {"path": path, "objects": [o.name for o in objs], "meshes": n_mesh, "take": written_take,
            "frames": [int(f0), int(f1)], "notes": notes, "seconds": round(time.perf_counter() - t0, 3)}


# --- round-trip check ----------------------------------------------------------------------------
def world_heads(scene, arms, frames) -> list[dict[str, np.ndarray]]:
    """Per armature: bone name -> evaluated world heads [F,3] on ``frames``."""
    out = [{pb.name: np.empty((len(frames), 3)) for pb in a.pose.bones} for a in arms]
    for i, f in enumerate(frames):
        scene.frame_set(int(f))
        dg = bpy.context.evaluated_depsgraph_get()
        for a, rec in zip(arms, out, strict=True):
            ev = a.evaluated_get(dg)
            mw = np.array(ev.matrix_world)
            for pb in ev.pose.bones:
                h = np.array(pb.head)
                rec[pb.name][i] = mw[:3, :3] @ h + mw[:3, 3]
    return out


def check_fbx(context, arm, path, frames: tuple[int, int] | None = None, scale: float = 1.0) -> dict:
    """Re-import ``path`` and compare its armature with ``arm`` (evaluated world bone heads, every
    frame of ``frames``, bones of the same name). ``scale``: the character's size factor (the
    retarget's hips_scale), so the error is also given at the fighter's scale. Everything imported
    is removed again; the scene's timing is put back."""
    scene = context.scene
    f0, f1 = frames if frames is not None else action_frames(arm)
    keep = _timing(scene)
    before_obj, before_act = set(bpy.data.objects), set(bpy.data.actions)
    before_data = {k: set(getattr(bpy.data, k)) for k in ("armatures", "meshes", "materials", "images")}
    try:
        # anim_offset 0: the importer's default shifts the keys one frame later
        res = bpy.ops.import_scene.fbx(filepath=os.fspath(path), anim_offset=0.0)
        if "FINISHED" not in res:
            raise CharacterError(f"re-import of {path} returned {sorted(res)}")
        file_fps = scene.render.fps / scene.render.fps_base
        new = [o for o in bpy.data.objects if o not in before_obj]
        arms = [o for o in new if o.type == "ARMATURE"]
        if not arms:
            raise CharacterError(f"{path}: no armature after re-import")
        back = max(arms, key=lambda o: len(o.data.bones))
        act = back.animation_data.action if back.animation_data else None
        if act is None:
            raise CharacterError(f"{path}: the re-imported armature has no action")
        a0, a1 = act.frame_range
        keys = max((len(fc.keyframe_points) for fc in _fcurves(act, back)), default=0)
        _restore_timing(scene, keep)
        frames_list = list(range(int(f0), int(f1) + 1))
        mine, theirs = world_heads(scene, [arm, back], frames_list)
        common = sorted(set(mine) & set(theirs))
        err = np.zeros(len(frames_list))
        for n in common:
            err = np.maximum(err, np.linalg.norm(mine[n] - theirs[n], axis=-1))
        unit_m = scene.unit_settings.scale_length
        worst = float(err.max()) if common else float("nan")
        return {"bones": len(common), "bones_target": len(mine), "bones_file": len(theirs),
                "max_mm": worst * 1000.0 * unit_m,
                "max_mm_fighter": worst * 1000.0 * unit_m / max(scale, 1e-12),
                "worst_frame": frames_list[int(err.argmax())] if common else None,
                "frames": len(frames_list), "file_fps": round(file_fps, 6),
                "file_frames": [round(a0), round(a1)], "file_keys": keys,
                "meshes": len([o for o in new if o.type == "MESH"])}
    finally:
        for o in [o for o in bpy.data.objects if o not in before_obj]:
            bpy.data.objects.remove(o)
        for a in [a for a in bpy.data.actions if a not in before_act]:
            bpy.data.actions.remove(a)
        for k, old in before_data.items():
            coll = getattr(bpy.data, k)
            for d in [d for d in coll if d not in old and d.users == 0]:
                coll.remove(d)
        _restore_timing(scene, keep)


def _fcurves(action, obj):
    """The F-curves of ``action`` for ``obj``'s slot (layered actions) or all of them."""
    try:
        from bpy_extras import anim_utils

        slot = obj.animation_data.action_slot
        bag = anim_utils.action_get_channelbag_for_slot(action, slot)
        if bag is not None:
            return list(bag.fcurves)
    except (AttributeError, ImportError, RuntimeError):
        pass
    return list(getattr(action, "fcurves", []))
