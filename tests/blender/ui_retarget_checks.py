"""Blender-side checks for the add-on's Retarget panel and headless ``afv retarget`` (track R2), run with
``tests/blender/test_retarget_ui.py``'s ``run_ui_check``:

    blender -b --factory-startup --disable-autoexec -noaudio --python ui_retarget_checks.py -- job.json

job.json: {"result_path", "addon_parent", "tests_dir", "mode", ...}

- characters: {"out"}: synthetic characters built with the engine tests' rig builders
  (``retarget_checks.build_rig``, odd rolls, skinned meshes) and saved as files: ``syn_unity.fbx`` (Unity
  Humanoid names, written from a 24 fps scene), ``syn_unity.glb``, ``syn_anat.blend`` (anatomical names,
  centimetre scale, next to a second small armature and an unrelated mesh, plus a text block registered
  as a module that would write ``script_ran.txt`` if it ever ran) and ``syn_squashed.blend`` (a
  non-uniformly scaled object: the engine refuses it).
- ui: {"take", "fbx", "glb", "blend", "squashed", "expected", "presets", "out"}: register the add-on,
  import the take, then drive the Retarget panel's operators: Import character (fps kept), the guessed
  map (stored on the object), a missing required bone and a bone used twice (readable refusals),
  Retarget, Save / Load map, Export character through the one-shot ``afv retarget --list-presets``
  fallback (a fake afv) and again from the cache with another preset, the FBX round trip
  (``retarget_io.check_fbx``), an engine refusal (non-uniform scale), fighter B, the .blend and GLB
  imports, the target poll, switching targets, and the panel's draw code against a recording layout.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import bpy


# --- characters ----------------------------------------------------------------------------------
def _fresh(fps=30):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = fps, 1.0
    return scene


def characters(job):
    import retarget_checks as rc

    out = job["out"]
    res = {}
    scene = _fresh(24)  # the FBX importer sets the scene to the file's fps: 24 here, the take is 30
    arm, exp = rc.build_rig(scene, "SynUnity", rc.style_unity, 35.0, seed=3)
    res["fbx"] = os.path.join(out, "syn_unity.fbx")
    bpy.ops.export_scene.fbx(filepath=res["fbx"], add_leaf_bones=False, bake_anim=False)
    res["glb"] = os.path.join(out, "syn_unity.glb")
    bpy.ops.export_scene.gltf(filepath=res["glb"], export_format="GLB", export_animations=False)
    res["expected_unity"] = exp
    res["unity_bones"] = len(arm.data.bones)

    scene = _fresh()
    arm, exp = rc.build_rig(scene, "SynAnat", rc.style_anatomical, 25.0, unit=100.0, seed=4)
    extra = bpy.data.objects.new("Extra", bpy.data.armatures.new("Extra"))
    scene.collection.objects.link(extra)
    bpy.context.view_layer.objects.active = extra
    bpy.ops.object.mode_set(mode="EDIT")
    for i in range(2):
        eb = extra.data.edit_bones.new(f"b{i}")
        eb.head, eb.tail = (3.0, 0.0, i), (3.0, 0.0, i + 1.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.mesh.primitive_cube_add(location=(5.0, 0.0, 0.0))
    bpy.context.active_object.name = "Prop"
    marker = os.path.join(out, "script_ran.txt")
    txt = bpy.data.texts.new("startup_hook.py")
    txt.write(f"open({marker!r}, 'w').write('ran')\n")
    txt.use_module = True
    res["blend"] = os.path.join(out, "syn_anat.blend")
    res["marker"] = marker
    bpy.ops.wm.save_as_mainfile(filepath=res["blend"], copy=True)
    res["expected_anat"] = exp
    res["anat_bones"] = len(arm.data.bones)

    scene = _fresh()
    arm, _ = rc.build_rig(scene, "SynSquashed", rc.style_unity, 35.0, scale=(1.0, 1.3, 1.0), seed=5)
    res["squashed"] = os.path.join(out, "syn_squashed.blend")
    bpy.ops.wm.save_as_mainfile(filepath=res["squashed"], copy=True)

    # Python drivers that write a marker file when they run: on the armature object (a pose bone),
    # the mesh data, a shape key and the mesh's material node tree
    scene = _fresh()
    arm, _ = rc.build_rig(scene, "SynScripted", rc.style_unity, 35.0, seed=6)
    mobj = bpy.data.objects["SynScripted_mesh"]
    markers = {k: os.path.join(out, f"driver_{k}.txt").replace("\\", "/")
               for k in ("object", "data", "shape_key", "material")}

    def drive(fc, key):
        fc.driver.type = "SCRIPTED"
        fc.driver.expression = f"(open({markers[key]!r}, 'a').write('x') and 0) + frame * 0"

    drive(arm.pose.bones["Spine"].driver_add("location", 0), "object")
    mobj.data["afv_p"] = 0.0
    drive(mobj.data.driver_add('["afv_p"]'), "data")
    mobj.shape_key_add(name="Basis")
    drive(mobj.shape_key_add(name="Bulge").driver_add("value"), "shape_key")
    mat = bpy.data.materials.new("SynMat")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    drive(bsdf.inputs["Roughness"].driver_add("default_value"), "material")
    mobj.data.materials.append(mat)
    res["scripted"] = os.path.join(out, "syn_scripted.blend")
    res["driver_markers"] = markers
    bpy.ops.wm.save_as_mainfile(filepath=res["scripted"], copy=True)

    scene = _fresh()  # the armature disabled in viewports and unselectable
    arm, _ = rc.build_rig(scene, "SynHidden", rc.style_unity, 35.0, seed=7)
    arm.hide_viewport, arm.hide_select = True, True
    res["hidden"] = os.path.join(out, "syn_hidden.blend")
    bpy.ops.wm.save_as_mainfile(filepath=res["hidden"], copy=True)

    # a third file whose mesh data has a Python driver, linked (not appended) into a character file
    scene = _fresh()
    bpy.ops.mesh.primitive_cube_add(size=0.2)
    lib_cube = bpy.context.active_object
    lib_cube.name, lib_cube.data.name = "LibCube", "LibCubeMesh"
    lib_cube.data["afv_p"] = 0.0
    res["linked_marker"] = os.path.join(out, "driver_linked.txt").replace("\\", "/")
    fc = lib_cube.data.driver_add('["afv_p"]')
    fc.driver.type = "SCRIPTED"
    fc.driver.expression = f"(open({res['linked_marker']!r}, 'a').write('x') and 0) + frame * 0"
    lib_path = os.path.join(out, "lib_driver.blend")
    bpy.ops.wm.save_as_mainfile(filepath=lib_path)
    scene = _fresh()
    arm, _ = rc.build_rig(scene, "SynLinked", rc.style_unity, 35.0, seed=8)
    with bpy.data.libraries.load(lib_path, link=True) as (_src, dst):
        dst.meshes = ["LibCubeMesh"]
    user = bpy.data.objects.new("UsesLinkedMesh", dst.meshes[0])
    scene.collection.objects.link(user)
    user.parent = arm
    res["linked"] = os.path.join(out, "syn_linked.blend")
    bpy.ops.wm.save_as_mainfile(filepath=res["linked"], copy=True)

    res["broken"] = os.path.join(out, "broken.blend")
    with open(res["broken"], "wb") as fh:
        fh.write(b"BLENDER-v502RENDH" + bytes(range(256)) * 8)
    return res


# --- the panel ------------------------------------------------------------------------------------
class Layout:
    """Records what the panel draws (labels, props, operators); every call returns a sub-layout."""

    def __init__(self, log):
        self.log = log
        self.alert = False
        self.scale_y = 1.0

    def _sub(self, *a, **kw):
        return Layout(self.log)

    row = column = box = _sub

    def split(self, *a, **kw):
        return Layout(self.log)

    def label(self, text="", **kw):
        self.log.append(("label", text, self.alert))

    def prop(self, data, name, **kw):
        getattr(data, name)
        self.log.append(("prop", name))

    def prop_search(self, data, name, sdata, sname, **kw):
        getattr(data, name)
        getattr(sdata, sname)
        self.log.append(("prop_search", name))

    def operator(self, idname, **kw):
        mod, op = idname.split(".")
        assert hasattr(getattr(bpy.ops, mod), op), idname
        self.log.append(("operator", idname))
        return type("OpProps", (), {})()

    def template_list(self, cls, list_id, data, prop, adata, aprop, **kw):
        assert hasattr(bpy.types, cls), cls
        getattr(data, prop)
        getattr(adata, aprop)
        self.log.append(("template_list", cls))


def _draw(ui_mod, context):
    log = []
    panel = type("P", (), {"layout": Layout(log)})()
    ui_mod.AFV_PT_retarget.draw(panel, context)
    return log


def _call(fn, **kw):
    """(result set or None, error message or None) of an operator call (errors raise in background)."""
    try:
        return sorted(fn(**kw)), None
    except RuntimeError as e:
        return None, str(e)


def _finish(ops):
    ops.STATE.job.join(60)
    for _ in range(100):
        if ops._poll() is None:
            return
    raise RuntimeError("job did not finish")


def _data_count():
    """Local data blocks of the kinds an append brings (for "nothing left behind")."""
    return sum(len([d for d in coll if d.library is None]) for coll in (
        bpy.data.objects, bpy.data.armatures, bpy.data.meshes, bpy.data.materials, bpy.data.actions,
        bpy.data.collections))


def _fcurve(act):
    return next(fc for layer in act.layers for st in layer.strips for bag in st.channelbags
                for fc in bag.fcurves)


def reuse_cases(context, props, tgt):
    """Retarget again after each thing a user may do to the previous action (the reviewer's cases):
    only an untouched, active action of the same character object may be replaced."""
    def alive(uid):
        return any(a.session_uid == uid for a in bpy.data.actions)

    def case(setup):
        _call(bpy.ops.afv.retarget)  # a fresh, untouched action of this panel
        prev = tgt.animation_data.action
        uid, name = prev.session_uid, prev.name
        extra = setup(prev) or {}
        n = len(bpy.data.actions)
        res, err = _call(bpy.ops.afv.retarget)
        rep = json.loads(props.report) if props.report else {}
        return {"result": res, "error": err, "prev_alive": alive(uid), "prev_name": name,
                "new_actions": len(bpy.data.actions) - n, "active": tgt.animation_data.action.name,
                "replaced": any("replaced this panel" in x for x in rep.get("notes", [])),
                **{k: v() for k, v in extra.items()}}

    def renamed(prev):
        prev.name = "jab_keep"

    def nla(prev):
        ad = tgt.animation_data
        track = ad.nla_tracks.new()
        track.strips.new(prev.name, int(prev.frame_range[0]), prev)
        ad.action = None

    def copied(prev):
        c = prev.copy()
        c.name, c.use_fake_user = "jab_backup", True
        return {"copy_alive": lambda: "jab_backup" in bpy.data.actions}

    def interp(prev):
        _fcurve(prev).keyframe_points[0].interpolation = "CONSTANT"

    def fmod(prev):
        _fcurve(prev).modifiers.new("NOISE")

    def moved(prev):
        _fcurve(prev).keyframe_points[0].co.y += 0.1

    def other_object(prev):
        e = bpy.data.objects.new("UsesIt", None)
        context.scene.collection.objects.link(e)
        e.animation_data_create().action = prev

    out = {"unchanged": case(lambda prev: None), "renamed": case(renamed), "nla": case(nla),
           "copy": case(copied), "interpolation": case(interp), "modifier": case(fmod), "key": case(moved),
           "other_object": case(other_object)}
    for t in list(tgt.animation_data.nla_tracks):
        tgt.animation_data.nla_tracks.remove(t)

    # another character object with the same name, given the panel's untouched action
    _call(bpy.ops.afv.retarget)
    prev = tgt.animation_data.action
    uid, name = prev.session_uid, tgt.name
    twin = tgt.copy()  # same map (custom property), same data
    context.scene.collection.objects.link(twin)
    tgt.animation_data.action = None
    tgt.name = "Original_renamed"
    twin.name = name
    twin.animation_data.action = prev
    props.target = twin
    n = len(bpy.data.actions)
    res, err = _call(bpy.ops.afv.retarget)
    out["same_name_other_character"] = {"result": res, "error": err, "prev_alive": alive(uid),
                                        "new_actions": len(bpy.data.actions) - n, "twin": twin.name}
    props.target = None
    bpy.data.objects.remove(twin)
    tgt.name = name
    tgt.animation_data.action = prev
    props.target = tgt
    return out


def ui(job):
    import afv_addon
    from afv_addon import builder, ops, retarget_io, retarget_ui, wsl
    from afv_addon import retarget as rt

    afv_addon.register()
    out = {}
    try:
        scene = bpy.context.scene
        builder.import_take(job["take"], scene)
        fps0 = scene.render.fps / scene.render.fps_base
        rng0 = (scene.frame_start, scene.frame_end)
        props = scene.afv_retarget
        arms = builder.fighters(scene)
        out["poll_fighter"] = retarget_ui._target_poll(None, arms["A"])
        out["draw_empty"] = _draw(retarget_ui, bpy.context)

        res, err = _call(bpy.ops.afv.retarget_import, filepath=job["fbx"])
        tgt = props.target
        out["import"] = {"result": res, "error": err, "target": tgt.name if tgt else None,
                         "fps": scene.render.fps / scene.render.fps_base, "fps_before": fps0,
                         "range": [scene.frame_start, scene.frame_end], "range_before": list(rng0),
                         "status": ops.STATE.status, "collection": tgt.users_collection[0].name,
                         "poll_target": retarget_ui._target_poll(None, tgt)}
        stored = rt.map_from_object(tgt)
        out["guess"] = {"rows": {r.name: r.bone for r in props.bones if r.bone},
                        "stored": stored["bones"] if stored else None,
                        "problems": retarget_ui.problems(props)}
        out["draw_mapped"] = _draw(retarget_ui, bpy.context)
        rows = {r.name: r for r in props.bones}

        bpy.context.view_layer.objects.active = tgt
        bpy.ops.object.mode_set(mode="EDIT")
        n_act = len(bpy.data.actions)
        res, err = _call(bpy.ops.afv.retarget)
        out["edit_mode"] = {"result": res, "error": err, "new_actions": len(bpy.data.actions) - n_act}
        bpy.ops.object.mode_set(mode="OBJECT")
        try:
            props.target = arms["A"]
            assigned = props.target == arms["A"]
        except (TypeError, ValueError, AttributeError) as e:
            assigned = f"refused: {e}"
        res, err = (_call(bpy.ops.afv.retarget) if assigned is True else (None, None))
        out["fighter_target"] = {"assigned": assigned, "result": res, "error": err}
        if props.target != tgt:
            props.target = tgt

        foot = rows["LeftFoot"].bone
        rows["LeftFoot"].bone = ""
        n_act = len(bpy.data.actions)
        res, err = _call(bpy.ops.afv.retarget)
        out["missing"] = {"result": res, "error": err, "panel_error": props.error,
                          "stored_has_foot": "LeftFoot" in rt.map_from_object(tgt)["bones"],
                          "new_actions": len(bpy.data.actions) - n_act,
                          "draw": [x for x in _draw(retarget_ui, bpy.context) if x[0] == "label" and x[2]]}
        rows["LeftFoot"].bone = foot
        hand = rows["LeftHand"].bone
        rows["LeftHand"].bone = rows["RightHand"].bone
        res, err = _call(bpy.ops.afv.retarget)
        out["twice"] = {"result": res, "error": err, "problems": retarget_ui.problems(props),
                        "new_actions": len(bpy.data.actions) - n_act}
        rows["LeftHand"].bone = hand

        props.role, props.lock_feet = "A", True
        res, err = _call(bpy.ops.afv.retarget)
        rep = json.loads(props.report) if props.report else None
        out["retarget"] = {"result": res, "error": err, "report": rep,
                           "new_actions": len(bpy.data.actions) - n_act,
                           "active": tgt.animation_data.action.name if tgt.animation_data else None,
                           "undo_flag": "UNDO" in retarget_ui.AFV_OT_retarget.bl_options,
                           "draw": _draw(retarget_ui, bpy.context)}

        out["reuse"] = reuse_cases(bpy.context, props, tgt)
        _call(bpy.ops.afv.retarget)
        rep = json.loads(props.report)

        real_drift = retarget_io.rest_drift
        retarget_io.rest_drift = lambda arm: (3, 0.0012)  # as Girush's file drift
        try:
            res, err = _call(bpy.ops.afv.retarget)
            drift_rep = json.loads(props.report)
            draw = _draw(retarget_ui, bpy.context)
            bpy.context.view_layer.objects.active = tgt
            bpy.ops.object.mode_set(mode="POSE")  # Fix rest matrices gives the mode back
            res2, err2 = _call(bpy.ops.afv.retarget_fix_rest)
        finally:
            retarget_io.rest_drift = real_drift
        out["drift"] = {"result": res, "error": err, "notes": drift_rep["notes"],
                        "action": drift_rep["action"],
                        "fix_button": ("operator", "afv.retarget_fix_rest") in [tuple(x[:2]) for x in draw],
                        "fix": [res2, err2], "mode": bpy.context.mode, "fix_status": err2 or "",
                        "draw_after": ("operator", "afv.retarget_fix_rest") in
                        [tuple(x[:2]) for x in _draw(retarget_ui, bpy.context)]}
        rep = drift_rep
        bpy.ops.object.mode_set(mode="OBJECT")

        props.hips_scale = 0.61  # a float32 property: the file must say 0.61
        map_path = os.path.join(job["out"], "ui.afvmap.json")
        res, err = _call(bpy.ops.afv.retarget_save_map, filepath=map_path)
        saved = rt.read_map(map_path)
        rows["Spine"].bone = ""
        props.facing = "+X"
        res2, err2 = _call(bpy.ops.afv.retarget_load_map, filepath=map_path)
        with open(map_path, encoding="utf-8") as fh:
            raw_hips = json.load(fh)["hips_scale"]
        out["map_io"] = {"save": [res, err], "load": [res2, err2], "saved": saved["bones"],
                         "rows": {r.name: r.bone for r in props.bones if r.bone}, "facing": props.facing,
                         "hips_file": raw_hips, "hips_loaded": round(props.hips_scale, 6)}
        props.hips_scale = 0.0

        calls = []
        script = f"import sys; sys.stdout.write(open({job['presets_line']!r}).read())"

        def fake_argv(args, distro=None, afv=None):
            calls.append(list(args))
            return [sys.executable, "-c", script]

        ops.STATE.serve_default = False  # no afv serve here: the one-shot fallback
        wsl.afv_argv = fake_argv
        retarget_ui.PRESETS.clear()
        props.preset = "unity_humanoid"
        fbx_a = os.path.join(job["out"], "ui_A.fbx")
        bpy.context.view_layer.objects.active = tgt
        bpy.ops.object.mode_set(mode="EDIT")
        res, err = _call(bpy.ops.afv.retarget_export, filepath=fbx_a)
        out["export_edit"] = {"result": res, "error": err, "job": ops.STATE.job is not None,
                              "calls": len(calls),
                              "panel_error": props.error}
        bpy.ops.object.mode_set(mode="OBJECT")
        script = f"import sys; sys.stdout.write(open({job['presets_partial']!r}).read())"
        res, err = _call(bpy.ops.afv.retarget_export, filepath=fbx_a)
        if ops.STATE.job is not None:
            _finish(ops)
        out["export_fetch_fail"] = {"result": res, "error": err, "panel_error": props.error,
                                    "status": ops.STATE.status, "exists": os.path.isfile(fbx_a),
                                    "presets": sorted(retarget_ui.PRESETS)}
        retarget_ui.PRESETS.clear()
        calls.clear()
        script = f"import sys; sys.stdout.write(open({job['presets_line']!r}).read())"
        res, err = _call(bpy.ops.afv.retarget_export, filepath=fbx_a)
        pending = ops.STATE.job is not None
        if pending:
            _finish(ops)
        frames = tuple(rep["frames"]) if rep else None
        check = (retarget_io.check_fbx(bpy.context, tgt, fbx_a, frames, scale=rep["hips_scale"])
                 if os.path.isfile(fbx_a) and rep else None)
        out["export"] = {"result": res, "error": err, "pending": pending, "calls": calls,
                         "presets": sorted(retarget_ui.PRESETS), "exists": os.path.isfile(fbx_a),
                         "status": ops.STATE.status,
                         "check": check,
                         "fps_after": scene.render.fps / scene.render.fps_base}
        props.preset = "unreal"
        fbx_u = os.path.join(job["out"], "ui_A_unreal.fbx")
        res, err = _call(bpy.ops.afv.retarget_export, filepath=fbx_u)
        folder = os.path.join(job["out"], "a_folder.fbx")
        os.makedirs(folder, exist_ok=True)
        res2, err2 = _call(bpy.ops.afv.retarget_export, filepath=folder)
        out["export_folder"] = {"result": res2, "error": err2, "panel_error": props.error}
        out["export_cached"] = {"result": res, "error": err, "calls": len(calls),
                                "exists": os.path.isfile(fbx_u), "job": ops.STATE.job is not None}

        n_act = len(bpy.data.actions)
        tgt.scale = (1.0, 1.5, 1.0)
        res, err = _call(bpy.ops.afv.retarget)
        out["refused"] = {"result": res, "error": err, "panel_error": props.error,
                          "new_actions": len(bpy.data.actions) - n_act}
        tgt.scale = (1.0, 1.0, 1.0)

        props.role = "B"
        res, err = _call(bpy.ops.afv.retarget)
        out["role_b"] = {"result": res, "error": err,
                         "report": json.loads(props.report) if props.report else None}

        fbx_tgt = tgt
        names_before = set(bpy.data.objects.keys())
        res, err = _call(bpy.ops.afv.retarget_import, filepath=job["blend"])
        new = sorted(set(bpy.data.objects.keys()) - names_before)
        anat = props.target
        out["blend"] = {"result": res, "error": err, "target": anat.name if anat else None, "new": new,
                        "fps": scene.render.fps / scene.render.fps_base,
                        "marker": os.path.exists(job["marker"]), "texts": sorted(bpy.data.texts.keys()),
                        "rows": {r.name: r.bone for r in props.bones if r.bone},
                        "missing": retarget_ui.problems(props)["missing"]}
        props.target = fbx_tgt  # back: the rows come from the map stored on it
        out["switch_back"] = {r.name: r.bone for r in props.bones if r.bone}
        res, err = _call(bpy.ops.afv.retarget_import, filepath=job["glb"])
        out["glb"] = {"result": res, "error": err, "target": props.target.name if props.target else None,
                      "missing": retarget_ui.problems(props)["missing"]}
        res, err = _call(bpy.ops.afv.retarget_import, filepath=job["squashed"])
        props.role = "A"
        res2, err2 = _call(bpy.ops.afv.retarget)
        out["squashed"] = {"import": [res, err], "retarget": [res2, err2], "panel_error": props.error}
        res, err = _call(bpy.ops.afv.retarget_import, filepath=os.path.join(job["out"], "nope.fbx"))
        out["bad_file"] = {"result": res, "error": err}

        real_drift = retarget_io.rest_drift
        retarget_io.rest_drift = lambda arm: (2, 0.001)  # makes the import re-derive the rest matrices
        try:
            res, err = _call(bpy.ops.afv.retarget_import, filepath=job["hidden"])
        finally:
            retarget_io.rest_drift = real_drift
        hid = props.target
        out["hidden"] = {"result": res, "error": err, "target": hid.name if hid else None,
                         "hide_get": hid.hide_get() if hid else None,
                         "hide_viewport": hid.hide_viewport if hid else None,
                         "hide_select": hid.hide_select if hid else None, "mode": bpy.context.mode,
                         "status": ops.STATE.status}
        real_norm = retarget_io.normalize_rest

        def boom(context, arm):
            raise RuntimeError("Cannot edit hidden object")

        retarget_io.normalize_rest = boom
        n_obj, n_coll, n_data = len(bpy.data.objects), len(bpy.data.collections), _data_count()
        try:
            res, err = _call(bpy.ops.afv.retarget_import, filepath=job["hidden"])
        finally:
            retarget_io.normalize_rest = real_norm
        out["import_fails"] = {"result": res, "error": err, "objects_left": len(bpy.data.objects) - n_obj,
                               "collections_left": len(bpy.data.collections) - n_coll,
                               "data_left": _data_count() - n_data}
        n_obj, n_coll, n_lib = len(bpy.data.objects), len(bpy.data.collections), len(bpy.data.libraries)
        n_data = _data_count()
        res, err = _call(bpy.ops.afv.retarget_import, filepath=job["linked"])
        out["linked"] = {"result": res, "error": err, "objects_left": len(bpy.data.objects) - n_obj,
                         "collections_left": len(bpy.data.collections) - n_coll,
                         "libraries_left": len(bpy.data.libraries) - n_lib,
                         "data_left": _data_count() - n_data}
        n_obj, n_coll = len(bpy.data.objects), len(bpy.data.collections)
        res, err = _call(bpy.ops.afv.retarget_import, filepath=job["broken"])
        out["broken"] = {"result": res, "error": err, "panel_error": props.error,
                         "objects_left": len(bpy.data.objects) - n_obj,
                         "collections_left": len(bpy.data.collections) - n_coll}

        lst = retarget_ui.AFV_UL_retarget_bones
        fake = type("L", (), {"filter_name": "", "bitflag_filter_item": 1 << 30})()
        flags, _order = lst.filter_items(fake, bpy.context, props, "bones")
        props.show_fingers = True
        flags2, _ = lst.filter_items(fake, bpy.context, props, "bones")
        fake.filter_name = "*Foot*"
        flags3, _ = lst.filter_items(fake, bpy.context, props, "bones")
        log = []
        for item in list(props.bones)[:3]:
            lst.draw_item(fake, bpy.context, Layout(log), props, item, 0, props, "bone_index", 0)
        out["list"] = {"shown": sum(1 for f in flags if f), "shown_fingers": sum(1 for f in flags2 if f),
                       "shown_foot": sum(1 for f in flags3 if f), "rows": len(props.bones), "draw": log}
    finally:
        afv_addon.unregister()
    return out


def scripted(job):
    """Import a .blend whose drivers run Python, in a session where Python drivers do run (Auto Run on:
    a control driver made here proves it)."""
    from afv_addon import retarget_io

    scene = bpy.context.scene
    ctl = bpy.data.objects.new("Control", None)
    scene.collection.objects.link(ctl)
    fc = ctl.driver_add("location", 0)
    fc.driver.type = "SCRIPTED"
    fc.driver.expression = f"(open({job['control']!r}, 'a').write('x') and 0) + frame * 0"
    for f in range(3):
        scene.frame_set(f)
    out = {"autoexec_pref": bpy.context.preferences.filepaths.use_scripts_auto_execute,
           "control_ran": os.path.exists(job["control"])}
    res = retarget_io.import_character(bpy.context, job["blend"])
    for f in range(6):
        scene.frame_set(f)
    bpy.context.view_layer.update()
    out["notes"] = res["notes"]
    out["ran"] = {k: os.path.exists(p) for k, p in job["markers"].items()}
    muted = []
    for idb in [*bpy.data.objects, *bpy.data.meshes, *bpy.data.shape_keys, *bpy.data.materials]:
        for owner in (idb, getattr(idb, "node_tree", None)):
            ad = getattr(owner, "animation_data", None)
            for d in (ad.drivers if ad is not None else ()):
                muted.append([idb.name, d.data_path, d.mute])
    out["drivers"] = muted
    return out


def hide(job):
    """Save a copy of ``blend`` with its biggest armature disabled in viewports (run with Auto Run off)."""
    bpy.ops.wm.open_mainfile(filepath=job["blend"], load_ui=False)
    arm = max((o for o in bpy.data.objects if o.type == "ARMATURE"), key=lambda o: len(o.data.bones))
    arm.hide_viewport = True
    bpy.ops.wm.save_as_mainfile(filepath=job["out"], copy=True)
    return {"armature": arm.name}


def main():
    job_path = sys.argv[sys.argv.index("--") + 1]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    sys.path.insert(0, job["addon_parent"])
    sys.path.insert(0, job["tests_dir"])
    result = {"ok": False}
    try:
        modes = {"characters": characters, "ui": ui, "scripted": scripted, "hide": hide}
        result["out"] = modes[job["mode"]](job)
        result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"])
    with open(job["result_path"], "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
