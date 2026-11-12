"""Blender-side checks for tests/blender (run headless through afv.export.blender.run_blender).

job.json: {"result_path": .., "addon_parent": <repo>/blender, "mode": "reopen"|"reimport"|"addon", ...}

- reopen:   {"blend"}: open a saved take .blend; report its NLA, camera, clip (path, missing), markers.
- reimport: {"blend", "take2", "role", "bone", "frame"}: key a rotation on the tweak track, re-import
            the take from ``take2``, report what survived and the evaluated pose (bone heads, Blender
            world) plus the solved and final pose basis of ``bone`` at ``frame``.
- addon:    {"take"}: register the add-on, import with the builder, run Next/Previous issue and the
            Re-import operator (no WSL refresh: the add-on isn't in the preferences here).
- jobs:     {"take"}: the background-job lifecycle with a fake afv (Blender's own Python): a file
            load mid-job, Cancel, a failing callback, Export paused for a tracking review and
            "Export anyway", unregister with a job running.
- instances: {"take", "take2"}: import the take twice, re-import one copy.
- moved:    {"blend"}: open a .blend whose project folder moved; find the take again and re-import.
- cleanup:  {"take", "shot_dir", "fake", "cfg", "log", "cfg_review", "log_review", "cfg_slow", "log_slow"}:
            the slice-2 tools against tests/blender/fake_serve.py (Blender's own Python): paint a
            contact, pin poses, smoothing, hands off, re-run range, ID swap, a paused run and "Run
            anyway", Cancel, and the one-shot fallback when afv serve cannot start.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback


def _nla(obj):
    return [{"name": t.name, "strips": [{"action": s.action.name if s.action else None, "blend": s.blend_type,
                                         "keys": sum(len(fc.keyframe_points) for fc in _curves(s)),
                                         "start": s.frame_start, "end": s.frame_end,
                                         "action_range": list(s.action.frame_range) if s.action else None}
                                        for s in t.strips]}
            for t in obj.animation_data.nla_tracks]


def _curves(strip):
    from bpy_extras import anim_utils

    bag = anim_utils.action_get_channelbag_for_slot(strip.action, strip.action_slot)
    return list(bag.fcurves) if bag is not None else []


def _arms(scene):
    return {o["afv_role"]: o for o in scene.objects if o.get("afv_kind") == "armature"}


def reopen(job):
    import bpy

    bpy.ops.wm.open_mainfile(filepath=job["blend"])
    scene = bpy.context.scene
    cam = scene.camera
    bg = cam.data.background_images[0] if cam and len(cam.data.background_images) else None
    clip = bg.clip if bg else None
    return {
        "fighters": {r: _nla(o) for r, o in _arms(scene).items()},
        "camera": cam.name if cam else None,
        "show_background": bool(cam and cam.data.show_background_images),
        "clip_path": clip.filepath if clip else None,
        "clip_abs": bpy.path.abspath(clip.filepath) if clip else None,
        "clip_size": list(clip.size) if clip else None,
        "clip_frame_start": clip.frame_start if clip else None,
        "clip_frames": clip.frame_duration if clip else None,
        "markers": sorted((m.frame, m.name) for m in scene.timeline_markers),
        "fps": scene.render.fps / scene.render.fps_base,
        "frames": [scene.frame_start, scene.frame_end],
        "has_take": "afv_take" in scene,
        "view_camera": [s.region_3d.view_perspective for sc in bpy.data.screens for a in sc.areas
                        if a.type == "VIEW_3D" for s in a.spaces if s.type == "VIEW_3D"],
    }


def reimport(job):
    import bpy
    from afv_addon import builder

    bpy.ops.wm.open_mainfile(filepath=job["blend"])
    scene = bpy.context.scene
    obj = _arms(scene)[job["role"]]
    bone, frame = job["bone"], int(job["frame"])
    track = next(t for t in obj.animation_data.nla_tracks if t.name == builder.TRACK_TWEAKS)
    strip = track.strips[0]
    from bpy_extras import anim_utils

    bag = anim_utils.action_ensure_channelbag_for_slot(strip.action, strip.action_slot)
    q = (math.cos(0.25), math.sin(0.25), 0.0, 0.0)  # 28.6 degrees about the bone's X
    for c in range(4):
        fc = bag.fcurves.new(f'pose.bones["{bone}"].rotation_quaternion', index=c, group_name=bone)
        fc.keyframe_points.insert(frame, q[c])
    solved_before = next(t for t in obj.animation_data.nla_tracks if t.name == builder.TRACK_SOLVED).strips[0]
    before_action = solved_before.action.name
    tweak_action = strip.action.name
    info = builder.reimport_take(scene, job["take2"])
    scene.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    solved = next(t for t in obj.animation_data.nla_tracks if t.name == builder.TRACK_SOLVED).strips[0]
    basis = [c.evaluate(frame) for c in sorted(
        (c for c in _curves(solved) if c.data_path == f'pose.bones["{bone}"].rotation_quaternion'),
        key=lambda c: c.array_index)]
    out = {
        "summary": info,
        "before_action": before_action,
        "tweak_action_before": tweak_action,
        "nla": _nla(obj),
        "solved_action": solved.action.name,
        "old_action_exists": before_action in bpy.data.actions and before_action != solved.action.name,
        "tweak_q": list(q),
        "solved_basis": basis,
        "final_basis": list(ev.pose.bones[bone].rotation_quaternion),
        "bones": [pb.name for pb in ev.pose.bones],
        "heads": [list(ev.matrix_world @ pb.head) for pb in ev.pose.bones],
        "rest_heads": [list(b.head_local) for b in obj.data.bones],
        "frame": frame,
        "proxies": sorted(o.name for o in scene.objects if o.get("afv_kind") == "proxy"),
    }
    if job.get("take3"):  # the user's strip settings survive a longer re-import
        strip = next(t for t in obj.animation_data.nla_tracks if t.name == builder.TRACK_TWEAKS).strips[0]
        strip.name = "my fixes"
        strip.mute = True
        strip.blend_in = 2.0
        strip.frame_start_ui = 3.0  # slid by 3 frames
        set_before = [strip.frame_start, strip.frame_end]
        builder.reimport_take(scene, job["take3"])
        strip = next(t for t in obj.animation_data.nla_tracks if t.name == builder.TRACK_TWEAKS).strips[0]
        out["strip_after"] = {"name": strip.name, "mute": strip.mute, "blend_in": strip.blend_in,
                              "start": strip.frame_start, "end": strip.frame_end, "before": set_before,
                              "action_range": list(strip.action.frame_range),
                              "action_frames": [strip.action_frame_start, strip.action_frame_end],
                              "keys": sum(len(fc.keyframe_points) for fc in _curves(strip)),
                              "strips": len(next(t for t in obj.animation_data.nla_tracks
                                                 if t.name == builder.TRACK_TWEAKS).strips)}
    return out


def addon(job):
    import afv_addon
    import bpy
    from afv_addon import builder

    afv_addon.register()
    try:
        scene = bpy.context.scene
        builder.import_take(job["take"], scene)
        spans = builder.scene_spans(scene)
        visited = []
        scene.frame_set(scene.frame_start)
        for _ in range(len(spans) + 1):
            bpy.ops.afv.issue(direction=1)
            visited.append(scene.frame_current)
        bpy.ops.afv.issue(direction=-1)
        back = scene.frame_current
        res = bpy.ops.afv.reimport()
        return {
            "registered": hasattr(bpy.types, "AFV_PT_panel"),
            "operators": [hasattr(bpy.ops.afv, n) for n in ("import_take", "reimport", "issue", "export")],
            "export_preset": scene.afv.export_preset,
            "span_starts": sorted(s["start"] for s in spans),
            "visited": visited,
            "back": back,
            "reimport": list(res),
        }
    finally:
        afv_addon.unregister()


REVIEW_SCRIPT = (
    "import json,sys\n"
    "print(json.dumps({'event': 'tracking_review', 'message': 'shot 007: 1 hint' + chr(10) + 'id swap at 40',"
    " 'hints': [{'kind': 'swap'}], 'review_video': None}))\n"
    "print(json.dumps({'event': 'result', 'outcome': 'needs_review', 'exports': []}))\n"
    "sys.exit(3)"
)


def _fake_afv(calls, script):
    def argv(args, distro=None, afv=None):
        calls.append(list(args))
        return [sys.executable, "-c", script]
    return argv


def _finish(ops):
    ops.STATE.job.join(60)
    for _ in range(50):
        if ops._poll() is None:
            return
    raise RuntimeError("job did not finish")


def jobs(job):
    import afv_addon
    import bpy
    from afv_addon import builder, ops, wsl

    afv_addon.register()
    ops.STATE.serve_default = False  # slice 1: the one-shot path (the worker: mode "cleanup")
    out, calls, applied = {}, [], []
    registered = True
    try:
        quick = "import json,time; time.sleep(1.5); print(json.dumps({'event': 'take', 'path': 'x'}))"
        wsl.afv_argv = _fake_afv(calls, quick)
        ops.run_afv(["take"], "t1", lambda j: applied.append("t1"))
        out["load"] = {"timer_started": bpy.app.timers.is_registered(ops._poll)}
        bpy.ops.wm.read_homefile(use_empty=True)  # the user opens another file mid-job
        out["load"].update(timer_after_load=bpy.app.timers.is_registered(ops._poll), busy_during=ops.busy())
        _finish(ops)
        out["load"].update(applied=list(applied), status=ops.STATE.status, busy_after=ops.busy(),
                           job_cleared=ops.STATE.job is None)

        wsl.afv_argv = _fake_afv(calls, "import time; time.sleep(60)")
        ops.run_afv(["take"], "t2", lambda j: applied.append("t2"))
        res = bpy.ops.afv.cancel()
        out["cancel"] = {"op": list(res), "job_cleared": ops.STATE.job is None, "busy": ops.busy(),
                         "timer": bpy.app.timers.is_registered(ops._poll), "status": ops.STATE.status}

        wsl.afv_argv = _fake_afv(calls, "print('{}')")
        ops.run_afv(["take"], "t3", lambda j: 1 / 0)
        _finish(ops)
        out["callback_error"] = {"status": ops.STATE.status, "error": ops.STATE.error, "busy": ops.busy()}

        ops.run_afv(["take"], "t5", lambda j: applied.append("t5"))
        _finish(ops)
        out["normal"] = {"applied": "t5" in applied, "busy": ops.busy()}

        def chain(j):
            ops.run_afv(["take"], "c2", lambda j2: applied.append("c2"))

        ops.run_afv(["take"], "c1", chain)
        ops.STATE.job.join(60)
        first = None
        for _ in range(50):
            first = ops._poll()
            if ops.STATE.job is not None and ops.STATE.job.label == "c2":
                break
        out["chain"] = {"poll_returned": first, "timer": bpy.app.timers.is_registered(ops._poll)}
        _finish(ops)
        out["chain"]["applied"] = "c2" in applied

        ops._do_import(job["stale_take"], bpy.context)
        out["stale_import"] = {"level": ops.STATE.level, "error": ops.STATE.error, "status": ops.STATE.status}

        scene = bpy.context.scene
        builder.import_take(job["take"], scene)
        wsl.afv_argv = _fake_afv(calls, REVIEW_SCRIPT)
        bpy.ops.afv.export()
        _finish(ops)
        out["review"] = {"args": calls[-1], "review": ops.STATE.review, "status": ops.STATE.status}
        ok = "import json; print(json.dumps({'event': 'result', 'outcome': 'ok', 'exports': ['007_A.fbx']}))"
        wsl.afv_argv = _fake_afv(calls, ok)
        rv = ops.STATE.review
        builder.import_take(job["take"], scene)  # another copy becomes the active one meanwhile
        scene.afv.export_preset = "godot"  # the panel changed since the review
        bpy.ops.afv.export(yes=True, instance=rv["instance"], reviewed_preset=rv["preset"])
        _finish(ops)
        out["anyway"] = {"args": calls[-1], "status": ops.STATE.status, "error": ops.STATE.error}

        wsl.afv_argv = _fake_afv(calls, "import time; time.sleep(60)")
        ops.run_afv(["take"], "t4", lambda j: applied.append("t4"))
        afv_addon.unregister()
        registered = False
        out["unregister"] = {"job_cleared": ops.STATE.job is None,
                             "timer": bpy.app.timers.is_registered(ops._poll), "busy": ops.busy()}
        out["applied"] = applied
        return out
    finally:
        if registered:
            afv_addon.unregister()


def instances(job):
    import bpy
    from afv_addon import builder

    scene = bpy.context.scene
    a = builder.import_take(job["take"], scene)
    b = builder.import_take(job["take"], scene)
    arms = {(o["afv_instance"], o["afv_role"]): o for o in scene.objects if o.get("afv_kind") == "armature"}

    def solved(inst):
        o = arms[(inst, "A")]
        track = next(t for t in o.animation_data.nla_tracks if t.name == builder.TRACK_SOLVED)
        return track.strips[0].action.name

    before = {i: solved(i) for i in (a["instance"], b["instance"])}
    rec = builder.record_for(scene, arms[(a["instance"], "A")])  # the user selected copy a
    info = builder.reimport_take(scene, job["take2"], rec=rec)
    return {
        "instances": [a["instance"], b["instance"]],
        "records": sorted(r["instance"] for r in builder.instance_records(scene)),
        "record_for": rec["instance"],
        "reimported": info["instance"],
        "cameras": sum(1 for o in scene.objects if o.get("afv_kind") == "camera"),
        "armatures": len(arms),
        "before": before,
        "after": {i: solved(i) for i in (a["instance"], b["instance"])},
        "scene_active": builder.scene_take(scene)["instance"],
    }


def mismatch(job):
    """A .blend copied next to another project's take: nothing resolves, nothing runs."""
    import afv_addon
    import bpy
    from afv_addon import builder, ops, wsl

    bpy.ops.wm.open_mainfile(filepath=job["blend"])
    afv_addon.register()
    try:
        scene = bpy.context.scene
        out = {}
        try:
            builder.resolve_take_json(builder.scene_take(scene))
            out["resolve"] = "found"
        except builder.BuildError as e:
            out["resolve"] = str(e)
        try:
            builder.reimport_take(scene)
            out["reimport"] = "done"
        except builder.BuildError as e:
            out["reimport"] = str(e)
        calls = []
        wsl.afv_argv = _fake_afv(calls, "print('{}')")
        try:
            out["export_op"] = list(bpy.ops.afv.export())
        except RuntimeError as e:  # the operator reported an error
            out["export_op"] = ["CANCELLED", str(e)]
        out["calls"] = calls
        out["job"] = ops.STATE.job is not None
        return out
    finally:
        afv_addon.unregister()


def savecopy(job):
    """Save Copy writes the copy's relative take path and leaves the open file's alone."""
    import afv_addon
    import bpy
    from afv_addon import builder

    afv_addon.register()
    try:
        scene = bpy.context.scene
        builder.import_take(job["take"], scene)
        bpy.ops.wm.save_as_mainfile(filepath=job["a"])
        rel_a = builder.scene_take(scene)["take_json_rel"]
        bpy.ops.wm.save_as_mainfile(filepath=job["b"], copy=True)
        rel_open = builder.scene_take(scene)["take_json_rel"]
        file_open = bpy.data.filepath
    finally:
        afv_addon.unregister()
    bpy.ops.wm.open_mainfile(filepath=job["b"])
    rec_b = builder.scene_take(bpy.context.scene)
    return {"rel_a": rel_a, "rel_open_after_copy": rel_open, "open_file": file_open,
            "rel_in_copy": rec_b["take_json_rel"],
            "resolved_in_copy": builder.resolve_take_json(rec_b)}


def legacy(job):
    """A scene from before instances and project uids still re-imports, and is upgraded."""
    import bpy
    from afv_addon import builder

    bpy.ops.wm.open_mainfile(filepath=job["blend"])
    scene = bpy.context.scene
    rec = builder.scene_take(scene)
    old_id = rec["take_id"].rsplit(":", 1)[0]
    for idb in list(bpy.data.collections):
        if builder.PROP_TAKE in idb:
            del idb[builder.PROP_TAKE]
    for o in scene.objects:
        if builder.OBJ_INSTANCE in o:
            del o[builder.OBJ_INSTANCE]
            o[builder.OBJ_TAKE] = old_id
    old = {k: v for k, v in rec.items() if k not in ("instance", "take_json_rel")}
    old["take_id"] = old_id
    scene[builder.PROP_TAKE] = json.dumps(old)
    info = builder.reimport_take(scene)
    new = builder.scene_take(scene)
    tagged = [(o.get(builder.OBJ_KIND), o.get(builder.OBJ_INSTANCE), o.get(builder.OBJ_TAKE))
              for o in scene.objects if o.get(builder.OBJ_KIND)]
    return {"old_id": old_id, "new_id": new["take_id"], "instance": new.get("instance"),
            "reimported": info["take_id"], "tagged": tagged,
            "collections_with_record": sum(1 for c in bpy.data.collections if builder.PROP_TAKE in c)}


def moved(job):
    import bpy
    from afv_addon import builder

    bpy.ops.wm.open_mainfile(filepath=job["blend"])
    scene = bpy.context.scene
    rec = builder.scene_take(scene)
    resolved = builder.resolve_take_json(rec)
    info = builder.reimport_take(scene)
    cam = scene.camera
    clip = cam.data.background_images[0].clip
    return {"stored": rec["take_json"], "rel": rec.get("take_json_rel"), "resolved": resolved,
            "reimported": info["take_id"], "clip": clip.filepath,
            "clip_exists": os.path.isfile(bpy.path.abspath(clip.filepath)),
            "stored_after": builder.scene_take(scene)["take_json"]}


# --- slice 2: the cleanup tools against the fake afv serve -----------------------------------------
def _run_all(ops, timeout=120.0):
    """Drive the add-on's timer until no job is left (a chain starts its next step from a callback)."""
    import time

    t0 = time.monotonic()
    while ops.STATE.job is not None:
        if time.monotonic() - t0 > timeout:
            raise RuntimeError(f"job {ops.STATE.job.label} did not finish")
        ops.STATE.job.join(0.5)
        ops._poll()


def _solved(obj):
    from afv_addon import builder

    track = next(t for t in obj.animation_data.nla_tracks if t.name == builder.TRACK_SOLVED)
    return track.strips[0].action


def _log_lines(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def _edit(shot_dir, name):
    p = os.path.join(shot_dir, "edits", f"{name}.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def cleanup(job):
    import afv_addon
    import bpy
    from afv_addon import builder, ops, overlay, serve, wsl
    from afv_addon import cleanup as cl

    def fake_serve(cfg):
        serve.serve_argv = lambda distro, afv: [sys.executable, job["fake"], cfg]

    afv_addon.register()
    out = {}
    shot = job["shot_dir"]
    try:
        ops.STATE.serve_default = True
        fake_serve(job["cfg"])
        scene = bpy.context.scene
        ops._do_import(job["take"], bpy.context)
        rec = builder.scene_take(scene)
        out["after_import"] = {"range": [scene.afv.range_start, scene.afv.range_end],
                               "smooth": [scene.afv.smooth_strength, scene.afv.smooth_arms],
                               "prefix": rec.get("prefix"), "tracks": overlay.tracks_for(scene) is not None}
        arm = _arms(scene)["A"]
        vl = bpy.context.view_layer
        vl.objects.active = arm
        arm.select_set(True)
        bpy.ops.object.mode_set(mode="POSE")

        # pins: the unposed pose at a frame is the take's own local rotations
        scene.frame_set(6)
        names = [b.name for b in arm.data.bones]
        pin = cl.pose_pin(arm, rec, "A", 6, 8, names)
        out["pin_all"] = {"bones": names, "rot": [pin["bones"][n] for n in names], "root": pin["root"],
                          "hips": arm.get(builder.OBJ_HIPS)}

        # paint a contact: edit -> run (range) -> take -> motion-only re-import
        cam = scene.camera
        cam_action = cam.animation_data.action.as_pointer()
        tweak = next(t for t in arm.animation_data.nla_tracks if t.name == builder.TRACK_TWEAKS)
        tweak_action = tweak.strips[0].action.name
        arm.data.bones.active = arm.data.bones["LeftFoot"]
        scene.afv.range_start, scene.afv.range_end = 3, 8
        scene.frame_end = 15  # the user shortened the scene: a range re-run still re-imports motion only
        scene.frame_set(5)
        before = _solved(arm).name
        res = list(bpy.ops.afv.paint_contact(state="plant"))
        _run_all(ops)
        out["paint"] = {
            "op": res, "status": ops.STATE.status, "error": ops.STATE.error,
            "contacts": _edit(shot, "contacts"), "solved_before": before, "solved_after": _solved(arm).name,
            "camera_same": cam.animation_data.action.as_pointer() == cam_action,
            "frame": scene.frame_current, "timings": ops.STATE.timings.get("contacts"),
            "tracks": [t.name for t in arm.animation_data.nla_tracks],
            "tweak_action": next(t for t in arm.animation_data.nla_tracks
                                 if t.name == builder.TRACK_TWEAKS).strips[0].action.name == tweak_action,
            "worker": ops.worker_status(),
            "mode": arm.mode,
            "scene_end": scene.frame_end,
            "bones": [pb.name for pb in arm.pose.bones],
            "heads": [list(arm.matrix_world @ pb.head)
                      for pb in arm.evaluated_get(bpy.context.evaluated_depsgraph_get()).pose.bones],
        }

        # pin a posed bone (no key): pins.json holds R basis R^-1 of the displayed pose
        scene.frame_set(10)
        for pbone in arm.pose.bones:
            pbone.select = False  # Blender 5: selection lives on pose bones
        bone = arm.data.bones["LeftForeArm"]
        arm.pose.bones["LeftForeArm"].select = True
        arm.data.bones.active = bone
        pb = arm.pose.bones["LeftForeArm"]
        q = (math.cos(0.3), 0.0, math.sin(0.3), 0.0)
        pb.rotation_quaternion = q
        bpy.context.view_layer.update()
        rest = list(bone.matrix_local.to_quaternion())
        res = list(bpy.ops.afv.pin())
        _run_all(ops)
        out["pin"] = {"op": res, "pins": _edit(shot, "pins"), "rest": rest, "basis": list(q),
                      "status": ops.STATE.status}
        hips = arm.get(builder.OBJ_HIPS)
        arm.pose.bones[hips].select = True
        scene.frame_set(12)
        root_want = list(arm.evaluated_get(bpy.context.evaluated_depsgraph_get()).pose.bones[hips].head)
        bpy.ops.afv.pin()
        _run_all(ops)
        out["pin_root"] = {"pins": _edit(shot, "pins"), "head": root_want}
        for pbone in arm.pose.bones:  # a second pin at frame 12 on another bone: merged
            pbone.select = False
        arm.pose.bones["RightArm"].select = True
        arm.data.bones.active = arm.data.bones["RightArm"]
        bpy.ops.afv.pin()
        _run_all(ops)
        out["pin_merged"] = _edit(shot, "pins")
        bpy.ops.afv.pin_delete(index=0)
        _run_all(ops)
        out["pin_deleted"] = _edit(shot, "pins")

        # smoothing sliders, applied while the tweak strip is in NLA tweak mode
        ad = arm.animation_data
        ttrack = next(t for t in ad.nla_tracks if t.name == builder.TRACK_TWEAKS)
        builder.enter_tweak_mode(arm, (ttrack.name, ttrack.strips[0].name))
        tweak_on = ad.use_tweak_mode
        scene.afv.smooth_strength = 0.5
        scene.afv.smooth_arms = 0.25
        bpy.ops.afv.smooth_apply()
        _run_all(ops)
        out["smooth"] = {"status": ops.STATE.status, "timings": ops.STATE.timings.get("smooth"),
                         "tweak_before": tweak_on, "tweak_after": ad.use_tweak_mode,
                         "tweak_action": ad.action is not None and ad.action == ttrack.strips[0].action}
        ad.use_tweak_mode = False

        # hands off, re-run range (a full-shot stage: full re-import), ID swap
        scene.afv.hands_side = "left"
        bpy.ops.afv.hands(off=True)
        _run_all(ops)
        out["hands"] = _edit(shot, "hands")
        scene.afv.rerun_stage = "pose3d"
        cam_action = cam.animation_data.action.as_pointer()
        bpy.ops.afv.rerun()
        _run_all(ops)
        out["rerun"] = {"camera_rebuilt": cam.animation_data.action.as_pointer() != cam_action,
                        "status": ops.STATE.status}
        scene.frame_set(9)
        cl.swap_ids(bpy.context, builder.scene_take(scene), [1, 2])
        _run_all(ops)
        out["ids"] = {"ids": _edit(shot, "ids"), "status": ops.STATE.status}

        # Revert last fix twice: the swap (ids.json goes), then hands off (hands.json goes)
        from afv_addon import edits as ed

        hist_before = [h["label"] for h in ed.history(shot)]
        res1 = list(bpy.ops.afv.revert())
        _run_all(ops)
        after1 = {"ids": _edit(shot, "ids"), "status": ops.STATE.status}
        res2 = list(bpy.ops.afv.revert())
        _run_all(ops)
        hands_after = _edit(shot, "hands")
        with open(os.path.join(shot, "shot.json"), encoding="utf-8") as fh:
            smooth_set = json.load(fh)["settings"]["params"].get("smooth")
        res3 = list(bpy.ops.afv.revert())  # the smoothing Apply: shot-level settings as before
        _run_all(ops)
        with open(os.path.join(shot, "shot.json"), encoding="utf-8") as fh:
            smooth_reverted = json.load(fh)["settings"]["params"].get("smooth")
        out["revert"] = {"ops": [res1, res2, res3], "history_before": hist_before, "after_swap": after1,
                         "hands": hands_after, "status": ops.STATE.status, "smooth_set": smooth_set,
                         "smooth_reverted": smooth_reverted, "sliders": [scene.afv.smooth_strength,
                                                                         scene.afv.smooth_arms],
                         "history_after": [h["label"] for h in ed.history(shot)],
                         "history_file": os.path.isfile(os.path.join(shot, ".addon", "history.json")),
                         "old_file": os.path.isfile(os.path.join(shot, "edits", ".addon_history.json"))}
        out["log"] = _log_lines(job["log"])

        # a run paused on tracking hints, then "Run anyway"
        ops.shutdown_serve(wait=5)
        fake_serve(job["cfg_review"])
        n_before = len(ed.history(shot))
        bpy.ops.afv.smooth_apply()
        _run_all(ops)
        n_paused = len(ed.history(shot))
        rv = ops.STATE.review
        out["review"] = {"kind": rv and rv.get("kind"), "hints": rv and rv.get("hints"),
                         "status": ops.STATE.status}
        res = list(bpy.ops.afv.go_on())
        _run_all(ops)
        out["go_on"] = {"op": res, "status": ops.STATE.status, "error": ops.STATE.error,
                        "log": _log_lines(job["log_review"]),
                        "history": [n_before, n_paused, len(ed.history(shot))]}

        # a re-solve that fails after its edit: reported, and Revert puts the edit back
        ops.shutdown_serve(wait=5)
        fake_serve(job["cfg_fail"])
        arm.data.bones.active = arm.data.bones["LeftFoot"]
        scene.afv.range_start, scene.afv.range_end = 20, 22
        before_fail = _edit(shot, "contacts")
        bpy.ops.afv.paint_contact(state="free")
        _run_all(ops)
        failed = dict(ops.STATE.failed or {})
        written = _edit(shot, "contacts")
        bpy.ops.afv.revert()
        _run_all(ops)
        out["failed"] = {"failed": failed, "written": written, "before": before_fail,
                         "reverted": _edit(shot, "contacts")}

        # cancel a running request
        ops.shutdown_serve(wait=5)
        fake_serve(job["cfg_slow"])
        bpy.ops.afv.smooth_apply()
        ops.STATE.job.join(1.0)
        ops._poll()
        import time

        time.sleep(0.5)
        bpy.ops.afv.cancel()
        time.sleep(0.5)
        out["cancel"] = {"job": ops.STATE.job is not None, "status": ops.STATE.status,
                         "log": [r["cmd"] for r in _log_lines(job["log_slow"])]}

        # no afv serve (an afv without it): the same tools run as one-shot commands
        ops.shutdown_serve(wait=5)
        serve.serve_argv = lambda distro, afv: [sys.executable, "-c", "import sys; sys.exit(2)"]
        calls = []
        ok = ("import json; print(json.dumps({'event': 'result', 'outcome': 'ok', 'shot': '007', "
              "'stages': [], 'exports': []}))")
        wsl.afv_argv = _fake_afv(calls, ok)
        arm.data.bones.active = arm.data.bones["RightToeBase"]
        scene.afv.range_start, scene.afv.range_end = 15, 18
        bpy.ops.afv.paint_contact(state="free")
        _run_all(ops)
        out["fallback"] = {"calls": calls, "serve_off": ops.STATE.serve_off, "status": ops.STATE.status,
                           "error": ops.STATE.error, "contacts": _edit(shot, "contacts"),
                           "worker": ops.worker_status()}
        client_before = ops.STATE.serve
        bpy.ops.afv.restart_worker()
        out["restart"] = {"serve_off": ops.STATE.serve_off, "client_gone": ops.STATE.serve is None,
                          "had_client": client_before is not None}
        bpy.ops.object.mode_set(mode="OBJECT")
        return out
    finally:
        afv_addon.unregister()


def pins2(job):
    """Pins on other rigs (unity_generic world / extracted, a bone prefix) and with tweak keys."""
    import bpy
    from afv_addon import builder
    from afv_addon import cleanup as cl
    from bpy_extras import anim_utils

    out = {}
    for key, tj in job["takes"].items():
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        builder.import_take(tj, scene)
        rec = builder.scene_take(scene)
        arm = _arms(scene)["A"]
        vl = bpy.context.view_layer
        vl.objects.active = arm
        arm.select_set(True)
        bpy.ops.object.mode_set(mode="POSE")
        f = int(job["frame"])
        scene.frame_set(f)
        names = [b.name for b in arm.data.bones]
        pin = cl.pose_pin(arm, rec, "A", f, 8, names)
        res = {"rec": {k: rec.get(k) for k in ("preset", "prefix", "root_motion")}, "pin": pin,
               "hips": arm.get(builder.OBJ_HIPS), "names": names}
        if job.get("tweak") == key:  # tweak keys (strip slid by 2 frames): the pin is still the solve
            hips = arm.get(builder.OBJ_HIPS)
            fore = next(n for n in names if n.endswith("LeftForeArm"))
            track = next(t for t in arm.animation_data.nla_tracks if t.name == builder.TRACK_TWEAKS)
            strip = track.strips[0]
            strip.frame_start_ui = 2.0
            bag = anim_utils.action_ensure_channelbag_for_slot(strip.action, strip.action_slot)
            q = (math.cos(0.4), 0.0, 0.0, math.sin(0.4))
            for c in range(4):
                fc = bag.fcurves.new(f'pose.bones["{fore}"].rotation_quaternion', index=c, group_name=fore)
                fc.keyframe_points.insert(f - 2, q[c])
            for c, v in enumerate((0.05, -0.03, 0.02)):
                fc = bag.fcurves.new(f'pose.bones["{hips}"].location', index=c, group_name=hips)
                fc.keyframe_points.insert(f - 2, v)
            # a pushed-down layer above it: half influence (blend in), a key with negative w
            arm_bone = next(n for n in names if n.endswith("LeftArm"))
            ad = arm.animation_data
            layer = ad.nla_tracks.new(prev=track)
            layer.name = "my layer"
            act = bpy.data.actions.new("my_layer")
            lslot = act.slots.new(id_type="OBJECT", name=arm.name)
            lbag = anim_utils.action_ensure_channelbag_for_slot(act, lslot)
            qn = (-0.8, 0.0, 0.6, 0.0)  # negative w: Blender's pow turns the long way
            for bone_name in (arm_bone, fore):
                for c in range(4):
                    fc = lbag.fcurves.new(f'pose.bones["{bone_name}"].rotation_quaternion', index=c,
                                          group_name=bone_name)
                    fc.keyframe_points.insert(0, qn[c])
                    fc.keyframe_points.insert(30, qn[c])
            lstrip = layer.strips.new("my layer", f - 2, act)
            lstrip.action_slot = lslot
            lstrip.blend_type = "COMBINE"
            lstrip.extrapolation = "HOLD_FORWARD"
            lstrip.use_sync_length = False
            lstrip.blend_in = 4.0  # influence 0.5 at f
            scene.frame_set(f - 1)
            scene.frame_set(f)
            ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
            head_shown = list(ev.pose.bones[hips].head)
            shown = {n: list(ev.pose.bones[n].matrix_basis.to_quaternion()) for n in (fore, arm_bone)}
            tw = cl.pose_pin(arm, rec, "A", f, 8, [fore, hips, arm_bone])
            ours = {n: list(cl.tweak_basis(arm, n, f)[0]) for n in (fore, arm_bone)}
            for t in ad.nla_tracks:  # Blender's own layers: shown vs the solve alone
                if t.name != builder.TRACK_SOLVED:
                    t.mute = True
            scene.frame_set(f - 1)
            scene.frame_set(f)
            ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
            solved = {n: list(ev.pose.bones[n].matrix_basis.to_quaternion()) for n in (fore, arm_bone)}
            for t in ad.nla_tracks:
                t.mute = False
            lstrip.blend_type = "ADD"
            try:
                cl.pose_pin(arm, rec, "A", f, 8, [fore])
                refused = None
            except cl.edits.EditError as e:
                refused = str(e)
            lstrip.blend_type = "COMBINE"
            res["tweak"] = {"pin": tw, "fore": fore, "arm": arm_bone, "shown_head": head_shown,
                            "shown": shown, "solved": solved, "ours": ours, "refused": refused}
        out[key] = res
    return out


def main():
    job_path = sys.argv[sys.argv.index("--") + 1]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    sys.path.insert(0, job["addon_parent"])
    result = {"ok": False}
    try:
        modes = {"reopen": reopen, "reimport": reimport, "addon": addon, "jobs": jobs, "instances": instances,
                 "moved": moved, "mismatch": mismatch, "savecopy": savecopy, "legacy": legacy,
                 "cleanup": cleanup, "pins2": pins2}
        result["out"] = modes[job["mode"]](job)
        result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"])
    with open(job["result_path"], "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
