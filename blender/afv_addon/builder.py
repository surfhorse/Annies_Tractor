"""Build an afv take in Blender: both fighters, the solved camera, the proxy video and the QC strip.

Used by the add-on (Import take / Re-import) and by ``headless.py`` (``afv blender``). bpy + numpy
only. The rig and its animation are built exactly as ``blender/afv_export.py`` builds the FBX
export (same arrays, same bone rolls, same pose-basis maths), so the take matches the exports:

- armature data in afv's Y-up frame under an armature object rotated +90 degrees about X (Blender's
  Z-up scene; the floor y = 0 is Blender's ground grid z = 0);
- bone rest orientation is Blender's (Y along the bone, Z towards ``rest_zaxis``); the pose basis is
  derived from the bone's actual ``matrix_local``: basis_b = R_b^-1 (delta_parent^-1 delta_b) R_b and,
  for translating bones, loc_b = R_b^-1 (delta_parent^-1 (head_b - head_parent) - (rest_b - rest_parent)).

Animation layout per fighter (NLA): track "AFV solved" holds the solved motion as one strip (blend
Replace); track "AFV tweaks" above it holds an empty action strip in Combine mode (quaternions
multiply, locations add) for the user's corrections. Keys made outside tweak mode go to a new active
action, which is also combined on top (``action_blend_type``). Re-import swaps only the solved strip's
action and stretches the tweak strip to the new frame range.

Instances: every Import gets its own instance id (objects, the collection and the scene record carry
it), so the same take imported twice is two independent copies; Re-import acts on one instance. The
take.json path is stored absolute and relative to the .blend (``relativize_records`` at save), and
``resolve_take_json`` finds it again after the project or the .blend moved.

The camera: the package gives camera-to-world (looks down -Z, +Y up: Blender's camera convention) in
the bodies' Y-up world; the object gets the same +90 degrees about X. Horizontal sensor fit, 36 mm
sensor, lens = focal_px * 36 / width, principal point via shift_x/shift_y, render size = the shot's.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path

import bpy
import numpy as np
from bpy_extras import anim_utils
from mathutils import Vector

from . import takeio

TRACK_SOLVED = "AFV solved"
TRACK_TWEAKS = "AFV tweaks"
PROP_TAKE = "afv_take"  # scene: JSON record of the take (takeio.TakeData.record)
PROP_SPANS = "afv_spans"  # scene: JSON list of QC spans (markers are their visible form)
OBJ_TAKE = "afv_take_id"
OBJ_ROLE = "afv_role"
OBJ_KIND = "afv_kind"  # armature | proxy | camera
OBJ_INSTANCE = "afv_instance"  # one id per Import
OBJ_HIPS = "afv_hips"  # armature: the rig's hips bone (its head is the pins' "root")
PLANT_PROP = "afv_plant_"


class BuildError(Exception):
    """The take cannot be built or re-imported here; the message is fit to show the user."""


# --- quaternion helpers (wxyz, numpy; identical to afv_export.py) ----------------------------------
def qmul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qrot(q, v):
    u = q[..., 1:]
    w = q[..., :1]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


Q_YUP_TO_ZUP = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])  # +90 deg about X


def yup_to_zup(v):
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)


def continuous(q):
    q = np.array(q, dtype=np.float64)
    for i in range(1, q.shape[0]):
        d = np.sum(q[i] * q[i - 1], axis=-1, keepdims=True)
        q[i] = np.where(d < 0.0, -q[i], q[i])
    return q


# --- small bpy helpers ------------------------------------------------------------------------------
def _view_layer(scene):
    ctx = bpy.context
    return ctx.view_layer if ctx.scene == scene and ctx.view_layer is not None else scene.view_layers[0]


def _ensure_object_mode(scene):
    """Object mode (edit bones need it); returns (active object name, its mode) to restore."""
    vl = _view_layer(scene)
    active = vl.objects.active
    if active is None:
        return None
    before = (active.name, active.mode)
    if active.mode != "OBJECT":
        with bpy.context.temp_override(scene=scene, view_layer=vl, active_object=active, object=active):
            bpy.ops.object.mode_set(mode="OBJECT")
    return before


def _restore_mode(scene, before):
    """Back to Pose Mode on the object that was active (the cleanup tools are used from there)."""
    if before is None or before[1] != "POSE":
        return
    obj = scene.objects.get(before[0])
    if obj is None or obj.type != "ARMATURE":
        return
    vl = _view_layer(scene)
    if obj.hide_get(view_layer=vl):
        return
    vl.objects.active = obj
    obj.select_set(True, view_layer=vl)
    with bpy.context.temp_override(scene=scene, view_layer=vl, active_object=obj, object=obj):
        bpy.ops.object.mode_set(mode="POSE")


def _make_active(obj, scene):
    vl = _view_layer(scene)
    for o in scene.objects:
        if o is not None and o.select_get(view_layer=vl):
            o.select_set(False, view_layer=vl)
    if obj.hide_get(view_layer=vl) or obj.hide_viewport:
        raise BuildError(f"{obj.name} is hidden: unhide it to re-import")
    vl.objects.active = obj
    obj.select_set(True, view_layer=vl)
    return vl


def _new_action(name, id_type, slot_name):
    action = bpy.data.actions.new(name)
    slot = action.slots.new(id_type=id_type, name=slot_name)
    bag = anim_utils.action_ensure_channelbag_for_slot(action, slot)
    return action, slot, bag


def _key(bag, path, index, frames, values, group, interpolation="LINEAR"):
    fc = bag.fcurves.new(path, index=index, group_name=group)
    n = len(frames)
    fc.keyframe_points.add(n)
    co = np.stack([np.asarray(frames, dtype=np.float64), np.asarray(values, dtype=np.float64)], axis=1)
    fc.keyframe_points.foreach_set("co", co.ravel().tolist())
    for kp in fc.keyframe_points:
        kp.interpolation = interpolation
    fc.update()
    return fc


def _set_range(action, frames):
    f0, f1 = float(frames[0]), float(frames[-1])
    action.use_frame_range = True
    action.frame_start = f0
    action.frame_end = max(f1, f0 + 1.0)


def _take_objects(scene, rec):
    """The objects of one imported instance (records without an instance: by take id)."""
    inst = rec.get("instance")
    out = {"armature": {}, "proxy": {}, "camera": None}
    for o in scene.objects:
        if inst is not None:
            if o.get(OBJ_INSTANCE) != inst:
                continue
        elif o.get(OBJ_TAKE) != rec.get("take_id"):
            continue
        kind = o.get(OBJ_KIND)
        if kind == "camera":
            out["camera"] = o
        elif kind in ("armature", "proxy"):
            out[kind][o.get(OBJ_ROLE)] = o
    return out


def fighters(scene, rec=None) -> dict:
    """Role -> armature object of one imported take instance (``rec``: default the scene's active
    take); {} when the scene has no take."""
    rec = rec if rec is not None else scene_take(scene)
    return dict(_take_objects(scene, rec)["armature"]) if rec is not None else {}


def _tag(obj, rec, role, kind):
    obj[OBJ_TAKE] = rec["take_id"]
    obj[OBJ_INSTANCE] = rec["instance"]
    if role is not None:
        obj[OBJ_ROLE] = role
    obj[OBJ_KIND] = kind


# --- fighters ---------------------------------------------------------------------------------------
def _edit_bones(obj, fighter, create, scene):
    data = fighter.arrays
    vl = _make_active(obj, scene)
    with bpy.context.temp_override(scene=scene, view_layer=vl, active_object=obj, object=obj):
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            _set_edit_bones(obj, fighter, create, data)
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")


def _set_edit_bones(obj, fighter, create, data):
    ebs = obj.data.edit_bones
    made = {}
    for i, (n, parent) in enumerate(fighter.bones):
        eb = ebs.new(n) if create else ebs[n]
        eb.head = Vector(data["rest_head"][i])
        eb.tail = Vector(data["rest_tail"][i])
        eb.align_roll(Vector(data["rest_zaxis"][i]))
        if create:
            eb.use_connect = False
            eb.use_deform = True
            eb.use_inherit_rotation = True
            eb.inherit_scale = "FULL"
            eb.use_local_location = True
            if parent is not None:
                eb.parent = made[parent]
            made[n] = eb


def build_armature(fighter, coll, rec, scene):
    arm = bpy.data.armatures.new(fighter.name)
    arm.display_type = "STICK"
    obj = bpy.data.objects.new(fighter.name, arm)
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (math.pi / 2.0, 0.0, 0.0)  # Y-up armature data in the Z-up scene (as the FBX)
    obj.show_in_front = True
    coll.objects.link(obj)
    _tag(obj, rec, fighter.role, "armature")
    if fighter.hips:
        obj[OBJ_HIPS] = fighter.hips
    _edit_bones(obj, fighter, True, scene)
    for pb in obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    return obj


def rest_matches(obj, fighter, tol=1e-5):
    bones = obj.data.bones
    if sorted(b.name for b in bones) != sorted(fighter.bone_names):
        return None  # a different rig
    rh, rt = fighter.arrays["rest_head"], fighter.arrays["rest_tail"]
    for i, n in enumerate(fighter.bone_names):
        b = bones[n]
        head, tail = np.array(b.head_local), np.array(b.tail_local)
        if np.abs(head - rh[i]).max() > tol or np.abs(tail - rt[i]).max() > tol:
            return False
    return True


def bake_solved(obj, fighter, frames, name):
    """The solved motion as a new action (not assigned): returns (action, slot)."""
    data = fighter.arrays
    names = fighter.bone_names
    parents = [p for _, p in fighter.bones]
    pidx = [names.index(p) if p is not None else -1 for p in parents]
    bones = obj.data.bones
    rest_q = np.array([list(bones[n].matrix_local.to_quaternion()) for n in names])  # [B,4] wxyz
    rest_h = np.array([list(bones[n].head_local) for n in names])  # [B,3]
    delta = np.asarray(data["pose_delta"], dtype=np.float64)  # [F,B,4]
    head = np.asarray(data["pose_head"], dtype=np.float64)  # [F,B,3]
    translating = set(fighter.translating)
    action, slot, bag = _new_action(name, "OBJECT", obj.name)
    frames = np.asarray(frames, dtype=np.float64)
    for b, n in enumerate(names):
        pb = obj.pose.bones[n]
        pb.rotation_mode = "QUATERNION"
        p = pidx[b]
        local = delta[:, b] if p < 0 else qmul(qconj(delta[:, p]), delta[:, b])
        basis = qmul(qmul(qconj(rest_q[b]), local), rest_q[b])
        basis = continuous(basis / np.linalg.norm(basis, axis=-1, keepdims=True))
        dpath = f'pose.bones["{n}"]'
        for c in range(4):
            _key(bag, dpath + ".rotation_quaternion", c, frames, basis[:, c], n)
        if n in translating:
            if p < 0:
                arm_off = head[:, b] - rest_h[b]
            else:
                arm_off = qrot(qconj(delta[:, p]), head[:, b] - head[:, p]) - (rest_h[b] - rest_h[p])
            loc = qrot(qconj(rest_q[b])[None], arm_off)
            for c in range(3):
                _key(bag, dpath + ".location", c, frames, loc[:, c], n)
    plant = data.get("plant")
    if plant is not None and plant.size:
        for c, ch in enumerate(fighter.plant_channels):
            prop = PLANT_PROP + ch
            obj[prop] = 0.0
            _key(bag, f'["{prop}"]', 0, frames, plant[:, c].astype(np.float64), "AFV plants", "CONSTANT")
    _set_range(action, frames)
    return action, slot


def _strip(track, action, slot, frames, blend, extrapolation):
    strip = track.strips.new(action.name, int(frames[0]), action)
    strip.action_slot = slot
    strip.blend_type = blend
    strip.extrapolation = extrapolation
    strip.use_sync_length = False
    return strip


def _track(obj, name):
    ad = obj.animation_data
    if ad is None:
        return None
    for t in ad.nla_tracks:
        if t.name == name:
            return t
    return None


def setup_nla(obj, action, slot, frames, name):
    ad = obj.animation_data or obj.animation_data_create()
    ad.action = None
    ad.action_blend_type = "COMBINE"  # keys made outside tweak mode also add on top of the solve
    solved = ad.nla_tracks.new()
    solved.name = TRACK_SOLVED
    _strip(solved, action, slot, frames, "REPLACE", "HOLD")
    tweaks = ad.nla_tracks.new(prev=solved)
    tweaks.name = TRACK_TWEAKS
    tw, tw_slot, _ = _new_action(f"{name}_tweaks", "OBJECT", obj.name)
    _set_range(tw, frames)  # the strip is its user: it is saved, and goes away with the armature
    _strip(tweaks, tw, tw_slot, frames, "COMBINE", "NOTHING")
    return solved, tweaks


def extend_tweaks(obj, frames):
    """Lengthen the tweak action's range (and the strip's view of it) to cover ``frames``. The strip
    itself stays: its name, mute, offset, influence, blending, scale and repeat are the user's."""
    track = _track(obj, TRACK_TWEAKS)
    if track is None or not len(track.strips):
        return
    ad = obj.animation_data
    if ad.use_tweak_mode:
        ad.use_tweak_mode = False
    strip = track.strips[0]
    action = strip.action
    if action is None:
        return
    a0, a1 = (float(v) for v in action.frame_range)
    lo, hi = min(float(frames[0]), a0), max(float(frames[-1]), a1)
    if (lo, hi) == (a0, a1) and (strip.action_frame_start, strip.action_frame_end) == (a0, a1):
        return
    _set_range(action, [lo, hi])
    # widen the strip's action range on the side that grows first; frame_start (offset) stays put
    if hi > strip.action_frame_end:
        strip.action_frame_end = max(hi, lo + 1.0)
    if lo < strip.action_frame_start:
        strip.action_frame_start = lo


def leave_tweak_mode(obj):
    """Leave NLA tweak mode; returns (track, strip) names to go back to (None if it was off)."""
    ad = obj.animation_data
    if ad is None or not ad.use_tweak_mode:
        return None
    track = ad.nla_tracks.active
    strip = next((s for s in track.strips if s.action == ad.action), None) if track is not None else None
    if strip is None and track is not None:
        strip = next((s for s in track.strips if s.select), None)
    ad.use_tweak_mode = False
    return (track.name, strip.name) if track is not None and strip is not None else None


def enter_tweak_mode(obj, where):
    """Back into tweak mode on the strip ``leave_tweak_mode`` reported."""
    ad = obj.animation_data
    if where is None or ad is None or ad.use_tweak_mode:
        return
    track = next((t for t in ad.nla_tracks if t.name == where[0]), None)
    strip = track.strips.get(where[1]) if track is not None else None
    if strip is None:
        return
    for t in ad.nla_tracks:
        t.select = t == track
        for s in t.strips:
            s.select = s == strip
    ad.nla_tracks.active = track
    ad.use_tweak_mode = True


def replace_solved(obj, action, slot, frames):
    track = _track(obj, TRACK_SOLVED)
    if track is None:
        raise BuildError(f"{obj.name} has no '{TRACK_SOLVED}' NLA track; use Import take for a fresh copy")
    old = [s.action for s in track.strips if s.action is not None]
    for s in list(track.strips):
        track.strips.remove(s)
    _strip(track, action, slot, frames, "REPLACE", "HOLD")
    for a in old:
        if a is not action and a.users == 0:
            bpy.data.actions.remove(a)


def build_proxy_mesh(fighter, arm_obj, coll, rec):
    """Octahedron per bone, rigidly skinned (like the export proxy, a bit wider), coloured per fighter."""
    data = fighter.arrays
    names = fighter.bone_names
    verts, faces = [], []
    for i in range(len(names)):
        h = data["rest_head"][i]
        t = data["rest_tail"][i]
        y = t - h
        length = float(np.linalg.norm(y))
        y = y / max(length, 1e-9)
        z = data["rest_zaxis"][i] - np.dot(data["rest_zaxis"][i], y) * y
        z = z / max(float(np.linalg.norm(z)), 1e-9)
        x = np.cross(y, z)
        w = min(max(0.12 * length, 0.008), 0.05)  # chunkier than the export proxy: easier to see
        c = h + 0.2 * (t - h)
        base = len(verts)
        verts += [h, t, c + w * x, c + w * z, c - w * x, c - w * z]
        for a in range(4):
            r0, r1 = base + 2 + a, base + 2 + (a + 1) % 4
            faces += [(base, r1, r0), (base + 1, r0, r1)]
    mesh = bpy.data.meshes.new(fighter.name + "_proxy")
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    mesh.update()
    mat = bpy.data.materials.new(fighter.name + "_proxy")
    mat.diffuse_color = tuple(fighter.color)
    mesh.materials.append(mat)
    mobj = bpy.data.objects.new(fighter.name + "_proxy", mesh)
    mobj.color = tuple(fighter.color)
    coll.objects.link(mobj)
    for i, n in enumerate(names):
        vg = mobj.vertex_groups.new(name=n)
        vg.add(list(range(6 * i, 6 * i + 6)), 1.0, "REPLACE")
    mobj.parent = arm_obj
    mod = mobj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm_obj
    mobj.hide_select = True  # pick bones, not the proxy
    _tag(mobj, rec, fighter.role, "proxy")
    return mobj


def _remove_proxy(mobj):
    mesh = mobj.data
    mats = list(mesh.materials)
    bpy.data.objects.remove(mobj)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    for m in mats:
        if m is not None and m.users == 0:
            bpy.data.materials.remove(m)


# --- camera and video -------------------------------------------------------------------------------
def camera_world(take):
    """Blender-world camera per frame: location [F,3], rotation [F,4] wxyz, lens mm [F]."""
    cam = take.camera
    loc = yup_to_zup(cam["pos"])
    rot = continuous(qmul(Q_YUP_TO_ZUP, np.asarray(cam["rot"], dtype=np.float64)))
    lens = np.asarray(cam["focal_px"], dtype=np.float64) * float(cam["sensor_width_mm"]) / float(cam["width"])
    return loc, rot, lens


def key_camera(obj, take):
    cam = take.camera
    frames = take.frames
    data = obj.data
    w, h = float(cam["width"]), float(cam["height"])
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = float(cam["sensor_width_mm"])
    data.shift_x = -(float(cam["cx"]) - w / 2.0) / w
    data.shift_y = (float(cam["cy"]) - h / 2.0) / w
    loc, rot, lens = camera_world(take)
    depth = np.linalg.norm(loc, axis=-1).max() if len(loc) else 10.0
    data.clip_start = 0.05
    data.clip_end = float(max(200.0, 4.0 * depth))
    for idb in (obj, data):
        ad = idb.animation_data
        if ad is not None and ad.action is not None:
            old = ad.action
            ad.action = None
            if old.users == 0:
                bpy.data.actions.remove(old)
    name = f"{take.shot_id}_camera"
    action, slot, bag = _new_action(name, "OBJECT", obj.name)
    obj.rotation_mode = "QUATERNION"
    for c in range(3):
        _key(bag, "location", c, frames, loc[:, c], "Object Transforms")
    for c in range(4):
        _key(bag, "rotation_quaternion", c, frames, rot[:, c], "Object Transforms")
    _set_range(action, frames)
    obj.animation_data_create()
    obj.animation_data.action = action
    obj.animation_data.action_slot = slot
    lens_action, lens_slot, lbag = _new_action(f"{name}_lens", "CAMERA", data.name)
    _key(lbag, "lens", 0, frames, lens, "Lens")
    _set_range(lens_action, frames)
    data.animation_data_create()
    data.animation_data.action = lens_action
    data.animation_data.action_slot = lens_slot


def build_camera(take, coll, rec):
    data = bpy.data.cameras.new(f"{take.shot_id}_camera")
    obj = bpy.data.objects.new(f"{take.shot_id}_camera", data)
    coll.objects.link(obj)
    _tag(obj, rec, None, "camera")
    key_camera(obj, take)
    return obj


def attach_video(cam_obj, take, scene):
    """The proxy as the camera's background (frame i of the proxy = scene frame i)."""
    data = cam_obj.data
    path = take.video_path()
    for bg in list(data.background_images):
        data.background_images.remove(bg)
    if path is None:
        data.show_background_images = False
        return None
    clip = None
    for c in bpy.data.movieclips:
        if Path(bpy.path.abspath(c.filepath)).resolve() == Path(path).resolve():
            clip = c
            break
    if clip is None:
        clip = bpy.data.movieclips.load(path)
    first = int((take.meta.get("video") or {}).get("first_frame", 0))
    clip.frame_start = first  # scene frame at which the clip's first frame shows
    clip.frame_offset = 0
    data.show_background_images = True
    bg = data.background_images.new()
    bg.source = "MOVIE_CLIP"
    bg.clip = clip
    bg.use_camera_clip = False
    bg.clip_user.use_render_undistorted = False
    bg.frame_method = "STRETCH"  # the proxy fills the camera frame exactly (same aspect)
    bg.alpha = 1.0
    bg.display_depth = "BACK"
    scene.active_clip = clip
    return clip


# --- scene ------------------------------------------------------------------------------------------
def set_timing(scene, take, cam_obj):
    fps = take.fps
    scene.render.fps = max(1, round(fps))
    scene.render.fps_base = scene.render.fps / fps
    scene.frame_start = take.frame_start
    scene.frame_end = take.frame_end
    scene.frame_current = take.frame_start
    if cam_obj is not None:
        scene.render.resolution_x = int(take.camera["width"])
        scene.render.resolution_y = int(take.camera["height"])
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = 1.0
        scene.render.pixel_aspect_y = 1.0
        scene.camera = cam_obj


def set_markers(scene, spans):
    """One timeline marker per QC span (at its start); markers of the previous import are replaced."""
    try:
        old = json.loads(scene.get(PROP_SPANS, "[]"))
    except ValueError:
        old = []
    old_keys = {(s["label"][:63], int(s["start"])) for s in old}
    for m in list(scene.timeline_markers):
        if (m.name, m.frame) in old_keys:
            scene.timeline_markers.remove(m)
    for s in spans:
        scene.timeline_markers.new(s["label"][:63], frame=int(s["start"]))
    scene[PROP_SPANS] = json.dumps(spans)


def _loads(raw):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def scene_take(scene):
    """The record of the scene's active take instance (the last imported or re-imported), or None."""
    return _loads(scene.get(PROP_TAKE))


def instance_records(scene):
    """Records of every take instance in ``scene`` (from their collections)."""
    out = []
    for coll in scene.collection.children_recursive:
        rec = _loads(coll.get(PROP_TAKE))
        if rec is not None:
            out.append(rec)
    return out


def record_by_instance(scene, instance):
    for rec in instance_records(scene):
        if rec.get("instance") == instance:
            return rec
    rec = scene_take(scene)
    return rec if rec is not None and rec.get("instance") == instance else None


def record_for(scene, obj=None):
    """The instance ``obj`` belongs to (e.g. the selected armature), else the scene's active one."""
    inst = obj.get(OBJ_INSTANCE) if obj is not None else None
    if inst is not None:
        for rec in instance_records(scene):
            if rec.get("instance") == inst:
                return rec
    return scene_take(scene)


def _collection(scene, rec):
    for coll in scene.collection.children_recursive:
        r = _loads(coll.get(PROP_TAKE))
        if r is not None and r.get("instance") == rec.get("instance"):
            return coll
    return None


def _rel_to_blend(path, blend_path):
    if not blend_path or not path:
        return ""
    try:
        return os.path.relpath(path, os.path.dirname(blend_path))
    except ValueError:  # another drive
        return ""


def _store(scene, coll, rec):
    rec["take_json_rel"] = _rel_to_blend(rec.get("take_json"), bpy.data.filepath)
    raw = json.dumps(rec)
    if coll is not None:
        coll[PROP_TAKE] = raw
    scene[PROP_TAKE] = raw


def relativize_records(blend_path):
    """Store each take.json path relative to ``blend_path`` (call before saving there)."""
    for idb in list(bpy.data.collections) + list(bpy.data.scenes):
        rec = _loads(idb.get(PROP_TAKE))
        if rec is not None:
            rec["take_json_rel"] = _rel_to_blend(rec.get("take_json"), blend_path)
            idb[PROP_TAKE] = json.dumps(rec)


def resolve_take_json(rec):
    """take.json of a record: relative to the .blend first (both moved together), then the absolute
    path, then a take.json next to the .blend."""
    cands = []
    here = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else ""
    if here and rec.get("take_json_rel"):
        cands.append(os.path.join(here, rec["take_json_rel"]))
    if rec.get("take_json"):
        cands.append(rec["take_json"])
    if here:
        cands.append(os.path.join(here, takeio.TAKE_FILE))
    others = []
    for c in dict.fromkeys(os.path.normpath(c) for c in cands):
        try:
            if not os.path.isfile(c):
                continue
            found = takeio.read_json_cached(c).get("take_id", "")
        except (OSError, ValueError):
            continue
        if takeio.ids_match(rec.get("take_id", ""), found):
            return os.path.normpath(c)
        others.append(f"{os.path.normpath(c)} is take {found!r}")
    why = f"; found only other takes: {'; '.join(others)} (a .blend copied next to another project?)" \
        if others else f" (looked for {', '.join(cands)})"
    raise BuildError(f"cannot find take {rec.get('take_id')!r}{why}. Nothing was run; use Import take")


def scene_spans(scene):
    try:
        return json.loads(scene.get(PROP_SPANS, "[]"))
    except ValueError:
        return []


def set_camera_views(cam_obj):
    """Every 3D viewport looks through the take camera (where the video shows)."""
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D" and space.region_3d is not None:
                    space.region_3d.view_perspective = "CAMERA"
                    space.shading.color_type = "OBJECT"


def _build_fighter(f, take, coll, rec, scene):
    obj = build_armature(f, coll, rec, scene)
    action, slot = bake_solved(obj, f, take.frames, f"{f.name}_solved")
    setup_nla(obj, action, slot, take.frames, f.name)
    build_proxy_mesh(f, obj, coll, rec)
    return obj


def import_take(path, scene=None):
    """Build the take at ``path`` (take.json, take folder or shot folder) into ``scene`` as a new
    instance; it becomes the scene's active take."""
    take = takeio.load_take(path)
    scene = scene or bpy.context.scene
    _ensure_object_mode(scene)
    rec = {**take.record(), "instance": uuid.uuid4().hex[:8]}
    coll = bpy.data.collections.new(f"AFV {take.shot_id} [{rec['instance']}]")
    scene.collection.children.link(coll)
    arms = {f.role: _build_fighter(f, take, coll, rec, scene) for f in take.fighters}
    cam_obj = None
    clip = None
    if take.camera is not None:
        cam_obj = build_camera(take, coll, rec)
        clip = attach_video(cam_obj, take, scene)
    set_timing(scene, take, cam_obj)
    set_markers(scene, take.spans)
    _store(scene, coll, rec)
    scene.frame_set(take.frame_start)
    return summary(scene, take, arms, cam_obj, clip, coll, rec)


def reimport_take(scene=None, path=None, rec=None, motion_only=False):
    """Replace only the solved motion (base NLA strip), camera, video and markers of one take
    instance in ``scene`` (``rec``: default the scene's active one) with the package at ``path``
    (default: the one it came from, found again with ``resolve_take_json``). The tweak track and
    everything the user added stay; the tweak strip is stretched to a longer range. A changed rest
    pose (new bone lengths) is applied to the existing armature; a different rig is refused.

    ``motion_only`` (the cleanup tools' re-solves: contacts, smooth, pins, hands): the camera, video
    and scene timing are left alone (the frame current stays too); fighters and QC markers update."""
    scene = scene or bpy.context.scene
    rec = rec or scene_take(scene)
    if rec is None:
        raise BuildError("this scene has no afv take: use Import take")
    take = takeio.load_take(path or resolve_take_json(rec))
    if not takeio.ids_match(rec["take_id"], take.take_id):
        raise BuildError(f"the package is take {take.take_id!r}, the scene holds {rec['take_id']!r}")
    objs = _take_objects(scene, rec)  # found with the old record (an older take id, no instance)
    old_range = (rec.get("frame_start"), rec.get("frame_end"))
    rec = {**rec, **take.record(), "instance": rec.get("instance") or uuid.uuid4().hex[:8]}
    for kind in ("armature", "proxy"):
        for role, o in objs[kind].items():
            _tag(o, rec, role, kind)  # upgrades objects of older imports
    if objs["camera"] is not None:
        _tag(objs["camera"], rec, None, "camera")
    mode = _ensure_object_mode(scene)
    tweaking = {role: leave_tweak_mode(o) for role, o in objs["armature"].items()}
    coll = _collection(scene, rec)
    if coll is None:
        for o in list(objs["armature"].values()) + ([objs["camera"]] if objs["camera"] else []):
            coll = o.users_collection[0] if o.users_collection else None
            break
    if coll is None:
        coll = scene.collection
    arms = {}
    for f in take.fighters:
        obj = objs["armature"].get(f.role)
        if obj is None:  # a fighter the scene does not have yet
            arms[f.role] = _build_fighter(f, take, coll, rec, scene)
            continue
        same = rest_matches(obj, f)
        if same is None:
            raise BuildError(f"{obj.name}: the take's rig has other bones (preset changed?); "
                             "use Import take for a new copy")
        if not same:
            _edit_bones(obj, f, False, scene)
            if objs["proxy"].get(f.role) is not None:
                _remove_proxy(objs["proxy"][f.role])
            build_proxy_mesh(f, obj, coll, rec)
        if f.hips:
            obj[OBJ_HIPS] = f.hips
        action, slot = bake_solved(obj, f, take.frames, f"{f.name}_solved")
        replace_solved(obj, action, slot, take.frames)
        extend_tweaks(obj, take.frames)
        arms[f.role] = obj
    cam_obj = objs["camera"]
    clip = None
    frame = scene.frame_current
    if motion_only and (take.frame_start, take.frame_end) != old_range:
        motion_only = False  # the take's frames changed (not by a range re-run): rebuild timing
    if take.camera is not None and not motion_only:
        if cam_obj is None:
            cam_obj = build_camera(take, coll, rec)
        else:
            key_camera(cam_obj, take)
        clip = attach_video(cam_obj, take, scene)
    if not motion_only:
        set_timing(scene, take, cam_obj)
    set_markers(scene, take.spans)
    _store(scene, coll if coll is not scene.collection else None, rec)
    for role, where in tweaking.items():
        if role in arms:
            enter_tweak_mode(arms[role], where)
    scene.frame_set(frame if motion_only else take.frame_start)
    _restore_mode(scene, mode)
    return summary(scene, take, arms, cam_obj, clip, coll, rec)


def summary(scene, take, arms, cam_obj, clip, coll, rec):
    out = {
        "take_id": take.take_id,
        "instance": rec["instance"],
        "stale": takeio.pipeline_warnings(take),
        "stale_summary": takeio.stale_summary(take),
        "collection": coll.name if coll is not None else None,
        "fps": scene.render.fps / scene.render.fps_base,
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "markers": len(scene.timeline_markers),
        "spans": len(take.spans),
        "fighters": {},
        "camera": None,
    }
    for role, obj in arms.items():
        ad = obj.animation_data
        tracks = []
        for t in ad.nla_tracks:
            tracks.append({"name": t.name, "strips": [
                {"action": s.action.name if s.action else None, "blend": s.blend_type,
                 "start": s.frame_start, "end": s.frame_end} for s in t.strips]})
        out["fighters"][role] = {"object": obj.name, "bones": len(obj.data.bones), "tracks": tracks,
                                 "active_action": ad.action.name if ad.action else None,
                                 "action_blend_type": ad.action_blend_type}
    if cam_obj is not None:
        out["camera"] = {"object": cam_obj.name, "lens0": cam_obj.data.lens,
                         "video": bpy.path.abspath(clip.filepath) if clip else None,
                         "clip_frame_start": clip.frame_start if clip else None,
                         "clip_frames": clip.frame_duration if clip else None,
                         "clip_size": list(clip.size) if clip else None}
    return out
