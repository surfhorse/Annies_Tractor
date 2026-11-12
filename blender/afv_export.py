"""Blender-side exporter for afv (run headless; do not import afv here).

    blender -b --factory-startup -noaudio --python-exit-code 1 --python afv_export.py -- job.json

job.json::

    {"result_path": "...", "jobs": [{
        "name": clip name (scene and action name), "armature_name": "Armature",
        "npz": path, "fps": 30.0, "frame_count": F,
        "bones": [{"name": .., "parent": .. or null}, ...]   (parents before children),
        "translating": [bone names with animated location],
        "proxy_mesh": true,
        "outputs": [{"format": "fbx"|"glb", "path": .., "unit_scale": 1.0 (scene scale_length),
                     "yup_object": true|false, "args": {exporter keyword arguments}}]}]}

The scene is built once per distinct (``unit_scale``, ``yup_object``): e.g. Unreal FBX in cm, GLB
always in metres because the glTF exporter ignores the scene unit scale.

NPZ arrays (armature space = afv's internal frame: Y-up, faces +Z, left +X; metres; wxyz):
    rest_head [B,3], rest_tail [B,3], rest_zaxis [B,3]   bone rest head/tail and roll reference
    pose_head [F,B,3]                                     head position per frame
    pose_delta [F,B,4]                                    world rotation away from rest per frame

``yup_object`` (FBX): the armature object is rotated +90 degrees about X, so the scene is Blender's
usual Z-up (character faces -Y) while the armature data stays Y-up. The FBX exporter's axis
conversion (forward -Z, up Y) is -90 degrees about X on root objects, which cancels the object
rotation: the FBX "Armature" node (and the proxy mesh) come out with identity transforms and the
bones directly in Y-up. Otherwise (GLB, whose exporter removes the armature object and would drop
its rotation) the data is converted to Z-up and the object stays identity.

Bone rest orientation is Blender's (Y along the bone, Z towards ``rest_zaxis``); the pose basis is
derived from the bone's actual ``matrix_local`` so the evaluated pose reproduces pose_head and
pose_delta exactly: basis_b = R_b^-1 (delta_parent^-1 delta_b) R_b, and for translating bones
loc_b = R_b^-1 (delta_parent^-1 (head_b - head_parent) - (rest_head_b - rest_head_parent)).
"""

from __future__ import annotations

import json
import math
import sys
import traceback

import bpy
import numpy as np
from bpy_extras import anim_utils
from mathutils import Vector

SET_ARGS = {"object_types"}  # exporter arguments that Blender wants as sets


# --- quaternion helpers (wxyz, numpy, broadcasting) ----------------------------------------------
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


Q_YUP_TO_ZUP = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # +90 deg about X, wxyz


def to_zup(data):
    """Arrays re-expressed in Blender's Z-up frame: (x, y, z) -> (x, -z, y), rotations conjugated."""
    def vec(v):
        return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)
    out = {k: vec(data[k]) for k in ("rest_head", "rest_tail", "rest_zaxis", "pose_head")}
    out["pose_delta"] = qmul(qmul(Q_YUP_TO_ZUP, data["pose_delta"]), qconj(Q_YUP_TO_ZUP))
    return out


def continuous(q):
    q = q.copy()
    for i in range(1, q.shape[0]):
        d = np.sum(q[i] * q[i - 1], axis=-1, keepdims=True)
        q[i] = np.where(d < 0.0, -q[i], q[i])
    return q


# --- scene building --------------------------------------------------------------------------------
def reset_scene(job, unit_scale):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = job["name"]
    us = scene.unit_settings
    us.system = "METRIC"
    us.scale_length = float(unit_scale)
    us.length_unit = "CENTIMETERS" if unit_scale < 0.1 else "METERS"
    fps = float(job["fps"])
    scene.render.fps = max(1, round(fps))
    scene.render.fps_base = scene.render.fps / fps
    scene.frame_start = 0
    scene.frame_end = int(job["frame_count"]) - 1
    scene.frame_current = 0
    return scene


def build_armature(job, data, k, yup_object):
    names = [b["name"] for b in job["bones"]]
    parents = [b["parent"] for b in job["bones"]]
    arm_data = bpy.data.armatures.new(job["name"])
    obj = bpy.data.objects.new(job["armature_name"], arm_data)
    if yup_object:
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (math.pi / 2.0, 0.0, 0.0)  # Y-up armature data in the Z-up scene
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = {}
    for i, n in enumerate(names):
        eb = arm_data.edit_bones.new(n)
        eb.head = Vector(data["rest_head"][i] * k)
        eb.tail = Vector(data["rest_tail"][i] * k)
        eb.align_roll(Vector(data["rest_zaxis"][i]))
        eb.use_connect = False
        eb.use_deform = True
        eb.use_inherit_rotation = True
        eb.inherit_scale = "FULL"
        eb.use_local_location = True
        if parents[i] is not None:
            eb.parent = ebs[parents[i]]
        ebs[n] = eb
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def bake_action(job, data, obj, k):
    names = [b["name"] for b in job["bones"]]
    parents = [b["parent"] for b in job["bones"]]
    pidx = [names.index(p) if p is not None else -1 for p in parents]
    bones = obj.data.bones
    rest_q = np.array([list(bones[n].matrix_local.to_quaternion()) for n in names])  # [B,4] wxyz
    rest_h = np.array([list(bones[n].head_local) for n in names])  # [B,3] Blender units
    delta = data["pose_delta"]  # [F,B,4]
    head = data["pose_head"] * k  # [F,B,3]
    n_frames = delta.shape[0]
    translating = set(job["translating"])

    action = bpy.data.actions.new(job["name"])
    obj.animation_data_create()
    slot = action.slots.new(id_type="OBJECT", name=job["armature_name"])
    obj.animation_data.action = action
    obj.animation_data.action_slot = slot
    bag = anim_utils.action_ensure_channelbag_for_slot(action, slot)
    frames = np.arange(n_frames, dtype=np.float64)

    def key(path, index, values, group):
        fc = bag.fcurves.new(path, index=index, group_name=group)
        fc.keyframe_points.add(n_frames)
        fc.keyframe_points.foreach_set("co", np.stack([frames, values], axis=1).ravel().tolist())
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
        fc.update()

    for b, n in enumerate(names):
        pb = obj.pose.bones[n]
        pb.rotation_mode = "QUATERNION"
        p = pidx[b]
        local = delta[:, b] if p < 0 else qmul(qconj(delta[:, p]), delta[:, b])
        basis = qmul(qmul(qconj(rest_q[b]), local), rest_q[b])
        basis = continuous(basis / np.linalg.norm(basis, axis=-1, keepdims=True))
        dpath = f'pose.bones["{n}"]'
        for c in range(4):
            key(dpath + ".rotation_quaternion", c, basis[:, c], n)
        if n in translating:
            if p < 0:
                arm_off = head[:, b] - rest_h[b]
            else:
                arm_off = qrot(qconj(delta[:, p]), head[:, b] - head[:, p]) - (rest_h[b] - rest_h[p])
            loc = qrot(qconj(rest_q[b])[None], arm_off)
            for c in range(3):
                key(dpath + ".location", c, loc[:, c], n)
    return action


def build_proxy_mesh(job, data, obj, k):
    """Small octahedron per bone, rigidly skinned: makes the file a skinned character for engines."""
    names = [b["name"] for b in job["bones"]]
    verts, faces = [], []
    for i in range(len(names)):
        h = data["rest_head"][i] * k
        t = data["rest_tail"][i] * k
        y = t - h
        length = float(np.linalg.norm(y))
        y = y / max(length, 1e-9)
        z = data["rest_zaxis"][i] - np.dot(data["rest_zaxis"][i], y) * y
        z = z / max(float(np.linalg.norm(z)), 1e-9)
        x = np.cross(y, z)
        w = min(max(0.08 * length, 0.004 * k), 0.03 * k)
        c = h + 0.2 * (t - h)
        base = len(verts)
        verts += [h, t, c + w * x, c + w * z, c - w * x, c - w * z]
        for a in range(4):
            r0, r1 = base + 2 + a, base + 2 + (a + 1) % 4
            faces += [(base, r1, r0), (base + 1, r0, r1)]
    mesh = bpy.data.meshes.new(job["name"] + "_proxy")
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    mesh.update()
    mobj = bpy.data.objects.new("Proxy", mesh)
    bpy.context.scene.collection.objects.link(mobj)
    for i, n in enumerate(names):
        vg = mobj.vertex_groups.new(name=n)
        vg.add(list(range(6 * i, 6 * i + 6)), 1.0, "REPLACE")
    mobj.parent = obj
    mod = mobj.modifiers.new("Armature", "ARMATURE")
    mod.object = obj
    return mobj


def export(out, fps):
    args = {k: (set(v) if k in SET_ARGS else v) for k, v in out["args"].items()}
    render = bpy.context.scene.render
    if out["format"] == "fbx":
        res = bpy.ops.export_scene.fbx(filepath=out["path"], **args)
    elif out["format"] == "glb":
        # Blender's glTF exporter converts frames to seconds with fps * fps_base (it should divide),
        # so for fractional rates set fps_base the way it expects; keys are per frame, unaffected.
        saved = render.fps_base
        render.fps_base = fps / render.fps
        try:
            res = bpy.ops.export_scene.gltf(filepath=out["path"], **args)
        finally:
            render.fps_base = saved
    else:
        raise ValueError(f"unknown format {out['format']}")
    if "FINISHED" not in res:
        raise RuntimeError(f"export {out['format']} returned {res}")


def run_job(job):
    yup = dict(np.load(job["npz"]))
    written = []
    groups = sorted({(float(o["unit_scale"]), bool(o["yup_object"])) for o in job["outputs"]})
    for unit_scale, yup_object in groups:
        data = yup if yup_object else to_zup(yup)
        reset_scene(job, unit_scale)
        k = 1.0 / unit_scale  # metres -> Blender units
        obj = build_armature(job, data, k, yup_object)
        bake_action(job, data, obj, k)
        if job.get("proxy_mesh", True):
            build_proxy_mesh(job, data, obj, k)
        bpy.context.scene.frame_set(0)
        for out in job["outputs"]:
            if (float(out["unit_scale"]), bool(out["yup_object"])) == (unit_scale, yup_object):
                export(out, float(job["fps"]))
                written.append(out["path"])
    return {"name": job["name"], "files": written}


def main():
    job_path = sys.argv[sys.argv.index("--") + 1]
    with open(job_path, encoding="utf-8") as f:
        spec = json.load(f)
    result = {"ok": False, "jobs": [], "blender": bpy.app.version_string}
    try:
        for job in spec["jobs"]:
            result["jobs"].append(run_job(job))
        result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"])
    with open(spec["result_path"], "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
