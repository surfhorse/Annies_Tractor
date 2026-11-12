"""Re-import exported files in headless Blender and sample them (afv round-trip test).

    blender -b --factory-startup -noaudio --python-exit-code 1 --python afv_inspect.py -- job.json

job.json: {"result_path": .., "files": [{"path": .., "format": "fbx"|"glb"|"bvh", "fps": 30.0,
"npz": output path, "bvh_scale": 0.01}]}

For each file (in an empty scene) the result lists the armatures, their bones (parent, top-level),
every object with its type and parent, the scene fps after import, keyframe counts and frame range
of the armature action, and the armature's object transform. The NPZ gets ``heads`` [F,B,3]: world
head position of every bone (Blender world, Z-up, metres) sampled at each distinct key frame of the
action (``key_frames``, which can be fractional: Blender's FBX importer maps key times to frames
with the *integer* scene fps), with ``bones`` naming the B axis, and ``rots`` [F,B,3,3]: each bone's
world rotation away from its rest pose (pose x rest^-1, conjugated into Blender world), which is
independent of how the importer orients bones.
"""

from __future__ import annotations

import json
import sys
import traceback

import bpy
import numpy as np
from bpy_extras import anim_utils


def import_file(f):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    fps = float(f["fps"])
    scene.render.fps = max(1, round(fps))
    if f["format"] == "bvh":
        scene.render.fps_base = scene.render.fps / fps
    elif f["format"] == "glb":
        # Blender's glTF importer converts seconds to frames with fps * fps_base: set it so that
        # product is the real rate and keys land on integer frames.
        scene.render.fps_base = fps / scene.render.fps
    if f["format"] == "fbx":
        bpy.ops.import_scene.fbx(filepath=f["path"], use_anim=True, anim_offset=0.0,
                                 automatic_bone_orientation=False, ignore_leaf_bones=False,
                                 primary_bone_axis="Y", secondary_bone_axis="X", global_scale=1.0)
    elif f["format"] == "glb":
        bpy.ops.import_scene.gltf(filepath=f["path"], disable_bone_shape=True)
    elif f["format"] == "bvh":
        bpy.ops.import_anim.bvh(filepath=f["path"], global_scale=float(f.get("bvh_scale", 0.01)),
                                frame_start=0, use_fps_scale=False, update_scene_fps=True,
                                update_scene_duration=True, rotate_mode="NATIVE",
                                axis_forward="-Z", axis_up="Y")
    else:
        raise ValueError(f["format"])
    return scene


def _unit_cols(m):
    return m / np.linalg.norm(m, axis=0, keepdims=True)


def action_info(obj):
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return None
    action = ad.action
    bag = anim_utils.action_get_channelbag_for_slot(action, ad.action_slot)
    curves = list(bag.fcurves) if bag is not None else []
    counts = [len(fc.keyframe_points) for fc in curves]
    frames = [kp.co[0] for fc in curves for kp in fc.keyframe_points]
    keys = sorted({round(float(x), 4) for x in frames})
    return {
        "key_frames": keys,
        "name": action.name,
        "fcurves": len(curves),
        "keys_min": min(counts) if counts else 0,
        "keys_max": max(counts) if counts else 0,
        "frame_min": min(frames) if frames else None,
        "frame_max": max(frames) if frames else None,
    }


def inspect(f):
    scene = import_file(f)
    objs = list(scene.objects)
    arms = [o for o in objs if o.type == "ARMATURE"]
    info = {
        "path": f["path"],
        "format": f["format"],
        "fps": scene.render.fps / scene.render.fps_base,
        "objects": [{"name": o.name, "type": o.type, "parent": o.parent.name if o.parent else None}
                    for o in objs],
        "armatures": [],
    }
    for a in arms:
        info["armatures"].append({
            "name": a.name,
            "parent": a.parent.name if a.parent else None,
            "matrix_world": [list(r) for r in a.matrix_world],
            "bones": [{"name": b.name, "parent": b.parent.name if b.parent else None} for b in a.data.bones],
            "top_bones": [b.name for b in a.data.bones if b.parent is None],
            "action": action_info(a),
        })
    if len(arms) == 1 and info["armatures"][0]["action"] is not None:
        a = arms[0]
        keys = info["armatures"][0]["action"]["key_frames"]
        names = [b.name for b in a.data.bones]
        heads = np.zeros((len(keys), len(names), 3))
        rots = np.zeros((len(keys), len(names), 3, 3))
        rest = [_unit_cols(np.array(a.data.bones[n].matrix_local.to_3x3())) for n in names]
        for i, fr in enumerate(keys):
            whole = int(np.floor(fr + 1e-6))
            scene.frame_set(whole, subframe=max(0.0, fr - whole))
            mw = a.matrix_world
            m3 = _unit_cols(np.array(mw.to_3x3()))
            for j, n in enumerate(names):
                pb = a.pose.bones[n]
                heads[i, j] = list(mw @ pb.head)
                rots[i, j] = m3 @ _unit_cols(np.array(pb.matrix.to_3x3())) @ rest[j].T @ m3.T
        np.savez(f["npz"], heads=heads, rots=rots, bones=np.array(names), key_frames=np.array(keys))
        info["scene_fps_int"] = scene.render.fps
    return info


def main():
    job_path = sys.argv[sys.argv.index("--") + 1]
    with open(job_path, encoding="utf-8") as fh:
        spec = json.load(fh)
    result = {"ok": False, "files": [], "blender": bpy.app.version_string}
    try:
        for f in spec["files"]:
            result["files"].append(inspect(f))
        result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"])
    with open(spec["result_path"], "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
