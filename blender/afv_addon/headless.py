"""Headless take builder (``afv blender``); run by afv through ``afv.export.blender.run_blender``:

    blender -b --factory-startup -noaudio --python-exit-code 1 --python headless.py -- job.json

job.json: {"result_path": .., "take": take.json path, "out_blend": .blend path or null,
"sample": null or {"npz": path, "frames": [scene frames] or null (all)}}

Builds the take into an empty file with the same builder the add-on uses and saves it (paths made
relative to the .blend). With ``sample`` it also writes, per fighter, every bone's evaluated world
head (Blender world, Z-up, metres; ``<role>/heads`` [S,B,3], ``<role>/bones`` names) and its pixel
position through the scene camera with ``bpy_extras.object_utils.world_to_camera_view`` and the
scene's render size (``<role>/pixels`` [S,B,2], x right, y down), plus ``frames`` and ``lens``.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so ``afv_addon`` imports as a package

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view

from afv_addon import builder


def sample(scene, frames, npz):
    objs = builder._take_objects(scene, builder.scene_take(scene))
    cam = scene.camera
    w = scene.render.resolution_x * scene.render.resolution_percentage / 100.0
    h = scene.render.resolution_y * scene.render.resolution_percentage / 100.0
    out = {"frames": np.asarray(frames, dtype=np.int64)}
    heads = {r: [] for r in objs["armature"]}
    pixels = {r: [] for r in objs["armature"]}
    lens = []
    for fr in frames:
        scene.frame_set(int(fr))
        dg = bpy.context.evaluated_depsgraph_get()
        cam_eval = cam.evaluated_get(dg) if cam is not None else None
        if cam_eval is not None:
            lens.append(cam_eval.data.lens)
        for role, obj in objs["armature"].items():
            ev = obj.evaluated_get(dg)
            mw = ev.matrix_world
            hs, px = [], []
            for pb in ev.pose.bones:
                co = mw @ pb.head
                hs.append(list(co))
                if cam_eval is not None:
                    ndc = world_to_camera_view(scene, cam_eval, co)
                    px.append([ndc.x * w, (1.0 - ndc.y) * h] if ndc.z > 0 else [np.nan, np.nan])
            heads[role].append(hs)
            pixels[role].append(px)
    for role, obj in objs["armature"].items():
        out[f"{role}/heads"] = np.asarray(heads[role])
        out[f"{role}/bones"] = np.array([pb.name for pb in obj.pose.bones])
        if cam is not None:
            out[f"{role}/pixels"] = np.asarray(pixels[role])
    out["lens"] = np.asarray(lens)
    np.savez(npz, **out)


def main():
    job_path = sys.argv[sys.argv.index("--") + 1]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    result = {"ok": False, "blender": bpy.app.version_string}
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        info = builder.import_take(job["take"], scene)
        result["summary"] = info
        cam = scene.camera
        if cam is not None:
            builder.set_camera_views(cam)
        if job.get("sample"):
            s = job["sample"]
            frames = s.get("frames")
            if frames is None:
                frames = list(range(scene.frame_start, scene.frame_end + 1))
            sample(scene, frames, s["npz"])
            scene.frame_set(scene.frame_start)
        out = job.get("out_blend")
        if out:
            bpy.context.preferences.filepaths.save_version = 0  # no .blend1 backups
            builder.relativize_records(out)  # the take.json path relative to the .blend too
            bpy.ops.wm.save_as_mainfile(filepath=out, check_existing=False)
            bpy.ops.file.make_paths_relative()
            bpy.ops.wm.save_mainfile()
            result["blend"] = out
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
