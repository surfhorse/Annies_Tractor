"""Headless retarget (``afv retarget``); run by afv through ``afv.export.blender.run_blender``:

    blender -b --factory-startup -noaudio --python-exit-code 1 --python retarget_headless.py -- job.json

job.json::

    {"result_path": .., "take": take.json, "character": .fbx | .glb | .blend, "map": *.afvmap.json | null,
     "roles": ["A", "B"], "lock_feet": true, "fbx_args": {FBX exporter arguments of the preset},
     "shot_id": "001", "out_dir": folder for the FBX files, "save_map": path | null, "check": false}

Builds the take into an empty file (``builder.import_take``, as ``afv blender``), imports the character
(``retarget_io.import_character``: a .blend's text blocks and scripts are never read, its Python
drivers are muted; ``--factory-startup`` keeps Auto Run Python Scripts off as well), maps it (the map
file, else ``retarget.guess_map``), retargets each role onto the same armature (so all files share one
origin, the take's world) and exports each one as ``<shot>_<role>_<character>.fbx`` (character =
``retarget_io.safe_name`` of the file stem; the action and the FBX clip carry the same name) with the
preset's FBX arguments. With ``check`` every file is re-imported and compared with the retargeted
armature (``retarget_io.check_fbx``).

The result (JSON, always written; exit 0 unless something unexpected crashed)::

    {"ok": true, "target", "bones", "character_name", "map_source": "file" | "guess", "map": {..},
     "saved_map": path | null, "import": {"seconds", "notes"}, "files": {role: path},
     "roles": {"A": {"report", "export", "check"}}}

or ``{"ok": false, "refused": message, "missing": [required bones not mapped], ...what was done}`` for
anything the user can fix (an unreadable file, a bad map, missing bones, an engine refusal).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so ``afv_addon`` imports as a package

import bpy

from afv_addon import builder, retarget_io
from afv_addon import retarget as rt


class Refused(Exception):
    def __init__(self, message, missing=()):
        super().__init__(message)
        self.missing = list(missing)


def _bone_map(job, tgt):
    if job.get("map"):
        return rt.read_map(job["map"]), "file"
    return rt.new_map(rt.guess_map(tgt), target=tgt.name), "guess"


def run(job, out):
    """Fill ``out`` (the result) step by step, so a refusal still reports what was done."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ctx = bpy.context
    scene = ctx.scene
    builder.import_take(job["take"], scene)
    objs = builder.fighters(scene)
    out["roles"], out["files"] = {}, {}
    try:
        imp = retarget_io.import_character(ctx, job["character"])
    except (OSError, RuntimeError) as e:
        raise Refused(f"{job['character']}: cannot import the character: {e}") from None
    tgt = imp["armature"]
    name = retarget_io.safe_name(Path(job["character"]).stem)
    out.update(target=tgt.name, bones=len(tgt.data.bones), objects=[o.name for o in imp["objects"]],
               meshes=len(retarget_io.character_meshes(scene, tgt)), character_name=name,
               **{"import": {"seconds": imp["seconds"], "notes": imp["notes"]}})
    data, source = _bone_map(job, tgt)
    out.update(map=data, map_source=source, saved_map=None)
    if job.get("save_map"):
        rt.write_map(job["save_map"], data)
        out["saved_map"] = job["save_map"]
    absent = sorted(c for c, b in data["bones"].items() if b not in tgt.data.bones)
    if absent:
        raise Refused(f"the map names bones that {tgt.name!r} does not have: "
                      + ", ".join(f"{c} -> {data['bones'][c]}" for c in absent))
    missing = rt.missing_required(data)
    if missing:
        raise Refused(f"required bones not mapped on {tgt.name!r}: {', '.join(missing)}", missing)
    for role in job["roles"]:
        src = objs.get(role)
        if src is None:
            raise Refused(f"the take has no fighter {role} (fighters: {', '.join(sorted(objs))})")
        stem = f"{job['shot_id']}_{role}_{name}"
        rep = rt.retarget(scene, src, tgt, data, lock_feet=bool(job.get("lock_feet", True)), action_name=stem)
        frames = tuple(rep["frames"])
        exp = retarget_io.export_fbx(ctx, tgt, os.path.join(job["out_dir"], f"{stem}.fbx"), job["fbx_args"],
                                     frames)
        res = {"report": rep, "export": exp}
        if job.get("check"):
            t = time.perf_counter()
            res["check"] = retarget_io.check_fbx(ctx, tgt, exp["path"], frames, scale=rep["hips_scale"])
            res["check"]["seconds"] = round(time.perf_counter() - t, 3)
        out["roles"][role] = res
        out["files"][role] = exp["path"]
    out["fps"] = scene.render.fps / scene.render.fps_base
    out["ok"] = True


def main():
    job_path = sys.argv[sys.argv.index("--") + 1]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    result = {"ok": False, "blender": bpy.app.version_string}
    code = 0
    try:
        run(job, result)
    except Refused as e:
        result.update(refused=str(e), missing=e.missing)
    except (ValueError, retarget_io.CharacterError) as e:  # the engine's / importer's "fix this" refusals
        result.update(refused=str(e), missing=[])
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"])
        code = 1
    with open(job["result_path"], "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
