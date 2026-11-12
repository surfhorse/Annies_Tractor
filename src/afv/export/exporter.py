"""Export takes to BVH / FBX / GLB on the canonical humanoid rig.

``export_takes`` handles any number of fighters in one Blender run; each fighter gets its own files
(``<name>.fbx`` / ``.glb`` / ``.bvh`` + ``<name>.json`` metadata). Positions are never re-centred,
so attacker and defender files share one world origin and line up when loaded together.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from afv.core.skeletons import get_skeleton
from afv.core.types import Skeleton, Take
from afv.export import blender as bl
from afv.export.bvh import write_bvh
from afv.export.mapping import Mapping
from afv.export.presets import FORMATS, EnginePreset, get_preset
from afv.export.retarget import RigMotion, retarget, rig_fk
from afv.export.rig import build_rig
from afv.export.root_motion import apply_root_motion
from afv.motion.rotations import xyzw_to_wxyz

FORWARD = np.array([0.0, 0.0, 1.0])  # character forward (internal frame = Blender armature space)
UP = np.array([0.0, 1.0, 0.0])


@dataclass
class ExportRequest:
    """One take to export as ``<name>.*``; the fighter's bone lengths come from ``take.rest_offsets``."""

    take: Take
    name: str
    skeleton: Skeleton | None = None  # default: get_skeleton(take.skeleton)


@dataclass
class ExportedTake:
    name: str
    role: str
    files: dict[str, Path]
    motion: RigMotion = field(repr=False)
    metadata: dict = field(default_factory=dict, repr=False)


def take_motion(
    take: Take,
    preset: EnginePreset,
    *,
    skeleton: Skeleton | None = None,
    mapping: Mapping | None = None,
    root_motion: str | None = None,
    prefix: str | None = None,
    fingers: bool = True,
    root_smooth_s: float = 0.2,
    root_yaw_smooth_s: float = 0.5,
) -> RigMotion:
    """Build the fighter's rig (own bone lengths: ``take.rest_offsets``) and the export-ready motion."""
    sk = skeleton if skeleton is not None else get_skeleton(take.skeleton)
    rest_offsets = take.rest_offsets
    mode = root_motion or take.root_motion
    if mode not in preset.allowed_root_motion:
        raise ValueError(
            f"preset {preset.name} does not allow root motion {mode!r} "
            f"(allowed: {preset.allowed_root_motion})"
        )
    rig = build_rig(
        sk, mapping, rest_offsets=rest_offsets, prefix=preset.prefix if prefix is None else prefix,
        root_bone=preset.root_bone, fingers=fingers,
    )
    m = retarget(rig, np.asarray(take.rot), np.asarray(take.root), take.fps)
    return apply_root_motion(m, mode, smooth_s=root_smooth_s, yaw_smooth_s=root_yaw_smooth_s)


def blender_arrays(m: RigMotion) -> dict[str, np.ndarray]:
    """Arrays for ``blender/afv_export.py``: armature space (afv's Y-up frame), metres, wxyz.

    The Blender armature object carries the Y-up -> Z-up rotation (+90 degrees about X), so the data
    itself is not converted.
    """
    rig = m.rig
    pos, wrot = rig_fk(m)
    head = rig.rest_positions()
    tail = rig.tails.copy()
    y = tail - head
    y /= np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-12)
    ref = np.where(np.abs(y @ FORWARD)[:, None] > 0.9, UP, FORWARD)  # bone Z axis reference
    z = ref - np.sum(ref * y, axis=-1, keepdims=True) * y
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    return {
        "rest_head": head,
        "rest_tail": tail,
        "rest_zaxis": z,
        "pose_head": pos,
        "pose_delta": xyzw_to_wxyz(wrot),
    }


def _metadata(req: ExportRequest, m: RigMotion, preset: EnginePreset, files: dict[str, Path]) -> dict:
    t = req.take
    return {
        "name": req.name,
        "role": t.role,
        "preset": preset.name,
        "root_motion": m.root_motion,
        "fps": m.fps,
        "frame_count": m.num_frames,
        "speed_factor": t.speed_factor,
        "source_skeleton": t.skeleton,
        "files": {k: v.name for k, v in files.items()},
        "rig": m.rig.describe(),
        "provenance": t.provenance,
        "axes": "Y-up, faces +Z, left +X, metres (FBX/GLB/BVH files per preset; BVH in cm)",
    }


def export_takes(
    requests: Sequence[ExportRequest],
    out_dir: Path,
    *,
    preset: str | EnginePreset = "unity_humanoid",
    formats: Sequence[str] | None = None,
    root_motion: str | None = None,
    prefix: str | None = None,
    fingers: bool = True,
    root_smooth_s: float = 0.2,
    root_yaw_smooth_s: float = 0.5,
    mapping: Mapping | None = None,
    proxy_mesh: bool = True,
    bvh_scale: float = 100.0,
    blender: Path | None = None,
    keep_job: bool = False,
    job_root: Path | None = None,
) -> list[ExportedTake]:
    """Export each request's take; FBX/GLB for all of them in a single headless Blender run.

    Blender's job files go to a unique ``_blender_job_*`` folder under ``job_root`` (default
    ``out_dir``), removed on success unless ``keep_job``.
    """
    p = get_preset(preset)
    fmts = tuple(formats) if formats is not None else p.formats
    if bad := set(fmts) - set(FORMATS):
        raise ValueError(f"unknown formats {sorted(bad)}; expected {FORMATS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [r.name for r in requests]
    if len(set(names)) != len(names):
        raise ValueError(f"export names must be unique: {names}")

    results: list[ExportedTake] = []
    blender_jobs: list[dict] = []
    job_dir = out_dir
    need_blender = any(f in fmts for f in ("fbx", "glb"))
    exe = None
    if need_blender:
        exe = blender or bl.find_blender()
        if exe is None:
            raise bl.BlenderError("FBX/GLB export needs Blender; set AFV_BLENDER or install Blender 5.2")
        root = Path(job_root) if job_root is not None else out_dir
        root.mkdir(parents=True, exist_ok=True)
        job_dir = Path(tempfile.mkdtemp(prefix="_blender_job_", dir=root))  # unique per run

    for req in requests:
        m = take_motion(
            req.take, p, skeleton=req.skeleton, mapping=mapping, root_motion=root_motion, prefix=prefix,
            fingers=fingers, root_smooth_s=root_smooth_s, root_yaw_smooth_s=root_yaw_smooth_s,
        )
        files: dict[str, Path] = {}
        if "bvh" in fmts:
            files["bvh"] = write_bvh(m, out_dir / f"{req.name}.bvh", scale=bvh_scale)
        outputs = []
        for f in ("fbx", "glb"):
            if f in fmts:
                files[f] = out_dir / f"{req.name}.{f}"
                args = p.fbx if f == "fbx" else p.glb
                # glTF is metres by definition (its exporter ignores the scene unit scale)
                unit = p.unit_scale if f == "fbx" else 1.0
                # FBX: Y-up data under a +90 X object, so the FBX Armature node is identity;
                # GLB: Z-up data (its exporter removes the armature object and its rotation)
                outputs.append({"format": f, "path": bl.host_path(files[f], exe), "unit_scale": unit,
                                "yup_object": f == "fbx", "args": args})
        if outputs:
            npz = job_dir / f"{req.name}.npz"
            np.savez(npz, **blender_arrays(m))
            rig = m.rig
            blender_jobs.append({
                "name": req.name,
                "armature_name": p.armature_name,
                "npz": bl.host_path(npz, exe),
                "fps": m.fps,
                "frame_count": m.num_frames,
                "bones": [
                    {"name": n, "parent": rig.names[q] if q >= 0 else None}
                    for n, q in zip(rig.names, rig.parents, strict=True)
                ],
                "translating": [rig.names[b] for b in m.translating],
                "proxy_mesh": proxy_mesh,
                "outputs": outputs,
            })
        results.append(ExportedTake(req.name, req.take.role, files, m))

    if blender_jobs:
        bl.run_blender("afv_export.py", {"jobs": blender_jobs}, job_dir, blender=exe)
        for r in results:
            for f in ("fbx", "glb"):
                if f in r.files and not r.files[f].is_file():
                    raise bl.BlenderError(f"Blender reported success but {r.files[f]} is missing")
    if need_blender and not keep_job:
        shutil.rmtree(job_dir, ignore_errors=True)

    for req, r in zip(requests, results, strict=True):
        r.metadata = _metadata(req, r.motion, p, r.files)
        meta_path = out_dir / f"{req.name}.json"
        meta_path.write_text(json.dumps(r.metadata, indent=1))
        r.files["json"] = meta_path
    return results


def export_take(take: Take, out_dir: Path, name: str, **kwargs) -> ExportedTake:
    """Export one take (see ``export_takes`` for the keyword arguments)."""
    return export_takes([ExportRequest(take, name)], out_dir, **kwargs)[0]
