"""``export`` stage: finished BodyMotion -> one file set per fighter on the export rig.

Files go to ``ctx.work_dir/exports/`` (the runner publishes them to the shot's ``exports/``):
``<name>_<role>.fbx|.glb|.bvh`` plus ``<name>_<role>.json`` metadata with provenance (source path,
sha256, title/url, shot start/duration from ``afv.core.project.shot_context``). The result lists them
by role and format, relative to the stage folder. Blender's job files go to ``tmp/`` (dropped).

Input: the ``hands_face`` result (the last motion stage; ``smooth`` is accepted when called directly).

Assumptions:
- the floor is y = 0 (the contacts stage guarantees it); the extracted root sits on it;
- retiming: the motion (30 Hz shot timeline) is resampled with ``afv.motion.retime`` to ``fps``
  (None = the motion's rate), re-pinning planted feet with the contacts stage's ``ContactParams``
  (``inherit_params``). The shot's ``speed_factor`` (injected by the runner from shot.json,
  ``shot_fields``) describes the FOOTAGE: its speed relative to real time (0.25 = a 4x slow-motion
  replay, 1.25 = undercranked/fast footage). Export plays it back at real time, so retime gets the
  playback multiplier ``1 / speed_factor`` (a 0.25 replay becomes ~4x shorter). At the motion's rate
  and speed 1 nothing is resampled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

from afv.core.project import ProjectError, shot_context
from afv.core.skeletons import get_skeleton
from afv.core.stage import Ctx, StageError, register
from afv.core.types import BodyMotion, LicenceRecord, Result, Take, register_result_kind
from afv.export import blender as bl
from afv.export.exporter import ExportRequest, export_takes
from afv.export.presets import PRESETS, get_preset
from afv.motion.contacts import ContactParams
from afv.motion.retime import retime


@register_result_kind
@dataclass
class ExportManifest(Result):
    """Result of the export stage: the files written per role (paths relative to the stage folder)."""

    kind: ClassVar[str] = "export"
    preset: str = ""
    root_motion: str = ""
    fps: float = 0.0
    files: dict[str, dict[str, str]] = field(default_factory=dict)  # role -> format -> relative path
    rig_bones: list[str] = field(default_factory=list)


MAX_EXPORT_FPS = 240.0


class ExportParams(BaseModel):
    preset: str = "unity_humanoid"
    formats: list[Literal["fbx", "glb", "bvh"]] | None = None  # None: the preset's formats
    root_motion: Literal["in_place", "world", "extracted"] | None = None  # None: the preset's default
    prefix: str | None = None  # bone-name prefix; None: the preset's ("" = plain Mixamo names)
    # export rate; None = the motion's (30 Hz shot timeline)
    fps: float | None = Field(None, gt=0.0, le=MAX_EXPORT_FPS, allow_inf_nan=False)
    # the footage's speed vs real time (shot.json, injected by the runner); 0.25 = 4x slow-motion replay
    speed_factor: float = Field(1.0, gt=0.0, le=100.0, allow_inf_nan=False)
    hold_contacts: bool = True  # re-pin planted feet after resampling
    contact: ContactParams = Field(default_factory=ContactParams)  # = the contacts stage's (inherited)
    fingers: bool = True
    root_smooth_s: float = Field(0.2, ge=0.0)  # ground-track filter of the extracted root
    root_yaw_smooth_s: float = Field(0.5, ge=0.0)  # heading filter of the extracted root
    name: str = ""  # file stem; "" = shot folder name. Files are <name>_<role>.<ext>
    roles: list[str] | None = None  # None: every actor in the motion
    proxy_mesh: bool = True

    @field_validator("preset")
    @classmethod
    def _known_preset(cls, v: str) -> str:
        if v not in PRESETS:
            raise ValueError(f"unknown export preset {v!r}; choose one of {sorted(PRESETS)}")
        return v


def _provenance(ctx: Ctx) -> dict[str, Any]:
    try:
        sc = shot_context(ctx.shot_dir)
    except (ProjectError, OSError, ValueError) as e:  # not inside an afv project (tests, ad hoc use)
        return {"shot_dir": str(ctx.shot_dir), "note": f"no shot context: {e}"}
    return {
        "shot_id": sc.shot_id,
        "shot_dir": sc.shot_dir,
        "source_path": sc.source_path,
        "source_sha256": sc.source_sha256,
        "source_title": sc.source_title,
        "source_url": sc.source_url,
        "start": sc.start,
        "duration": sc.duration,
        "speed_factor": sc.speed_factor,
    }


@register("default")
class ExportStage:
    name: ClassVar[str] = "export"
    # 3: hands_face input, retime wired; 4: speed_factor = footage speed; 5: retime output changed
    # 6: retime re-pins with the contacts v4 algorithm; 7: ... with contacts v5 (pivoting toe plants)
    version: ClassVar[str] = "7"
    env: ClassVar[str] = "cpu"
    inputs: ClassVar[tuple[str, ...]] = ("hands_face",)
    shot_fields: ClassVar[tuple[str, ...]] = ("speed_factor",)
    inherit_params: ClassVar[dict[str, str]] = {"contact": "contacts"}
    licences: ClassVar[tuple[LicenceRecord, ...]] = ()
    Params: ClassVar[type[BaseModel]] = ExportParams
    weight_hash: ClassVar[str] = ""

    def run(self, ctx: Ctx, inputs: dict[str, Result], params: BaseModel) -> Result:
        assert isinstance(params, ExportParams)
        motion = inputs.get("hands_face", inputs.get("smooth"))
        if not isinstance(motion, BodyMotion):
            raise StageError("export needs the hands_face (or smooth) stage's BodyMotion")
        if params.preset not in PRESETS:
            raise StageError(f"unknown export preset {params.preset!r}; choose one of {sorted(PRESETS)}")
        src_fps = float(motion.fps or 30.0)
        fps = float(params.fps) if params.fps is not None else src_fps
        speed = float(params.speed_factor)  # footage speed vs real time
        playback = 1.0 / speed  # retime's multiplier: a 0.25x replay plays 4x faster (real time)
        retimed = not (math.isclose(fps, src_fps, rel_tol=1e-9) and speed == 1.0)
        if retimed:
            ctx.progress(-1.0, f"retiming {src_fps:g} -> {fps:g} fps, footage speed x{speed:g} -> real time")
            try:
                motion = retime(
                    motion, fps, playback, hold_contacts=params.hold_contacts, contact_params=params.contact
                )
            except (KeyError, ValueError, ArithmeticError, MemoryError) as e:
                raise StageError(f"export: retiming failed: {type(e).__name__}: {e}") from e
        preset = get_preset(params.preset)
        mode = params.root_motion or preset.root_motion
        if mode not in preset.allowed_root_motion:
            raise StageError(f"preset {preset.name} does not support root motion {mode!r}")
        roles = params.roles or list(motion.actors)
        missing = [r for r in roles if r not in motion.actors]
        if missing:
            raise StageError(f"no motion for roles {missing}; have {sorted(motion.actors)}")
        stem = params.name or ctx.shot_dir.name
        sk = get_skeleton(motion.skeleton)
        prov = _provenance(ctx)
        prov.update(
            speed_factor=speed, speed_factor_applied=True, playback_multiplier=playback,
            source_fps=src_fps, retimed=retimed,
        )
        if motion.num_frames:
            prov["timestamps"] = [float(motion.timestamps[0]), float(motion.timestamps[-1])]
        requests = []
        for role in roles:
            a = motion.actors[role]
            take = Take(
                skeleton=motion.skeleton, fps=motion.fps, speed_factor=speed, rot=a.rot, root=a.root,
                root_motion=mode, contact_channels=list(motion.contact_channels), contact=a.contact,
                role=role, provenance=dict(prov), rest_offsets=a.rest_offsets,
            )
            requests.append(ExportRequest(take, f"{stem}_{role}", sk))
        ctx.progress(0.1, f"exporting {len(requests)} take(s) with preset {preset.name}")
        out_dir = ctx.work_dir / "exports"
        try:
            out = export_takes(
                requests, out_dir, preset=preset, formats=params.formats, root_motion=mode,
                prefix=params.prefix, fingers=params.fingers, root_smooth_s=params.root_smooth_s,
                root_yaw_smooth_s=params.root_yaw_smooth_s, proxy_mesh=params.proxy_mesh,
                job_root=ctx.work_dir / "tmp",
            )
        except (bl.BlenderError, ValueError, KeyError, OSError) as e:
            raise StageError(f"export failed: {e}") from e
        ctx.progress(1.0, "export done")
        return ExportManifest(
            frames=motion.frames,
            timestamps=motion.timestamps,
            preset=preset.name,
            root_motion=mode,
            fps=float(motion.fps),
            files={r.role: {k: f"exports/{v.name}" for k, v in r.files.items()} for r in out},
            rig_bones=list(out[0].motion.rig.names) if out else [],
        )
