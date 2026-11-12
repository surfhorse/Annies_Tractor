"""Module F: export to game engines (canonical humanoid rig, BVH, FBX/GLB via headless Blender).

The ``export`` stage lives in ``afv.export.stage`` (import it to register the stage).
"""

from afv.export.bvh import bvh_fk, read_bvh, write_bvh
from afv.export.exporter import ExportedTake, ExportRequest, export_take, export_takes, take_motion
from afv.export.mapping import Mapping, get_mapping, register_mapping
from afv.export.presets import PRESETS, EnginePreset, get_preset
from afv.export.retarget import RigMotion, mapped_source_positions, retarget, rig_fk
from afv.export.rig import HUMANOID, ExportRig, build_rig
from afv.export.root_motion import ROOT_MOTION_MODES, apply_root_motion

__all__ = [
    "HUMANOID",
    "PRESETS",
    "ROOT_MOTION_MODES",
    "EnginePreset",
    "ExportRequest",
    "ExportRig",
    "ExportedTake",
    "Mapping",
    "RigMotion",
    "apply_root_motion",
    "build_rig",
    "bvh_fk",
    "export_take",
    "export_takes",
    "get_mapping",
    "get_preset",
    "mapped_source_positions",
    "read_bvh",
    "register_mapping",
    "retarget",
    "rig_fk",
    "take_motion",
    "write_bvh",
]
