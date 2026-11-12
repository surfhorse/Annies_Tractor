"""Engine presets (spec: Export formats and engine presets). Data only.

Every preset stores the **full** Blender exporter argument set, because Blender's defaults break
game exports (leaf bones added, keys simplified away, all actions exported, ...).

Scale ("Unreal's x100 handled once"): Blender's scene unit is metres for every preset except
Unreal, which builds the Blender scene in centimetres (``unit_scale`` 0.01, i.e. 1 Blender unit =
1 cm and all positions x100) and writes the FBX with ``FBX_SCALE_ALL``, so the file declares cm
(UnitScaleFactor 1) and no node carries a x100 scale. The metre presets with ``FBX_SCALE_ALL``
write metre values with UnitScaleFactor 100 (1 unit = 1 m), so nodes keep scale 1 in Unity/Godot.

Axes: for FBX the armature data stays in afv's Y-up frame under an armature object rotated +90
degrees about X (Z-up Blender scene, character faces -Y); the FBX exporter's conversion (forward
``-Z`` / up ``Y``) cancels that rotation, so the file is Y-up, the character faces +Z with its left
hand on +X (Mixamo's convention), and the "Armature" Null and proxy mesh nodes have identity
transforms (no -90 degree parent for Unity/Godot to compensate; Unreal drops the node anyway).
glTF is +Y up, +Z front by definition (``export_yup``); its data is built Z-up in Blender.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from afv.export.root_motion import ROOT_MOTION_MODES

FORMATS: tuple[str, ...] = ("fbx", "glb", "bvh")

FBX_ARGS: dict[str, Any] = {
    # what goes in the file
    "use_selection": False,
    "use_visible": False,
    "use_active_collection": False,
    "object_types": ["ARMATURE", "MESH"],
    "use_custom_props": False,
    "use_metadata": True,
    "path_mode": "AUTO",
    "embed_textures": False,
    "batch_mode": "OFF",
    # axes and scale
    "axis_forward": "-Z",
    "axis_up": "Y",
    "global_scale": 1.0,
    "apply_unit_scale": True,
    "apply_scale_options": "FBX_SCALE_ALL",
    "use_space_transform": True,
    "bake_space_transform": False,
    # armature
    "add_leaf_bones": False,
    "primary_bone_axis": "Y",
    "secondary_bone_axis": "X",
    "use_armature_deform_only": True,
    "armature_nodetype": "NULL",
    # mesh (proxy only)
    "use_mesh_modifiers": True,
    "mesh_smooth_type": "OFF",
    "use_triangles": False,
    "use_tspace": False,
    # animation
    "bake_anim": True,
    "bake_anim_use_all_bones": True,
    "bake_anim_use_nla_strips": False,
    "bake_anim_use_all_actions": False,
    "bake_anim_force_startend_keying": True,
    "bake_anim_step": 1.0,
    "bake_anim_simplify_factor": 0.0,
}

GLB_ARGS: dict[str, Any] = {
    "export_format": "GLB",
    "use_selection": False,
    "use_visible": False,
    "use_active_collection": False,
    "export_yup": True,
    "export_apply": False,
    "export_extras": False,
    "export_cameras": False,
    "export_lights": False,
    "export_materials": "NONE",
    "export_skins": True,
    "export_def_bones": True,
    "export_leaf_bone": False,
    "export_armature_object_remove": True,
    "export_hierarchy_flatten_bones": False,
    "export_rest_position_armature": True,
    "export_morph": False,
    "export_animations": True,
    "export_animation_mode": "ACTIONS",
    "export_frame_range": True,
    "export_frame_step": 1,
    "export_force_sampling": True,
    "export_optimize_animation_size": False,
    "export_anim_slide_to_zero": False,
    "export_negative_frame": "SLIDE",
    "export_bake_animation": False,
    "export_anim_single_armature": True,
    "export_reset_pose_bones": True,
    "export_current_frame": False,
}


@dataclass(frozen=True)
class EnginePreset:
    name: str
    label: str
    root_bone: bool
    root_motion: str  # default mode
    allowed_root_motion: tuple[str, ...]
    formats: tuple[str, ...]  # default files written
    engine_notes: str
    prefix: str = ""
    unit_scale: float = 1.0  # Blender scene scale_length for FBX: 1.0 = metres, 0.01 = cm (GLB: always m)
    fbx: dict[str, Any] = field(default_factory=lambda: dict(FBX_ARGS))
    glb: dict[str, Any] = field(default_factory=lambda: dict(GLB_ARGS))
    armature_name: str = "Armature"  # Unreal drops an armature node with exactly this name

    def __post_init__(self) -> None:
        if self.root_motion not in self.allowed_root_motion:
            raise ValueError(f"{self.name}: default root motion not allowed")
        bad = set(self.allowed_root_motion) - set(ROOT_MOTION_MODES)
        if bad or ("extracted" in self.allowed_root_motion and not self.root_bone):
            raise ValueError(f"{self.name}: bad root-motion modes {self.allowed_root_motion}")
        if set(self.formats) - set(FORMATS):
            raise ValueError(f"{self.name}: unknown formats {self.formats}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "root_bone": self.root_bone,
            "root_motion": self.root_motion,
            "allowed_root_motion": list(self.allowed_root_motion),
            "formats": list(self.formats),
            "prefix": self.prefix,
            "unit_scale": self.unit_scale,
            "armature_name": self.armature_name,
            "engine_notes": self.engine_notes,
            "fbx": dict(self.fbx),
            "glb": dict(self.glb),
        }


_NO_ROOT = ("world", "in_place")
_WITH_ROOT = ("extracted", "world", "in_place")

PRESETS: dict[str, EnginePreset] = {
    p.name: p
    for p in (
        EnginePreset(
            "unity_humanoid",
            "Unity, Humanoid",
            root_bone=False,
            root_motion="world",
            allowed_root_motion=_NO_ROOT,
            formats=("fbx", "bvh"),
            engine_notes="Mixamo-style rig, no root bone. World motion stays on Hips; choose in-place or "
            "root motion in the clip import settings (Root Transform Position/Rotation, Bake Into Pose).",
        ),
        EnginePreset(
            "unity_generic",
            "Unity, Generic",
            root_bone=True,
            root_motion="extracted",
            allowed_root_motion=_WITH_ROOT,
            formats=("fbx", "bvh"),
            engine_notes="Adds a root bone with horizontal translation and yaw; set it as the Root node.",
        ),
        EnginePreset(
            "unreal",
            "Unreal",
            root_bone=True,
            root_motion="extracted",
            allowed_root_motion=_WITH_ROOT,
            formats=("fbx", "bvh"),
            unit_scale=0.01,
            engine_notes="Adds a root bone at the floor with horizontal translation and yaw; enable root "
            "motion on the animation. Scene built in cm, FBX declares cm (x100 applied once).",
        ),
        EnginePreset(
            "godot",
            "Godot",
            root_bone=True,
            root_motion="extracted",
            allowed_root_motion=_WITH_ROOT,
            formats=("glb", "bvh"),
            engine_notes="Adds a root bone carrying the root motion; point AnimationMixer.root_motion_track "
            "at it.",
        ),
        EnginePreset(
            "blender",
            "Blender",
            root_bone=False,
            root_motion="world",
            allowed_root_motion=_NO_ROOT,
            formats=("fbx", "glb", "bvh"),
            engine_notes="No root bone; world motion on Hips.",
        ),
    )
}


def get_preset(name: str | EnginePreset) -> EnginePreset:
    if isinstance(name, EnginePreset):
        return name
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown export preset {name!r}; known: {sorted(PRESETS)}") from None
