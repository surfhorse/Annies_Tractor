"""Shared contract: typed stage results, takes and licence records.

Conventions (spec: Data model / Export):
- World frame is right-handed, Y-up, metres, seconds.
- Quaternions are stored as **xyzw** (scalar last). Blender uses wxyz; convert only via
  ``afv.motion.rotations.xyzw_to_wxyz`` / ``wxyz_to_xyzw``.
- Rotations in motion records are **local** (relative to the parent joint), applied on top of the
  skeleton's rest pose; ``root`` is the world translation of the skeleton's root joint (pelvis/hips).
- Shot timeline: every per-frame array of a shot is on GEM-X's normalised **30 Hz** timeline for
  that shot (P0: GEM-X resamples to 30 fps itself). ``frames`` holds frame indices on that timeline
  (int64, 0-based) and ``timestamps`` the matching source time in seconds (shot start + i/30,
  float64). Retiming to the game's frame rate happens only in the smooth/export stages.
- Bulk arrays are float32 on disk (NPZ); timestamps are float64.

Every result is a frozen-ish dataclass that serialises to one NPZ (arrays) plus a JSON-safe
``attrs`` dict. Use ``save_result`` / ``load_result``; never pickle.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Skeleton:
    """A joint hierarchy with a rest pose.

    ``parents[i]`` is the index of joint i's parent, -1 for the root. Parents must precede
    children (topological order). ``rest_offsets[i]`` is joint i's translation from its parent in
    the rest pose, in the parent's frame (metres); for the root it is the rest position in world.
    ``rest_rotations`` are optional local rest rotations (xyzw); None means identity for all.
    """

    name: str
    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    rest_offsets: np.ndarray  # [J,3] float64
    rest_rotations: np.ndarray | None = None  # [J,4] xyzw or None

    def __post_init__(self) -> None:
        j = len(self.joint_names)
        if len(self.parents) != j:
            raise ValueError(f"{self.name}: {j} joints but {len(self.parents)} parents")
        if len(set(self.joint_names)) != j:
            raise ValueError(f"{self.name}: duplicate joint names")
        for i, p in enumerate(self.parents):
            if p >= i or p < -1:
                raise ValueError(f"{self.name}: joint {self.joint_names[i]} parent {p} not before it")
        if self.parents.count(-1) != 1 or self.parents[0] != -1:
            raise ValueError(f"{self.name}: exactly one root, at index 0")
        off = np.asarray(self.rest_offsets, dtype=np.float64)
        if off.shape != (j, 3):
            raise ValueError(f"{self.name}: rest_offsets shape {off.shape}, expected {(j, 3)}")
        object.__setattr__(self, "rest_offsets", off)
        if self.rest_rotations is not None:
            rr = np.asarray(self.rest_rotations, dtype=np.float64)
            if rr.shape != (j, 4):
                raise ValueError(f"{self.name}: rest_rotations shape {rr.shape}, expected {(j, 4)}")
            object.__setattr__(self, "rest_rotations", rr)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def index(self, joint: str) -> int:
        try:
            return self.joint_names.index(joint)
        except ValueError:
            raise KeyError(f"{self.name}: no joint {joint!r}") from None

    def children(self, i: int) -> list[int]:
        return [k for k, p in enumerate(self.parents) if p == i]

    def with_offsets(self, rest_offsets: np.ndarray) -> Skeleton:
        """Same hierarchy with per-actor bone offsets (each fighter's locked bone lengths)."""
        return Skeleton(self.name, self.joint_names, self.parents, rest_offsets, self.rest_rotations)


# ---------------------------------------------------------------------------
# Licences (NFR-12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LicenceRecord:
    """Licence facts for one component or checkpoint, including transitive downloads.

    ``commercial_output_ok`` is True only when both code and weight licences allow commercial use
    of the outputs; None means unknown (which fails ``afv models --check``).
    """

    component: str
    code_licence: str
    weight_licence: str
    training_data_terms: str
    commercial_output_ok: bool | None
    source_url: str = ""
    weight_sha256: str = ""
    licence_text_path: str = ""  # dated copy, relative to the repo's models/ folder
    retrieved: str = ""  # ISO date the licence text was copied
    notes: str = ""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

ROLE_A = "A"  # fighter A (e.g. attacker)
ROLE_B = "B"  # fighter B (e.g. defender)
Role = Literal["A", "B"]


@dataclass
class Result:
    """Base for every stage result. Subclasses add array fields and plain attributes.

    Array-valued fields (np.ndarray) go to NPZ; everything else must be JSON-serialisable and goes
    to ``attrs``. Nested per-actor results are stored under ``"<role>/<field>"`` keys.
    """

    kind: ClassVar[str] = "result"
    frames: np.ndarray  # [F] int64 source frame indices
    timestamps: np.ndarray  # [F] float64 seconds

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=np.int64)
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        if self.frames.ndim != 1 or self.frames.shape != self.timestamps.shape:
            raise ValueError("frames and timestamps must be equal-length 1-D arrays")

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])


@dataclass
class Tracks(Result):
    """Person boxes with persistent IDs (stage detect_track).

    ``boxes[f, t]`` is (x0, y0, x1, y1) in source pixels for track ``track_ids[t]``; NaN when the
    track is absent on that frame. ``scores`` is the matched detector confidence, 0 where absent
    or where the track's box matches no detection (detect_track writes masks to masks/<id>/).
    ``fighters`` maps role -> chosen track id (empty until the user picks).
    """

    kind: ClassVar[str] = "tracks"
    track_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # [T]
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 4), np.float32))  # [F,T,4]
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))  # [F,T]
    fighters: dict[str, int] = field(default_factory=dict)


@dataclass
class ActorKeypoints:
    kp: np.ndarray  # [F,K,2] pixels
    conf: np.ndarray  # [F,K] 0..1


@dataclass
class Keypoints2D(Result):
    """2D keypoints per fighter (stage kp2d). ``keypoint_set`` names the layout, e.g. 'vitpose77'."""

    kind: ClassVar[str] = "keypoints2d"
    keypoint_set: str = ""
    keypoint_names: list[str] = field(default_factory=list)
    actors: dict[str, ActorKeypoints] = field(default_factory=dict)


@dataclass
class CameraTrack(Result):
    """Per-frame camera (stage camera).

    ``cam_rot`` [F,4] xyzw and ``cam_pos`` [F,3] give the camera-to-world transform (camera looks
    down its local -Z, +Y up, OpenGL/Blender convention). ``focal_px`` [F] focal length in pixels;
    principal point (cx, cy) and image size are constant per shot. The ground plane is
    ``dot(ground_normal, x) + ground_offset = 0`` in world; with gravity aligned, normal = +Y.
    """

    kind: ClassVar[str] = "camera"
    cam_rot: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    cam_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    focal_px: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    width: int = 0
    height: int = 0
    cx: float = 0.0
    cy: float = 0.0
    ground_normal: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    ground_offset: float = 0.0
    static: bool = False


@dataclass
class ActorMotion:
    """One fighter's motion on ``skeleton``: local rotations, root translation, contacts, confidence."""

    rot: np.ndarray  # [F,J,4] xyzw local
    root: np.ndarray  # [F,3] world, metres
    conf: np.ndarray | None = None  # [F,J] 0..1 per-joint confidence, optional
    contact: np.ndarray | None = None  # [F,C] 0..1, channels named by contact_channels
    rest_offsets: np.ndarray | None = None  # [J,3] this fighter's bone offsets; None = skeleton's
    # [F,3] GEM-X's own world-rollout root trajectory, expressed in this result's world (same frames).
    # Drifts against the camera over a shot but is temporally smooth: a velocity prior for pair/contacts.
    root_prior: np.ndarray | None = None
    # [F] bool: frames whose 2D evidence is too weak for GEM-X's depth (pose3d: a truncated/occluded
    # body or an interpolated box). None = unknown; consumers that do not know it ignore it.
    weak: np.ndarray | None = None


@dataclass
class BodyMotion(Result):
    """World-grounded motion per fighter in one shared world (stages pose3d, contacts, smooth)."""

    kind: ClassVar[str] = "body_motion"
    skeleton: str = ""  # registry name, e.g. "soma_provisional"
    fps: float = 0.0  # nominal sample rate of the arrays (timestamps are authoritative)
    contact_channels: list[str] = field(default_factory=list)  # e.g. ["l_heel","l_toe","r_heel","r_toe"]
    actors: dict[str, ActorMotion] = field(default_factory=dict)
    # Translation (metres) applied to the camera stage's world to get this result's world. Stages that
    # move the world (pose3d floor re-fit, contacts floor refinement) ADD to it; anything reprojecting
    # with the CameraTrack must use cam_pos + world_shift. After the contacts stage the floor is y = 0.
    world_shift: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class Take:
    """One fighter's finished motion, ready to export (record 'Take' in the spec)."""

    skeleton: str
    fps: float
    speed_factor: float
    rot: np.ndarray  # [F,J,4] xyzw local
    root: np.ndarray  # [F,3] world metres, Y-up
    root_motion: Literal["in_place", "world", "extracted"] = "world"
    contact_channels: list[str] = field(default_factory=list)
    contact: np.ndarray | None = None  # [F,C]
    role: str = ROLE_A
    provenance: dict[str, Any] = field(default_factory=dict)  # source path/sha256/title/url/in-out times
    rest_offsets: np.ndarray | None = None  # [J,3] this fighter's bone offsets; None = skeleton's


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

RESULT_KINDS: dict[str, type[Result]] = {
    c.kind: c for c in (Result, Tracks, Keypoints2D, CameraTrack, BodyMotion)
}


def register_result_kind(cls: type[Result]) -> type[Result]:
    """Make a module-specific Result subclass (e.g. an export manifest) loadable."""
    RESULT_KINDS[cls.kind] = cls
    return cls
_NESTED: dict[str, type] = {"keypoints2d": ActorKeypoints, "body_motion": ActorMotion}
_ATTRS_KEY = "__attrs__"


def result_to_arrays(res: Result) -> dict[str, np.ndarray]:
    """Flatten a result to NPZ-ready arrays; non-array fields go to a JSON string under __attrs__."""
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {"kind": res.kind}
    for f in dataclasses.fields(res):
        v = getattr(res, f.name)
        if isinstance(v, np.ndarray):
            arrays[f.name] = _store(f.name, v)
        elif f.name == "actors":
            roles = []
            for role, actor in v.items():
                roles.append(role)
                for af in dataclasses.fields(actor):
                    av = getattr(actor, af.name)
                    if av is not None:
                        arrays[f"{role}/{af.name}"] = _store(af.name, np.asarray(av))
            attrs["actors"] = roles
        else:
            attrs[f.name] = v
    arrays[_ATTRS_KEY] = np.array(json.dumps(attrs))
    return arrays


def _store(name: str, a: np.ndarray) -> np.ndarray:
    if name == "timestamps":
        return a.astype(np.float64)
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32)
    return a


def result_from_arrays(arrays: dict[str, np.ndarray]) -> Result:
    attrs = json.loads(str(arrays[_ATTRS_KEY]))
    kind = attrs.pop("kind")
    cls = RESULT_KINDS[kind]
    kwargs: dict[str, Any] = {}
    roles = attrs.pop("actors", None)
    for f in dataclasses.fields(cls):
        if f.name in arrays:
            kwargs[f.name] = np.asarray(arrays[f.name])
        elif f.name in attrs:
            kwargs[f.name] = attrs[f.name]
    if roles is not None:
        nested = _NESTED[kind]
        actors = {}
        for role in roles:
            fkw = {}
            for af in dataclasses.fields(nested):
                key = f"{role}/{af.name}"
                if key in arrays:
                    fkw[af.name] = np.asarray(arrays[key])
            actors[role] = nested(**fkw)
        kwargs["actors"] = actors
    return cls(**kwargs)


def save_result(res: Result, path: Path) -> None:
    """Write ``res`` as a compressed NPZ at ``path`` (caller handles atomic placement)."""
    np.savez_compressed(path, **result_to_arrays(res))


def load_result(path: Path) -> Result:
    with np.load(path, allow_pickle=False) as z:
        return result_from_arrays({k: z[k] for k in z.files})
