"""Take package for Blender (``afv take``) and the headless take .blend (``afv blender``).

Blender's Python has numpy but neither pydantic nor afv, so a take is handed over as plain files in
``<shot>/take/`` (the add-on and ``blender/afv_addon/headless.py`` read only these):

``take.json``
    format/version, shot (id, fps, frame range, speed factor, timestamps range), the project's WSL
    path (so the add-on can run ``afv`` again), the rig preset, one entry per fighter (bones with
    parents, translating bones, contact/plant channels, its NPZ), the camera (image size, principal
    point, sensor width, the world shift applied), the proxy video (absolute Windows path and a path
    relative to the take folder), the QC spans for the confidence strip, the export presets, and
    the stage results it was made from. Written last, atomically: it is the package's commit.
``<name>_<role>.<token>.npz``
    per fighter (``<token>`` is new on every write: see "Commit" below), armature space (afv's
    frame: Y-up, faces +Z, metres, quaternions **wxyz**), exactly
    the arrays the exporter hands to ``blender/afv_export.py``: ``rest_head`` / ``rest_tail`` /
    ``rest_zaxis`` [B,3], ``pose_head`` [F,B,3], ``pose_delta`` [F,B,4]; plus the rig's local motion
    ``rot`` [F,B,4] (xyzw, rest rotations identity) and ``loc`` [F,B,3], ``contact`` [F,C] and
    ``plant`` [F,P] (uint8 plant masks, ``plant_channels``).
``camera.<token>.npz``
    ``pos`` [F,3] camera centre and ``rot`` [F,4] camera-to-world (wxyz; looks down -Z, +Y up) in
    the **bodies' world** (``cam_pos + world_shift``, Y-up), ``focal_px`` [F].
``timeline.<token>.npz``
    ``frames`` [F] (shot timeline, 30 Hz) and ``timestamps`` [F] (source seconds).
``qc.<token>.npz``
    the per-frame curves the spans were cut from (``<role>/<kind>``).
``tracks.<token>.npz`` (optional, take.json ``tracks``)
    detect_track's person boxes for the add-on's box overlay (pick fighters / swap IDs): ``boxes``
    [F,T,4] (x0, y0, x1, y1) in **camera image pixels** (take.json ``tracks.image_size``; source
    pixels scaled to the camera's width/height), NaN where an object is absent, on the take's
    frames; ``scores`` [F,T]. take.json ``tracks``: ``ids`` (the tracked object ids, the ids
    ``edits/ids.json`` swaps), ``fighters`` (role -> object id), ``labels`` (object id -> label from
    ``edits/fighters.json``), ``image_size``.

Commit: every write uses fresh NPZ names, then replaces ``take.json`` atomically, then deletes the
NPZ files the previous ``take.json`` listed. A reader that opened ``take.json`` therefore always finds
the files it names; nothing else in the folder is touched (``take.blend`` lives there too), and a
non-empty folder without a ``take.json`` is refused.

Paths: the shot and project folders are stored relative to the take folder (``shot_rel``,
``project_rel``) as well as absolute (WSL), so a moved project still works; the video likewise.
``take_id`` is ``<project folder>:<shot>:<project uid>`` (uid from the source hash and the project's
creation time, so two projects with the same folder name differ).

The take is on the shot timeline (never retimed), so the proxy video, the camera and both fighters
stay frame-locked; ``speed_factor`` is recorded for the export. The rig is the export rig of the
shot's export preset (same bones, rest and motion as the exports); an ``in_place`` root-motion
setting is shown as ``world`` (the overlay needs the world path).
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from afv.core.cache import atomic_write_bytes, atomic_write_json
from afv.core.project import ProjectError, Shot, shot_context
from afv.core.skeletons import get_skeleton
from afv.core.types import BodyMotion, CameraTrack, Take
from afv.export.exporter import blender_arrays, take_motion
from afv.export.presets import PRESETS, get_preset
from afv.motion.rotations import make_continuous, xyzw_to_wxyz

TAKE_DIR = "take"
TAKE_FILE = "take.json"
TAKE_FORMAT = "afv_take"
TAKE_VERSION = 1
MOTION_STAGES = ("hands_face", "smooth")  # the export stage's input first
SENSOR_WIDTH_MM = 36.0
FIGHTER_COLORS = {"A": [0.85, 0.25, 0.2, 1.0], "B": [0.2, 0.45, 0.9, 1.0]}
HEADLESS_SCRIPT = "afv_addon/headless.py"


class TakeError(RuntimeError):
    """The take cannot be written (missing results, bad settings); the message is fit to show."""


# ---------------------------------------------------------------------------
# QC spans (confidence strip)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QcThresholds:
    """Per-frame limits (warn, bad) that cut the confidence strip's spans.

    ``low_conf`` is "below" (median joint confidence of the frame); the others are "above".
    ``reproj_px`` is for a 720 px tall image and scales with the image height.
    """

    low_conf: tuple[float, float] = (0.35, 0.2)
    reproj_px: tuple[float, float] = (25.0, 45.0)
    skate_m_s: tuple[float, float] = (0.05, 0.15)
    penetration_m: tuple[float, float] = (0.01, 0.03)
    pops: tuple[float, float] = (1.0, 3.0)
    weak_2d: tuple[float, float] = (0.5, 2.0)  # [Z, round 3] pose3d's ActorMotion.weak (always "warn")
    merge_gap: int = 2  # flagged frames this close are one span


QC_KINDS: dict[str, tuple[str, str, str, float]] = {  # kind -> (threshold field, label, unit, display scale)
    "low_conf": ("low_conf", "low conf", "", 1.0),
    "reproj": ("reproj_px", "reproj", "px", 1.0),
    "slide": ("skate_m_s", "slide", "cm/s", 100.0),
    "penetration": ("penetration_m", "floor pen", "cm", 100.0),
    "pop": ("pops", "pop", "", 1.0),
    "weak_2d": ("weak_2d", "weak 2D", "", 1.0),  # [Z, round 3]
}


def _runs(mask: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) index runs of ``mask``; runs separated by <= ``gap`` frames merge."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > gap + 1:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


def qc_spans(
    curves: dict[str, dict[str, np.ndarray]],
    frames: np.ndarray,
    thresholds: QcThresholds | None = None,
    image_height: int = 720,
) -> list[dict[str, Any]]:
    """Spans per fighter and kind where a curve crosses its warn limit (sorted by start frame).

    Each span: ``role``, ``kind``, ``start``/``end`` (inclusive scene frames), ``severity``
    (``warn``/``bad``), ``peak`` (worst value, display units), ``unit`` and a short ``label``.
    """
    thresholds = thresholds or QcThresholds()
    frames = np.asarray(frames, dtype=np.int64)
    spans = []
    for role, kinds in curves.items():
        for kind, values in kinds.items():
            if kind not in QC_KINDS:
                continue
            tfield, label, unit, scale = QC_KINDS[kind]
            warn, bad = getattr(thresholds, tfield)
            if kind == "reproj":
                warn, bad = warn * image_height / 720.0, bad * image_height / 720.0
            v = np.asarray(values, dtype=np.float64)
            finite = np.isfinite(v)
            below = kind == "low_conf"
            if below:
                flag = finite & (v < warn)
            else:
                flag = finite & ((v >= warn) if kind in ("pop", "weak_2d") else (v > warn))  # [Z, round 3]
            for s, e in _runs(flag, thresholds.merge_gap):
                seg = v[s : e + 1][finite[s : e + 1]]
                peak = float(seg.min() if below else seg.max())
                severe = peak < bad if below else peak >= bad
                shown = peak * scale
                text = f"{shown:.2f}" if kind == "low_conf" else f"{shown:.0f}{unit}"
                text = "" if kind == "weak_2d" else text  # [Z, round 3] named, no number
                spans.append({
                    "role": role,
                    "kind": kind,
                    "start": int(frames[s]),
                    "end": int(frames[e]),
                    "severity": "bad" if severe else "warn",
                    "peak": round(shown, 4),
                    "unit": unit,
                    "label": f"{role} {label} {text}".rstrip() + (" !" if severe else ""),
                })
    spans.sort(key=lambda d: (d["start"], d["role"], d["kind"]))
    return spans


def qc_curves(
    motion: BodyMotion,
    camera: CameraTrack | None = None,
    kp2d: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    """Per-frame QC curves per fighter and the plant masks per foot channel.

    ``camera`` is in the camera stage's world (``motion.world_shift`` is applied here) and
    ``kp2d[role]`` [F,J,3] are GEM-X's 2D keypoints (x, y, conf) on the motion's joints: with both,
    ``reproj`` is the frame's median reprojection error over keypoints with conf > 0.5.
    """
    from afv.motion.contacts import actor_skeleton
    from afv.motion.fk import forward_kinematics
    from afv.motion.ik import get_roles
    from afv.qc.metrics import per_frame_metrics

    roles = get_roles(get_skeleton(motion.skeleton))
    metrics = per_frame_metrics(motion, roles)
    curves: dict[str, dict[str, np.ndarray]] = {}
    plants: dict[str, dict[str, np.ndarray]] = {}
    for role, a in motion.actors.items():
        m = metrics[role]
        c: dict[str, np.ndarray] = {
            "slide": np.asarray(m["skate"], dtype=np.float64),
            "penetration": np.asarray(m["penetration"], dtype=np.float64),
            "pop": np.asarray(m["pops"], dtype=np.float64),
        }
        if a.conf is not None:
            c["low_conf"] = np.median(np.asarray(a.conf, dtype=np.float64), axis=1)
        if getattr(a, "weak", None) is not None:  # [Z, round 3] pose3d's weak 2D evidence spans
            c["weak_2d"] = np.asarray(a.weak, dtype=np.float64).reshape(-1)
        if camera is not None and kp2d is not None and role in kp2d:
            kp = np.asarray(kp2d[role], dtype=np.float64)
            sk = actor_skeleton(motion, a)
            if kp.shape[:2] == (motion.num_frames, sk.num_joints):
                pos, _ = forward_kinematics(sk, a.rot, a.root)
                uv = project_points(camera, pos, motion.frames, motion.world_shift)
                err = np.linalg.norm(uv - kp[..., :2], axis=-1)
                good = (kp[..., 2] > 0.5) & np.isfinite(err)
                with np.errstate(invalid="ignore"):
                    med = np.array([np.median(e[g]) if g.any() else np.nan
                                    for e, g in zip(err, good, strict=True)])
                c["reproj"] = med
        curves[role] = c
        plants[role] = {k[len("contact_") :]: m[k] > 0.5 for k in m if k.startswith("contact_")}
    return curves, plants


# ---------------------------------------------------------------------------
# Camera in the bodies' world
# ---------------------------------------------------------------------------


def camera_in_world(
    cam: CameraTrack, frames: np.ndarray, world_shift: Sequence[float]
) -> dict[str, np.ndarray]:
    """Camera per motion frame in the bodies' world: ``pos`` [F,3], ``rot`` [F,4] wxyz, ``focal_px`` [F]."""
    from afv.qc.camera import camera_index

    idx = camera_index(cam, frames)
    pos = np.asarray(cam.cam_pos, dtype=np.float64)[idx] + np.asarray(world_shift, dtype=np.float64)
    rot = make_continuous(np.asarray(cam.cam_rot, dtype=np.float64)[idx])
    f = np.asarray(cam.focal_px, dtype=np.float64).reshape(-1)
    focal = f[idx] if f.size > 1 else np.full(len(idx), float(f[0]) if f.size else 1.0)
    return {"pos": pos, "rot": xyzw_to_wxyz(rot), "focal_px": focal}


def project_points(
    cam: CameraTrack, points: np.ndarray, frames: np.ndarray, world_shift: Sequence[float] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    """The pipeline's own pinhole projection (``afv.qc.camera.project``) of points [F,N,3] in the
    bodies' world (``world_shift`` from the camera's world) to pixels [F,N,2] (x right, y down)."""
    from afv.qc.camera import project

    shift = np.asarray(world_shift, dtype=np.float64)
    return project(np.asarray(points, dtype=np.float64) - shift, cam, frames)


# ---------------------------------------------------------------------------
# Paths (the package is read by Windows Blender)
# ---------------------------------------------------------------------------


def windows_path(p: Path) -> str:
    """``p`` as Windows sees it (``wslpath -w`` inside WSL; unchanged elsewhere)."""
    p = Path(p).absolute()
    if not sys.platform.startswith("linux") or shutil.which("wslpath") is None:
        return str(p)
    base, tail = p, []
    while not base.exists() and base.parent != base:  # wslpath wants an existing path
        tail.append(base.name)
        base = base.parent
    try:
        out = subprocess.run(["wslpath", "-w", str(base)], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return str(p)
    win = out.stdout.strip()
    return "\\".join([win.rstrip("\\"), *reversed(tail)]) if tail else win


def _rel(target: Path, start: Path) -> str:
    import os

    try:
        return Path(os.path.relpath(Path(target).resolve(), Path(start).resolve())).as_posix()
    except ValueError:
        return ""


# ---------------------------------------------------------------------------
# Writing the package
# ---------------------------------------------------------------------------


@dataclass
class RigSettings:
    """Which export rig the take uses (defaults: the shot's export settings)."""

    preset: str = "unity_humanoid"
    root_motion: str | None = None
    prefix: str | None = None
    fingers: bool = True
    root_smooth_s: float = 0.2
    root_yaw_smooth_s: float = 0.5
    name: str = ""  # export.name: file stem of the exports ("" = the shot folder name)

    def mode(self) -> str:
        """The root-motion mode shown in the take (validated like the export stage does)."""
        if self.preset not in PRESETS:
            raise TakeError(f"unknown export preset {self.preset!r}; choose one of {sorted(PRESETS)}")
        p = get_preset(self.preset)
        mode = self.root_motion or p.root_motion
        if mode not in p.allowed_root_motion:
            raise TakeError(f"preset {p.name} does not support root motion {mode!r} "
                            f"(allowed: {', '.join(p.allowed_root_motion)})")
        return "world" if mode == "in_place" else mode


@dataclass
class TrackBoxes:
    """detect_track's boxes on the take's frames, in camera image pixels (see ``tracks_on_frames``)."""

    ids: list[int]
    boxes: np.ndarray  # [F,T,4] x0, y0, x1, y1; NaN where absent
    scores: np.ndarray  # [F,T]
    fighters: dict[str, int]
    image_size: tuple[int, int]
    labels: dict[int, str] = field(default_factory=dict)


def tracks_on_frames(
    tracks: Any, frames: np.ndarray, scale: tuple[float, float] = (1.0, 1.0),
    image_size: tuple[int, int] = (0, 0), labels: dict[int, str] | None = None,
) -> TrackBoxes:
    """A ``Tracks`` result resampled onto ``frames`` (NaN boxes where it has no frame) with its
    source-pixel boxes scaled by ``scale`` (camera width / source width, height likewise)."""
    frames = np.asarray(frames, dtype=np.int64)
    tf = np.asarray(tracks.frames, dtype=np.int64)
    boxes = np.asarray(tracks.boxes, dtype=np.float64)
    scores = np.asarray(tracks.scores, dtype=np.float64)
    n_t = len(tracks.track_ids)
    out = np.full((frames.size, n_t, 4), np.nan)
    sc = np.zeros((frames.size, n_t))
    if tf.size and boxes.ndim == 3 and boxes.shape[0] == tf.size:
        idx = np.clip(np.searchsorted(tf, frames), 0, tf.size - 1)
        hit = tf[idx] == frames
        sx, sy = scale
        out[hit] = boxes[idx[hit]] * np.array([sx, sy, sx, sy])
        if scores.shape == boxes.shape[:2]:
            sc[hit] = scores[idx[hit]]
    return TrackBoxes(ids=[int(i) for i in tracks.track_ids], boxes=out, scores=sc,
                      fighters={str(k): int(v) for k, v in dict(tracks.fighters).items()},
                      image_size=(int(image_size[0]), int(image_size[1])), labels=dict(labels or {}))


@dataclass
class TakeInputs:
    """Everything a take is made of (``take_inputs`` reads it from a shot)."""

    motion: BodyMotion
    camera: CameraTrack | None = None
    shot_id: str = "shot"
    name: str = ""  # file stem; "" = shot_id
    speed_factor: float = 1.0
    project_dir: str = ""
    project_uid: str = ""  # unique per project (see take_id)
    shot_dir: Path | None = None  # stored relative to the take folder
    video: Path | None = None  # proxy (or normalised shot) video
    kp2d: dict[str, np.ndarray] | None = None  # role -> [F,J,3] GEM-X 2D keypoints
    tracks: TrackBoxes | None = None  # detect_track's boxes for the box overlay
    provenance: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, Any] = field(default_factory=dict)  # stage -> {version_dir, output_hash}
    notes: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)  # why the stored results may be out of date
    stale_summary: str = ""  # one short, actionable line for the above


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def write_take(
    out_dir: Path,
    inp: TakeInputs,
    rig: RigSettings | None = None,
    thresholds: QcThresholds | None = None,
) -> Path:
    """Write the take package into ``out_dir``; returns the path of ``take.json`` (see "Commit")."""
    rig = rig or RigSettings()
    thresholds = thresholds or QcThresholds()
    mode = rig.mode()
    motion = inp.motion
    if not motion.actors:
        raise TakeError("the motion has no fighters")
    out_dir = Path(out_dir)
    previous = _previous_files(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preset = get_preset(rig.preset)
    token = uuid.uuid4().hex[:8]
    sk = get_skeleton(motion.skeleton)
    fps = float(motion.fps or 30.0)
    frames = np.asarray(motion.frames, dtype=np.int64)
    stem = inp.name or inp.shot_id
    height = inp.camera.height if inp.camera is not None and inp.camera.height else 720

    curves, plants = qc_curves(motion, inp.camera, inp.kp2d)
    spans = qc_spans(curves, frames, thresholds, height)

    written: list[str] = []
    try:  # every file of this write is removed again unless take.json names it
        fighters = []
        for role, a in motion.actors.items():
            take = Take(
                skeleton=motion.skeleton, fps=fps, speed_factor=inp.speed_factor, rot=a.rot, root=a.root,
                root_motion=mode, contact_channels=list(motion.contact_channels), contact=a.contact,
                role=role, rest_offsets=a.rest_offsets,
            )
            try:
                m = take_motion(
                    take, preset, skeleton=sk, root_motion=mode, prefix=rig.prefix, fingers=rig.fingers,
                    root_smooth_s=rig.root_smooth_s, root_yaw_smooth_s=rig.root_yaw_smooth_s,
                )
            except (KeyError, ValueError) as e:
                raise TakeError(f"fighter {role}: cannot build the export rig: {e}") from e
            arrays = blender_arrays(m)
            plant_channels = list(plants.get(role, {}))
            arrays = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()}
            arrays["rot"] = np.asarray(m.rot, dtype=np.float64)
            arrays["loc"] = np.asarray(m.loc, dtype=np.float64)
            n = m.num_frames
            if a.contact is not None:
                arrays["contact"] = np.asarray(a.contact, dtype=np.float32)
            arrays["plant"] = (
                np.stack([plants[role][c] for c in plant_channels], axis=1).astype(np.uint8)
                if plant_channels else np.zeros((n, 0), np.uint8)
            )
            name = f"{stem}_{role}"
            npz = f"{name}.{token}.npz"
            atomic_write_bytes(out_dir / npz, _npz_bytes(arrays))
            written.append(npz)
            rg = m.rig
            fighters.append({
                "role": role,
                "name": name,
                "npz": npz,
                "frame_count": n,
                "bones": [{"name": b, "parent": rg.names[p] if p >= 0 else None}
                          for b, p in zip(rg.names, rg.parents, strict=True)],
                "translating": [rg.names[b] for b in m.translating],
                "hips": rg.names[rg.hips],
                "root_bone": rg.names[rg.root] if rg.root is not None else None,
                "contact_channels": list(motion.contact_channels) if a.contact is not None else [],
                "plant_channels": plant_channels,
                "color": FIGHTER_COLORS.get(role, [0.6, 0.6, 0.6, 1.0]),
            })

        camera_meta = None
        if inp.camera is not None and len(inp.camera.cam_pos):
            cam = inp.camera
            cw = camera_in_world(cam, frames, motion.world_shift)
            atomic_write_bytes(out_dir / f"camera.{token}.npz", _npz_bytes(cw))
            written.append(f"camera.{token}.npz")
            camera_meta = {
                "npz": f"camera.{token}.npz",
                "width": int(cam.width),
                "height": int(cam.height),
                "cx": float(cam.cx),
                "cy": float(cam.cy),
                "sensor_width_mm": SENSOR_WIDTH_MM,
                "static": bool(cam.static),
                "world_shift": [float(v) for v in motion.world_shift],
                "convention": "camera-to-world in the bodies' world (Y-up); looks down -Z, +Y up; wxyz",
            }
        timeline, qc_npz = f"timeline.{token}.npz", f"qc.{token}.npz"
        atomic_write_bytes(out_dir / timeline, _npz_bytes(
            {"frames": frames, "timestamps": np.asarray(motion.timestamps, dtype=np.float64)}))
        atomic_write_bytes(out_dir / qc_npz, _npz_bytes(
            {f"{r}/{k}": np.asarray(v, dtype=np.float32) for r, c in curves.items() for k, v in c.items()}))
        written += [timeline, qc_npz]

        tracks_meta = None
        if inp.tracks is not None and inp.tracks.ids:
            tb = inp.tracks
            tnpz = f"tracks.{token}.npz"
            atomic_write_bytes(out_dir / tnpz, _npz_bytes({
                "boxes": np.asarray(tb.boxes, dtype=np.float32),
                "scores": np.asarray(tb.scores, dtype=np.float32),
                "ids": np.asarray(tb.ids, dtype=np.int64)}))
            written.append(tnpz)
            tracks_meta = {"npz": tnpz, "ids": [int(i) for i in tb.ids], "fighters": dict(tb.fighters),
                           "labels": {str(k): v for k, v in tb.labels.items()},
                           "image_size": [int(v) for v in tb.image_size],
                           "convention": "boxes x0, y0, x1, y1 in camera image pixels (y down); NaN = absent"}

        video = None
        if inp.video is not None and Path(inp.video).is_file():
            video = {"path_win": windows_path(inp.video), "path_rel": _rel(inp.video, out_dir),
                     "first_frame": 0}
        elif inp.video is not None:
            inp.notes.append(f"video {inp.video} not found")

        meta = {
            "format": TAKE_FORMAT,
            "version": TAKE_VERSION,
            "take_id": f"{Path(inp.project_dir).name or 'project'}:{inp.shot_id}"
                       + (f":{inp.project_uid}" if inp.project_uid else ""),
            "shot": {
                "id": inp.shot_id,
                "fps": fps,
                "frame_start": int(frames[0]) if frames.size else 0,
                "frame_end": int(frames[-1]) if frames.size else 0,
                "frame_count": int(frames.size),
                "speed_factor": float(inp.speed_factor),
                "timestamps": ([float(motion.timestamps[0]), float(motion.timestamps[-1])]
                               if frames.size else []),
            },
            "project_dir": inp.project_dir,
            "shot_rel": _rel(inp.shot_dir, out_dir) if inp.shot_dir is not None else "",
            "project_rel": (_rel(Path(inp.shot_dir).parent.parent, out_dir)
                            if inp.shot_dir is not None else ""),
            "rig": {"preset": preset.name, "root_motion": mode, "prefix": rig.prefix if rig.prefix is not None
                    else preset.prefix, "fingers": rig.fingers, "root_bone": preset.root_bone},
            "axes": "afv world: Y-up right-handed metres; fighters face +Z; quaternions wxyz. Blender: "
                    "armature objects carry +90 deg about X (as the FBX export), the camera is converted.",
            "skeleton": motion.skeleton,
            "fighters": fighters,
            "camera": camera_meta,
            "tracks": tracks_meta,
            "video": video,
            "timeline": timeline,
            "qc": {"npz": qc_npz, "spans": spans, "thresholds": asdict(thresholds)},
            "presets": [{"name": p.name, "label": p.label, "formats": list(p.formats)}
                        for p in PRESETS.values()],
            "provenance": inp.provenance,
            "sources": inp.sources,
            "notes": inp.notes,
            "stale": inp.stale,
            "stale_summary": inp.stale_summary,
        }
        path = out_dir / TAKE_FILE
        atomic_write_json(path, meta)  # the commit
    except BaseException:
        if not _committed(out_dir, token):  # e.g. Ctrl+C just after the replace: the new take stands
            for n in written:
                (out_dir / n).unlink(missing_ok=True)
        raise
    keep = set(written)
    for n in previous - keep:
        (out_dir / n).unlink(missing_ok=True)
    prefixes = {"camera", "timeline", "qc", "tracks", *(f["name"] for f in fighters)}
    for p in out_dir.glob("*.npz"):  # orphans of earlier writes that failed before their commit
        m = _TOKEN_NPZ.match(p.name)
        if m and m.group(1) in prefixes and p.name not in keep:
            p.unlink(missing_ok=True)
    return path


_TOKEN_NPZ = re.compile(r"^(.+)\.([0-9a-f]{8})\.npz$")
_OWN_ARTEFACTS = re.compile(r"^(take\.blend1?|.+\.[0-9a-f]{8}\.npz)$")


def _committed(out_dir: Path, token: str) -> bool:
    """Does take.json name this write's files (the commit happened)?"""
    try:
        meta = json.loads((out_dir / TAKE_FILE).read_text())
    except (OSError, ValueError):
        return False
    return any(f".{token}." in n for n in package_files(meta))


def package_files(meta: dict[str, Any]) -> set[str]:
    """The NPZ files a take.json names (relative to its folder)."""
    names = {f["npz"] for f in meta.get("fighters", [])}
    names |= {meta.get("timeline", ""), (meta.get("qc") or {}).get("npz", "")}
    names.add((meta.get("camera") or {}).get("npz", ""))
    names.add((meta.get("tracks") or {}).get("npz", ""))
    return {n for n in names if n and "/" not in n and "\\" not in n}


def _previous_files(out_dir: Path) -> set[str]:
    """Files of the take already in ``out_dir`` (deleted after the new commit). Refuses a non-empty
    folder that is not a take, so a wrong ``--out`` never deletes anything."""
    if not out_dir.exists():
        return set()
    if not out_dir.is_dir():
        raise TakeError(f"{out_dir} is not a folder")
    jp = out_dir / TAKE_FILE
    if not jp.is_file():  # our own artefacts (a .blend, orphaned NPZs, Blender job folders) are fine
        others = [p.name for p in out_dir.iterdir()
                  if not p.name.startswith(".") and not _OWN_ARTEFACTS.match(p.name)]
        if others:
            raise TakeError(f"{out_dir} is not empty and holds no take ({', '.join(sorted(others)[:5])}); "
                            "choose an empty folder or a take folder")
        return set()
    try:
        meta = json.loads(jp.read_text())
    except (OSError, ValueError):
        return set()  # unreadable: keep its files rather than guess
    if meta.get("format") != TAKE_FORMAT:
        raise TakeError(f"{jp} is not an afv take; not overwriting it")
    return package_files(meta)


# ---------------------------------------------------------------------------
# From a shot
# ---------------------------------------------------------------------------


def _source(shot: Shot, stage: str) -> dict[str, Any]:
    meta = shot.meta(stage) or {}
    return {"version_dir": shot.stage_dir(stage).name, "output_hash": meta.get("output_hash"),
            "version": meta.get("version")}


def _kp2d(shot: Shot, roles: Sequence[str], notes: list[str]) -> dict[str, np.ndarray] | None:
    """GEM-X's 2D keypoints per fighter (pose3d's ``gemx/<role>/soma_fk.npz``), if present."""
    out = {}
    base = shot.stage_dir("pose3d") / "gemx"
    for role in roles:
        p = base / role / "soma_fk.npz"
        if not p.is_file():
            continue
        try:
            with np.load(p, allow_pickle=False) as z:
                out[role] = np.asarray(z["kp2d"], dtype=np.float64)
        except (OSError, KeyError, ValueError) as e:
            notes.append(f"cannot read 2D keypoints {p}: {e}")
    return out or None


def _track_boxes(shot: Shot, frames: np.ndarray, camera: CameraTrack | None, notes: list[str],
                 source_size: tuple[int, int] | None) -> TrackBoxes | None:
    """detect_track's boxes for the overlay, in the camera image's pixels (None without tracks)."""
    if not shot.has_result("detect_track"):
        return None
    try:
        tr = shot.load_result("detect_track")
    except Exception as e:  # the overlay just has no boxes
        notes.append(f"cannot read detect_track's boxes: {e}")
        return None
    if not len(getattr(tr, "track_ids", ())):
        return None
    w, h = (int(camera.width), int(camera.height)) if camera is not None else (0, 0)
    sw, sh = source_size or (w, h)
    if not (w and h):
        w, h = sw, sh
    scale = (w / sw, h / sh) if sw and sh else (1.0, 1.0)
    labels: dict[int, str] = {}
    try:
        for o in (shot.read_edit("fighters") or {}).get("objects", []) or []:
            labels[int(o["id"])] = str(o.get("label", ""))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return tracks_on_frames(tr, frames, scale, (w, h), labels)


def _copy_weak_mask(shot: Shot, motion: BodyMotion, notes: list[str]) -> None:
    """[Z, round 3] pose3d's ``ActorMotion.weak`` onto the motion's fighters that lack it (later stages
    do not carry it), for the confidence strip's "weak 2D" spans."""
    if not shot.has_result("pose3d"):
        return
    try:
        p3 = shot.load_result("pose3d")
    except Exception as e:  # the strip just lacks the spans
        notes.append(f"cannot read pose3d's weak 2D mask: {e}")
        return
    for role, a in motion.actors.items():
        src = getattr(p3, "actors", {}).get(role)
        w = getattr(src, "weak", None)
        if getattr(a, "weak", None) is None and w is not None and len(w) == motion.num_frames:
            a.weak = np.asarray(w, dtype=bool)


def _provenance(shot: Shot) -> dict[str, Any]:
    """Source facts for the take (as the export's metadata records them)."""
    try:
        sc = shot_context(shot.dir)
    except (ProjectError, OSError, ValueError) as e:
        return {"shot_dir": str(shot.dir), "note": f"no shot context: {e}"}
    return {
        "shot_id": sc.shot_id, "shot_dir": sc.shot_dir, "source_path": sc.source_path,
        "source_sha256": sc.source_sha256, "source_title": sc.source_title, "source_url": sc.source_url,
        "start": sc.start, "duration": sc.duration, "speed_factor": sc.speed_factor,
    }


def stale_results(shot: Shot, stage: str) -> list[str]:
    """Why ``stage``'s stored result may be out of date: any result in its input chain whose recorded
    input hash differs from that input's current result, e.g. hands_face made from an older smooth,
    or pair/contacts (the motion's camera inputs) made from an older camera."""
    out, seen, todo = [], set(), [stage]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        meta = shot.meta(name) or {}
        for inp, h in (meta.get("inputs") or {}).items():
            cur = shot.meta(inp) or {}
            if cur.get("output_hash") is None:
                continue
            if h != cur.get("output_hash"):
                out.append(f"{name} was made from an older {inp} result")
            todo.append(inp)
    return sorted(set(out))


def plan_stale(shot: Shot, stage: str) -> list[str]:
    """Stages up to ``stage`` that ``afv run`` would re-run because their settings or code changed
    (stages only waiting on those are left out); [] if unsure."""
    try:
        from afv.core.runner import Runner

        rep = Runner(shot).plan(to=stage)
    except Exception:  # the plan is advisory only
        return []
    out = []
    for s in rep.stages:
        if s.state in ("fresh", "unavailable", "pending", "planned"):
            continue
        # (a stage blocked by an invalid edit file is listed with the file and the field: not hidden)
        why = s.reason if len(s.reason) <= 100 else s.reason[:97] + "..."
        out.append(f"{s.name}: {s.state}" + (f" ({why})" if why else ""))
    return out


def stale_summary(shot_id: str, chain: Sequence[str], plan: Sequence[str]) -> str:
    """One actionable line for ``stale_results`` (chain) and ``plan_stale`` (plan) messages. A stage that
    is blocked by an invalid edit file (``Runner._plan``) is not an old result: the line names the file
    instead (and ``afv run`` would refuse until it is fixed)."""
    import re

    files = list(dict.fromkeys(m.group(1) for p in plan if (m := re.search(r"\((edits/[\w.-]+\.json)", p))))
    blocked = {p.split(":")[0] for p in plan if re.search(r"\(edits/[\w.-]+\.json|its edit file must", p)}
    stages = list(dict.fromkeys(
        [p.split(":")[0] for p in plan if p.split(":")[0] not in blocked] + [c.split(" ")[0] for c in chain]
    ))
    parts = []
    if files:
        which = ", ".join(files)
        parts.append(f"{which} {'are' if len(files) > 1 else 'is'} invalid: fix or delete that entry "
                     f"(details: afv status {shot_id})")
    if stages:
        what = ", ".join(stages[:4]) + (" ..." if len(stages) > 4 else "")
        parts.append(f"{what} result{'s are' if len(stages) > 1 else ' is'} older than the current "
                     f"settings or code: run `afv run {shot_id}` (details: afv status {shot_id})")
    return "; ".join(parts)


def project_uid(shot: Shot) -> str:
    import hashlib

    info = shot.project.info
    return hashlib.sha1(f"{info.source.sha256}|{info.created}".encode()).hexdigest()[:8]


def take_inputs(shot: Shot, motion_stage: str | None = None, *, check_plan: bool = True) -> TakeInputs:
    """Read a shot's finished motion, camera, proxy video, 2D keypoints and tracking boxes.

    ``check_plan=False`` skips asking the runner which stages ``afv run`` would redo (its code-hash
    scan costs ~2 s on the Windows drive); the input-chain check (``stale_results``) still runs."""
    import afv.media  # noqa: F401  (registers the ingest result kind)

    stages = (motion_stage,) if motion_stage else MOTION_STAGES
    stage = next((s for s in stages if shot.has_result(s)), None)
    if stage is None:
        raise TakeError(f"shot {shot.id}: no finished motion yet (run the pipeline: afv run {shot.id})")
    motion = shot.load_result(stage)
    if not isinstance(motion, BodyMotion):
        raise TakeError(f"shot {shot.id}: stage {stage} is not a body motion")
    notes: list[str] = []
    sources = {"motion": {"stage": stage, **_source(shot, stage)}}
    camera = None
    if shot.has_result("camera"):
        cam = shot.load_result("camera")
        if isinstance(cam, CameraTrack) and len(cam.cam_pos):
            camera = cam
            sources["camera"] = _source(shot, "camera")
    else:
        notes.append("no camera result: the take has no camera")
    video = None
    source_size = None
    if shot.has_result("ingest"):
        media = shot.load_result("ingest")
        if getattr(media, "width", 0) and getattr(media, "height", 0):
            source_size = (int(media.width), int(media.height))
        d = shot.stage_dir("ingest")
        for name in (getattr(media, "proxy", ""), getattr(media, "video", "")):
            if name and (d / name).is_file():
                video = d / name
                break
        sources["ingest"] = _source(shot, "ingest")
    if video is None:
        notes.append("no proxy video: the camera has no background")
    kp2d = _kp2d(shot, list(motion.actors), notes)
    if kp2d:
        sources["pose3d"] = _source(shot, "pose3d")
    _copy_weak_mask(shot, motion, notes)  # [Z, round 3]
    tracks = _track_boxes(shot, np.asarray(motion.frames), camera, notes, source_size)
    if tracks is not None:
        sources["detect_track"] = _source(shot, "detect_track")
    stale = stale_results(shot, stage)
    plan = plan_stale(shot, stage) if check_plan else []
    summary = stale_summary(shot.id, stale, plan)
    stale = sorted(set(stale) | {f"afv run would re-run {s}" for s in plan})
    return TakeInputs(
        motion=motion, camera=camera, shot_id=shot.id, speed_factor=float(shot.info.speed_factor),
        project_dir=str(shot.project.root), project_uid=project_uid(shot), shot_dir=shot.dir,
        video=video, kp2d=kp2d, tracks=tracks, provenance=_provenance(shot), sources=sources, notes=notes,
        stale=stale, stale_summary=summary,
    )


def shot_rig_settings(shot: Shot, preset: str | None = None) -> RigSettings:
    """The shot's resolved export settings (project < shot), as the export stage would use them."""
    from afv.core.runner import Runner

    runner = Runner(shot)
    cls, _ = runner.resolve("export")
    p = runner.params_for("export", cls)
    return RigSettings(
        preset=preset or p.preset, root_motion=p.root_motion, prefix=p.prefix, fingers=p.fingers,
        root_smooth_s=p.root_smooth_s, root_yaw_smooth_s=p.root_yaw_smooth_s, name=p.name,
    )


def take_dir(shot: Shot) -> Path:
    return shot.dir / TAKE_DIR


def make_take(shot: Shot, out_dir: Path | None = None, *, preset: str | None = None,
              motion_stage: str | None = None, check_plan: bool = True) -> Path:
    """Write ``<shot>/take/`` (or ``out_dir``) from the shot's current results; returns take.json.
    ``check_plan=False``: see ``take_inputs`` (for a refresh right after ``afv run``)."""
    inp = take_inputs(shot, motion_stage, check_plan=check_plan)
    rig = shot_rig_settings(shot, preset)
    inp.name = rig.name or shot.dir.name  # the exports' stem (ExportStage: params.name or shot folder)
    return write_take(out_dir or take_dir(shot), inp, rig)


# ---------------------------------------------------------------------------
# Headless .blend and the alignment check
# ---------------------------------------------------------------------------


def build_blend(
    take_json: Path,
    out_blend: Path | None,
    *,
    sample_frames: Sequence[int] | None = None,
    samples_npz: Path | None = None,
    blender: Path | None = None,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the take scene in headless Blender (``blender/afv_addon/headless.py``) and save it to
    ``out_blend`` (None: don't save). With ``samples_npz``, Blender also writes per-fighter bone
    heads (Blender world, by bone name) and their pixel positions through the Blender camera
    (``world_to_camera_view``) at ``sample_frames`` (None: every frame)."""
    from afv.export import blender as bl

    exe = blender or bl.find_blender()
    if exe is None:
        raise bl.BlenderError("Blender not found; set AFV_BLENDER or install Blender 5.2")
    take_json = Path(take_json)
    job: dict[str, Any] = {"take": bl.host_path(take_json, exe),
                           "out_blend": bl.host_path(out_blend, exe) if out_blend else None}
    if samples_npz is not None:
        job["sample"] = {"npz": bl.host_path(samples_npz, exe),
                         "frames": None if sample_frames is None else [int(f) for f in sample_frames]}
    if out_blend:
        Path(out_blend).parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    tmp = None
    if work_dir is None:
        tmp = Path(tempfile.mkdtemp(prefix=".blender_job_", dir=take_json.parent))
    res = bl.run_blender(HEADLESS_SCRIPT, job, Path(work_dir) if work_dir else tmp, blender=exe)
    if tmp is not None:  # kept (with Blender's log) when Blender failed
        shutil.rmtree(tmp, ignore_errors=True)
    if not res.get("ok"):
        raise bl.BlenderError(f"Blender could not build the take: {res.get('error', res)}")
    return res


def load_take_arrays(take_json: Path) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    meta = json.loads(Path(take_json).read_text())
    arrays = {}
    for f in meta["fighters"]:
        with np.load(Path(take_json).parent / f["npz"]) as z:
            arrays[f["role"]] = {k: z[k] for k in z.files}
    return meta, arrays


def check_alignment(take_json: Path, samples_npz: Path, camera: CameraTrack | None = None) -> dict[str, Any]:
    """Compare Blender's samples with the package: 3D heads (mm) and, with ``camera`` (the camera
    stage's, in its own world), pixels through the Blender camera vs ``project_points`` (px)."""
    from afv.motion.rotations import zup_to_yup_vec

    meta, arrays = load_take_arrays(take_json)
    with np.load(samples_npz) as z:
        s = {k: z[k] for k in z.files}
    frames = np.asarray(s["frames"], dtype=np.int64)
    tl = np.load(Path(take_json).parent / meta["timeline"])
    all_frames = np.asarray(tl["frames"], dtype=np.int64)
    rows = np.searchsorted(all_frames, frames)
    ws = meta["camera"]["world_shift"] if meta.get("camera") else [0.0, 0.0, 0.0]
    out: dict[str, Any] = {"frames": int(frames.size), "fighters": {}}
    for f in meta["fighters"]:
        role, names = f["role"], [b["name"] for b in f["bones"]]
        bnames = [str(n) for n in s[f"{role}/bones"]]
        sel = [bnames.index(n) for n in names]
        heads_bl = s[f"{role}/heads"][:, sel]
        want = arrays[role]["pose_head"][rows]
        err3 = np.linalg.norm(zup_to_yup_vec(heads_bl) - want, axis=-1)
        rec: dict[str, Any] = {"bones": len(names), "max_mm": float(err3.max() * 1000),
                               "mean_mm": float(err3.mean() * 1000)}
        if camera is not None and f"{role}/pixels" in s:
            px_bl = s[f"{role}/pixels"][:, sel]
            px = project_points(camera, want, frames, ws)
            e = np.linalg.norm(px_bl - px, axis=-1)
            ok = np.isfinite(e)
            rec.update(px_max=float(e[ok].max()) if ok.any() else None,
                       px_mean=float(e[ok].mean()) if ok.any() else None,
                       px_points=int(ok.sum()))
        out["fighters"][role] = rec
    return out


def take_summary(take_json: Path) -> dict[str, Any]:
    """Short description of a written package (for the CLI)."""
    meta = json.loads(Path(take_json).read_text())
    spans = meta["qc"]["spans"]
    return {
        "take_id": meta["take_id"],
        "frames": meta["shot"]["frame_count"],
        "fps": meta["shot"]["fps"],
        "preset": meta["rig"]["preset"],
        "fighters": [f["role"] for f in meta["fighters"]],
        "camera": meta["camera"] is not None,
        "video": (meta["video"] or {}).get("path_win"),
        "spans": len(spans),
        "bad_spans": sum(1 for s in spans if s["severity"] == "bad"),
        "tracks": len((meta.get("tracks") or {}).get("ids", [])),
        "notes": meta["notes"],
        "stale": meta.get("stale", []),
        "stale_summary": meta.get("stale_summary", ""),
    }


def blend_shot(
    shot: Shot,
    out: Path | None = None,
    *,
    preset: str | None = None,
    check: bool = False,
    sample_frames: Sequence[int] | None = None,
    blender: Path | None = None,
) -> dict[str, Any]:
    """``afv blender``: refresh the shot's take package, build ``out`` (default
    ``<shot>/take/take.blend``) in headless Blender; with ``check``, also measure the scene against
    the package (3D heads, mm) and the camera overlay against the pipeline's projection (px)."""
    tj = make_take(shot, preset=preset)
    out = Path(out) if out is not None else take_dir(shot) / "take.blend"
    import tempfile

    tmp = None
    samples = None
    try:
        if check:
            shot.work_root.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix="blender_check_", dir=shot.work_root))
            samples = tmp / "samples.npz"
        res = build_blend(tj, out, samples_npz=samples, sample_frames=sample_frames, blender=blender)
        report: dict[str, Any] = {"take": str(tj), "blend": str(out), "summary": res.get("summary"),
                                  **{k: v for k, v in take_summary(tj).items() if k.startswith("stale")}}
        if samples is not None:
            cam = shot.load_result("camera") if shot.has_result("camera") else None
            report["check"] = check_alignment(tj, samples, cam if isinstance(cam, CameraTrack) else None)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return report
