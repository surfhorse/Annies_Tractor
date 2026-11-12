"""Golden-file regression helpers: a small numeric summary of every stage of one shot, and a
comparison against the stored summary with per-key tolerances.

Only numbers, ids, file names and hashes are stored (``golden_shots.json``), never footage or frames.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import afv.export.stage  # registers the export manifest kind
import afv.media.result  # noqa: F401  (registers the ingest result kind)
from afv.core.project import Shot
from afv.core.skeletons import get_skeleton
from afv.core.types import BodyMotion, CameraTrack, Tracks
from afv.export.bvh import read_bvh
from afv.motion.fk import forward_kinematics, rest_positions
from afv.motion.rotations import qconj, qmul, to_rotvec

# (pattern, abs tolerance, rel tolerance); first match wins; no match = exact.
# Calibration (2026-11-10, final round 2 code): the golden values come from one AFV_GOLDEN_UPDATE run per
# shot, then ONE repeat deterministic run per shot (close_boxing, ufc_cross, who_am_i) was compared with
# them. Each tolerance is about 4x the largest run-to-run difference seen for its key pattern over the
# three shots (the observed maximum is noted), with a small floor where the repeat matched exactly
# (detection/tracking, strikes, reprojection were bit-identical; GEM-X/ViPE GPU kernels are not, so the
# camera and motion numbers move slightly). One repeat per shot is a small sample: widen a tolerance only
# with a new repeat measurement, never just to pass. The quantised BVH hash differed in all three repeats
# and is recorded only (INFO_KEYS).
TOLERANCES: list[tuple[str, float, float]] = [
    ("detect_track.coverage.*", 0.01, 0.0),  # observed 0
    ("detect_track.box_center_median.*", 2.0, 0.0),  # observed 0 px
    ("detect_track.box_height_median.*", 2.0, 0.0),  # observed 0 px
    ("detect_track.flagged_frames", 2.0, 0.0),  # observed 0
    ("camera.focal_px", 0.0, 0.025),  # observed 0.6 % (43.7 px of 7300)
    ("camera.rot_travel_deg", 0.2, 0.0),  # observed 0.04 deg
    ("camera.height_m", 0.07, 0.0),  # observed 0.016 m
    ("camera.distance_m", 0.45, 0.0),  # observed 0.107 m (telephoto: ~18 m away)
    ("pose3d.*.height_m", 0.005, 0.0),  # observed 0.0003 m
    ("pose3d.*.root_path_m", 0.05, 0.025),  # observed 0.104 m = 0.6 % (telephoto depth noise)
    ("pose3d.*.root_mean_m", 0.06, 0.0),  # observed 0.013 m
    ("pose3d.*.reprojection_px", 0.5, 0.0),  # observed 0
    ("pose3d.world_shift", 0.015, 0.0),  # observed 0.003 m
    # Round 4 (2026-11-10): close_boxing A's smoothed root path was bimodal (3.972-3.975 / 4.065-4.092 m)
    # because contacts' root solve amplified one-frame plant flips. Round 5 (2026-11-11) made that solve
    # convex (track M); real repeats of the new code (GPU runs + the CPU chain on real pose3d inputs):
    # smooth root path close_boxing A 4.040-4.062, B 3.905-3.948 (spread 0.043 m, the largest), ufc_cross
    # A 5.397-5.434, B 0.005 spread, who_am_i <= 0.010. pair.distance_median_m is not a decision flip:
    # it follows GEM-X's run-to-run depth drift (who_am_i 2.220-2.251 m, continuous), so it keeps 0.045.
    ("pair.distance_median_m", 0.045, 0.0),  # observed 0.031 m (who_am_i, GEM-X depth noise)
    ("contacts.*.contact_frames", 8.0, 0.05),  # observed 10 frames of 671 (1.5 %)
    ("contacts.*.mean_contact_foot_speed_cm_s", 0.8, 0.0),  # observed 0.20 cm/s
    ("contacts.*.max_penetration_cm", 0.2, 0.0),  # observed 0
    ("smooth.*.strikes", 1.0, 0.0),  # observed 0
    ("smooth.*.root_path_m", 0.06, 0.0),  # observed 0.043 m (close_boxing B, round 5; see above)
    ("hands_face.*.metacarpal_p95_deg", 0.5, 0.0),  # observed 0
    ("hands_face.*.metacarpal_max_deg", 0.5, 0.0),  # observed 0.01 deg
    ("fk.*.rel_mean_m", 0.005, 0.0),  # observed 0.001 m
    ("fk.*.rel_p95_m", 0.01, 0.0),  # observed 0.002 m
    # Round 3 (2026-11-10) repeat: ufc A RightShin moved 0.23 / 0.57 deg (mean / p95). Near-straight knees
    # hit the IK's forward/hyperextension choice and the contacts trim rules, which amplify GPU noise.
    ("fk.*Shin.world_angle_mean_deg", 0.9, 0.0),  # observed 0.23 deg
    ("fk.*Shin.world_angle_p95_deg", 2.3, 0.0),  # observed 0.57 deg
    ("fk.*.world_angle_mean_deg", 0.2, 0.0),  # observed 0.03 deg
    ("fk.*.world_angle_p95_deg", 0.25, 0.0),  # observed 0.04 deg
    ("export60.channel_mean_abs", 0.0, 0.002),  # observed 0.04 %
    ("export60.per_channel_mean_abs", 2.0, 0.05),  # cm / degrees per BVH channel; observed 1.06 (1.3 %)
]


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _qc(shot: Shot, stage: str) -> dict[str, Any]:
    p = shot.stage_dir(stage) / "qc.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def _path_len(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(x, np.float64), axis=0), axis=-1).sum())


def summarise_shot(shot: Shot) -> dict[str, Any]:
    """Flat {key: value} summary of the shot's current stage results."""
    s: dict[str, Any] = {}
    media = shot.load_result("ingest")
    sdir = shot.stage_dir("ingest")
    s["ingest.frames"] = media.num_frames
    s["ingest.size"] = [int(media.width), int(media.height)]  # type: ignore[attr-defined]
    s["ingest.shot_mp4_sha256"] = _sha256(sdir / "shot.mp4")

    tr = shot.load_result("detect_track")
    assert isinstance(tr, Tracks)
    s["detect_track.frames"] = tr.num_frames
    s["detect_track.track_ids"] = [int(t) for t in tr.track_ids]
    s["detect_track.fighters"] = dict(sorted(tr.fighters.items()))
    for k, tid in enumerate(tr.track_ids):
        b = np.asarray(tr.boxes[:, k], np.float64)
        ok = np.isfinite(b[:, 0])
        s[f"detect_track.coverage.{tid}"] = _r(ok.mean())
        if ok.any():
            c = (b[ok, :2] + b[ok, 2:]) / 2
            s[f"detect_track.box_center_median.{tid}"] = [_r(v, 1) for v in np.median(c, axis=0)]
            s[f"detect_track.box_height_median.{tid}"] = _r(np.median(b[ok, 3] - b[ok, 1]), 1)
    tq = _qc(shot, "detect_track").get("summary", {})
    s["detect_track.flagged_frames"] = int(tq.get("flagged_frame_count", 0))

    cam = shot.load_result("camera")
    assert isinstance(cam, CameraTrack)
    s["camera.frames"] = cam.num_frames
    s["camera.focal_px"] = _r(np.median(cam.focal_px), 1)
    q = np.asarray(cam.cam_rot, np.float64)
    step = np.linalg.norm(to_rotvec(qmul(qconj(q[:-1]), q[1:])), axis=-1)
    s["camera.rot_travel_deg"] = _r(np.rad2deg(step.sum()), 2)
    s["camera.height_m"] = _r(np.median(cam.cam_pos[:, 1]), 3)
    s["camera.distance_m"] = _r(np.median(np.linalg.norm(cam.cam_pos[:, [0, 2]], axis=-1)), 3)

    pose = shot.load_result("pose3d")
    assert isinstance(pose, BodyMotion)
    s["pose3d.frames"] = pose.num_frames
    s["pose3d.skeleton"] = pose.skeleton
    s["pose3d.world_shift"] = [_r(v, 3) for v in pose.world_shift]
    pq = _qc(shot, "pose3d").get("summary", {})
    sk = get_skeleton(pose.skeleton)
    for role, a in sorted(pose.actors.items()):
        rest_y = rest_positions(sk.with_offsets(np.asarray(a.rest_offsets, np.float64)))[:, 1]
        s[f"pose3d.{role}.height_m"] = _r(rest_y.max() - rest_y.min(), 4)  # T-pose height (bone lengths)
        s[f"pose3d.{role}.root_path_m"] = _r(_path_len(a.root), 3)
        s[f"pose3d.{role}.root_mean_m"] = [_r(v, 3) for v in np.mean(a.root, axis=0)]
        rep = (pq.get(role) or {}).get("reprojection_px")
        if rep is not None:
            s[f"pose3d.{role}.reprojection_px"] = _r(rep, 1)

    pair = shot.load_result("pair")
    assert isinstance(pair, BodyMotion)
    roots = [np.asarray(a.root, np.float64) for _, a in sorted(pair.actors.items())]
    if len(roots) == 2:
        s["pair.distance_median_m"] = _r(np.median(np.linalg.norm(roots[0] - roots[1], axis=-1)), 3)

    cq = _qc(shot, "contacts").get("fighters", {})
    for role, f in sorted(cq.items()):
        for k in ("contact_frames", "mean_contact_foot_speed_cm_s", "max_penetration_cm"):
            s[f"contacts.{role}.{k}"] = _r(f[k], 3)

    sm = shot.load_result("smooth")
    assert isinstance(sm, BodyMotion)
    strikes = _qc(shot, "smooth").get("strikes", {})
    for role, a in sorted(sm.actors.items()):
        s[f"smooth.{role}.strikes"] = len(strikes.get(role, []))
        s[f"smooth.{role}.root_path_m"] = _r(_path_len(a.root), 3)

    s.update(fk_summary(shot.load_result("hands_face")))  # type: ignore[arg-type]

    for role, f in sorted(_qc(shot, "hands_face").get("fighters", {}).items()):
        mc = f.get("gemx_metacarpals") or {}
        if mc:
            s[f"hands_face.{role}.metacarpal_p95_deg"] = mc["p95_deg"]
            s[f"hands_face.{role}.metacarpal_max_deg"] = mc["max_deg"]

    ex = shot.load_result("export")
    s["export.frames"] = ex.num_frames
    s["export.fps"] = float(ex.fps)  # type: ignore[attr-defined]
    s["export.preset"] = ex.preset  # type: ignore[attr-defined]
    files = ex.files  # type: ignore[attr-defined]
    s["export.files"] = {r: sorted(Path(p).name for p in f.values()) for r, f in sorted(files.items())}
    s["export.rig_bones"] = len(ex.rig_bones)  # type: ignore[attr-defined]
    return s


FK_EFFECTORS = ("LeftHand", "RightHand", "LeftFoot", "RightFoot", "Head")  # wrists, ankles, head
FK_ROTATIONS = ("Hips", "Chest", "Head", "LeftForeArm", "RightShin")


def fk_summary(bm: BodyMotion) -> dict[str, Any]:
    """Per fighter: effector positions relative to the root (mean vector, p95 distance) and world
    rotation angles of a few joints (mean, p95), from FK of the final motion."""
    out: dict[str, Any] = {}
    sk0 = get_skeleton(bm.skeleton)
    for role, a in sorted(bm.actors.items()):
        sk = sk0.with_offsets(np.asarray(a.rest_offsets, np.float64)) if a.rest_offsets is not None else sk0
        pos, wrot = forward_kinematics(sk, np.asarray(a.rot, np.float64), np.asarray(a.root, np.float64))
        root = pos[:, 0]
        for n in FK_EFFECTORS:
            rel = pos[:, sk.index(n)] - root
            out[f"fk.{role}.{n}.rel_mean_m"] = [_r(v, 3) for v in rel.mean(axis=0)]
            out[f"fk.{role}.{n}.rel_p95_m"] = _r(np.percentile(np.linalg.norm(rel, axis=-1), 95), 3)
        for n in FK_ROTATIONS:
            ang = np.rad2deg(np.linalg.norm(to_rotvec(wrot[:, sk.index(n)]), axis=-1))
            out[f"fk.{role}.{n}.world_angle_mean_deg"] = _r(ang.mean(), 2)
            out[f"fk.{role}.{n}.world_angle_p95_deg"] = _r(np.percentile(ang, 95), 2)
    return out


def export60_summary(shot: Shot, name: str) -> dict[str, Any]:
    """The 60 fps BVH export (``<name>_A.bvh``): frame count, a hash of its channels quantised to 0.1
    (BVH is in cm and degrees), their mean magnitude, and the mean magnitude of every channel.

    The hash is recorded but NOT compared (``INFO_KEYS``): two deterministic runs of close_boxing gave
    different hashes (GEM-X/ViPE GPU kernels are not bit-exact, and any 0.05 rounding boundary flips
    it). The per-channel means carry the regression check with a tolerance instead."""
    bvh = read_bvh(shot.exports_dir / f"{name}_A.bvh")
    q = np.round(np.asarray(bvh.motion, np.float64), 1) + 0.0  # +0.0: no -0.0 in the hash
    return {
        "export60.bvh_frames": len(bvh.motion),
        "export60.fps": round(float(bvh.fps), 6),
        "export60.channels_q01_sha256": hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(),
        "export60.channel_mean_abs": _r(np.abs(q).mean(), 4),
        "export60.per_channel_mean_abs": [_r(v, 2) for v in np.abs(q).mean(axis=0)],
    }


# Recorded for inspection, never compared (not reproducible bit for bit across GPU runs).
INFO_KEYS: tuple[str, ...] = ("export60.channels_q01_sha256",)


def _tol(key: str) -> tuple[float, float] | None:
    for pat, a, r in TOLERANCES:
        if fnmatch.fnmatchcase(key, pat):
            return a, r
    return None


def _close(got: Any, want: Any, tol: tuple[float, float] | None) -> bool:
    if isinstance(want, list) and isinstance(got, list):
        return len(got) == len(want) and all(_close(g, w, tol) for g, w in zip(got, want, strict=True))
    if isinstance(want, dict) and isinstance(got, dict):
        return got.keys() == want.keys() and all(_close(got[k], want[k], tol) for k in want)
    if tol is not None and isinstance(want, int | float) and isinstance(got, int | float):
        a, r = tol
        return math.isfinite(got) and abs(got - want) <= a + r * abs(want)
    return got == want


def compare(got: dict[str, Any], want: dict[str, Any]) -> list[str]:
    """Readable differences (empty = within tolerance). Keys missing on either side are differences."""
    out = []
    for k in sorted(set(got) | set(want)):
        if k in INFO_KEYS:
            continue
        if k not in got:
            out.append(f"{k}: missing (golden {want[k]!r})")
        elif k not in want:
            out.append(f"{k}: new key (got {got[k]!r}); update the golden file")
        elif not _close(got[k], want[k], _tol(k)):
            out.append(f"{k}: got {got[k]!r}, golden {want[k]!r} (tolerance {_tol(k) or 'exact'})")
    return out
