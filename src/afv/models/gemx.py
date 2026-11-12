"""GEM-X + SOMA adapter: worker output -> ``BodyMotion`` on the ``soma77`` skeleton; ``pose3d`` stage.

SOMA facts (py-soma-x 0.3.3, as configured by GEM-X: reference pose 0.2.0, procedural twist on)
-------------------------------------------------------------------------------------------------
* 77 public joints, root ``Hips`` (SOMA's virtual ``Root`` is dropped; it is always identity).
  ``transl`` is exactly the Hips world position; ``global_orient`` is the Hips rotation relative
  to its T-pose orientation.
* Zero pose = T-pose: Y up, faces +Z, character's left = +X, metres. SOMA joint frames are not
  world-aligned in the T-pose (bone-aligned axes, "joint orient").
* Bone lengths come from identity (128 PCA coeffs) + 56 local bone scales; GEM-X predicts them per
  frame but they are constant per shot (avgbeta); SOMA's adapter averages them over the shot.

soma77 conventions in afv
-------------------------
Rest pose = SOMA's T-pose with **identity rest rotations**: ``rest_offsets[j]`` is the world-aligned
T-pose bone vector (the fighter's own lengths), ``rest_offsets[root]`` = (0, hip height above the
T-pose sole, 0). A local rotation is the joint's rotation relative to its T-pose orientation:
``A[j] = W[j] @ W0[j]^T`` (W = SOMA joint world rotation, W0 = its T-pose orientation) and
``local[j] = A[parent]^T @ A[j]``. Then ``afv.motion.fk`` reproduces SOMA's joints exactly, because
SOMA's FK is rigid in its joint frames: ``p[j] = p[parent] + W[parent] l[j]`` with
``W0[parent] l[j] = rest_offsets[j]``.

Contacts: GEM-X's 6 ``static_conf_logits`` -> sigmoid probabilities on channels named by the
soma77 joint they refer to (``GEMX_CONTACT_CHANNELS``). Per-joint confidence = GEM-X's ViTPose77
keypoint confidence (same 77-joint order), clipped to [0, 1], times ``UNOBSERVED_CONF_WEIGHT`` (0)
on frames whose box was interpolated or held (spans listed in qc.json).

Floor: the camera stage's ground is a box-height prior. pose3d re-fits the floor from GEM-X's
solved feet (contact-weighted median over both fighters), moves the world so the floor is y = 0
and records the move in ``BodyMotion.world_shift`` (reproject with ``cam_pos + world_shift``).

``ActorMotion.root_prior``: GEM-X's own world-rollout hips trajectory in the same world (one
best-fit yaw + translation per fighter, then the same floor shift) - smooth, a velocity prior for
the pair stage's depth.

World: the shot world of the camera stage (``afv.models.camera``). ``root_mode="camera"``
maps GEM-X's camera-frame body into it through the camera, so every joint projects exactly where
GEM-X's own overlay draws it; depth along the view is GEM-X's (noisy for telephoto shots; the pair
stage smooths it). ``root_mode="gemx_world"`` instead uses GEM-X's world rollout, aligned to the
shot world by one yaw + translation per fighter (smoother, but drifts off the image).
``root_mode="auto"`` (the pose3d default) is "camera" unless the camera-based root moves across the
camera ray far faster than GEM-X's rollout (``decide_root_mode``), which means the camera's
rotations are wrong: over most of the shot, the whole shot falls back to "gemx_world"; within
1 s windows elsewhere, only those frames follow the rollout's velocity, pinned to the camera root
where they meet it (``prior_root_in_spans``; flagged in qc.json either way). The camera root's
lateral motion is distance x camera angle, so it is only as good as the camera's rotations: the
camera stage measures them from the background (``afv.models.camera.flow_rotations``) because
ViPE's were wrong on telephoto shots.

Zoom: the camera's per-frame ``focal_px`` reaches GEM-X as its per-frame K (``K_fullimg``), so GEM-X
reads a zoom as a zoom, not as the fighter approaching. ``zoom_corrected_incam`` still corrects the
depth of results solved with another focal (identity when GEM-X had the camera's K).

GEM-X's raw output is cached per shot (``GemxOutputCache``): the gravity re-level pass changes the
shot world by one rotation, which GEM-X never sees (it is given rotations relative to camera 0,
``afv.models.camera.gemx_network_rotations``), so that pass reuses the first pass's network run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, Field

from afv.core.skeletons import get_skeleton
from afv.core.stage import CacheOnlyMiss, Ctx, StageError, register
from afv.core.types import ActorMotion, BodyMotion, CameraTrack, LicenceRecord, Result, Skeleton, Tracks
from afv.models.camera import (
    SHOT_FPS,
    WORKERS,
    camera_to_world_cv,
    clean_mask,
    env_fingerprint,
    feet_data_malformed,
    feet_refute_relevel,
    feet_tilt_after_relevel,
    ffmpeg_identity,
    file_sha256,
    gemx_network_rotations,
    gemx_provided_camera,
    gravity_disagrees,
    leveled_track,
    mask_spans,
    project,
    relevel_fraction,
    rolling_nanmedian,
    run_env_worker,
    shifted_camera,
    shot_source,
    shot_timeline,
    source_identity,
)
from afv.motion.fk import forward_kinematics, rest_positions
from afv.motion.rotations import from_matrix, make_continuous

SOMA77 = "soma77"
# GEM-X static_conf_logits order and the soma77 joint each channel refers to
# (gem.motion.postprocess.contact.SOMA_CONTACT_JOINT_IDS = 69, 70, 74, 75, 14, 42).
GEMX_CONTACT_LOGITS: tuple[str, ...] = (
    "left_ankle",
    "left_foot",
    "right_ankle",
    "right_foot",
    "left_wrist",
    "right_wrist",
)
GEMX_CONTACT_CHANNELS: tuple[str, ...] = (
    "LeftFoot",
    "LeftToeBase",
    "RightFoot",
    "RightToeBase",
    "LeftHand",
    "RightHand",
)
_Y = np.array([0.0, 1.0, 0.0])
# Foot joints whose contact channels anchor the floor (GEM-X's ankle/foot channels).
FLOOR_JOINTS: tuple[str, ...] = GEMX_CONTACT_CHANNELS[:4]
# Confidence multiplier on frames whose box was interpolated/held (no SAM 2 mask): the 2D
# evidence there is ViTPose on a guessed crop.
UNOBSERVED_CONF_WEIGHT = 0.0
# Floor scale (``scale_to_floor``). The stage and ``body_motion_from_fk`` share the default.
FLOOR_SCALE_MAX = 0.05
# Contact weight (GEM-X ankle/foot contact probability summed over the 4 foot channels and observed
# frames) a fighter needs before it is scaled. A planted foot gives ~2 per frame (heel + toe), so 10 is
# about one plant of 5 frames (1/6 s). Its weighted median sole height then has a standard error of
# ~1.25 sigma / sqrt(10) = 0.4 sigma, i.e. 1.5-2 cm for the 3-5 cm per-frame sole noise of telephoto
# shots: well under the 4-15 cm gaps it corrects, so the scale follows the fighter, not noise.
FLOOR_SCALE_MIN_CONTACT = 10.0

# Root/prior consistency: the camera-based root's speed across the camera ray vs GEM-X's own rollout,
# both over LATERAL_WINDOW frames (per-frame speeds are dominated by GEM-X's 2D jitter).
# Measured on the Media Samples (0.5 s windows): with background-flow rotations the camera root is
# 0.9-2.2x the prior and follows the 2D hips (the rollout under-moves); with P0's wrong ViPE
# rotations it was 2.8-5.4x. Flag (and, with root_mode="auto", fall back) above both thresholds.
LATERAL_WINDOW = 15
LATERAL_RATIO_FLAG = 3.0
LATERAL_EXCESS_FLAG = 0.5  # m/s
# root_mode="auto" also decides per window: rolling medians over LATERAL_DECISION_WINDOW frames of
# those speeds (pooled over the fighters); disagreeing spans of at least LOCAL_FALLBACK_MIN_FRAMES
# follow the rollout locally, more than LOCAL_FALLBACK_MAX_FRACTION of the shot falls back entirely.
# On the Media Samples (round 3, flow rotations) no window qualifies; ref walk 001 frames ~150-180
# disagree, but for fighter A only (GEM-X's rollout under-moving a fast walk, see below).
LATERAL_DECISION_WINDOW = 30
LOCAL_FALLBACK_MIN_FRAMES = 15
LOCAL_FALLBACK_MAX_FRACTION = 0.5
# A window (and the shot-level test) only counts when every fighter shows at least this fraction of the
# largest angular excess (a camera rotation error is shared; ref walk 001 frames 140-170: A 0.04 rad/s,
# B ~0.002, GEM-X's rollout under-moving A). Not bounded by the camera's own rotation rate: an estimate
# panning the wrong way slides the root by |w_est| + |w_true| (round 2: ViPE on Who Am I).
SHARED_EXCESS_FRACTION = 0.5

# Weak 2D evidence (``weak_2d_frames``): GEM-X's depth comes from the body's image size, so on a
# truncated or occluded body it is a guess (Close boxing B folded over the ropes, frames 81-91: depth
# running from 21 to 25 m). A frame is weak when fewer than WEAK_BODY_FRACTION of DEPTH_BODY_JOINTS
# (torso and legs: what sets the body's size) have keypoint confidence > 0.5 (5-frame rolling median),
# unless the joints that are seen pin GEM-X's body (``weak_2d_frames``), or its box was
# interpolated/held. pose3d does not move the body there: it publishes the mask as
# ``ActorMotion.weak`` (``conf`` stays GEM-X's, so visible limbs keep their weight downstream) and
# lists the spans in qc.json (``weak_2d_spans``); the pair stage's depth solve then lets those frames
# follow GEM-X's rollout and their neighbours (``afv.motion.pair``). Media Samples: Close boxing A
# 34-37 and 78-81, B 81-91; ref walk 001 B 16-18; nothing elsewhere.
DEPTH_BODY_JOINTS: tuple[str, ...] = (
    "Hips",
    "Spine2",
    "Chest",
    "Neck1",
    "Head",
    "LeftArm",
    "RightArm",
    "LeftLeg",
    "RightLeg",
    "LeftShin",
    "RightShin",
    "LeftFoot",
    "RightFoot",
)
WEAK_BODY_FRACTION = 0.5
GOOD_FIT_MIN_JOINTS = 8
GOOD_FIT_RATIO = 1.5


# ---------------------------------------------------------------------------------------------
# Worker output
# ---------------------------------------------------------------------------------------------


@dataclass
class SomaFK:
    """One fighter's ``soma_worker`` FK NPZ (see that module for the exact fields)."""

    joint_names: tuple[str, ...]
    parents: tuple[int, ...]
    incam_rot: np.ndarray  # [T,J,3,3] OpenCV camera frame
    incam_pos: np.ndarray  # [T,J,3]
    global_rot: np.ndarray  # [T,J,3,3] GEM-X world (Y up)
    global_pos: np.ndarray
    tpose_pos: np.ndarray  # [J,3] hips at origin
    tpose_rot: np.ndarray  # [J,3,3]
    sole_y: float
    contact_logits: np.ndarray  # [T,6]
    kp2d: np.ndarray  # [T,J,3]
    K: np.ndarray  # [T,3,3]
    # [T,3,3] the rotations GEM-X was given: since pose3d v4 relative to camera 0
    # (``afv.models.camera.gemx_network_rotations``; so is ``camera.R_w2c`` in motion.pt), not the
    # shot world's. Shot-world cameras come from the CameraTrack.
    R_w2c: np.ndarray
    observed: np.ndarray  # [T] bool
    meta: dict[str, Any] = field(default_factory=dict)
    boxes_xyxy: np.ndarray | None = None  # [T,4] the (clipped, filled) boxes GEM-X cropped from

    @property
    def num_frames(self) -> int:
        return int(self.incam_pos.shape[0])

    @classmethod
    def load(cls, path: str | Path) -> SomaFK:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            a = {k: z[k] for k in z.files if k != "meta"}
        f64 = lambda k: np.asarray(a[k], dtype=np.float64)  # noqa: E731
        return cls(
            joint_names=tuple(meta["joint_names"]),
            parents=tuple(int(p) for p in meta["parents"]),
            incam_rot=f64("incam_rot"),
            incam_pos=f64("incam_pos"),
            global_rot=f64("global_rot"),
            global_pos=f64("global_pos"),
            tpose_pos=f64("tpose_pos"),
            tpose_rot=f64("tpose_rot"),
            sole_y=float(a["sole_y"]),
            contact_logits=f64("contact_logits"),
            kp2d=f64("kp2d"),
            K=f64("K"),
            R_w2c=f64("R_w2c"),
            observed=np.asarray(a["observed"], dtype=bool),
            meta=meta,
            boxes_xyxy=f64("boxes_xyxy") if "boxes_xyxy" in a else None,
        )


# ---------------------------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------------------------


def soma77() -> Skeleton:
    return get_skeleton(SOMA77)


def check_joints(fk: SomaFK, sk: Skeleton) -> None:
    if tuple(fk.joint_names) != tuple(sk.joint_names) or tuple(fk.parents) != tuple(sk.parents):
        raise StageError(f"SOMA worker joints do not match skeleton {sk.name}")
    if sk.rest_rotations is not None and not np.allclose(sk.rest_rotations, [0, 0, 0, 1]):
        raise StageError(f"{sk.name}: conversion assumes identity rest rotations")


def fighter_rest_offsets(fk: SomaFK) -> np.ndarray:
    """This fighter's bone offsets [J,3]: world-aligned T-pose bone vectors (shot-locked lengths)."""
    off = np.empty_like(fk.tpose_pos)
    for j, p in enumerate(fk.parents):
        off[j] = fk.tpose_pos[j] - fk.tpose_pos[p] if p >= 0 else (0.0, -fk.sole_y, 0.0)
    return off


def world_to_local(world_rot: np.ndarray, tpose_rot: np.ndarray, parents: tuple[int, ...]) -> np.ndarray:
    """SOMA joint world rotations [T,J,3,3] -> afv local xyzw quaternions [T,J,4] (identity rest)."""
    rel = np.einsum("tjab,jcb->tjac", world_rot, tpose_rot)  # A = W @ W0^T
    local = np.empty_like(rel)
    for j, p in enumerate(parents):
        local[:, j] = rel[:, j] if p < 0 else np.swapaxes(rel[:, p], -1, -2) @ rel[:, j]
    return make_continuous(from_matrix(local), axis=0)


def zoom_corrected_incam(fk: SomaFK, cam: CameraTrack) -> np.ndarray:
    """GEM-X's camera-frame joints [T,J,3] with its depth corrected for a zoom.

    GEM-X solves with one constant K (focal f0); when the camera's focal ``f(t)`` differs (a measured
    zoom) its depth from the body's image size is off by ``f(t) / f0`` while x/z is right for f0. The
    body is moved along the optical axis by ``z_root (f(t)/f0 - 1)``: the root then projects to the
    same pixel with f(t) as GEM-X's with f0 and the body keeps its size on the image. Identity when
    the camera's focal is GEM-X's.
    """
    pos = np.array(fk.incam_pos, dtype=np.float64)
    f_cam = np.broadcast_to(np.asarray(cam.focal_px, dtype=np.float64), (fk.num_frames,))
    f_gemx = 0.5 * (fk.K[:, 0, 0] + fk.K[:, 1, 1])
    ratio = f_cam / f_gemx
    if np.allclose(ratio, 1.0, rtol=1e-4, atol=0.0):
        return pos
    pos[..., 2] += (pos[:, 0, 2] * (ratio - 1.0))[:, None]
    return pos


def incam_to_world(fk: SomaFK, cam: CameraTrack) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame SOMA joints -> shot world through the camera: (rot [T,J,3,3], pos [T,J,3]).
    Depth is corrected for a zoom (``zoom_corrected_incam``)."""
    if cam.num_frames != fk.num_frames:
        raise StageError(f"camera has {cam.num_frames} frames, GEM-X result has {fk.num_frames}")
    r, c = camera_to_world_cv(cam)
    rot = np.einsum("tab,tjbc->tjac", r, fk.incam_rot)
    pos = np.einsum("tab,tjb->tja", r, zoom_corrected_incam(fk, cam)) + c[:, None]
    return rot, pos


def _yaw_only(m: np.ndarray) -> np.ndarray:
    """Closest rotation about +Y to a rotation matrix."""
    yaw = np.arctan2(m[0, 2] - m[2, 0], m[0, 0] + m[2, 2])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def gemx_world_to_world(fk: SomaFK, cam: CameraTrack) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """GEM-X world rollout placed in the shot world by one yaw + translation per fighter.

    Yaw from the hips orientations (robust when a boxer barely moves), translation = median offset
    of the hips trajectory from its camera-consistent position.
    """
    cam_rot, cam_pos = incam_to_world(fk, cam)
    m = np.einsum("tab,tcb->tac", cam_rot[:, 0], fk.global_rot[:, 0])  # R_{W<-G} per frame
    u, _, vt = np.linalg.svd(m.mean(0))
    r_wg = _yaw_only(u @ vt)
    t = np.median(cam_pos[:, 0] - fk.global_pos[:, 0] @ r_wg.T, axis=0)
    rot = np.einsum("ab,tjbc->tjac", r_wg, fk.global_rot)
    pos = fk.global_pos @ r_wg.T + t
    yaw_deg = float(np.degrees(np.arctan2(r_wg[0, 2], r_wg[0, 0])))
    return rot, pos, {"yaw_deg": yaw_deg, "offset_m": [float(v) for v in t]}


def contact_probabilities(logits: np.ndarray) -> np.ndarray:
    """Sigmoid of GEM-X's contact logits, numerically safe."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(logits, dtype=np.float64)))


def observed_weight(observed: np.ndarray) -> np.ndarray:
    """Per-frame confidence multiplier: 1 where the box was observed, ``UNOBSERVED_CONF_WEIGHT`` else."""
    return np.where(np.asarray(observed, dtype=bool), 1.0, UNOBSERVED_CONF_WEIGHT)


def actor_from_world(fk: SomaFK, world_rot: np.ndarray, world_pos: np.ndarray, sk: Skeleton) -> ActorMotion:
    """``ActorMotion`` on ``sk`` from SOMA joint world rotations/positions in the afv world."""
    check_joints(fk, sk)
    return ActorMotion(
        rot=world_to_local(world_rot, fk.tpose_rot, sk.parents).astype(np.float32),
        root=world_pos[:, 0].astype(np.float32),
        conf=(np.clip(fk.kp2d[..., 2], 0.0, 1.0) * observed_weight(fk.observed)[:, None]).astype(np.float32),
        contact=contact_probabilities(fk.contact_logits).astype(np.float32),
        rest_offsets=fighter_rest_offsets(fk).astype(np.float32),
    )


def actor_from_fk(
    fk: SomaFK,
    cam: CameraTrack,
    sk: Skeleton | None = None,
    *,
    root_mode: str = "camera",
    local_spans: list[list[int]] | None = None,
) -> tuple[ActorMotion, dict[str, Any]]:
    """One fighter in the shot world, plus ``root_prior``: GEM-X's world-rollout hips trajectory placed
    in the same world by ONE best-fit rigid yaw (from the hips orientations over the shot) +
    translation (median offset to the camera-based hips), so its velocities are directly comparable
    with ``root``'s. It drifts against the camera but is temporally smooth (a velocity prior).

    ``local_spans`` (inclusive frame spans, camera mode): the root follows ``root_prior``'s velocity
    there (``prior_root_in_spans``), the rest of the body moving with it."""
    sk = sk or soma77()
    if root_mode not in ("camera", "gemx_world", "auto"):
        raise ValueError(f"unknown root_mode {root_mode!r}")
    if root_mode == "auto":  # one fighter: the shot-level decision on its own
        used, evidence = decide_root_mode({"_": fk}, cam)
        actor, info = actor_from_fk(fk, cam, sk, root_mode=used, local_spans=evidence["local_fallback_spans"])
        info.update(root_mode="auto", lateral_check=evidence["fighters"]["_"])
        return actor, info
    info: dict[str, Any] = {"root_mode": root_mode, "root_mode_used": root_mode}
    g_rot, g_pos, align = gemx_world_to_world(fk, cam)
    info["root_prior_alignment"] = align
    rot, pos = (g_rot, g_pos) if root_mode == "gemx_world" else incam_to_world(fk, cam)
    if root_mode == "gemx_world":
        info["alignment"] = align
    actor = actor_from_world(fk, rot, pos, sk)
    actor.root_prior = g_pos[:, 0].astype(np.float32)
    if local_spans and root_mode == "camera":
        actor.root = prior_root_in_spans(actor.root, actor.root_prior, local_spans).astype(np.float32)
        info["local_root_fallback_spans"] = [list(map(int, s)) for s in local_spans]
    return actor, info


def prior_root_in_spans(root: np.ndarray, prior: np.ndarray, spans: list[list[int]]) -> np.ndarray:
    """``root`` [T,3] with ``prior``'s velocity inside each inclusive frame span: anchored on the last
    frame before and the first frame after the span (the camera root there), the end mismatch spread
    linearly over the span; a span at the start or end of the shot is anchored on its one side."""
    root = np.asarray(root, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    out = root.copy()
    n = len(root)
    for s, e in spans:
        s, e = max(0, int(s)), min(n - 1, int(e))
        a, b = s - 1, e + 1
        if e < s or (a < 0 and b >= n):
            continue
        seg = prior[s : e + 1]
        if a < 0:
            out[s : e + 1] = root[b] + seg - prior[b]
        elif b >= n:
            out[s : e + 1] = root[a] + seg - prior[a]
        else:
            alpha = (np.arange(s, e + 1) - a) / (b - a)
            mismatch = (root[b] - root[a]) - (prior[b] - prior[a])
            out[s : e + 1] = root[a] + seg - prior[a] + alpha[:, None] * mismatch
    return out


def lateral_speeds(fk: SomaFK, cam: CameraTrack) -> tuple[np.ndarray, np.ndarray]:
    """Across-ray speeds (m/s, ``LATERAL_WINDOW``-frame windows) of this fighter's camera-based hips and
    of GEM-X's rollout; index i covers frames i .. i + LATERAL_WINDOW."""
    _, centre = camera_to_world_cv(cam)
    _, pos = incam_to_world(fk, cam)
    _, g_pos, _ = gemx_world_to_world(fk, cam)
    return lateral_speed(pos[:, 0], centre), lateral_speed(g_pos[:, 0], centre)


def camera_root_lateral(fk: SomaFK, cam: CameraTrack) -> dict[str, float]:
    """Median across-ray speeds (m/s) of this fighter's camera-based hips and of GEM-X's rollout."""
    lat, prior = lateral_speeds(fk, cam)
    return {
        "camera_m_s": round(float(np.median(lat)), 3) if len(lat) else 0.0,
        "prior_m_s": round(float(np.median(prior)), 3) if len(prior) else 0.0,
    }


def disagreeing_frames(
    camera_lateral: np.ndarray,
    prior_lateral: np.ndarray,
    n_frames: int,
    shared: np.ndarray | None = None,
    *,
    window: int = LATERAL_DECISION_WINDOW,
    min_len: int = LOCAL_FALLBACK_MIN_FRAMES,
) -> np.ndarray:
    """Frames [n_frames] where the camera root's lateral speed disagrees with the rollout's
    (``lateral_disagrees`` on centred ``window`` rolling medians of the ``lateral_speed`` series, each
    index marking the frames its speed spans). ``shared`` [same length] limits it to the windows where
    every fighter shows the excess (``shared_excess``): otherwise one fighter's hips really move across
    the image and GEM-X's rollout under-moves them (ref walk 001, frames 140-170: A 1.4 m/s vs 0.45 m/s,
    B in agreement). Gaps up to ``window / 2`` are closed and spans shorter than ``min_len`` frames
    dropped."""
    out = np.zeros(n_frames, bool)
    m = len(camera_lateral)
    if m == 0:
        return out
    mc, mp = rolling_nanmedian(camera_lateral, window), rolling_nanmedian(prior_lateral, window)
    with np.errstate(invalid="ignore"):
        bad = (mc > LATERAL_RATIO_FLAG * mp) & (mc - mp > LATERAL_EXCESS_FLAG)
    if shared is not None:
        bad &= np.asarray(shared, dtype=bool)
    span = n_frames - m  # frames per speed sample - 1
    for i in np.flatnonzero(bad):
        out[i : i + span + 1] = True
    return clean_mask(out, merge_gap=max(1, int(window)) // 2, min_len=min_len)


def shared_excess(
    lats: list[np.ndarray],
    priors: list[np.ndarray],
    distances: list[np.ndarray],
    *,
    window: int = LATERAL_DECISION_WINDOW,
    fraction: float = SHARED_EXCESS_FRACTION,
) -> np.ndarray:
    """Windows (``lateral_speed`` indices) where every fighter's camera root out-runs its rollout by an
    angular rate (excess m/s over distance, rolling medians) of at least ``fraction`` of the largest:
    a camera rotation error adds the same angular rate to everyone, a fighter GEM-X's rollout
    under-moves adds it to that fighter only. One fighter: everywhere."""
    if len(lats) < 2:
        return np.ones(len(lats[0]) if lats else 0, bool)
    rate = np.array(
        [
            rolling_nanmedian((lat - prior) / np.maximum(dist, 1e-3), window)
            for lat, prior, dist in zip(lats, priors, distances, strict=True)
        ]
    )
    top = rate.max(axis=0)
    with np.errstate(invalid="ignore"):
        return (top > 0.0) & (rate.min(axis=0) >= fraction * top)


def decide_root_mode(fks: dict[str, SomaFK], cam: CameraTrack) -> tuple[str, dict[str, Any]]:
    """``root_mode="auto"`` for a whole shot: "camera", or "gemx_world" for EVERY fighter when the
    camera-based roots move across the ray far faster than GEM-X's rollouts and the camera can be the
    cause, either over the shot (``lateral_disagrees`` on the fighters' mean speeds) or over more than
    ``LOCAL_FALLBACK_MAX_FRACTION`` of it in 1 s windows (``disagreeing_frames``). Shorter disagreeing
    windows are fixed locally: "camera" with ``local_fallback_spans`` (``prior_root_in_spans``).

    "The camera can be the cause": every fighter shows the same angular excess (``shared_excess``): a
    wrong camera rotation slides everyone by the same angle, whichever way it turns, while GEM-X's
    rollout under-moving one fighter does not (``not_the_camera`` in the evidence; one fighter: no
    cross-check). One decision keeps the fighters in one world. Returns the mode and the evidence (per
    fighter and pooled)."""
    per, lats, priors, dists = {}, [], [], []
    _, centre = camera_to_world_cv(cam)
    for role, fk in fks.items():
        lat, prior = lateral_speeds(fk, cam)
        lats.append(lat)
        priors.append(prior)
        _, pos = incam_to_world(fk, cam)
        dists.append(np.linalg.norm(pos[:, 0] - centre, axis=1)[: len(lat)])
        per[role] = {
            "camera_m_s": round(float(np.median(lat)), 3) if len(lat) else 0.0,
            "prior_m_s": round(float(np.median(prior)), 3) if len(prior) else 0.0,
        }
    cam_m = float(np.mean([v["camera_m_s"] for v in per.values()])) if per else 0.0
    prior_m = float(np.mean([v["prior_m_s"] for v in per.values()])) if per else 0.0
    n = cam.num_frames
    bad = np.zeros(n, bool)
    if lats and len(lats[0]):
        bad = disagreeing_frames(
            np.mean(lats, axis=0),
            np.mean(priors, axis=0),
            n,
            shared_excess(lats, priors, dists),
        )
    # the shot-level test, with the same sharing condition on the shot's medians
    rates = [
        (v["camera_m_s"] - v["prior_m_s"]) / max(float(np.median(d)) if len(d) else 0.0, 1e-3)
        for v, d in zip(per.values(), dists, strict=True)
    ]
    shared = len(rates) < 2 or (max(rates) > 0.0 and min(rates) >= SHARED_EXCESS_FRACTION * max(rates))
    shot_disagrees = lateral_disagrees(cam_m, prior_m)
    mostly = n > 0 and bad.mean() > LOCAL_FALLBACK_MAX_FRACTION
    whole = (shot_disagrees and shared) or mostly
    used = "gemx_world" if whole else "camera"
    spans = [] if whole else mask_spans(bad)
    return used, {
        "used": used,
        "camera_m_s": round(cam_m, 3),
        "prior_m_s": round(prior_m, 3),
        **({"not_the_camera": True} if shot_disagrees and not whole else {}),
        "disagreeing_fraction": round(float(bad.mean()), 3) if n else 0.0,
        "local_fallback_spans": spans,
        "fighters": per,
    }


def weak_2d_frames(fk: SomaFK) -> np.ndarray:
    """Frames [T] where this fighter's 2D evidence is too weak for GEM-X's depth (see
    ``WEAK_BODY_FRACTION``): few torso and leg joints seen and GEM-X's body not pinned by the joints
    that are seen (``GOOD_FIT_MIN_JOINTS`` seen joints fitted within ``GOOD_FIT_RATIO`` x the fighter's
    usual fit: GEM-X's body size is fixed per shot, so a good fit of several joints fixes the depth;
    UFC cross B 202-206, partly hidden by A: 11 joints at 8 px), or an interpolated box. Gaps up to 3
    frames are closed, blips under 3 frames dropped."""
    sk = soma77()
    idx = [sk.index(n) for n in DEPTH_BODY_JOINTS]
    kp = np.asarray(fk.kp2d, dtype=np.float64)
    frac = rolling_nanmedian((kp[:, idx, 2] > 0.5).mean(axis=1), 5)
    inc = np.asarray(fk.incam_pos, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = np.stack(
            [
                fk.K[:, None, 0, 0] * inc[..., 0] / inc[..., 2] + fk.K[:, None, 0, 2],
                fk.K[:, None, 1, 1] * inc[..., 1] / inc[..., 2] + fk.K[:, None, 1, 2],
            ],
            axis=-1,
        )
    err = np.linalg.norm(uv - kp[..., :2], axis=-1)  # GEM-X's own 2D fit (pose3d's projection)
    seen = kp[..., 2] > 0.5
    n_seen = seen.sum(axis=1)
    fit = np.array([np.median(err[t][seen[t]]) if n_seen[t] else np.nan for t in range(len(n_seen))])
    usual = fit[(n_seen >= GOOD_FIT_MIN_JOINTS) & (frac >= WEAK_BODY_FRACTION)]
    good_fit = np.zeros(len(fit), bool)
    if np.isfinite(usual).any():
        with np.errstate(invalid="ignore"):
            good_fit = (n_seen >= GOOD_FIT_MIN_JOINTS) & (fit <= GOOD_FIT_RATIO * float(np.nanmedian(usual)))
    weak = ((frac < WEAK_BODY_FRACTION) & ~good_fit) | ~np.asarray(fk.observed, dtype=bool)
    return clean_mask(weak, merge_gap=3, min_len=3)


def body_motion_from_fk(
    fks: dict[str, SomaFK],
    cam: CameraTrack,
    *,
    root_mode: str = "camera",
    refit_floor: bool = True,
    floor_scale_max: float = FLOOR_SCALE_MAX,
    frames: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> tuple[BodyMotion, dict[str, Any]]:
    """Both fighters on soma77, plus QC ``{"floor": ..., "fighters": {role: ...}}``.

    Frames with weak 2D evidence (``weak_2d_frames``) are published as ``ActorMotion.weak`` (QC
    ``weak_2d_spans``) and left out of the floor fit; their depth is for the pair stage to fix.

    ``floor_scale_max`` > 0: fighters whose soles disagree with the shared floor are scaled about the
    camera centre (``scale_to_floor``, at most that fraction), which keeps every joint's image exactly.

    With ``refit_floor`` the floor is re-fitted from GEM-X's solved feet (both fighters together)
    and the world is moved so it is y = 0: ``BodyMotion.world_shift`` = that move relative to the
    camera stage's world (reproject with ``shifted_camera(cam, motion.world_shift)``).
    """
    sk = soma77()
    actors, info = {}, {}
    decision: dict[str, Any] | None = None
    mode = root_mode
    if root_mode == "auto":
        mode, decision = decide_root_mode(fks, cam)
    spans = decision["local_fallback_spans"] if decision is not None else None
    for role, fk in fks.items():
        actors[role], info[role] = actor_from_fk(fk, cam, sk, root_mode=mode, local_spans=spans)
        if decision is not None:
            info[role].update(root_mode="auto", lateral_check=decision["fighters"][role])
    observed = {}
    for role, fk in fks.items():  # publish the weak mask (ActorMotion.weak; pair reads it)
        weak = weak_2d_frames(fk)
        actors[role].weak = weak
        info[role]["weak_2d_spans"] = mask_spans(weak)
        observed[role] = np.asarray(fk.observed, dtype=bool) & ~weak
    shift = np.zeros(3)
    floor: dict[str, Any] = {"refit": False}
    if refit_floor:
        floor_y, floor = fit_floor(actors, observed, sk)
        if floor_scale_max > 0.0 and mode != "gemx_world":
            floor["scale"] = scale_to_floor(actors, floor, cam, sk, floor_scale_max, observed)
            for role, sc in floor["scale"].items():
                if sc["scale"] != 1.0:  # these were measured on the unscaled fighter
                    for k in ("root_prior_alignment", "lateral_check"):
                        if isinstance(info[role].get(k), dict):
                            info[role][k] = {**info[role][k], "before_scale": True}
            if any(v.get("scale", 1.0) != 1.0 for v in floor["scale"].values()):
                before = floor
                floor_y, floor = fit_floor(actors, observed, sk)
                floor |= {
                    "scale": before["scale"],
                    "before_scale": {
                        k: before.get(k)
                        for k in (
                            "per_fighter_floor_m",
                            "fighter_floor_spread_m",
                            "floor_y_in_camera_world_m",
                        )
                    },
                }
        shift[1] = -floor_y
        for a in actors.values():
            a.root = (np.asarray(a.root, dtype=np.float64) + shift).astype(np.float32)
            a.root_prior = (np.asarray(a.root_prior, dtype=np.float64) + shift).astype(np.float32)
    motion = BodyMotion(
        frames=cam.frames if frames is None else frames,
        timestamps=cam.timestamps if timestamps is None else timestamps,
        skeleton=sk.name,
        fps=SHOT_FPS,
        contact_channels=list(GEMX_CONTACT_CHANNELS),
        actors=actors,
        world_shift=[float(v) for v in shift],
    )
    if motion.num_frames != cam.num_frames:
        raise StageError("BodyMotion timeline does not match the camera")
    view = shifted_camera(cam, motion.world_shift)
    fighters = {
        role: {
            **info[role],
            **fighter_qc(fks[role], view, actors[role], sk),
            "unobserved_spans": unobserved_spans(fks[role].observed),
        }
        for role in fks
    }
    out: dict[str, Any] = {"floor": floor, "fighters": fighters, "gravity": gravity_check(fks, view, actors)}
    if decision is not None:
        out["root_mode"] = {"requested": "auto", **{k: v for k, v in decision.items() if k != "fighters"}}
    return motion, out


def gravity_check(fks: dict[str, SomaFK], cam: CameraTrack, actors: dict[str, ActorMotion]) -> dict[str, Any]:
    """Camera height above the floor and how GEM-X's gravity (people stand upright) disagrees with the
    camera stage's in PITCH, which is what decides that height at telephoto range: a pitch error
    d tilts the floor under the camera by distance x tan(d) (1 deg at 20 m = 0.35 m). GeoCalib's
    pitch is weakly observed on telephoto frames (stated 3-6 deg per frame on the Media Samples).
    ``cam`` must be in the motion's world (``shifted_camera``).

    Per frame, the camera's down-pitch against the shot's up and against GEM-X's up; the difference
    is the median of the PER-FRAME differences over every fighter's observed frames (pooled), and the
    reported pitches are medians over that same frame set (+ = GEM-X says the camera looks further
    down, i.e. sits higher). Also: the height that difference implies at the fighters' median distance,
    the roll difference (reported, not acted on), and ``gemx_up_cam0`` (GEM-X's up in camera 0's
    OpenCV axes, for ``afv.models.camera.relevel_override``). Without any usable frame (no fighter
    observed, non-finite GEM-X output) the angles are None and ``error`` says why.

    ``feet_tilt_deg`` (``feet_tilt``): the pitch error the planted feet show, an independent check of
    that difference (same sign: + = the camera should look further down).
    """
    r, c = camera_to_world_cv(cam)
    view, right = r[:, :, 2], r[:, :, 0]
    height = float(np.median(np.asarray(cam.cam_pos, dtype=np.float64)[:, 1]))
    cam_down_all = np.degrees(np.arcsin(np.clip(-view[:, 1], -1.0, 1.0)))
    cam_down, gem_down, diff, roll, ups, dists = [], [], [], [], [], []
    for role, fk in fks.items():
        up = gemx_up_in_world(fk, cam)
        with np.errstate(invalid="ignore", divide="ignore"):
            up = up / np.linalg.norm(up, axis=1, keepdims=True)
        ok = np.asarray(fk.observed, dtype=bool) & np.isfinite(up).all(axis=1)
        if not ok.any():
            continue
        g = np.degrees(np.arcsin(np.clip(-np.sum(view * up, axis=1), -1.0, 1.0)))
        cam_down.append(cam_down_all[ok])
        gem_down.append(g[ok])
        diff.append(g[ok] - cam_down_all[ok])
        # roll: GEM-X's up leaning along the camera's right axis vs the shot's (0 for the shot's own up)
        roll.append(np.degrees(np.arcsin(np.clip(np.sum(right * up, axis=1) - right[:, 1], -1.0, 1.0)))[ok])
        ups.append(np.einsum("ba,tb->ta", r[0], up)[ok])  # in camera 0's OpenCV axes
        root = np.asarray(actors[role].root, dtype=np.float64)
        d = np.linalg.norm((root - c)[:, [0, 2]], axis=1)[ok]
        dists.append(d[np.isfinite(d)])
    out: dict[str, Any] = {"camera_height_m": round(height, 3) if np.isfinite(height) else None}
    out |= feet_tilt(fks, cam, actors)
    keys = ("camera_down_pitch_deg", "gemx_down_pitch_deg", "pitch_difference_deg", "roll_difference_deg")
    keys += ("fighter_distance_m", "height_change_if_gemx_m", "gemx_up_cam0", "camera_up_cam0", "frames_used")
    if not diff or not np.concatenate(dists).size:
        return {**out, **dict.fromkeys(keys), "error": "no observed frame with a finite GEM-X gravity"}
    dd = float(np.median(np.concatenate(diff)))
    dist = float(np.median(np.concatenate(dists)))
    up0 = np.median(np.concatenate(ups), axis=0)
    up0 /= np.linalg.norm(up0)
    return {
        **out,
        "camera_down_pitch_deg": round(float(np.median(np.concatenate(cam_down))), 2),
        "gemx_down_pitch_deg": round(float(np.median(np.concatenate(gem_down))), 2),
        "pitch_difference_deg": round(dd, 2),
        "roll_difference_deg": round(float(np.median(np.concatenate(roll))), 2),
        "fighter_distance_m": round(dist, 2),
        "height_change_if_gemx_m": round(dist * float(np.tan(np.radians(dd))), 3),
        "gemx_up_cam0": [round(float(v), 5) for v in up0],
        "camera_up_cam0": [round(float(v), 5) for v in r[0].T @ _Y],  # the shot's up, same axes
        "frames_used": int(sum(len(x) for x in diff)),
    }


# Feet tilt: per estimate (all fighters, or one) this many weighted planted frames and this spread of
# distances from the camera; block bootstrap over FEET_TILT_BLOCK frames, FEET_TILT_RESAMPLES times,
# for a FEET_TILT_CONFIDENCE two-sided interval.
FEET_TILT_MIN_FRAMES = 20
FEET_TILT_MIN_RANGE_M = 1.0
FEET_TILT_BLOCK = 10
FEET_TILT_RESAMPLES = 400
FEET_TILT_CONFIDENCE = 0.95
FEET_KEYPOINTS: tuple[str, ...] = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")


def _weighted_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted least-squares slope through per-group-centred data (no intercept)."""
    return float(np.sum(w * x * y) / max(float(np.sum(w * x * x)), 1e-12))


def _tilt_samples(fks, cam, actors) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Per fighter (x, y, w, frame): planted frames' distance and lowest sole height, each centred on
    the fighter's own weighted mean, weighted by the foot keypoints' confidence."""
    _, c = camera_to_world_cv(cam)
    sk = soma77()
    feet = [sk.index(n) for n in FEET_KEYPOINTS]
    out = {}
    for role, fk in fks.items():
        a = actors[role]
        w = np.clip(np.asarray(fk.kp2d[:, feet, 2], dtype=np.float64).max(axis=1), 0.0, 1.0)
        planted = (np.asarray(a.contact, dtype=np.float64)[:, : len(FLOOR_JOINTS)].max(axis=1) > 0.5) & (
            w > 0
        )
        sole = sole_heights(a, sk).min(axis=1)
        dist = np.linalg.norm((np.asarray(a.root, dtype=np.float64) - c)[:, [0, 2]], axis=1)
        ok = planted & np.isfinite(sole) & np.isfinite(dist)
        if ok.sum() < 3:
            continue
        x, y, ww = dist[ok], sole[ok], w[ok]
        x = x - np.sum(ww * x) / np.sum(ww)
        y = y - np.sum(ww * y) / np.sum(ww)
        out[role] = (x, y, ww, np.flatnonzero(ok))
    return out


def _tilt_deg(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float | None:
    """Robust weighted slope as an angle (refitted without residuals beyond 3 robust sigmas), or
    None below ``FEET_TILT_MIN_FRAMES`` (summed weight) or ``FEET_TILT_MIN_RANGE_M`` of spread."""
    if (
        len(x) == 0
        or w.sum() < FEET_TILT_MIN_FRAMES
        or np.percentile(x, 95) - np.percentile(x, 5) < FEET_TILT_MIN_RANGE_M
    ):
        return None
    return _robust_tilt(x, y, w)


def _robust_tilt(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted slope (degrees), refitted twice without residuals beyond 3 robust sigmas."""
    keep = np.ones(len(x), bool)
    for _ in range(2):
        k = _weighted_slope(x[keep], y[keep], w[keep])
        res = y - k * x
        mad = 1.4826 * float(np.median(np.abs(res[keep] - np.median(res[keep]))))
        keep = np.abs(res) <= 3.0 * max(mad, 1e-3)
    return float(np.degrees(np.arctan(_weighted_slope(x[keep], y[keep], w[keep]))))


def feet_tilt(fks: dict[str, SomaFK], cam: CameraTrack, actors: dict[str, ActorMotion]) -> dict[str, Any]:
    """The camera pitch error the planted feet show: ``atan`` of the slope of the planted soles'
    height on their horizontal distance from the camera (one slope for all fighters, each with its own
    floor offset; weighted least squares, refitted without residuals beyond 3 robust sigmas).

    A pitch error d tilts the solved floor by d about the camera (far soles too high when the camera
    should look further down), so ``feet_tilt_deg`` has the sign of ``pitch_difference_deg``. Planted
    frames: any foot contact channel > 0.5, weighted by the foot keypoints' 2D confidence. Also
    ``feet_tilt_ci_deg`` (a ``FEET_TILT_CONFIDENCE`` block-bootstrap interval: depth noise is
    correlated over frames; time blocks shared by the fighters, robust estimator resampled) and
    ``feet_tilt_per_fighter_deg``. It is NOT a pitch estimator: depth
    errors that grow with distance tilt it too (UFC cross, no re-level: +4.8 deg at 3-5 m). It is
    only used to refute a re-level it significantly contradicts (``afv.models.camera.
    feet_refute_relevel``). None (with ``feet_tilt_note``) below ``FEET_TILT_MIN_FRAMES`` of planted
    weight or ``FEET_TILT_MIN_RANGE_M`` of spread.
    """
    samples = _tilt_samples(fks, cam, actors)
    out: dict[str, Any] = {
        "feet_tilt_deg": None,
        "feet_tilt_ci_deg": None,
        "feet_tilt_per_fighter_deg": {},
        "feet_tilt_frames": int(sum(len(s[0]) for s in samples.values())),
        "feet_distance_spread_m": 0.0,
    }
    if not samples:
        return out | {"feet_tilt_note": "no planted frames with seen feet"}
    x = np.concatenate([s[0] for s in samples.values()])
    y = np.concatenate([s[1] for s in samples.values()])
    w = np.concatenate([s[2] for s in samples.values()])
    out["feet_distance_spread_m"] = round(float(np.percentile(x, 95) - np.percentile(x, 5)), 2)
    for role, (xr, yr, wr, _) in samples.items():
        t = _tilt_deg(xr, yr, wr)
        out["feet_tilt_per_fighter_deg"][role] = None if t is None else round(t, 2)
    tilt = _tilt_deg(x, y, w)
    if tilt is None:
        return out | {"feet_tilt_note": "too little planted weight or spread of distance"}
    # block bootstrap of the robust estimator: FEET_TILT_BLOCK-frame time blocks, each carrying every
    # fighter's samples of those frames (the fighters share the camera's errors at a time)
    blocks = np.concatenate([fr // FEET_TILT_BLOCK for *_, fr in samples.values()])
    ids = np.unique(blocks)
    members = [np.flatnonzero(blocks == b) for b in ids]
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(FEET_TILT_RESAMPLES):
        sel = np.concatenate([members[i] for i in rng.integers(0, len(ids), len(ids))])
        vals.append(_robust_tilt(x[sel], y[sel], w[sel]))
    lo, hi = np.percentile(vals, [50 * (1 - FEET_TILT_CONFIDENCE), 50 * (1 + FEET_TILT_CONFIDENCE)])
    return out | {
        "feet_tilt_deg": round(tilt, 2),
        "feet_tilt_ci_deg": [round(float(lo), 2), round(float(hi), 2)],
    }


def stature_m(actor: ActorMotion, sk: Skeleton) -> float:
    """The fighter's T-pose height of ``HeadEnd`` above the sole (from its own bone offsets; the rest
    root offset puts the sole at y = 0)."""
    rest = rest_positions(sk.with_offsets(np.asarray(actor.rest_offsets, dtype=np.float64)))
    return float(rest[sk.index("HeadEnd"), 1])


def scale_actor(actor: ActorMotion, centres: np.ndarray, s: float) -> None:
    """Scale a fighter about the camera centre (per frame) by ``s``: root, root_prior and bone lengths.
    Every joint ``p`` becomes ``c + s (p - c)``, so its projection is unchanged (a camera cannot tell
    a bigger fighter further away from a smaller one closer by)."""
    c = np.asarray(centres, dtype=np.float64)
    actor.root = (c + s * (np.asarray(actor.root, dtype=np.float64) - c)).astype(np.float32)
    if actor.root_prior is not None:
        prior = np.asarray(actor.root_prior, dtype=np.float64)
        actor.root_prior = (c + s * (prior - c)).astype(np.float32)
    actor.rest_offsets = (np.asarray(actor.rest_offsets, dtype=np.float64) * s).astype(np.float32)


def _sole_residual(sole: np.ndarray, w: np.ndarray, cy: np.ndarray, s: float, floor_y: float) -> float:
    """Contact-weighted median sole height after scaling by ``s`` about each frame's camera centre,
    minus the floor (non-increasing in ``s`` while the camera is above the soles)."""
    return _weighted_median((cy + s * (sole - cy)).ravel(), w.ravel()) - floor_y


def _solve_scale(sole, w, cy, floor_y, lo, hi, iterations=50) -> float:
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if _sole_residual(sole, w, cy, mid, floor_y) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def scale_to_floor(
    actors: dict[str, ActorMotion],
    floor: dict[str, Any],
    cam: CameraTrack,
    sk: Skeleton,
    max_change: float,
    observed: dict[str, np.ndarray] | None = None,
    *,
    min_contact: float = FLOOR_SCALE_MIN_CONTACT,
) -> dict[str, dict[str, Any]]:
    """Make each fighter's soles meet the shared floor by scaling it about the camera centre.

    GEM-X's body size and depth trade off (monocular scale ambiguity): a fighter whose planted soles
    sit above or below the shared floor (``fit_floor``, contact-weighted over both fighters) is scaled
    by the ``s`` that puts the contact-weighted median of its scaled sole heights on the floor. Each
    frame is scaled about that frame's camera centre (so every joint keeps its image exactly) and
    ``s`` is solved with those same per-frame centres (bisection; the median moves monotonically with
    ``s`` while the camera is above the soles), so it stays right when the camera height changes over
    the shot. ``s`` is capped to ``1 +- max_change``; bone lengths (stature) change by ``s``.
    Measured on the Media Samples: 1-4 % (stature 1.60-1.78 m after); an 8-frame shot needed 15 %
    (capped). Left alone (with the reason in the returned qc): fighters with contact weight below
    ``min_contact`` (``FLOOR_SCALE_MIN_CONTACT``), a floor not fitted from contacts, a camera not above
    the soles. Modifies ``actors`` in place.
    """
    _, c = camera_to_world_cv(cam)
    floor_y = floor.get("floor_y_in_camera_world_m")
    out: dict[str, dict[str, Any]] = {}
    for role, actor in actors.items():
        before = stature_m(actor, sk)
        obs = np.ones(len(actor.root), bool) if observed is None else np.asarray(observed[role], dtype=bool)
        w = np.asarray(actor.contact, dtype=np.float64)[:, : len(FLOOR_JOINTS)] * obs[:, None]
        weight = float(w.sum())
        info: dict[str, Any] = {
            "scale": 1.0,
            "contact_weight": round(weight, 1),
            "stature_m": round(before, 3),
        }
        if floor_y is None or floor.get("method") != "contact_weighted_median":
            out[role] = info | {"reason": "floor not fitted from contacts"}
            continue
        if weight < min_contact:
            out[role] = info | {"reason": f"too little contact ({weight:.1f} < {min_contact:g})"}
            continue
        sole = sole_heights(actor, sk)
        cy = np.broadcast_to(c[:, 1:2], sole.shape)
        above = _weighted_median((cy - sole).ravel(), w.ravel())
        if above <= 0.05:
            out[role] = info | {"reason": "camera not above the soles"}
            continue
        gap = _sole_residual(sole, w, cy, 1.0, float(floor_y))
        if abs(gap) < 1e-3:
            out[role] = info | {"reason": "already on the floor", "gap_m": round(gap, 4)}
            continue
        want = _solve_scale(sole, w, cy, float(floor_y), 0.5, 2.0)
        s_ = float(np.clip(want, 1.0 - max_change, 1.0 + max_change))
        scale_actor(actor, c, s_)
        out[role] = {
            "scale": round(s_, 4),
            "wanted": round(want, 4),
            "capped": bool(abs(want - s_) > 1e-6),
            "gap_m": round(gap, 4),
            "contact_weight": round(weight, 1),
            "stature_before_m": round(before, 3),
            "stature_after_m": round(stature_m(actor, sk), 3),
        }
    return out


def unobserved_spans(observed: np.ndarray) -> list[list[int]]:
    """Inclusive [first, last] frame spans where the fighter's box was interpolated or held."""
    miss = ~np.asarray(observed, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[0, miss.astype(np.int8), 0]))
    return [[int(s0), int(s1) - 1] for s0, s1 in zip(edges[::2], edges[1::2], strict=True)]


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cw, 0.5 * cw[-1])])


def sole_heights(actor: ActorMotion, sk: Skeleton) -> np.ndarray:
    """Floor height implied by each ``FLOOR_JOINTS`` joint per frame [T,4]: the joint's height minus
    its height above the sole in this fighter's T-pose (true while that part of the foot is planted)."""
    skr = sk.with_offsets(np.asarray(actor.rest_offsets, dtype=np.float64))
    idx = [sk.index(n) for n in FLOOR_JOINTS]
    pos, _ = forward_kinematics(skr, actor.rot, actor.root)
    return pos[:, idx, 1] - rest_positions(skr)[idx, 1]


def fit_floor(
    actors: dict[str, ActorMotion],
    observed: dict[str, np.ndarray],
    sk: Skeleton,
    *,
    min_weight: float = 3.0,
    fallback_percentile: float = 10.0,
) -> tuple[float, dict[str, Any]]:
    """Floor height (camera-stage world) from the solved feet of all fighters together.

    Contact-weighted median of the implied sole heights (weight = GEM-X ankle/foot contact
    probability x box observed); with too little contact (total weight < ``min_weight``) the
    ``fallback_percentile`` of the per-frame lowest sole on observed frames. The median is robust
    to the telephoto depth noise that moves the feet up and down along the view.
    """
    xs, ws, lows, per = [], [], [], {}
    for role, a in actors.items():
        sole = sole_heights(a, sk)
        obs = np.asarray(observed[role], dtype=bool)
        w = np.asarray(a.contact, dtype=np.float64)[:, : len(FLOOR_JOINTS)] * obs[:, None]
        xs.append(sole.ravel())
        ws.append(w.ravel())
        lows.append(sole.min(axis=1)[obs] if obs.any() else sole.min(axis=1))
        per[role] = round(_weighted_median(sole.ravel(), w.ravel()), 4) if w.sum() >= min_weight else None
    x, w = np.concatenate(xs), np.concatenate(ws)
    if w.sum() >= min_weight:
        floor_y, method = _weighted_median(x, w), "contact_weighted_median"
    else:
        floor_y = float(np.percentile(np.concatenate(lows), fallback_percentile))
        method = "lowest_sole_percentile"
    vals = [v for v in per.values() if v is not None]
    return floor_y, {
        "refit": True,
        "method": method,
        "floor_y_in_camera_world_m": round(floor_y, 4),
        "per_fighter_floor_m": per,
        "fighter_floor_spread_m": round(max(vals) - min(vals), 4) if len(vals) > 1 else None,
        "contact_weight": round(float(w.sum()), 1),
    }


# ---------------------------------------------------------------------------------------------
# QC and helpers
# ---------------------------------------------------------------------------------------------


def gemx_up_in_world(fk: SomaFK, cam: CameraTrack) -> np.ndarray:
    """GEM-X's own gravity direction per frame, expressed in the shot world [T,3]."""
    up_cv = np.einsum("tab,tcb,c->ta", fk.incam_rot[:, 0], fk.global_rot[:, 0], _Y)
    r, _ = camera_to_world_cv(cam)
    return np.einsum("tab,tb->ta", r, up_cv)


def fighter_qc(fk: SomaFK, cam: CameraTrack, actor: ActorMotion, sk: Skeleton) -> dict[str, Any]:
    """Reprojection vs GEM-X's 2D keypoints, GEM-X vs shot gravity, feet height. ``cam`` must be the
    camera in the motion's world (``shifted_camera(cam, motion.world_shift)``)."""
    pos, _ = forward_kinematics(sk.with_offsets(actor.rest_offsets), actor.rot, actor.root)
    uv, depth = project(cam, pos)
    good = fk.kp2d[..., 2] > 0.5
    err = np.linalg.norm(uv - fk.kp2d[..., :2], axis=-1)
    up = gemx_up_in_world(fk, cam)
    ang = np.degrees(np.arccos(np.clip(up @ _Y, -1.0, 1.0)))
    feet = [sk.index(n) for n in ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")]
    return {
        "reprojection_px_median": float(np.median(err[good])) if good.any() else None,
        "gravity_disagreement_deg_median": float(np.median(ang)),
        "feet_min_height_m_median": float(np.median(pos[:, feet, 1].min(axis=1))),
        "sole_min_height_m_median": float(np.median(sole_heights(actor, sk).min(axis=1))),
        "depth_m_median": float(np.median(depth[:, 0])),
        "observed_fraction": float(fk.observed.mean()),
        **_root_speeds(actor),
        **root_lateral_speeds(actor, cam),
    }


def _root_speeds(actor: ActorMotion) -> dict[str, float | None]:
    """Median hips speed (m/s, 30 Hz) of the camera-based root and of GEM-X's world root prior."""
    if actor.root.shape[0] < 2:
        return {"root_speed_m_s_median": None, "root_prior_speed_m_s_median": None}
    out: dict[str, float | None] = {
        "root_speed_m_s_median": float(
            np.median(np.linalg.norm(np.diff(actor.root, axis=0), axis=1)) * SHOT_FPS
        )
    }
    prior = actor.root_prior
    out["root_prior_speed_m_s_median"] = (
        None if prior is None else float(np.median(np.linalg.norm(np.diff(prior, axis=0), axis=1)) * SHOT_FPS)
    )
    return out


def lateral_disagrees(root_lateral: float, prior_lateral: float) -> bool:
    """True when the camera-based root moves across the ray much faster than GEM-X's rollout."""
    return (
        root_lateral > LATERAL_RATIO_FLAG * prior_lateral
        and root_lateral - prior_lateral > LATERAL_EXCESS_FLAG
    )


def lateral_speed(root: np.ndarray, centre: np.ndarray, window: int = LATERAL_WINDOW) -> np.ndarray:
    """Speed (m/s, 30 Hz) of ``root`` [T,3] across the ray from ``centre`` [T,3], over ``window`` frames."""
    root, centre = np.asarray(root, np.float64), np.asarray(centre, np.float64)
    w = min(window, len(root) - 1)
    if w < 1:
        return np.zeros(0)
    ray = root - centre
    ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-9)
    d = (root[w:] - root[:-w]) / w
    across = d - np.sum(d * ray[:-w], axis=1, keepdims=True) * ray[:-w]
    return np.linalg.norm(across, axis=1) * SHOT_FPS


def root_lateral_speeds(actor: ActorMotion, cam: CameraTrack) -> dict[str, Any]:
    """Median across-ray speeds of the camera-based root and of ``root_prior`` (``cam`` in the motion's
    world) and their ratio; a ratio far above 1 means the camera's rotations move the root (P0 telephoto
    bug: ViPE rotations), since GEM-X's rollout is independent of them."""
    if actor.root_prior is None or len(actor.root) <= 1:
        return {"root_lateral_m_s": None, "root_prior_lateral_m_s": None, "root_lateral_ratio": None}
    _, c = camera_to_world_cv(cam)
    a = float(np.median(lateral_speed(actor.root, c)))
    b = float(np.median(lateral_speed(actor.root_prior, c)))
    return {
        "root_lateral_m_s": round(a, 3),
        "root_prior_lateral_m_s": round(b, 3),
        "root_lateral_ratio": round(a / max(b, 1e-3), 2),
    }


def camera_from_gemx(fk: SomaFK, *, up: Literal["gemx", "level"] = "gemx", start: float = 0.0) -> CameraTrack:
    """A shot-world camera from a GEM-X result alone (offline P0 conversion and tests).

    Uses GEM-X's ``R_w2c`` and K; world up = GEM-X's mean gravity (``"gemx"``) or camera 0 level.
    The camera sits at (0, 0, 0); no ground placement.
    """
    r_v_cv = np.swapaxes(fk.R_w2c, -1, -2)
    up_v = None
    if up == "gemx":
        up_cv = np.einsum("tab,tcb,c->ta", fk.incam_rot[:, 0], fk.global_rot[:, 0], _Y)
        ups = np.einsum("tab,tb->ta", r_v_cv, up_cv)
        up_v = ups.mean(0) / np.linalg.norm(ups.mean(0))
    k = fk.K[0]
    video = fk.meta.get("normalized_video", {})
    w = int(video.get("width") or round(2 * k[0, 2]))
    h = int(video.get("height") or round(2 * k[1, 2]))
    return leveled_track(
        r_v_cv,
        None,
        0.5 * (fk.K[:, 0, 0] + fk.K[:, 1, 1]),
        up_v=up_v,
        width=w,
        height=h,
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        start=start,
    )


# ---------------------------------------------------------------------------------------------
# GEM-X raw-output cache: <shot>/.cache/pose3d/<key>-<id>/ (the re-level pass reuses the network run)
# ---------------------------------------------------------------------------------------------

GEMX_CACHE_VERSION = "1"
GEMX_CACHE_KEEP = 6  # newest entries kept per shot
GEMX_CACHE_ROTATION_TOL = 1e-5  # max |difference| of the network rotations' matrix entries
GEMX_CACHE_K_RTOL = 1e-6  # max |difference| of K's entries relative to the focal
GEMX_CACHE_TMP_MAX_AGE_S = 3600.0
_GEMX_SCRIPTS = ("gemx_worker.py", "soma_worker.py")
_GEMX_CONFIG = Path.home() / "afv/gem-x/weights/body_video_native77.yaml"  # the worker's default


def _gemx_config_identity(request: dict[str, Any]) -> str:
    """sha256 of the GEM-X body-video config YAML the worker loads (``config`` or the default); its
    checkpoints are covered by ``GEMX_WEIGHT_HASH``."""
    path = Path(request.get("config") or _GEMX_CONFIG)
    try:
        return file_sha256(path)
    except OSError as e:
        return f"unreadable {path}: {e}"


_REQUEST_FILES = ("video", "boxes_npz", "out_dir", "camera")  # keyed by content / tolerance instead


@dataclass
class GemxOutputCache:
    """GEM-X's raw output (the gemx worker's ``out_dir``: per fighter ``motion.pt`` and ``soma_fk.npz``,
    plus its response) per shot in ``<shot>/.cache/pose3d/``, reused when GEM-X's input is the same.

    What the re-level pass changes: the shot world turns by one rotation (new up), so ``R_w2c``
    changes by that constant rotation and nothing else GEM-X reads does (frames, boxes, K, and the
    relative rotations ``cam_angvel`` it conditions on). GEM-X is therefore given rotations relative to
    camera 0 (``gemx_network_rotations``), equal between the passes up to float noise (~1e-7, from the
    float32 camera quaternions), and its output is reused.

    Exact key (sha256 of JSON): cache version; the gemx and soma worker scripts' sha256; the gemx
    environment's fingerprint and GEM-X's weight hash; the source video's identity, the pinned ffmpeg
    and the shot interval (they decide the decoded frames); the supplied boxes (array hash); every
    other request field (fighters and slots, device, postprocess, lod, deterministic, seed). Compared
    within tolerance: the network rotations (``GEMX_CACHE_ROTATION_TOL``) and the per-frame K
    (``GEMX_CACHE_K_RTOL``). Entries carry their key material, inputs and each file's sha256 and are
    verified on every read (a bad entry is deleted); they are written to a temporary folder and
    renamed. ``Ctx.force`` skips the read and replaces the matching entry. The newest
    ``GEMX_CACHE_KEEP`` entries are kept; entries of other worker-script versions are dropped.

    Key policy for GEM-X itself: the config YAML is hashed (``_gemx_config_identity``), the weights
    by ``GEMX_WEIGHT_HASH``, the environment by ``env_fingerprint`` (interpreter, pyvenv.cfg,
    site-packages size/mtime, gem-x git HEAD). UNCOMMITTED edits to the gem-x sources (an editable
    install) are NOT seen: after patching GEM-X locally, re-run with ``afv run --from pose3d`` (force).
    ``motion.pt`` and ``soma_fk.npz`` carry the rotations GEM-X was given, relative to camera 0.
    """

    root: Path
    material: dict[str, Any] | None
    events: list[str] = field(default_factory=list)
    hit_entry: str | None = None

    @classmethod
    def for_run(cls, ctx: Ctx, request: dict[str, Any], boxes: np.ndarray) -> GemxOutputCache:
        root = Path(ctx.shot_dir) / ".cache" / "pose3d"
        try:
            b = np.ascontiguousarray(boxes, dtype=np.float32)
            material = {
                "cache": GEMX_CACHE_VERSION,
                "scripts": {n: file_sha256(WORKERS / n) for n in _GEMX_SCRIPTS},
                "env": env_fingerprint("gemx"),
                "config": _gemx_config_identity(request),
                "weights": GEMX_WEIGHT_HASH,
                "source": source_identity(ctx, request["video"]),
                "ffmpeg": ffmpeg_identity(),
                "boxes": hashlib.sha256(b.tobytes() + str(b.shape).encode()).hexdigest(),
                "camera_mode": request["camera"]["mode"],
                "frame_range": list(ctx.frame_range) if ctx.frame_range else None,
                "request": {k: v for k, v in request.items() if k not in _REQUEST_FILES},
            }
        except Exception as e:  # e.g. the source cannot be identified: run uncached
            return cls(root, None, [f"uncached ({type(e).__name__}: {e})"])
        cache = cls(root, json.loads(json.dumps(material, sort_keys=True, default=str)))
        cache.sweep()
        return cache

    @property
    def key(self) -> str:
        return hashlib.sha256(json.dumps(self.material, sort_keys=True).encode()).hexdigest()[:32]

    def sweep(self) -> None:
        import time

        for tmp in self.root.glob(".tmp-*") if self.root.exists() else ():
            try:
                if time.time() - tmp.stat().st_mtime > GEMX_CACHE_TMP_MAX_AGE_S:
                    shutil.rmtree(tmp, ignore_errors=True)
            except OSError:
                pass

    def _entries(self) -> list[Path]:
        """This key's entries, newest first."""
        if self.material is None or not self.root.exists():
            return []
        found = [(m.stat().st_mtime, m.parent) for m in self.root.glob(f"{self.key}-*/manifest.json")]
        return [p for _, p in sorted(found, reverse=True)]

    @staticmethod
    def _same_input(entry: Path, rotations: np.ndarray, k: np.ndarray) -> bool:
        with np.load(entry / "inputs.npz") as z:
            r0, k0 = z["rotations"], z["K"]
        if r0.shape != rotations.shape or k0.shape != k.shape:
            return False
        dr = float(np.abs(r0.astype(np.float64) - rotations).max()) if r0.size else 0.0
        dk = float(np.abs(k0.astype(np.float64) - k).max()) if k0.size else 0.0
        return dr <= GEMX_CACHE_ROTATION_TOL and dk <= GEMX_CACHE_K_RTOL * max(float(np.abs(k0).max()), 1.0)

    def load(self, rotations: np.ndarray, k: np.ndarray, out_dir: Path) -> dict[str, Any] | None:
        """The cached response (paths under ``out_dir``, files copied there) or None."""
        if self.material is None:
            return None
        for entry in self._entries():
            try:  # verify: anything wrong with the entry itself deletes it
                man = json.loads((entry / "manifest.json").read_text())
                if man["material"] != self.material:
                    raise ValueError("key material differs")
                same = self._same_input(entry, rotations, k)
                for rel, sha in man["files"].items():
                    if same and file_sha256(entry / "files" / rel) != sha:
                        raise ValueError(f"{rel} changed")
            except Exception:
                shutil.rmtree(entry, ignore_errors=True)
                continue
            if not same:
                continue
            try:
                for rel in man["files"]:
                    dst = Path(out_dir) / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(entry / "files" / rel, dst)
            except OSError:
                break  # a valid entry that cannot be copied out is kept; run GEM-X
            os.utime(entry / "manifest.json")  # recently used: kept by prune
            self.hit_entry = entry.name
            self.events.append("hit")
            return _swap_prefix(man["response"], "@out", str(out_dir))
        self.events.append("miss")
        return None

    def store(self, rotations: np.ndarray, k: np.ndarray, out_dir: Path, response: dict[str, Any]) -> None:
        """Best effort: a failure is reported, never raised."""
        if self.material is None:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / f".tmp-{uuid.uuid4().hex}"
            (tmp / "files").mkdir(parents=True)
            try:
                files = {}
                for f in sorted(p for p in Path(out_dir).rglob("*") if p.is_file()):
                    rel = f.relative_to(out_dir).as_posix()
                    (tmp / "files" / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(f, tmp / "files" / rel)
                    files[rel] = file_sha256(tmp / "files" / rel)
                np.savez(tmp / "inputs.npz", rotations=np.asarray(rotations), K=np.asarray(k))
                manifest = {
                    "material": self.material,
                    "files": files,
                    "response": _swap_prefix(response, str(out_dir), "@out"),
                }
                (tmp / "manifest.json").write_text(json.dumps(manifest, default=str))
                for old in self._entries():  # the entry this one replaces (forced, or changed files)
                    try:
                        if self._same_input(old, rotations, k):
                            shutil.rmtree(old, ignore_errors=True)
                    except Exception:
                        shutil.rmtree(old, ignore_errors=True)
                entry = self.root / f"{self.key}-{uuid.uuid4().hex[:8]}"
                os.replace(tmp, entry)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            self.prune(keep=entry)
            self.events.append("stored")
        except Exception as e:
            self.events.append(f"not stored ({type(e).__name__}: {e})")

    def prune(self, keep: Path, max_entries: int = GEMX_CACHE_KEEP) -> None:
        """Drop entries made by other worker scripts (they can never hit) and all but the newest
        ``max_entries`` (``keep`` always stays)."""
        assert self.material is not None
        rest: list[tuple[float, Path]] = []
        for m in self.root.glob("*/manifest.json"):
            if m.parent == keep:
                continue
            try:
                other = json.loads(m.read_text())["material"]
            except Exception:
                shutil.rmtree(m.parent, ignore_errors=True)
                continue
            if other.get("scripts") != self.material["scripts"]:
                shutil.rmtree(m.parent, ignore_errors=True)
                continue
            rest.append((m.stat().st_mtime, m.parent))
        for _, entry in sorted(rest, reverse=True)[max(0, max_entries - 1) :]:
            shutil.rmtree(entry, ignore_errors=True)

    def report(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": ", ".join(self.events) or "unused"}
        if self.material is not None:
            out["key"] = self.key
        if self.hit_entry:
            out["entry"] = self.hit_entry
        return out


def _swap_prefix(obj: Any, old: str, new: str) -> Any:
    """``obj`` with every string starting with ``old`` re-rooted at ``new``."""
    if isinstance(obj, dict):
        return {k: _swap_prefix(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_swap_prefix(v, old, new) for v in obj]
    if isinstance(obj, str) and obj.startswith(old):
        return new + obj[len(old) :]
    return obj


# ---------------------------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------------------------

_GEMX_HASHES = """\
e244d27f2834eb6382c5f6b9cbf2a38681b685514a1b2e86f8f3d2592d614959  body_s240000.ckpt
c7790f4587e518bcea40cb8281c9be2233d37f6caa668835462542760dcf82aa  motion_native77_e774_s292175.ckpt
8e3eec71dbd6bb745eba8c6debe3e4b97bcfd1d01f300099f2eedf50b9a80c8e  vitpose.pth
b7719b634e0fc1f001005605d117d77448c8d4a110923149df58529c68112ed6  motion_stats_raw.npz
5829af4a5333abdba5ad449360b868faacee99d395971e62f35c0e7169432b63  SOMA_neutral.npz
9853e19ab51625ffaa7d9e7ccf0e443272c2740837604e6bacd66604e8edf509  SOMA_procedural_transforms.json
a952cb7bdf4ccd801e355d53122d395b985b69d0d9181439edad5081bb171856  SOMA_template_rig.usda
"""
GEMX_WEIGHT_HASH = hashlib.sha256(_GEMX_HASHES.encode()).hexdigest()

_GEMX_LICENCES = (
    LicenceRecord(
        "GEM-X body + motion + vitpose77",
        "Apache-2.0 (some files OpenMDW-1.1)",
        "NVIDIA Open Model License",
        "not reviewed",
        True,
        "https://huggingface.co/nvidia/GEM-X",
        weight_sha256=GEMX_WEIGHT_HASH,
        licence_text_path="licences/2026-11-09/GEM-X_hf_model_card.md",
        retrieved="2026-11-09",
        notes="gem-x commit 4aef1452; revocable if guardrails are circumvented",
    ),
    LicenceRecord(
        "SOMA-X v0.3.3 assets",
        "Apache-2.0",
        "Apache-2.0",
        "not reviewed",
        True,
        "https://huggingface.co/nvidia/SOMA-X",
        licence_text_path="licences/2026-11-09/SOMA-X_hf_model_card.md",
        retrieved="2026-11-09",
    ),
)


class Pose3dParams(BaseModel):
    # "auto": camera, unless the camera-based roots move across the ray far faster than GEM-X's own
    # rollouts (``decide_root_mode``: a wrong camera rotation); then gemx_world for the whole shot, or
    # the rollout's velocity only over the 1 s windows where they do (most of the shot: whole shot).
    root_mode: Literal["camera", "gemx_world", "auto"] = "auto"
    # Fighters whose soles disagree with the shared floor are scaled about the camera centre (image
    # unchanged) by at most this fraction (``scale_to_floor``); 0 = off.
    floor_scale_max: float = Field(FLOOR_SCALE_MAX, ge=0.0, le=0.15)
    gemx_postprocess: bool = False  # GEM-X's own contact/IK (our contacts stage replaces it)
    device: str = "cuda"
    lod: Literal["mid", "low", "xlo"] = "mid"  # SOMA LOD for FK (joints differ <= 2.5 mm, fingers)
    video: str | None = None  # overrides the shot context (afv.core.project.shot_context)
    start: float | None = None
    duration: float | None = None
    keep_motion_pt: bool = True  # keep GEM-X's motion.pt in the work dir (gem-render overlays)
    timeout_s: float = 3600.0
    cache: bool = True  # reuse GEM-X's raw output from <shot>/.cache/pose3d (``GemxOutputCache``)


@register("gemx")
class Pose3dGemx:
    """GEM-X motion-video per fighter with supplied boxes and the shot camera (provided mode)."""

    name: ClassVar[str] = "pose3d"
    # 2: root_mode auto, zoom depth fix, lateral QC; 3: floor scale; 4: per-frame K (zoom) into GEM-X,
    # rotations relative to camera 0, windowed root_mode auto, raw-output cache; 5: weak 2D evidence
    # published as conf 0 (no depth hold), feet-limited re-level; 6: weak mask as ActorMotion.weak
    version: ClassVar[str] = "6"
    env: ClassVar[str] = "gemx"
    inputs: ClassVar[tuple[str, ...]] = ("detect_track", "camera")
    licences: ClassVar[tuple[LicenceRecord, ...]] = _GEMX_LICENCES
    Params: ClassVar[type[BaseModel]] = Pose3dParams
    weight_hash: ClassVar[str] = GEMX_WEIGHT_HASH

    def run(self, ctx: Ctx, inputs: dict[str, Result], params: BaseModel) -> Result:
        assert isinstance(params, Pose3dParams)
        tracks, cam = inputs.get("detect_track"), inputs.get("camera")
        if not isinstance(tracks, Tracks) or not isinstance(cam, CameraTrack):
            raise StageError("pose3d needs detect_track (Tracks) and camera (CameraTrack) results")
        if not tracks.fighters:
            raise StageError("pose3d: no fighters picked (edits/fighters.json)")
        if tracks.num_frames != cam.num_frames:
            raise StageError(f"pose3d: tracks have {tracks.num_frames} frames, camera {cam.num_frames}")
        work = Path(ctx.work_dir)
        src = shot_source(ctx, params.video, params.start, params.duration)

        def say(msg: str) -> None:
            ctx.progress(-1.0, msg)

        boxes = np.asarray(tracks.boxes, np.float32)
        np.savez_compressed(work / "boxes.npz", boxes=boxes)
        r_w2c, k = gemx_provided_camera(cam)
        rotations = gemx_network_rotations(r_w2c)  # GEM-X's input; camera 0 = its world
        np.save(work / "R_w2c.npy", rotations)
        np.save(work / "K.npy", k)
        fighters = []
        for role, tid in sorted(tracks.fighters.items()):
            hits = np.flatnonzero(tracks.track_ids == tid)
            if not len(hits):
                raise StageError(f"pose3d: fighter {role} track {tid} not in tracks")
            fighters.append({"role": role, "slot": int(hits[0])})
        request = {
            "command": "pose3d",
            **src,
            "boxes_npz": str(work / "boxes.npz"),
            "fighters": fighters,
            "camera": {
                "mode": "provided",
                "rotations_npy": str(work / "R_w2c.npy"),
                "intrinsics": np.median(k, axis=0).astype(float).tolist(),
                "intrinsics_npy": str(work / "K.npy"),
            },
            "device": params.device,
            "postprocess": params.gemx_postprocess,
            "lod": params.lod,
            "out_dir": str(work / "gemx"),
            "deterministic": ctx.deterministic,
            "seed": ctx.seed,
        }
        cache = GemxOutputCache.for_run(ctx, request, boxes) if params.cache else None
        resp = None
        if cache is not None and not ctx.force:
            resp = cache.load(rotations, k, work / "gemx")
            if resp is not None:
                say(f"GEM-X output reused from {cache.hit_entry} (same network input)")
        if resp is None:
            if getattr(ctx, "cache_only", False):  # never launch GEM-X on a cache-only run
                why = (
                    "the GEM-X cache is off"
                    if cache is None
                    else f"no cached GEM-X output ({cache.report()['status']})"
                )
                raise CacheOnlyMiss(f"pose3d: {why}")
            resp = run_env_worker(
                "gemx", "gemx_worker.py", request, work, progress=ctx.progress, timeout=params.timeout_s
            )
            if cache is not None:
                cache.store(rotations, k, work / "gemx", resp)
        if int(resp["frames"]) != cam.num_frames:
            raise StageError(f"pose3d: GEM-X solved {resp['frames']} frames, camera has {cam.num_frames}")
        fks = {role: SomaFK.load(f["soma_fk"]) for role, f in resp["fighters"].items()}
        if not params.keep_motion_pt:
            for f in resp["fighters"].values():
                Path(f["motion_pt"]).unlink(missing_ok=True)
        frames, timestamps = shot_timeline(cam.num_frames, src["start"], tracks)
        motion, qc = body_motion_from_fk(
            fks,
            cam,
            root_mode=params.root_mode,
            floor_scale_max=params.floor_scale_max,
            frames=frames,
            timestamps=timestamps,
        )
        qc["worker"] = {k2: resp.get(k2) for k2 in ("timings", "peak_vram_mib", "ffmpeg")}
        qc["gemx_cache"] = cache.report() if cache is not None else {"status": "off"}
        qc["gravity"]["relevelled"] = camera_relevelled(ctx, qc["gravity"])
        qc["summary"] = pose3d_summary(motion, qc)
        (work / "qc.json").write_text(json.dumps(qc, indent=2, default=str))
        if qc["summary"]["flags"]:
            say(f"pose3d QC flags: {qc['summary']['flags']}")
        for role in fks:
            q = qc["fighters"][role]
            rep = q["reprojection_px_median"]
            say(
                f"pose3d {role}: reprojection {'n/a' if rep is None else f'{rep:.1f}'} px, GEM-X gravity "
                f"off by {q['gravity_disagreement_deg_median']:.1f} deg, "
                f"feet min height {q['feet_min_height_m_median']:+.2f} m"
            )
        return motion


def camera_relevelled(ctx: Ctx, gravity: dict[str, Any]) -> bool:
    """This camera carries the runner's re-level (``<shot>/relevel.json``, written by
    ``afv.core.runner``): a record exists and the camera's pitch in camera 0 (``camera_up_cam0``) is its
    override's, within 0.2 deg (the pitch-only override keeps GeoCalib's roll)."""
    try:
        rec = json.loads((Path(ctx.shot_dir) / "relevel.json").read_text())
        up = np.asarray(rec["override"]["up_cam0"], dtype=np.float64)
        cam_up = np.asarray(gravity["camera_up_cam0"], dtype=np.float64)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    pitch = lambda u: float(np.degrees(np.arctan2(u[2], -u[1])))  # noqa: E731
    return bool(abs(pitch(up) - pitch(cam_up)) < 0.2)


def _interval_excludes_zero(ci: Any) -> bool:
    try:
        return ci is not None and len(ci) == 2 and (min(ci) > 0.0 or max(ci) < 0.0)
    except TypeError:
        return False


def pose3d_summary(motion: BodyMotion, qc: dict[str, Any]) -> dict[str, Any]:
    """Short QC for meta.json (the runner reads qc.json's "summary")."""
    flags: list[str] = []
    if qc["floor"].get("method") == "lowest_sole_percentile":
        flags.append("floor_from_lowest_feet_no_contacts")
    spread = qc["floor"].get("fighter_floor_spread_m")
    if spread is not None and spread > 0.15:
        flags.append("fighters_disagree_on_floor")
    for role, sc in (qc["floor"].get("scale") or {}).items():
        if sc.get("capped"):
            flags.append(f"{role}_floor_scale_capped")
    out: dict[str, Any] = {
        "world_shift": motion.world_shift,
        "floor_method": qc["floor"].get("method"),
        "flags": flags,
    }
    for role, q in qc["fighters"].items():
        rep_px = q["reprojection_px_median"]
        out[role] = {
            "reprojection_px": None if rep_px is None else round(rep_px, 1),
            "gravity_off_deg": round(q["gravity_disagreement_deg_median"], 1),
            "observed": round(q["observed_fraction"], 3),
            "unobserved_spans": len(q["unobserved_spans"]),
        }
        if q["gravity_disagreement_deg_median"] > 8.0:
            flags.append(f"{role}_gravity_disagrees")
        # The camera-based root's speeds: from the auto decision when there was one (after a fallback
        # the result's root IS the rollout, so its own speeds would hide the evidence).
        check = q.get("lateral_check") or {}
        a = check.get("camera_m_s", q.get("root_lateral_m_s"))
        b = check.get("prior_m_s", q.get("root_prior_lateral_m_s"))
        if a is not None and b is not None:
            out[role]["root_lateral_m_s"] = a
            out[role]["root_prior_lateral_m_s"] = b
            if lateral_disagrees(a, b):
                flags.append(f"{role}_root_disagrees_with_prior")
        out[role]["root_mode_used"] = q.get("root_mode_used", "camera")
        weak = q.get("weak_2d_spans") or []
        if weak:
            out[role]["weak_2d_frames"] = int(sum(e - s + 1 for s, e in weak))
            flags.append(f"{role}_weak_2d_evidence")
    grav = qc.get("gravity")
    if grav is not None:
        out["camera_height_m"] = grav["camera_height_m"]
        out["gravity_pitch_difference_deg"] = grav["pitch_difference_deg"]
        if grav["camera_height_m"] is not None and grav["camera_height_m"] < 0.0:
            flags.append("camera_below_floor")
        if feet_data_malformed(grav):
            flags.append("gravity_feet_unreadable")
        if grav.get("relevelled") and _interval_excludes_zero(grav.get("feet_tilt_ci_deg")):
            flags.append("gravity_relevel_feet_still_disagree")  # a warning: the feet lean off the new pitch
        if gravity_disagrees(grav):  # the runner's re-level pass: afv.models.camera.relevel_override
            flags.append("gravity_pitch_disagrees")
            if relevel_fraction(grav) < 1.0:
                flags.append("gravity_relevel_limited_by_feet")
                out["relevel_fraction"] = relevel_fraction(grav)
        elif feet_refute_relevel(grav):
            flags.append("gravity_relevel_refuted_by_feet")
        if grav.get("feet_tilt_deg") is not None:
            out["feet_tilt_deg"] = grav["feet_tilt_deg"]
            out["feet_tilt_ci_deg"] = grav.get("feet_tilt_ci_deg")
            out["feet_tilt_after_relevel_ci_deg"] = feet_tilt_after_relevel(grav)
        if grav.get("error"):
            flags.append("gravity_unchecked")
    decision = qc.get("root_mode")
    if decision is not None:
        out["root_mode"] = decision
        if decision.get("used") == "gemx_world":
            flags.append("root_fallback_gemx_world")
        elif decision.get("local_fallback_spans"):
            flags.append("root_local_fallback_gemx_world")
    return out


def load_motion_fk(
    path: str | Path, cam: CameraTrack | None = None, *, root_mode: str = "camera", role: str = "A"
) -> tuple[BodyMotion, CameraTrack, dict[str, Any]]:
    """Convenience: one ``soma_worker`` FK NPZ -> (BodyMotion, camera used, QC).

    Without ``cam`` the camera is rebuilt from GEM-X's own ``R_w2c``/K with GEM-X gravity
    (``camera_from_gemx``), e.g. for the P0 runs.
    """
    fk = SomaFK.load(path)
    cam = cam or camera_from_gemx(fk)
    motion, qc = body_motion_from_fk({role: fk}, cam, root_mode=root_mode)
    return motion, cam, qc
