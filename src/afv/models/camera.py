"""Camera conventions, the shot's world frame, and the ``camera`` stage (vipe / static / background).

Conventions
-----------
* afv ``CameraTrack``: ``cam_rot``/``cam_pos`` = camera-to-world, OpenGL camera axes (x right,
  y up, looks down -Z), world right-handed Y-up metres.
* GEM-X / ViPE / SOMA "incam": OpenCV camera axes (x right, y down, z forward). GEM-X stores
  world-to-camera rotations ``R_w2c`` in an arbitrary world (ViPE: its own SLAM world; background
  mode: camera 0); only their relative rotations (``cam_angvel``) condition the model.
* ``R_{W<-cv} = R_{W<-gl} @ CV_TO_GL`` with ``CV_TO_GL = diag(1, -1, -1)`` (its own inverse).
* Pixels: ``u = fx * x / z + cx`` in OpenCV camera coordinates, the same K GEM-X uses for its
  overlay and its 2D keypoints (normalised 30 Hz video pixels).

The shot world
--------------
Built once per shot by this stage and shared by every later stage (pose3d, pair, contacts, QC):

1. Up: gravity from GeoCalib up vectors (ViPE's bundled model) on a dozen frames, rotated into the
   camera world and averaged (``gravity="geocalib"``), or "camera 0 is level" (``"level"``).
2. Heading: world -Z is camera 0's viewing direction projected on the ground (so the scene's
   front, +Z, faces the camera).
3. Ground and origin: from the fighters' boxes (feet on the ground, a person-height prior), the
   ground is put at y = 0 and the origin under the fighters' median position. Camera position is
   constant (rotation-only camera, as GEM-X assumes) unless ``use_translation``.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import shutil
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel

from afv.core.stage import CacheOnlyMiss, Ctx, StageError, register
from afv.core.types import CameraTrack, LicenceRecord, Result, Tracks
from afv.motion.rotations import from_matrix, to_matrix

CV_TO_GL = np.diag([1.0, -1.0, -1.0])
SHOT_FPS = 30.0
STANCE_HEIGHT_M = 1.42  # a fighter's box height in stance (crouched, knees bent), for the ground prior
WORKERS = Path(__file__).resolve().parent / "workers"

# ---------------------------------------------------------------------------------------------
# Convention helpers
# ---------------------------------------------------------------------------------------------


def camera_to_world_cv(cam: CameraTrack) -> tuple[np.ndarray, np.ndarray]:
    """``(R [F,3,3], c [F,3])``: OpenCV camera axes to world, and camera centre in world."""
    r_gl = to_matrix(np.asarray(cam.cam_rot, dtype=np.float64))
    return r_gl @ CV_TO_GL, np.asarray(cam.cam_pos, dtype=np.float64)


def world_to_camera_cv(cam: CameraTrack) -> tuple[np.ndarray, np.ndarray]:
    """``(R [F,3,3], t [F,3])`` with ``x_cv = R x_world + t`` (GEM-X / OpenCV extrinsics)."""
    r, c = camera_to_world_cv(cam)
    rt = np.swapaxes(r, -1, -2)
    return rt, -np.einsum("fij,fj->fi", rt, c)


def intrinsics_matrix(cam: CameraTrack) -> np.ndarray:
    """K [F,3,3] (square pixels, constant principal point)."""
    f = np.asarray(cam.focal_px, dtype=np.float64)
    k = np.zeros((f.shape[0], 3, 3))
    k[:, 0, 0] = k[:, 1, 1] = f
    k[:, 0, 2], k[:, 1, 2], k[:, 2, 2] = cam.cx, cam.cy, 1.0
    return k


def project(cam: CameraTrack, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points [F,N,3] to pixels [F,N,2]; also returns depth along the view [F,N]."""
    r, t = world_to_camera_cv(cam)
    x = np.einsum("fij,fnj->fni", r, np.asarray(points, dtype=np.float64)) + t[:, None]
    k = intrinsics_matrix(cam)
    uvw = np.einsum("fij,fnj->fni", k, x)
    return uvw[..., :2] / uvw[..., 2:3], x[..., 2]


def shifted_camera(cam: CameraTrack, world_shift) -> CameraTrack:
    """The camera expressed in a result's world: ``cam_pos + world_shift`` (``BodyMotion.world_shift``)."""
    shift = np.asarray(world_shift, dtype=np.float64).reshape(3)
    out = copy.copy(cam)
    out.cam_pos = np.asarray(cam.cam_pos, dtype=np.float64) + shift
    n = np.asarray(cam.ground_normal, dtype=np.float64)
    out.ground_offset = float(cam.ground_offset - n @ shift)
    return out


def gemx_provided_camera(cam: CameraTrack) -> tuple[np.ndarray, np.ndarray]:
    """GEM-X ``provided`` camera: ``R_w2c`` [F,3,3] (OpenCV, this world) and K [F,3,3] per frame.

    GEM-X's networks take a per-frame K (``K_fullimg`` [T,3,3]: the pose features' crop rays, the
    motion network's CLIFF box features and its camera-frame translation decode), so a zoom
    (``CameraTrack.focal_px`` per frame) reaches GEM-X as it is; only its ``motion_intrinsics``
    config helper insists on one 3x3 K, which the gemx worker bypasses (``intrinsics_npy``).
    """
    r, _ = world_to_camera_cv(cam)
    return r.astype(np.float32), intrinsics_matrix(cam).astype(np.float32)


def gemx_network_rotations(r_w2c: np.ndarray) -> np.ndarray:
    """``R_w2c`` re-expressed with camera 0 as the world (``R_w2c[t] @ R_w2c[0]^T``, float32).

    GEM-X conditions only on the relative rotations (``cam_angvel = R[t+1] R[t]^T``) and rolls its
    world out in its own gravity-view frame, so this gives it the same input as ``r_w2c`` itself
    while making that input independent of the shot world's orientation: the gravity re-level pass
    (a constant world rotation) then hands GEM-X the same rotations, up to float noise, and its raw
    output can be reused (``afv.models.gemx.GemxOutputCache``).
    """
    r = np.asarray(r_w2c, dtype=np.float64)
    return np.einsum("tab,cb->tac", r, r[0]).astype(np.float32)


def make_camera_track(
    r_c2w_cv: np.ndarray,
    centres: np.ndarray,
    focal_px: np.ndarray,
    *,
    width: int,
    height: int,
    cx: float,
    cy: float,
    frames: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    start: float = 0.0,
    static: bool = False,
) -> CameraTrack:
    """CameraTrack from OpenCV camera-to-world rotations and camera centres in an afv world."""
    n = r_c2w_cv.shape[0]
    frames = np.arange(n) if frames is None else np.asarray(frames)
    timestamps = start + np.arange(n) / SHOT_FPS if timestamps is None else np.asarray(timestamps)
    q = from_matrix(np.asarray(r_c2w_cv, dtype=np.float64) @ CV_TO_GL)
    return CameraTrack(
        frames=frames,
        timestamps=timestamps,
        cam_rot=q,
        cam_pos=np.broadcast_to(np.asarray(centres, dtype=np.float64), (n, 3)).copy(),
        focal_px=np.broadcast_to(np.asarray(focal_px, dtype=np.float64), (n,)).copy(),
        width=int(width),
        height=int(height),
        cx=float(cx),
        cy=float(cy),
        static=static,
    )


def level_rotation(up: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """``R_{W<-V}``: world Y = ``up``, world -Z = ``forward`` projected on the ground plane."""
    y = np.asarray(up, dtype=np.float64)
    y = y / np.linalg.norm(y)
    f = np.asarray(forward, dtype=np.float64)
    f = f - (f @ y) * y
    if np.linalg.norm(f) < 1e-6:
        raise ValueError("camera looks straight up or down; heading is undefined")
    z = -f / np.linalg.norm(f)
    x = np.cross(y, z)
    return np.stack([x, y, z])


def average_up(r_v_from_cv: np.ndarray, up_cam: np.ndarray, frames: np.ndarray, weights=None) -> np.ndarray:
    """Weighted mean world-up direction in the camera world V from per-frame camera-frame ups."""
    ups = np.einsum("nij,nj->ni", r_v_from_cv[np.asarray(frames)], np.asarray(up_cam, dtype=np.float64))
    ups /= np.linalg.norm(ups, axis=1, keepdims=True)
    w = np.ones(len(ups)) if weights is None else np.asarray(weights, dtype=np.float64)
    mean = (ups * w[:, None]).sum(0)
    if np.linalg.norm(mean) < 1e-6:
        raise ValueError("gravity estimates cancel out")
    return mean / np.linalg.norm(mean)


def leveled_track(
    r_v_from_cv: np.ndarray,
    centres_v: np.ndarray | None,
    focal_px: np.ndarray,
    *,
    up_v: np.ndarray | None,
    width: int,
    height: int,
    cx: float,
    cy: float,
    frames: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    start: float = 0.0,
    static: bool = False,
) -> CameraTrack:
    """Re-express an OpenCV camera trajectory (camera-to-V) in a gravity-aligned afv world.

    ``up_v`` is world up in V (None: camera 0 is level). Camera 0 sits at the origin; call
    ``place_on_ground`` to move the origin to the ground under the fighters.
    """
    r_v_from_cv = np.asarray(r_v_from_cv, dtype=np.float64)
    if up_v is None:
        up_v = r_v_from_cv[0] @ np.array([0.0, -1.0, 0.0])
    r_wv = level_rotation(up_v, r_v_from_cv[0] @ np.array([0.0, 0.0, 1.0]))
    r_w = np.einsum("ij,fjk->fik", r_wv, r_v_from_cv)
    if centres_v is None:
        centres = np.zeros(3)
    else:
        c = np.asarray(centres_v, dtype=np.float64)
        centres = (c - c[0]) @ r_wv.T
    return make_camera_track(
        r_w,
        centres,
        focal_px,
        width=width,
        height=height,
        cx=cx,
        cy=cy,
        frames=frames,
        timestamps=timestamps,
        start=start,
        static=static,
    )


def place_on_ground(
    cam: CameraTrack,
    boxes: np.ndarray,
    *,
    person_height: float = STANCE_HEIGHT_M,
    default_camera_height: float = 1.5,
    min_usable: int = 10,
    min_usable_fraction: float = 0.2,
) -> tuple[CameraTrack, dict[str, Any]]:
    """Shift the world so the ground is y = 0 and the origin is under the people in ``boxes``.

    ``boxes`` [F,N,4] (x0, y0, x1, y1 pixels, NaN when absent). Each box gives a horizontal
    distance from its height (``person_height`` prior: a fighter's box in stance, ~1.42 m, not
    standing height; feet on the ground at the box bottom) and so a ground height under the
    camera; medians over all boxes are used. Boxes cut by the top or bottom image border (feet or
    head out of frame) are ignored; ``info["flag"] = "few_usable_boxes"`` when fewer than
    ``min_usable`` boxes (or ``min_usable_fraction`` of the present ones) are usable. This is only a
    prior: pose3d re-fits the floor from GEM-X's solved feet (``BodyMotion.world_shift``).
    """
    r, c = camera_to_world_cv(cam)
    kinv = np.linalg.inv(intrinsics_matrix(cam))
    heights, points = [], []
    margin = 3.0  # pixels
    present = 0
    for f in range(min(cam.num_frames, boxes.shape[0])):
        for b in boxes[f]:
            if not np.isfinite(b).all() or b[3] - b[1] < 8:
                continue
            present += 1
            if cam.height and (b[1] <= margin or b[3] >= cam.height - 1 - margin):
                continue
            u = 0.5 * (b[0] + b[2])
            rays = r[f] @ kinv[f] @ np.array([[u, b[3], 1.0], [u, b[1], 1.0]]).T  # [3,2] bottom, top
            horiz = np.linalg.norm(rays[[0, 2]], axis=0)
            if np.any(horiz < 1e-6):
                continue
            slope = rays[1] / horiz  # dy per metre of horizontal distance
            gap = slope[1] - slope[0]  # top minus bottom
            if gap <= 1e-4:
                continue
            dist = person_height / gap
            ground_y = c[f, 1] + dist * slope[0]
            foot = c[f] + dist * rays[:, 0] / horiz[0]
            heights.append(ground_y)
            points.append(foot[[0, 2]])
    info: dict[str, Any] = {
        "ground_samples": len(heights),
        "boxes_present": present,
        "usable_fraction": round(len(heights) / present, 3) if present else 0.0,
        "person_height_m": person_height,
    }
    if len(heights) < min_usable or (present and len(heights) / present < min_usable_fraction):
        info["flag"] = "few_usable_boxes"
    if len(heights) >= 3:
        ground_y = float(np.median(heights))
        origin_xz = np.median(np.asarray(points), axis=0)
        info["method"] = "boxes_person_height_prior"
    else:
        ground_y = float(c[0, 1] - default_camera_height)
        origin_xz = np.zeros(2)
        info["method"] = "default_camera_height"
    shift = np.array([origin_xz[0], ground_y, origin_xz[1]])
    cam.cam_pos = np.asarray(cam.cam_pos, dtype=np.float64) - shift
    cam.ground_normal = [0.0, 1.0, 0.0]
    cam.ground_offset = 0.0
    info["camera_height_m"] = round(float(cam.cam_pos[0, 1]), 3)
    info["camera_distance_m"] = round(float(np.linalg.norm(cam.cam_pos[0, [0, 2]])), 3)
    return cam, info


# ---------------------------------------------------------------------------------------------
# Camera rotation (+ zoom) from background feature flow
# ---------------------------------------------------------------------------------------------
#
# Why: on telephoto shots ViPE's rotations do not explain the background's image motion (measured on
# the Media Samples: ref walk 3.6x too fast, Who Am I mirrored, Close boxing ~0 while it pans and
# zooms 1.9x): with almost no parallax its metric-depth prior lets translation absorb the pan and the
# rotation compensates. GEM-X's world motion and our camera-based root use rotations only, so the
# root then slides across the ray while the 2D hips are smooth. A rotation-only camera (plus zoom) is
# measured directly from the background instead: per frame pair, one rotation and one relative zoom
# that map the background features of frame t onto frame t+1 (``fit_rotation_zoom``), chained over
# the shot (``flow_rotations``). The lateral world motion this gives (distance x angle) does not
# depend on the absolute focal, which stays ViPE's.
#
# Zoom: the fitted relative zooms, chained, give the focal per frame relative to ViPE's (``zoom=
# "apply"``, the default): ViPE calibrates one focal (GeoCalib on frames 0, 1 s and 2 s, then frozen),
# so the chain is anchored there (``vipe_focal_frames``). It reaches the CameraTrack and GEM-X's K;
# GEM-X then reads a growing body under a zoom-in as a zoom, not as the fighter walking towards the
# camera. Measured on the Media Samples (round 3): UFC cross 1.21x, background residual 0.20 px during
# the zoom vs 0.18 px elsewhere (a dolly, which ViPE read it as, would not fit one zoom: its
# rotations rise to 0.9 px there); Close boxing 1.53x, 0.54 vs 0.27 px (fast pan with motion blur).


def unproject(p: np.ndarray, focal: float, cx: float, cy: float) -> np.ndarray:
    """Unit OpenCV rays [N,3] through pixels ``p`` [N,2]."""
    p = np.asarray(p, dtype=np.float64)
    r = np.c_[(p[:, 0] - cx) / focal, (p[:, 1] - cy) / focal, np.ones(len(p))]
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def _project_rays(rays: np.ndarray, focal: float, cx: float, cy: float) -> np.ndarray:
    return np.c_[cx + focal * rays[:, 0] / rays[:, 2], cy + focal * rays[:, 1] / rays[:, 2]]


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R minimising sum |b - R a|^2 (rows are vectors)."""
    u, _, vt = np.linalg.svd(b.T @ a)
    d = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    return u @ np.diag([1.0, 1.0, d]) @ vt


def _fit_rz(a: np.ndarray, p1: np.ndarray, focal: float, cx: float, cy: float, zoom: bool, iterations: int):
    """Least-squares rotation (+ zoom) from unit rays ``a`` of frame t to pixels ``p1`` of frame t+1."""
    r, s = np.eye(3), 1.0
    for _ in range(iterations if zoom else 1):
        r = _kabsch(a, unproject(p1, focal * s, cx, cy))
        if not zoom:
            break
        q = a @ r.T
        q = focal * q[:, :2] / q[:, 2:3]
        s = float(np.sum((p1 - (cx, cy)) * q) / max(float(np.sum(q * q)), 1e-12))
    return r, s


def fit_rotation_zoom(
    p0: np.ndarray,
    p1: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
    *,
    zoom: bool = True,
    iterations: int = 6,
    threshold_px: float = 1.0,
    ransac_iterations: int = 100,
    seed: int = 0,
) -> tuple[np.ndarray, float, float]:
    """Rotation ``R`` (OpenCV axes, ``x_{t+1} = R x_t``) and relative zoom ``s = f_{t+1} / f_t`` that
    map pixels ``p0`` of frame t (focal ``focal``) onto ``p1`` of frame t+1, and the median inlier
    residual (px).

    RANSAC on this model (2-point samples, ``threshold_px``), each hypothesis scored by how much of
    the frame its inliers cover (cells of an 8 x 8 grid over the principal-point-centred image, then
    inlier count): the background is the motion spread over the frame, not the most textured one (a
    moving crowd can carry more corners than a still wall). A homography (8 DOF) is not used: it can
    bend to fit a still and a moving region within a pixel. The model is refitted on the inliers
    (alternating Kabsch on unit rays with a closed-form zoom), twice.
    """
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    n = len(p0)
    a_all = unproject(p0, focal, cx, cy)

    def errors(r: np.ndarray, s: float) -> np.ndarray:
        return np.linalg.norm(_project_rays(a_all @ r.T, focal * s, cx, cy) - p1, axis=1)

    keep = np.ones(n, bool)
    if n >= 6:
        rng = np.random.default_rng(seed)
        cell = (
            np.clip(p0[:, 0] * 4 // max(cx, 1.0), 0, 7) * 8 + np.clip(p0[:, 1] * 4 // max(cy, 1.0), 0, 7)
        ).astype(int)
        best = (-1, -1)
        for _ in range(ransac_iterations):
            i, j = rng.choice(n, 2, replace=False)
            r, s = _fit_rz(a_all[[i, j]], p1[[i, j]], focal, cx, cy, zoom, 3)
            inl = errors(r, s) < threshold_px
            score = (len(np.unique(cell[inl])), int(inl.sum()))
            if score > best:
                best, keep = score, inl
        if best[1] < 3:
            keep = np.ones(n, bool)
    r, s = _fit_rz(a_all[keep], p1[keep], focal, cx, cy, zoom, iterations)
    for _ in range(2):
        new = errors(r, s) < threshold_px
        if new.sum() < 3 or np.array_equal(new, keep):
            break
        keep = new
        r, s = _fit_rz(a_all[keep], p1[keep], focal, cx, cy, zoom, iterations)
    return r, s, float(np.median(errors(r, s)[keep]))


@dataclass
class FlowCamera:
    """Rotation-only camera measured from background flow; pair t is frames (t, t+1)."""

    r_v_cv: np.ndarray  # [T,3,3] OpenCV camera-to-V rotations, V = camera 0
    focal_px: np.ndarray  # [T] per-frame focal (constant unless the zoom was applied)
    zoom: np.ndarray  # [T] measured focal relative to the reference (reported even if not applied)
    residual_px: np.ndarray  # [T-1] median background residual per pair (NaN: too few points)
    valid: np.ndarray  # [T-1] bool, pairs with enough background points
    zoom_applied: bool


def vipe_focal_frames(n: int, gap_s: float = 1.0) -> list[int]:
    """Frames ViPE calibrates its one focal on (GeoCalib, shared intrinsics; ViPE's
    ``IntrinsicEstimationProcessor``: 0, ``gap``, ``2 gap`` with ``gap = min(1 s, (n - 1) // 2)``)."""
    gap = max(0, min(int(gap_s * SHOT_FPS), (n - 1) // 2))
    return [0, gap, 2 * gap]


def _normalise_zoom(z: np.ndarray, reference: list[int] | None) -> np.ndarray:
    """Chained zoom relative to the geometric mean at ``reference`` frames (None: the median)."""
    if reference is None:
        return z / np.median(z)
    ref = np.clip(np.asarray(reference, dtype=int), 0, len(z) - 1)
    return z / float(np.exp(np.mean(np.log(z[ref]))))


def _fill_invalid(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-column linear interpolation over invalid rows (constant velocity through gaps); 0 if none valid."""
    x = np.array(x, dtype=np.float64)
    idx = np.flatnonzero(valid)
    if not len(idx):
        return np.zeros_like(x)
    t = np.arange(len(x))
    for c in range(x.shape[1]):
        x[:, c] = np.interp(t, idx, x[idx, c])
    return x


def flow_rotations(
    p0s: np.ndarray,
    p1s: np.ndarray,
    counts: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
    *,
    zoom: bool = True,
    apply_zoom: bool = False,
    min_zoom_range: float = 0.10,
    min_points: int = 12,
    zoom_reference: list[int] | None = None,
) -> FlowCamera:
    """Chain per-pair rotations and zooms (``fit_rotation_zoom``) of background tracks ``p0s``/``p1s``
    [T-1,M,2] (``counts`` [T-1] valid points per pair, the vipe worker's ``flow`` output).

    ``focal`` is ViPE's focal, valid at the frames ``zoom_reference`` (``vipe_focal_frames``; None:
    the shot median). With ``zoom`` each pair's relative zoom is fitted with its rotation (so a zoom
    is not misread as rotation) and reported as ``zoom`` (chained, relative to the reference frames'
    geometric mean). Only with ``apply_zoom`` and a zoom range ``max(Z) / min(Z) - 1`` above
    ``min_zoom_range`` (smaller ranges come from parallax of a moving camera and drift) is the
    per-frame focal ``focal * Z(t)`` used and the rotations refitted with it (``zoom_applied``).
    Pairs with fewer than ``min_points`` points get the interpolated angular velocity (and zoom
    rate) of their neighbours.
    """
    counts = np.asarray(counts)
    n_pairs = len(counts)
    valid = counts >= min_points

    def solve(focals: np.ndarray, with_zoom: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rv, logs, res = np.zeros((n_pairs, 3)), np.zeros((n_pairs, 1)), np.full(n_pairs, np.nan)
        for t in np.flatnonzero(valid):
            m = int(counts[t])
            r, s, e = fit_rotation_zoom(p0s[t, :m], p1s[t, :m], float(focals[t]), cx, cy, zoom=with_zoom)
            rv[t], logs[t], res[t] = _rotvec(r), np.log(s), e
        return _fill_invalid(rv, valid), _fill_invalid(logs, valid)[:, 0], res

    def cumulative(logs: np.ndarray) -> np.ndarray:
        return _normalise_zoom(np.exp(np.r_[0.0, np.cumsum(logs)]), zoom_reference)

    const = np.full(n_pairs + 1, float(focal))
    rv, logs, res = solve(const, zoom)
    z = cumulative(logs)
    applied = bool(zoom and apply_zoom and z.max() / z.min() - 1.0 > min_zoom_range)
    focals = const
    if applied:
        rv, logs, res = solve(focal * z, True)
        z = cumulative(logs)
        focals = focal * z
    rel = _rotmat(rv)  # x_{t+1} = rel[t] x_t
    r_v_cv = np.empty((n_pairs + 1, 3, 3))
    r_v_cv[0] = np.eye(3)
    for t in range(n_pairs):
        r_v_cv[t + 1] = r_v_cv[t] @ rel[t].T
    return FlowCamera(r_v_cv, focals, z, res, valid, applied)


def rotation_residual_px(
    p0s: np.ndarray,
    p1s: np.ndarray,
    counts: np.ndarray,
    r_v_cv: np.ndarray,
    focal_px: np.ndarray | float,
    cx: float,
    cy: float,
    *,
    fit_zoom: bool = False,
) -> np.ndarray:
    """Median pixel residual per pair [T-1] of a rotation-only camera (OpenCV camera-to-V rotations,
    per-frame or one focal) on ALL background tracks: how well it explains the background's image
    motion (NaN where a pair has no points). ``fit_zoom`` gives every pair its best relative zoom
    for the camera's rotation (closed form), so cameras with and without a zoom model compare fairly.
    """
    out = np.full(len(counts), np.nan)
    f = np.broadcast_to(np.asarray(focal_px, np.float64), (len(r_v_cv),))
    for t in np.flatnonzero(np.asarray(counts) > 0):
        m = int(counts[t])
        rel = r_v_cv[t + 1].T @ r_v_cv[t]
        q = unproject(p0s[t, :m], f[t], cx, cy) @ rel.T
        q = q[:, :2] / q[:, 2:3]
        d = np.asarray(p1s[t, :m], np.float64) - (cx, cy)
        f1 = f[t + 1]
        if fit_zoom:
            f1 = float(np.sum(d * q) / max(float(np.sum(q * q)), 1e-12))
        out[t] = float(np.median(np.linalg.norm(f1 * q - d, axis=1)))
    return out


def relative_drift_deg(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """Angle (degrees) [T] between two cameras' rotations relative to their own frame 0: how far one
    has drifted from the other over the shot (each camera may live in its own world)."""
    rel_a = np.einsum("ba,tbc->tac", r_a[0], r_a)  # camera t -> camera 0
    rel_b = np.einsum("ba,tbc->tac", r_b[0], r_b)
    d = np.einsum("tba,tbc->tac", rel_a, rel_b)
    return np.degrees(np.arccos(np.clip((np.trace(d, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))


def drift_rate_deg_s(r_a: np.ndarray, r_b: np.ndarray, window: int = int(SHOT_FPS)) -> float:
    """Largest drift between two cameras over any ``window`` frames (degrees per second)."""
    n = len(r_a)
    w = min(window, n - 1)
    if w < 1:
        return 0.0
    rel_a = np.einsum("tba,tbc->tac", r_a[:-w], r_a[w:])  # camera t+w -> camera t
    rel_b = np.einsum("tba,tbc->tac", r_b[:-w], r_b[w:])
    d = np.einsum("tba,tbc->tac", rel_a, rel_b)
    ang = np.degrees(np.arccos(np.clip((np.trace(d, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
    return float(ang.max()) * SHOT_FPS / w


def fighter_distance_m(boxes: np.ndarray, focal: float, height: int, *, default: float = 10.0) -> float:
    """Median camera-to-fighter distance from box heights (stance-height prior), boxes [F,K,4];
    boxes cut by the top/bottom border are skipped."""
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    h = b[:, 3] - b[:, 1]
    ok = np.isfinite(b).all(axis=1) & (h > 8) & (b[:, 1] > 3) & (b[:, 3] < height - 4)
    return float(np.median(focal * STANCE_HEIGHT_M / h[ok])) if ok.sum() >= 3 else default


def vipe_disagreement(
    vipe_px: float | None,
    flow_px: float | None,
    drift_m: float,
    drift_m_s: float,
    *,
    ratio: float = 3.0,
    floor_px: float = 0.75,
    max_drift_m: float = 0.3,
    max_drift_m_s: float = 0.1,
) -> list[str]:
    """Why ViPE's rotations disagree with the background (empty: they agree).

    ``residual``: its median per-pair residual is above ``max(ratio x flow's, floor_px)`` (relative,
    so a steady rate bias of a pixel per frame is caught on telephoto shots where flow fits to 0.2 px);
    ``drift``: relative to the flow camera it drifts more than ``max_drift_m`` over the shot or
    ``max_drift_m_s`` within any second, as metres at the fighters' distance (what the root slides).
    """
    out = []
    if vipe_px is not None and flow_px is not None and vipe_px > max(ratio * flow_px, floor_px):
        out.append("residual")
    if drift_m > max_drift_m or drift_m_s > max_drift_m_s:
        out.append("drift")
    return out


def relative_rotations(r_v_cv: np.ndarray) -> np.ndarray:
    """Per-pair camera increments [T-1,3,3] of camera-to-V rotations: ``x_{t+1} = rel[t] x_t``."""
    r = np.asarray(r_v_cv, dtype=np.float64)
    return np.einsum("tba,tbc->tac", r[1:], r[:-1])


def splice_rotations(r_a: np.ndarray, r_b: np.ndarray, use_b: np.ndarray) -> np.ndarray:
    """Camera-to-V rotations [T,3,3] chained from ``r_a[0]`` with ``r_b``'s increment on the pairs
    where ``use_b`` [T-1] and ``r_a``'s elsewhere (each camera may live in its own world V)."""
    rel_a, rel_b = relative_rotations(r_a), relative_rotations(r_b)
    out = np.empty((len(rel_a) + 1, 3, 3))
    out[0] = np.asarray(r_a, dtype=np.float64)[0]
    for t, use in enumerate(np.asarray(use_b, dtype=bool)):
        out[t + 1] = out[t] @ (rel_b[t] if use else rel_a[t]).T
    return out


def rolling_nanmedian(x: np.ndarray, window: int) -> np.ndarray:
    """Median of the finite values in a centred ``window`` around each index (NaN where none)."""
    x = np.asarray(x, dtype=np.float64)
    half = max(1, int(window)) // 2
    out = np.full(len(x), np.nan)
    for t in range(len(x)):
        seg = x[max(0, t - half) : t + half + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[t] = float(np.median(seg))
    return out


def mask_spans(mask: np.ndarray) -> list[list[int]]:
    """Inclusive [first, last] index spans where ``mask`` is True."""
    m = np.asarray(mask, dtype=np.int8)
    edges = np.flatnonzero(np.diff(np.r_[0, m, 0]))
    return [[int(a), int(b) - 1] for a, b in zip(edges[::2], edges[1::2], strict=True)]


def clean_mask(mask: np.ndarray, *, merge_gap: int, min_len: int) -> np.ndarray:
    """``mask`` with gaps of at most ``merge_gap`` between True spans filled, then spans shorter than
    ``min_len`` dropped (a local decision must not flicker)."""
    m = np.asarray(mask, dtype=bool).copy()
    spans = mask_spans(m)
    for (_, e0), (s1, _) in itertools.pairwise(spans):
        if s1 - e0 - 1 <= merge_gap:
            m[e0 + 1 : s1] = True
    for s, e in mask_spans(m):
        if e - s + 1 < min_len:
            m[s : e + 1] = False
    return m


def disagreeing_pairs(
    res_vipe: np.ndarray,
    res_flow: np.ndarray,
    r_vipe: np.ndarray,
    r_flow: np.ndarray,
    distance_m: float,
    *,
    window: int = int(SHOT_FPS),
    ratio: float = 3.0,
    floor_px: float = 0.75,
    max_flow_px: float = 1.5,
    max_drift_m_s: float = 0.1,
    min_len: int = 10,
) -> np.ndarray:
    """Frame pairs [T-1] where ViPE's rotations do not explain the background but flow's do: the
    windowed form of ``vipe_disagreement``.

    Per pair, over a centred ``window``: ViPE's median residual above ``max(ratio x flow's,
    floor_px)`` while flow's stays within ``max_flow_px``; or the two cameras drifting apart faster than
    ``max_drift_m_s`` at the fighters' distance (the window's mean per-pair rotation difference, a
    vector, so jitter that cancels is not drift). Gaps up to ``window / 2`` are closed and spans
    shorter than ``min_len`` pairs dropped.
    """
    n = len(res_vipe)
    if n == 0:
        return np.zeros(0, bool)
    mv, mf = rolling_nanmedian(res_vipe, window), rolling_nanmedian(res_flow, window)
    with np.errstate(invalid="ignore"):
        bad = (mv > np.maximum(ratio * mf, floor_px)) & (mf <= max_flow_px)
    # per-pair difference of the two cameras' increments (camera frame t), its centred window mean
    diff = np.einsum("tab,tcb->tac", relative_rotations(r_flow), relative_rotations(r_vipe))
    dv = np.array([_rotvec(d) for d in diff])
    half = max(1, int(window)) // 2
    csum = np.r_[np.zeros((1, 3)), np.cumsum(dv, axis=0)]
    lo, hi = np.clip(np.arange(n) - half, 0, n), np.clip(np.arange(n) + half + 1, 0, n)
    mean = (csum[hi] - csum[lo]) / (hi - lo)[:, None]
    bad |= np.linalg.norm(mean, axis=1) * distance_m * SHOT_FPS > max_drift_m_s
    return clean_mask(bad, merge_gap=half, min_len=min_len)


def _rotvec(r: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(r).as_rotvec()


def _rotmat(rv: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(rv).as_matrix()


# ---------------------------------------------------------------------------------------------
# Worker bridge (module A's subprocess bridge; this wrapper keeps one call shape for C's stages)
# ---------------------------------------------------------------------------------------------

Progress = Callable[[float, str], None]


def run_env_worker(
    env: str,
    script: str,
    request: dict[str, Any],
    work_dir: Path,
    *,
    gpu: bool | None = None,
    timeout: float | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run ``workers/<script>`` in model environment ``env``; return its JSON response.

    Thin wrapper over ``afv.models.envs.run_worker`` (GPU lock, streaming log, timeout, readable
    errors). Requests are named ``<script stem>_<command>`` in ``work_dir/worker``.
    """
    from afv.models.envs import run_worker

    name = f"{Path(script).stem}_{request.get('command', 'run')}"
    return run_worker(env, script, request, work_dir, timeout=timeout, gpu=gpu, name=name, progress=progress)


# ---------------------------------------------------------------------------------------------
# Gravity re-level pass (runner: camera -> pose3d -> if needed camera(up_cam0) -> pose3d, once)
# ---------------------------------------------------------------------------------------------

GRAVITY_PITCH_FLAG_DEG = 3.0
GRAVITY_HEIGHT_FLAG_M = 0.5


def gravity_disagrees(gravity: dict[str, Any] | None) -> bool:
    """pose3d's qc ``gravity`` section says the camera stage's pitch is off: GEM-X's people-based
    gravity differs by more than ``GRAVITY_PITCH_FLAG_DEG`` in pitch AND that moves the camera by more
    than ``GRAVITY_HEIGHT_FLAG_M`` at the fighters' distance (flag ``gravity_pitch_disagrees``), and the
    planted feet do not refute it (``relevel_fraction`` > 0)."""
    return _gemx_pitch_disagrees(gravity) and relevel_fraction(gravity) > 0.0


def _gemx_pitch_disagrees(gravity: dict[str, Any] | None) -> bool:
    if not gravity:
        return False
    try:
        return (
            abs(float(gravity["pitch_difference_deg"])) > GRAVITY_PITCH_FLAG_DEG
            and abs(float(gravity["height_change_if_gemx_m"])) > GRAVITY_HEIGHT_FLAG_M
        )
    except (KeyError, TypeError, ValueError):
        return False


# The feet tilt's interval is widened by this allowance for its own bias (half the re-level trigger;
# the UFC cross control, no re-level, reads +4.8 deg at 3-5 m, where the feet's distance spread is
# small), and a correction the feet limit to under FEET_MIN_RELEVEL_DEG is no re-level.
FEET_TILT_BIAS_DEG = 1.5
# The re-level decision rule's version, stored in the runner's relevel.json: a record made by another
# rule is dropped and the decision made again. 2: feet-limited fraction of GEM-X's pitch difference.
RELEVEL_RULE = "2"
FEET_MIN_RELEVEL_DEG = 0.5


def relevel_fraction(gravity: dict[str, Any] | None) -> float:
    """The fraction of GEM-X's pitch difference p a re-level applies: 1 without a usable feet
    measurement; else p clipped to the feet's interval ``feet_tilt_ci_deg`` widened by
    ``FEET_TILT_BIAS_DEG`` (the correction closest to GEM-X's that the feet do not significantly
    contradict), as a fraction of p; 0 when that leaves less than ``FEET_MIN_RELEVEL_DEG`` or the
    other sign (``feet_refute_relevel``). A fighter whose own estimate reaches p
    (``feet_tilt_per_fighter_deg``) keeps the full re-level (the fighters must agree to limit it).
    Idempotent: after applying f p the feet read their interval minus f p, which then contains 0 or
    sits beyond it, so the rest of p is clipped to ~0. 0 when GEM-X asks for nothing.

    The feet tilt (pose3d's ``afv.models.gemx.feet_tilt``) is not a pitch estimator (depth errors tilt
    it too), so it only limits GEM-X's correction, never adds one. Media Samples (round 3): Who Am I
    (camera 0.42 m below the floor) p = +4.8, feet [1.1, 2.3]: 80 %; ref walk 001 p = +3.6, feet
    [-1.9, 1.3]: 79 %; Close boxing p = +4.3, feet [-2.3, -1.6] (both fighters lean the other way):
    refuted."""
    if not _gemx_pitch_disagrees(gravity):
        return 0.0
    assert gravity is not None
    if feet_data_malformed(gravity):
        return 0.0  # never a full re-level on feet data that cannot be read (flag gravity_feet_unreadable)
    try:
        p = float(gravity["pitch_difference_deg"])
        ci = gravity.get("feet_tilt_ci_deg")
        if ci is None:
            return 1.0  # no feet measurement: GEM-X alone
        sign = float(np.sign(p))
        per = [float(v) for v in (gravity.get("feet_tilt_per_fighter_deg") or {}).values() if v is not None]
        if any(sign * v >= abs(p) for v in per):
            return 1.0
        lo, hi = min(ci) - FEET_TILT_BIAS_DEG, max(ci) + FEET_TILT_BIAS_DEG
        applied = sign * float(np.clip(p, lo, hi))
        return 0.0 if applied < FEET_MIN_RELEVEL_DEG else round(applied / abs(p), 4)
    except (KeyError, TypeError, ValueError):
        return 0.0


def feet_data_malformed(gravity: dict[str, Any] | None) -> bool:
    """The feet fields of pose3d's gravity QC are present but unreadable (not 2 finite numbers, or a
    non-numeric per-fighter value): ``relevel_fraction`` then refuses to re-level."""
    if not gravity:
        return False
    try:
        ci = gravity.get("feet_tilt_ci_deg")
        if ci is not None and (len(ci) != 2 or not np.all(np.isfinite(np.asarray(ci, dtype=np.float64)))):
            return True
        per = gravity.get("feet_tilt_per_fighter_deg") or {}
        if not isinstance(per, dict):
            return True
        vals = [v for v in per.values() if v is not None]
        return not np.all(np.isfinite(np.asarray(vals, dtype=np.float64)))
    except (TypeError, ValueError):
        return True


def feet_refute_relevel(gravity: dict[str, Any] | None) -> bool:
    """GEM-X asks for a re-level, but the planted feet leave none of it (``relevel_fraction`` 0; flag
    ``gravity_relevel_refuted_by_feet``)."""
    return _gemx_pitch_disagrees(gravity) and relevel_fraction(gravity) == 0.0


def feet_tilt_after_relevel(gravity: dict[str, Any] | None) -> list[float] | None:
    """The feet-tilt interval predicted after a re-level by ``pitch_difference_deg`` (degrees)."""
    try:
        ci = (gravity or {}).get("feet_tilt_ci_deg")
        p = float((gravity or {})["pitch_difference_deg"])
        if ci is None or len(ci) != 2:
            return None
        return [round(float(ci[0]) - p, 2), round(float(ci[1]) - p, 2)]
    except (KeyError, TypeError, ValueError):
        return None


UP_OVERRIDE_MAX_DEG = 10.0  # an override further than this from GeoCalib's up is flagged


def validate_up_cam0(up: Any) -> np.ndarray:
    """``CameraParams.up_cam0`` as a unit vector: 3 finite values, non-zero, pointing UP in OpenCV
    camera axes (y < 0: y points down); raises ``StageError`` otherwise."""
    u = np.asarray(up, dtype=np.float64).reshape(-1)
    if u.shape != (3,) or not np.isfinite(u).all() or np.linalg.norm(u) < 1e-9:
        raise StageError(f"up_cam0 must be 3 finite numbers, not all zero: {up!r}")
    u = u / np.linalg.norm(u)
    if u[1] >= 0.0:
        raise StageError(f"up_cam0 {u.round(4).tolist()} does not point up (OpenCV axes: y < 0 is up)")
    return u


def _pitch_roll_deg(u: np.ndarray) -> tuple[float, float]:
    """Pitch (towards +z, forward) and roll (towards +x) of an OpenCV-axes up vector, degrees."""
    return float(np.degrees(np.arctan2(u[2], -u[1]))), float(np.degrees(np.arctan2(u[0], -u[1])))


def pitch_only_up(override: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``reference`` (GeoCalib's up, camera 0 OpenCV axes) rotated about camera 0's x axis until its
    pitch is ``override``'s: the re-level pass changes only the pitch it was triggered by and keeps
    GeoCalib's roll (well observed even on telephoto)."""
    a = np.radians(_pitch_roll_deg(override)[0] - _pitch_roll_deg(reference)[0])
    c, s_ = np.cos(a), np.sin(a)
    x, y, z = reference
    out = np.array([x, -(c * (-y) - s_ * z), s_ * (-y) + c * z])
    return out / np.linalg.norm(out)


def relevel_override(pose3d_qc: dict[str, Any] | None) -> dict[str, Any] | None:
    """Camera params for the re-level pass, or None when none is needed.

    Reads pose3d's ``qc.json`` dict: ``qc["gravity"]["pitch_difference_deg"]``,
    ``qc["gravity"]["height_change_if_gemx_m"]`` (the ``gravity_disagrees`` test) and
    ``qc["gravity"]["gemx_up_cam0"]`` (GEM-X's up in camera 0's OpenCV axes). Returns
    ``{"up_cam0": [x, y, z]}`` to merge into the camera stage's params; re-run camera (its ViPE /
    flow / normalise outputs come from ``<shot>/.cache/camera``, so only the CPU part runs) and then
    pose3d, once. The planted feet can limit it to a fraction of GEM-X's pitch difference
    (``relevel_fraction``; the up then goes that fraction of the way from the camera's up,
    ``qc["gravity"]["camera_up_cam0"]``, to GEM-X's) or refute it.
    """
    gravity = (pose3d_qc or {}).get("gravity")
    if not gravity_disagrees(gravity):
        return None
    assert gravity is not None
    up = gravity.get("gemx_up_cam0")
    if up is None or len(up) != 3 or not np.all(np.isfinite(up)) or np.linalg.norm(up) < 1e-6:
        return None
    f = relevel_fraction(gravity)
    if f >= 1.0:
        return {"up_cam0": [float(v) for v in up]}
    cam_up = gravity.get("camera_up_cam0")
    if cam_up is None or len(cam_up) != 3 or not np.all(np.isfinite(cam_up)):
        return None  # limited by the feet, but the camera's own up is unknown: never the full re-level
    part = _slerp_unit(np.asarray(cam_up, dtype=np.float64), np.asarray(up, dtype=np.float64), f)
    return {"up_cam0": [round(float(v), 6) for v in part]}  # part of the way


def _slerp_unit(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    ang = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
    if ang < 1e-9:
        return b
    out = (np.sin((1.0 - t) * ang) * a + np.sin(t * ang) * b) / np.sin(ang)
    return out / np.linalg.norm(out)


# ---------------------------------------------------------------------------------------------
# Worker output cache: <shot>/.cache/camera/<key>/ (the re-level pass re-runs only the CPU part)
# ---------------------------------------------------------------------------------------------

CACHE_VERSION = "1"
_OUTPUT_KEYS = ("out_npz", "out_mp4")  # request keys naming files a worker writes
_INPUT_FILE_KEYS = ("video", "boxes_npz")  # request keys naming files a worker reads


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_identity(ctx: Ctx, video: str) -> str:
    """The source video's identity for cache keys: the project's sha256 when this is the project's
    source, always with size and mtime (a replaced file never matches)."""
    path = Path(video)
    st = path.stat()
    sha = ""
    try:
        from afv.core.project import shot_context

        sc = shot_context(Path(ctx.shot_dir))
        if Path(sc.source_path).resolve() == path.resolve():
            sha = sc.source_sha256
    except Exception:
        pass
    return (
        f"sha256={sha}:size={st.st_size}:mtime={st.st_mtime_ns}"
        if sha
        else f"{path.resolve()}:{st.st_size}:{st.st_mtime_ns}"
    )


def env_fingerprint(env: str) -> str:
    """Identity of a model environment for cache keys: its interpreter's realpath, pyvenv.cfg and
    site-packages (size/mtime: an install or upgrade changes them) and its repo's git HEAD commit."""
    try:
        from afv.models.envs import get_env

        spec = get_env(env)
    except Exception as e:
        return f"unknown env {env}: {e}"
    parts = [f"exe={os.path.realpath(spec.executable)}"]
    venv = Path(spec.executable).parent.parent
    for f in [venv / "pyvenv.cfg", *sorted(venv.glob("lib/python*/site-packages"))]:
        try:
            st = f.stat()
            parts.append(f"{f.name}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            pass
    if spec.cwd is not None:
        head = Path(spec.cwd) / ".git" / "HEAD"
        try:
            ref = head.read_text().strip()
            if ref.startswith("ref:"):
                ref_file = head.parent / ref.split(":", 1)[1].strip()
                ref = ref_file.read_text().strip() if ref_file.exists() else ref
            parts.append(f"git={ref}")
        except OSError:
            pass
    return ";".join(parts)


def ffmpeg_identity() -> str:
    """The pinned ffmpeg binary (``afv.media.ffmpeg.ffmpeg_path``): realpath, size, mtime."""
    try:
        from afv.media.ffmpeg import ffmpeg_path

        path = os.path.realpath(ffmpeg_path())
        st = os.stat(path)
        return f"{path}:{st.st_size}:{st.st_mtime_ns}"
    except Exception as e:
        return f"unknown ffmpeg: {e}"


# Which base fields reach each worker command (so e.g. switching vipe -> static keeps the normalised
# video, and only ViPE depends on the stage's weights). Unknown commands get all of them.
_BASE_FIELDS: dict[str, tuple[str, ...]] = {
    "normalize": ("frame_range", "ffmpeg"),
    "vipe": ("frame_range", "deterministic", "seed", "weights"),
    "gravity": ("frame_range", "deterministic", "seed"),
    "flow": ("frame_range",),
    "background_camera": ("frame_range",),
}
CACHE_KEEP_PER_COMMAND = 8  # newest entries kept per worker command (older ones are pruned)
CACHE_TMP_MAX_AGE_S = 3600.0  # orphaned temporary folders (a killed run) older than this are swept


@dataclass
class WorkerCache:
    """Content-keyed cache of the camera stage's worker outputs (ViPE + GeoCalib, flow tracks,
    normalised video, background camera, GeoCalib gravity) in ``<shot>/.cache/camera/<key>/``.

    Key = sha256 of: cache version; the command, its worker script's name and sha256; the fingerprint
    of the environment it runs in (``env_fingerprint``); the ``base`` fields that reach that command
    (``_BASE_FIELDS``: frame range, deterministic/seed for GPU solves, the stage's weight hash for ViPE,
    the pinned ffmpeg's identity for normalising); the request without its file paths; and the CONTENT
    of its inputs (the source video by ``source_id``; the normalised video and other input files by
    sha256; ``boxes_npz`` by its boxes array). CPU-side parameters (gravity override, ground fit,
    rotation choice, zoom handling) reach no worker, so changing them hits.

    ``force`` (``Ctx.force``: the user forced the stage) recomputes every call and refreshes its entry.
    Entries are written to a temporary folder and renamed (never half-written), carry their full key
    material and each file's sha256, and are verified on every read: a mismatch deletes the entry and
    recomputes; failing to copy a VALID entry to the work folder recomputes without deleting it.
    Orphaned temporary folders older than ``CACHE_TMP_MAX_AGE_S`` are swept; after each store, entries
    of the same command made by another version of its worker script are pruned, and only the newest
    ``CACHE_KEEP_PER_COMMAND`` per command are kept.
    """

    root: Path
    base: dict[str, Any]
    source_id: str
    force: bool = False
    events: list[list[str]] = field(default_factory=list)
    cache_only: bool = False  # [afv-core C, round 3] Ctx.cache_only: a miss raises CacheOnlyMiss

    def __post_init__(self) -> None:
        self._env_fp: dict[str, str] = {}
        self.sweep()

    def sweep(self, max_age_s: float = CACHE_TMP_MAX_AGE_S) -> None:
        import time

        if not self.root.exists():
            return
        now = time.time()
        for tmp in self.root.glob(".tmp-*"):
            try:
                if now - tmp.stat().st_mtime > max_age_s:
                    shutil.rmtree(tmp, ignore_errors=True)
            except OSError:
                pass

    def key(self, env: str, script: str, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        cmd = str(request.get("command"))
        inputs: dict[str, str] = {}
        for k in _INPUT_FILE_KEYS:
            if k not in request:
                continue
            if k == "video" and cmd == "normalize":
                inputs[k] = self.source_id
            elif k == "boxes_npz":
                with np.load(request[k]) as z:
                    a = np.ascontiguousarray(z["boxes"], dtype=np.float32)
                inputs[k] = hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest()
            else:
                inputs[k] = file_sha256(request[k])
        if env not in self._env_fp:
            self._env_fp[env] = env_fingerprint(env)
        fields = _BASE_FIELDS.get(cmd, tuple(self.base))
        material = {
            "cache": CACHE_VERSION,
            "command": cmd,
            "script_name": script,
            "script": file_sha256(WORKERS / script),
            "env": self._env_fp[env],
            "base": {k: self.base.get(k) for k in fields},
            "request": {k: v for k, v in request.items() if k not in _OUTPUT_KEYS + _INPUT_FILE_KEYS},
            "inputs": inputs,
        }
        blob = json.dumps(material, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:32], json.loads(blob)

    def call(
        self, env: str, script: str, request: dict[str, Any], work_dir: Path, **kw: Any
    ) -> dict[str, Any]:
        cmd = str(request.get("command"))
        try:
            key, material = self.key(env, script, request)
        except Exception as e:  # unreadable input: run uncached
            self.events.append([cmd, f"uncached ({e})"])
            if self.cache_only:  # [afv-core C, round 3] cache-only run: never launch a worker
                raise CacheOnlyMiss(f"camera {cmd}: worker input unreadable for the cache ({e})") from None
            return run_env_worker(env, script, request, work_dir, **kw)
        entry = self.root / key
        if not self.force:
            resp = self._load(entry, request, material)
            if resp is not None:
                self.events.append([cmd, "hit"])
                return resp
        if self.cache_only:  # [afv-core C, round 3] cache-only run: never launch a worker
            raise CacheOnlyMiss(f"camera {cmd}: no cached {script} output for this request")
        resp = run_env_worker(env, script, request, work_dir, **kw)
        how = "forced" if self.force else "miss"
        try:
            self._store(entry, request, resp, material)
            self.prune(material, keep=entry)
            self.events.append([cmd, how])
        except Exception as e:  # caching is best effort
            self.events.append([cmd, f"{how}, not stored ({e})"])
        return resp

    def _load(self, entry: Path, request: dict[str, Any], material: dict[str, Any]) -> dict[str, Any] | None:
        manifest = entry / "manifest.json"
        if not manifest.exists():
            return None
        try:  # verify: anything wrong with the entry itself deletes it
            m = json.loads(manifest.read_text())
            if m["material"] != material:
                raise ValueError("key material differs")
            for k, f in m["files"].items():
                if k not in request or file_sha256(entry / f["name"]) != f["sha256"]:
                    raise ValueError(f"{k} missing or changed")
        except Exception:
            shutil.rmtree(entry, ignore_errors=True)
            return None
        try:  # a valid entry that cannot be copied out (disk full, permissions) is kept
            for k, f in m["files"].items():
                Path(request[k]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(entry / f["name"], request[k])
        except OSError:
            return None
        os.utime(manifest)  # recently used: kept by prune
        return _swap_paths(m["response"], {f"@{k}": str(request[k]) for k in m["files"]})

    def _store(
        self, entry: Path, request: dict[str, Any], resp: dict[str, Any], material: dict[str, Any]
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / f".tmp-{uuid.uuid4().hex}"
        tmp.mkdir()
        try:
            files = {}
            for k in _OUTPUT_KEYS:
                if k in request and Path(request[k]).exists():
                    name = k + Path(request[k]).suffix
                    shutil.copyfile(request[k], tmp / name)
                    files[k] = {"name": name, "sha256": file_sha256(tmp / name)}
            swapped = _swap_paths(resp, {str(request[k]): f"@{k}" for k in files})
            (tmp / "manifest.json").write_text(
                json.dumps({"material": material, "files": files, "response": swapped}, default=str)
            )
            if entry.exists():
                shutil.rmtree(entry, ignore_errors=True)
            os.replace(tmp, entry)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def prune(self, material: dict[str, Any], keep: Path, max_entries: int = CACHE_KEEP_PER_COMMAND) -> None:
        """Drop this command's entries made by another worker-script version, and all but the newest
        ``max_entries`` of the rest (``keep`` always stays)."""
        same: list[tuple[float, Path]] = []
        for m in self.root.glob("*/manifest.json"):
            try:
                other = json.loads(m.read_text())["material"]
            except Exception:
                continue
            if other.get("command") != material["command"] or m.parent == keep:
                continue
            if (
                other.get("script_name") == material["script_name"]
                and other.get("script") != material["script"]
            ):
                shutil.rmtree(m.parent, ignore_errors=True)
                continue
            same.append((m.stat().st_mtime, m.parent))
        for _, entry in sorted(same, reverse=True)[max(0, max_entries - 1) :]:
            shutil.rmtree(entry, ignore_errors=True)

    def summary(self) -> dict[str, str]:
        return {cmd: how for cmd, how in self.events}


def _swap_paths(obj: Any, mapping: dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: _swap_paths(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_swap_paths(v, mapping) for v in obj]
    return mapping.get(obj, obj) if isinstance(obj, str) else obj


_CACHE: ContextVar[WorkerCache | None] = ContextVar("afv_camera_cache", default=None)


def _worker(env: str, script: str, request: dict[str, Any], work_dir: Path, **kw: Any) -> dict[str, Any]:
    """``run_env_worker`` through the current camera run's cache (if any)."""
    cache = _CACHE.get()
    if cache is None:
        return run_env_worker(env, script, request, work_dir, **kw)
    return cache.call(env, script, request, work_dir, **kw)


# ---------------------------------------------------------------------------------------------
# Shot source (project/shot JSON written by module A; params can override)
# ---------------------------------------------------------------------------------------------


def shot_source(ctx: Ctx, video: str | None, start: float | None, duration: float | None) -> dict[str, Any]:
    """ORIGINAL source video, start and duration of this shot (``afv.core.project.shot_context``).

    Params override the project's values (tests, one-off runs). Every normalisation (camera stage,
    pose3d) re-encodes from this original source + interval, so all stages share GEM-X's frames.
    """
    shot: dict[str, Any] = {}
    try:
        from afv.core.project import shot_context

        sc = shot_context(Path(ctx.shot_dir))
        shot = {"video": sc.source_path, "start": sc.start, "duration": sc.duration}
    except Exception as e:  # no project around this shot folder
        if video is None:
            raise StageError(f"no shot context for {ctx.shot_dir} and no params.video: {e}") from e
    src = video or shot["video"]
    start = start if start is not None else shot.get("start", 0.0)
    duration = duration if duration is not None else shot.get("duration")
    return {
        "video": str(src),
        "start": float(start or 0.0),
        "duration": None if duration is None else float(duration),
    }


def shot_timeline(n: int, start: float, tracks: Tracks | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``frames``/``timestamps`` of the shot's 30 Hz timeline (taken from tracks when they match)."""
    if tracks is not None and tracks.num_frames == n:
        return tracks.frames, tracks.timestamps
    return np.arange(n), start + np.arange(n) / SHOT_FPS


def fighter_boxes(tracks: Tracks) -> np.ndarray:
    """Boxes [F,K,4] of the picked fighters (all tracks if none are picked)."""
    ids = list(tracks.fighters.values()) or list(tracks.track_ids)
    cols = [int(np.flatnonzero(tracks.track_ids == i)[0]) for i in ids]
    return np.asarray(tracks.boxes, dtype=np.float64)[:, cols]


# ---------------------------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------------------------

_VIPE_LICENCES = (
    LicenceRecord(
        "ViPE",
        "Apache-2.0",
        "none",
        "not reviewed",
        True,
        "https://github.com/nv-tlabs/vipe",
        retrieved="2026-11-09",
        licence_text_path="licences/2026-11-09/ViPE_LICENSE.txt",
    ),
    LicenceRecord(
        "DROID-SLAM droid.pth (via ViPE)",
        "BSD-3-Clause",
        "repo licence (assumed)",
        "not reviewed",
        True,
        weight_sha256="46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526",
        retrieved="2026-11-09",
    ),
    LicenceRecord(
        "MoGe-2 ViT-L (via ViPE)",
        "MIT",
        "MIT",
        "not reviewed",
        True,
        weight_sha256="3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5",
        retrieved="2026-11-09",
    ),
    LicenceRecord(
        "GroundingDINO Swin-T (via ViPE)",
        "Apache-2.0",
        "repo licence (assumed)",
        "not reviewed",
        True,
        weight_sha256="3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
        retrieved="2026-11-09",
    ),
    LicenceRecord(
        "bert-base-uncased (via ViPE)",
        "Apache-2.0",
        "Apache-2.0",
        "not reviewed",
        True,
        weight_sha256="68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3",
        retrieved="2026-11-09",
    ),
    LicenceRecord(
        "SAM ViT-B (via ViPE)",
        "Apache-2.0",
        "Apache-2.0",
        "not reviewed",
        True,
        weight_sha256="ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
        retrieved="2026-11-09",
    ),
    LicenceRecord(
        "DeAOT R50 (via ViPE)",
        "BSD-3-Clause",
        "repo licence (assumed)",
        "not reviewed",
        True,
        weight_sha256="7e8a8d83310739bac02817f6bf48b6bbe2bbd7d5325722f1084088eb3aee1e06",
        retrieved="2026-11-09",
    ),
)
_GEOCALIB = LicenceRecord(
    "GeoCalib pinhole (via ViPE)",
    "Apache-2.0",
    "CC BY 4.0",
    "not reviewed",
    True,
    "https://github.com/cvg/GeoCalib",
    weight_sha256="86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac",
    retrieved="2026-11-09",
    notes="credit required (CC BY 4.0); used for gravity and ViPE's initial focal",
)


class CameraParams(BaseModel):
    gravity: Literal["geocalib", "level"] = "geocalib"
    # World up in camera 0's OpenCV axes (x right, y down, z forward) for a re-level pass, e.g. pose3d's
    # qc.json "gravity" -> "gemx_up_cam0" (``relevel_override``) when GeoCalib's pitch is weak (telephoto).
    # "pitch": GeoCalib's up rotated to this up's pitch (GeoCalib's roll kept; GeoCalib needed);
    # "full": this up as it is.
    up_cam0: list[float] | None = None
    up_override: Literal["pitch", "full"] = "pitch"
    gravity_samples: int = 12
    pipeline: str = "pose_only_long"  # ViPE pipeline (vipe mode)
    use_translation: bool = False  # ViPE translations (GEM-X itself uses rotations only)
    focal_px: float | None = None  # static/background: None = GEM-X default (image diagonal)
    person_height_m: float = STANCE_HEIGHT_M  # ground prior: fighters' box height in stance
    default_camera_height_m: float = 1.5
    # vipe mode: where the rotations come from. "auto" = ViPE's while they explain the background's
    # image motion (``vipe_disagreement`` below), else background flow when that is usable (enough
    # points, small residual); "flow" = always flow; "vipe" = ViPE only, no check.
    # ViPE is kept where it works: it models translation (parallax of close, moving wide-lens cameras
    # biases a rotation-only fit; UFC cross: 2.3 deg of pitch drift over 10 s) and bundle-adjusts.
    # "auto" decides per window too (``disagreeing_pairs``): where ViPE fails locally (UFC cross: during
    # its zoom, which ViPE reads as a dolly) flow's increments replace ViPE's there ("vipe+flow");
    # beyond max_splice_fraction of the shot, or with ViPE's shot median off, flow is used throughout.
    rotation_source: Literal["auto", "flow", "vipe"] = "auto"
    rotation_window: int = 30  # frame pairs per local ViPE-vs-flow decision (1 s)
    min_splice_pairs: int = 10  # shorter local disagreements are ignored
    max_splice_fraction: float = 0.5
    # flow: "apply" (default) fits the zoom with the rotations and, above min_zoom_range, makes
    # focal_px per frame (CameraTrack and GEM-X's K; anchored at ViPE's calibration frames; with ViPE's
    # rotations kept or spliced too, but rotation_source="vipe" runs no flow, so no zoom is measured
    # there); "flag" only measures and flags it (one focal per shot); "off" fits
    # rotations only.
    zoom: Literal["apply", "flag", "off"] = "apply"
    min_zoom_range: float = 0.10  # smaller "zooms" are within what parallax and drift produce
    flow_min_valid: float = 0.8  # fraction of frame pairs with enough background points
    flow_max_residual_px: float = 1.5  # median background residual of the flow camera
    # ViPE disagrees (``vipe_disagreement``) when its residual > max(ratio x flow's, floor) or it drifts
    # from the flow camera by > max_drift_m (or > max_drift_m_s in any second) at the fighters' distance.
    # UFC cross (wide, moving camera, parallax): ViPE 0.52 px vs flow 0.18 px and 0.14 m of drift: kept.
    vipe_residual_ratio: float = 3.0
    vipe_residual_floor_px: float = 0.75
    max_drift_m: float = 0.3
    max_drift_m_s: float = 0.1
    video: str | None = None  # overrides the shot context (afv.core.project.shot_context)
    start: float | None = None
    duration: float | None = None
    timeout_s: float = 1800.0
    cache: bool = True  # reuse worker outputs from <shot>/.cache/camera (``WorkerCache``)


QC_FILE = "qc.json"  # the runner puts qc.json's "summary" into meta.json


class _CameraStage:
    """Shared run(); subclasses pick the rotation source."""

    name: ClassVar[str] = "camera"
    # 2: background-flow rotations, zoom measured; 3: zoom applied (per-frame focal), local ViPE/flow splice
    version: ClassVar[str] = "3"
    inputs: ClassVar[tuple[str, ...]] = ("detect_track",)
    Params: ClassVar[type[BaseModel]] = CameraParams
    mode: ClassVar[str] = ""

    def run(self, ctx: Ctx, inputs: dict[str, Result], params: BaseModel) -> Result:
        assert isinstance(params, CameraParams)
        cache = None
        if params.cache:
            src = shot_source(ctx, params.video, params.start, params.duration)
            try:
                source_id = source_identity(ctx, src["video"])
            except OSError:
                source_id = ""
            if source_id:
                base = {
                    "weights": getattr(self, "weight_hash", ""),
                    "deterministic": bool(ctx.deterministic),
                    "seed": int(ctx.seed),
                    "frame_range": list(ctx.frame_range) if ctx.frame_range else None,
                    "ffmpeg": ffmpeg_identity(),
                }
                force = bool(getattr(ctx, "force", False))
                cache = WorkerCache(Path(ctx.shot_dir) / ".cache" / "camera", base, source_id, force=force,
                                    cache_only=bool(getattr(ctx, "cache_only", False)))  # fmt: skip
        if cache is None and getattr(ctx, "cache_only", False):  # [afv-core C, round 3]
            raise CacheOnlyMiss("camera: the worker cache is off for this shot")
        token = _CACHE.set(cache)
        try:
            return self._run(ctx, inputs, params)
        finally:
            _CACHE.reset(token)

    def _run(self, ctx: Ctx, inputs: dict[str, Result], params: CameraParams) -> Result:
        tracks = inputs.get("detect_track")
        if tracks is not None and not isinstance(tracks, Tracks):
            raise StageError("camera: detect_track input is not a Tracks result")
        src = shot_source(ctx, params.video, params.start, params.duration)
        work = Path(ctx.work_dir)

        def say(msg: str) -> None:
            ctx.progress(-1.0, msg)

        say("normalising the shot to GEM-X's 30 Hz timeline")
        norm = _worker(
            "gemx",
            "gemx_worker.py",
            {"command": "normalize", **src, "out_mp4": str(work / "shot30.mp4")},
            work,
            gpu=False,
            progress=ctx.progress,
        )
        n, width, height = int(norm["frame_count"]), int(norm["width"]), int(norm["height"])
        if tracks is not None and tracks.num_frames != n:
            raise StageError(f"camera: tracks have {tracks.num_frames} frames, GEM-X's timeline has {n}")
        frames, timestamps = shot_timeline(n, src["start"], tracks)
        r_v_cv, centres_v, focal, cx, cy, up_v, info = self.solve(ctx, params, norm, tracks, say)
        ups = info.pop("_up", None)  # raw GeoCalib ups (camera frames), to re-express in another V
        # One focal per shot from the solver (ViPE freezes its intrinsics; ViPE's per-frame focal is in
        # qc.json); a zoom measured by the background flow makes it per frame (zoom="apply").
        focal = np.full(n, float(np.median(focal)))
        flags: list[str] = []
        if self.mode == "vipe" and params.rotation_source != "vipe":
            r_flow, zoom_focal = self._flow_camera(
                ctx, params, norm, tracks, r_v_cv, float(focal[0]), cx, cy, info, flags
            )
            if r_flow is not None:
                r_v_cv = r_flow
                if params.use_translation:
                    flags.append("translation_dropped")  # ViPE's translations live in ViPE's world
                centres_v = None
                if ups is not None:
                    up_v = average_up(r_v_cv, *ups)
            if zoom_focal is not None:
                focal = zoom_focal
                if params.use_translation and centres_v is not None:
                    # ViPE read the zoom as a dolly: its translations would count the zoom twice
                    centres_v = None
                    flags.append("translation_dropped")
        override = validate_up_cam0(params.up_cam0) if params.up_cam0 is not None else None
        full_override = override is not None and params.up_override == "full"
        if up_v is None and params.gravity == "geocalib" and not full_override:
            # A focal prior only when it is known (ViPE's or the user's); GEM-X's diagonal default is
            # a guess that, fixed in GeoCalib, tilts the world (~2 deg on telephoto shots).
            known = float(np.median(focal)) if (self.mode == "vipe" or params.focal_px) else None
            up_v = self._geocalib_up(ctx, params, norm, r_v_cv, known, info)
        if override is not None:
            up_v = self._apply_up_override(override, params, r_v_cv, up_v, info, flags)
        cam = leveled_track(
            r_v_cv,
            centres_v if params.use_translation else None,
            focal,
            up_v=up_v,
            width=width,
            height=height,
            cx=cx,
            cy=cy,
            frames=frames,
            timestamps=timestamps,
            static=self.mode == "static",
        )
        boxes = fighter_boxes(tracks) if tracks is not None else np.zeros((0, 0, 4))
        cam, ground = place_on_ground(
            cam,
            boxes,
            person_height=params.person_height_m,
            default_camera_height=params.default_camera_height_m,
        )
        info["ground"] = ground
        info["ffmpeg"] = norm.get("ffmpeg")
        flags += [ground["flag"]] if "flag" in ground else []
        if not str(info.get("gravity", "")).startswith(("geocalib", "override")):
            flags.append("gravity_not_estimated")
        info["summary"] = {
            "mode": self.mode,
            "frames": n,
            "focal_px": round(float(np.median(focal)), 2),
            "focal_px_range": [round(float(np.min(focal)), 2), round(float(np.max(focal)), 2)],
            "rotation_source": info.get("rotation_source", self.mode),
            **(
                {
                    "zoom_range": info["flow"].get("zoom_range"),
                    "zoom_applied": info["flow"].get("zoom_applied", False),
                }
                if "flow" in info
                else {}
            ),
            "gravity": info.get("gravity"),
            "camera_height_m": ground["camera_height_m"],
            "camera_distance_m": ground["camera_distance_m"],
            "ground_method": ground["method"],
            "ground_samples": ground["ground_samples"],
            "flags": flags,
        }
        cache = _CACHE.get()
        info["cache"] = cache.summary() if cache is not None else "off"
        (work / QC_FILE).write_text(json.dumps(info, indent=2, default=str))
        f_lo, f_hi = float(np.min(focal)), float(np.max(focal))
        say(
            f"camera {self.mode}: {n} frames, focal "
            + (f"{f_lo:.0f} px, " if f_hi - f_lo < 0.5 else f"{f_lo:.0f}-{f_hi:.0f} px (zoom), ")
            + f"gravity {info.get('gravity')}, "
            f"camera height {ground['camera_height_m']} m ({ground['method']})"
            + (f" FLAGS {flags}" if flags else "")
        )
        return cam

    def solve(self, ctx, params, norm, tracks, say):  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def _apply_up_override(
        override: np.ndarray,
        params: CameraParams,
        r_v_cv: np.ndarray,
        up_v: np.ndarray | None,
        info: dict[str, Any],
        flags: list[str],
    ) -> np.ndarray:
        """World up (in V) from ``up_cam0``: pitch-only on GeoCalib's up (default) or as given."""
        report: dict[str, Any] = {"up_cam0": override.round(5).tolist(), "mode": params.up_override}
        used = override
        if up_v is not None:
            ref = r_v_cv[0].T @ (np.asarray(up_v, dtype=np.float64) / np.linalg.norm(up_v))
            (p_o, r_o), (p_r, r_r) = _pitch_roll_deg(override), _pitch_roll_deg(ref)
            angle = float(np.degrees(np.arccos(np.clip(override @ ref, -1.0, 1.0))))
            report |= {
                "pitch_change_deg": round(p_o - p_r, 3),
                "roll_difference_deg": round(r_o - r_r, 3),
                "angle_from_geocalib_deg": round(angle, 3),
            }
            if params.up_override == "pitch":
                used = pitch_only_up(override, ref)
            if angle > UP_OVERRIDE_MAX_DEG:
                flags.append("up_override_far_from_geocalib")
        elif params.up_override == "pitch":
            report["note"] = "no GeoCalib up to keep the roll of: used as given"
        info["gravity"] = "override"
        info["up_override"] = report
        return r_v_cv[0] @ used

    @staticmethod
    def _flow_camera(
        ctx: Ctx,
        params: CameraParams,
        norm: dict[str, Any],
        tracks: Tracks | None,
        r_vipe: np.ndarray,
        focal: float,
        cx: float,
        cy: float,
        info: dict[str, Any],
        flags: list[str],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Background-flow rotations (+ zoom), checked against ViPE's on the same background tracks.

        Returns ``(rotations, focal)``: camera-to-V rotations replacing ViPE's (flow's throughout, or
        ViPE's with flow's increments over the windows where ViPE fails, ``disagreeing_pairs``), None to
        keep ViPE's; and the per-frame focal when a measured zoom is applied (None: one focal)."""
        work = Path(ctx.work_dir)
        n = r_vipe.shape[0]
        try:
            req: dict[str, Any] = {
                "command": "flow",
                "video": norm["out_mp4"],
                "out_npz": str(work / "flow.npz"),
            }
            if tracks is not None and tracks.num_frames:
                np.savez_compressed(work / "flow_boxes.npz", boxes=np.asarray(tracks.boxes, np.float32))
                req["boxes_npz"] = str(work / "flow_boxes.npz")
            resp = _worker(
                "vipe",
                "vipe_worker.py",
                req,
                work,
                gpu=False,
                progress=ctx.progress,
                timeout=params.timeout_s,
            )
            with np.load(resp["out_npz"]) as z:
                p0, p1, count = z["p0"], z["p1"], z["count"]
                static_fraction = float(z["static_fraction"])
            if len(count) != n - 1:
                raise ValueError(f"flow has {len(count)} frame pairs for {n} frames")
            fc = flow_rotations(
                p0,
                p1,
                count,
                focal,
                cx,
                cy,
                zoom=params.zoom != "off",
                apply_zoom=params.zoom == "apply",
                min_zoom_range=params.min_zoom_range,
                zoom_reference=vipe_focal_frames(n),
            )
        except CacheOnlyMiss:  # [afv-core C, round 3] not a flow failure: the run is cache-only
            raise
        except Exception as e:  # ViPE's rotations stay; reported
            info["rotation_source"] = "vipe"
            info["flow"] = {"error": str(e)}
            flags.append("flow_rotation_failed")
            ctx.progress(-1.0, f"background flow failed, keeping ViPE's rotations: {e}")
            return None, None
        # Both cameras on the same points with the focal the camera will have (per frame when the zoom
        # is applied), each pair with its best zoom on top (a fair rotation comparison).
        res_vipe = rotation_residual_px(p0, p1, count, r_vipe, fc.focal_px, cx, cy, fit_zoom=True)
        res_flow = rotation_residual_px(p0, p1, count, fc.r_v_cv, fc.focal_px, cx, cy, fit_zoom=True)
        vipe_med, flow_med = _nanstat(res_vipe, 50), _nanstat(res_flow, 50)
        usable = bool(
            fc.valid.mean() >= params.flow_min_valid
            and flow_med is not None
            and flow_med <= params.flow_max_residual_px
        )
        dist = fighter_distance_m(
            tracks.boxes if tracks is not None else np.zeros((0, 4)), focal, norm["height"]
        )
        drift_deg = relative_drift_deg(r_vipe, fc.r_v_cv)
        drift_m = float(np.radians(drift_deg.max())) * dist
        drift_m_s = float(np.radians(drift_rate_deg_s(r_vipe, fc.r_v_cv))) * dist
        reasons = (
            vipe_disagreement(
                vipe_med,
                flow_med,
                drift_m,
                drift_m_s,
                ratio=params.vipe_residual_ratio,
                floor_px=params.vipe_residual_floor_px,
                max_drift_m=params.max_drift_m,
                max_drift_m_s=params.max_drift_m_s,
            )
            if usable
            else ([] if vipe_med is None or vipe_med <= params.flow_max_residual_px else ["residual"])
        )
        # Local decision: the windows where ViPE fails while flow explains the background.
        bad = np.zeros(len(count), bool)
        if usable and params.rotation_source == "auto":
            bad = disagreeing_pairs(
                res_vipe,
                res_flow,
                r_vipe,
                fc.r_v_cv,
                dist,
                window=params.rotation_window,
                ratio=params.vipe_residual_ratio,
                floor_px=params.vipe_residual_floor_px,
                max_flow_px=params.flow_max_residual_px,
                max_drift_m_s=params.max_drift_m_s,
                min_len=params.min_splice_pairs,
            )
        source, rotations = "vipe", None
        spliced_drift_m = None
        if params.rotation_source == "flow" or (
            usable and ("residual" in reasons or (len(bad) and bad.mean() > params.max_splice_fraction))
        ):
            source, rotations = "flow", fc.r_v_cv
        elif usable and bad.any():
            spliced = splice_rotations(r_vipe, fc.r_v_cv, bad)
            spliced_drift_m = float(np.radians(relative_drift_deg(spliced, fc.r_v_cv).max())) * dist
            if spliced_drift_m > params.max_drift_m:  # ViPE drifts outside the windows too
                source, rotations = "flow", fc.r_v_cv
            else:
                source, rotations = "vipe+flow", spliced
        elif usable and reasons:  # slow drift over the whole shot, no window to blame
            source, rotations = "flow", fc.r_v_cv
        apply_zoom = fc.zoom_applied and (usable or source == "flow")
        focal_out = fc.focal_px if apply_zoom else None
        zoom_range = float(fc.zoom.max() / fc.zoom.min())
        info["flow"] = {
            "pairs": len(count),
            "valid_fraction": round(float(fc.valid.mean()), 3),
            "median_points": float(np.median(count)) if len(count) else 0.0,
            "static_fraction": round(static_fraction, 4),
            "residual_px": {"median": flow_med, "p90": _nanstat(res_flow, 90)},
            "vipe_residual_px": {"median": vipe_med, "p90": _nanstat(res_vipe, 90)},
            "rotation_deg_per_s_median": {"flow": _rotation_rate(fc.r_v_cv), "vipe": _rotation_rate(r_vipe)},
            "zoom_range": round(zoom_range, 4),
            "zoom_reference_frames": vipe_focal_frames(n),
            "usable": usable,
            "fighter_distance_m": round(dist, 2),
            "vipe_drift": {
                "max_deg": round(float(drift_deg.max()), 3),
                "max_m": round(drift_m, 3),
                "max_m_per_s": round(drift_m_s, 3),
            },
            "vipe_disagrees": reasons,
            "vipe_disagrees_pairs": mask_spans(bad),
            "vipe_disagrees_fraction": round(float(bad.mean()), 3) if len(bad) else 0.0,
        }
        if spliced_drift_m is not None:
            info["flow"]["spliced_drift_m"] = round(spliced_drift_m, 3)
        if reasons:
            flags.append("vipe_rotation_disagrees")
        use = rotations is not None
        if not use and reasons:
            flags.append("camera_rotation_unreliable")
        elif not use and vipe_med is None:
            flags.append("camera_rotation_unverified")  # no background to check ViPE against
        if use and not usable:
            flags.append("flow_rotation_weak")
        info["flow"]["vipe_explains_background"] = not reasons and vipe_med is not None and not bad.any()
        info["flow"]["zoom_applied"] = focal_out is not None
        if zoom_range - 1.0 > params.min_zoom_range and focal_out is None:
            flags.append("zoom")  # one focal per shot: GEM-X reads the zoom as the fighters' depth
        final_r = r_vipe if rotations is None else rotations
        final_f = np.full(n, focal) if focal_out is None else focal_out
        res_final = rotation_residual_px(p0, p1, count, final_r, final_f, cx, cy)
        info["flow"]["camera_residual_px"] = {
            "median": _nanstat(res_final, 50),
            "p90": _nanstat(res_final, 90),
        }
        info["rotation_source"] = source
        ctx.progress(
            -1.0,
            f"rotations from {source}: background residual flow {flow_med} px, "
            f"ViPE {vipe_med} px, ViPE drift {drift_m:.2f} m ({drift_m_s:.2f} m/s)"
            + (f", ViPE replaced on pairs {mask_spans(bad)}" if source == "vipe+flow" else "")
            + f"; zoom x{zoom_range:.3f}"
            + (" (applied)" if focal_out is not None else ""),
        )
        return rotations, focal_out

    @staticmethod
    def _geocalib_up(ctx, params, norm, r_v_cv, focal: float | None, info) -> np.ndarray | None:
        work = Path(ctx.work_dir)
        try:
            resp = _worker(
                "vipe",
                "vipe_worker.py",
                {
                    "command": "gravity",
                    "video": norm["out_mp4"],
                    "out_npz": str(work / "gravity.npz"),
                    "focal_px": focal,
                    "gravity_samples": params.gravity_samples,
                },
                work,
                progress=ctx.progress,
                timeout=params.timeout_s,
            )
            z = np.load(resp["out_npz"])
            info["gravity"] = "geocalib"
            info["geocalib"] = {
                "focal_prior_px": focal,
                "geocalib_focal_px": resp.get("geocalib_focal_px"),
                "samples": len(z["up_frames"]),
            }
            return average_up(r_v_cv, z["up_cam"], z["up_frames"], z["up_weight"])
        except CacheOnlyMiss:  # [afv-core C, round 3] not a GeoCalib failure: the run is cache-only
            raise
        except Exception as e:  # level fallback, reported
            info["gravity"] = f"level (GeoCalib failed: {e})"
            ctx.progress(-1.0, f"GeoCalib gravity failed, assuming a level camera: {e}")
            return None


def _nanstat(x: np.ndarray, q: float) -> float | None:
    """Percentile of the finite values, None (JSON null) when there are none."""
    x = np.asarray(x, dtype=np.float64)
    return round(float(np.nanpercentile(x, q)), 3) if np.isfinite(x).any() else None


def _rotation_rate(r_v_cv: np.ndarray) -> float:
    """Median frame-to-frame rotation of a camera (degrees per second at 30 Hz)."""
    if len(r_v_cv) < 2:
        return 0.0
    rel = np.einsum("tba,tbc->tac", r_v_cv[:-1], r_v_cv[1:])
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return round(float(np.degrees(np.median(np.arccos(cos)))) * SHOT_FPS, 4)


def _default_focal(params: CameraParams, width: int, height: int) -> float:
    return float(params.focal_px) if params.focal_px else float(np.hypot(width, height))


@register("vipe")
class CameraVipe(_CameraStage):
    """ViPE once per shot (pose_only_long, one focal per shot), plus GeoCalib gravity."""

    env: ClassVar[str] = "vipe"
    mode: ClassVar[str] = "vipe"
    licences: ClassVar[tuple[LicenceRecord, ...]] = (*_VIPE_LICENCES, _GEOCALIB)
    weight_hash: ClassVar[str] = "vipe:ead2e008bc055310121736a49ff306bba50d10dc"

    def solve(self, ctx, params, norm, tracks, say):
        work = Path(ctx.work_dir)
        resp = _worker(
            "vipe",
            "vipe_worker.py",
            {
                "command": "vipe",
                "video": norm["out_mp4"],
                "out_npz": str(work / "vipe.npz"),
                "pipeline": params.pipeline,
                "gravity": params.gravity == "geocalib",
                "gravity_samples": params.gravity_samples,
            },
            work,
            progress=ctx.progress,
            timeout=params.timeout_s,
        )
        z = np.load(resp["out_npz"])
        c2w, intr = z["c2w"].astype(np.float64), z["intrinsics"].astype(np.float64)
        if c2w.shape[0] != int(norm["frame_count"]):
            raise StageError(f"ViPE returned {c2w.shape[0]} frames, expected {norm['frame_count']}")
        info: dict[str, Any] = {"mode": "vipe", "pipeline": params.pipeline, "vipe": resp}
        up_v = None
        if "up_cam" in z.files:
            up_v = average_up(c2w[:, :3, :3], z["up_cam"], z["up_frames"], z["up_weight"])
            info["_up"] = (z["up_cam"], z["up_frames"], z["up_weight"])
            info["gravity"] = "geocalib"
        elif params.gravity == "geocalib":
            info["gravity"] = f"level ({resp.get('gravity')})"
        else:
            info["gravity"] = "level"
        focal = intr[:, :2].mean(axis=1)
        info["vipe_focal_px"] = {
            "min": float(focal.min()),
            "median": float(np.median(focal)),
            "max": float(focal.max()),
            "fx_minus_fy_max": float(np.abs(intr[:, 0] - intr[:, 1]).max()),
        }
        return (
            c2w[:, :3, :3],
            c2w[:, :3, 3],
            focal,
            float(np.median(intr[:, 2])),
            float(np.median(intr[:, 3])),
            up_v,
            info,
        )


@register("static")
class CameraStatic(_CameraStage):
    """Locked-off camera: identity rotations, one focal (GEM-X default unless given)."""

    env: ClassVar[str] = "cpu"
    mode: ClassVar[str] = "static"
    licences: ClassVar[tuple[LicenceRecord, ...]] = (_GEOCALIB,)
    weight_hash: ClassVar[str] = ""

    def solve(self, ctx, params, norm, tracks, say):
        n, w, h = int(norm["frame_count"]), int(norm["width"]), int(norm["height"])
        focal = np.full(n, _default_focal(params, w, h))
        info = {"mode": "static", "gravity": "level" if params.gravity == "level" else None}
        return np.tile(np.eye(3), (n, 1, 1)), None, focal, w / 2.0, h / 2.0, None, info


@register("background")
class CameraBackground(_CameraStage):
    """GEM-X's background rotation estimate (SIFT homographies), every tracked person masked."""

    env: ClassVar[str] = "gemx"
    mode: ClassVar[str] = "background"
    licences: ClassVar[tuple[LicenceRecord, ...]] = (_GEOCALIB,)
    weight_hash: ClassVar[str] = ""

    def solve(self, ctx, params, norm, tracks, say):
        if tracks is None or tracks.num_frames == 0:
            raise StageError("background camera needs detect_track boxes to mask the people")
        work = Path(ctx.work_dir)
        n, w, h = int(norm["frame_count"]), int(norm["width"]), int(norm["height"])
        np.savez_compressed(work / "boxes.npz", boxes=np.asarray(tracks.boxes, np.float32))
        ids = list(tracks.fighters.values()) or [int(tracks.track_ids[0])]
        slot = int(np.flatnonzero(tracks.track_ids == ids[0])[0])
        f = _default_focal(params, w, h)
        resp = _worker(
            "gemx",
            "gemx_worker.py",
            {
                "command": "background_camera",
                "video": norm["out_mp4"],
                "boxes_npz": str(work / "boxes.npz"),
                "slot": slot,
                "intrinsics": [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]],
                "out_npz": str(work / "background.npz"),
            },
            work,
            gpu=False,
            progress=ctx.progress,
            timeout=params.timeout_s,
        )
        z = np.load(resp["out_npz"])
        r_c2v = np.swapaxes(z["R_w2c"].astype(np.float64), -1, -2)
        info = {
            "mode": "background",
            "diagnostics": resp.get("diagnostics", {}),
            "gravity": "level" if params.gravity == "level" else None,
        }
        return r_c2v, None, np.full(n, f), w / 2.0, h / 2.0, None, info
