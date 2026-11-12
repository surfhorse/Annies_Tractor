"""Two fighters in one consistent world (stage ``pair``).

P0 found that GEM-X's per-fighter depth is unreliable with telephoto broadcast lenses (jumps up to
1.9 m per frame, bodies overlapping) while the image projection is good. So the pair stage only
moves each fighter **along the camera ray through its root**: a rigid translation along that ray
keeps the root's image position exactly; every other joint moves on the image by about
f * o * dz / d^2 pixels (o its offset from the root across the ray, dz the move along the ray, d the
distance; [Z, round 3] e.g. 12 px for o = 0.5 m, dz = 1 m at 20 m with f = 10000 px), so 2D
fidelity holds while the moves stay a small fraction of the distance.

Depth and floor (one linear least-squares problem, robustified)
---------------------------------------------------------------
Unknowns: the distance ``z_t`` along the ray for every frame of every fighter, plus one floor
height ``y0`` shared by both. Terms:

* data: ``z_t`` stays near GEM-X's distance (sigma = ``depth_sigma_rel`` x distance), Huber-robust
  so single-frame depth jumps are treated as outliers rather than averaged in;
* GEM-X's own root rollout (``ActorMotion.root_prior``, when present): its velocity along the
  ray is a robust prior on ``z[t+1] - z[t]``. The rollout drifts against the camera but is smooth,
  while the camera-consistent root jitters in depth, so the two are fused: absolute depth from the
  data term, frame-to-frame depth change from the rollout;
* smoothness: second differences of ``z`` (sigma ``acc_sigma`` m/frame^2, about the acceleration
  a fighter can produce) and a weak first-difference term;
* ground: when a foot is (likely) on the ground, the lowest sole point must sit at ``y0``. Because
  the ray is not horizontal, height along it is linear in ``z``, so the floor constrains depth too.
  With a rollout prior the floor sigma is divided by ``|ray . up|``: sole jitter would otherwise
  turn into depth wobble of jitter / sin(pitch) on telephoto shots, and the prior already gives
  the frame-to-frame depth; without one the floor is the main depth cue and keeps full weight.
  Weighted by the input contact probability when available, Huber-robust for jumps.
* local-mean floor (``floor_mean_sigma``): the contact-weighted mean over +-``floor_mean_window``
  frames of each fighter's sole height must sit at ``y0``, with the full (not pitch-divided) sigma.
  Averaging removes the sole jitter the pitch division guards against, so this fixes each fighter's
  *slowly varying* depth bias. On the real fixtures one fighter's soles sat 8-24 cm above the
  other's (pose3d's ``fighter_floor_spread_m``): GEM-X's per-fighter depth (scale) is biased by ~3-4 %
  on telephoto shots (0.6-1 m at 17-27 m), and the pitch-divided per-frame rows were far weaker than
  the depth data term, so pair kept the bias and the floating fighter lost most plants downstream.
* weak 2D evidence ([Z, round 3]; ``ActorMotion.weak``, pose3d's mask for a truncated/occluded body
  or an interpolated box): GEM-X's depth there is a
  guess, so the data row keeps only an image-error budget (sigma = ``weak_image_budget_px`` x d^2 /
  (f x ``weak_offset_m``), the move that shifts a joint ``weak_offset_m`` off the ray by that many
  pixels), the floor rows are dropped and the frame is left out of the local means: depth there
  follows the rollout and the smoothness rows from the neighbouring frames, with no edge pops.
* local-mean depth (``depth_mean_sigma_rel``, NOT robust): the same window's mean depth stays near
  GEM-X's (sigma = that fraction of the distance). The Huber data rows reject single-frame jumps,
  but they also let a consistent pull win; this keeps the floor correction a plausible scale error.
  At close range (ufc: a 10 cm gap at 4 m would need a 25 % depth change) or on nearly horizontal
  rays (Who Am I) the gap is mostly left, since no plausible depth explains it.

Moving both fighters and the floor together along their rays is (nearly) unobservable, so the
common depth offset comes from the data term (GEM-X's mean depth) unless ``camera_ground_sigma``
says to trust the camera's ground plane height, which then fixes absolute depth as well.

Solved with sparse normal equations and IRLS (a few iterations, milliseconds for a shot).

Finally the shot is translated vertically so the shared floor is y = 0 (project convention,
``floor_to_zero``). That one constant offset is the only change that is not along a camera ray; it
is added to ``BodyMotion.world_shift``; camera rays here, and any later reprojection, use
``cam_pos + world_shift``.

Push-apart
----------
Bodies are approximated by capsules (torso, head, legs; arms optional, since gloves legitimately
touch the opponent). For every frame the extra depth separation that removes all capsule
overlap is found (the smallest separation from which the bodies stay apart), moving the nearer
fighter towards the camera and the farther one away along their own rays (again
projection-preserving). The depth order is fixed per push episode (the overlapping frames plus
everything their smoothed push can reach): the order that needs less total push on the overlapping
frames, ties broken by GEM-X's median order there. (A per-frame order from the smoothed depth
difference flipped when two people passed each other on screen at the same GEM-X depth, and both
roots jumped by the whole push: 72 cm in one frame on ref walk.) Ramp frames whose GEM-X order is the
other one get the separation that keeps them clear, up to what the overlapping frames need (beyond
that GEM-X has the fighters pass through each other: a long crossing is left to overlap briefly
rather than pushed by max_push). The per-frame requirement is smoothed with a max-then-box filter
that never undercuts it, so the push never pops. Without a camera, fighters
are pushed apart horizontally along the line between their pelvises and only floor heights are
aligned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict
from scipy.linalg import solveh_banded
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.sparse.linalg import spsolve

from afv.core.skeletons import get_skeleton
from afv.core.types import ActorMotion, BodyMotion, CameraTrack, Skeleton
from afv.motion.contacts import (
    Floor,
    actor_skeleton,
    cover_smooth,
    fit_floor,
    model_probabilities,
    sanitize_motion,
    segments,
    sole_points,
)
from afv.motion.fk import forward_kinematics
from afv.motion.ik import BodyRoles, Capsule, get_roles
from afv.qc.camera import camera_centres, camera_index


class PairParams(BaseModel):
    """Parameters of the pair stage."""

    model_config = ConfigDict(extra="forbid")

    depth_sigma_rel: float = 0.03
    depth_sigma_min: float = 0.05
    acc_sigma: float = 0.005  # m / frame^2
    vel_sigma: float = 0.5  # m / frame (weak)
    prior_vel_sigma: float = 0.01  # m / frame: depth velocity follows GEM-X's own root rollout
    floor_sigma: float = 0.03  # m, divided by |ray . up| (see module docstring)
    floor_min_ray_pitch: float = 0.02
    # m; local mean (+-``floor_mean_window`` frames, contact-weighted) of each fighter's sole height at
    # the shared floor, NOT divided by the ray pitch (see module docstring); None = off
    floor_mean_sigma: float | None = 0.02
    floor_mean_window: int = 15
    # The local mean depth (same window) stays near GEM-X's with this relative sigma, NOT robust: the
    # Huber data rows reject single-frame jumps, but a consistent offset must stay a plausible depth
    # (scale) error, or the floor rows would pull a close-range fighter far off its image size.
    depth_mean_sigma_rel: float | None = 0.03
    floor_weight_no_contact: float = 0.5
    camera_ground_sigma: float | None = None  # m; trust CameraTrack.ground_offset as the floor height
    huber_k: float = 1.5
    irls_iterations: int = 8
    push_apart: bool = True
    include_arms: bool = False
    push_tolerance: float = 0.0
    max_push: float = 1.5
    push_smooth_frames: int = 3
    floor_to_zero: bool = True  # translate the shot vertically so the shared floor is y = 0
    # [Z, round 3] weak 2D evidence (ActorMotion.weak): the data row's image-error budget (module doc)
    weak_image_budget_px: float = 20.0
    weak_offset_m: float = 0.5


@dataclass
class PairReport:
    floor: Floor
    depth_change: dict[str, np.ndarray]  # [F] metres moved along the ray
    push: np.ndarray  # [F] extra separation applied
    penetration_before: np.ndarray  # [F] max capsule overlap (m) after the depth solve
    penetration_after: np.ndarray
    floor_shift: float = 0.0  # metres added to every root's y after the solve (floor -> y = 0)


# ---------------------------------------------------------------------------
# Capsules
# ---------------------------------------------------------------------------


def segment_distance(p0, p1, q0, q1) -> np.ndarray:
    """Closest distance between segments p0p1 and q0q1 (broadcasting over leading axes)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a = np.maximum(np.sum(d1 * d1, -1), 1e-12)
    e = np.maximum(np.sum(d2 * d2, -1), 1e-12)
    f = np.sum(d2 * r, -1)
    c = np.sum(d1 * r, -1)
    b = np.sum(d1 * d2, -1)
    den = a * e - b * b
    s = np.where(den > 1e-12, np.clip((b * f - c * e) / np.maximum(den, 1e-12), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    diff = (p0 + d1 * s[..., None]) - (q0 + d2 * t[..., None])
    return np.linalg.norm(diff, axis=-1)


def _caps(sk: Skeleton, caps: tuple[Capsule, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([sk.index(c.a) for c in caps]),
        np.array([sk.index(c.b) for c in caps]),
        np.array([c.radius for c in caps]),
    )


def capsule_penetration(
    pos_a: np.ndarray, pos_b: np.ndarray, sk_a: Skeleton, sk_b: Skeleton, caps: tuple[Capsule, ...]
) -> np.ndarray:
    """Largest capsule overlap [F] (metres, <= 0 means separated) between two posed bodies."""
    ia0, ia1, ra = _caps(sk_a, caps)
    ib0, ib1, rb = _caps(sk_b, caps)
    d = segment_distance(pos_a[:, ia0, None], pos_a[:, ia1, None], pos_b[:, None, ib0], pos_b[:, None, ib1])
    return np.max(ra[:, None] + rb[None, :] - d, axis=(1, 2))


# ---------------------------------------------------------------------------
# Depth + floor solve
# ---------------------------------------------------------------------------


def _floor_weights(roles: BodyRoles, bm: BodyMotion, actor: ActorMotion, p: PairParams) -> np.ndarray:
    probs = [
        v for v in model_probabilities(roles, bm.contact_channels, actor.contact).values() if v is not None
    ]
    if probs:
        return np.clip(np.max(np.stack(probs), axis=0), 0.0, 1.0)
    return np.full(bm.num_frames, p.floor_weight_no_contact)


def _huber(res: np.ndarray, k: float) -> np.ndarray:
    a = np.abs(res)
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-12))


def _solve_bordered_banded(lhs: sp.spmatrix, rvec: np.ndarray, max_band: int = 256) -> np.ndarray:
    """Solve the SPD normal equations whose last unknown (the shared floor ``y0``) couples to all others
    while the rest is banded (frames couple only within the smoothing / window rows): banded Cholesky
    on the band plus a Schur complement for ``y0``, linear in the shot length (a general sparse LU
    was superlinear: 4.9 s at 3000 frames). Falls back to ``spsolve`` for a wide band."""
    n = lhs.shape[0] - 1
    k_mat = sp.csr_matrix(lhs[:n, :n])
    coo = k_mat.tocoo()
    band = int(np.abs(coo.row - coo.col).max()) if coo.nnz else 0
    if n <= 0 or band > max_band:
        return spsolve(sp.csc_matrix(lhs), rvec)
    ab = np.zeros((band + 1, n))
    for k in range(band + 1):
        ab[k, : n - k] = k_mat.diagonal(-k)  # lower form: ab[k, j] = K[j + k, j]
    border = np.asarray(lhs[:n, n].todense()).ravel()
    sol = solveh_banded(ab, np.column_stack([rvec[:n], border]), lower=True)
    u, v = sol[:, 0], sol[:, 1]
    y = (rvec[n] - border @ u) / (float(lhs[n, n]) - border @ v)
    return np.r_[u - v * y, y]


def solve_depth_and_floor(
    bm: BodyMotion, cam: CameraTrack, roles: BodyRoles, p: PairParams
) -> tuple[dict[str, np.ndarray], float, dict[str, np.ndarray]]:
    """New distances along the ray per fighter, the shared floor height, and the ray directions."""
    up = np.asarray(roles.up, dtype=np.float64)
    names = list(bm.actors)
    n_frames = bm.num_frames
    centres = camera_centres(cam, bm.frames) + np.asarray(bm.world_shift, dtype=np.float64)
    focal = np.asarray(cam.focal_px, dtype=np.float64)  # [Z, round 3] per frame (zoom)
    focal = (
        focal[camera_index(cam, bm.frames)]
        if focal.size > 1
        else np.full(n_frames, float(focal.reshape(-1)[0]))
    )
    blocks = []
    for r in names:
        a = bm.actors[r]
        sk = actor_skeleton(bm, a)
        pos, _ = forward_kinematics(sk, a.rot, a.root)
        root = np.asarray(a.root, dtype=np.float64)
        ray = root - centres
        z0 = np.linalg.norm(ray, axis=-1)
        u = ray / np.maximum(z0, 1e-9)[:, None]
        _, low = sole_points(sk, roles, pos, up)
        s = low @ up - root @ up  # lowest sole height relative to the root
        dprior = None
        if a.root_prior is not None:
            prior = np.asarray(a.root_prior, dtype=np.float64)
            if prior.shape == root.shape and np.all(np.isfinite(prior)):
                dprior = np.sum((prior[1:] - prior[:-1]) * u[:-1], axis=-1)  # along-ray velocity
        # [Z, round 3] weak 2D evidence: pose3d's ActorMotion.weak (module doc)
        w_in = getattr(a, "weak", None)
        weak = np.zeros(n_frames, bool) if w_in is None else np.asarray(w_in, dtype=bool).reshape(-1)
        if weak.shape != (n_frames,):
            weak = np.zeros(n_frames, bool)
        wf = np.where(weak, 0.0, _floor_weights(roles, bm, a, p))
        blocks.append({"z0": z0, "u": u, "s": s, "wf": wf, "dprior": dprior, "weak": weak})
    n_var = n_frames * len(names) + 1
    iy = n_var - 1
    # Rows are assembled once (vectorised, as COO triplets); IRLS only changes the robust weights.
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    sig: list[np.ndarray] = []
    robust: list[np.ndarray] = []
    n_rows = 0

    def add(r_loc: np.ndarray, c: np.ndarray, v: np.ndarray, b: np.ndarray, sigma, rob: bool) -> None:
        """Rows ``n_rows + r_loc`` (one entry per triplet), right-hand sides ``b`` [m]."""
        nonlocal n_rows
        m = len(b)
        rows.append(n_rows + np.asarray(r_loc, dtype=np.int64))
        cols.append(np.asarray(c, dtype=np.int64))
        vals.append(np.asarray(v, dtype=np.float64))
        rhs.append(np.asarray(b, dtype=np.float64))
        sig.append(np.broadcast_to(np.asarray(sigma, dtype=np.float64), (m,)).copy())
        robust.append(np.full(m, rob))
        n_rows += m

    def window(hw: int, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Triplets (row t, frame s, weight[s]) of the +-``hw`` windows with weight > 0."""
        offs = np.arange(-hw, hw + 1)
        t_idx = np.repeat(np.arange(n_frames), len(offs))
        s_idx = t_idx + np.tile(offs, n_frames)
        ok = (s_idx >= 0) & (s_idx < n_frames)
        t_idx, s_idx = t_idx[ok], s_idx[ok]
        w = weight[s_idx]
        keep = w > 0
        return t_idx[keep], s_idx[keep], w[keep]

    t_all = np.arange(n_frames)
    t_in = np.arange(1, n_frames - 1)
    t_pair = np.arange(n_frames - 1)
    for k, blk in enumerate(blocks):
        o = k * n_frames
        z0, u, s, wf, dprior = blk["z0"], blk["u"], blk["s"], blk["wf"], blk["dprior"]
        uy = u @ up
        sig_z = np.maximum(p.depth_sigma_min, p.depth_sigma_rel * z0)
        weak = blk["weak"]
        if weak.any():  # [Z, round 3] only an image-error budget on weak 2D evidence
            budget = p.weak_image_budget_px * z0**2 / (focal * max(p.weak_offset_m, 1e-3))
            sig_z = np.where(weak, np.maximum(sig_z, budget), sig_z)
        add(t_all, o + t_all, np.ones(n_frames), z0, sig_z, True)
        # Floor: (C + z u).up + s - y0 = 0  ->  (u.up) z - y0 = -(C.up + s). Sole jitter dh maps to depth
        # jitter dh / |u.up|, so the floor sigma is scaled by 1 / |u.up|: on telephoto shots the floor
        # fixes depth on average without driving frame-to-frame wobble.
        tf = np.flatnonzero(wf > 1e-3)
        if tf.size:
            pitch = np.maximum(np.abs(uy[tf]), p.floor_min_ray_pitch) if dprior is not None else 1.0
            r_loc = np.repeat(np.arange(tf.size), 2)
            c = np.stack([o + tf, np.full(tf.size, iy)], axis=1).reshape(-1)
            v = np.stack([uy[tf], -np.ones(tf.size)], axis=1).reshape(-1)
            add(r_loc, c, v, -(centres[tf] @ up + s[tf]), p.floor_sigma / pitch / np.sqrt(wf[tf]), True)
        hw = max(1, p.floor_mean_window)
        if p.floor_mean_sigma is not None and p.floor_mean_sigma > 0:
            h_c = centres @ up + s  # sole height = (u.up) z + C.up + s
            ti, si, w = window(hw, np.where(wf > 1e-3, wf, 0.0))
            tot = np.bincount(ti, w, minlength=n_frames)
            good = tot >= 3.0
            sel = good[ti]
            ti, si, w = ti[sel], si[sel], w[sel]
            rows_t = np.flatnonzero(good)
            loc = np.full(n_frames, -1)
            loc[rows_t] = np.arange(rows_t.size)
            r_loc = np.r_[loc[ti], np.arange(rows_t.size)]
            c = np.r_[o + si, np.full(rows_t.size, iy)]
            v = np.r_[w * uy[si] / tot[ti], -np.ones(rows_t.size)]
            b = -np.bincount(ti, w * h_c[si], minlength=n_frames)[rows_t] / tot[rows_t]
            add(r_loc, c, v, b, p.floor_mean_sigma, True)
        if p.depth_mean_sigma_rel is not None and p.depth_mean_sigma_rel > 0:
            ti, si, _ = window(hw, np.where(blk["weak"], 0.0, 1.0))  # [Z, round 3] weak frames left out
            cnt = np.bincount(ti, minlength=n_frames).astype(np.float64)
            rows_t = np.flatnonzero(cnt > 0)
            loc = np.full(n_frames, -1)
            loc[rows_t] = np.arange(rows_t.size)
            mean_z = np.bincount(ti, z0[si], minlength=n_frames)[rows_t] / cnt[rows_t]
            add(
                loc[ti],
                o + si,
                1.0 / cnt[ti],
                mean_z,
                np.maximum(p.depth_sigma_min, p.depth_mean_sigma_rel * mean_z),
                False,
            )
        two = np.repeat(np.arange(n_frames - 1), 2)
        c_vel = np.stack([o + t_pair, o + t_pair + 1], axis=1).reshape(-1)
        v_vel = np.tile([-1.0, 1.0], n_frames - 1)
        if dprior is not None:
            add(two, c_vel, v_vel, dprior, p.prior_vel_sigma, True)
        if t_in.size:
            add(
                np.repeat(np.arange(t_in.size), 3),
                np.stack([o + t_in - 1, o + t_in, o + t_in + 1], axis=1).reshape(-1),
                np.tile([1.0, -2.0, 1.0], t_in.size),
                np.zeros(t_in.size),
                p.acc_sigma,
                False,
            )
        add(two, c_vel, v_vel, np.zeros(n_frames - 1), p.vel_sigma, False)
    if p.camera_ground_sigma is not None:
        # Camera ground plane dot(n, x) + offset = 0 with n = up, in the camera's world; this
        # result's world is that world + world_shift  ->  y0 = -offset + shift_y.
        y_cam = -float(cam.ground_offset) + float(np.asarray(bm.world_shift) @ up)
        add(np.zeros(1), np.array([iy]), np.ones(1), np.array([y_cam]), p.camera_ground_sigma, False)
    a_mat = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_rows, n_var)
    )
    b = np.concatenate(rhs)
    sig_v = np.concatenate(sig)
    rob_v = np.concatenate(robust)
    w = 1.0 / sig_v**2
    # Weak prior on y0 keeps the system regular when no floor rows exist.
    y_prior = float(
        np.median(np.concatenate([centres @ up + blk["z0"] * (blk["u"] @ up) + blk["s"] for blk in blocks]))
    )
    reg = sp.csc_matrix(([1e-6], ([iy], [iy])), shape=(n_var, n_var))
    x = None
    for _ in range(max(1, p.irls_iterations)):
        lhs = (a_mat.T @ sp.diags(w) @ a_mat).tocsc() + reg
        rvec = a_mat.T @ (w * b)
        rvec[iy] += 1e-6 * y_prior
        x = _solve_bordered_banded(lhs, rvec)
        res = (a_mat @ x - b) / sig_v
        w = np.where(rob_v, _huber(res, p.huber_k), 1.0) / sig_v**2
    assert x is not None
    z = {r: x[k * n_frames : (k + 1) * n_frames] for k, r in enumerate(names)}
    rays = {r: blocks[k]["u"] for k, r in enumerate(names)}
    return z, float(x[iy]), rays


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _shift(actor: ActorMotion, delta: np.ndarray, world: bool = False) -> ActorMotion:
    """Translate a fighter. ``world`` = a translation of the whole world (also moves
    ``root_prior``); otherwise a correction of the fighter within the world (prior unchanged)."""
    prior = actor.root_prior
    if world and prior is not None:
        prior = (np.asarray(prior, dtype=np.float64) + delta).astype(np.float32)
    return ActorMotion(
        rot=actor.rot,
        root=(np.asarray(actor.root, dtype=np.float64) + delta).astype(np.float32),
        conf=actor.conf,
        contact=actor.contact,
        rest_offsets=actor.rest_offsets,
        root_prior=prior,
        weak=getattr(actor, "weak", None),  # [Z, round 3] carried along
    )


def pair_fighters(
    bm: BodyMotion,
    cam: CameraTrack | None = None,
    params: PairParams | None = None,
    roles: BodyRoles | None = None,
) -> tuple[BodyMotion, PairReport]:
    """Shared floor, smoothed depth along the camera rays, and push-apart (module docstring)."""
    p = params or PairParams()
    bm, _ = sanitize_motion(bm)
    roles = roles or get_roles(get_skeleton(bm.skeleton))
    up = np.asarray(roles.up, dtype=np.float64)
    names = list(bm.actors)
    n_frames = bm.num_frames
    actors = dict(bm.actors)
    depth_change: dict[str, np.ndarray] = {}
    rays: dict[str, np.ndarray] | None = None
    z: dict[str, np.ndarray] = {}
    if cam is not None and len(cam.cam_pos) > 0 and n_frames > 0:
        z, y0, rays = solve_depth_and_floor(bm, cam, roles, p)
        centres = camera_centres(cam, bm.frames) + np.asarray(bm.world_shift, dtype=np.float64)
        for r in names:
            z0 = np.linalg.norm(np.asarray(bm.actors[r].root, dtype=np.float64) - centres, axis=-1)
            depth_change[r] = z[r] - z0
            actors[r] = _shift(bm.actors[r], depth_change[r][:, None] * rays[r])
    else:
        # No camera: align each fighter's own floor to the common (mean) floor, vertically.
        heights = {}
        for r in names:
            a = bm.actors[r]
            sk = actor_skeleton(bm, a)
            _, low = sole_points(sk, roles, forward_kinematics(sk, a.rot, a.root)[0], up)
            heights[r] = fit_floor(low).height(np.zeros(3)) * -1.0
        y0 = float(np.mean(list(heights.values()))) if heights else 0.0
        for r in names:
            depth_change[r] = np.zeros(n_frames)
            actors[r] = _shift(bm.actors[r], np.broadcast_to((y0 - heights[r]) * up, (n_frames, 3)))
    out = BodyMotion(
        frames=bm.frames,
        timestamps=bm.timestamps,
        skeleton=bm.skeleton,
        fps=bm.fps,
        contact_channels=list(bm.contact_channels),
        actors=actors,
        world_shift=list(bm.world_shift),
    )
    pen0 = pen1 = push = np.zeros(n_frames)
    if p.push_apart and len(names) == 2 and n_frames > 0:
        out, pen0, pen1, push = push_apart(out, roles, p, rays, z)
    shift = 0.0
    if p.floor_to_zero:
        shift = -y0
        out.actors = {
            r: _shift(a, np.broadcast_to(shift * up, (n_frames, 3)), world=True)
            for r, a in out.actors.items()
        }
        out.world_shift = [float(v) for v in np.asarray(out.world_shift, dtype=np.float64) + shift * up]
        y0 = 0.0
    return out, PairReport(Floor.horizontal(y0), depth_change, push, pen0, pen1, shift)


def push_apart(
    bm: BodyMotion,
    roles: BodyRoles,
    p: PairParams,
    rays: dict[str, np.ndarray] | None,
    depth: dict[str, np.ndarray] | None,
) -> tuple[BodyMotion, np.ndarray, np.ndarray, np.ndarray]:
    """Separate interpenetrating bodies (module docstring). Returns motion, overlap before/after,
    and the applied extra separation per frame."""
    ra, rb = list(bm.actors)
    caps = tuple(c for c in roles.capsules if p.include_arms or c.group != "arm")
    if not caps:
        z = np.zeros(bm.num_frames)
        return bm, z, z, z
    a, b = bm.actors[ra], bm.actors[rb]
    sk_a, sk_b = actor_skeleton(bm, a), actor_skeleton(bm, b)
    pa = forward_kinematics(sk_a, a.rot, a.root)[0]
    pb = forward_kinematics(sk_b, b.rot, b.root)[0]
    up = np.asarray(roles.up, dtype=np.float64)
    n_frames = bm.num_frames
    if rays is not None and depth:
        diff = median_filter(depth[rb] - depth[ra], size=15, mode="nearest")
        base_a, base_b = -rays[ra], rays[rb]  # sgn +1: A is nearer, so A comes forward
    else:
        ip = sk_a.index(roles.pelvis)
        d = pb[:, ip] - pa[:, ip]
        d = d - (d @ up)[:, None] * up
        d = median_filter(d, size=(15, 1), mode="nearest")
        diff = np.ones(n_frames)
        d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        base_a, base_b = -d, d

    def overlap(sep: np.ndarray, sgn: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        """Largest capsule overlap with an extra separation ``sep`` in order ``sgn`` (frames ``idx``)."""
        t = np.arange(n_frames) if idx is None else idx
        h = 0.5 * (sep * sgn)[:, None, None]
        return capsule_penetration(pa[t] + h * base_a[t, None], pb[t] + h * base_b[t, None], sk_a, sk_b, caps)

    def clear_from(idx: np.ndarray, sgn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per frame of ``idx`` the smallest extra separation from which the bodies stay apart for
        every larger one up to ``max_push``, and whether such a separation exists. In the wrong depth
        order the overlap is not monotone (they pass through each other), so a plain bisection from 0
        can stop on a separation that a slightly larger smoothed push drives back into the other body."""
        grid = np.linspace(0.0, p.max_push, 41)
        bad = np.stack([overlap(np.full(len(idx), g), sgn, idx) > p.push_tolerance for g in grid])
        stays = np.flip(np.logical_and.accumulate(np.flip(~bad, axis=0), axis=0), axis=0)
        ok = stays.any(axis=0)
        j = np.where(ok, np.argmax(stays, axis=0), len(grid) - 1)
        lo = grid[np.maximum(j - 1, 0)]
        hi = grid[j]
        refine = ok & (j > 0)
        for _ in range(10):  # the last crossing lies in (grid[j-1], grid[j]]
            mid = 0.5 * (lo + hi)
            b = overlap(mid, sgn, idx) > p.push_tolerance
            lo = np.where(refine & b, mid, lo)
            hi = np.where(refine & ~b, mid, hi)
        return np.where(ok, np.where(j > 0, hi, 0.0), p.max_push), ok

    def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
        return maximum_filter1d(mask.astype(np.uint8), size=2 * radius + 1, mode="constant") > 0

    pen0 = overlap(np.zeros(n_frames), np.ones(n_frames))
    need = np.zeros(n_frames)
    sgn = np.ones(n_frames)
    hit = pen0 > p.push_tolerance
    reach = 4 * max(p.push_smooth_frames, 0)  # how far cover_smooth spreads a requirement
    two_orders = rays is not None and bool(depth)
    episodes = dilate(hit, 2 * reach)  # everything one episode's push can touch (see below)
    for s0, e0 in segments(episodes):
        # The depth order must not flip while the fighters are pushed apart: a flip moves both by the
        # whole push in one frame (ref walk: two people passing each other on screen at the same GEM-X
        # depth, a 72 cm root jump). So one order per episode: the one needing less total push on the
        # overlapping frames (a frame that cannot be cleared within max_push costs 10 x max_push),
        # ties broken by GEM-X's median order on them.
        h = s0 + np.flatnonzero(hit[s0:e0])
        orders = (1.0, -1.0) if two_orders else (1.0,)
        cand = {}
        for o in orders:
            nd, ok = clear_from(h, np.full(len(h), o))
            cand[o] = (nd, float(np.sum(np.where(ok, nd, 10.0 * p.max_push))))
        if two_orders and abs(cand[1.0][1] - cand[-1.0][1]) <= 1e-6:
            order = 1.0 if np.median(diff[h]) >= 0.0 else -1.0
        else:
            order = min(cand, key=lambda o: cand[o][1])
        need[h] = cand[order][0]
        # Ramp frames (no overlap at zero push, but within the smoothed push's reach): in the wrong
        # order there (GEM-X has the fighters on the other side) the push would drive one body into the
        # other. If GEM-X keeps the chosen order on all overlapping frames, such a frame gets the
        # separation that keeps it clear, up to what the overlapping frames need. If GEM-X has them
        # pass through each other while overlapping (a crossing), some frames must overlap whatever
        # the order; clearing the far side would push up to max_push and ramp back fast, so the ramp
        # is left as is (the bodies overlap briefly while the push eases out).
        ramp = s0 + np.flatnonzero((dilate(hit, reach) & ~hit)[s0:e0])
        crossing = two_orders and bool(np.any(np.where(diff[h] >= 0.0, 1.0, -1.0) != order))
        if ramp.size and not crossing:
            nd, ok = clear_from(ramp, np.full(len(ramp), order))
            need[ramp] = np.where(ok & (nd <= need[h].max() + 1e-9), nd, 0.0)
        sgn[s0:e0] = order
    # cover_smooth spreads ``need`` (hit frames and ramp frames, i.e. within ``reach`` of a hit) by up
    # to ``reach`` more, so the push stays inside its episode; the mask removes float residue there.
    push = np.where(episodes, np.minimum(cover_smooth(need, p.push_smooth_frames), p.max_push), 0.0)
    dir_a, dir_b = sgn[:, None] * base_a, sgn[:, None] * base_b
    half = 0.5 * push[:, None]
    actors = dict(bm.actors)
    actors[ra] = _shift(a, half * dir_a)
    actors[rb] = _shift(b, half * dir_b)
    out = BodyMotion(
        frames=bm.frames,
        timestamps=bm.timestamps,
        skeleton=bm.skeleton,
        fps=bm.fps,
        contact_channels=list(bm.contact_channels),
        actors=actors,
        world_shift=list(bm.world_shift),
    )
    return out, pen0, overlap(push, sgn), push
