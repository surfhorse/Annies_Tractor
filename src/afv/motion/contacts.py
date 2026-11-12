"""Foot contacts: detection, floor fit, skate removal and ground penetration (stage ``contacts``).

Design
------
GEM-X's world root is the weak part of its output (P0 and the real-data review: feet inside
GEM-X's own contacts still move 0.3-2 m/s after the pair stage), so the root, not only the legs,
must be corrected: pinning such feet with leg IK alone stretches the legs, forces big pelvis drops,
snaps knees and moves the body on screen. The stage therefore runs:

1. **Floor.** One floor for both fighters, fitted robustly (iteratively reweighted least squares
   with Tukey's biweight) to the lowest sole point per frame, refitted on candidate plants. The
   world is gravity aligned, so by default only its height is fitted (``floor_tilt`` fits a capped
   tilt).
2. **Candidate plants.** Per channel (heel = foot joint, toe = toe joint) a log-odds sum of
   GEM-X's contact probability, the sole height above the floor and a *weak* speed vote (the root
   is not trusted yet), hysteresis (0.6/0.4), gap filling and a minimum duration.
3. **Root correction.** One sparse robust least-squares solve per fighter for a smooth root offset
   ``d(t)`` that makes the candidate plants stand still: plant rows ask each planted joint (plus
   ``d``) to stay on one per-plant pin in the floor plane (foot height is left to the legs: on a
   tilted ray, depth would otherwise be bent to fix it); prior rows make ``d`` cheap *along the
   camera ray* (depth is what GEM-X gets wrong, and moving along the ray does not change the image)
   and limited across it: per frame up to about ``root_image_px`` pixels at the fighter's distance
   (enough to remove GEM-X's 2D jitter of the hips) but only ``root_image_mean_px`` for the local
   mean over +-``root_image_window`` frames, so the body never drifts on screen; along the ray
   ``root_depth_sigma`` keeps depth anchored where there are no plants; smoothness rows (second
   differences) keep ``d`` from adding jitter, and its rate along the ray is limited
   (``root_max_speed`` + ``root_speed_ratio`` x the input root speed) so the body never lurches in
   depth between two stances. Plant rows are Huber-robust, so a wrong GEM-X contact
   is outvoted rather than obeyed. Without a camera the across-ray term is isotropic. A fighter
   whose own input plants already stand still (this stage's output fed back in) is not corrected, so
   re-running the stage does not move it.
4. **Final plants.** Re-detected on the corrected motion with the full speed vote plus a hard
   veto (a foot moving faster than ``speed_veto`` cannot be planted whatever the model says), and
   a travel cap: a plant whose joint travels more than ``max_plant_travel`` is split at its fastest
   frame (pieces shorter than the minimum are dropped), and a toe plant spanning two heel plants of its
   foot (a pivot over the ball) is split between them (its pin comes from one heel plant's foot pin,
   :func:`split_pivot_toe_plants`). Steps 3-4 iterate ``root_iterations`` times.
   Painted contacts (``edits/contacts.json``) then override: "plant" spans forced on, "free" off.
5. **Rigid foot pins and leg IK.** Per plant one rigid foot transform: the heel pin is the median
   ankle position (never below the floor; a hover of up to ``max_plant_hover`` is kept, since
   pulling a straight-legged stance down would have to lower the hips on screen; a pin the leg
   cannot reach with ``pin_reach_drop`` of pelvis drop is raised further, never above the input's
   own ankle height nor ``max_pin_raise`` above the clearance, see :func:`_reachable_pin`), the
   foot orientation the mean planted foot orientation pitched so the toe joint sits at its clearance
   (up to ``max_plant_hover`` above, :func:`_level_foot`: a flat plant has the whole sole on the
   floor), and a toe pin that overlaps it is derived from them (so heel and toe can never disagree
   and twist the foot); a toe-only plant gets its own median pin and drags the ankle rigidly. On
   toe-only frames (push-off, toe-first landing) the ankle target is put on the sphere of foot length
   around the toe pin and the toe is held by the toe correction, so the foot pivots on the ball
   (:func:`_pivot_ankle`). Outside a plant the correction offset
   is continued with matching value and rate and eased out over ``blend_frames`` (smootherstep), so
   the foot's velocity is continuous; with ``knee_blend`` the blend frames' hip-ankle distance is
   then re-shaped so the knee angle is eased instead (:func:`_knee_space_blend`: near a straight leg
   the knee angle is extremely sensitive to that distance). Analytic two-bone IK
   (:mod:`afv.motion.ik`, pole preserved, knee kept on its side of the hinge) puts the ankle on its
   target; the foot blends to the pinned orientation; a toe correction is applied only where the
   ankle reached its target.
6. **Residual pelvis.** Only what the root correction could not absorb: the pelvis is lowered when a
   pinned ankle (on a fully planted frame) is beyond ``reach`` x leg length (a slightly bent knee
   avoids the IK singularity) and beyond the input leg's own extension, smoothed with a
   max-then-box filter that never undercuts the requirement and capped (``max_pelvis_drop``;
   lifting a sunk body as a whole, ``lift_pelvis``, is off by default: the legs bend instead and
   the hips stay on the image).
   Unplanted legs keep their pose relative to the pelvis (they are never sent to world targets they
   cannot reach); any foot below the floor is lifted onto it (exactly in plants, with a C1 smooth
   maximum for swinging feet). Plant frames whose pin stays more than ``max_pin_miss`` away are
   unplanted and solved again (up to three re-solves while a re-solve misses elsewhere; frames
   still missing after the last are unplanted without another solve; painted plants are kept;
   so are toe plants whose pin would float more than ``max_pin_raise`` above the floor, and frames
   whose foot's lowest sole point is more than ``max_plant_hover`` above the floor: a raised heel is a
   heel lift only while the ball is down, :func:`foot_sole_heights`), so are frames whose planted
   joint steps faster than ``max_plant_step`` to a planted neighbour, and the nearest plant-edge
   frame of a foot pop or knee snap (a foot or knee moving ``max_edge_pop`` / ``max_knee_snap_deg``
   more in one frame than the input's) is unplanted the same way, and frames
   at a plant's ends whose planted joint still moves faster than ``max_edge_slide`` (a leg that stops
   reaching its pin, within ``max_pin_miss``) are flagged unplanted, so a "planted" foot never slides
   or floats; the leg-IK correction is lightly smoothed in time (``ik_smooth``), each leg held where
   its own foot is planted or blending into/out of a plant, then re-solved exactly there, and a
   swinging foot the smoothing pushed into the floor is lifted back by a smoothed lift (no kink).

Later stages (smooth, retime) re-pin with :func:`hold_contacts`: leg IK onto the reference motion's
planted joints frame by frame (``pin_mode="ref"``), no root correction, a pelvis drop only where a
planted target is beyond the leg's reach (the IK's own reach for its knee hinge), and an exact floor
clamp, so re-pinning an unchanged motion is a no-op and those stages are idempotent.

**Stability to GPU-level input noise** ([M, round 5]). GEM-X's run-to-run noise (mostly a 1-2.5 cm
depth drift along the camera ray) moves the plant probabilities by ~0.01, so frames whose probability
sits that close to a threshold flip between repeat runs; with a hard decision that cannot be avoided.
The root-correction solve is now convex (one minimum, see :func:`solve_root_correction`), so such a
flip moves the root by about 1-1.5 cm instead of up to 8 cm. What still flips, measured on 8 real
repeat inputs of close_boxing: candidate plant onsets crossing ``on_threshold`` (A RightFoot frame
36: 0.604 vs 0.596) and a 5-frame plant at the clip end lost to the pin-miss rule, i.e. contact
frames A 99/105/107 and B 116/119; root steps of 1.1 cm (A) / 1.45 cm (B) at the flip and a smoothed
root path spread of 1.6 / 2.8 cm. ufc_cross A keeps a structural flip: a long candidate toe plant
(frames 121-168) is one plant or two at a 3-frame gap, ~3 cm of smoothed root path. Tried and not
kept (each reduced the flips but lost plants): onsets needing two frames at ``on_threshold``, a
higher onset threshold (0.7), and hysteresis on the 3-frame mean of the plant log-odds (contact frame
spread 0 on close_boxing, but A 99-107 -> 87 frames; who_am_i B 195 -> 187). ``max_plant_step``
2.5 cm/s (was 3) trades a little frame-count stability for planted steps: on ufc_cross A the real
repeats' contact frames are 705/715/705 (spread 10, 4 at 3 cm/s; root paths unchanged), while the
worst planted steps drop well under the 3 cm/s target (ufc_cross B 2.94 -> 0.03, A <= 2.4,
ref_walk/001 B 2.76 -> 2.29 cm/s).

**Floor at y = 0** (project convention): the shot is translated vertically so the fitted floor is
y = 0; the shift is added to ``BodyMotion.world_shift`` (reprojection uses ``cam_pos +
world_shift``) and to ``root_prior``, and reported in ``qc.json``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple

import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d
from scipy.sparse.linalg import spsolve

from afv.core.skeletons import get_skeleton
from afv.core.stage import StageError
from afv.core.types import ActorMotion, BodyMotion, CameraTrack, Skeleton
from afv.motion.fk import forward_kinematics
from afv.motion.ik import (
    BodyRoles,
    ContactRole,
    LegRoles,
    LegTarget,
    floor_lift,
    get_roles,
    knee_hinge_local,
    leg_max_reach,
    quat_axis_angle,
    solve_leg_ik,
)
from afv.motion.rotations import make_continuous, qconj, qmul, qrotate, slerp

HOVER_TOL = 1e-3  # m; float32 storage and IK rounding on top of ``ContactParams.max_plant_hover``
# Root-correction solve (see solve_root_correction): at most ROOT_SOLVE_ITERS iterations; it stops once
# the rate limit's active set is unchanged and the estimated distance to the converged solve,
# step x rho / (1 - rho), is below ROOT_SOLVE_TOL (m). The iteration converges linearly (~0.95 per step
# on ufc_cross), but can be fast for a few steps and slow after: rho is the largest of the last three
# step ratios, floored at ROOT_SOLVE_RHO_MIN (a ref_walk crop's ratios ~0.2 for 4 steps, then 0.8-0.94:
# unfloored it stopped 1.06 mm from converged). Measured: <= 0.11 mm on 79 crops (45-220 frames) of 5
# real shots, <= 0.043 mm on full real inputs and harness draws.
ROOT_SOLVE_ITERS = 150
ROOT_SOLVE_TOL = 1e-4
ROOT_SOLVE_RHO_MIN = 0.9


class ContactParams(BaseModel):
    """Parameters of contact detection and the foot fix (also the ``contacts`` stage params)."""

    model_config = ConfigDict(extra="forbid")

    on_threshold: float = 0.6
    off_threshold: float = 0.4
    min_plant_frames: int = 3
    max_gap_frames: int = 2
    height_threshold: float = 0.05  # m of sole height below which the height cue votes "contact"
    height_softness: float = 0.015
    speed_threshold: float = 0.35  # m/s horizontal speed below which the speed cue votes "contact"
    speed_softness: float = 0.10
    speed_veto: float = 0.6  # m/s; faster than this a foot is not planted, whatever the model says
    speed_veto_softness: float = 0.05  # m/s per logit of the veto
    max_plant_travel: float = 0.15  # m; longer plants are split at their fastest frame
    model_weight: float = 1.0
    height_weight: float = 1.0
    speed_weight: float = 1.0
    speed_weight_with_model: float = 0.5
    candidate_speed_weight: float = 0.15  # speed vote before the root is corrected
    # Root correction (step 3).
    root_correction: bool = True
    root_iterations: int = 2
    root_plant_sigma: float = 0.01  # m, horizontal stillness of a planted joint
    root_plant_vertical_sigma: float = 0.0  # m, planted joint at its clearance; 0 = legs handle height
    root_depth_sigma: float = 0.1  # m, offset along the camera ray (cheap: invisible in the image)
    root_image_px: float = 4.0  # px per frame across the ray (removes 2D jitter of GEM-X's hips)
    root_image_mean_px: float = 2.0  # px for the local mean (+-7 frames) across the ray: no drift on screen
    root_image_window: int = 7
    root_free_sigma: float = 0.05  # m, across-ray sigma when there is no camera
    root_acc_sigma: float = 0.004  # m/frame^2, smoothness of the offset
    root_huber: float = 2.0  # Huber threshold of plant rows, in sigmas
    root_skip_speed: float = 0.02  # m/s: plants already this still need no root correction
    # Rate limit of the offset (m/s; with a camera its component along the ray, else all of it):
    # |d(t) - d(t-1)| * fps <= root_max_speed + root_speed_ratio x the input root's own speed.
    # Without it the depth offset swung up to 50 cm in 8 plant-free frames between two stances (hips
    # 11 cm/frame against the input's 3); plants the slower offset cannot hold still are left to the
    # final detection's speed veto and the pin-miss rules.
    root_max_speed: float = Field(0.4, ge=0.0)
    root_speed_ratio: float = Field(0.5, ge=0.0)
    # Foot fix (steps 5-6).
    blend_frames: int = 4
    reach: float = 0.96  # max hip-ankle distance / leg length for pinned legs (see module doc)
    max_pelvis_drop: float = 0.025
    max_pin_miss: float = 0.01  # m; plant frames whose pin stays further away are unplanted
    # m/s; frames at either end of a plant whose planted joint still moves faster than this (a leg
    # that stops reaching its pin, within ``max_pin_miss``) are flagged unplanted (the motion is kept)
    max_edge_slide: float = Field(0.05, ge=0.0)
    # m/s; a planted joint whose solved position steps faster than this between two planted frames (a
    # leg that reaches its pin on one frame and misses it, within ``max_pin_miss``, on the next) has the
    # worse of the two frames unplanted and solved again: 1 cm of allowed miss is a 30 cm/s jump.
    # [M, round 5] 2.5 cm/s (was 3): the converged root solve keeps more plant-edge frames, whose steps
    # then sat just under 3 cm/s (ufc_cross B frame 144 2.94, projects/ufc_cross B frame 249 2.84);
    # with a margin under the 3 cm/s target they are back at or below the round-4 level (B max 0.03,
    # ref_walk/001 B 2.29) for 1-2 plant frames per fighter
    max_plant_step: float = Field(0.025, ge=0.0)
    # m; a foot joint stepping this much more than the input's (and over twice it) next to a plant (a
    # pin far from where the input foot was at the plant's edge, or two pins a short gap apart) has
    # the nearest plant-edge frame unplanted and is solved again
    max_edge_pop: float = Field(0.03, ge=0.0)
    # deg; a knee whose angle changes this much more in one frame than the input's (a pinned leg
    # snapping straight and back next to its plant) is treated like an edge pop
    max_knee_snap_deg: float = Field(12.0, ge=0.0)
    knee_blend: bool = True  # ease the knee angle into/out of plants (see _knee_space_blend)
    ik_smooth: float = 1.0  # strength of the smoothing of the leg-IK correction (0 = off)
    # m a pin may stay above the floor (keeps the hips on the image), and the most a planted foot's
    # lowest sole point (heel or ball) may float: frames above it are unplanted; QC reports the same
    max_plant_hover: float = 0.02
    # Reach-aware heel pins: a pin the leg cannot reach (with this much pelvis drop) is raised, at most
    # to the input's own ankle height and to ``max_pin_raise`` above the clearance, until it is reached
    # on this quantile of the plant's frames. Frames it still cannot reach are unplanted (pin miss).
    pin_reach: bool = True
    pin_reach_drop: float = Field(0.0125, ge=0.0, le=0.1)
    pin_reach_quantile: float = Field(0.9, ge=0.0, le=1.0)
    # A planted contact must not float (a game's foot IK keyed on contact flags would snap it down).
    # 3 cm = the noise of GEM-X's planted ankle height (within-fighter spread of planted sole heights
    # 1.3-2.5 cm MAD on the real fixtures) plus the 1-2 cm a boxer's heel lifts while the ball of the
    # foot still bears weight; a heel joint higher than that above its clearance is not on the floor.
    # A raised heel counts as lifted only while its ball is down (``max_plant_hover``, see above).
    max_pin_raise: float = Field(0.03, ge=0.0, le=0.1)
    lift_pelvis: bool = False  # lift a sunk body as a whole (default: legs bend instead)
    max_pelvis_lift: float = 0.10
    # Radius r of the drop's never-undercut smoothing: a drop is held >= 4r+1 frames around every
    # frame needing it. r = 3 (13 frames, 0.43 s at 30 fps) is about a walking step, so on a gait the
    # pelvis never came back up between steps and the residual became a constant lowering (ref walk:
    # mean 2.2 cm from a raw requirement of 0.8 cm); r = 2 (9 frames) follows the gait. It also sets
    # the median filter size (2r+1) of the sink measure used by ``lift_pelvis``.
    pelvis_smooth_frames: int = Field(2, ge=0)
    fix_penetration: bool = True
    floor_soft_width: float = 0.01  # m; smooth floor clamp for unplanted feet (no velocity kink)
    floor_tilt: bool = False
    max_tilt_deg: float = 8.0
    floor_iterations: int = 2
    floor_to_zero: bool = True  # translate the shot vertically so the fitted floor is y = 0

    # Accepted and ignored (deprecated): briefly in the round 2 checkpoint (f451709), off by default,
    # then removed (vertical stillness rows in the root correction cost 2D fidelity). Old configs and
    # stored stage params that name it still validate.
    _DEPRECATED: ClassVar[tuple[str, ...]] = ("root_plant_still_sigma",)

    @model_validator(mode="before")
    @classmethod
    def _drop_deprecated(cls, data: Any) -> Any:
        if isinstance(data, dict) and any(k in data for k in cls._DEPRECATED):
            data = {k: v for k, v in data.items() if k not in cls._DEPRECATED}
        return data


# ---------------------------------------------------------------------------
# Floor
# ---------------------------------------------------------------------------


@dataclass
class Floor:
    """Plane ``dot(normal, x) + offset = 0``; ``height(x)`` is the signed distance above it."""

    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    offset: float = 0.0

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=np.float64)
        self.normal = n / np.linalg.norm(n)
        self.offset = float(self.offset)

    @classmethod
    def horizontal(cls, y: float = 0.0) -> Floor:
        return cls(np.array([0.0, 1.0, 0.0]), -float(y))

    def height(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.normal + self.offset

    def project(self, x: np.ndarray, h: float | np.ndarray = 0.0) -> np.ndarray:
        """Move points along the normal so they sit at height ``h`` above the floor."""
        x = np.asarray(x, dtype=np.float64)
        return x + (np.asarray(h) - self.height(x))[..., None] * self.normal

    def to_dict(self) -> dict[str, Any]:
        return {"normal": [float(v) for v in self.normal], "offset": self.offset}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Floor:
        return cls(np.asarray(d["normal"], dtype=np.float64), float(d["offset"]))


def _tukey_irls(design: np.ndarray, y: np.ndarray, w0: np.ndarray, iters: int = 15) -> np.ndarray:
    w = w0.copy()
    beta = np.linalg.lstsq(design * w[:, None] ** 0.5, y * w**0.5, rcond=None)[0]
    for _ in range(iters):
        r = y - design @ beta
        scale = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 0.005)
        u = r / (4.685 * scale)
        w = w0 * np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
        if w.sum() < 1e-9:
            break
        beta = np.linalg.lstsq(design * w[:, None] ** 0.5, y * w**0.5, rcond=None)[0]
    return beta


def fit_floor(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    tilt: bool = False,
    max_tilt_deg: float = 8.0,
    min_spread: float = 0.3,
) -> Floor:
    """Robust floor plane through sole points [N,3] (Y-up world). See module docstring."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    ok = np.all(np.isfinite(p), axis=1) & (w > 0)
    p, w = p[ok], w[ok]
    if len(p) == 0:
        return Floor.horizontal(0.0)
    y = p[:, 1]
    if tilt and len(p) >= 10:
        xz = p[:, [0, 2]]
        cov = np.cov((xz - np.average(xz, axis=0, weights=w)).T, aweights=w)
        if np.sqrt(max(np.linalg.eigvalsh(cov)[0], 0.0)) >= min_spread:
            a, b, c = _tukey_irls(np.column_stack([p[:, 0], p[:, 2], np.ones(len(p))]), y, w)
            if np.degrees(np.arctan(np.hypot(a, b))) <= max_tilt_deg:
                n = np.array([-a, 1.0, -b])
                norm = np.linalg.norm(n)
                return Floor(n / norm, -c / norm)
    (c,) = _tukey_irls(np.ones((len(p), 1)), y, w)
    return Floor.horizontal(float(c))


# ---------------------------------------------------------------------------
# Input hygiene
# ---------------------------------------------------------------------------


def sanitize_motion(bm: BodyMotion) -> tuple[BodyMotion, dict[str, int]]:
    """Fill non-finite frames of each fighter (root linear, rotations slerp between the nearest
    finite frames, contacts 0) so one bad frame cannot spread NaN through the solves.

    Returns the motion (a copy only when something was filled) and the filled-frame count per
    fighter. Raises ``StageError`` when a fighter has no finite frame at all.
    """
    filled: dict[str, int] = {}
    actors = dict(bm.actors)
    for r, a in bm.actors.items():
        rot = np.asarray(a.rot, dtype=np.float64)
        root = np.asarray(a.root, dtype=np.float64)
        good = np.all(np.isfinite(root), axis=1) & np.all(np.isfinite(rot), axis=(1, 2))
        good &= np.all(np.linalg.norm(rot, axis=-1) > 1e-6, axis=1) if rot.size else good
        n_bad = int((~good).sum())
        filled[r] = n_bad
        if n_bad == 0:
            continue
        if not good.any():
            raise StageError(f"fighter {r}: no finite frame in the motion")
        idx = np.flatnonzero(good)
        t = np.arange(len(good))
        new_root = np.stack([np.interp(t, idx, root[idx, k]) for k in range(3)], axis=1)
        new_rot = rot.copy()
        bad = np.flatnonzero(~good)
        lo = idx[np.clip(np.searchsorted(idx, bad) - 1, 0, len(idx) - 1)]
        hi = idx[np.clip(np.searchsorted(idx, bad), 0, len(idx) - 1)]
        w = np.where(hi == lo, 0.0, (bad - lo) / np.maximum(hi - lo, 1))
        new_rot[bad] = slerp(rot[lo], rot[hi], np.broadcast_to(w[:, None], (len(bad), rot.shape[1])))
        contact = None if a.contact is None else np.where(np.isfinite(a.contact), a.contact, 0.0)
        conf = None if a.conf is None else np.where(np.isfinite(a.conf), a.conf, 0.0)
        if conf is not None:
            conf[bad] = 0.0
        prior = a.root_prior
        if prior is not None:
            prior = np.asarray(prior, dtype=np.float64)
            pg = np.all(np.isfinite(prior), axis=1)
            if pg.any() and not pg.all():
                pi = np.flatnonzero(pg)
                prior = np.stack([np.interp(t, pi, prior[pi, k]) for k in range(3)], axis=1)
            elif not pg.any():
                prior = None
        actors[r] = ActorMotion(
            rot=new_rot.astype(np.float32),
            root=new_root.astype(np.float32),
            conf=conf,
            contact=contact,
            rest_offsets=a.rest_offsets,
            root_prior=prior,
            weak=getattr(a, "weak", None),  # [Z, round 3] pose3d's weak 2D mask, kept for pair
        )
    if not any(filled.values()):
        return bm, filled
    return _with_actors(bm, actors), filled


def _with_actors(bm: BodyMotion, actors: dict[str, ActorMotion], **kw: Any) -> BodyMotion:
    return BodyMotion(
        frames=bm.frames,
        timestamps=bm.timestamps,
        skeleton=bm.skeleton,
        fps=bm.fps,
        contact_channels=kw.get("contact_channels", list(bm.contact_channels)),
        actors=actors,
        world_shift=kw.get("world_shift", list(bm.world_shift)),
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-3, 1.0 - 1e-3)
    return np.log(p / (1.0 - p))


def contact_probability(
    height: np.ndarray,
    speed: np.ndarray,
    model: np.ndarray | None,
    p: ContactParams,
    speed_weight: float | None = None,
    veto: bool = True,
) -> np.ndarray:
    """Fused contact probability from sole height [F], horizontal speed [F] and model prob [F].

    ``veto`` adds an unbounded penalty above ``speed_veto`` (one logit per ``speed_veto_softness``),
    so a foot moving at 1 m/s is not a contact even with a 0.99 model probability.
    """
    lh = np.clip((p.height_threshold - height) / p.height_softness, -8.0, 8.0)
    lv = np.clip((p.speed_threshold - speed) / p.speed_softness, -8.0, 8.0)
    if speed_weight is None:
        speed_weight = p.speed_weight if model is None else p.speed_weight_with_model
    logit = p.height_weight * lh + speed_weight * lv
    if model is not None:
        logit = logit + p.model_weight * _logit(model)
    if veto:
        logit = logit - np.maximum(0.0, speed - p.speed_veto) / p.speed_veto_softness
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0)))


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Runs of True as (start, end) with end exclusive."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


def hysteresis_mask(prob: np.ndarray, p: ContactParams) -> np.ndarray:
    """Hysteresis threshold, then fill gaps <= max_gap_frames, then drop plants < min_plant_frames."""
    on = np.zeros(len(prob), dtype=bool)
    state = False
    for i, v in enumerate(prob):
        state = v >= p.off_threshold if state else v >= p.on_threshold
        on[i] = state
    segs = segments(on)
    for (_, e0), (s1, _) in itertools.pairwise(segs):
        if s1 - e0 <= p.max_gap_frames:
            on[e0:s1] = True
    for s, e in segments(on):
        if e - s < p.min_plant_frames:
            on[s:e] = False
    return on


def one_sided_speeds(
    pos: np.ndarray, normal: np.ndarray, fps: float, span: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """In-plane speeds [F] from local linear fits over the ``span`` + 1 frames ending at each
    frame (backward) and starting at it (forward)."""
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    if n < 2:
        return np.zeros(n), np.zeros(n)
    span = max(1, min(span, n - 1))
    k = np.arange(span + 1, dtype=np.float64) - span / 2.0
    k /= np.sum(k * k)
    padded = np.concatenate([np.repeat(pos[:1], span, 0), pos, np.repeat(pos[-1:], span, 0)])
    # Fit over frames [i, i + span] of the padded array = original frames [i - span, i].
    fits = sum(k[i] * padded[i : i + n + span] for i in range(span + 1)) * fps
    out = []
    for v in (fits[:n], fits[span : span + n]):
        v = v - (v @ normal)[:, None] * normal
        out.append(np.linalg.norm(v, axis=-1))
    return out[0], out[1]


def horizontal_speed(pos: np.ndarray, normal: np.ndarray, fps: float, span: int = 3) -> np.ndarray:
    """Speed [F] of positions [F,3] within the floor plane.

    Slopes of local linear fits over ``span`` + 1 frames ending at the frame (backward) and starting
    at it (forward), combined by their geometric mean, then a 3-frame median. A fit averages
    per-frame jitter out (a finite difference would amplify it). A symmetric window would smear
    swing speed into the first and last stance frames (plants found late, lost early); the slower
    one-sided fit would do the opposite (plants held into the swing). The geometric mean is low only
    when both sides are slow-ish and rises as soon as one side moves fast.
    """
    back, fwd = one_sided_speeds(pos, normal, fps, span)
    if len(back) < 2:
        return back
    return median_filter(np.sqrt(back * fwd), size=3, mode="nearest")


def cap_plant_travel(
    mask: np.ndarray, pos: np.ndarray, normal: np.ndarray, fps: float, p: ContactParams
) -> np.ndarray:
    """Split plants whose joint travels more than ``max_plant_travel`` (in the floor plane) at
    their fastest frame, recursively; drop pieces shorter than ``min_plant_frames``."""
    out = np.asarray(mask, dtype=bool).copy()
    speed = np.zeros(len(pos))
    if len(pos) > 1:
        d = np.diff(pos, axis=0)
        d = d - (d @ normal)[:, None] * normal
        speed[1:] = np.linalg.norm(d, axis=-1)
    stack = segments(out)
    while stack:
        s, e = stack.pop()
        q = pos[s:e] - (pos[s:e] @ normal)[:, None] * normal
        if e - s < p.min_plant_frames:
            out[s:e] = False
            continue
        if np.linalg.norm(q - np.median(q, axis=0), axis=-1).max() * 2.0 <= p.max_plant_travel:
            continue
        cut = s + 1 + int(np.argmax(speed[s + 1 : e]))  # speed[cut] = step cut-1 -> cut
        out[cut] = False if e - s > 2 else out[cut]
        stack.extend([(s, cut), (cut + 1, e)] if cut + 1 < e else [(s, cut)])
    return out


# ---------------------------------------------------------------------------
# Painted contacts
# ---------------------------------------------------------------------------


def painted_spans(
    edits: Any, role: str, roles: BodyRoles | list[str], frames: np.ndarray
) -> dict[str, list[tuple[int, int, bool]]]:
    """Parse ``edits/contacts.json`` for one fighter into {channel: [(i0, i1, plant), ...]}.

    Accepted forms: ``{"spans": [span, ...]}``, ``{"A": [span, ...], "B": [...]}`` or a list of
    spans. A span is ``{"role": "A", "start": f0, "end": f1, "state": "plant"|"free"}`` plus
    either ``"channels": [...]`` or ``"foot": "left"|"right"`` (all channels of that leg, resolved
    through the roles, so it works for any skeleton). ``start``/``end`` are shot-timeline frame
    numbers, end exclusive. A span that resolves to no channel raises ``ValueError``.
    """
    if not edits:
        return {}
    channels = [c.channel for c in roles.contacts] if isinstance(roles, BodyRoles) else list(roles)
    if isinstance(edits, dict) and "spans" in edits:
        spans = list(edits["spans"])
    elif isinstance(edits, dict):
        spans = [{**s, "role": r} for r, ss in edits.items() if isinstance(ss, list) for s in ss]
    else:
        spans = list(edits)
    out: dict[str, list[tuple[int, int, bool]]] = {}
    frames = np.asarray(frames)
    for s in spans:
        if s.get("role", role) != role:
            continue
        if "channels" in s:
            chans = list(s["channels"])
        elif isinstance(roles, BodyRoles):
            chans = roles.side_channels(str(s.get("foot", "")))
        else:
            raise ValueError("painted span by 'foot' needs BodyRoles to resolve the side")
        if not chans:
            raise ValueError(f"painted span {s} resolves to no contact channel")
        state = str(s.get("state", "plant")).lower()
        if state not in ("plant", "free"):
            raise ValueError(f"painted contact state must be 'plant' or 'free', got {state!r}")
        i0 = int(np.searchsorted(frames, int(s["start"]), side="left"))
        i1 = int(np.searchsorted(frames, int(s["end"]), side="left"))
        for c in chans:
            if c not in channels:
                raise ValueError(f"painted contact channel {c!r} not in {channels}")
            out.setdefault(c, []).append((i0, i1, state == "plant"))
    return out


# ---------------------------------------------------------------------------
# Root correction
# ---------------------------------------------------------------------------


def _perp_basis(n: np.ndarray) -> np.ndarray:
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(n, e1)])


def camera_geometry(
    bm: BodyMotion, actor: ActorMotion, cam: CameraTrack | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-frame unit ray from the camera (shifted into this world) to the root [F,3] and metres
    per pixel at the root's distance [F]; None without a usable camera."""
    if cam is None or len(cam.cam_pos) == 0:
        return None
    from afv.qc.camera import camera_index

    idx = camera_index(cam, bm.frames)
    centre = np.asarray(cam.cam_pos, dtype=np.float64)[idx] + np.asarray(bm.world_shift, dtype=np.float64)
    ray = np.asarray(actor.root, dtype=np.float64) - centre
    dist = np.linalg.norm(ray, axis=-1)
    f = np.asarray(cam.focal_px, dtype=np.float64)
    f = f[idx] if f.size > 1 else np.full(len(idx), float(f.reshape(-1)[0]) if f.size else 1000.0)
    return ray / np.maximum(dist, 1e-9)[:, None], dist / np.maximum(f, 1.0)


def solve_root_correction(
    sk: Skeleton,
    roles: BodyRoles,
    pos: np.ndarray,
    masks: dict[str, np.ndarray],
    floor: Floor,
    params: ContactParams,
    geometry: tuple[np.ndarray, np.ndarray] | None = None,
    fps: float = 30.0,
    base: np.ndarray | None = None,
    fixed: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Smooth root offset [F,3] that makes the plants in ``masks`` stand still (module doc, step 3),
    changing by at most ``root_max_speed`` + ``root_speed_ratio`` x the input root speed (m/s).
    ``base``: the offset earlier iterations already applied to ``pos`` [F,3]; the limit holds for the
    total (``base`` + the result), against the root speed of ``pos`` without it. ``fixed`` = (mask
    [F], values [F,3]): frames whose result is held at ``values`` (range re-runs: the previous
    solution outside the editable zone, so the solve there is a boundary condition)."""
    n_frames = pos.shape[0]
    plants = [
        (c, s, e) for c in roles.contacts for s, e in segments(masks.get(c.channel, np.zeros(n_frames, bool)))
    ]
    if n_frames < 3 or not plants:
        out = np.zeros((n_frames, 3))
        if fixed is not None:
            out[fixed[0]] = np.asarray(fixed[1], dtype=np.float64)[fixed[0]]
        return out
    clear = roles.clearances(sk)
    nrm = floor.normal
    basis = _perp_basis(nrm)  # two in-floor directions
    n_d = 3 * n_frames
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    sig: list[np.ndarray] = []
    rob: list[np.ndarray] = []
    nrow = 0

    def add(r_idx, c_idx, v, b, s, robust):
        nonlocal nrow
        m = len(b)
        rows.append(np.repeat(nrow + np.arange(m), r_idx))
        cols.append(c_idx.reshape(-1))
        vals.append(v.reshape(-1))
        rhs.append(b)
        sig.append(np.broadcast_to(s, (m,)).astype(np.float64))
        rob.append(np.full(m, robust))
        nrow += m

    pin_var = n_d
    for c, s, e in plants:
        t = np.arange(s, e)
        p = pos[t, sk.index(c.joint)]
        for k in range(2):  # horizontal: e_k . (p + d_t) - pin_k = 0
            ek = basis[k]
            ci = np.stack([3 * t, 3 * t + 1, 3 * t + 2, np.full(len(t), pin_var + k)], axis=1)
            v = np.tile(np.r_[ek, -1.0], (len(t), 1))
            add(4, ci, v, -(p @ ek), params.root_plant_sigma, True)
        if params.root_plant_vertical_sigma > 0:
            # vertical: n . (p + d_t) + offset - clearance = 0 (off by default: on a tilted ray
            # depth is cheap and would be bent to fix foot height, which is the legs' job)
            ci = np.stack([3 * t, 3 * t + 1, 3 * t + 2], axis=1)
            add(
                3,
                ci,
                np.tile(nrm, (len(t), 1)),
                clear[c.channel] - floor.height(p),
                params.root_plant_vertical_sigma,
                True,
            )
        pin_var += 2
    t = np.arange(n_frames)
    ci = np.stack([3 * t, 3 * t + 1, 3 * t + 2], axis=1)
    if geometry is not None:
        u, mpp = geometry
        a = np.cross(u, nrm)
        a_n = np.linalg.norm(a, axis=-1, keepdims=True)
        a = np.where(a_n > 1e-6, a / np.maximum(a_n, 1e-9), basis[0])
        b = np.cross(u, a)
        add(3, ci, u, np.zeros(n_frames), params.root_depth_sigma, False)
        img_sig = np.maximum(params.root_image_px * mpp, 1e-3)
        add(3, ci, a, np.zeros(n_frames), img_sig, False)
        add(3, ci, b, np.zeros(n_frames), img_sig, False)
        # The local mean of the across-ray offset is held much tighter: per-frame 2D jitter of the
        # hips may be removed, a shift of the body on screen may not.
        hw = max(1, params.root_image_window)
        lo = np.clip(t - hw, 0, n_frames - 1)
        hi = np.clip(t + hw, 0, n_frames - 1)
        win = [np.arange(l0, h0 + 1) for l0, h0 in zip(lo, hi, strict=True)]
        mean_sig = np.maximum(params.root_image_mean_px * mpp, 1e-3)
        for vec in (a, b):
            r_idx = np.array([3 * len(w) for w in win])
            c_idx = np.concatenate([np.stack([3 * w, 3 * w + 1, 3 * w + 2], axis=1).reshape(-1) for w in win])
            v = np.concatenate([(vec[w] / len(w)).reshape(-1) for w in win])
            m = len(win)
            rows.append(np.repeat(nrow + np.arange(m), r_idx))
            cols.append(c_idx)
            vals.append(v)
            rhs.append(np.zeros(m))
            sig.append(mean_sig)
            rob.append(np.zeros(m, dtype=bool))
            nrow += m
    else:
        for k in range(3):
            ek = np.eye(3)[k]
            s_k = params.root_free_sigma if abs(ek @ nrm) < 0.9 else 0.01
            add(3, ci, np.tile(ek, (n_frames, 1)), np.zeros(n_frames), s_k, False)
    if fixed is not None and np.any(fixed[0]):
        tf = np.flatnonzero(fixed[0])
        vf = np.asarray(fixed[1], dtype=np.float64)[tf]
        for k in range(3):  # stiff rows: d_k(t) = value (sigma 1 um)
            add(1, (3 * tf + k)[:, None], np.ones((len(tf), 1)), vf[:, k], 1e-6, False)
    for k in range(3):  # smoothness of each component
        tt = np.arange(1, n_frames - 1)
        ci2 = np.stack([3 * (tt - 1) + k, 3 * tt + k, 3 * (tt + 1) + k], axis=1)
        add(3, ci2, np.tile([1.0, -2.0, 1.0], (len(tt), 1)), np.zeros(len(tt)), params.root_acc_sigma, False)
    # Rate-limit rows dir . (d(t) - d(t-1)) = target: on (with the target projected onto the allowed
    # speed) exactly where the current solve exceeds the limit, off elsewhere (see below). With a camera
    # only the offset along the ray is limited (that is where it swung; across the ray the image rows
    # already hold it to a few pixels, and limiting it too kept the hips off their image position).
    tt = np.arange(1, n_frames)
    if geometry is not None:
        dirs = [np.asarray(geometry[0], dtype=np.float64)[tt]]
    else:
        dirs = [np.tile(np.eye(3)[k], (len(tt), 1)) for k in range(3)]
    v_first = nrow
    for dvec in dirs:
        ci_v = np.concatenate(
            [np.stack([3 * (tt - 1) + k for k in range(3)], 1), np.stack([3 * tt + k for k in range(3)], 1)],
            1,
        )
        add(6, ci_v, np.concatenate([-dvec, dvec], axis=1), np.zeros(len(tt)), 1.0, False)
    v_rows = np.arange(v_first, nrow).reshape(len(dirs), len(tt))
    base = np.zeros((n_frames, 3)) if base is None else np.asarray(base, dtype=np.float64)
    d_base = np.diff(base, axis=0)
    root_step = np.linalg.norm(np.diff(pos[:, 0] - base, axis=0), axis=-1)  # input root speed, m/frame
    v_lim = (params.root_max_speed + params.root_speed_ratio * root_step * fps) / max(fps, 1e-9)
    n_var = pin_var
    a_mat = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nrow, n_var)
    )
    b_vec = np.concatenate(rhs)
    s_vec = np.concatenate(sig)
    r_vec = np.concatenate(rob)
    # The limit is a one-sided penalty: a stiff row (v_sig) pulls the rate back to the bound on exactly
    # the frames where the current solve exceeds it, and is off elsewhere. With the Huber plant rows
    # the objective is convex, so these iterations (IRLS + the penalty's active set) settle on its one
    # minimum, which moves continuously with the input. [M, round 5] Rows used to be switched on with
    # the current rate as their target and kept on: the result depended on the iteration path, and one
    # candidate plant frame more or less (a plant probability crossing its threshold on GPU-level input
    # noise) moved the offset by 8 cm (close_boxing A; 1.1 cm without the limit, 1.2 cm now).
    v_on = np.zeros(len(tt), dtype=bool)
    # m/frame: the penalty's stiffness (floored: both limit parameters may be 0, so may the limit)
    v_sig = max(0.02 * float(v_lim.min()), 1e-4) if len(tt) else 1.0
    s_vec[v_rows.reshape(-1)] = np.inf
    w = 1.0 / s_vec**2
    x = np.zeros(n_var)
    limited = params.root_max_speed > 0 or params.root_speed_ratio > 0
    d_prev: np.ndarray | None = None
    steps: list[float] = []
    # Only the robust (plant) rows' and the rate rows' weights or targets change between iterations: the
    # normal equations of the other rows are built once.
    var = r_vec.copy()
    var[v_rows.reshape(-1)] = True
    a_c, a_v = a_mat[~var], a_mat[var]
    w_c = w[~var]
    lhs_c = (a_c.T @ (sp.diags(w_c) @ a_c)).tocsc() + sp.identity(n_var, format="csc") * 1e-9
    rhs_c = a_c.T @ (w_c * b_vec[~var])
    for _ in range(ROOT_SOLVE_ITERS if limited else 5):
        w_v = w[var]
        lhs = lhs_c + (a_v.T @ (sp.diags(w_v) @ a_v)).tocsc()
        x = spsolve(lhs, rhs_c + a_v.T @ (w_v * b_vec[var]))
        res = np.abs(a_mat @ x - b_vec) / s_vec
        hub = np.where(res <= params.root_huber, 1.0, params.root_huber / np.maximum(res, 1e-12))
        w = np.where(r_vec, hub, 1.0) / s_vec**2
        if not limited or len(tt) == 0:
            continue
        dv = np.diff(x[:n_d].reshape(n_frames, 3), axis=0) + d_base  # total offset rate
        comp = np.stack([np.sum(dv * dvec, axis=-1) for dvec in dirs])  # [K, F-1]
        sp_d = np.linalg.norm(comp, axis=0)
        on = sp_d > v_lim
        scale = np.where(on, v_lim / np.maximum(sp_d, 1e-12), 1.0)  # target: the bound, this direction
        for k, dvec in enumerate(dirs):
            b_vec[v_rows[k]] = comp[k] * scale - np.sum(d_base * dvec, axis=-1)
            s_vec[v_rows[k]] = np.where(on, v_sig, np.inf)
        w[v_rows.reshape(-1)] = 1.0 / s_vec[v_rows.reshape(-1)] ** 2
        d_now = x[:n_d]
        same_set = np.array_equal(on, v_on)
        v_on, d_prev, step = on, d_now, (None if d_prev is None else float(np.abs(d_now - d_prev).max()))
        if step is None:
            continue
        steps.append(step if same_set else np.inf)
        if len(steps) >= 4 and np.isfinite(steps[-4:]).all():
            # remaining distance to the fixed point ~ step x rho / (1 - rho) for a linear contraction
            # rho; rho from the last three step ratios (the largest, to be safe)
            rho = max(steps[-1] / max(steps[-2], 1e-15), steps[-2] / max(steps[-3], 1e-15),
                      steps[-3] / max(steps[-4], 1e-15))  # fmt: skip
            rho = max(rho, ROOT_SOLVE_RHO_MIN)
            if rho < 1.0 and steps[-1] * rho / (1.0 - rho) < ROOT_SOLVE_TOL:
                break
    return x[:n_d].reshape(n_frames, 3)


# ---------------------------------------------------------------------------
# Targets and the foot fix
# ---------------------------------------------------------------------------


def channel_offsets(
    cur: np.ndarray,
    ref: np.ndarray | None,
    mask: np.ndarray,
    floor: Floor,
    clearance: float,
    blend: int,
    pins: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Per-frame correction offsets [F,3] (target - cur) and weights [F] for one channel.

    Pins are given (one per segment of ``mask``) or taken from ``ref`` (median over each plant, on
    the floor at ``clearance``); inside a plant the target is the pin. Outside it the offset
    ``o = target - cur`` is continued so that it matches both value and rate at the plant edge
    (inside a plant ``o`` changes at ``-v_cur``): ``o(edge + k) = w_k (o_edge - k v_edge)`` with a
    smootherstep ``w`` (1 at the plant, 0 ``blend`` + 1 frames away). The foot's velocity is
    therefore continuous (zero) across the plant boundary and the input's own motion shape is kept.
    """
    n_frames = len(mask)
    off = np.zeros((n_frames, 3))
    acc = np.zeros((n_frames, 3))
    wacc = np.zeros(n_frames)
    inside = np.zeros(n_frames, dtype=bool)
    out_pins = []
    for i, (s, e) in enumerate(segments(mask)):
        if pins is not None:
            pin = np.asarray(pins[i], dtype=np.float64)
        else:
            assert ref is not None
            pin = floor.project(np.median(ref[s:e], axis=0), clearance)
        out_pins.append(pin)
        off[s:e] = pin - cur[s:e]  # a pin is one point [3] or a per-frame trajectory [e-s,3]
        inside[s:e] = True
        if blend > 0:
            k = np.arange(1, blend + 1)
            x = (blend + 1 - k) / (blend + 1)  # 1 next to the plant, -> 0 far away
            w = x**3 * (10.0 - 15.0 * x + 6.0 * x * x)  # smootherstep: C2 at both ends
            # Continue the offset with its own rate at the edge (for a fixed pin: -v_cur).
            span = min(2, e - s - 1)
            o_in = off[s:e]
            r_s = (o_in[span] - o_in[0]) / span if span > 0 else np.zeros(3)
            r_e = (o_in[-1] - o_in[-1 - span]) / span if span > 0 else np.zeros(3)
            before = s - k
            ok = before >= 0
            acc[before[ok]] += w[ok, None] * (o_in[0] - k[ok, None] * r_s)
            wacc[before[ok]] += w[ok]
            after = e - 1 + k
            ok = after < n_frames
            acc[after[ok]] += w[ok, None] * (o_in[-1] + k[ok, None] * r_e)
            wacc[after[ok]] += w[ok]
    out = ~inside
    off[out] = acc[out] / np.maximum(1.0, wacc[out])[:, None]
    weight = np.where(inside, 1.0, np.minimum(1.0, wacc))
    return off, weight, out_pins


def cover_smooth(x: np.ndarray, radius: int) -> np.ndarray:
    """Smooth a non-negative requirement without ever dropping below it (max filter radius 2r,
    then two box filters radius r)."""
    if radius <= 0 or len(x) == 0:
        return np.asarray(x, dtype=np.float64)
    m = maximum_filter1d(np.asarray(x, dtype=np.float64), size=4 * radius + 1, mode="nearest")
    m = uniform_filter1d(m, size=2 * radius + 1, mode="nearest")
    return uniform_filter1d(m, size=2 * radius + 1, mode="nearest")


def channel_positions(sk: Skeleton, roles: BodyRoles, pos: np.ndarray) -> dict[str, np.ndarray]:
    return {c.channel: pos[:, sk.index(c.joint)] for c in roles.contacts}


def _mean_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = np.where(np.sum(q * q[:1], axis=-1, keepdims=True) < 0, -q, q)
    m = q.mean(axis=0)
    return m / np.linalg.norm(m)


def _level_foot(
    pin: np.ndarray, rot: np.ndarray, v_loc: np.ndarray, floor: Floor, lo: float, hi: float
) -> np.ndarray:
    """Pitch a planted foot orientation ``rot`` (world, xyzw) about its lateral axis through the
    ankle pin so the toe joint (``pin + rot v_loc``) sits between heights ``lo`` (its clearance:
    on the floor) and ``hi`` above the floor; unchanged when it already does. The caller lowers
    ``hi`` by the heel pin's own lift (:func:`leg_pins`), so a raised heel gets its ball on the floor
    rather than a whole foot hovering at the top of the band.

    GEM-X's planted foot is rarely flat relative to the rest pose, so the toe derived from a heel
    pin with the mean planted foot orientation used to land up to 3.4 cm below the floor (the floor
    clamp then lifted it off its pin and the pin-miss rule unplanted the toe: most of the toe
    channels' lost plants on the real shots) or up to 3.4 cm above it (a floating toe contact)."""
    n = floor.normal
    u = qrotate(rot, v_loc)
    ell = float(np.linalg.norm(u))
    h_pin = float(floor.height(pin))
    h_toe = h_pin + float(u @ n)
    want = float(np.clip(h_toe, lo, hi))
    axis = np.cross(u, n)
    s = float(np.linalg.norm(axis))
    if abs(want - h_toe) < 1e-6 or ell < 1e-6 or s < 1e-9:
        return rot
    a0 = np.arcsin(np.clip(float(u @ n) / ell, -1.0, 1.0))
    a1 = np.arcsin(np.clip((want - h_pin) / ell, -1.0, 1.0))
    return qmul(quat_axis_angle(axis / s, np.asarray(a1 - a0)), rot)


def _reachable_pin(
    pin: np.ndarray,
    max_height: float,
    floor: Floor,
    pos: np.ndarray,
    sk: Skeleton,
    leg: LegRoles,
    reach: float,
    drop: float,
    quantile: float,
) -> tuple[np.ndarray, float, bool]:
    """Raise a heel pin along the floor normal until the leg reaches it (hip-ankle distance <=
    max(``reach`` x leg length, the input's own extension), with the pelvis lowered by ``drop``) on
    ``quantile`` of the plant frames ``pos``, but never above floor height ``max_height`` (the caller
    passes the lower of the input's median ankle height and ``max_pin_raise`` above the clearance).
    Returns the pin, the raise (m) and whether the cap was hit. Frames that would need a higher pin
    are not pinned in the air: the leg misses the capped pin there (by more than ``max_pin_miss``,
    after the full pelvis drop) and the pin-miss rule unplants them.

    A foot that hovers above the shared floor in the input (on real footage: a fighter whose soles
    disagree with the other's, or a slightly tilted world) cannot be pulled down onto it by a leg that
    is already (nearly) straight: the pinned foot then misses its pin and slides, or the plant is
    dropped. Pinning it a little higher, still at or below where the input had it, keeps the plant and
    the foot still; a pin far above the floor would be a floating contact, hence the cap."""
    n = floor.normal
    thigh = pos[:, sk.index(leg.thigh)]
    knee, ankle = pos[:, sk.index(leg.calf)], pos[:, sk.index(leg.foot)]
    length = np.linalg.norm(knee - thigh, axis=-1) + np.linalg.norm(ankle - knee, axis=-1)
    d_in = np.linalg.norm(thigh - ankle, axis=-1)
    r = np.maximum(reach * length, np.minimum(d_in, 0.995 * length))
    rel = thigh - drop * n - pin
    up = rel @ n
    horiz2 = np.sum(rel * rel, axis=-1) - up**2
    ok = horiz2 < r * r  # frames a higher pin cannot help (too far sideways) are left to the miss rule
    if not ok.any():
        return pin, 0.0, False
    # Raise needed per frame: the leg reaches the pin once up - raise <= sqrt(r^2 - horiz^2).
    need = np.maximum(up[ok] - np.sqrt(r[ok] ** 2 - horiz2[ok]), 0.0)
    want = float(np.quantile(need, quantile))
    room = max(0.0, max_height - float(floor.height(pin)))
    raise_by = min(want, room)
    return pin + raise_by * n, raise_by, want > room + 1e-9


@dataclass
class LegPins:
    """Rigid foot pins of one leg: per heel plant the ankle pin and (if the toe is planted too)
    the foot orientation; per toe plant its pin (derived from the heel pin when they overlap).
    ``raised``/``capped``: per heel plant, the reach raise (m) and whether its cap was hit;
    ``floating``: per toe plant, whether its pin (derived from a flat heel plant) sits more than
    ``max_pin_raise`` above the toe's clearance (a toe that high is not in contact)."""

    heel: list[np.ndarray]
    heel_rot: list[np.ndarray | None]
    toe: list[np.ndarray]
    raised: list[float] = field(default_factory=list)
    capped: list[bool] = field(default_factory=list)
    floating: list[bool] = field(default_factory=list)


def leg_pins(
    sk: Skeleton,
    roles: BodyRoles,
    leg_index: int,
    ref_pos: np.ndarray,
    ref_wrot: np.ndarray,
    masks: dict[str, np.ndarray],
    floor: Floor,
    max_hover: float = 0.0,
    per_frame: bool = False,
    reach: tuple[float, float, float, float] | None = None,
    max_toe_height: float | None = None,
) -> LegPins:
    """``reach`` = (reach fraction, pelvis drop allowance, quantile, max raise above the clearance):
    heel pins are raised (never above the input's own median ankle height) until the leg reaches
    them on that quantile of the plant's frames (:func:`_reachable_pin`). ``max_toe_height``: toe
    pins higher than this above the clearance are flagged ``floating``."""
    leg = roles.legs[leg_index]
    clear = roles.clearances(sk)
    n_frames = ref_pos.shape[0]
    heel = next((c for c in roles.contacts if c.leg == leg_index and c.kind == "heel"), None)
    toe = next((c for c in roles.contacts if c.leg == leg_index and c.kind == "toe"), None)
    ia = sk.index(leg.foot)
    hm = masks.get(heel.channel, np.zeros(n_frames, bool)) if heel else np.zeros(n_frames, bool)
    tm = masks.get(toe.channel, np.zeros(n_frames, bool)) if toe else np.zeros(n_frames, bool)
    it = sk.index(leg.toe) if leg.toe else None
    heel_pins, heel_rots, raised, capped = [], [], [], []
    seg_of = np.full(n_frames, -1)

    def on_floor(p: np.ndarray, c: float) -> np.ndarray:
        # Never below the floor; up to ``max_hover`` above it is kept as the input had it.
        return floor.project(p, float(np.clip(floor.height(p), c, c + max_hover)))

    if per_frame:
        # Re-pin mode (later stages): hold the reference motion's planted joints frame by frame, so
        # re-pinning a result onto itself changes nothing.
        heel_pins = [ref_pos[s:e, ia] for s, e in segments(hm)]
        heel_rots = [ref_wrot[s:e, ia] if tm[s:e].any() else None for s, e in segments(hm)]
        toe_pins = [ref_pos[s:e, it] for s, e in segments(tm)] if it is not None else []
        return LegPins(heel_pins, heel_rots, toe_pins)
    for i, (s, e) in enumerate(segments(hm)):
        med = np.median(ref_pos[s:e, ia], axis=0)
        pin = on_floor(med, clear[heel.channel])
        up_by, cap_hit = 0.0, False
        if reach is not None:
            fraction, drop, quantile, max_raise = reach
            top = min(float(floor.height(med)), clear[heel.channel] + max_raise)
            pin, up_by, cap_hit = _reachable_pin(
                pin, top, floor, ref_pos[s:e], sk, leg, fraction, drop, quantile
            )
            # Count only the float cap (``max_pin_raise``); the input's own height capping the
            # raise (a foot at or below the pin already) is the ordinary case for the pelvis drop.
            cap_hit = cap_hit and top >= clear[heel.channel] + max_raise - 1e-9
        heel_pins.append(pin)
        raised.append(up_by)
        capped.append(cap_hit)
        flat = np.arange(s, e)[tm[s:e]]
        rq = _mean_quat(ref_wrot[flat, ia]) if len(flat) else None
        if rq is not None and toe is not None and it is not None:
            # A flat plant puts the whole sole on the floor: the toe joint at its clearance (up to
            # ``max_hover`` above), not wherever the mean planted foot pitch takes it from the pin.
            v_loc = np.mean(qrotate(qconj(ref_wrot[flat, ia]), ref_pos[flat, it] - ref_pos[flat, ia]), axis=0)
            c_t = clear[toe.channel]
            lift = max(0.0, float(floor.height(pin)) - clear[heel.channel])  # heel sole above the floor
            rq = _level_foot(pin, rq, v_loc, floor, c_t, c_t + max(0.0, max_hover - lift))
        heel_rots.append(rq)
        seg_of[s:e] = i
    toe_pins, floating = [], []
    for s, e in segments(tm):
        assert toe is not None and it is not None
        hs = seg_of[s:e]
        hs = hs[hs >= 0]
        k = int(np.bincount(hs).argmax()) if len(hs) else -1
        if k >= 0 and heel_rots[k] is not None:
            flat = np.arange(s, e)[hm[s:e]]
            v_loc = np.mean(qrotate(qconj(ref_wrot[flat, ia]), ref_pos[flat, it] - ref_pos[flat, ia]), axis=0)
            toe_pins.append(heel_pins[k] + qrotate(heel_rots[k], v_loc))
        else:
            toe_pins.append(on_floor(np.median(ref_pos[s:e, it], axis=0), clear[toe.channel]))
        floating.append(
            max_toe_height is not None
            and float(floor.height(toe_pins[-1])) > clear[toe.channel] + max_toe_height
        )
    return LegPins(heel_pins, heel_rots, toe_pins, raised, capped, floating)


@dataclass
class FootFix:
    rot: np.ndarray
    root: np.ndarray
    pelvis_drop: np.ndarray  # [F] metres lowered along the floor normal (negative = lifted)
    weights: dict[str, np.ndarray]
    pins: dict[str, list[np.ndarray]]
    miss: dict[str, np.ndarray] = field(default_factory=dict)  # [F] distance from pin in plants
    raised_plants: int = 0  # heel plants whose pin was raised to be reachable
    capped_plants: int = 0  # ... of which needed more than ``max_pin_raise`` (frames then unplanted)
    floating: dict[str, np.ndarray] = field(default_factory=dict)  # [F] plant frames whose pin floats
    # [F] planted frames whose joint steps faster than ``max_plant_step`` to a planted neighbour (the
    # worse-missing frame of each such pair)
    stepping: dict[str, np.ndarray] = field(default_factory=dict)


class _LegPlan(NamedTuple):
    """Per-leg intermediate state of :func:`apply_foot_targets`."""

    leg: LegRoles
    heel: ContactRole | None
    toe: ContactRole | None
    pins: LegPins
    ankle_off: np.ndarray  # [F,3] ankle correction offset
    w_h: np.ndarray  # [F] heel plant weight (1 in plants, eased out around them)
    w_t0: np.ndarray  # [F] toe plant weight
    w_any: np.ndarray  # [F] max of the two
    planted: np.ndarray | None = None  # [F] the leg's plant mask (heel or toe planted)
    hinge: np.ndarray | None = None  # knee hinge (thigh frame) the IK must use (re-pin mode)


LURCH_M = 0.06  # m: 3-step displacement this much longer than the input's ...
LURCH_RATIO = 1.6  # ... and this many times it is a lurch


def foot_lurches(p_out: np.ndarray, p_in: np.ndarray) -> np.ndarray:
    """Frames [F] whose 3-step displacement (frame t-2 to t+1, clipped at the clip ends) is more than
    ``LURCH_M`` longer than the input's and ``LURCH_RATIO`` times it: a foot catching up with (or
    leaving) a pin, or a body lurch. The input's is the largest of the three 3-step windows around
    the frame. Filling a one-frame stall of the input (16, 4, 18 cm -> 13, 12, 13 cm) keeps the 3-step
    displacement, and filling a two-frame stall in a 10 m/s kick (Who Am I) keeps the windowed one, so
    neither counts (a single-frame step comparison counted both); a multi-frame catch-up still does."""
    n = len(p_out)
    t = np.arange(n)
    lo, hi = np.maximum(t - 2, 0), np.minimum(t + 1, n - 1)
    d_out = np.linalg.norm(p_out[hi] - p_out[lo], axis=-1)
    d_in = np.linalg.norm(p_in[hi] - p_in[lo], axis=-1)
    d_in = np.maximum.reduce([d_in, np.r_[d_in[1:], 0.0], np.r_[0.0, d_in[:-1]]])
    out = (d_out - d_in > LURCH_M) & (d_out > LURCH_RATIO * d_in)
    out[0] = False
    return out


def _knee_angles(sk: Skeleton, leg: LegRoles, pos: np.ndarray) -> np.ndarray:
    """Unsigned knee bend [F] (rad, 0 = straight) from joint positions."""
    h, k, a = (pos[:, sk.index(j)] for j in (leg.thigh, leg.calf, leg.foot))
    t, c = k - h, a - k
    cos = np.sum(t * c, -1) / np.maximum(np.linalg.norm(t, axis=-1) * np.linalg.norm(c, axis=-1), 1e-12)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _pivot_ankle(
    sk: Skeleton,
    leg: LegRoles,
    pos: np.ndarray,
    ankle_off: np.ndarray,
    masks: dict[str, np.ndarray],
    heel: ContactRole,
    toe: ContactRole,
    pins: LegPins,
    floor: Floor,
    params: ContactParams,
    ref_ankle: np.ndarray | None = None,
) -> np.ndarray:
    """Ankle offsets [F,3] with the ankle target of toe-only plant frames (toe planted, heel not)
    moved onto the sphere of foot length around the toe pin; outside those frames the correction is
    continued and eased out like a plant offset (:func:`channel_offsets`), never re-aimed at another
    plant's pin.

    The foot then pivots about its pinned ball (push-off, toe-first landing) and the toe correction
    can put the toe exactly on its pin. Before, the ankle followed the heel plant's eased-out offset
    (or the toe offset as a rigid translation), so the ankle-toe distance did not fit the pin and
    the toe missed it by 1.4-5 cm on almost every such frame (ref walk A: 44 of 46), which unplanted
    them. Next to a flat plant the projection is the identity (the derived toe pin is exactly one
    foot length from the heel pin). In re-pin mode (``ref_ankle`` given: the reference motion's
    ankle positions) those frames take the reference's own ankle, projected the same way (retime's
    position-interpolated reference shortens the foot by a chord error of ~2 mm), with no easing, so
    re-pinning an unchanged motion stays a no-op."""
    n_frames = len(pos)
    tm = masks.get(toe.channel, np.zeros(n_frames, bool))
    only = tm & ~masks.get(heel.channel, np.zeros(n_frames, bool))
    if not only.any():
        return ankle_off
    ia, it = sk.index(leg.foot), sk.index(leg.toe)
    # Toe pin per toe-only frame: its own plant's pin (a point, or a trajectory in re-pin mode).
    tp = np.zeros((n_frames, 3))
    for (s, e), pin in zip(segments(tm), pins.toe, strict=True):
        tp[s:e] = pin
    ell = np.linalg.norm(pos[:, it] - pos[:, ia], axis=-1)
    a = pos[:, ia] + ankle_off if ref_ankle is None else np.asarray(ref_ankle, dtype=np.float64)
    r = a - tp
    on_sphere = tp + ell[:, None] * r / np.maximum(np.linalg.norm(r, axis=-1), 1e-9)[:, None]
    if ref_ankle is not None:
        return np.where(only[:, None], on_sphere - pos[:, ia], ankle_off)
    corr = on_sphere - a  # only meaningful on toe-only frames
    segs = segments(only)
    off, _, _ = channel_offsets(
        np.zeros((n_frames, 3)), None, only, floor, 0.0, params.blend_frames, [corr[s:e] for s, e in segs]
    )
    # Heel-planted frames keep their heel pin: the eased continuation must not leak into them (next to
    # a flat plant the correction is ~0 anyway, see above).
    off[masks.get(heel.channel, np.zeros(n_frames, bool))] = 0.0
    return ankle_off + off


def apply_foot_targets(
    sk: Skeleton,
    roles: BodyRoles,
    rot: np.ndarray,
    root: np.ndarray,
    masks: dict[str, np.ndarray],
    floor: Floor,
    params: ContactParams,
    ref: tuple[np.ndarray, np.ndarray] | None = None,
    allow_pelvis: bool = True,
    soft_floor: bool = True,
    pin_mode: str = "median",
    ref_fk: tuple[np.ndarray, np.ndarray] | None = None,
    fps: float = 30.0,
) -> FootFix:
    """Pin every plant in ``masks`` with rigid foot pins and keep feet out of the floor.

    ``ref`` = (rot, root) of the motion the pins are taken from (default: this one); later stages
    pass their (already pinned) input so contacts are held exactly. ``allow_pelvis`` permits the
    residual pelvis lift/drop (contacts stage only); with it off only the legs move.
    ``soft_floor`` uses the C1 floor clamp for swinging feet (else the exact, idempotent clamp).
    ``pin_mode`` "median" makes one rigid pin per plant (contacts stage); "ref" holds the
    reference's planted joints frame by frame (later stages: re-pinning is then a no-op on an
    unchanged motion). ``ref_fk`` = precomputed (positions [F,J,3], world rotations [F,J,4]) of
    the reference, instead of ``ref``.
    """
    rot = np.asarray(rot, dtype=np.float64)
    root = np.asarray(root, dtype=np.float64)
    n_frames = rot.shape[0]
    clear = roles.clearances(sk)
    nrm = floor.normal
    width = params.floor_soft_width if soft_floor else 0.0
    pos, wrot = forward_kinematics(sk, rot, root)
    fwd = np.asarray(roles.forward)
    lift = np.zeros(n_frames)
    if allow_pelvis and params.lift_pelvis and params.fix_penetration:
        sole = np.stack([floor.height(pos[:, sk.index(c.joint)]) - clear[c.channel] for c in roles.contacts])
        sink = median_filter(
            np.maximum(0.0, -sole.min(axis=0)), size=2 * params.pelvis_smooth_frames + 1, mode="nearest"
        )
        lift = np.minimum(cover_smooth(sink, params.pelvis_smooth_frames), params.max_pelvis_lift)
        if lift.any():
            root = root + lift[:, None] * nrm
            pos, _ = forward_kinematics(sk, rot, root)
    if ref_fk is not None:
        ref_pos, ref_wrot = ref_fk
    elif ref is None:
        ref_pos, ref_wrot = forward_kinematics(sk, rot, root)
    else:
        ref_pos, ref_wrot = forward_kinematics(sk, ref[0], ref[1])
    cur = channel_positions(sk, roles, pos)
    weights: dict[str, np.ndarray] = {}
    all_pins: dict[str, list[np.ndarray]] = {}
    plans: list[_LegPlan] = []
    drop = np.zeros(n_frames)
    for li, leg in enumerate(roles.legs):
        heel = next((c for c in roles.contacts if c.leg == li and c.kind == "heel"), None)
        toe = next((c for c in roles.contacts if c.leg == li and c.kind == "toe"), None)
        pins = leg_pins(
            sk,
            roles,
            li,
            ref_pos,
            ref_wrot,
            masks,
            floor,
            params.max_plant_hover,
            per_frame=pin_mode == "ref",
            reach=(
                params.reach,
                params.pin_reach_drop if allow_pelvis else 0.0,
                params.pin_reach_quantile,
                params.max_pin_raise,
            )
            if params.pin_reach
            else None,
            max_toe_height=None if pin_mode == "ref" else params.max_pin_raise,
        )
        ia = sk.index(leg.foot)
        zeros = np.zeros(n_frames)
        off_h = np.zeros((n_frames, 3))
        w_h = zeros
        if heel is not None:
            hm = masks.get(heel.channel, np.zeros(n_frames, bool))
            off_h, w_h, all_pins[heel.channel] = channel_offsets(
                cur[heel.channel], None, hm, floor, clear[heel.channel], params.blend_frames, pins.heel
            )
            weights[heel.channel] = w_h
        off_t0 = np.zeros((n_frames, 3))
        w_t0 = zeros
        if toe is not None:
            tm = masks.get(toe.channel, np.zeros(n_frames, bool))
            off_t0, w_t0, all_pins[toe.channel] = channel_offsets(
                cur[toe.channel], None, tm, floor, clear[toe.channel], params.blend_frames, pins.toe
            )
            weights[toe.channel] = w_t0
        w_any = np.maximum(w_h, w_t0)
        ankle_off = off_h + (1.0 - w_h)[:, None] * off_t0  # toe-only plants drag the ankle rigidly
        if heel is not None and toe is not None and leg.toe is not None:
            ref_ankle = ref_pos[:, ia] if pin_mode == "ref" else None
            ankle_off = _pivot_ankle(
                sk, leg, pos, ankle_off, masks, heel, toe, pins, floor, params, ref_ankle
            )
        # Pelvis drop needed to reach the (world-fixed) pinned targets.
        hip = pos[:, sk.index(leg.thigh)]
        target_pre = pos[:, ia] + ankle_off
        length = np.linalg.norm(pos[:, sk.index(leg.calf)] - hip, axis=-1) + np.linalg.norm(
            pos[:, ia] - pos[:, sk.index(leg.calf)], axis=-1
        )
        rel = hip - target_pre
        b = rel @ nrm
        # Allowed extension: ``reach`` x leg length, or the input's own extension if it was already
        # straighter (a straight-legged stance is not a reason to lower the pelvis).
        d_in = np.linalg.norm(hip - pos[:, ia], axis=-1)
        r2 = np.maximum(params.reach * length, np.minimum(d_in, 0.995 * length)) ** 2
        hinge = None
        if pin_mode == "ref":
            # Re-pin (later stages): only a target the leg cannot reach at all lowers the pelvis (an
            # unchanged motion reaches every target, so re-pinning it stays a no-op). "At all" is the
            # IK's own reach for this knee hinge, which is shorter than the bones' sum when the calf
            # leaves the hinge plane; [M, round 5] with the sum, a nearly straight planted leg missed
            # its pin by ~0.1 mm on the first or last frames of a plant, and the planted step there
            # grew (projects/ufc_cross B frame 249: contacts 2.29, smooth 2.70 cm/s; ufc_cross A frame 283
            # under harness noise: 2.87 -> 3.19 cm/s).
            hinge = knee_hinge_local(sk, wrot, pos, leg, fwd)
            r2 = (leg_max_reach(sk, wrot, pos, leg, hinge) * (1.0 - 1e-6)) ** 2
        dist2 = np.sum(rel * rel, axis=-1)
        disc = b * b - (dist2 - r2)
        need = np.where(dist2 > r2, np.where(disc >= 0, b - np.sqrt(np.maximum(disc, 0.0)), b), 0.0)
        need = np.where(need > 1e-5, need, 0.0)
        # Only fully planted frames ask for a drop: their targets are world-fixed pins. Blend-frame targets
        # are eased (and, with ``knee_blend``, re-shaped in knee angle), so a stretch there is no reason
        # to lower the pelvis; counting them held the drop over whole gaits (ref walk B 2.0 -> 1.7 cm).
        # (The plant masks, not w >= 1: frames of a short gap between two plants sum to w >= 1.)
        planted = np.zeros(n_frames, dtype=bool)
        for c in (heel, toe):
            if c is not None:
                planted |= masks.get(c.channel, np.zeros(n_frames, bool))
        drop = np.maximum(drop, np.where(planted, np.clip(need, 0.0, params.max_pelvis_drop), 0.0))
        plans.append(_LegPlan(leg, heel, toe, pins, ankle_off, w_h, w_t0, w_any, planted, hinge))
    if allow_pelvis:
        drop = np.minimum(cover_smooth(drop, params.pelvis_smooth_frames), params.max_pelvis_drop)
    else:
        drop = np.zeros(n_frames)
    new_root = root - drop[:, None] * nrm
    pos2 = forward_kinematics(sk, rot, new_root)[0] if drop.any() else pos
    targets: list[LegTarget] = []
    for leg, heel, toe, pins, ankle_off, w_h, w_t0, w_any, leg_planted, hinge in plans:
        ia = sk.index(leg.foot)
        # Planted (w = 1) targets are world-fixed; unplanted legs follow the pelvis (w = 0).
        ankle = pos2[:, ia] + ankle_off + (w_any * drop)[:, None] * nrm
        if params.knee_blend:
            ankle = _knee_space_blend(sk, leg, pos2, ankle, leg_planted, params.blend_frames)
        active = w_any > 0
        if params.fix_penetration and heel is not None:
            h = floor.height(ankle)
            want = floor_lift(h, clear[heel.channel], width, w_h)
            lift_ok = want > h + 1e-5  # ignore float32 rounding of a foot resting on the floor
            ankle += np.where(lift_ok, want - h, 0.0)[:, None] * nrm
            active |= lift_ok
        foot_rot = foot_w = None
        if heel is not None and any(r is not None for r in pins.heel_rot):
            foot_rot = np.tile([0.0, 0.0, 0.0, 1.0], (n_frames, 1))
            foot_w = np.zeros(n_frames)
            hm = masks.get(heel.channel, np.zeros(n_frames, bool))
            for (s, e), rq in zip(segments(hm), pins.heel_rot, strict=True):
                if rq is None:
                    continue
                lo, hi = max(0, s - params.blend_frames), min(n_frames, e + params.blend_frames)
                # The blend frames of a plant must not overwrite the frames of a neighbouring heel
                # plant (they did: the next plant's orientation turned the last frames of this one,
                # and the toe left its pin by up to 0.8 cm for 2-3 frames).
                sel = np.arange(lo, hi)
                sel = sel[((sel >= s) & (sel < e)) | ~hm[sel]]
                if rq.ndim == 2:  # per-frame (re-pin mode): the reference's own foot rotation
                    foot_rot[sel] = ref_wrot[sel, sk.index(leg.foot)]
                else:
                    foot_rot[sel] = rq
                foot_w[sel] = np.maximum(foot_w[sel], np.minimum(w_h[sel], w_t0[sel]))
            # Frames of a short gap between two pinned plants turn from one pinned orientation to the
            # next (they took the later plant's at full weight: a 22 deg/frame snap on ufc B).
            segs_h = segments(hm)
            for i in range(len(segs_h) - 1):
                (_, e0), (s1, _) = segs_h[i], segs_h[i + 1]
                q0, q1 = pins.heel_rot[i], pins.heel_rot[i + 1]
                if q0 is None or q1 is None or q0.ndim != 1 or s1 - e0 > 2 * params.blend_frames:
                    continue
                f = (np.arange(e0, s1) - e0 + 1.0) / (s1 - e0 + 1.0)
                foot_rot[e0:s1] = slerp(np.tile(q0, (s1 - e0, 1)), np.tile(q1, (s1 - e0, 1)), f)
            active |= foot_w > 0
        toe_target = toe_w = toe_floor = None
        if toe is not None and leg.toe is not None:
            tm = masks.get(toe.channel, np.zeros(n_frames, bool))
            rigid = pos2[:, sk.index(leg.toe)] + (ankle - pos2[:, ia])
            off_t, w_t, _ = channel_offsets(
                rigid, None, tm, floor, clear[toe.channel], params.blend_frames, pins.toe
            )
            toe_target = rigid + off_t
            flat_w = np.zeros(n_frames) if foot_w is None else foot_w
            toe_w = (w_t > 0) * (1.0 - flat_w)  # flat plants are held by the foot orientation
            if heel is not None:
                # ... but a toe planted while its heel is not (push-off, toe-first landing) is held by
                # the toe correction itself, not by the heel plant's eased-out orientation.
                toe_w = np.where(tm & ~masks.get(heel.channel, np.zeros(n_frames, bool)), 1.0, toe_w)
            active |= w_t > 0
            if params.fix_penetration:
                toe_floor = (nrm, floor.offset, clear[toe.channel], width, w_t)
                active |= floor.height(rigid) < clear[toe.channel] + width - 1e-5
        targets.append(
            LegTarget(
                leg,
                ankle,
                active,
                toe_target,
                toe_w,
                toe_floor,
                hinge_local=hinge,
                foot_rot=foot_rot,
                foot_rot_weight=foot_w,
            )
        )
    new_rot = make_continuous(solve_leg_ik(sk, rot, new_root, targets, forward=fwd))
    if params.ik_smooth > 0 and n_frames >= 5 and pin_mode == "median":
        new_rot = _smooth_leg_correction(
            sk, roles, rot, new_rot, new_root, targets, plans, params, fwd, floor
        )
    out_pos = forward_kinematics(sk, new_rot, new_root)[0]
    miss = {}
    for c in roles.contacts:
        m = masks.get(c.channel, np.zeros(n_frames, bool))
        d = np.zeros(n_frames)
        for (s, e), pin in zip(segments(m), all_pins.get(c.channel, []), strict=True):
            d[s:e] = np.linalg.norm(out_pos[s:e, sk.index(c.joint)] - pin, axis=-1)
        miss[c.channel] = d
    raised = sum(int(x > 1e-4) for pl in plans for x in pl.pins.raised)
    capped = sum(int(x) for pl in plans for x in pl.pins.capped)
    floating = {}
    for pl in plans:
        if pl.toe is None or not pl.pins.floating:
            continue
        fl = np.zeros(n_frames, dtype=bool)
        tm = masks.get(pl.toe.channel, np.zeros(n_frames, bool))
        for (s, e), up in zip(segments(tm), pl.pins.floating, strict=True):
            fl[s:e] = up
        floating[pl.toe.channel] = fl
    stepping = {}
    if pin_mode == "median":
        fps_step = params.max_plant_step / max(fps, 1e-9)  # m per frame
        for c in roles.contacts:
            m = masks.get(c.channel, np.zeros(n_frames, bool))
            st = np.zeros(n_frames, dtype=bool)
            both = np.flatnonzero(m[1:] & m[:-1])
            if both.size:
                p_c = out_pos[:, sk.index(c.joint)]
                fast = both[np.linalg.norm(p_c[both + 1] - p_c[both], axis=-1) > fps_step]
                d = miss[c.channel]
                worse = np.where(d[fast + 1] >= d[fast], fast + 1, fast)
                st[worse] = True
            stepping[c.channel] = st
        # Edge pops: a foot that leaves (or reaches) its pin much faster than the input foot moved
        # (a pin far from where the input foot was at the plant's edge, or two pins a short gap apart)
        # gets the nearest plant edge frame unplanted, so the next solve has one more frame to move.
        # These only trim: a plant keeps at least ``min_plant_frames`` (ref walk/002 B lost its only one).
        edge = {c.channel: np.zeros(n_frames, dtype=bool) for c in roles.contacts}
        for li, leg in enumerate(roles.legs):
            chans = [c.channel for c in roles.contacts if c.leg == li]
            planted = np.zeros(n_frames, dtype=bool)
            for ch in chans:
                planted |= masks.get(ch, np.zeros(n_frames, bool))
            if not planted.any():
                continue
            pops = np.zeros(n_frames, dtype=bool)
            for jn in (leg.foot, leg.toe):
                if jn is None:
                    continue
                j = sk.index(jn)
                so = np.linalg.norm(np.diff(out_pos[:, j], axis=0), axis=-1)
                si = np.linalg.norm(np.diff(pos[:, j], axis=0), axis=-1)
                pops[1:] |= (so - si > params.max_edge_pop) & (so > 2.0 * si)
            k_out, k_in = _knee_angles(sk, leg, out_pos), _knee_angles(sk, leg, pos)
            snap = np.zeros(n_frames, dtype=bool)
            snap[1:] = np.abs(np.diff(k_out)) - np.abs(np.diff(k_in)) > np.radians(params.max_knee_snap_deg)
            inside = planted & np.r_[False, planted[:-1]]  # steps between two planted frames
            for t in np.flatnonzero((pops & ~inside) | snap):
                if inside[t]:  # a snap within a plant: the straighter (struggling) of the two frames
                    ext = np.linalg.norm(
                        out_pos[:, sk.index(leg.thigh)] - out_pos[:, sk.index(leg.foot)], axis=-1
                    )
                    cand = [t if ext[t] >= ext[t - 1] else t - 1]
                else:  # next to a plant: its nearest edge frame on either side
                    back = [k for k in range(t - 1, max(-1, t - 2 - params.blend_frames), -1) if planted[k]]
                    fwd = [k for k in range(t, min(n_frames, t + 1 + params.blend_frames)) if planted[k]]
                    cand = back[:1] + fwd[:1]
                for k in cand:
                    for ch in chans:
                        if masks.get(ch, np.zeros(n_frames, bool))[k]:
                            edge[ch][k] = True
        for ch, e_m in edge.items():
            for s0, e0 in segments(masks.get(ch, np.zeros(n_frames, bool))):
                if (e0 - s0) - int(e_m[s0:e0].sum()) >= params.min_plant_frames:
                    stepping[ch][s0:e0] |= e_m[s0:e0]
        # A planted foot touches the floor with its lowest sole point (heel or ball): a raised heel pin
        # is a heel lift only while the ball is down. Frames whose whole foot is higher than
        # ``max_plant_hover`` float (a raised heel-only plant, or a pin miss upwards).
        lowest = foot_sole_heights(sk, roles, out_pos, floor)
        for c in roles.contacts:
            up = lowest[c.leg] > params.max_plant_hover + HOVER_TOL
            floating[c.channel] = floating.get(c.channel, np.zeros(n_frames, dtype=bool)) | up
    return FootFix(
        new_rot, new_root, drop - lift, weights, all_pins, miss, raised, capped, floating, stepping
    )


def _knee_angle(d: np.ndarray, l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """Knee bend (0 = straight) of a two-bone leg whose hip-ankle distance is ``d``."""
    return np.arccos(np.clip((d * d - l1 * l1 - l2 * l2) / np.maximum(2.0 * l1 * l2, 1e-12), -1.0, 1.0))


def _smootherstep_tail(blend: int) -> tuple[np.ndarray, np.ndarray]:
    """Frames 1..``blend`` away from a plant and their easing weights (as :func:`channel_offsets`)."""
    k = np.arange(1, blend + 1)
    x = (blend + 1 - k) / (blend + 1)  # 1 next to the plant, -> 0 far away
    return k, x**3 * (10.0 - 15.0 * x + 6.0 * x * x)


def _knee_space_blend(
    sk: Skeleton, leg: LegRoles, pos: np.ndarray, ankle: np.ndarray, planted: np.ndarray, blend: int
) -> np.ndarray:
    """Ankle targets [F,3] whose blend into/out of the plants (``planted`` = the leg's plant mask) is
    eased in knee angle instead of position.

    The offsets are eased in position space (``channel_offsets``), but near a straight leg the knee
    angle is extremely sensitive to the hip-ankle distance (0.99 -> 0.96 of the leg length is a
    ~25 degree bend), so a few centimetres of eased offset in one frame snapped the knee (ref walk:
    27 deg/frame on a straight-legged walk). Here each blend frame keeps the eased target's direction
    from the hip, but its distance is the one whose knee angle is the input's own knee angle plus the
    knee correction (target's knee angle minus the input's) of the plants around it, built exactly
    like the position offsets: inside a plant the correction is the target's; outside, each plant's
    edge correction is continued with its own edge rate (so the knee's angular velocity is continuous
    across the edge) and eased by that plant's smootherstep weight ``w_k``, and the plants are
    combined as ``sum(w_k c_k) / max(1, sum(w_k))``. With two close plants both contribute over the
    whole gap, so the knee moves smoothly from one plant's correction to the other's (taking only the
    nearest plant's switched mid-gap: a 17 deg step in a 5-frame gap). At plant frames and frames no
    plant reaches nothing changes, and a re-pin of an unchanged motion (zero offsets) is a no-op.

    ``pos`` is the motion after the pelvis drop (hip, knee and the input ankle all moved with it),
    ``ankle`` the eased targets (world-fixed in plants)."""
    planted = np.asarray(planted, dtype=bool)
    n_frames = len(planted)
    plants = segments(planted)
    if not plants or blend <= 0 or planted.all():
        return ankle
    hip = pos[:, sk.index(leg.thigh)]
    knee = pos[:, sk.index(leg.calf)]
    ank_in = pos[:, sk.index(leg.foot)]
    l1 = np.linalg.norm(knee - hip, axis=-1)
    l2 = np.linalg.norm(ank_in - knee, axis=-1)
    th_in = _knee_angle(np.linalg.norm(ank_in - hip, axis=-1), l1, l2)
    corr = _knee_angle(np.linalg.norm(ankle - hip, axis=-1), l1, l2) - th_in
    k, wk = _smootherstep_tail(blend)
    acc = np.zeros(n_frames)
    wacc = np.zeros(n_frames)
    for s, e in plants:
        c = corr[s:e]
        span = min(2, e - s - 1)
        r_s = (c[span] - c[0]) / span if span > 0 else 0.0
        r_e = (c[-1] - c[-1 - span]) / span if span > 0 else 0.0
        for t, val in ((s - k, c[0] - k * r_s), (e - 1 + k, c[-1] + k * r_e)):
            ok = (t >= 0) & (t < n_frames)
            np.add.at(acc, t[ok], wk[ok] * val[ok])
            np.add.at(wacc, t[ok], wk[ok])
    t = np.flatnonzero(~planted & (wacc > 0))
    if not t.size:
        return ankle
    th = np.clip(th_in[t] + acc[t] / np.maximum(1.0, wacc[t]), 0.0, np.pi)
    d_new = np.sqrt(np.maximum(l1[t] ** 2 + l2[t] ** 2 + 2.0 * l1[t] * l2[t] * np.cos(th), 0.0))
    rel = ankle[t] - hip[t]
    dist = np.linalg.norm(rel, axis=-1)
    out = ankle.copy()
    ok = dist > 1e-9
    out[t[ok]] = hip[t[ok]] + rel[ok] / dist[ok, None] * d_new[ok, None]
    return out


def _smooth_lift(need: np.ndarray, free: np.ndarray, radius: int = 2) -> np.ndarray:
    """Smooth a per-frame floor-lift requirement [F] without undercutting it, separately inside each
    run of ``free`` (unplanted) frames, so the lift never leaks into a plant; 0 on planted frames."""
    out = np.zeros(len(need))
    req = np.where((need > 1e-4) & free, need, 0.0)
    for s, e in segments(free):
        if req[s:e].any():
            out[s:e] = cover_smooth(req[s:e], radius)
    return out


def _smooth_leg_correction(
    sk: Skeleton,
    roles: BodyRoles,
    rot: np.ndarray,
    new_rot: np.ndarray,
    root: np.ndarray,
    targets: list[LegTarget],
    plans: list[_LegPlan],
    params: ContactParams,
    fwd: np.ndarray,
    floor: Floor | None = None,
) -> np.ndarray:
    """Smooth the IK correction (local delta rotations of the leg chains) in time, holding planted
    frames (with ``knee_blend`` also the blend frames) almost fixed (data weight 100), then re-solve
    exactly there: frame-to-frame IK changes on swinging legs and singular poses are spread instead
    of popping."""
    from afv.motion.ik import chain
    from afv.motion.rotations import qmul
    from afv.motion.smooth import smooth_rotations

    # Everything is per leg: a leg is held (and re-solved exactly) only where *its own* foot is
    # planted. With the union of both legs, a swinging leg was re-solved exactly on every frame the
    # other foot was planted and kept the smoothed correction elsewhere, so it alternated between
    # two solutions (knee snaps), and its floor clamp was skipped whenever the other foot was down.
    n_frames = rot.shape[0]
    planted_leg = [
        pl.planted if pl.planted is not None else (pl.w_h >= 1.0) | (pl.w_t0 >= 1.0) for pl in plans
    ]
    if params.knee_blend:
        # The blend frames are already eased in knee angle (_knee_space_blend): held and re-solved
        # exactly like the plants. Smoothing them with the plant held at its (often much more bent)
        # knee overshot it (ref walk: 1.5 -> 27 -> 42 deg on frames planned as 10 -> 21 -> 28).
        planted_leg = [pl.w_any > 0 for pl in plans]
    leg_joints = [chain(sk, leg.thigh, leg.foot) for leg in roles.legs]
    joints = sorted({j for js in leg_joints for j in js})
    delta = make_continuous(qmul(new_rot[:, joints], qconj(rot[:, joints])))
    hold_w = np.ones((n_frames, len(joints)))
    for js, pl in zip(leg_joints, planted_leg, strict=True):
        cols = [joints.index(j) for j in js]
        hold_w[:, cols] = np.where(pl, 100.0, 1.0)[:, None]
    lam = params.ik_smooth * np.ones_like(hold_w)
    delta_s = smooth_rotations(delta, hold_w, 0.2 * lam, 4.0 * lam, 4)
    out = new_rot.copy()
    out[:, joints] = qmul(delta_s, rot[:, joints])
    exact = []
    for tg, planted in zip(targets, planted_leg, strict=True):
        tg2 = LegTarget(
            tg.leg,
            tg.ankle,
            tg.active & planted,
            tg.toe,
            tg.toe_weight,
            tg.floor,
            tg.hinge_local,
            tg.foot_rot,
            tg.foot_rot_weight,
        )
        exact.append(tg2)
    out = make_continuous(solve_leg_ik(sk, out, root, exact, forward=fwd))
    # Smoothing may have pushed a swinging foot below the floor. Lift it back, but not with a per-frame
    # clamp: a foot grazing the floor for a frame or two would get a velocity kink (on the real fixtures
    # 2-4x the swing's typical vertical acceleration). The lift each frame needs is smoothed in time
    # with the never-undercut filter (so the foot still ends at or above its clearance) and applied to
    # the ankle (heel) and then, after that rigid raise, to the toe; an exact clamp stays as a safety.
    if params.fix_penetration and floor is not None:
        pos = forward_kinematics(sk, out, root)[0]
        clear = roles.clearances(sk)
        n = floor.normal
        fixes = []
        for li, leg in enumerate(roles.legs):
            chans = [ch for ch in roles.contacts if ch.leg == li]
            heel = next((ch for ch in chans if ch.kind == "heel"), None)
            toe = next((ch for ch in chans if ch.kind == "toe"), None)
            free = ~planted_leg[li]
            ia = sk.index(leg.foot)
            ankle = pos[:, ia].copy()
            lift_a = np.zeros(n_frames)
            if heel is not None:
                lift_a = np.maximum(0.0, clear[heel.channel] - floor.height(ankle)) * free
            lift_a = _smooth_lift(lift_a, free)
            ankle += lift_a[:, None] * n
            toe_t = toe_w = tf = None
            lift_t = np.zeros(n_frames)
            if toe is not None and leg.toe is not None:
                toe_p = pos[:, sk.index(leg.toe)] + lift_a[:, None] * n
                lift_t = np.maximum(0.0, clear[toe.channel] - floor.height(toe_p)) * free
                lift_t = _smooth_lift(lift_t, free)
                toe_t = toe_p + lift_t[:, None] * n
                toe_w = (lift_t > 0).astype(np.float64)
                tf = (n, floor.offset, clear[toe.channel], 0.0, None)
            low = free & ((lift_a > 0) | (lift_t > 0))
            if not low.any():
                continue
            fixes.append(LegTarget(leg, ankle, low, toe_t, toe_w, tf))
        if fixes:
            out = make_continuous(solve_leg_ik(sk, out, root, fixes, forward=fwd))
    return out


# ---------------------------------------------------------------------------
# Whole-shot driver
# ---------------------------------------------------------------------------


def actor_skeleton(bm: BodyMotion, actor: ActorMotion) -> Skeleton:
    sk = get_skeleton(bm.skeleton)
    return sk.with_offsets(actor.rest_offsets) if actor.rest_offsets is not None else sk


def model_probabilities(
    roles: BodyRoles, channels: list[str], contact: np.ndarray | None
) -> dict[str, np.ndarray | None]:
    """Input contact probability per foot channel (first matching source), None if absent."""
    out: dict[str, np.ndarray | None] = {}
    for c in roles.contacts:
        out[c.channel] = None
        if contact is None:
            continue
        for src in (c.channel, *c.sources):
            if src in channels:
                out[c.channel] = np.asarray(contact[:, channels.index(src)], dtype=np.float64)
                break
    return out


def consumed_channels(roles: BodyRoles, channels: list[str]) -> set[str]:
    used = set()
    for c in roles.contacts:
        for src in (c.channel, *c.sources):
            if src in channels:
                used.add(src)
                break
    return used


def channel_sole_heights(
    sk: Skeleton, roles: BodyRoles, pos: np.ndarray, floor: Floor
) -> dict[str, np.ndarray]:
    """Height above ``floor`` of each contact channel's sole point [F]: the channel joint's height
    minus its clearance (its height above the sole with the foot flat in the rest pose). The one
    sole reference of the stage (detection, floor fit, pins, floor clamps) and of the QC metrics."""
    clear = roles.clearances(sk)
    return {c.channel: floor.height(pos[:, sk.index(c.joint)]) - clear[c.channel] for c in roles.contacts}


def foot_sole_heights(sk: Skeleton, roles: BodyRoles, pos: np.ndarray, floor: Floor) -> np.ndarray:
    """Lowest sole point of each leg's foot above ``floor`` [L,F] (min over its heel and toe
    channels, see :func:`channel_sole_heights`): what touches the floor, so a planted foot is on
    the floor when this is at most ``max_plant_hover`` (a heel lifted 3 cm off a planted ball is)."""
    h = channel_sole_heights(sk, roles, pos, floor)
    out = np.full((len(roles.legs), len(pos)), np.inf)
    for c in roles.contacts:
        out[c.leg] = np.minimum(out[c.leg], h[c.channel])
    return out


def sole_points(
    sk: Skeleton, roles: BodyRoles, pos: np.ndarray, up: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sole points [F,C,3] (joint minus clearance along ``up``) and the lowest per frame [F,3]."""
    clear = roles.clearances(sk)
    pts = np.stack([pos[:, sk.index(c.joint)] - clear[c.channel] * up for c in roles.contacts], axis=1)
    lowest = pts[np.arange(len(pts)), np.argmin(pts @ up, axis=1)]
    return pts, lowest


@dataclass
class ActorContactInfo:
    probs: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    fix: FootFix | None = None
    root_offset: np.ndarray | None = None  # [F,3] root correction (step 3)
    raised_plants: int = 0  # final heel plants whose pin was raised to be reachable
    capped_plants: int = 0  # heel plants (first solve) that needed more than ``max_pin_raise``
    dropped_plants: int = 0  # plants (any channel, first solve) with no frame left at the end
    miss_unplanted_frames: int = 0  # plant frames unplanted by the pin-miss rule and re-solved
    unsolved_unplanted_frames: int = 0  # ... still missing after the last re-solve (not re-solved)
    slide_trimmed_frames: int = 0  # plant-end frames unflagged because the planted joint still slid
    pivot_split_frames: int = 0  # toe frames unflagged to split toe plants spanning two heel plants
    held_holes_filled: int = 0  # frames of 1-2 frame flag holes re-merged because the joint stayed put
    # [K,F,3] the total root offset after each root-correction iteration and [K,C,F] the plant masks
    # it used (range re-runs replay them)
    offset_iters: np.ndarray | None = None
    mask_iters: np.ndarray | None = None
    replayed: bool = False  # range re-run: every root solve was replayed (the previous run's data)


@dataclass
class ContactReport:
    floor: Floor  # the floor of the *output* motion (y = 0 when ``floor_to_zero``)
    actors: dict[str, ActorContactInfo]
    floor_shift: float = 0.0  # metres added to every root's y (output = input + shift)
    filled_frames: dict[str, int] = field(default_factory=dict)
    zone: tuple[int, int] | None = None  # range re-run: the frames re-solved (indices, end exclusive)


@dataclass
class FrozenContacts:
    """A range re-run of :func:`clean_contacts` (``afv.motion.ranges``): only ``editable`` frames
    (and whole plants that touch them, at most ``max_extend`` frames further) are re-solved; the
    root offsets elsewhere are held at the previous run's (``offset_iters`` per fighter, [K,F,3]: the
    total offset after each root-correction iteration), so the solve there is a boundary condition
    and the rest of the shot is reproduced. ``masks``: the previous result's plant masks per fighter."""

    editable: tuple[int, int]
    offset_iters: dict[str, np.ndarray]
    masks: dict[str, dict[str, np.ndarray]]
    max_extend: int = 45
    # [K,C,F] the plant masks each root-correction iteration used (roles.contacts order): where a
    # fighter's masks are unchanged the previous offsets are replayed (the robust solve is path
    # dependent: re-solving identical data inside a fixed boundary moved a root by up to 5 cm)
    mask_iters: dict[str, np.ndarray] | None = None


def frozen_zone(
    editable: tuple[int, int], masks: list[dict[str, np.ndarray]], n_frames: int, max_extend: int
) -> tuple[int, int]:
    """``editable`` grown (repeatedly) to whole plant segments of any of ``masks`` that touch it, at
    most ``max_extend`` frames beyond ``editable`` on each side."""
    e0, e1 = max(0, editable[0]), min(n_frames, editable[1])
    lo_cap, hi_cap = max(0, e0 - max_extend), min(n_frames, e1 + max_extend)
    segs = [seg for mm in masks for m in mm.values() for seg in segments(np.asarray(m, dtype=bool))]
    z0, z1 = e0, e1
    while True:
        a, b = z0, z1
        for s0, e0_ in segs:
            if s0 < b and e0_ > a:
                a, b = min(a, s0), max(b, e0_)
        a, b = max(a, lo_cap), min(b, hi_cap)
        if (a, b) == (z0, z1):
            return z0, z1
        z0, z1 = a, b


ZONE_TAPER = 8  # frames: a range re-run's root offset change eases to 0 at the zone edges (C1 seam)


def floor_zero_shift(floor: Floor, roots: list[np.ndarray]) -> tuple[float, Floor]:
    """Vertical shift that puts ``floor`` at y = 0 under the fighters (exact for a horizontal
    floor; for a tilted one, at the fighters' mean position), and the floor after the shift."""
    c = (
        np.mean(np.concatenate([np.asarray(r, dtype=np.float64) for r in roots]), axis=0)
        if roots
        else np.zeros(3)
    )
    n = floor.normal
    y_plane = -(n[0] * c[0] + n[2] * c[2] + floor.offset) / n[1]
    dy = -float(y_plane)
    return dy, Floor(n, floor.offset - n[1] * dy)


def detect_actor(
    sk: Skeleton,
    roles: BodyRoles,
    pos: np.ndarray,
    floor: Floor,
    fps: float,
    model: dict[str, np.ndarray | None],
    params: ContactParams,
    painted: dict[str, list[tuple[int, int, bool]]] | None = None,
    final: bool = True,
) -> ActorContactInfo:
    """Per-channel probabilities and plant masks. ``final`` False = candidate plants (weak speed
    vote, no veto or travel cap: the root is not corrected yet)."""
    clear = roles.clearances(sk)
    probs, masks = {}, {}
    for c in roles.contacts:
        p = pos[:, sk.index(c.joint)]
        h = floor.height(p) - clear[c.channel]
        v = horizontal_speed(p, floor.normal, fps)
        sw = None if final else params.candidate_speed_weight
        probs[c.channel] = contact_probability(h, v, model.get(c.channel), params, sw, veto=final)
        m = hysteresis_mask(probs[c.channel], params)
        # Onset refinement: the landing frame reads as moving (its backward window is swing); take
        # it into the plant when it is on the floor and its forward window is still.
        _, fwd = one_sided_speeds(p, floor.normal, fps)
        mdl = model.get(c.channel)
        for s, _e in segments(m):
            certain_off = mdl is not None and mdl[s - 1] < 0.05  # e.g. this stage's own output
            if (
                s > 0
                and not certain_off
                and h[s - 1] < params.height_threshold
                and fwd[s - 1] < 1.5 * params.speed_threshold
            ):
                m[s - 1] = True
        if final:
            m = cap_plant_travel(m, p, floor.normal, fps, params)
        masks[c.channel] = m
    split = 0
    if final:
        split = split_pivot_toe_plants(sk, roles, pos, masks, floor.normal, params, fps)
    for c in roles.contacts:
        for i0, i1, plant in (painted or {}).get(c.channel, []):
            masks[c.channel][i0:i1] = plant
    return ActorContactInfo(probs, masks, pivot_split_frames=split)


def split_pivot_toe_plants(
    sk: Skeleton,
    roles: BodyRoles,
    pos: np.ndarray,
    masks: dict[str, np.ndarray],
    normal: np.ndarray,
    params: ContactParams,
    fps: float = 30.0,
) -> int:
    """Split (in place) every toe plant that overlaps more than one heel plant of its foot, at the
    toe's fastest frame in each gap between those heel plants; pieces shorter than
    ``min_plant_frames`` are dropped. Returns the number of frames unflagged.

    When the ball moved between the pieces (their median toe positions in the floor plane differ by
    more than ``max_pin_miss``) it is a step, not a pivot: the cut is widened to the frames the toe
    needs to travel that far at ``speed_threshold``, so the foot is not pulled from one pin to the
    other in a single frame (ufc A: pins ~7 cm apart, 24 deg/frame).

    A toe plant's pin is derived from one heel plant's rigid foot pin; on a pivot over the ball (heel
    lifted, swung and set down again) the other heel plant's foot pin put the toe elsewhere, so every
    one of those frames missed its pin by 2-4 cm and was unplanted (Who Am I A, 23 of 23 frames)."""
    removed = 0
    for li in range(len(roles.legs)):
        heel = next((c for c in roles.contacts if c.leg == li and c.kind == "heel"), None)
        toe = next((c for c in roles.contacts if c.leg == li and c.kind == "toe"), None)
        if heel is None or toe is None or heel.channel not in masks or toe.channel not in masks:
            continue
        hm, tm = masks[heel.channel], masks[toe.channel]
        p = pos[:, sk.index(toe.joint)]
        step = np.zeros(len(p))
        if len(p) > 1:
            d = np.diff(p, axis=0)
            step[1:] = np.linalg.norm(d - (d @ normal)[:, None] * normal, axis=-1)
        for s, e in segments(tm):
            hs = segments(hm[s:e])
            if len(hs) < 2:
                continue
            for (s0, e0), (s1, e1) in itertools.pairwise(hs):
                gap = np.arange(s + e0, s + s1)  # toe-only frames between two heel plants
                cut = int(gap[np.argmax(step[gap])])
                q0 = np.median(p[s + s0 : cut], axis=0)
                q1 = np.median(p[cut + 1 : s + e1], axis=0)
                dq = q1 - q0
                moved = float(np.linalg.norm(dq - (dq @ normal) * normal))
                width = 1
                if moved > params.max_pin_miss:
                    width = max(1, int(np.ceil(moved * fps / max(params.speed_threshold, 1e-6))))
                lo = max(s + e0, cut - (width - 1) // 2)  # never under the heel plants
                hi = min(s + s1, lo + width)
                removed += int(tm[lo:hi].sum())
                tm[lo:hi] = False
        for s, e in segments(tm):
            if e - s < params.min_plant_frames:
                tm[s:e] = False
                removed += e - s
    return removed


def _keep_clean_floor(
    floor: Floor,
    sks: dict[str, Skeleton],
    roles: BodyRoles,
    pos: dict[str, np.ndarray],
    models: dict[str, dict[str, np.ndarray | None]],
    params: ContactParams,
    up: np.ndarray,
) -> Floor:
    """Keep y = 0 when the input already follows the post-contacts convention: its planted soles
    (per the input's contact probabilities) all sit within [0, ``max_plant_hover``] of y = 0. A
    refit would move the floor by the kept hover, so re-running the stage would not be a no-op."""
    hs = []
    for r in pos:
        clear = roles.clearances(sks[r])
        for c in roles.contacts:
            pm = models[r].get(c.channel)
            if pm is None:
                return floor
            m = pm > 0.5
            hs.append(pos[r][m, sks[r].index(c.joint)] @ up - clear[c.channel])
    h = np.concatenate(hs) if hs else np.zeros(0)
    if h.size and np.mean((h > -5e-3) & (h < params.max_plant_hover + 1e-2)) >= 0.8:
        return Floor.horizontal(0.0)
    return floor


def _missed_frames(
    fix: FootFix, masks: dict[str, np.ndarray], forced: dict[str, np.ndarray], params: ContactParams
) -> dict[str, np.ndarray]:
    """Planted, non-painted frames whose pin the leg misses (> ``max_pin_miss``), whose (toe) pin
    or foot floats, or whose joint steps faster than ``max_plant_step`` to a planted neighbour."""
    out = {}
    for ch, d in fix.miss.items():
        up = fix.floating.get(ch, np.zeros(len(d), dtype=bool))
        st = fix.stepping.get(ch, np.zeros(len(d), dtype=bool))
        out[ch] = ((d > params.max_pin_miss) | up | st) & masks[ch] & ~forced[ch]
    return out


def _trim_sliding_edges(
    sk: Skeleton,
    roles: BodyRoles,
    fix: FootFix,
    masks: dict[str, np.ndarray],
    forced: dict[str, np.ndarray],
    fps: float,
    params: ContactParams,
) -> int:
    """Unflag (in place) frames at either end of each non-painted plant whose planted joint moves
    faster than ``max_edge_slide`` to its in-plant neighbour, then pieces shorter than
    ``min_plant_frames``; returns the number of frames removed. Only the flags change: such frames
    are where a leg stops reaching its pin (within ``max_pin_miss``), and a game's foot IK keyed on
    the flag would pin a sliding foot."""
    pos = forward_kinematics(sk, fix.rot, fix.root)[0]
    bad = {}
    for c in roles.contacts:
        m = masks.get(c.channel)
        if m is None:
            continue
        step = np.linalg.norm(np.diff(pos[:, sk.index(c.joint)], axis=0), axis=-1) * fps
        b = np.zeros(len(m), dtype=bool)
        for s, e in segments(m):
            i = s
            while i < e - 1 and step[i] > params.max_edge_slide:  # frame i -> i + 1
                b[i] = True
                i += 1
            j = e - 1
            while j > i and step[j - 1] > params.max_edge_slide:  # frame j - 1 -> j
                b[j] = True
                j -= 1
        bad[c.channel] = b & ~forced[c.channel]
    return _unplant(masks, bad, forced, params)


def _fill_held_holes(
    sk: Skeleton,
    roles: BodyRoles,
    fix: FootFix,
    masks: dict[str, np.ndarray],
    free_paint: dict[str, np.ndarray],
    fps: float,
    params: ContactParams,
) -> int:
    """Re-flag (in place) holes of at most 2 frames between two plants of a channel when its joint
    stays put through them (every step slower than ``max_plant_step``): the unplant rules can leave a
    one-frame hole in a foot the solve still holds (moved 0.0 cm), and a game's foot IK would let go
    of the foot for that frame. Painted "free" frames stay free. Returns the number of frames."""
    pos = forward_kinematics(sk, fix.rot, fix.root)[0]
    lim = params.max_plant_step / max(fps, 1e-9)
    filled = 0
    for c in roles.contacts:
        m = masks.get(c.channel)
        if m is None:
            continue
        p = pos[:, sk.index(c.joint)]
        for (_, e0), (s1, _) in itertools.pairwise(segments(m)):
            if s1 - e0 > 2 or free_paint[c.channel][e0:s1].any():
                continue
            steps = np.linalg.norm(np.diff(p[e0 - 1 : s1 + 1], axis=0), axis=-1)
            if steps.max() <= lim:
                m[e0:s1] = True
                filled += s1 - e0
    return filled


def _unplant(
    masks: dict[str, np.ndarray],
    bad: dict[str, np.ndarray],
    forced: dict[str, np.ndarray],
    params: ContactParams,
) -> int:
    """Unplant ``bad`` frames in place, then non-painted pieces shorter than ``min_plant_frames``;
    returns the number of frames removed."""
    n0 = sum(int(m.sum()) for m in masks.values())
    for ch, b in bad.items():
        m = masks[ch] & ~b
        for s0, e0 in segments(m):
            if e0 - s0 < params.min_plant_frames and not forced[ch][s0:e0].any():
                m[s0:e0] = False
        masks[ch] = m
    return n0 - sum(int(m.sum()) for m in masks.values())


def clean_contacts(
    bm: BodyMotion,
    params: ContactParams | None = None,
    roles: BodyRoles | None = None,
    painted: Any = None,
    floor: Floor | None = None,
    camera: CameraTrack | None = None,
    frozen: FrozenContacts | None = None,
    zero_shift: float | None = None,
) -> tuple[BodyMotion, ContactReport]:
    """Detect contacts, correct the root and fix skate/penetration for every fighter of ``bm`` on
    one shared floor (module docstring). ``camera`` (the camera stage's track) makes the root
    correction prefer depth along the camera ray, which keeps the image unchanged. ``frozen``: a
    range re-run (see :class:`FrozenContacts`; needs ``floor``); ``zero_shift``: the vertical shift
    to apply instead of the one the floor gives (a range re-run keeps the previous world).

    Returns a new BodyMotion whose ``contact_channels`` are the roles' foot channels (binary plant
    masks) followed by any input channels not used as foot evidence (e.g. GEM-X wrists, passed
    through), and a report with the floor and per-channel probabilities/masks.
    """
    params = params or ContactParams()
    bm, filled = sanitize_motion(bm)
    fps = bm.fps or 30.0
    base = get_skeleton(bm.skeleton)
    roles = roles or get_roles(base)
    up = np.asarray(roles.up, dtype=np.float64)
    sks = {r: actor_skeleton(bm, a) for r, a in bm.actors.items()}
    roots = {r: np.asarray(a.root, dtype=np.float64) for r, a in bm.actors.items()}
    pos = {r: forward_kinematics(sks[r], a.rot, roots[r])[0] for r, a in bm.actors.items()}
    models = {r: model_probabilities(roles, bm.contact_channels, a.contact) for r, a in bm.actors.items()}
    paints = {r: painted_spans(painted, r, roles, bm.frames) for r in bm.actors}

    def detect(final: bool) -> dict[str, ActorContactInfo]:
        return {
            r: detect_actor(sks[r], roles, pos[r], floor, fps, models[r], params, paints[r], final)
            for r in bm.actors
        }

    if floor is None:
        # First guess: lowest sole point per frame, trusted more when it is slow.
        pts, w = [], []
        for r in bm.actors:
            _, low = sole_points(sks[r], roles, pos[r], up)
            speed = horizontal_speed(low, up, fps)
            pts.append(low)
            w.append(np.where(speed < 2 * params.speed_threshold, 1.0, 0.2))
        floor = fit_floor(np.concatenate(pts), np.concatenate(w), params.floor_tilt, params.max_tilt_deg)
        for _ in range(params.floor_iterations):
            cand = detect(final=False)
            pts = []
            for r in bm.actors:
                spts, _ = sole_points(sks[r], roles, pos[r], up)
                for ci, c in enumerate(roles.contacts):
                    pts.append(spts[cand[r].masks[c.channel], ci])
            pts = [p for p in pts if len(p)]
            if pts:
                floor = fit_floor(np.concatenate(pts), None, params.floor_tilt, params.max_tilt_deg)
        floor = _keep_clean_floor(floor, sks, roles, pos, models, params, up)
    offsets = {r: np.zeros((bm.num_frames, 3)) for r in bm.actors}
    infos = detect(final=not params.root_correction)
    chans = [c.channel for c in roles.contacts]
    # A fighter whose own input plants (its contact channels) already stand still, e.g. this stage's
    # output fed back in, needs no root correction: re-running the stage must not move it.
    still = {}
    for r in bm.actors:
        steps = []
        for c in roles.contacts:
            pm = models[r].get(c.channel)
            if pm is None:
                continue
            for s0, e0 in segments(pm > 0.5):
                if e0 - s0 > 1:
                    steps.append(
                        np.linalg.norm(np.diff(pos[r][s0:e0, sks[r].index(c.joint)], axis=0), axis=-1)
                    )
        still[r] = bool(steps) and float(np.concatenate(steps).mean()) * fps < params.root_skip_speed
    zone = None
    if frozen is not None:
        zone = frozen_zone(
            frozen.editable,
            [i.masks for i in infos.values()] + list(frozen.masks.values()),
            bm.num_frames,
            frozen.max_extend,
        )
    roots0 = {r: v.copy() for r, v in roots.items()}
    pos0 = {r: v.copy() for r, v in pos.items()}
    infos0 = infos
    # Range re-runs: solve inside the zone, then grow the zone over every final plant (of this solve
    # or the previous result) that crosses its edges and solve again, so no plant is cut by the zone.
    for _attempt in range(4):
        roots = {r: v.copy() for r, v in roots0.items()}
        pos = {r: v.copy() for r, v in pos0.items()}
        infos = infos0
        offsets = {r: np.zeros((bm.num_frames, 3)) for r in bm.actors}
        iters: dict[str, list[np.ndarray]] = {r: [] for r in bm.actors}
        used_masks: dict[str, list[np.ndarray]] = {r: [] for r in bm.actors}
        replayed = {r: True for r in bm.actors}
        fixed_out = None
        if zone is not None:
            fixed_out = np.ones(bm.num_frames, dtype=bool)
            fixed_out[zone[0] : zone[1]] = False
        if params.root_correction:
            for it in range(max(1, params.root_iterations)):
                for r, a in bm.actors.items():
                    used_masks[r].append(np.stack([infos[r].masks[c] for c in chans]))
                    if still[r]:
                        iters[r].append(offsets[r].copy())
                        continue
                    fixed = None
                    replay = False
                    if frozen is not None and fixed_out is not None:
                        prev_it = np.asarray(frozen.offset_iters[r], dtype=np.float64)[it]
                        fixed = (fixed_out, prev_it - offsets[r])
                        pm = None if frozen.mask_iters is None else frozen.mask_iters.get(r)
                        replay = pm is not None and np.array_equal(np.asarray(pm)[it], used_masks[r][-1])
                    if replay and fixed is not None:  # the same data as the previous run: its solution
                        d = prev_it - offsets[r]
                    else:
                        replayed[r] = False
                        geo = camera_geometry(bm, ActorMotion(a.rot, roots[r]), camera)
                        d = solve_root_correction(
                            sks[r], roles, pos[r], infos[r].masks, floor, params, geo, fps, offsets[r], fixed
                        )
                    if fixed is not None:  # exactly the previous offsets outside the zone
                        d[fixed_out] = fixed[1][fixed_out]
                    offsets[r] += d
                    roots[r] = roots[r] + d
                    pos[r] = forward_kinematics(sks[r], a.rot, roots[r])[0]
                    iters[r].append(offsets[r].copy())
                infos = detect(final=True)
        if frozen is None or zone is None:
            break
        grown = frozen_zone(
            zone, [i.masks for i in infos.values()] + list(frozen.masks.values()), bm.num_frames,
            frozen.max_extend + max(0, frozen.editable[0] - zone[0], zone[1] - frozen.editable[1]),
        )  # fmt: skip
        grown = (
            max(grown[0], frozen.editable[0] - frozen.max_extend),
            min(grown[1], frozen.editable[1] + frozen.max_extend),
        )
        if grown == zone:
            break
        zone = grown
    if frozen is not None and zone is not None and params.root_correction:
        # the change of the root offset eases to 0 at the zone edges (no acceleration seam there)
        t = np.arange(bm.num_frames)
        u = np.ones(bm.num_frames)
        if zone[0] > 0:
            u = np.minimum(u, np.clip((t - zone[0] + 1) / (ZONE_TAPER + 1.0), 0.0, 1.0))
        if zone[1] < bm.num_frames:
            u = np.minimum(u, np.clip((zone[1] - t) / (ZONE_TAPER + 1.0), 0.0, 1.0))
        taper = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
        taper[fixed_out if fixed_out is not None else np.zeros(bm.num_frames, bool)] = 0.0
        for r in bm.actors:
            if replayed[r] or still[r] or not iters[r]:
                continue
            prev_final = np.asarray(frozen.offset_iters[r], dtype=np.float64)[-1]
            offsets[r] = prev_final + (offsets[r] - prev_final) * taper[:, None]
            iters[r][-1] = offsets[r].copy()
            roots[r] = roots0[r] + offsets[r]

    in_ch = list(bm.contact_channels)
    used = consumed_channels(roles, in_ch)
    passthrough = [c for c in in_ch if c not in used and c not in {k.channel for k in roles.contacts}]
    out_channels = [c.channel for c in roles.contacts] + passthrough
    # PM convention: after the contacts stage the floor is y = 0 (export assumes it). Detection
    # does not depend on a vertical translation, so the shot is shifted here, before the fix.
    shift = 0.0
    if zero_shift is not None:
        shift = float(zero_shift)
        floor = Floor(floor.normal, floor.offset - floor.normal[1] * shift)
    elif params.floor_to_zero:
        shift, floor = floor_zero_shift(floor, list(roots.values()))
    up_shift = np.array([0.0, shift, 0.0])
    actors = {}
    for r, a in bm.actors.items():
        info = infos[r]
        info.root_offset = offsets[r]
        info.offset_iters = np.stack(iters[r]) if iters[r] else np.zeros((0, bm.num_frames, 3))
        info.replayed = bool(frozen is not None and replayed.get(r, False))
        info.mask_iters = (
            np.stack(used_masks[r]) if used_masks[r] else np.zeros((0, len(chans), bm.num_frames), bool)
        )
        info.fix = apply_foot_targets(
            sks[r], roles, a.rot, roots[r] + up_shift, info.masks, floor, params, fps=fps
        )
        first = {ch: m.copy() for ch, m in info.masks.items()}
        info.capped_plants = info.fix.capped_plants
        forced = {ch: np.zeros(bm.num_frames, dtype=bool) for ch in info.masks}
        for ch in forced:  # painted plants are kept as painted
            for i0, i1, plant in paints[r].get(ch, []):
                forced[ch][i0:i1] = plant

        # A pin the leg cannot reach (after the capped pelvis drop) is not a plant: unplant those
        # frames (dropping pieces that become too short) and solve again. Pins are re-derived from the
        # remaining frames, so a re-solve can miss elsewhere: up to three re-solves; frames
        # that still miss after the last one are unplanted without another solve (the motion keeps
        # that solve; those frames are just no longer flagged as planted). Painted plants stay.
        for _ in range(3):
            bad = _missed_frames(info.fix, info.masks, forced, params)
            if not any(b.any() for b in bad.values()):
                break
            info.miss_unplanted_frames += _unplant(info.masks, bad, forced, params)
            info.fix = apply_foot_targets(
                sks[r], roles, a.rot, roots[r] + up_shift, info.masks, floor, params, fps=fps
            )
        else:
            bad = _missed_frames(info.fix, info.masks, forced, params)
            info.unsolved_unplanted_frames = _unplant(info.masks, bad, forced, params)
        info.slide_trimmed_frames = _trim_sliding_edges(
            sks[r], roles, info.fix, info.masks, forced, fps, params
        )
        free_paint = {ch: np.zeros(bm.num_frames, dtype=bool) for ch in info.masks}
        for ch in free_paint:
            for i0, i1, plant in paints[r].get(ch, []):
                free_paint[ch][i0:i1] = not plant
        info.held_holes_filled = _fill_held_holes(
            sks[r], roles, info.fix, info.masks, free_paint, fps, params
        )
        info.raised_plants = info.fix.raised_plants
        info.dropped_plants = sum(
            int(not info.masks[ch][s0:e0].any()) for ch, m in first.items() for s0, e0 in segments(m)
        )
        cols = [info.masks[c.channel].astype(np.float32) for c in roles.contacts]
        if a.contact is not None:
            cols += [np.asarray(a.contact[:, in_ch.index(c)], dtype=np.float32) for c in passthrough]
        else:
            cols += [np.zeros(bm.num_frames, np.float32) for _ in passthrough]
        prior = (
            None
            if a.root_prior is None
            else (np.asarray(a.root_prior, np.float64) + up_shift).astype(np.float32)
        )
        actors[r] = ActorMotion(
            rot=info.fix.rot.astype(np.float32),
            root=info.fix.root.astype(np.float32),
            conf=a.conf,
            contact=np.stack(cols, axis=1) if cols else None,
            rest_offsets=a.rest_offsets,
            root_prior=prior,
        )
    ws = [float(bm.world_shift[0]), float(bm.world_shift[1]) + shift, float(bm.world_shift[2])]
    out = _with_actors(bm, actors, contact_channels=out_channels, world_shift=ws)
    return out, ContactReport(floor, infos, shift, filled, zone)


def contact_masks(
    bm: BodyMotion, actor: ActorMotion, roles: BodyRoles, threshold: float = 0.5
) -> dict[str, np.ndarray]:
    """Plant masks per foot channel read back from a contacts-stage result."""
    out = {}
    for c in roles.contacts:
        if actor.contact is not None and c.channel in bm.contact_channels:
            out[c.channel] = actor.contact[:, bm.contact_channels.index(c.channel)] > threshold
        else:
            out[c.channel] = np.zeros(bm.num_frames, dtype=bool)
    return out


def hold_contacts(
    bm: BodyMotion,
    rot_by_role: dict[str, np.ndarray],
    root_by_role: dict[str, np.ndarray],
    roles: BodyRoles,
    params: ContactParams | None = None,
    floor: Floor | None = None,
    ref_fk: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, FootFix]:
    """Re-pin the plants of a contacts-stage result ``bm`` after another stage changed the motion
    (smooth, retime): leg IK only, onto ``bm``'s own planted joints frame by frame (or onto
    ``ref_fk[role]`` = reference positions/world rotations), no root moves except a pelvis drop
    (capped, smoothed) where a planted target is out of the leg's reach (smoothed hips can put a
    straight stance leg's hip a few mm too far: the pinned foot then stepped 1-2 mm), exact floor
    clamp on y = 0 (the post-contacts convention). Runs for every fighter, plants or not."""
    params = params or ContactParams()
    floor = floor or Floor.horizontal(0.0)
    out = {}
    for r, a in bm.actors.items():
        sk = actor_skeleton(bm, a)
        masks = contact_masks(bm, a, roles)
        out[r] = apply_foot_targets(
            sk,
            roles,
            rot_by_role[r],
            root_by_role[r],
            masks,
            floor,
            params,
            ref=(np.asarray(a.rot, np.float64), np.asarray(a.root, np.float64)),
            allow_pelvis=True,  # with pin_mode "ref": only for targets out of reach
            soft_floor=False,
            pin_mode="ref",
            ref_fk=None if ref_fk is None else ref_fk.get(r),
        )
    return out


def floor_from_contacts(bm: BodyMotion, roles: BodyRoles, params: ContactParams | None = None) -> Floor:
    """Re-derive the shared floor from a contacts-stage result (planted sole points)."""
    params = params or ContactParams()
    up = np.asarray(roles.up, dtype=np.float64)
    pts, low_all = [], []
    for a in bm.actors.values():
        sk = actor_skeleton(bm, a)
        pos = forward_kinematics(sk, a.rot, a.root)[0]
        spts, low = sole_points(sk, roles, pos, up)
        low_all.append(low)
        masks = contact_masks(bm, a, roles)
        for ci, c in enumerate(roles.contacts):
            pts.append(spts[masks[c.channel], ci])
    pts = [p for p in pts if len(p)]
    src = np.concatenate(pts) if pts else np.concatenate(low_all)
    return fit_floor(src, None, params.floor_tilt, params.max_tilt_deg)
