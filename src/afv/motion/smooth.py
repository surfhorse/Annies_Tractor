"""Strike-aware motion smoothing (stage ``smooth``).

Design
------
**Sliding-window least squares in a local tangent space.** For every frame ``t`` and joint, the
rotations of the window ``t-k .. t+k`` are mapped into the tangent space at the frame's own
rotation (``r_s = log(q_t^-1 q_s)``), where rotations behave like vectors and no global
unwrapping is needed (no singularities as long as a window turns < 180 degrees). The smoothed
tangent trajectory minimises::

    sum_s c_s |x_s - r_s|^2 + sum_s lv_s |x_{s+1} - x_s|^2 + sum_s la_s |x_{s+1} - 2 x_s + x_{s-1}|^2

and the centre value is mapped back (``q_t' = q_t exp(x_t)``). This is a local Whittaker/Tikhonov
smoother: the acceleration term removes jitter, the velocity term damps drift, and because the
weights are per frame they can vary in time and per joint: confidence lowers ``c`` (low-confidence
frames are filled by their neighbours), strike easing lowers ``lv``/``la`` around strike peaks, and
per-region strength scales them per joint. Each window is a small dense SPD system solved in
batches (only the centre row of the inverse is needed). The root translation uses the same solver
directly on positions.

**Strength** is a time-scale multiplier: ``la = lambda_acc * s^4``, ``lv = lambda_vel * s^2`` (the
cut-off frequency of this smoother scales as ``la^-1/4``), so strength 2 roughly halves the cut-off.
The window grows with it (``half_window x max(1, s)``, up to 28 frames, i.e. strength 4), so the
global slider and each per-region slider (``region_strength``: torso/head/arms/hands/legs/root,
0 = that region passes through) smooth monotonically more as they go up.

**Pops.** A single-frame pop goes out and comes back: frame ``t`` is far from the midpoint of its
neighbours, while the constant-velocity predictions from the left (``2 q[t-1] - q[t-2]``) and from
the right (``2 q[t+1] - q[t+2]``) *agree* with each other. A strike or a sudden start also has a
large midpoint deviation, but there the two predictions disagree (the velocity really changed).
A pop is flagged when the deviation exceeds ``pop_angle_deg``, ``pop_ratio`` x the left/right
disagreement and ``pop_isolation`` x the deviation two frames away (on a parabola, i.e. sustained
acceleration, the two predictions agree although nothing popped). Pop frames get data weight 0
and are replaced by the smooth fit (NFR-4: no single-frame pops).

**Strikes.** Peaks of wrist/ankle speed relative to the pelvis above ``strike_min_speed`` (wrists)
or ``kick_min_speed`` (ankles), prominent against a *short* local baseline (the higher of the two
adjacent valleys within ``strike_valley_frames``, so every punch of a flurry counts; a long median
would hide them), away from the clip ends, *straight* (net/path over +-2 frames, which rejects
out-and-back pops) and *long* enough (rejects jitter) are strikes. Around each peak the smoothing
of the striking limb is eased to ``strike_ease`` x and its torso ancestors to
``strike_ease_torso`` x within ``strike_hold_s`` of the peak, returning to full strength over
``strike_taper_s`` with a raised cosine; only strong, straight strikes also ease the root. Every
frame where an effector itself is faster than its strike threshold (after pop repair) eases its limb
the same way, since inside a flurry the speed stays high between punches. So peak
speed survives (NFR-4: >= 80% of unsmoothed speed) and the strength change itself is smooth.

**Contacts held.** Smoothing moves feet, so planted channels from the input (the contacts stage
result) are re-pinned afterwards with the contacts stage's leg IK onto the input's own pins, and
every foot is kept above the y = 0 floor. Only the legs move: no new root or pelvis offsets, so the
stage is idempotent (smoothing at strength 0 returns its input).

**Pinned poses** (``edits/pins.json``, :mod:`afv.motion.pins`) are applied after smoothing and
before the contacts are re-pinned: a C1 displacement over +-radius frames makes the motion pass
exactly through each pinned export-rig pose.

**One-Euro fallback** (``method="one_euro"``): the classic adaptive low-pass filter, applied
forward then backward (zero phase), on root positions and on rotations in the tangent space of the
previous filtered value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.ndimage import distance_transform_edt

from afv.core.types import ActorMotion, BodyMotion, Skeleton
from afv.motion.contacts import (
    ContactParams,
    actor_skeleton,
    hold_contacts,
    sanitize_motion,
)
from afv.motion.fk import forward_kinematics
from afv.motion.ik import REGIONS, BodyRoles, get_roles
from afv.motion.pins import Pin, PinReport, apply_pins, pin_errors
from afv.motion.rotations import make_continuous, qconj, qmul


class SmoothParams(BaseModel):
    """Parameters of the smoother (also the ``smooth`` stage params)."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["window", "one_euro"] = "window"
    strength: float = 1.0
    region_strength: dict[str, float] = Field(default_factory=dict)  # torso/head/arms/hands/legs/root
    half_window: int = 7
    lambda_vel: float = 0.2
    lambda_acc: float = 10.0
    root_lambda_vel: float = 0.2
    root_lambda_acc: float = 10.0
    use_confidence: bool = True
    pop_angle_deg: float = 6.0
    pop_root_m: float = 0.03
    pop_ratio: float = 1.5  # deviation / left-right disagreement
    pop_isolation: float = 2.5  # deviation / deviation two frames away
    strike_min_speed: float = 2.5  # m/s, wrists (punches), relative to the pelvis
    kick_min_speed: float = 4.0  # m/s, ankles (kicks), relative to the pelvis; lower keeps leg jitter
    strike_valley_frames: int = 6  # local baseline: the lowest speed within this many frames each side
    strike_prominence: float = 1.5  # peak / higher of the two adjacent valleys
    strike_edge_frames: int = 4  # peaks this close to the clip ends are ignored
    strike_straightness: float = 0.5  # net / path over +-2 frames (out-and-back pops are not strikes)
    strike_min_travel: float = 0.12  # m over +-2 frames, wrists
    kick_min_travel: float = 0.25  # m over +-2 frames, ankles
    root_ease_min_speed: float = 5.0  # only strong, straight strikes also ease the root
    root_ease_straightness: float = 0.8
    strike_ease: float = 0.01  # limb joints of the striking chain
    strike_ease_torso: float = 0.1  # pelvis/spine ancestors and the root (they carry other limbs too)
    strike_hold_s: float = 0.10  # full easing within +- this of a peak
    strike_taper_s: float = 0.10  # then a raised-cosine return to full smoothing
    hold_contacts: bool = True
    contact: ContactParams = Field(default_factory=ContactParams)
    one_euro_min_cutoff: float = 1.5  # Hz
    one_euro_beta: float = 0.3
    one_euro_d_cutoff: float = 1.0

    @field_validator("region_strength")
    @classmethod
    def _known_regions(cls, v: dict[str, float]) -> dict[str, float]:
        known = (*REGIONS, "root")
        bad = sorted(k for k in v if k not in known)
        if bad:
            raise ValueError(f"unknown smoothing region(s) {bad}; regions are {list(known)}")
        neg = sorted(k for k, x in v.items() if not np.isfinite(x) or x < 0.0)
        if neg:
            raise ValueError(f"region strength must be a finite number >= 0 (regions {neg})")
        return v


# ---------------------------------------------------------------------------
# Tangent-space helpers
# ---------------------------------------------------------------------------


def qlog(q: np.ndarray) -> np.ndarray:
    """Rotation vector of unit xyzw quaternions (shortest arc)."""
    q = np.asarray(q, dtype=np.float64)
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    v = q[..., :3]
    s = np.linalg.norm(v, axis=-1)
    ang = 2.0 * np.arctan2(s, q[..., 3])
    scale = np.where(s > 1e-12, ang / np.maximum(s, 1e-12), 2.0)
    return v * scale[..., None]


def qexp(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    ang = np.linalg.norm(r, axis=-1)
    scale = np.where(ang > 1e-12, np.sin(0.5 * ang) / np.maximum(ang, 1e-12), 0.5)
    return np.concatenate([r * scale[..., None], np.cos(0.5 * ang)[..., None]], axis=-1)


# ---------------------------------------------------------------------------
# Windowed solver
# ---------------------------------------------------------------------------


def _window_index(n_frames: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_frames)[:, None] + np.arange(-k, k + 1)[None, :]
    valid = (idx >= 0) & (idx < n_frames)
    return np.clip(idx, 0, n_frames - 1), valid


def window_solve(
    r: np.ndarray, c: np.ndarray, lv: np.ndarray, la: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Centre values of the per-window least-squares problems (see module docstring).

    ``r`` [F,K,M,D] window values, ``c`` [F,K,M] data weights, ``lv`` [F,K,M] and ``la`` [F,K,M]
    per-frame smoothness weights (absolute-frame values gathered into each window), ``valid``
    [F,K] whether a window slot is inside the clip. Returns [F,M,D].
    """
    n_frames, n_k, n_m, n_d = r.shape
    k = n_k // 2
    eye = np.eye(n_k)
    d1 = np.diff(eye, axis=0)
    d2 = np.diff(eye, n=2, axis=0)
    v1 = (valid[:, :-1] & valid[:, 1:])[:, :, None]
    v2 = (valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:])[:, :, None]
    w_v = lv[:, :-1] * v1
    w_a = la[:, 1:-1] * v2
    w_c = np.where(valid[:, :, None], c, 1e-6) + 1e-9
    out = np.empty((n_frames, n_m, n_d))
    chunk = max(1, (6000 * 225) // max(n_m * n_k * n_k, 1))  # ~11 MB of systems per batch
    e = np.zeros((n_k, 1))
    e[k] = 1.0
    for f0 in range(0, n_frames, chunk):
        sl = slice(f0, min(f0 + chunk, n_frames))
        cc = w_c[sl].transpose(0, 2, 1).reshape(-1, n_k)
        vv = w_v[sl].transpose(0, 2, 1).reshape(-1, n_k - 1)
        aa = w_a[sl].transpose(0, 2, 1).reshape(-1, n_k - 2)
        a_mat = (d1.T[None] * vv[:, None, :]) @ d1 + (d2.T[None] * aa[:, None, :]) @ d2
        a_mat[:, np.arange(n_k), np.arange(n_k)] += cc
        y = np.linalg.solve(a_mat, np.broadcast_to(e, (a_mat.shape[0], n_k, 1)))[..., 0]
        rr = r[sl].transpose(0, 2, 1, 3).reshape(-1, n_k, n_d)
        out[sl] = np.einsum("nk,nkd->nd", y * cc, rr).reshape(-1, n_m, n_d)
    return out


def smooth_rotations(
    q: np.ndarray, c: np.ndarray, lam_v: np.ndarray, lam_a: np.ndarray, k: int
) -> np.ndarray:
    """Windowed tangent-space smoothing of rotations [F,J,4] with per-frame/joint weights [F,J]."""
    n_frames = q.shape[0]
    if n_frames < 3 or k < 1:
        return q.copy()
    idx, valid = _window_index(n_frames, k)
    qw = q[idx]  # [F,K,J,4]
    r = qlog(qmul(qconj(q[:, None]), qw))
    # A window that turns more than 180 degrees (a spin at > ~26 deg/frame over 15 frames) wraps
    # the log map; unwrap outward from the centre so the tangent trajectory stays continuous.
    for o, prev in [(o, o - 1) for o in range(k + 1, 2 * k + 1)] + [(o, o + 1) for o in range(k - 1, -1, -1)]:
        cur, ref = r[:, o], r[:, prev]
        ang = np.linalg.norm(cur, axis=-1, keepdims=True)
        alt = cur - 2.0 * np.pi * cur / np.maximum(ang, 1e-12)
        use = np.linalg.norm(alt - ref, axis=-1) < np.linalg.norm(cur - ref, axis=-1)
        r[:, o] = np.where(use[..., None], alt, cur)
    r[~valid] = 0.0
    x = window_solve(r, c[idx], lam_v[idx], lam_a[idx], valid)
    return make_continuous(qmul(q, qexp(x)))


def smooth_vectors(p: np.ndarray, c: np.ndarray, lam_v: np.ndarray, lam_a: np.ndarray, k: int) -> np.ndarray:
    """Windowed smoothing of vectors [F,M,D] with weights [F,M]."""
    n_frames = p.shape[0]
    if n_frames < 3 or k < 1:
        return p.copy()
    idx, valid = _window_index(n_frames, k)
    r = p[idx] - p[:, None]
    r[~valid] = 0.0
    return p + window_solve(r, c[idx], lam_v[idx], lam_a[idx], valid)


# ---------------------------------------------------------------------------
# Pops and strikes
# ---------------------------------------------------------------------------


def _qangle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return 2.0 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0))


def _pop_test(
    dev: np.ndarray, cons: np.ndarray, threshold: float, ratio: float, iso_ratio: float
) -> np.ndarray:
    """``dev`` [F,...] midpoint deviation, ``cons`` [F,...] left/right extrapolation disagreement
    (both 0 on the two edge frames). A pop is large, inconsistent with its neighbours' motion and
    isolated (``iso_ratio`` x the deviation two frames away: sustained acceleration deviates about
    equally there, and on a parabola the two extrapolations agree although nothing popped)."""
    pad = np.zeros((2, *dev.shape[1:]))
    d = np.concatenate([pad, dev, pad])
    near = np.maximum(d[:-4], d[4:])
    return (dev > threshold) & (dev > ratio * cons) & (dev > iso_ratio * near)


def rotation_spikes(
    q: np.ndarray, angle_deg: float = 6.0, ratio: float = 1.5, iso_ratio: float = 2.5
) -> np.ndarray:
    """Single-frame rotation pops [F,J] (see module docstring). ``q`` must be continuous."""
    dev = np.zeros(q.shape[:-1])
    cons = np.zeros(q.shape[:-1])
    if len(q) >= 5:
        dev[2:-2] = _qangle(q[2:-2], q[1:-3] + q[3:-1])
        cons[2:-2] = _qangle(2.0 * q[1:-3] - q[:-4], 2.0 * q[3:-1] - q[4:])
    return _pop_test(dev, cons, np.radians(angle_deg), ratio, iso_ratio)


def position_spikes(
    p: np.ndarray, threshold: float = 0.02, ratio: float = 1.5, iso_ratio: float = 2.5
) -> np.ndarray:
    """Single-frame position pops [F,...] of positions [F,...,3] (same test as rotations)."""
    dev = np.zeros(p.shape[:-1])
    cons = np.zeros(p.shape[:-1])
    if len(p) >= 5:
        dev[2:-2] = np.linalg.norm(p[2:-2] - 0.5 * (p[1:-3] + p[3:-1]), axis=-1)
        cons[2:-2] = np.linalg.norm((2.0 * p[1:-3] - p[:-4]) - (2.0 * p[3:-1] - p[4:]), axis=-1)
    return _pop_test(dev, cons, threshold, ratio, iso_ratio)


@dataclass
class Strike:
    effector: str
    frame: int  # index into the motion's frames
    speed: float  # m/s, unsmoothed
    strong: bool = False  # fast and straight: also eases the root


def effector_speed(p: np.ndarray, fps: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if len(p) < 2:
        return np.zeros(len(p))
    return np.linalg.norm(np.gradient(p, axis=0), axis=-1) * fps


def detect_strikes(
    positions: dict[str, np.ndarray],
    fps: float,
    params: SmoothParams | None = None,
    min_speed: dict[str, float] | None = None,
    min_travel: dict[str, float] | None = None,
) -> list[Strike]:
    """Strike peaks of end-effector speed (see module docstring). ``positions``: name -> [F,3]
    (normally relative to the pelvis); ``min_speed``/``min_travel`` override the wrist defaults per
    effector.

    A peak counts when it is a local maximum (+-2 frames), at least ``strike_edge_frames`` from the
    clip ends, faster than the minimum speed, ``strike_prominence`` x faster than the *higher* of
    the two adjacent valleys (lowest speed within ``strike_valley_frames`` each side: a short local
    baseline, so every punch of a flurry counts), and straight and long enough over +-2 frames.
    """
    params = params or SmoothParams()
    out: list[Strike] = []
    edge = max(2, params.strike_edge_frames)
    vw = max(1, params.strike_valley_frames)
    for name, p in positions.items():
        vmin = (min_speed or {}).get(name, params.strike_min_speed)
        tmin = (min_travel or {}).get(name, params.strike_min_travel)
        n_frames = len(p)
        if n_frames < 2 * edge + 1:
            continue
        raw = effector_speed(p, fps)
        # Decide on a lightly filtered path ([1,2,1]/4): jitter must neither hide a peak nor
        # inflate the path length of the straightness test. The reported speed stays raw.
        pe = np.concatenate([p[:1], p, p[-1:]])
        p = 0.25 * pe[:-2] + 0.5 * pe[1:-1] + 0.25 * pe[2:]
        sp = effector_speed(p, fps)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=-1)
        csum = np.concatenate([[0.0], np.cumsum(seg)])
        for t in range(edge, n_frames - edge):
            if sp[t] < vmin or sp[t] < sp[max(0, t - 2) : t + 3].max():
                continue
            left = sp[max(0, t - vw) : t].min()
            right = sp[t + 1 : t + 1 + vw].min()
            if sp[t] < params.strike_prominence * max(left, right):
                continue
            path = csum[t + 2] - csum[t - 2]
            net = np.linalg.norm(p[t + 2] - p[t - 2])
            if path <= 0 or net < params.strike_straightness * path or net < tmin:
                continue
            v = float(raw[t - 1 : t + 2].max())
            strong = v >= params.root_ease_min_speed and net >= params.root_ease_straightness * path
            out.append(Strike(name, int(t), v, strong))
    return out


def strike_ease_map(
    sk: Skeleton,
    strikes: list[Strike],
    n_frames: int,
    fps: float,
    params: SmoothParams,
    regions: list[str] | None = None,
    fast: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothing multipliers [F,J] for joints and [F] for the root, dipping at strike peaks.

    Limb joints of the striking chain dip to ``strike_ease``; its torso ancestors (``regions``
    "torso"/"head") and the root only to ``strike_ease_torso``, since they also carry the head and
    the other limbs, whose jitter would otherwise show through during the strike."""
    ease = np.ones((n_frames, sk.num_joints))
    root = np.ones(n_frames)
    hold = params.strike_hold_s * fps
    taper = max(1.0, params.strike_taper_s * fps)
    t = np.arange(n_frames)
    regions = regions or ["arms"] * sk.num_joints

    def profile(depth: float, frame: int) -> np.ndarray:
        u = np.clip((np.abs(t - frame) - hold) / taper, 0.0, 1.0)
        return depth + (1.0 - depth) * (0.5 - 0.5 * np.cos(np.pi * u))

    def apply(effector: str, limb: np.ndarray, torso: np.ndarray) -> None:
        j = sk.index(effector)
        while j >= 0:
            prof = torso if regions[j] in ("torso", "head") else limb
            ease[:, j] = np.minimum(ease[:, j], prof)
            j = sk.parents[j]

    for s in strikes:
        limb, torso = profile(params.strike_ease, s.frame), profile(params.strike_ease_torso, s.frame)
        apply(s.effector, limb, torso)
        if s.strong:
            root = np.minimum(root, torso)
    # Every fast frame of an effector eases its limb too (a flurry keeps the speed up between
    # punches, so not every punch is a prominent peak).
    for name, mask in (fast or {}).items():
        if not np.any(mask):
            continue
        dist = distance_transform_edt(~np.asarray(mask, dtype=bool))
        u = np.clip((dist - hold) / taper, 0.0, 1.0)
        shape = 0.5 - 0.5 * np.cos(np.pi * u)
        apply(
            name,
            params.strike_ease + (1.0 - params.strike_ease) * shape,
            params.strike_ease_torso + (1.0 - params.strike_ease_torso) * shape,
        )
    return ease, root


# ---------------------------------------------------------------------------
# One-Euro fallback
# ---------------------------------------------------------------------------


def _alpha(cutoff: np.ndarray, fps: float) -> np.ndarray:
    return 1.0 / (1.0 + fps / (2.0 * np.pi * cutoff))


def one_euro_vectors(
    x: np.ndarray, fps: float, min_cutoff: float, beta: float, d_cutoff: float
) -> np.ndarray:
    """Zero-phase One-Euro filter over axis 0 of [F,...,D] (norm over the last axis)."""

    def run(v: np.ndarray) -> np.ndarray:
        out = np.empty_like(v)
        xh = v[0].copy()
        dxh = np.zeros_like(xh)
        out[0] = xh
        ad = _alpha(np.asarray(d_cutoff), fps)
        for i in range(1, len(v)):
            dxh = dxh + ad * ((v[i] - xh) * fps - dxh)
            a = _alpha(min_cutoff + beta * np.linalg.norm(dxh, axis=-1, keepdims=True), fps)
            xh = xh + a * (v[i] - xh)
            out[i] = xh
        return out

    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return x.copy()
    return run(run(x)[::-1])[::-1]


def one_euro_rotations(
    q: np.ndarray, fps: float, min_cutoff: float, beta: float, d_cutoff: float
) -> np.ndarray:
    """Zero-phase One-Euro on rotations [F,J,4] in the tangent space of the previous output."""

    def run(v: np.ndarray) -> np.ndarray:
        out = np.empty_like(v)
        qh = v[0].copy()
        dxh = np.zeros((*v.shape[1:-1], 3))
        out[0] = qh
        ad = _alpha(np.asarray(d_cutoff), fps)
        for i in range(1, len(v)):
            delta = qlog(qmul(qconj(qh), v[i]))
            dxh = dxh + ad * (delta * fps - dxh)
            a = _alpha(min_cutoff + beta * np.linalg.norm(dxh, axis=-1, keepdims=True), fps)
            qh = qmul(qh, qexp(a * delta))
            out[i] = qh
        return out

    q = make_continuous(q)
    if len(q) < 2:
        return q.copy()
    return make_continuous(run(run(q)[::-1])[::-1])


# ---------------------------------------------------------------------------
# Actor / shot drivers
# ---------------------------------------------------------------------------


@dataclass
class SmoothInfo:
    strikes: list[Strike] = field(default_factory=list)
    rotation_pops: int = 0
    root_pops: int = 0
    pins: list[PinReport] = field(default_factory=list)


def _end_weights(n_frames: int, boost: tuple[float, ...] = (1e4, 30.0, 3.0)) -> np.ndarray:
    """Data-weight multipliers [F] holding the first and last frames to the input (1 elsewhere)."""
    w = np.ones(n_frames)
    for i, b in enumerate(boost[: max(0, n_frames // 2)]):
        w[i] = max(w[i], b)
        w[n_frames - 1 - i] = max(w[n_frames - 1 - i], b)
    return w


MAX_HALF_WINDOW = 28  # frames (strength 4 at the default half window); stronger saturates


def half_windows(params: SmoothParams, strength: np.ndarray, n_frames: int) -> np.ndarray:
    """Half window per strength: ``half_window`` x max(1, strength), capped (see ``smooth_actor``)."""
    k = np.ceil(params.half_window * np.maximum(1.0, np.asarray(strength, np.float64)) - 1e-9)
    cap = max(params.half_window, min(MAX_HALF_WINDOW, n_frames - 1))
    return np.clip(k, params.half_window, cap).astype(int)


def smooth_actor(
    sk: Skeleton,
    roles: BodyRoles,
    actor: ActorMotion,
    fps: float,
    params: SmoothParams | None = None,
) -> tuple[ActorMotion, SmoothInfo]:
    """Smooth one fighter (no contact handling; :func:`smooth_motion` re-pins afterwards)."""
    params = params or SmoothParams()
    q = make_continuous(np.asarray(actor.rot, dtype=np.float64))
    root = np.asarray(actor.root, dtype=np.float64)
    n_frames, n_j = q.shape[:2]
    info = SmoothInfo()

    pops = rotation_spikes(q, params.pop_angle_deg, params.pop_ratio, params.pop_isolation)
    rpops = position_spikes(root, params.pop_root_m, params.pop_ratio, params.pop_isolation)
    info.rotation_pops, info.root_pops = int(pops.sum()), int(rpops.sum())

    # Strike detection on the pop-repaired motion so pops cannot pose as strikes.
    q_rep = q.copy()
    if pops.any():
        mid = np.zeros_like(q)
        mid[1:-1] = q[:-2] + q[2:]
        mid[1:-1] /= np.linalg.norm(mid[1:-1], axis=-1, keepdims=True)
        q_rep[pops] = mid[pops]
    root_rep = root.copy()
    if rpops.any():
        rmid = root.copy()
        rmid[1:-1] = 0.5 * (root[:-2] + root[2:])
        root_rep[rpops] = rmid[rpops]
    pos_rep, _ = forward_kinematics(sk, q_rep, root_rep)
    # Effectors relative to the pelvis: walking or stepping translation is not a strike.
    pel = pos_rep[:, sk.index(roles.pelvis)]
    eff = {n: pos_rep[:, sk.index(n)] - pel for n in (*roles.wrists, *roles.ankles)}
    vmin = {
        **{n: params.strike_min_speed for n in roles.wrists},
        **{n: params.kick_min_speed for n in roles.ankles},
    }
    tmin = {
        **{n: params.strike_min_travel for n in roles.wrists},
        **{n: params.kick_min_travel for n in roles.ankles},
    }
    info.strikes = detect_strikes(eff, fps, params, vmin, tmin)
    fast = {}
    for n, pth in eff.items():
        pe = np.concatenate([pth[:1], pth, pth[-1:]])
        sp = effector_speed(0.25 * pe[:-2] + 0.5 * pe[1:-1] + 0.25 * pe[2:], fps)
        fast[n] = sp >= vmin[n]

    if params.method == "one_euro":
        args = (fps, params.one_euro_min_cutoff, params.one_euro_beta, params.one_euro_d_cutoff)
        q_s = one_euro_rotations(q_rep, *args)
        root_s = one_euro_vectors(root_rep, *args)
    else:
        regions = roles.regions(sk)
        ease, ease_root = strike_ease_map(sk, info.strikes, n_frames, fps, params, regions, fast)
        reg = np.array([params.region_strength.get(r, 1.0) for r in regions], dtype=np.float64)
        s = params.strength * reg[None, :] * np.ones((n_frames, 1))
        c = np.ones((n_frames, n_j))
        if params.use_confidence and actor.conf is not None:
            c = np.clip(np.asarray(actor.conf, dtype=np.float64), 0.05, 1.0)
        c = np.where(pops, 0.0, c)
        # Clip ends: a window there has neighbours on one side only, so the end frames were pulled along
        # the trend (feet moved 6-9 cm on frame 0). They are held to the input, easing in over 3 frames.
        ends = _end_weights(n_frames)
        c = c * ends[:, None]
        lv, la = params.lambda_vel * s**2 * ease, params.lambda_acc * s**4 * ease
        # The window must grow with the strength (time scale), or a strong slider saturates and
        # stops smoothing more (the per-region sliders must be monotone): joints are solved in
        # groups of equal half window; strength <= 1 keeps ``half_window``.
        k_joint = half_windows(params, params.strength * reg, n_frames)
        q_s = np.empty_like(q)
        for k in np.unique(k_joint):
            sel = k_joint == k
            q_s[:, sel] = smooth_rotations(q[:, sel], c[:, sel], lv[:, sel], la[:, sel], int(k))
        rs = params.strength * params.region_strength.get("root", 1.0)
        cr = (np.where(rpops, 0.0, 1.0) * ends)[:, None]
        root_s = smooth_vectors(
            root[:, None],
            cr,
            (params.root_lambda_vel * rs**2 * ease_root)[:, None],
            (params.root_lambda_acc * rs**4 * ease_root)[:, None],
            int(half_windows(params, np.array([rs]), n_frames)[0]),
        )[:, 0]

    out = ActorMotion(
        rot=make_continuous(q_s),
        root=root_s,
        conf=actor.conf,
        contact=actor.contact,
        rest_offsets=actor.rest_offsets,
        root_prior=actor.root_prior,
    )
    return out, info


def smooth_motion(
    bm: BodyMotion,
    params: SmoothParams | None = None,
    roles: BodyRoles | None = None,
    pins: dict[str, list[Pin]] | None = None,
) -> tuple[BodyMotion, dict[str, SmoothInfo]]:
    """Smooth every fighter of ``bm`` (normally the contacts-stage result), pass through the
    pinned poses ``pins`` (``afv.motion.pins``: {role: pins}, frame indices into ``bm``) and hold
    its contacts."""
    from afv.core.skeletons import get_skeleton

    params = params or SmoothParams()
    bm, _ = sanitize_motion(bm)
    fps = bm.fps or 30.0
    base = get_skeleton(bm.skeleton)
    roles = roles or get_roles(base)
    actors, infos = {}, {}
    rigs: dict = {}
    for r, a in bm.actors.items():
        actors[r], infos[r] = smooth_actor(actor_skeleton(bm, a), roles, a, fps, params)
        if pins and pins.get(r):
            # Pins after smoothing, before the re-pin: planted legs stay on their plants.
            actors[r].rot, actors[r].root, infos[r].pins = apply_pins(
                base, a, actors[r].rot, actors[r].root, pins[r], fps, rigs
            )
    if params.hold_contacts:
        # Leg IK back onto the input's pins (and the y = 0 floor) for every fighter; no root or
        # pelvis moves, so smoothing an already smooth result changes nothing.
        fixes = hold_contacts(
            bm,
            {r: a.rot for r, a in actors.items()},
            {r: a.root for r, a in actors.items()},
            roles,
            params.contact,
        )
        for r, f in fixes.items():
            actors[r].rot, actors[r].root = f.rot, f.root
    for a in actors.values():
        a.rot = make_continuous(np.asarray(a.rot, dtype=np.float64)).astype(np.float32)
        a.root = np.asarray(a.root, dtype=np.float32)
    for r, a in actors.items():
        if pins and pins.get(r):
            pin_errors(base, bm.actors[r], a.rot, a.root, pins[r], infos[r].pins, fps, rigs)
    out = BodyMotion(
        frames=bm.frames,
        timestamps=bm.timestamps,
        skeleton=bm.skeleton,
        fps=bm.fps,
        contact_channels=list(bm.contact_channels),
        actors=actors,
        world_shift=list(bm.world_shift),  # smoothing does not move the world
    )
    return out, infos
