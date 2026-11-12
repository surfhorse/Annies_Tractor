"""CPU-only noise harness for the motion stages (round 5, track M): how much do the golden keys of
pair -> contacts -> smooth move when the pose3d (and camera) input carries GPU-level run-to-run
noise, and are they bimodal?

Usage (WSL, the afv env; ``RUN_*`` are project folders of the same shot from repeat runs, e.g. golden
test projects made with ``AFV_GOLDEN_DIR``; the first shot of each is used)::

    python tests/motion/noise_harness.py measure RUN_A RUN_B [RUN_C ...] [--out noise.json]
    python tests/motion/noise_harness.py run RUN_A --noise noise.json [--n 12] [--seed 0]
        [--scale 1.0] [--parts root,prior,rot,contact,cam,focal] [--jobs 4] [--out s.json] [--dump d.npz]
    python tests/motion/noise_harness.py mix RUN_A RUN_B [RUN_C ...] [--n 16] [--seed 0] [--out s.json]
    python tests/motion/noise_harness.py interp RUN_A RUN_B [--k 11] [--out s.json] [--dump d.npz]

``measure`` compares the pose3d and camera results of the repeat runs (same frames) and writes the
noise model. GEM-X's run-to-run root difference is depth: along the camera ray through the root, a
slow drift over the clip (Legendre trend up to degree 3, shared by both fighters and the camera's
offset along its axis) plus a small residual; across the ray ~1 mm. Rotation, contact-logit, camera
rotation and focal differences are modelled as correlated Gaussian noise of the measured RMS.

``run`` perturbs RUN_A N times with that model (x ``scale``; ``--parts`` for ablations). ``mix`` uses
the real differences instead: inputs A + sum_i w_i (B_i - A) with w_i ~ N(0, 1 / (2 m)) for m other runs.
``interp`` walks one real difference: A + t (B - A) for t in [-0.5, 1.5] (a step in a key along t is
a decision flipping). Each runs the three CPU stages exactly as the stages do (default params, no
edits) and prints per golden key (and diagnostics ``x.*``) the spread and the modes (values grouped
where consecutive sorted values are more than ``--gap`` apart). ``--dump`` keeps per-sample plant
masks, roots and probabilities to find which decision flipped.

The model holds the difference of two runs (sqrt(2) x one run's noise) and the median over the
measured pairs; a ``run`` draw is scaled by 1 / sqrt(2) and a ``mix`` uses w_i ~ N(0, 1 / (2 m)), so
one draw has about one run's noise. Even so the harness spreads are an upper bound (independent
Gaussian parts, extrapolated mixes): use them to find and compare decision flips, never to set golden
tolerances; those come from real repeat runs.

Nothing here touches project folders: results are computed in memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from afv.core.project import Project
from afv.core.skeletons import get_skeleton
from afv.core.types import BodyMotion, CameraTrack
from afv.motion.contacts import ContactParams, clean_contacts, contact_masks, segments
from afv.motion.ik import get_roles
from afv.motion.pair import PairParams, pair_fighters
from afv.motion.rotations import from_rotvec, make_continuous, qconj, qmul, to_rotvec
from afv.motion.smooth import SmoothParams, smooth_motion
from afv.qc.camera import camera_centres
from afv.qc.metrics import per_frame_metrics, summarise

LOGIT_CLIP = 1e-3
PARTS = ("root", "prior", "rot", "contact", "cam", "focal")


def _rays(bm: BodyMotion, cam: CameraTrack, root: np.ndarray) -> np.ndarray:
    """Unit camera rays [F,3] through ``root`` (this result's world)."""
    c = camera_centres(cam, bm.frames) + np.asarray(bm.world_shift, np.float64)
    ray = np.asarray(root, np.float64) - c
    return ray / np.maximum(np.linalg.norm(ray, axis=-1, keepdims=True), 1e-9)


def load_inputs(root: Path) -> tuple[BodyMotion, CameraTrack]:
    proj = Project(Path(root))
    shot = proj.shot(proj.shot_ids()[0])
    bm, cam = shot.load_result("pose3d"), shot.load_result("camera")
    assert isinstance(bm, BodyMotion) and isinstance(cam, CameraTrack)
    return bm, cam


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


def _corr_len(d: np.ndarray) -> float:
    """Lag (frames) where the mean normalised autocorrelation of difference signals [F, ...] drops
    to 0.5 (1 for white noise)."""
    x = np.asarray(d, np.float64).reshape(len(d), -1)
    x = x - x.mean(axis=0)
    var = (x * x).sum(axis=0)
    x = x[:, var > 1e-18]
    if x.shape[1] == 0 or len(x) < 4:
        return 1.0
    v = (x * x).sum()
    for lag in range(1, len(x) // 2):
        if (x[lag:] * x[:-lag]).sum() / v < 0.5:
            return float(lag)
    return float(len(x) // 2)


TREND_DEG = 3


def _legendre(n: int) -> np.ndarray:
    """[n, TREND_DEG + 1] Legendre polynomials on the clip (x from -1 to 1)."""
    x = np.linspace(-1.0, 1.0, n)
    return np.stack([np.polynomial.legendre.Legendre.basis(k)(x) for k in range(TREND_DEG + 1)], axis=1)


def _trend(d: np.ndarray) -> tuple[list[float], float, float]:
    """A run-to-run difference signal [F] as a slow trend (|Legendre coefficients| up to TREND_DEG:
    GEM-X's depth differences between repeat runs are a drift over the clip, not jitter) plus a
    residual (RMS, correlation length). The coefficients are signed: both fighters' (and the
    camera's) drifts are drawn with one shared random factor per degree, as they move together."""
    b = _legendre(len(d))
    c = np.linalg.lstsq(b, d, rcond=None)[0]
    res = d - b @ c
    return c.tolist(), float(_rms(res)), _corr_len(res)


def _rms(x: np.ndarray, axis: Any = None) -> np.ndarray:
    return np.sqrt(np.mean(np.square(np.asarray(x, np.float64)), axis=axis))


def _qangle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        to_rotvec(qmul(qconj(np.asarray(a, np.float64)), np.asarray(b, np.float64))), axis=-1
    )


def measure(runs: list[Path]) -> dict[str, Any]:
    """Noise model from repeat runs (project folders): statistics of each run's difference to the first."""
    out = measure_data([load_inputs(r) for r in runs])
    out["runs"] = [str(r) for r in runs]
    return out


def measure_pair(a: tuple[BodyMotion, CameraTrack], b: tuple[BodyMotion, CameraTrack]) -> dict[str, Any]:
    return measure_data([a, b])


def measure_data(data: list[tuple[BodyMotion, CameraTrack]]) -> dict[str, Any]:
    """Noise model from repeat runs' (pose3d, camera) results: each one's difference to the first."""
    bm0, cam0 = data[0]
    out: dict[str, Any] = {"frames": bm0.num_frames, "actors": {}, "pairs": []}
    pairs = []
    for bm1, cam1 in data[1:]:
        assert bm1.num_frames == bm0.num_frames, "repeat runs must cover the same frames"
        p: dict[str, Any] = {"world_shift_m": (np.subtract(bm1.world_shift, bm0.world_shift)).tolist()}
        for r, a0 in bm0.actors.items():
            a1 = bm1.actors[r]
            q: dict[str, Any] = {}
            # GEM-X's run-to-run root noise is depth: along the camera ray through the root (constant
            # part + slowly varying part); across the ray it is ~0.1 mm
            u = _rays(bm0, cam0, a0.root)
            dr = np.asarray(a1.root, np.float64) - np.asarray(a0.root, np.float64)
            al = np.sum(dr * u, axis=-1)
            q["root_along_trend_m"], q["root_along_resid_m"], q["root_corr"] = _trend(al)
            q["root_across_rms_m"] = _rms(dr - al[:, None] * u, axis=0).tolist()
            q["root_max_m"] = float(np.abs(dr).max())
            if a0.root_prior is not None and a1.root_prior is not None:
                dp = np.asarray(a1.root_prior, np.float64) - np.asarray(a0.root_prior, np.float64)
                ap = np.sum(dp * u, axis=-1)
                q["prior_along_trend_m"], q["prior_along_resid_m"], q["prior_corr"] = _trend(ap)
                q["prior_across_rms_m"] = _rms(dp - ap[:, None] * u, axis=0).tolist()
            ang = _qangle(a0.rot, a1.rot)  # [F,J]
            q["rot_rms_rad"] = _rms(ang, axis=0).tolist()  # per joint
            q["rot_rms_deg_all"] = float(np.degrees(_rms(ang)))
            q["rot_max_deg"] = float(np.degrees(ang.max()))
            rv = to_rotvec(qmul(qconj(np.asarray(a0.rot, np.float64)), np.asarray(a1.rot, np.float64)))
            q["rot_corr"] = _corr_len(rv)
            if a0.contact is not None and a1.contact is not None:
                dl = _logit(a1.contact) - _logit(a0.contact)
                q["contact_logit_rms"] = _rms(dl, axis=0).tolist()
                q["contact_abs_max"] = float(np.abs(np.asarray(a1.contact) - np.asarray(a0.contact)).max())
                q["contact_corr"] = _corr_len(dl)
            if a0.conf is not None and a1.conf is not None:
                q["conf_rms"] = float(_rms(np.asarray(a1.conf) - np.asarray(a0.conf)))
            w0, w1 = getattr(a0, "weak", None), getattr(a1, "weak", None)
            if w0 is not None and w1 is not None:
                q["weak_frames_differ"] = int(np.sum(np.asarray(w0, bool) != np.asarray(w1, bool)))
            p[r] = q
        dc = np.asarray(cam1.cam_pos, np.float64) - np.asarray(cam0.cam_pos, np.float64)
        p["cam_pos_offset_m"] = dc.mean(axis=0).tolist()  # a vector: drawn as N(0,1) x this
        p["cam_pos_rms_m"] = _rms(dc - dc.mean(axis=0), axis=0).tolist()
        p["cam_pos_corr"] = _corr_len(dc)
        ca = _qangle(cam0.cam_rot, cam1.cam_rot)
        p["cam_rot_rms_rad"] = float(_rms(ca))
        p["cam_rot_corr"] = _corr_len(
            to_rotvec(qmul(qconj(np.asarray(cam0.cam_rot, np.float64)), np.asarray(cam1.cam_rot, np.float64)))
        )
        df = np.asarray(cam1.focal_px, np.float64) / np.asarray(cam0.focal_px, np.float64) - 1.0
        p["focal_rel_rms"] = float(_rms(df))
        pairs.append(p)
    out["pairs"] = pairs
    # The model: the median RMS per quantity over the pairs (a typical pair), the median correlation length.
    model: dict[str, Any] = {"actors": {}}
    for r in bm0.actors:
        qs = [p[r] for p in pairs]
        m: dict[str, Any] = {}
        for k in (
            "root_along_resid_m",
            "root_across_rms_m",
            "prior_along_resid_m",
            "prior_across_rms_m",
            "rot_rms_rad",
            "contact_logit_rms",
        ):
            vals = [np.asarray(q[k]) for q in qs if k in q]
            if vals:
                m[k] = np.median(np.stack(vals), axis=0).tolist()  # a typical pair, not the largest
        for k in ("root_corr", "prior_corr", "rot_corr", "contact_corr"):
            vals = [q[k] for q in qs if k in q]
            if vals:
                m[k] = float(np.median(vals))
        model["actors"][r] = m
    model["cam_pos_rms_m"] = np.median(
        np.stack([np.asarray(p["cam_pos_rms_m"]) for p in pairs]), axis=0
    ).tolist()
    # The drifts (signed, coherent across fighters and the camera) from the pair with the median one.
    by_size = sorted(
        pairs, key=lambda p: sum(float(np.linalg.norm(p[r]["root_along_trend_m"])) for r in bm0.actors)
    )
    big = by_size[(len(by_size) - 1) // 2]
    model["cam_pos_offset_m"] = big["cam_pos_offset_m"]
    for r in bm0.actors:
        for k in ("root_along_trend_m", "prior_along_trend_m"):
            if k in big[r]:
                model["actors"][r][k] = big[r][k]
    for k in ("cam_rot_rms_rad", "focal_rel_rms"):
        model[k] = float(np.median([p[k] for p in pairs]))
    for k in ("cam_pos_corr", "cam_rot_corr"):
        model[k] = float(np.median([p[k] for p in pairs]))
    model["world_shift_rms_m"] = _rms(np.array([p["world_shift_m"] for p in pairs]), axis=0).tolist()
    out["model"] = model
    return out


def _colored(rng: np.random.Generator, shape: tuple[int, ...], corr: float) -> np.ndarray:
    """Unit-RMS Gaussian noise [F, ...], correlated over time with autocorrelation 0.5 at ``corr``
    frames (Gaussian filter sigma = corr / 1.665)."""
    x = rng.standard_normal(shape)
    if corr > 1.0:
        x = gaussian_filter1d(x, sigma=corr / 1.665, axis=0, mode="reflect")
    s = _rms(x, axis=0)
    return x / np.maximum(s, 1e-12)


def perturb(
    bm: BodyMotion,
    cam: CameraTrack,
    model: dict[str, Any],
    rng: np.random.Generator,
    scale: float = 1.0,
    parts: tuple[str, ...] = (),
) -> tuple[BodyMotion, CameraTrack]:
    """``bm``/``cam`` plus one draw of the measured noise (x ``scale``; only the noise ``parts``,
    default all). The model holds differences of two runs, each with its own noise, so a difference is
    sqrt(2) x one run's noise: a draw is scaled by 1 / sqrt(2) to be one run's."""
    parts = parts or PARTS
    scale = scale / np.sqrt(2.0)
    n = bm.num_frames
    actors = {}
    z = rng.standard_normal(TREND_DEG + 1)  # shared drift factors (fighters, priors, camera offset)
    for r, a in bm.actors.items():
        m = model["actors"].get(r, {})
        root = np.asarray(a.root, np.float64)
        u = _rays(bm, cam, root)

        def depth_noise(pre: str, corr: float, m=m, u=u) -> np.ndarray:
            coef = np.asarray(m.get(f"{pre}_along_trend_m", np.zeros(TREND_DEG + 1))) * z
            al = _legendre(n) @ coef + m.get(f"{pre}_along_resid_m", 0.0) * _colored(rng, (n,), corr)
            across = np.asarray(m.get(f"{pre}_across_rms_m", np.zeros(3))) * _colored(rng, (n, 3), n / 3.0)
            across -= np.sum(across * u, axis=-1, keepdims=True) * u
            return scale * (al[:, None] * u + across)

        if "root" in parts and "root_along_trend_m" in m:
            root = root + depth_noise("root", m.get("root_corr", 1.0))
        prior = a.root_prior
        if "prior" in parts and prior is not None and "prior_along_trend_m" in m:
            prior = (np.asarray(prior, np.float64) + depth_noise("prior", m.get("prior_corr", 1.0))).astype(
                np.float32
            )
        rot = np.asarray(a.rot, np.float64)
        if "rot" in parts and "rot_rms_rad" in m:
            j = rot.shape[1]
            # per joint: a random rotation vector whose angle has the measured RMS (3 axes -> / sqrt 3)
            amp = scale * np.asarray(m["rot_rms_rad"])[None, :, None] / np.sqrt(3.0)
            rv = amp * _colored(rng, (n, j, 3), m.get("rot_corr", 1.0))
            rot = make_continuous(qmul(rot, from_rotvec(rv)))
        contact = a.contact
        if "contact" in parts and contact is not None and "contact_logit_rms" in m:
            lg = _logit(contact) + scale * np.asarray(m["contact_logit_rms"]) * _colored(
                rng, contact.shape, m.get("contact_corr", 1.0)
            )
            contact = (1.0 / (1.0 + np.exp(-lg))).astype(np.float32)
        actors[r] = replace(a, rot=rot.astype(np.float32), root=root.astype(np.float32), contact=contact,
                            root_prior=prior)  # fmt: skip
    ws = np.asarray(bm.world_shift, np.float64)
    if "cam" in parts and "world_shift_rms_m" in model:
        ws = ws + scale * np.asarray(model["world_shift_rms_m"]) * rng.standard_normal(3)
    bm2 = replace(bm, actors=actors, world_shift=[float(v) for v in ws])
    nc = len(cam.cam_pos)
    cpos = np.asarray(cam.cam_pos, np.float64)
    crot = np.asarray(cam.cam_rot, np.float64)
    foc = np.asarray(cam.focal_px, np.float64)
    if "cam" in parts:
        cpos = cpos + scale * (
            np.asarray(model.get("cam_pos_offset_m", np.zeros(3))) * z[0]
            + np.asarray(model["cam_pos_rms_m"]) * _colored(rng, (nc, 3), model.get("cam_pos_corr", 1.0))
        )
        crv = (
            scale
            * model["cam_rot_rms_rad"]
            / np.sqrt(3.0)
            * _colored(rng, (nc, 3), model.get("cam_rot_corr", 1.0))
        )
        crot = qmul(crot, from_rotvec(crv))
    if "focal" in parts:
        foc = foc * (1.0 + scale * model["focal_rel_rms"] * rng.standard_normal())
    cam2 = replace(cam, cam_pos=cpos.astype(np.float32), cam_rot=crot.astype(np.float32),
                   focal_px=foc.astype(np.float32))  # fmt: skip
    return bm2, cam2


def combine(
    base: tuple[BodyMotion, CameraTrack], others: list[tuple[BodyMotion, CameraTrack]], w: list[float]
) -> tuple[BodyMotion, CameraTrack]:
    """Inputs A + sum_i w_i (B_i - A) of real repeat runs (rotations in A's tangent space, contacts in
    logit space). One B with w = t walks the line from A (t = 0) to B (t = 1)."""
    bm0, cam0 = base

    def mix(get, f=lambda x: np.asarray(x, np.float64), g=lambda x: x):
        x0 = get(bm0, cam0)
        if x0 is None or any(get(*o) is None for o in others):
            return x0
        y = f(x0) + sum(wi * (f(get(*o)) - f(x0)) for wi, o in zip(w, others, strict=True))
        return g(y)

    actors = {}
    for r, a0 in bm0.actors.items():
        q0 = np.asarray(a0.rot, np.float64)
        rv = sum(
            wi * to_rotvec(qmul(qconj(q0), np.asarray(o[0].actors[r].rot, np.float64)))
            for wi, o in zip(w, others, strict=True)
        )
        f32 = lambda x: np.asarray(x, np.float32)  # noqa: E731
        sig = lambda x: (1.0 / (1.0 + np.exp(-x))).astype(np.float32)  # noqa: E731
        actors[r] = replace(
            a0,
            rot=qmul(q0, from_rotvec(rv)).astype(np.float32),
            root=mix(lambda bm, c, r=r: bm.actors[r].root, g=f32),
            root_prior=mix(lambda bm, c, r=r: bm.actors[r].root_prior, g=f32),
            conf=mix(lambda bm, c, r=r: bm.actors[r].conf, g=f32),
            contact=mix(lambda bm, c, r=r: bm.actors[r].contact, f=_logit, g=sig),
        )
    ws = mix(lambda bm, c: bm.world_shift)
    bm = replace(bm0, actors=actors, world_shift=[float(v) for v in ws])
    qc = np.asarray(cam0.cam_rot, np.float64)
    crv = sum(
        wi * to_rotvec(qmul(qconj(qc), np.asarray(o[1].cam_rot, np.float64)))
        for wi, o in zip(w, others, strict=True)
    )
    cam = replace(
        cam0,
        cam_pos=mix(lambda bm, c: c.cam_pos).astype(np.float32),
        cam_rot=qmul(qc, from_rotvec(crv)).astype(np.float32),
        focal_px=mix(lambda bm, c: c.focal_px).astype(np.float32),
    )
    return bm, cam


def _path_len(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(x, np.float64), axis=0), axis=-1).sum())


def run_chain(bm: BodyMotion, cam: CameraTrack, detail: bool = False) -> dict[str, Any]:
    """pair -> contacts -> smooth as the stages run them (default params, no edits); the golden keys
    of those stages plus diagnostics."""
    pair, prep = pair_fighters(bm, cam, PairParams())
    cont, crep = clean_contacts(pair, ContactParams(), painted=None, camera=cam)
    sm, infos = smooth_motion(cont, SmoothParams())
    roles = get_roles(get_skeleton(cont.skeleton))
    s: dict[str, Any] = {}
    roots = [np.asarray(a.root, np.float64) for _, a in sorted(pair.actors.items())]
    if len(roots) == 2:
        s["pair.distance_median_m"] = round(float(np.median(np.linalg.norm(roots[0] - roots[1], axis=-1))), 4)
    fq = summarise(per_frame_metrics(cont, roles, crep.floor, ref=pair), roles)
    sq = summarise(per_frame_metrics(sm, roles, None, ref=cont), roles)
    for r in sorted(cont.actors):
        s[f"contacts.{r}.contact_frames"] = fq[r]["contact_frames"]
        s[f"contacts.{r}.mean_contact_foot_speed_cm_s"] = round(fq[r]["mean_contact_foot_speed_cm_s"], 4)
        s[f"contacts.{r}.max_penetration_cm"] = round(fq[r]["max_penetration_cm"], 4)
        s[f"smooth.{r}.strikes"] = len(infos[r].strikes)
        s[f"smooth.{r}.root_path_m"] = round(_path_len(sm.actors[r].root), 4)
        # diagnostics (not golden keys)
        s[f"x.pair.{r}.root_path_m"] = round(_path_len(pair.actors[r].root), 4)
        s[f"x.contacts.{r}.root_path_m"] = round(_path_len(cont.actors[r].root), 4)
        s[f"x.contacts.{r}.plants"] = sum(
            len(segments(m)) for m in contact_masks(cont, cont.actors[r], roles).values()
        )
        s[f"x.smooth.{r}.max_skate_cm_s"] = round(
            float(per_frame_metrics(sm, roles)[r]["skate"].max() * 100), 3
        )
        s[f"x.smooth.{r}.p95_contact_speed_cm_s"] = round(sq[r]["p95_contact_foot_speed_cm_s"], 3)
        s[f"x.contacts.{r}.max_skate_cm_s"] = round(
            float(per_frame_metrics(cont, roles, crep.floor)[r]["skate"].max() * 100), 3
        )
        s[f"x.smooth.{r}.max_penetration_cm"] = round(sq[r]["max_penetration_cm"], 4)
        s[f"x.contacts.{r}.max_hover_cm"] = round(fq[r]["max_hover_in_contact_cm"] or 0.0, 3)
    s["x.pair.max_push_m"] = round(float(prep.push.max()) if prep.push.size else 0.0, 4)
    s["x.pair.push_frames"] = int((prep.push > 1e-6).sum())
    s["x.contacts.floor_shift_m"] = round(float(crep.floor_shift), 4)
    if detail:
        d: dict[str, np.ndarray] = {"pair/push": prep.push}
        for r in cont.actors:
            for ch, m in contact_masks(cont, cont.actors[r], roles).items():
                d[f"{r}/{ch}"] = m
            d[f"{r}/pair_root"] = np.asarray(pair.actors[r].root)
            d[f"{r}/contacts_root"] = np.asarray(cont.actors[r].root)
            d[f"{r}/smooth_root"] = np.asarray(sm.actors[r].root)
            d[f"{r}/strike_frames"] = np.array([i.frame for i in infos[r].strikes], dtype=np.int64)
            d[f"{r}/strike_eff"] = np.array([i.effector for i in infos[r].strikes] or [""])
            for ch, pr in crep.actors[r].probs.items():
                d[f"{r}/prob_{ch}"] = pr
        s["_detail"] = d
    return s


def _job(args: tuple) -> dict[str, Any]:
    kind, root, extra, seed, scale, detail, parts = args
    base = load_inputs(Path(root))
    if kind == "noise":
        if seed < 0:
            bm, cam = base
        else:
            bm, cam = perturb(*base, extra, np.random.default_rng(seed), scale, parts)
    else:  # interp / mix: real repeat differences
        others = [load_inputs(Path(o)) for o in extra["others"]]
        bm, cam = combine(base, others, extra["w"])
    out = run_chain(bm, cam, detail)
    out["_id"] = seed if kind == "noise" else extra["w"][0]
    return out


GAPS = (
    ("root_path", 0.01),
    ("distance", 0.01),
    ("_cm", 0.5),
    ("frames", 3.0),
    ("plants", 1.0),
    ("strikes", 1.0),
)


def _gap(key: str) -> float:
    """Smallest jump between sorted values that may separate two modes of ``key`` (its unit)."""
    return next((g for pat, g in GAPS if pat in key), 0.01)


def modes(values: list[float], gap: float) -> list[list[float]]:
    """Group sorted values into modes: a new mode starts at a jump larger than ``gap`` and than 4 x
    the median jump between neighbours (so evenly spread, continuous values stay one mode)."""
    v = sorted(values)
    if not v:
        return []
    steps = np.diff(v)
    cut = max(gap, 4.0 * float(np.median(steps))) if len(steps) else gap
    groups: list[list[float]] = [[v[0]]]
    for x, d in zip(v[1:], steps, strict=True):
        if d > cut:
            groups.append([x])
        else:
            groups[-1].append(x)
    return groups


def report(samples: list[dict[str, Any]], gap: float | None = None) -> dict[str, Any]:
    """Per key: range, spread, std and modes (``gap`` overrides the per-unit jump of metre keys)."""
    keys = sorted(k for k in samples[0] if not k.startswith("_"))
    rep = {}
    for k in keys:
        vals = [float(s[k]) for s in samples]
        g = _gap(k) if gap is None or not ("root_path" in k or "distance" in k) else gap
        rep[k] = {
            "min": min(vals),
            "max": max(vals),
            "spread": round(max(vals) - min(vals), 4),
            "std": round(float(np.std(vals)), 4),
            "modes": [[round(min(x), 4), round(max(x), 4), len(x)] for x in modes(vals, g)],
            "values": vals,
        }
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("runs", nargs="+", type=Path)
    m.add_argument("--out", type=Path)
    r = sub.add_parser("run")
    r.add_argument("run", type=Path)
    r.add_argument("--noise", type=Path, required=True)
    r.add_argument("--n", type=int, default=12)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--scale", type=float, default=1.0)
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--parts", default=",".join(PARTS), help="noise parts to apply: " + ",".join(PARTS))
    r.add_argument("--gap", type=float, default=0.01, help="m between modes of a metre key")
    r.add_argument("--out", type=Path)
    r.add_argument("--dump", type=Path, help="npz of per-sample masks, roots, push (flip analysis)")
    i = sub.add_parser("interp")
    i.add_argument("a", type=Path)
    i.add_argument("b", type=Path)
    i.add_argument("--k", type=int, default=11)
    i.add_argument("--jobs", type=int, default=4)
    i.add_argument("--gap", type=float, default=0.01)
    i.add_argument("--out", type=Path)
    i.add_argument("--dump", type=Path)
    x = sub.add_parser("mix")
    x.add_argument("a", type=Path)
    x.add_argument("others", nargs="+", type=Path)
    x.add_argument("--n", type=int, default=16)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--jobs", type=int, default=4)
    x.add_argument("--gap", type=float, default=0.01)
    x.add_argument("--out", type=Path)
    x.add_argument("--dump", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "measure":
        res = measure(a.runs)
        txt = json.dumps(res, indent=1)
        if a.out:
            a.out.write_text(txt)
        print(json.dumps(res["model"], indent=1)[:4000])
        for p in res["pairs"]:
            short = {
                k: v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if "rms_rad" not in kk}
                for k, v in p.items()
            }
            print(short)
        return 0
    detail = bool(getattr(a, "dump", None))
    if a.cmd == "run":
        model = json.loads(a.noise.read_text())["model"]
        parts = tuple(a.parts.split(","))
        jobs = [("noise", str(a.run), model, -1, 0.0, detail, parts)]  # the unperturbed input first
        jobs += [("noise", str(a.run), model, a.seed + k, a.scale, detail, parts) for k in range(a.n)]
    elif a.cmd == "interp":
        ts = np.linspace(-0.5, 1.5, a.k)
        jobs = [
            ("interp", str(a.a), {"others": [str(a.b)], "w": [float(t)]}, 0, 0.0, detail, PARTS) for t in ts
        ]
    if a.cmd == "mix":
        rng = np.random.default_rng(a.seed)
        jobs = [("mix", str(a.a), {"others": [str(o) for o in a.others], "w": [0.0] * len(a.others)}, 0, 0.0,
                 detail, PARTS)]  # fmt: skip
        others = [str(o) for o in a.others]
        for _ in range(a.n):
            # sum_i w_i (B_i - A) = sum_i w_i e_i - (sum_i w_i) e_A for run noises e: with w_i ~
            # N(0, 1 / (2 m)) its variance is one run's (m other runs)
            w = (rng.standard_normal(len(a.others)) / np.sqrt(2.0 * len(a.others))).tolist()
            jobs.append(("mix", str(a.a), {"others": others, "w": w}, 0, 0.0, detail, PARTS))
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        samples = list(ex.map(_job, jobs))
    if detail:
        flat = {f"s{k}/{n}": v for k, s in enumerate(samples) for n, v in s.pop("_detail").items()}
        np.savez_compressed(a.dump, ids=np.array([s["_id"] for s in samples], dtype=np.float64), **flat)
    rep = report(samples[1:] if a.cmd in ("run", "mix") else samples, a.gap)
    if a.cmd in ("run", "mix"):
        rep["_base"] = {k: v for k, v in samples[0].items() if not k.startswith("_")}
    else:
        rep["_t"] = [s["_id"] for s in samples]
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1))
    base = rep.get("_base", {})
    for k, v in rep.items():
        if k.startswith("_"):
            continue
        b = f" base {base[k]}" if k in base else ""
        print(f"{k:45s} spread {v['spread']:<8} modes {v['modes']}{b}")
    if a.cmd == "interp":
        for k in rep:
            if not k.startswith("_") and len(rep[k]["modes"]) > 1:
                print(k, list(zip(rep["_t"], rep[k]["values"], strict=True)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
