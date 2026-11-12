"""The spec's labelled metrics (Testing: metrics table), measured against hand labels.

Measuring on hand-labelled plants rather than on the tool's own detections keeps the gates
honest: a foot lock can drive skate on self-detected contacts to zero whatever the real quality.

==========================  =========================================================  ========
Metric                      How it is measured                                         Target
==========================  =========================================================  ========
Contact recall              labelled plant spans that the tool also detects            >= 90%
Slide within a plant        maximum horizontal foot travel inside each labelled span   <= 2 cm
Strike sharpness            effector speed at labelled peaks after / before smoothing  >= 80%
Ground penetration          mean foot depth below the floor during labelled plants     <= 1.5 cm
Reprojection error          projected joints vs 2D keypoints (``metrics``)             <= 15 px
==========================  =========================================================  ========

Labels file (JSON)::

    {"plants": [{"role": "A", "foot": "left", "start": 10, "end": 31}, ...],
     "strikes": [{"role": "A", "effector": "r_hand", "frame": 57}, ...]}

``start``/``end`` are shot-timeline frame numbers (end exclusive); a plant may name ``channels``
instead of ``foot``. ``frame`` may be approximate: the peak is searched within +-``peak_window``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from afv.core.skeletons import get_skeleton
from afv.core.types import BodyMotion
from afv.motion.contacts import Floor, actor_skeleton, contact_masks
from afv.motion.fk import forward_kinematics
from afv.motion.ik import BodyRoles, get_roles

TARGETS = {
    "contact_recall": 0.90,
    "slide_cm": 2.0,
    "strike_sharpness": 0.80,
    "penetration_cm": 1.5,
    "reprojection_px": 15.0,
}


@dataclass
class PlantLabel:
    role: str
    start: int
    end: int
    foot: str = ""
    channels: list[str] = field(default_factory=list)

    def channel_names(self, roles: BodyRoles) -> list[str]:
        """Channels of this label: explicit ``channels`` or the ``foot``'s leg via the roles (any
        skeleton). Raises ``ValueError`` when it resolves to nothing, so a metric is never vacuous."""
        known = {c.channel for c in roles.contacts}
        chans = list(self.channels) if self.channels else roles.side_channels(self.foot)
        bad = [c for c in chans if c not in known]
        if bad or not chans:
            raise ValueError(f"plant label {self}: unknown channels {bad or chans}; known {sorted(known)}")
        return chans


@dataclass
class StrikeLabel:
    role: str
    effector: str
    frame: int


@dataclass
class Labels:
    plants: list[PlantLabel] = field(default_factory=list)
    strikes: list[StrikeLabel] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Labels:
        return cls(
            [PlantLabel(**p) for p in d.get("plants", [])],
            [StrikeLabel(**s) for s in d.get("strikes", [])],
        )

    @classmethod
    def load(cls, path: Path) -> Labels:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _span(frames: np.ndarray, start: int, end: int) -> slice:
    return slice(int(np.searchsorted(frames, start)), int(np.searchsorted(frames, end)))


def contact_recall(
    bm: BodyMotion, labels: Labels, roles: BodyRoles | None = None, min_overlap: float = 0.5
) -> float:
    """Share of labelled plants of which >= ``min_overlap`` of frames are detected (any channel
    of the labelled foot)."""
    roles = roles or get_roles(get_skeleton(bm.skeleton))
    if not labels.plants:
        return float("nan")
    hits = 0
    for pl in labels.plants:
        masks = contact_masks(bm, bm.actors[pl.role], roles)
        sl = _span(bm.frames, pl.start, pl.end)
        det = np.zeros(bm.num_frames, dtype=bool)
        for ch in pl.channel_names(roles):
            det |= masks[ch]
        n = sl.stop - sl.start
        hits += int(n > 0 and det[sl].mean() >= min_overlap)
    return hits / len(labels.plants)


def _channel_positions(bm: BodyMotion, roles: BodyRoles) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for r, a in bm.actors.items():
        sk = actor_skeleton(bm, a)
        pos = forward_kinematics(sk, a.rot, a.root)[0]
        out[r] = {c.channel: pos[:, sk.index(c.joint)] for c in roles.contacts}
    return out


def slide_within_plants(
    bm: BodyMotion, labels: Labels, floor: Floor, roles: BodyRoles | None = None
) -> np.ndarray:
    """Maximum horizontal travel (m) of the labelled foot's joints inside each labelled span:
    the largest in-plane distance between any two frames of the span."""
    roles = roles or get_roles(get_skeleton(bm.skeleton))
    chp = _channel_positions(bm, roles)
    n = floor.normal
    out = []
    for pl in labels.plants:
        sl = _span(bm.frames, pl.start, pl.end)
        worst = 0.0
        for ch in pl.channel_names(roles):
            p = chp[pl.role][ch][sl]
            if len(p) < 2:
                continue
            p = p - (p @ n)[:, None] * n
            d = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
            worst = max(worst, float(d.max()))
        out.append(worst)
    return np.asarray(out)


def ground_penetration(bm: BodyMotion, labels: Labels, floor: Floor, roles: BodyRoles | None = None) -> float:
    """Mean depth (m) of the labelled foot's sole points below the floor during labelled plants."""
    roles = roles or get_roles(get_skeleton(bm.skeleton))
    chp = _channel_positions(bm, roles)
    depths = []
    for pl in labels.plants:
        sk = actor_skeleton(bm, bm.actors[pl.role])
        clear = roles.clearances(sk)
        sl = _span(bm.frames, pl.start, pl.end)
        for ch in pl.channel_names(roles):
            depths.append(np.maximum(0.0, clear[ch] - floor.height(chp[pl.role][ch][sl])))
    return float(np.concatenate(depths).mean()) if depths else 0.0


def effector_speeds(bm: BodyMotion, role: str, effector: str) -> np.ndarray:
    a = bm.actors[role]
    sk = actor_skeleton(bm, a)
    p = forward_kinematics(sk, a.rot, a.root)[0][:, sk.index(effector)]
    if len(p) < 2:
        return np.zeros(len(p))
    return np.linalg.norm(np.gradient(p, axis=0), axis=-1) * (bm.fps or 30.0)


def strike_sharpness(
    before: BodyMotion, after: BodyMotion, labels: Labels, peak_window: int = 3
) -> np.ndarray:
    """Per labelled strike: peak effector speed after smoothing / peak speed before (near the
    labelled frame)."""
    out = []
    for st in labels.strikes:
        sl = _span(before.frames, st.frame - peak_window, st.frame + peak_window + 1)
        v0 = effector_speeds(before, st.role, st.effector)[sl].max()
        v1 = effector_speeds(after, st.role, st.effector)[sl].max()
        out.append(v1 / v0 if v0 > 0 else float("nan"))
    return np.asarray(out)


def evaluate(
    after: BodyMotion,
    labels: Labels,
    floor: Floor,
    before: BodyMotion | None = None,
    roles: BodyRoles | None = None,
) -> dict[str, Any]:
    """All labelled metrics with pass/fail against the spec targets."""
    roles = roles or get_roles(get_skeleton(after.skeleton))
    res: dict[str, Any] = {}
    if labels.plants:
        rec = contact_recall(after, labels, roles)
        slide = slide_within_plants(after, labels, floor, roles)
        pen = ground_penetration(after, labels, floor, roles)
        res["contact_recall"] = {"value": rec, "pass": rec >= TARGETS["contact_recall"]}
        res["slide_cm"] = {
            "value": float(slide.max() * 100),
            "pass": float(slide.max() * 100) <= TARGETS["slide_cm"],
        }
        res["penetration_cm"] = {"value": pen * 100, "pass": pen * 100 <= TARGETS["penetration_cm"]}
    if labels.strikes and before is not None:
        sh = strike_sharpness(before, after, labels)
        res["strike_sharpness"] = {
            "value": float(np.nanmin(sh)),
            "pass": bool(np.nanmin(sh) >= TARGETS["strike_sharpness"]),
        }
    return res
