"""Frame-range re-runs of the motion stages (``afv run --from <stage> --frames a:b``).

A range-capable stage (``supports_range = True``) gets ``ctx.frame_range = (a, b)`` (shot-timeline
frames, ``b`` exclusive) and ``ctx.previous`` (its own previous full-length result). The runner
only gives a range when the stage's params equal the previous result's and its edits differ only
inside the range (otherwise it runs the stage in full).

Every stage solves the **whole shot** again (the motion stages take about a second for 10 s, and a
slice changes the solution through its edges: the contacts root correction couples the whole clip)
and then blends that solution into ``ctx.previous`` in a *zone* around the range: weight 1 inside
the zone, a smootherstep ramp (C2) to 0 over ``blend`` frames outside it, and exactly the previous
result beyond. Rotations slerp per joint, root and root prior lerp; plant masks are merged per
segment (a plant comes whole from one side, never spliced frame by frame). Where nothing is
blended (weight 1, or equal values) the values are copied, so the result is bit-identical to
``previous`` outside the zone + ramp.

* **contacts** (:class:`afv.motion.contacts.FrozenContacts`): the editable frames are the range
  +- ``CONTACTS_EDIT`` frames, grown to whole plants that touch them (at most
  ``CONTACTS_EXTEND`` more on each side); the root offsets outside them are held at the previous
  run's (read from the ``range_state.npz`` it wrote next to its result), so the root solve there
  is a boundary condition and the rest of the shot is reproduced. Its zone is that editable set,
  so its reach is at most :data:`CONTACTS_MAX_MARGIN` frames.
* **smooth**: zone = range +- (the contacts reach + the smoother's half window + 1): its input may
  have changed anywhere contacts may have, and smoothing spreads a change by one half window.
* **hands_face** (per frame): zone = range +- the largest smooth reach (strength 4).

Margins (per side, frames, beyond the range): contacts <= 65, smooth 85 (strength <= 1) to 106
(strength 4), hands_face 108; all stages ramp over the last ``blend`` of them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from afv.core.stage import Ctx, StageError
from afv.core.types import ActorMotion, BodyMotion
from afv.motion.contacts import segments
from afv.motion.rotations import slerp

CONTACTS_EDIT = 8  # contacts re-solves the range +- this many frames ...
CONTACTS_EXTEND = 45  # ... grown to whole plants touching them, by at most this much per side
BLEND = 12  # ramp back into the previous result (contacts, smooth)
CONTACTS_MAX_MARGIN = CONTACTS_EDIT + CONTACTS_EXTEND + BLEND  # 65
SMOOTH_MAX_HALF_WINDOW = 28  # afv.motion.smooth.MAX_HALF_WINDOW (strength 4)
HANDS_MARGIN = CONTACTS_MAX_MARGIN + SMOOTH_MAX_HALF_WINDOW + 1 + BLEND + 2  # 108
HANDS_BLEND = 2
STATE_FILE = "range_state.npz"


def smooth_margin(half_window: int) -> int:
    """smooth's reach beyond the range for a smoother of half window ``half_window``."""
    return CONTACTS_MAX_MARGIN + int(half_window) + 1 + BLEND


@dataclass(frozen=True)
class RangePlan:
    """Blend plan over the whole motion: weight 1 on ``[z0, z1)``, ramps to 0 outside, 0 outside
    the kept frames ``[k0, k1)``."""

    z0: int
    z1: int
    k0: int
    k1: int
    weights: np.ndarray  # [F] blend weight of the new solution


def smootherstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def range_indices(frames: np.ndarray, frame_range: tuple[int, int]) -> tuple[int, int]:
    """Indices [ia, ib) of shot-timeline frames ``frame_range`` in ``frames``."""
    frames = np.asarray(frames)
    a, b = int(frame_range[0]), int(frame_range[1])
    if b <= a:
        raise ValueError(f"empty frame range {a}:{b}")
    ia = int(np.searchsorted(frames, a, side="left"))
    ib = int(np.searchsorted(frames, b, side="left"))
    if len(frames) == 0 or ib <= 0 or ia >= len(frames):
        lo, hi = (int(frames[0]), int(frames[-1])) if len(frames) else (0, -1)
        raise ValueError(f"frame range {a}:{b} is outside the shot (frames {lo}..{hi})")
    return ia, ib


def zone_plan(n_frames: int, z0: int, z1: int, blend: int) -> RangePlan:
    """Weight 1 on [z0, z1) (clipped), a smootherstep ramp over ``blend`` frames on each side."""
    z0, z1 = max(0, z0), min(n_frames, z1)
    k0, k1 = max(0, z0 - blend), min(n_frames, z1 + blend)
    t = np.arange(n_frames, dtype=np.float64)
    w = np.zeros(n_frames)
    w[z0:z1] = 1.0
    if blend > 0:
        up = smootherstep((t - (z0 - blend - 0.5)) / (blend + 1.0))
        down = smootherstep(((z1 + blend - 0.5) - t) / (blend + 1.0))
        ramp = np.minimum(up, down)
        side = (t >= k0) & (t < k1) & (w < 1.0)
        w[side] = ramp[side]
    return RangePlan(z0, z1, k0, k1, w)


def plan_range(frames: np.ndarray, frame_range: tuple[int, int], margin: int, blend: int) -> RangePlan:
    """The plan of a stage that reaches ``margin`` frames beyond the range (the last ``blend`` of
    them ramp)."""
    ia, ib = range_indices(frames, frame_range)
    bl = max(0, min(blend, margin))
    return zone_plan(len(np.asarray(frames)), ia - (margin - bl), ib + (margin - bl), bl)


def check_previous(prev: object, ref: BodyMotion) -> str | None:
    """Why ``prev`` cannot take a range re-run of a stage whose new input is ``ref`` (None = it can)."""
    if not isinstance(prev, BodyMotion):
        return f"previous result is {type(prev).__name__}, not a body motion"
    same = prev.num_frames == ref.num_frames and np.array_equal(
        np.asarray(prev.frames), np.asarray(ref.frames)
    )
    if not same:
        return "previous result has different frames"
    if set(prev.actors) != set(ref.actors):
        return f"previous result has fighters {sorted(prev.actors)}, input {sorted(ref.actors)}"
    if prev.skeleton != ref.skeleton:
        return "previous result is on another skeleton"
    for r, a in prev.actors.items():
        if np.asarray(a.rot).shape != np.asarray(ref.actors[r].rot).shape:
            return f"fighter {r}: previous rotations have another shape"
    return None


def range_plan(ctx: Ctx, stage: str, inp: BodyMotion, margin: int, blend: int) -> RangePlan | None:
    """The blend plan of a range re-run, or None for a full run (no range, or no usable previous
    result: then the reason is reported through ``ctx.progress``)."""
    fr = ctx.frame_range
    if fr is None:
        return None
    why = check_previous(getattr(ctx, "previous", None), inp)
    if why:
        ctx.progress(-1.0, f"{stage}: {why}; re-solving the whole shot")
        return None
    try:
        return plan_range(inp.frames, (int(fr[0]), int(fr[1])), margin, blend)
    except ValueError as e:
        raise StageError(f"{stage}: {e}") from e


def range_summary(ctx: Ctx, bm: BodyMotion, plan: RangePlan | None, **extra: Any) -> dict[str, Any]:
    """qc.json entry of a range re-run: the range, the re-solved zone and the changed frames."""
    if plan is None or ctx.frame_range is None:
        return {}
    f = np.asarray(bm.frames)
    last = int(f[-1]) + 1

    def at(i: int) -> int:
        return int(f[i]) if i < len(f) else last

    return {
        "range": {
            "frames": [int(ctx.frame_range[0]), int(ctx.frame_range[1])],
            "zone": [at(plan.z0), at(plan.z1)],
            "changed": [at(plan.k0), at(plan.k1)],
            **extra,
        }
    }


def slice_motion(bm: BodyMotion, lo: int, hi: int) -> BodyMotion:
    """Frames ``[lo, hi)`` of every fighter (frame numbers and timestamps kept)."""

    def cut(x: np.ndarray | None) -> np.ndarray | None:
        return None if x is None else np.asarray(x)[lo:hi].copy()

    actors = {
        r: ActorMotion(
            rot=np.asarray(a.rot)[lo:hi].copy(),
            root=np.asarray(a.root)[lo:hi].copy(),
            conf=cut(a.conf),
            contact=cut(a.contact),
            rest_offsets=a.rest_offsets,
            root_prior=cut(a.root_prior),
            weak=cut(getattr(a, "weak", None)),
        )
        for r, a in bm.actors.items()
    }
    return BodyMotion(
        frames=np.asarray(bm.frames)[lo:hi].copy(),
        timestamps=np.asarray(bm.timestamps)[lo:hi].copy(),
        skeleton=bm.skeleton,
        fps=bm.fps,
        contact_channels=list(bm.contact_channels),
        actors=actors,
        world_shift=list(bm.world_shift),
    )


def merge_masks(prev: np.ndarray, new: np.ndarray, plan: RangePlan) -> np.ndarray:
    """Plant mask [F] with whole segments: the new solution's plants that touch the zone, the
    previous result's plants that do not (both clipped to the kept frames; beyond them: previous)."""
    prev = np.asarray(prev, dtype=bool)
    new = np.asarray(new, dtype=bool)
    out = prev.copy()
    k0, k1 = plan.k0, plan.k1
    for s0, e0 in segments(prev):
        if s0 < plan.z1 and e0 > plan.z0:
            out[max(s0, k0) : min(e0, k1)] = False
    for s0, e0 in segments(new):
        if s0 < plan.z1 and e0 > plan.z0:
            out[max(s0, k0) : min(e0, k1)] = True
    return out


def blend_into_previous(
    prev: BodyMotion, new: BodyMotion, plan: RangePlan, plant_channels: list[str] | None = None
) -> BodyMotion:
    """Full-length result: ``new`` (a whole-shot solve) blended into ``prev`` with ``plan``; exactly
    ``prev`` where the weight is 0. Contact channels in ``plant_channels`` are merged per segment
    (:func:`merge_masks`), others switch at weight 0.5."""
    if new.num_frames != prev.num_frames:
        raise ValueError(f"range solve has {new.num_frames} frames, previous {prev.num_frames}")
    if list(new.contact_channels) != list(prev.contact_channels):
        raise ValueError(
            f"range solve has contact channels {new.contact_channels}, previous {prev.contact_channels}"
        )
    k0, k1 = plan.k0, plan.k1
    w = plan.weights[k0:k1]
    plant_idx = [prev.contact_channels.index(c) for c in (plant_channels or []) if c in prev.contact_channels]
    actors = {}
    for r, pa in prev.actors.items():
        na = new.actors[r]
        rot = np.array(pa.rot, dtype=np.float32, copy=True)
        p32, n32 = np.asarray(pa.rot, np.float32)[k0:k1], np.asarray(na.rot, np.float32)[k0:k1]
        p_rot, n_rot = p32.astype(np.float64), n32.astype(np.float64)
        n_signed = np.where(np.sum(n_rot * p_rot, axis=-1, keepdims=True) < 0.0, -n_rot, n_rot)
        blended = slerp(p_rot, n_signed, np.broadcast_to(w[:, None], p_rot.shape[:2]))
        blended = np.where(np.sum(blended * p_rot, axis=-1, keepdims=True) < 0.0, -blended, blended)
        full_w = np.broadcast_to((w >= 1.0)[:, None, None], blended.shape)
        blended = np.where(full_w, n_signed, blended).astype(np.float32)
        same = np.all(p32 == n32, axis=-1, keepdims=True)
        rot[k0:k1] = np.where(same, p32, blended)

        root = np.array(pa.root, dtype=np.float32, copy=True)
        pr, nr = np.asarray(pa.root, np.float32)[k0:k1], np.asarray(na.root, np.float32)[k0:k1]
        mixed = ((1.0 - w[:, None]) * pr.astype(np.float64) + w[:, None] * nr.astype(np.float64)).astype(
            np.float32
        )
        root[k0:k1] = np.where((w >= 1.0)[:, None], nr, np.where(pr == nr, pr, mixed))

        def mix(p: np.ndarray | None, q: np.ndarray | None, switch: bool) -> np.ndarray | None:
            if p is None or q is None:
                return None if p is None else np.array(p, copy=True)
            out = np.array(p, copy=True)
            pk, qk = np.asarray(p)[k0:k1], np.asarray(q)[k0:k1]
            ww = w.reshape((-1,) + (1,) * (pk.ndim - 1))
            if switch:
                out[k0:k1] = np.where(ww >= 0.5, qk, pk)
            else:
                out[k0:k1] = np.where(ww >= 1.0, qk, ((1.0 - ww) * pk + ww * qk).astype(out.dtype))
            return out

        contact = mix(pa.contact, na.contact, True)
        if contact is not None and na.contact is not None:
            for ci in plant_idx:
                pm, nm = np.asarray(pa.contact)[:, ci] > 0.5, np.asarray(na.contact)[:, ci] > 0.5
                merged = merge_masks(pm, nm, plan)
                contact[k0:k1, ci] = np.where(merged[k0:k1], 1.0, 0.0).astype(contact.dtype)
        actors[r] = ActorMotion(
            rot=rot,
            root=root,
            conf=mix(pa.conf, na.conf, True),
            contact=contact,
            rest_offsets=pa.rest_offsets,
            root_prior=mix(pa.root_prior, na.root_prior, False),
            weak=getattr(pa, "weak", None),
        )
    return BodyMotion(
        frames=prev.frames,
        timestamps=prev.timestamps,
        skeleton=prev.skeleton,
        fps=prev.fps,
        contact_channels=list(prev.contact_channels),
        actors=actors,
        world_shift=list(prev.world_shift),
    )


# ---------------------------------------------------------------------------
# Stage state kept next to a result (contacts: the root offsets a range re-run replays)
# ---------------------------------------------------------------------------


def fingerprint(bm: BodyMotion) -> str:
    """Hash of a motion result's rotations and roots as stored (float32)."""
    h = hashlib.sha1()
    for r in sorted(bm.actors):
        a = bm.actors[r]
        h.update(r.encode())
        h.update(np.ascontiguousarray(np.asarray(a.rot, np.float32)).tobytes())
        h.update(np.ascontiguousarray(np.asarray(a.root, np.float32)).tobytes())
    return h.hexdigest()


def write_state(work_dir: Path, result: BodyMotion, arrays: dict[str, np.ndarray]) -> None:
    """``range_state.npz`` in the stage's work folder (kept with the result by the runner)."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(work / STATE_FILE, fingerprint=np.array(fingerprint(result)), **arrays)


def read_state(ctx: Ctx, stage: str, prev: BodyMotion) -> dict[str, np.ndarray] | None:
    """The state the run that made ``prev`` wrote: ``ctx.previous_dir/range_state.npz`` (the runner
    sets it); direct callers without it: the newest matching one among the stage's stored versions
    under ``ctx.shot_dir/stages/<stage>/``. It must carry ``prev``'s fingerprint; None otherwise."""
    want = fingerprint(prev)
    pdir = getattr(ctx, "previous_dir", None)
    if pdir is not None:
        files = [Path(pdir) / STATE_FILE]
    else:
        files = sorted((Path(ctx.shot_dir) / "stages" / stage).glob(f"v*/{STATE_FILE}"), reverse=True)
    for f in files:
        try:
            with np.load(f) as z:
                if str(z["fingerprint"]) == want:
                    return {k: z[k] for k in z.files if k != "fingerprint"}
        except (OSError, ValueError, KeyError):
            continue
    return None
