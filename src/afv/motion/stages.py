"""CPU stages of module E: ``pair``, ``contacts`` and ``smooth`` (registered as impl "default").

Each stage returns a ``BodyMotion`` and also writes its QC into the run's work folder:
``qc.json`` (summary per fighter, the NFR numbers) and ``qc_frames.npz`` (per-frame metrics,
keys ``<role>/<metric>``) for the Blender confidence strip. QC always covers the whole result.

``contacts`` and ``smooth`` (and ``hands_face``) support range re-runs (``afv.motion.ranges``):
with ``ctx.frame_range`` and ``ctx.previous`` they solve the whole shot again (contacts with the
root offsets outside the edited zone held at the previous run's, read from the ``range_state.npz``
every contacts run writes) and blend that into the previous result around the range. Without a
usable previous result (other frames or fighters, no state) they run the whole shot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from afv.core.skeletons import get_skeleton
from afv.core.stage import Ctx, StageError, register
from afv.core.types import BodyMotion, CameraTrack, LicenceRecord, Result
from afv.motion.contacts import (
    ContactParams,
    Floor,
    FrozenContacts,
    actor_skeleton,
    apply_foot_targets,
    clean_contacts,
    contact_masks,
    floor_from_contacts,
    segments,
)
from afv.motion.fk import forward_kinematics
from afv.motion.ik import REGIONS, get_roles, subtree
from afv.motion.pair import PairParams, pair_fighters
from afv.motion.pins import pin_errors, stage_pins
from afv.motion.ranges import (
    BLEND,
    CONTACTS_EDIT,
    CONTACTS_EXTEND,
    blend_into_previous,
    check_previous,
    fingerprint,
    range_indices,
    range_plan,
    range_summary,
    read_state,
    smooth_margin,
    write_state,
    zone_plan,
)
from afv.motion.smooth import SmoothParams, half_windows, smooth_motion
from afv.qc.metrics import per_frame_metrics, summarise


def _body(inputs: dict[str, Result], name: str) -> BodyMotion:
    res = inputs.get(name)
    if not isinstance(res, BodyMotion):
        raise StageError(f"expected a body_motion result from stage {name!r}, got {type(res).__name__}")
    if not res.actors:
        raise StageError(f"stage {name!r} result has no fighters")
    return res


def _write_qc(
    ctx: Ctx, bm: BodyMotion, floor: Floor | None, extra: dict[str, Any], ref: BodyMotion | None = None
) -> None:
    roles = get_roles(get_skeleton(bm.skeleton))
    metrics = per_frame_metrics(bm, roles, floor, ref=ref)
    summary = {
        "fighters": summarise(metrics, roles),
        "world_shift": [float(v) for v in bm.world_shift],
        **extra,
    }
    if floor is not None:
        summary["floor"] = floor.to_dict()
    work = Path(ctx.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    (work / "qc.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        work / "qc_frames.npz",
        **{f"{r}/{k}": np.asarray(v, dtype=np.float32) for r, m in metrics.items() for k, v in m.items()},
    )


def _same_plants(a: BodyMotion, b: BodyMotion, role: str) -> bool:
    x, y = a.actors[role].contact, b.actors[role].contact
    return x is not None and y is not None and np.array_equal(np.asarray(x) > 0.5, np.asarray(y) > 0.5)


def _repin_ramp(
    prev: BodyMotion, new: BodyMotion, out: BodyMotion, plan: Any, floor: Floor, params: ContactParams
) -> BodyMotion:
    """Re-pin planted feet where a range re-run's result mixes two solutions: on the blend ramp (a
    blend of both) each plant is held frame by frame on the planted joints of the side it was taken
    from (``merge_masks``: new plants touch the zone); a plant that runs past the kept frames (longer
    than the zone may grow) continues as the previous result's there, so over all its kept frames it
    is held where the previous result has it just outside them. Leg IK only, as smooth re-pins."""
    roles = get_roles(get_skeleton(out.skeleton))
    n = out.num_frames
    ramp = (plan.weights > 0.0) & (plan.weights < 1.0)
    blend = max(0, int(params.blend_frames))
    for r, a in out.actors.items():
        pa, na = prev.actors[r], new.actors[r]
        diff = np.any(a.rot != pa.rot, axis=(1, 2)) | np.any(a.root != pa.root, axis=1)
        if not (diff & (plan.weights > 0.0)).any():
            continue
        sk = actor_skeleton(out, a)
        ref_pos, ref_wrot = forward_kinematics(sk, pa.rot, pa.root)
        new_pos, new_wrot = forward_kinematics(sk, na.rot, na.root)
        masks = contact_masks(out, a, roles)
        apply = ramp.copy()
        for c in roles.contacts:
            joints = subtree(sk, sk.index(roles.legs[c.leg].thigh))
            for s0, e0 in segments(masks[c.channel]):
                if not (s0 < plan.z1 and e0 > plan.z0):
                    continue  # a plant of the previous result: its own planted joints (default)
                if s0 < plan.k0 or e0 > plan.k1:  # spliced: held where the previous result has it
                    anchor = plan.k1 if e0 > plan.k1 else plan.k0 - 1  # first frame outside, still planted
                    lo, hi = max(plan.k0, s0), min(plan.k1, e0)
                    ref_pos[lo:hi, joints] = ref_pos[anchor, joints]
                    ref_wrot[lo:hi, joints] = ref_wrot[anchor, joints]
                    apply[max(plan.k0, s0 - blend) : min(plan.k1, e0 + blend)] = True
                    continue
                ref_pos[s0:e0, joints] = new_pos[s0:e0, joints]
                ref_wrot[s0:e0, joints] = new_wrot[s0:e0, joints]
        apply &= diff
        apply[: plan.k0] = False
        apply[plan.k1 :] = False
        if not apply.any() or n == 0:
            continue
        fix = apply_foot_targets(
            sk, roles, a.rot, a.root, masks, floor, params, allow_pelvis=True, soft_floor=False,
            pin_mode="ref", ref_fk=(ref_pos, ref_wrot),
        )  # fmt: skip
        rot = np.array(a.rot, copy=True)
        rot[apply] = np.asarray(fix.rot, np.float32)[apply]
        root = np.array(a.root, copy=True)
        root[apply] = np.asarray(fix.root, np.float32)[apply]
        a.rot, a.root = rot, root
    return out


@register("default")
class PairStage:
    name: ClassVar[str] = "pair"
    # 3: per-episode push order, local floor/depth rows (fighter B floor)
    # 4: QC only (result unchanged): hover = the foot's lowest sole point, foot_* per-frame metrics
    # 5: [Z, round 3] weak 2D evidence down-weighted in the depth solve; 6: ActorMotion.weak carried along
    version: ClassVar[str] = "6"
    edits: ClassVar[tuple[str, ...]] = ()
    env: ClassVar[str] = "cpu"
    inputs: ClassVar[tuple[str, ...]] = ("pose3d", "camera")
    licences: ClassVar[tuple[LicenceRecord, ...]] = ()
    Params: ClassVar[type[BaseModel]] = PairParams
    weight_hash: ClassVar[str] = ""

    def run(self, ctx: Ctx, inputs: dict[str, Result], params: BaseModel) -> Result:
        bm = _body(inputs, "pose3d")
        cam = inputs.get("camera")
        cam = cam if isinstance(cam, CameraTrack) and len(cam.cam_pos) else None
        ctx.progress(0.1, "pair: depth and floor")
        out, rep = pair_fighters(bm, cam, PairParams.model_validate(params.model_dump()))
        _write_qc(
            ctx,
            out,
            rep.floor,
            {
                "camera_used": cam is not None,
                "floor_shift_m": rep.floor_shift,
                "max_depth_change_m": {
                    r: float(np.abs(d).max()) if d.size else 0.0 for r, d in rep.depth_change.items()
                },
                "max_overlap_before_cm": float(rep.penetration_before.max() * 100) if rep.push.size else 0.0,
                "max_overlap_after_cm": float(rep.penetration_after.max() * 100) if rep.push.size else 0.0,
            },
        )
        ctx.progress(1.0, "pair: done")
        return out


@register("default")
class ContactsStage:
    name: ClassVar[str] = "contacts"
    # 4: capped pin raise, knee-space blend, edge-slide trim, plant counts
    # 5: flat plants levelled (toe on the floor), pivot toe plants split, toe-only frames pivot about
    #    the ball, foot-level float rule, foot orientation of neighbouring plants kept apart; QC hover
    #    = the foot's lowest sole point; after review: eased pivot correction, forward knee bend,
    #    step / edge-pop / knee-snap unplant rules, gap orientation slerp, foot_pops QC
    # 6: [M, round 5] root-correction rate limit as a convex one-sided penalty (path-independent: one
    #    candidate plant frame more or less no longer moves the root by up to 8 cm); range re-runs'
    #    ramp re-pin reaches nearly straight legs exactly (IK reach); max_plant_step 3 -> 2.5 cm/s
    version: ClassVar[str] = "6"
    edits: ClassVar[tuple[str, ...]] = ("contacts",)
    env: ClassVar[str] = "cpu"
    inputs: ClassVar[tuple[str, ...]] = ("pair", "camera")
    licences: ClassVar[tuple[LicenceRecord, ...]] = ()
    Params: ClassVar[type[BaseModel]] = ContactParams
    weight_hash: ClassVar[str] = ""
    supports_range: ClassVar[bool] = True

    def run(self, ctx: Ctx, inputs: dict[str, Result], params: BaseModel) -> Result:
        bm = _body(inputs, "pair")
        cam = inputs.get("camera")
        cam = cam if isinstance(cam, CameraTrack) and len(cam.cam_pos) else None
        painted = ctx.edits.get("contacts", ctx.edits.get("contacts.json"))
        p = ContactParams.model_validate(params.model_dump())
        frozen, prev, shift, floor = self._range(ctx, bm, p)
        ctx.progress(0.1, "contacts: detect and pin" + (" (range)" if frozen else ""))
        plan = None
        try:
            if frozen is None or prev is None:
                out, rep = clean_contacts(bm, p, painted=painted, camera=cam)
            else:
                full, rep = clean_contacts(
                    bm, p, painted=painted, floor=floor, camera=cam, frozen=frozen, zero_shift=shift
                )
                assert rep.zone is not None
                for r, info in rep.actors.items():  # nothing changed for this fighter: keep it as it was
                    if info.replayed and _same_plants(full, prev, r):
                        full.actors[r] = prev.actors[r]
                plan = zone_plan(bm.num_frames, rep.zone[0], rep.zone[1], BLEND)
                channels = [c.channel for c in get_roles(get_skeleton(bm.skeleton)).contacts]
                out = blend_into_previous(prev, full, plan, plant_channels=channels)
                out = _repin_ramp(prev, full, out, plan, rep.floor, p)
        except (KeyError, ValueError) as e:
            raise StageError(f"contacts: {e}") from e
        write_state(
            ctx.work_dir,
            out,
            {
                "input": np.array(fingerprint(bm)),
                "params": np.array(json.dumps(p.model_dump(), sort_keys=True)),
                "floor": np.r_[rep.floor.normal, rep.floor.offset],  # output world
                "floor_shift": np.array(rep.floor_shift),
                **{f"{r}/offset_iters": np.asarray(i.offset_iters, np.float64) for r, i in rep.actors.items()
                   if i.offset_iters is not None},
                **{f"{r}/mask_iters": np.asarray(i.mask_iters, bool) for r, i in rep.actors.items()
                   if i.mask_iters is not None},
            },
        )  # fmt: skip
        roles = get_roles(get_skeleton(out.skeleton))
        masks = {r: contact_masks(out, a, roles) for r, a in out.actors.items()}  # the merged result's
        plants = {r: {ch: len(segments(m)) for ch, m in mm.items()} for r, mm in masks.items()}
        extra_range = {}
        if plan is not None and prev is not None:
            # changed plants of the result that run past the kept frames (spliced onto the previous
            # result's part of the same plant, where a pin that moved would slide)
            n_spliced = 0
            for r, mm in masks.items():
                a, b = out.actors[r], prev.actors[r]
                moved = np.any(a.rot != b.rot, axis=(1, 2)) | np.any(a.root != b.root, axis=1)
                for m in mm.values():
                    for s0, e0 in segments(m):
                        crosses = s0 < plan.z1 and e0 > plan.z0 and (s0 < plan.k0 or e0 > plan.k1)
                        n_spliced += int(crosses and bool(moved[s0:e0].any()))
            extra_range["spliced_plants"] = n_spliced
        _write_qc(
            ctx,
            out,
            rep.floor,
            {
                "plants": plants,
                "painted": bool(painted),
                "floor_shift_m": rep.floor_shift,
                "camera_used": cam is not None,
                "filled_frames": rep.filled_frames,
                "root_offset_max_m": {
                    r: float(np.linalg.norm(i.root_offset, axis=-1).max())
                    if i.root_offset is not None
                    else 0.0
                    for r, i in rep.actors.items()
                },
                "pelvis_drop_mean_m": {
                    r: float(i.fix.pelvis_drop.mean()) for r, i in rep.actors.items() if i.fix
                },
                "plant_fix_counts": {
                    r: {
                        k: int(getattr(i, k))
                        for k in (
                            "raised_plants",
                            "capped_plants",
                            "dropped_plants",
                            "miss_unplanted_frames",
                            "unsolved_unplanted_frames",
                            "slide_trimmed_frames",
                            "pivot_split_frames",
                            "held_holes_filled",
                        )
                    }
                    for r, i in rep.actors.items()
                },
                "pin_miss_p95_m": {
                    r: float(
                        np.percentile(
                            np.concatenate([m[m > 0] for m in i.fix.miss.values()] or [np.zeros(1)]), 95
                        )
                    )
                    if i.fix and any((m > 0).any() for m in i.fix.miss.values())
                    else 0.0
                    for r, i in rep.actors.items()
                },
                **range_summary(ctx, bm, plan, **extra_range),
            },
            ref=bm,
        )
        ctx.progress(1.0, "contacts: done")
        return out

    def _range(
        self, ctx: Ctx, bm: BodyMotion, p: ContactParams
    ) -> tuple[FrozenContacts | None, BodyMotion | None, float, Floor | None]:
        """The frozen-solve setup of a range re-run (None for a full run, with the reason reported)."""
        if ctx.frame_range is None:
            return None, None, 0.0, None
        prev = getattr(ctx, "previous", None)
        why = check_previous(prev, bm)
        state = read_state(ctx, self.name, prev) if why is None and isinstance(prev, BodyMotion) else None
        if why is None and state is None:
            why = "no range state (range_state.npz) for the previous result"
        if why is None and state is not None:
            if "input" in state and str(state["input"]) != fingerprint(bm):
                why = "the previous result was made from another input"
            elif "params" in state and str(state["params"]) != json.dumps(p.model_dump(), sort_keys=True):
                why = "the previous result was made with other params"
        if why is None and state is not None:
            k = max(1, p.root_iterations) if p.root_correction else 0
            for r in bm.actors:
                it = state.get(f"{r}/offset_iters")
                if it is None or (k and it.shape != (k, bm.num_frames, 3)):
                    why = f"the previous result's range state does not fit fighter {r}"
                    break
        if why is not None or state is None or not isinstance(prev, BodyMotion):
            ctx.progress(-1.0, f"contacts: {why}; re-solving the whole shot")
            return None, None, 0.0, None
        try:
            ia, ib = range_indices(bm.frames, ctx.frame_range)
        except ValueError as e:
            raise StageError(f"contacts: {e}") from e
        roles = get_roles(get_skeleton(prev.skeleton))
        # the previous run's floor and world, in the input's world (a range re-run must not re-fit them)
        shift = float(prev.world_shift[1]) - float(bm.world_shift[1])
        if "floor" in state:
            fl = np.asarray(state["floor"], dtype=np.float64)
            floor = Floor(fl[:3], float(fl[3]) + fl[1] * shift)
        elif p.floor_tilt:
            f_out = floor_from_contacts(prev, roles, p)
            floor = Floor(f_out.normal, f_out.offset + f_out.normal[1] * shift)
        else:
            floor = Floor.horizontal(-shift)
        frozen = FrozenContacts(
            editable=(ia - CONTACTS_EDIT, ib + CONTACTS_EDIT),
            offset_iters={r: state[f"{r}/offset_iters"] for r in bm.actors},
            mask_iters={r: state[f"{r}/mask_iters"] for r in bm.actors if f"{r}/mask_iters" in state},
            masks={r: contact_masks(prev, a, roles) for r, a in prev.actors.items()},
            max_extend=CONTACTS_EXTEND,
        )
        return frozen, prev, shift, floor


@register("default")
class SmoothStage:
    name: ClassVar[str] = "smooth"
    # 4: re-pins with the contacts v4 algorithm; 5: ... with v5; 6: pinned poses (edits/pins.json),
    # region_strength names validated, the window grows with strength > 1 (monotone sliders); the
    # default output (strength 1, no pins) is unchanged; 7: [M, round 5] the contact re-pin lowers the
    # pelvis for the IK's real reach, so planted joints follow contacts' exactly (no added planted step)
    version: ClassVar[str] = "7"
    edits: ClassVar[tuple[str, ...]] = ("pins",)
    supports_range: ClassVar[bool] = True
    env: ClassVar[str] = "cpu"
    inputs: ClassVar[tuple[str, ...]] = ("contacts",)
    inherit_params: ClassVar[dict[str, str]] = {"contact": "contacts"}  # re-pin as contacts did
    shot_fields: ClassVar[tuple[str, ...]] = ()  # works on the shot timeline; speed is export's business
    licences: ClassVar[tuple[LicenceRecord, ...]] = ()
    Params: ClassVar[type[BaseModel]] = SmoothParams
    weight_hash: ClassVar[str] = ""

    def run(self, ctx: Ctx, inputs: dict[str, Result], params: BaseModel) -> Result:
        bm = _body(inputs, "contacts")
        p = SmoothParams.model_validate(params.model_dump())
        pins = stage_pins(ctx, bm, self.name)
        # reach: contacts' + this smoother's (strength-grown) half window
        strengths = [p.strength * p.region_strength.get(r, 1.0) for r in (*REGIONS, "root")]
        k = int(half_windows(p, np.array(strengths), bm.num_frames).max())
        plan = range_plan(ctx, self.name, bm, smooth_margin(k), BLEND)
        prev = ctx.previous if plan is not None else None
        n_pins = sum(map(len, pins.values()))
        ctx.progress(0.1, "smooth" + (" (range)" if plan else "") + (f", {n_pins} pin(s)" if n_pins else ""))
        out, infos = smooth_motion(bm, p, pins=pins)
        if plan is not None and isinstance(prev, BodyMotion):
            out = blend_into_previous(prev, out, plan)
            base = get_skeleton(bm.skeleton)
            for r, ps in pins.items():  # pins QC on the merged result
                pin_errors(
                    base, bm.actors[r], out.actors[r].rot, out.actors[r].root, ps, infos[r].pins, bm.fps
                )
        else:
            plan = None
        _write_qc(
            ctx,
            out,
            None,
            {
                "strikes": {
                    r: [
                        {"effector": s.effector, "frame": int(bm.frames[s.frame]), "speed_m_s": s.speed}
                        for s in i.strikes
                    ]
                    for r, i in infos.items()
                },
                "pops_removed": {r: i.rotation_pops + i.root_pops for r, i in infos.items()},
                "pins": [rep.to_dict() for i in infos.values() for rep in i.pins],
                **range_summary(ctx, bm, plan, half_window=k),
            },
            ref=bm,
        )
        ctx.progress(1.0, "smooth: done")
        return out
