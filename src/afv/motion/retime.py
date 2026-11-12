"""Retiming: speed factor and resampling from the 30 Hz shot timeline to the export frame rate.

``speed_factor`` < 1 is slow motion (0.5 plays the action at half speed, so the clip lasts twice
as long). Output frame ``i`` samples source time ``i / out_fps * speed_factor`` (relative to the
first source sample); the last output frame is the last one that stays within the source.

Rotations are interpolated with slerp per joint (shortest arc), root translation, confidence and
any other float channels linearly; plant channels stay binary: a sample is planted only when both
source frames around it are (:func:`resample_plants`). Optionally, planted feet are re-pinned after
resampling (``hold_contacts``): slerp of local rotations between two pinned poses does not keep the foot
exactly still in between, and slow-motion makes that visible. Re-pinning moves only the legs
(no root or pelvis offsets), so retiming at the source rate with speed 1 changes nothing.
"""

from __future__ import annotations

import numpy as np

from afv.core.types import ActorMotion, BodyMotion
from afv.motion.rotations import make_continuous, slerp


def sample_times(n_src: int, src_fps: float, out_fps: float, speed_factor: float = 1.0) -> np.ndarray:
    """Fractional source frame positions [N] for each output frame."""
    if n_src < 1:
        return np.zeros(0)
    if speed_factor <= 0 or src_fps <= 0 or out_fps <= 0:
        raise ValueError("speed_factor, src_fps and out_fps must be positive")
    duration = (n_src - 1) / src_fps / speed_factor  # playback seconds
    n_out = int(np.floor(duration * out_fps + 1e-6)) + 1
    return np.minimum(np.arange(n_out) / out_fps * speed_factor * src_fps, n_src - 1)


def _interp_linear(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    w = (pos - i0).reshape(-1, *([1] * (x.ndim - 1)))
    return (1.0 - w) * x[i0] + w * x[i1]


def resample_rotations(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    q = make_continuous(np.asarray(rot, dtype=np.float64))
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(q) - 1)
    w = np.broadcast_to((pos - i0)[:, None], (len(pos), q.shape[1]))
    return make_continuous(slerp(q[i0], q[i1], w))


def resample_plants(contact: np.ndarray, pos: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Binary plant channels [F,C] (> 0.5 = planted) resampled at fractional source frames ``pos``.

    A sample is planted only when *both* source frames bracketing it are planted (a sample that
    falls exactly on a source frame takes that frame's state). A plant is a span of source frames
    pinned in place; the time between its last frame and the first lifted frame is already swing (the
    foot moves from the pin to the lifted pose), so a threshold on the interpolated value (``>= 0.5``)
    would call a sample half-way into that step planted and make the planted foot jump at every plant
    edge and across one-frame gaps between plants. A non-finite contact value counts as not
    planted; non-finite or out-of-range sample positions raise ``ValueError``.
    """
    pos = np.asarray(pos, dtype=np.float64)
    c = np.nan_to_num(np.asarray(contact, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0) > 0.5
    if pos.size and (not np.all(np.isfinite(pos)) or pos.min() < -eps or pos.max() > len(c) - 1 + eps):
        raise ValueError("resample_plants: sample positions must be finite and within the source")
    i0 = np.floor(pos + eps).astype(np.int64)
    i0 = np.clip(i0, 0, len(c) - 1)
    i1 = np.minimum(i0 + 1, len(c) - 1)
    on_frame = (np.abs(pos - i0) <= eps)[:, None]
    return np.where(on_frame, c[i0], c[i0] & c[i1])


def retime_actor(actor: ActorMotion, pos: np.ndarray, contact_mask: np.ndarray | None = None) -> ActorMotion:
    """Resample one fighter at fractional source frames ``pos``. ``contact_mask`` [C] marks which
    contact channels are binary plants (see :func:`resample_plants`); default: all. The others
    (e.g. GEM-X wrist probabilities) are interpolated linearly."""
    contact = None
    if actor.contact is not None:
        src = np.asarray(actor.contact, dtype=np.float64)
        c = _interp_linear(src, pos)
        binary = np.ones(c.shape[1], dtype=bool) if contact_mask is None else np.asarray(contact_mask)
        c[:, binary] = resample_plants(src[:, binary], pos).astype(np.float64)
        contact = c.astype(np.float32)
    return ActorMotion(
        rot=resample_rotations(actor.rot, pos).astype(np.float32),
        root=_interp_linear(np.asarray(actor.root, dtype=np.float64), pos).astype(np.float32),
        conf=None if actor.conf is None else _interp_linear(np.asarray(actor.conf), pos).astype(np.float32),
        contact=contact,
        rest_offsets=actor.rest_offsets,
        root_prior=None
        if actor.root_prior is None
        else _interp_linear(np.asarray(actor.root_prior, dtype=np.float64), pos).astype(np.float32),
    )


def retime(
    bm: BodyMotion,
    out_fps: float,
    speed_factor: float = 1.0,
    hold_contacts: bool = True,
    binary_channels: list[str] | None = None,
    contact_params=None,
) -> BodyMotion:
    """Resample a whole shot to ``out_fps`` with ``speed_factor``.

    The result's ``frames`` are output frame numbers 0..N-1 and ``timestamps`` the interpolated
    source times (so provenance in/out times stay in source seconds). ``binary_channels`` lists
    the contact channels to threshold (default: the skeleton's foot plant channels; probabilities
    such as GEM-X's wrist channels stay continuous).
    """
    from afv.core.skeletons import get_skeleton
    from afv.motion.ik import get_roles

    roles = get_roles(get_skeleton(bm.skeleton))
    src_fps = bm.fps or 30.0
    pos = sample_times(bm.num_frames, src_fps, out_fps, speed_factor)
    chans = list(bm.contact_channels)
    if binary_channels is None:
        binary_channels = [c.channel for c in roles.contacts]
    mask = np.array([c in binary_channels for c in chans], dtype=bool)
    actors = {r: retime_actor(a, pos, mask) for r, a in bm.actors.items()}
    out = BodyMotion(
        frames=np.arange(len(pos), dtype=np.int64),
        timestamps=_interp_linear(bm.timestamps, pos) if bm.num_frames else np.zeros(0),
        skeleton=bm.skeleton,
        fps=float(out_fps),
        contact_channels=chans,
        actors=actors,
        world_shift=list(bm.world_shift),
    )
    if hold_contacts and len(pos) > 1 and any(c.channel in chans for c in roles.contacts):
        from afv.motion.contacts import ContactParams, actor_skeleton
        from afv.motion.contacts import hold_contacts as _hold
        from afv.motion.fk import forward_kinematics

        # Leg IK only (no root/pelvis moves: idempotent), onto the *source* joints interpolated in
        # position space: slerped local rotations wobble planted feet between two pinned frames.
        ref_fk = {}
        for r, a in bm.actors.items():
            p_src, w_src = forward_kinematics(actor_skeleton(bm, a), a.rot, a.root)
            ref_fk[r] = (_interp_linear(p_src, pos), resample_rotations(w_src, pos))
        fixes = _hold(
            out,
            {r: a.rot for r, a in out.actors.items()},
            {r: a.root for r, a in out.actors.items()},
            roles,
            contact_params or ContactParams(),
            ref_fk=ref_fk,
        )
        for r, f in fixes.items():
            out.actors[r].rot = f.rot.astype(np.float32)
            out.actors[r].root = f.root.astype(np.float32)
    return out
