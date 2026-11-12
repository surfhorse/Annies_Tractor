"""Range re-runs (ctx.frame_range + ctx.previous) of contacts, smooth and hands_face on real GEM-X
motion: no-op ranges reproduce the previous result, edits stay local, margins nest, QC covers the
whole result, fallbacks, speed."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from real_metrics import load_clip, plant_speed, positions

import afv.motion.hands_face  # registers the stage
import afv.motion.stages  # noqa: F401
from afv.core.stage import Ctx, StageError, get_stage
from afv.core.types import BodyMotion, load_result, save_result
from afv.motion.ranges import (
    BLEND,
    CONTACTS_MAX_MARGIN,
    HANDS_BLEND,
    HANDS_MARGIN,
    SMOOTH_MAX_HALF_WINDOW,
    plan_range,
    slice_motion,
    smooth_margin,
    zone_plan,
)
from afv.motion.rotations import from_rotvec, qmul


def test_plan_weights():
    frames = np.arange(100, 200)
    plan = plan_range(frames, (140, 160), margin=16, blend=12)
    w = plan.weights
    assert (plan.z0, plan.z1, plan.k0, plan.k1) == (40 - 4, 60 + 4, 40 - 16, 60 + 16)
    assert np.all(w[: plan.k0] == 0) and np.all(w[plan.k1 :] == 0) and np.all(w[36:64] == 1)
    ramp = w[plan.k0 : plan.z0]
    assert np.all(np.diff(ramp) > 0) and 0 < ramp[0] < 0.01 and ramp[-1] < 1
    assert np.allclose(w[plan.k0 : plan.z0], w[plan.k1 - 1 : plan.z1 - 1 : -1])  # symmetric
    p2 = zone_plan(100, -5, 3, 12)
    assert p2.k0 == 0 and p2.z0 == 0 and p2.k1 == 15
    with pytest.raises(ValueError, match="outside the shot"):
        plan_range(frames, (300, 320), 16, 12)
    with pytest.raises(ValueError, match="empty"):
        plan_range(frames, (150, 150), 16, 12)


def test_margins_nest():
    """Each downstream stage's full-weight zone covers what its upstream stage may change plus its own
    reach (smooth: its strength-grown half window), up to strength 4."""
    for k in (7, 14, SMOOTH_MAX_HALF_WINDOW):
        assert smooth_margin(k) - BLEND >= CONTACTS_MAX_MARGIN + k + 1
    assert HANDS_MARGIN - HANDS_BLEND >= smooth_margin(SMOOTH_MAX_HALF_WINDOW)


class _Shot:
    """A shot folder laid out like the runner's (stages/<name>/vNNNN), so a range re-run finds the
    state its previous result left (range_state.npz)."""

    def __init__(self, root: Path):
        self.dir = Path(root)
        self.n = 0

    def run(self, name, inputs, edits=None, frame_range=None, previous=None, params=None, messages=None):
        cls = get_stage(name, "default")
        self.n += 1
        work = self.dir / "stages" / name / f"v{self.n:04d}"
        ctx = Ctx(
            shot_dir=self.dir, work_dir=work, edits=edits or {}, frame_range=frame_range, previous=previous
        )
        if messages is not None:
            ctx.progress = lambda f, m: messages.append(m)
        res = cls().run(ctx, inputs, cls.Params.model_validate(params or {}))
        save_result(res, work / "result.npz")  # what the runner stores and later stages load
        out = load_result(work / "result.npz")
        out.qc = json.loads((work / "qc.json").read_text())  # type: ignore[attr-defined]
        return out


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    bm, cam = load_clip("ufc_cross")  # real GEM-X fixture (120 frames, two fighters)
    shot = _Shot(tmp_path_factory.mktemp("shot"))
    out = {"shot": shot, "camera": cam}
    out["pair"] = shot.run("pair", {"pose3d": bm, "camera": cam})
    out["contacts"] = shot.run("contacts", {"pair": out["pair"], "camera": cam})
    out["smooth"] = shot.run("smooth", {"contacts": out["contacts"]})
    out["hands_face"] = shot.run("hands_face", {"smooth": out["smooth"]})
    return out


def _joint_diff_mm(a: BodyMotion, b: BodyMotion) -> dict[str, np.ndarray]:
    return {
        r: np.linalg.norm(positions(a, r)[1] - positions(b, r)[1], axis=-1).max(axis=1) * 1000
        for r in a.actors
    }


def _changed(a: BodyMotion, b: BodyMotion) -> np.ndarray:
    out = np.zeros(a.num_frames, bool)
    for r in a.actors:
        x, y = a.actors[r], b.actors[r]
        out |= np.any(x.rot != y.rot, axis=(1, 2)) | np.any(x.root != y.root, axis=1)
        if x.contact is not None:
            out |= np.any(x.contact != y.contact, axis=1)
    return out


@pytest.mark.parametrize("rng", [(0, 10), (40, 41), (50, 70), (110, 120)])
def test_noop_contacts_range_reproduces_previous(chain, rng):
    """A range re-run with nothing changed equals the previous result (the root offsets are replayed)."""
    f0 = int(chain["contacts"].frames[0])
    c1 = chain["shot"].run("contacts", {"pair": chain["pair"], "camera": chain["camera"]},
                           frame_range=(f0 + rng[0], f0 + rng[1]), previous=chain["contacts"])  # fmt: skip
    assert "range" in c1.qc
    assert max(float(v.max()) for v in _joint_diff_mm(c1, chain["contacts"]).values()) <= 1.0
    for r in c1.actors:
        assert np.array_equal(c1.actors[r].contact, chain["contacts"].actors[r].contact)
    assert c1.qc["plants"] == chain["contacts"].qc["plants"]  # QC of the whole result


def test_contacts_edit_stays_local_and_planted(chain):
    shot, prev = chain["shot"], chain["contacts"]
    f0 = int(prev.frames[0])
    a, b = f0 + 50, f0 + 70
    paint = {"spans": [{"role": "A", "start": a + 4, "end": a + 14, "foot": "right", "state": "free"}]}
    t = time.perf_counter()
    c1 = shot.run(
        "contacts", {"pair": chain["pair"], "camera": chain["camera"]}, {"contacts": paint}, (a, b), prev
    )
    s1 = shot.run("smooth", {"contacts": c1}, {}, (a, b), chain["smooth"])
    assert time.perf_counter() - t < 10.0  # spec: a contacts+smooth re-run of a 2 s range <= 30 s
    rg = c1.qc["range"]
    k0, k1 = rg["changed"][0] - f0, rg["changed"][1] - f0
    ch = _changed(c1, prev)
    assert not ch[:k0].any() and not ch[k1:].any() and ch[54:64].any()
    assert k0 >= 50 - CONTACTS_MAX_MARGIN and k1 <= 70 + CONTACTS_MAX_MARGIN
    assert np.array_equal(c1.actors["B"].rot, prev.actors["B"].rot)  # B's root solve is replayed
    ri = c1.contact_channels.index("RightFoot")
    assert not c1.actors["A"].contact[54:64, ri].any()  # the painted span is free
    for r in c1.actors:
        assert float(plant_speed(c1, r).max()) * 100 <= 3.0  # cm/s, every planted frame
        assert float(plant_speed(s1, r).max()) * 100 <= 3.0
    # smooth's range result equals a full smooth run on the new contacts (margins nest)
    sf = shot.run("smooth", {"contacts": c1})
    assert max(float(v.max()) for v in _joint_diff_mm(s1, sf).values()) <= 1.0
    # ... and the contacts result is close to a full re-run with the edit where it was painted. A full
    # re-run's root solve is global: the edit moves it 14-15 cm even 40 frames away (frames 0-20 and
    # 100-120 here); in the painted span the range result stays nearer to it than that drift.
    cf = shot.run("contacts", {"pair": chain["pair"], "camera": chain["camera"]}, {"contacts": paint})
    d = _joint_diff_mm(c1, cf)["A"]
    far = _joint_diff_mm(cf, prev)["A"]
    drift = min(float(far[:20].max()), float(far[100:].max()))
    assert float(d[54:64].max()) < max(30.0, drift), (d[54:64].max(), drift)


@pytest.mark.parametrize("strength", [1.0, 4.0])
def test_noop_smooth_range_at_strength(chain, strength):
    shot = chain["shot"]
    full = shot.run("smooth", {"contacts": chain["contacts"]}, params={"strength": strength})
    n = full.num_frames
    for a, b in ((5, 6), (50, 70), (n - 10, n)):
        f0 = int(full.frames[0])
        r1 = shot.run("smooth", {"contacts": chain["contacts"]}, {}, (f0 + a, f0 + b), full,
                      params={"strength": strength})  # fmt: skip
        assert max(float(v.max()) for v in _joint_diff_mm(r1, full).values()) <= 1.0, (strength, a, b)


def test_smooth_range_pins_qc_covers_every_pin(chain):
    shot, prev = chain["shot"], chain["smooth"]
    from afv.core.skeletons import get_skeleton
    from afv.core.types import Take
    from afv.export.exporter import take_motion
    from afv.export.presets import get_preset

    a0 = prev.actors["A"]
    pr = get_preset("blender")
    take = Take(skeleton=prev.skeleton, fps=prev.fps, speed_factor=1.0, rot=a0.rot, root=a0.root,
                root_motion="world", role="A", rest_offsets=a0.rest_offsets)  # fmt: skip
    m0 = take_motion(take, pr, skeleton=get_skeleton(prev.skeleton))
    f0 = int(prev.frames[0])
    lf = m0.rig.index("LeftForeArm")
    pins = {"pins": [
        {"role": "A", "frame": f0 + i, "radius": 6, "preset": "blender",
         "bones": {"LeftForeArm": qmul(m0.rot[i, lf], from_rotvec([0, 0.3, 0])).tolist()}}
        for i in (20, 60, 100)
    ]}  # fmt: skip
    full = shot.run("smooth", {"contacts": chain["contacts"]}, {"pins": pins})
    r1 = shot.run("smooth", {"contacts": chain["contacts"]}, {"pins": pins}, (f0 + 55, f0 + 65), full)
    assert [p["frame"] for p in r1.qc["pins"]] == [f0 + 20, f0 + 60, f0 + 100]
    assert all(p["max_error_deg"] < 0.1 for p in r1.qc["pins"])
    assert max(float(v.max()) for v in _joint_diff_mm(r1, full).values()) <= 1.0


def test_range_falls_back_to_full_run(chain, tmp_path):
    msgs: list[str] = []
    full = chain["smooth"]
    other = slice_motion(full, 0, full.num_frames - 1)  # a previous result with other frames
    shot = _Shot(tmp_path)
    rng = (int(full.frames[10]), int(full.frames[20]))
    res = shot.run("smooth", {"contacts": chain["contacts"]}, {}, rng, other, messages=msgs)
    assert any("re-solving the whole shot" in m for m in msgs) and "range" not in res.qc
    assert not _changed(res, full).any()
    # contacts without the previous run's range state (another shot folder): whole shot
    msgs.clear()
    f0 = int(full.frames[0])
    c = shot.run("contacts", {"pair": chain["pair"], "camera": chain["camera"]}, {}, (f0 + 10, f0 + 20),
                 chain["contacts"], messages=msgs)  # fmt: skip
    assert any("no range state" in m for m in msgs) and "range" not in c.qc
    with pytest.raises(StageError, match="outside the shot"):
        chain["shot"].run("smooth", {"contacts": chain["contacts"]}, {}, (10**6, 10**6 + 5), full)


def test_hands_face_range_and_hands_off(chain):
    shot = chain["shot"]
    cls = get_stage("hands_face", "default")
    assert cls.supports_range and set(cls.edits) == {"hands", "pins"}
    f0 = int(chain["smooth"].frames[0])
    a, b = f0 + 40, f0 + 60
    hands = {"spans": [{"role": "B", "start": a + 5, "end": a + 15, "side": "left"}]}
    h1 = shot.run("hands_face", {"smooth": chain["smooth"]}, {"hands": hands}, (a, b), chain["hands_face"])
    full = shot.run("hands_face", {"smooth": chain["smooth"]}, {"hands": hands})
    assert not _changed(h1, full).any()  # the span lies inside the range: same as a full run
    ch = np.flatnonzero(_changed(h1, chain["hands_face"]))
    assert ch.min() >= 45 - 6 and ch.max() < 55 + 6
    assert h1.qc["summary"]["hands_off_frames"] == {"A": 0, "B": 10}


def test_range_rerun_speed_on_ten_seconds(tmp_path):
    """Smooth on 12 s (360 frames, two fighters) through the stage interface, and a 2 s range."""
    bm, _ = load_clip("ufc_cross")
    long = _tile(bm, 3)  # 360 frames from the 120-frame fixture (no camera for the longer clip)
    shot = _Shot(tmp_path)
    cont = shot.run("contacts", {"pair": long})
    t = time.perf_counter()
    full = shot.run("smooth", {"contacts": cont})
    t_full = time.perf_counter() - t
    t = time.perf_counter()
    shot.run("contacts", {"pair": long}, {}, (int(cont.frames[150]), int(cont.frames[210])), cont)
    shot.run("smooth", {"contacts": cont}, {}, (int(cont.frames[150]), int(cont.frames[210])), full)
    t_range = time.perf_counter() - t
    assert t_full < 5.0 and t_range < 10.0, (t_full, t_range)


def _tile(bm: BodyMotion, k: int) -> BodyMotion:
    """``bm`` played forward, backward, forward ... (continuous) for a longer clip."""
    idx = np.concatenate([np.arange(bm.num_frames)[:: 1 if i % 2 == 0 else -1] for i in range(k)])
    out = slice_motion(bm, 0, bm.num_frames)
    n = len(idx)
    for a in out.actors.values():
        for name in ("rot", "root", "conf", "contact", "root_prior", "weak"):
            v = getattr(a, name)
            if v is not None:
                setattr(a, name, np.asarray(v)[idx])
    out.frames = np.arange(n) + int(bm.frames[0])
    out.timestamps = out.frames / bm.fps
    return out


def _slides(bm: BodyMotion, thr: float = 0.03) -> set[tuple[str, str, int]]:
    """(role, channel, frame) of planted frame pairs whose joint moves faster than ``thr`` m/s."""
    from afv.motion.contacts import contact_masks
    from afv.motion.ik import get_roles

    out = set()
    for r, a in bm.actors.items():
        sk, pos = positions(bm, r)
        roles = get_roles(sk)
        masks = contact_masks(bm, a, roles)
        for c in roles.contacts:
            m = masks[c.channel]
            v = np.linalg.norm(np.diff(pos[:, sk.index(c.joint)], axis=0), axis=-1) * bm.fps
            out |= {(r, c.channel, int(i)) for i in np.flatnonzero(m[1:] & m[:-1] & (v > thr))}
    return out


@pytest.mark.slow
@pytest.mark.parametrize("clip", ["ufc_cross", "close_boxing", "ref_walk", "who_am_i"])
def test_painted_span_sweep_adds_no_sliding_plant(clip, tmp_path):
    """The review sweep on real motion: a 10-frame painted Free span in a 14-frame range every 15
    frames, both feet and both fighters; contacts + smooth range re-runs never leave a planted foot
    moving faster than 3 cm/s where a full re-run with the same edit has none."""
    bm, cam = load_clip(clip)
    shot = _Shot(tmp_path)
    pair = shot.run("pair", {"pose3d": bm, "camera": cam})
    c0 = shot.run("contacts", {"pair": pair, "camera": cam})
    s0 = shot.run("smooth", {"contacts": c0})
    n, f0 = c0.num_frames, int(c0.frames[0])
    bad = []
    for a in range(10, n - 25, 15):
        for foot in ("left", "right"):
            for role in ("A", "B"):
                span = {"role": role, "start": f0 + a + 2, "end": f0 + a + 12, "foot": foot, "state": "free"}
                ed = {"contacts": {"spans": [span]}}
                rng = (f0 + a, f0 + a + 14)
                c1 = shot.run("contacts", {"pair": pair, "camera": cam}, ed, rng, c0)
                s1 = shot.run("smooth", {"contacts": c1}, {}, rng, s0)
                cf = shot.run("contacts", {"pair": pair, "camera": cam}, ed)
                sf = shot.run("smooth", {"contacts": cf})
                for res, ref in ((c1, cf), (s1, sf)):
                    ok = _slides(ref)
                    new = [
                        x for x in _slides(res) if not any((x[0], x[1], x[2] + d) in ok for d in (-1, 0, 1))
                    ]
                    if new:
                        bad.append((a, foot, role, new[:3]))
    assert not bad, bad
