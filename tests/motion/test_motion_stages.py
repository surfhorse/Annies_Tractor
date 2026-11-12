"""Stage registration, the pair -> contacts -> smooth chain through the Stage interface, and NFR-2."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from motion_synth import GEMX_CHANNELS, body, corrupt, model_probs, telephoto_camera, walk

import afv.motion.stages  # noqa: F401  (registers the stages)
from afv.core.stage import Ctx, Stage, get_stage
from afv.core.types import BodyMotion, load_result, save_result
from afv.motion.contacts import clean_contacts
from afv.motion.smooth import smooth_motion


def _two_fighters(seconds: float, seed: int = 0) -> BodyMotion:
    rng = np.random.default_rng(seed)
    a, ma = walk(seconds=seconds, origin=(-0.8, 0.0), heading=np.pi / 2, speed=0.4)
    b, mb = walk(seconds=seconds, origin=(0.8, 0.4), heading=-np.pi / 2, speed=0.4)
    a, b = corrupt(a, rng), corrupt(b, rng)
    a.contact, b.contact = model_probs(ma, rng), model_probs(mb, rng)
    return body({"A": a, "B": b}, GEMX_CHANNELS)


def test_stages_registered():
    for name in ("pair", "contacts", "smooth"):
        cls = get_stage(name, "default")
        assert cls.env == "cpu" and cls.weight_hash == "" and cls.licences == ()
        assert isinstance(cls(), Stage)


def test_chain_through_stage_interface(tmp_path):
    bm = _two_fighters(2.0)
    cam = telephoto_camera(bm.num_frames)
    results: dict = {"pose3d": bm, "camera": cam}
    for name in ("pair", "contacts", "smooth"):
        cls = get_stage(name, "default")
        work = tmp_path / name
        ctx = Ctx(shot_dir=tmp_path, work_dir=work, edits={})
        res = cls().run(ctx, {k: results[k] for k in cls.inputs}, cls.Params())
        assert isinstance(res, BodyMotion) and set(res.actors) == {"A", "B"}
        save_result(res, work / "result.npz")
        back = load_result(work / "result.npz")
        assert isinstance(back, BodyMotion) and back.contact_channels == res.contact_channels
        qc = json.loads((work / "qc.json").read_text())
        assert set(qc["fighters"]) == {"A", "B"}
        if name in ("pair", "contacts"):
            assert isinstance(qc["floor_shift_m"], float) and qc["floor"]["offset"] == pytest.approx(
                0.0, abs=1e-6
            )
        assert qc["world_shift"] == pytest.approx(list(back.world_shift), abs=1e-6)
        with np.load(work / "qc_frames.npz") as z:
            assert "A/skate" in z.files and z["A/skate"].shape == (bm.num_frames,)
        results[name] = back
    qc = json.loads((tmp_path / "smooth" / "qc.json").read_text())
    for r in ("A", "B"):
        assert qc["fighters"][r]["mean_contact_foot_speed_cm_s"] <= 3.0
        assert qc["fighters"][r]["pop_frames"] == 0


def test_contacts_stage_reads_painted_edits(tmp_path):
    bm = _two_fighters(2.0)
    cls = get_stage("contacts", "default")
    edits = {"contacts": {"A": [{"foot": "left", "start": 0, "end": 60, "state": "free"}]}}
    res = cls().run(Ctx(shot_dir=tmp_path, work_dir=tmp_path / "w", edits=edits), {"pair": bm}, cls.Params())
    ch = res.contact_channels
    assert res.actors["A"].contact[:, [ch.index("l_heel"), ch.index("l_toe")]].max() == 0.0
    assert res.actors["B"].contact[:, ch.index("l_heel")].max() == 1.0


def test_runtime_ten_seconds_two_fighters():
    bm = _two_fighters(10.0, seed=3)
    t0 = time.perf_counter()
    cont, _ = clean_contacts(bm)
    t1 = time.perf_counter()
    smooth_motion(cont)
    t2 = time.perf_counter()
    print(
        f"10 s x 2 fighters ({bm.num_frames} frames, {bm.actors['A'].rot.shape[1]} joints): "
        f"contacts {t1 - t0:.2f} s, smooth {t2 - t1:.2f} s"
    )
    assert t2 - t0 < 10.0  # NFR-2 asks <= 30 s for a 2 s re-run; this is 10 s of motion


def test_tiny_clips_do_not_crash():
    from afv.motion.retime import retime

    for n in (1, 2, 3, 5):
        bm = _two_fighters(n / 30.0 + 1e-6)
        assert bm.num_frames == n
        cont, _ = clean_contacts(bm)
        smooth_motion(cont)
        retime(cont, 24.0)
