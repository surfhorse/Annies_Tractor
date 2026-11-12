"""Range re-runs (``afv run --from S --frames a:b``) and cancelling a run."""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest
from pydantic import BaseModel

import afv.models.camera  # noqa: F401  (the runner's re-level check imports it: register it now, not mid-test)
from afv.core.provenance import format_provenance, shot_provenance
from afv.core.runner import RunError, Runner, parse_frames
from afv.core.stage import register
from afv.core.types import ActorMotion, BodyMotion

T = 10
SEEN: list[dict] = []  # what the range stages were given


def _motion(root: np.ndarray, like: BodyMotion) -> BodyMotion:
    rot = like.actors["A"].rot
    return BodyMotion(
        frames=like.frames, timestamps=like.timestamps, skeleton=like.skeleton, fps=30.0,
        actors={"A": ActorMotion(rot=rot, root=root.astype(np.float32))},
    )  # fmt: skip


def _marks(ctx) -> np.ndarray:
    """edits/marks.json {"spans": [{"start", "end", "value"}]} as a per-frame offset."""
    out = np.zeros(T, np.float32)
    for sp in (ctx.edits.get("marks") or {}).get("spans", []):
        out[sp["start"] : sp["end"]] += sp["value"]
    return out


class RangeSmooth:
    name, version, env, inputs, licences, weight_hash = "smooth", "1", "cpu", ("pose3d",), (), ""
    supports_range = True
    edits = ("marks",)

    class Params(BaseModel):
        strength: float = 0.5
        hold: bool = False  # wait for the test's cancel event inside the stage

    gate: threading.Event | None = None

    def run(self, ctx, inputs, params):
        SEEN.append({"stage": "smooth", "range": ctx.frame_range, "previous": ctx.previous is not None})
        (ctx.work_dir / "range_state.npz").write_bytes(b"state")  # kept in the result folder
        if ctx.frame_range is not None:  # the folder of the result given as ctx.previous
            assert ctx.previous_dir is not None and (ctx.previous_dir / "range_state.npz").is_file()
            assert ctx.previous_dir == ctx.shot_dir / "stages" / "smooth" / ctx.previous_dir.name
        else:
            assert ctx.previous_dir is None
        if params.hold and RangeSmooth.gate is not None:
            RangeSmooth.gate.wait(5)
            ctx.progress(0.5, "after the gate")  # raises once the run is cancelled
        m = inputs["pose3d"]
        new = m.actors["A"].root * params.strength + _marks(ctx)[:, None]
        if ctx.frame_range is not None:
            a, b = ctx.frame_range
            out = ctx.previous.actors["A"].root.copy()
            out[a:b] = new[a:b]
            new = out
            (ctx.work_dir / "qc.json").write_text(json.dumps({"range": {"frames": [a, b], "solved": [a, b]}}))
        return _motion(new, m)


class RangeHands:
    name, version, env, inputs, licences, weight_hash = "hands_face", "1", "cpu", ("smooth",), (), ""
    supports_range = True

    class Params(BaseModel):
        offset: float = 1.0

    fallback = False  # solve the whole shot even when given a range (as when the previous does not fit)

    def run(self, ctx, inputs, params):
        SEEN.append({"stage": "hands_face", "range": ctx.frame_range, "previous": ctx.previous is not None})
        m = inputs["smooth"]
        new = m.actors["A"].root + params.offset
        if ctx.frame_range is not None and not RangeHands.fallback:
            a, b = ctx.frame_range
            out = ctx.previous.actors["A"].root.copy()
            out[a:b] = new[a:b]
            new = out
            (ctx.work_dir / "qc.json").write_text(json.dumps({"summary": {"range": {"frames": [a, b]}}}))
        return _motion(new, m)


@pytest.fixture
def ranged(shot, fakes, do_pick):
    register("default")(RangeSmooth)
    register("default")(RangeHands)
    do_pick(shot)
    rep = Runner(shot).run(to="hands_face")
    assert rep.outcome == "ok", rep.message
    SEEN.clear()
    yield shot
    RangeHands.fallback = False


def go(shot, **kw) -> tuple:
    events: list[dict] = []
    params = kw.pop("params", None)
    rep = Runner(shot, report=events.append, params=params, cancel=kw.pop("cancel", None)).run(
        to=kw.pop("to", "hands_face"), **kw
    )
    return rep, events


def root(shot, stage) -> np.ndarray:
    return shot.load_result(stage).actors["A"].root


def states(rep) -> dict[str, str]:
    return {s.name: s.state for s in rep.stages}


def test_parse_frames():
    assert parse_frames("3:7") == (3, 7)
    assert parse_frames([0, 1]) == (0, 1)
    assert parse_frames(None) is None
    for bad in ("7:3", "3", "a:b", "-1:4", "4:4"):
        with pytest.raises(RunError):
            parse_frames(bad)


def mark(shot, *spans) -> None:
    shot.write_edit(
        "marks", {"spans": [{"role": "A", "side": "left", "start": x, "end": y, "value": v}
                            for x, y, v in spans]},
    )


def test_range_rerun_blends_and_counts_as_fresh(ranged):
    before_s, before_h = root(ranged, "smooth"), root(ranged, "hands_face")
    mark(ranged, (3, 5, 10.0))  # an edit inside the range; params unchanged
    rep, events = go(ranged, from_stage="smooth", frames=(2, 5))
    assert rep.outcome == "ok", rep.message
    assert SEEN == [
        {"stage": "smooth", "range": (2, 5), "previous": True},
        {"stage": "hands_face", "range": (2, 5), "previous": True},
    ]
    st = {s.name: s for s in rep.stages}
    assert st["smooth"].state == "ran" and st["smooth"].range == [2, 5]
    assert st["hands_face"].range == [2, 5]
    assert "frames 2:5" in st["smooth"].reason
    after_s = root(ranged, "smooth")
    np.testing.assert_array_equal(after_s[:3], before_s[:3])
    np.testing.assert_array_equal(after_s[5:], before_s[5:])
    np.testing.assert_allclose(after_s[3:5], before_s[3:5] + 10.0)
    np.testing.assert_allclose(root(ranged, "hands_face"), after_s + 1.0)
    assert not np.allclose(root(ranged, "hands_face"), before_h)

    meta = ranged.meta("smooth")
    assert meta["range"]["frames"] == [2, 5]
    assert meta["key_parts"]["frame_range"] == [2, 5]
    assert meta["key_parts"]["previous"] == meta["range"]["previous_output_hash"]
    assert meta["base_key"] != meta["key"]
    assert meta["range"]["stage"] == {"frames": [2, 5], "solved": [2, 5]}  # the stage's qc.json "range"
    assert ranged.meta("hands_face")["range"]["stage"] == {"frames": [2, 5]}  # from its qc summary
    assert [e["range"] for e in events if e["event"] == "stage_done" and "range" in e] == [[2, 5], [2, 5]]

    # a plain run keeps the range results: under the range rules they stand for the full run
    SEEN.clear()
    rep, _ = go(ranged)
    assert rep.outcome == "ok" and SEEN == []
    assert states(rep)["smooth"] == "fresh"
    plan = Runner(ranged).plan(to="hands_face")
    sm = next(s for s in plan.stages if s.name == "smooth")
    assert sm.state == "fresh" and sm.range == [2, 5] and "range re-run of frames 2:5" in sm.reason

    # with another setting it is stale, and says why (params, not frame_range)
    plan = Runner(ranged, params={"smooth": {"strength": 0.25}}).plan(to="hands_face")
    sm = next(s for s in plan.stages if s.name == "smooth")
    assert sm.state == "stale" and sm.reason.startswith("params: strength") and "frame_range" not in sm.reason


def test_range_needs_the_same_params(ranged):
    rep, _ = go(ranged, from_stage="smooth", frames=(2, 5), params={"smooth": {"strength": 0.25}})
    st = {s.name: s for s in rep.stages}
    assert st["smooth"].range is None and "params differ" in st["smooth"].reason
    assert SEEN[0] == {"stage": "smooth", "range": None, "previous": False}
    np.testing.assert_allclose(root(ranged, "smooth"), 0.25 * ranged.load_result("pose3d").actors["A"].root)


def test_edits_outside_the_range_force_a_full_run(ranged):
    mark(ranged, (3, 5, 1.0), (7, 9, 2.0))  # one span inside 2:5, one outside
    rep, _ = go(ranged, from_stage="smooth", frames=(2, 5))
    st = {s.name: s for s in rep.stages}
    assert st["smooth"].range is None and "edits/marks.json changed outside frames 2:5" in st["smooth"].reason
    assert (ranged.stage_dir("smooth") / "edits_used.json").is_file()
    mark(ranged, (3, 5, 4.0), (7, 9, 2.0))  # now only the span inside changes: a range re-run
    rep, _ = go(ranged, from_stage="smooth", frames=(2, 5))
    assert {s.name: s.range for s in rep.stages}["smooth"] == [2, 5]
    np.testing.assert_allclose(root(ranged, "smooth")[7:9, 0], 0.5 * root(ranged, "pose3d")[7:9, 0] + 2.0)


def test_range_chain_needs_the_replaced_upstream(ranged):
    # smooth re-runs in full (to=smooth): hands_face now lags it
    mark(ranged, (7, 9, 2.0))
    go(ranged, from_stage="smooth", to="smooth")
    SEEN.clear()
    rep, _ = go(ranged, from_stage="smooth", frames=(2, 5))
    st = {s.name: s for s in rep.stages}
    assert st["smooth"].range == [2, 5]  # its edits did not change since its own previous result
    hf = st["hands_face"]
    assert hf.range is None and "upstream smooth changed outside the range" in hf.reason
    np.testing.assert_allclose(root(ranged, "hands_face"), root(ranged, "smooth") + 1.0)


def test_a_stage_that_solved_the_whole_shot_is_a_full_result(ranged):
    RangeHands.fallback = True
    mark(ranged, (3, 5, 1.0))
    rep, _ = go(ranged, from_stage="smooth", frames=(2, 5))
    st = {s.name: s for s in rep.stages}
    assert st["smooth"].range == [2, 5]
    assert st["hands_face"].range is None and "solved the whole shot" in st["hands_face"].reason
    meta = ranged.meta("hands_face")
    assert "range" not in meta and "base_key" not in meta and meta["key_parts"]["frame_range"] is None
    rep, _ = go(ranged)
    assert states(rep)["hands_face"] == "fresh"


def test_results_keyed_with_absent_edits_as_none_stay_fresh(ranged):
    from afv.core.cache import atomic_write_json, stage_key

    sdir = ranged.stage_dir("smooth")
    meta = json.loads((sdir / "meta.json").read_text())
    assert meta["key_parts"]["edits"] == {}  # absent edits are left out now
    meta["key_parts"]["edits"] = {"marks": None}  # as results stored before were keyed
    meta["key"] = stage_key(meta["key_parts"])
    atomic_write_json(sdir / "meta.json", meta)
    plan = Runner(ranged).plan(to="hands_face")
    assert {s.name: s.state for s in plan.stages}["smooth"] == "fresh"
    mark(ranged, (1, 2, 1.0))
    plan = Runner(ranged).plan(to="hands_face")
    sm = next(s for s in plan.stages if s.name == "smooth")
    assert sm.state == "stale" and sm.reason == "edits: marks"


def test_range_history_and_provenance(ranged):
    mark(ranged, (2, 4, 1.0))
    go(ranged, from_stage="smooth", frames=(2, 5))
    first_base = ranged.meta("smooth")["range"]["base"]
    mark(ranged, (2, 4, 1.0), (6, 8, 1.0))
    go(ranged, from_stage="smooth", frames=(6, 9))
    rg = ranged.meta("smooth")["range"]
    assert rg["frames"] == [6, 9]
    assert [h["frames"] for h in rg["history"]] == [[2, 5], [6, 9]]
    assert rg["base"] == first_base  # the full run the chain started from
    prov = shot_provenance(ranged, hash_exports=False)
    assert prov["stages"]["smooth"]["range"]["frames"] == [6, 9]
    text = format_provenance({"source": {"path": "x", "sha256": "y"}, "shots": [prov]})
    assert "frames 2:5, 6:9" in text
    # a full forced run starts a new chain
    go(ranged, from_stage="smooth")
    assert "range" not in ranged.meta("smooth")


def test_range_falls_back_to_full_when_upstream_changed(ranged):
    rep, _ = go(ranged, from_stage="camera", frames=(2, 5), params={"camera": {"focal": 1200.0}})
    assert rep.outcome == "ok", rep.message
    st = {s.name: s for s in rep.stages}
    assert st["camera"].reason.endswith("in full (no range support)")
    assert "upstream pose3d changed outside the range" in st["smooth"].reason
    assert st["smooth"].range is None and "range" not in ranged.meta("smooth")
    assert SEEN == [
        {"stage": "smooth", "range": None, "previous": False},
        {"stage": "hands_face", "range": None, "previous": False},
    ]


def test_range_needs_from_and_frames_inside_the_shot(ranged):
    with pytest.raises(RunError, match="needs --from"):
        Runner(ranged).run(to="hands_face", frames=(1, 3))
    with pytest.raises(RunError, match=r"has frames 0\.\.9"):
        Runner(ranged).run(to="hands_face", from_stage="smooth", frames=(5, T + 1))
    rep, _ = go(ranged, from_stage="smooth", frames=(5, T))  # b == T is the end of the shot
    assert rep.outcome == "ok"


def test_plan_shows_range_intent(ranged):
    plan = Runner(ranged).plan(to="hands_face", from_stage="pose3d", frames=(1, 4))
    st = {s.name: s for s in plan.stages}
    assert st["pose3d"].reason == "--from (in full: no range support)"
    plan = Runner(ranged).plan(to="hands_face", from_stage="smooth", frames=(1, 4))
    st = {s.name: s for s in plan.stages}
    assert st["smooth"].state == "forced" and "frames 1:4" in st["smooth"].reason


def test_cancel_before_start(ranged):
    ev = threading.Event()
    ev.set()
    rep, events = go(ranged, from_stage="smooth", cancel=ev)
    assert rep.outcome == "cancelled"
    assert any(e["event"] == "cancelled" for e in events)
    assert SEEN == []


def test_cancel_during_a_stage_keeps_the_previous_result(ranged):
    before = ranged.meta("smooth")
    ev, gate = threading.Event(), threading.Event()
    RangeSmooth.gate = gate
    out: dict = {}

    def target():
        out["rep"], _ = go(ranged, from_stage="smooth", params={"smooth": {"hold": True}}, cancel=ev)

    th = threading.Thread(target=target)
    th.start()
    try:
        for _ in range(500):
            if SEEN:
                break
            threading.Event().wait(0.01)
        ev.set()
        gate.set()
        th.join(10)
    finally:
        RangeSmooth.gate = None
    rep = out["rep"]
    assert rep.outcome == "cancelled", rep.message
    assert states(rep)["smooth"] == "cancelled"
    assert ranged.meta("smooth")["output_hash"] == before["output_hash"]
    work = ranged.work_root
    assert not [p for p in work.iterdir() if p.is_dir() and p.name.startswith(("smooth-", "failed-smooth"))]


def test_edit_changes_are_compared_frame_by_frame():
    from afv.core.runner import _changes_inside
    from afv.motion.pins import DEFAULT_RADIUS

    shot = (0, 300)
    plant = {"role": "A", "start": 100, "end": 200, "state": "plant", "foot": "left"}
    old = {"spans": [plant]}
    painted = {"spans": [{**plant, "end": 150}, {**plant, "start": 170}, {**plant, "start": 150, "end": 170,
                                                                          "state": "free"}]}  # fmt: skip
    assert _changes_inside(old, painted, (150, 170), shot)  # painting over part of a span
    assert not _changes_inside(old, painted, (150, 165), shot)
    assert _changes_inside(old, {"spans": []}, (100, 200), shot)  # clearing it
    assert not _changes_inside(old, {"spans": []}, (120, 200), shot)
    right = {"spans": [{**plant, "foot": "right"}]}
    assert not _changes_inside(old, {"spans": [plant, *right["spans"]]}, (0, 100), shot)
    hands = {"spans": [{"role": "B", "start": 10, "end": 20, "side": "both"}]}
    assert _changes_inside(None, hands, (10, 20), shot) and not _changes_inside(None, hands, (12, 20), shot)
    assert not _changes_inside({"note": 1}, {"note": 2}, (0, 300), shot)  # other keys must be equal

    pin = {"role": "A", "frame": 100, "preset": "p", "bones": {"Hips": [0, 0, 0, 1]}, "root": None}
    r = DEFAULT_RADIUS
    assert _changes_inside({"pins": []}, {"pins": [pin]}, (100 - r, 101 + r), shot)  # no radius: default
    assert not _changes_inside({"pins": []}, {"pins": [pin]}, (100 - r + 1, 101 + r), shot)
    near_start = {**pin, "frame": 3, "radius": 8}
    assert _changes_inside({"pins": []}, {"pins": [near_start]}, (0, 12), shot)  # window clipped to the shot
    near_end = {**pin, "frame": 296, "radius": 8}
    assert _changes_inside({"pins": []}, {"pins": [near_end]}, (288, 300), shot)
