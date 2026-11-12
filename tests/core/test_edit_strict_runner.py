"""Stored edit files that are lax stop a run: the stages load edits through ``Runner._edits``, which checks
them with the same validator and says how to fix the file. ``run`` checks before anything runs; ``plan``
(``afv status``, ``--dry-run``) still shows the table, with the stages that read the file blocked."""

from __future__ import annotations

import json

import pytest

from afv.core.edits import EditError
from afv.core.runner import Runner
from afv.take import plan_stale

LAX_PINS = {"pins": [{"role": "A", "frame": "3", "preset": "p", "root": [0, 0, 0]}]}
GOOD_PINS = {"pins": [{"role": "A", "frame": 3, "preset": "p", "root": [0, 0, 0]}]}


def raw(shot, name, data):
    """A file as an older version or a hand edit would leave it (Shot.write_edit refuses lax data)."""
    p = shot.edit_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


@pytest.fixture
def done(shot, fakes, do_pick):
    """A shot run through smooth once."""
    do_pick(shot)
    rep = Runner(shot).run(to="smooth")
    assert rep.outcome == "ok", rep.message
    return shot


def test_shot_write_edit_refuses_lax_data_and_leaves_the_file(shot):
    shot.write_edit("pins", GOOD_PINS)
    before = shot.edit_path("pins").read_bytes()
    with pytest.raises(EditError, match=r"pins\[0\]\.frame: must be an integer >= 0, got '3'"):
        shot.write_edit("pins", LAX_PINS)
    assert shot.edit_path("pins").read_bytes() == before
    assert not list(shot.edit_path("pins").parent.glob("*.tmp"))
    for name, data in (("contacts", {"spans": [{"role": "A", "start": 1.0}]}), ("hands", []),
                       ("ids", {"swaps": [{"start": "1", "objects": [1, 2]}]}),
                       ("fighters", {"fighters": [True, False]})):
        with pytest.raises(EditError):
            shot.write_edit(name, data)
        assert not shot.edit_path(name).exists()


def test_a_lax_file_on_disk_makes_the_message_say_to_fix_it_first(shot):
    raw(shot, "pins", LAX_PINS)
    with pytest.raises(EditError) as e:  # data merged from the lax file keeps its bad value
        shot.write_edit("pins", LAX_PINS)
    assert "the file on disk is invalid too: fix or delete that entry in edits/pins.json" in str(e.value)
    raw(shot, "pins", [1])
    with pytest.raises(EditError, match=r"fix or delete the file edits/pins\.json"):
        shot.write_edit("pins", [1])
    shot.write_edit("pins", GOOD_PINS)  # a clean replacement is fine


def test_run_stops_before_any_stage_when_a_needed_edit_is_lax(shot, fakes, do_pick):
    do_pick(shot)
    raw(shot, "pins", LAX_PINS)
    fakes.calls.clear()
    with pytest.raises(EditError) as e:
        Runner(shot).run(to="smooth")
    msg = str(e.value)
    assert msg.startswith("edits/pins.json: pins[0].frame: must be an integer >= 0, got '3'")
    assert msg.endswith("(fix or delete that entry in edits/pins.json)")
    assert fakes.calls == []  # nothing ran: not ingest, detect_track, camera, pose3d
    assert not shot.has_result("detect_track")
    with pytest.raises(EditError):
        Runner(shot).check_edits("smooth")
    Runner(shot).check_edits("pose3d")  # smooth is not needed to get there


def test_range_and_from_runs_check_too(done, fakes):
    raw(done, "pins", LAX_PINS)
    fakes.calls.clear()
    with pytest.raises(EditError, match="fix or delete"):
        Runner(done).run(to="smooth", from_stage="smooth", frames=(1, 4))
    assert fakes.calls == []
    raw(done, "pins", GOOD_PINS)
    Runner(done).check_edits()  # fine again


def test_lax_fighters_stop_a_run_and_say_how_to_fix_it(done):
    raw(done, "fighters", {"fighters": ["1", "0"], "extra": []})
    want = r"fighters\[0\]: must be an integer.*fix or delete edits/fighters\.json first"
    with pytest.raises(EditError, match=want):
        Runner(done).run()


def test_plan_shows_the_table_with_the_stage_blocked(done):
    raw(done, "pins", LAX_PINS)
    plan = Runner(done).plan()
    states = {s.name: s.state for s in plan.stages}
    assert states["ingest"] == "fresh" and states["pose3d"] == "fresh" and states["smooth"] == "blocked"
    smooth = next(s for s in plan.stages if s.name == "smooth")
    assert smooth.reason.startswith("edits/pins.json: pins[0].frame:")
    assert plan.outcome == "failed" and "edits/pins.json" in plan.message
    raw(done, "pins", GOOD_PINS)
    assert Runner(done).plan().outcome == "planned"


def test_take_surfaces_the_bad_edit_file(done):
    assert plan_stale(done, "smooth") == []
    raw(done, "pins", LAX_PINS)
    lines = plan_stale(done, "smooth")
    assert len(lines) == 1 and lines[0].startswith("smooth: blocked (edits/pins.json: pins[0].frame")



def test_take_summary_names_the_file_not_an_old_result(done):
    from afv.take import stale_summary

    raw(done, "pins", LAX_PINS)
    plan = plan_stale(done, "smooth")
    s = stale_summary(done.id, [], plan)
    assert s == f"edits/pins.json is invalid: fix or delete that entry (details: afv status {done.id})"
    assert "older" not in s and "afv run" not in s
    # blocked stages after it are not "old results" either; real stale stages still are
    plan = [*plan, "export: blocked (after smooth: its edit file must be fixed first)",
            "camera: stale (params: zoom)"]
    s = stale_summary(done.id, ["hands_face was made from an older smooth result"], plan)
    assert s == (f"edits/pins.json is invalid: fix or delete that entry (details: afv status {done.id}); "
                 f"camera, hands_face results are older than the current settings or code: "
                 f"run `afv run {done.id}` (details: afv status {done.id})")
