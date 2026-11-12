"""Tracking QC pause: hints from detect_track's QC stop the run until fixed or acknowledged."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from afv.cli import app
from afv.core.review import REVIEW_FILE, format_review, tracking_hints
from afv.core.runner import Runner


def _obj(oid, role, runs, label=None):
    return {"id": oid, "role": role, "label": label or (f"fighter {role}" if role else f"person {oid}"),
            "flagged_runs": runs}  # fmt: skip


SWAP_QC = {
    "objects": [
        _obj(1, "A", [[3, 5, ["jump", "swap"]]]),
        _obj(2, "B", [[3, 5, ["swap"]], [7, 7, ["area_jump"]]]),
        _obj(3, None, [[1, 2, ["drift", "jump"]]], "referee"),
    ],
    "suggestions": [],
}
DRIFT_QC = {
    "objects": [_obj(1, "A", []), _obj(2, "B", [[82, 91, ["drift"]]])],
    "suggestions": [
        {"kind": "drift", "obj": 2, "frames": [82, 91], "frame": 82, "detection": 1, "box": [1, 2, 3, 4]}
    ],
}
CLEAN_QC = {"objects": [_obj(1, "A", []), _obj(2, "B", []), _obj(3, None, [[4, 4, ["area_jump"]]])],
            "suggestions": []}  # fmt: skip


# ---------------------------------------------------------------------------
# Hints from the QC summary
# ---------------------------------------------------------------------------


def test_hints_cover_fighters_only():
    hints = tracking_hints(SWAP_QC, "001")
    got = [(h["object"], h["kind"], h["frames"]) for h in hints]
    assert got == [(1, "identity", [3, 5]), (2, "identity", [3, 5])]
    assert "likely ID swap" in hints[0]["message"] and hints[0]["message"].startswith("fighter A")
    assert hints[0]["fix"] == (
        "afv pick 001 --frame 3, then afv pick 001 --prompt 3:1:DET; "
        "if the fighters traded identities from frame 3: afv pick 001 --swap 3:1:2"
    )


def test_drift_hint_uses_the_suggested_detection():
    (h,) = tracking_hints(DRIFT_QC, "7")
    assert h["kind"] == "drift" and h["role"] == "B"
    assert h["fix"] == "afv pick 7 --prompt 82:2:1"


def test_lost_fighter_and_untracked_person():
    qc = {
        "objects": [_obj(1, "A", [[0, 9, ["empty"]]]), _obj(2, "B", [])],
        "suggestions": [
            {"kind": "untracked_person", "frames": [10, 40], "frame": 12, "obj": 4, "detection": 2}
        ],
    }
    hints = tracking_hints(qc, "1")
    assert [h["kind"] for h in hints] == ["lost", "untracked_person"]
    assert "afv pick 1 --prompt 12:4:2 --label NAME" in hints[1]["fix"]


BAD_RUN = {"objects": [{"role": "A", "flagged_runs": [[1]]}]}


@pytest.mark.parametrize("qc", [None, {}, CLEAN_QC, {"objects": "bad"}, BAD_RUN])
def test_no_hints(qc):
    assert tracking_hints(qc, "1") == []


def test_format_review():
    rv = {"hints": tracking_hints(DRIFT_QC, "001"), "review_video": "/x/review.mp4", "acknowledged": False}
    text = format_review(rv, "001", paused=True)
    assert "stopped after detect_track" in text and "/x/review.mp4" in text
    assert "afv run 001 --yes" in text and "fighter B" in text


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@pytest.fixture
def qc(fakes, monkeypatch):
    """The summary the fake detect_track writes to qc.json once fighters are picked (mutable)."""
    state = {"summary": DRIFT_QC}
    orig = fakes.DetectTrack.run

    def run(self, ctx, inputs, params):
        res = orig(self, ctx, inputs, params)
        if res.fighters:
            (ctx.work_dir / "qc.json").write_text(json.dumps({"summary": state["summary"]}))
        return res

    monkeypatch.setattr(fakes.DetectTrack, "run", run)
    return state


def run(shot, fakes, to="smooth", **kw):
    fakes.calls.clear()
    events: list[dict] = []
    rep = Runner(shot, report=events.append, **kw).run(to=to)
    rep.events = events
    return rep


def test_pause_after_detect_track(shot, fakes, do_pick, qc):
    do_pick(shot)
    rep = run(shot, fakes, pause_on_hints=True)
    assert rep.outcome == "needs_review"
    assert fakes.calls == ["ingest", "detect_track"]  # camera (needs tracking) did not run
    assert rep.stages[-1].name == "camera" and rep.stages[-1].state == "blocked"
    ev = next(e for e in rep.events if e["event"] == "tracking_review")
    assert ev["paused"] is True and ev["hints"][0]["fix"] == "afv pick 001 --prompt 82:2:1"
    assert rep.review["hints"] == ev["hints"] and "afv run 001 --yes" in rep.message
    assert not (shot.dir / REVIEW_FILE).exists()

    plan = {s.name: s for s in Runner(shot).plan().stages}
    assert "pauses after this stage" in plan["detect_track"].reason
    assert plan["detect_track"].state == "fresh"

    rep = run(shot, fakes, pause_on_hints=True)  # still unacknowledged: pauses again, runs nothing
    assert rep.outcome == "needs_review" and fakes.calls == []

    rep = run(shot, fakes, pause_on_hints=True, yes=True)
    assert rep.outcome == "ok" and fakes.calls == ["camera:vipe", "pose3d", "smooth"]
    ev = next(e for e in rep.events if e["event"] == "tracking_review")
    assert ev["paused"] is False and ev["acknowledged"] is True
    rec = json.loads((shot.dir / REVIEW_FILE).read_text())["detect_track"]
    assert rec["output_hash"] == shot.meta("detect_track")["output_hash"] and rec["hints"] == 1

    # acknowledged for this tracking result: later runs go on (a smooth change re-runs smooth only)
    rep = run(shot, fakes, pause_on_hints=True, params={"smooth": {"strength": 0.9}})
    assert rep.outcome == "ok" and fakes.calls == ["smooth"]
    assert not any(e["event"] == "tracking_review" for e in rep.events)


def test_new_tracking_is_reviewed_again(shot, fakes, do_pick, qc):
    do_pick(shot)
    assert run(shot, fakes, pause_on_hints=True, yes=True).outcome == "ok"
    do_pick(shot, fighters=(0, 1))  # a different tracking result, still with a hint
    rep = run(shot, fakes, pause_on_hints=True)
    assert rep.outcome == "needs_review" and fakes.calls == ["detect_track"]


def test_resolved_hints_continue(shot, fakes, do_pick, qc):
    do_pick(shot)
    assert run(shot, fakes, pause_on_hints=True).outcome == "needs_review"
    qc["summary"] = CLEAN_QC
    do_pick(shot, extra=())  # e.g. after a re-prompt: detect_track runs again, without hints
    rep = run(shot, fakes, pause_on_hints=True)
    assert rep.outcome == "ok" and fakes.calls == ["detect_track", "camera:vipe", "pose3d", "smooth"]
    assert rep.review is None


def test_no_pause_when_nothing_after_tracking_runs(shot, fakes, do_pick, qc):
    do_pick(shot)
    rep = run(shot, fakes, to="detect_track", pause_on_hints=True)
    assert rep.outcome == "ok"
    ev = next(e for e in rep.events if e["event"] == "tracking_review")
    assert ev["paused"] is False and ev["acknowledged"] is False  # reported as a warning


def test_without_pause_hints_are_a_warning(shot, fakes, do_pick, qc):
    do_pick(shot)
    rep = run(shot, fakes)  # Runner default (direct use, golden runs): no pause
    assert rep.outcome == "ok" and fakes.calls[-1] == "smooth"
    assert [e["paused"] for e in rep.events if e["event"] == "tracking_review"] == [False]
    assert rep.events[-1]["event"] == "done"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def cli():
    r = CliRunner()

    def invoke(*args, code=0):
        res = r.invoke(app, [str(a) for a in args], catch_exceptions=False)
        assert res.exit_code == code, res.output
        return res

    return invoke


def test_cli_run_pauses_and_yes_goes_on(cli, project, shot, fakes, do_pick, qc):
    do_pick(shot)
    out = cli("run", "1", "-p", project.root, "--to", "smooth", code=3).output
    assert "tracking QC for shot 001: 1 hint - stopped after detect_track" in out
    assert "fix: afv pick 001 --prompt 82:2:1" in out and "camera        running" not in out

    res = cli("run", "1", "-p", project.root, "--to", "smooth", "--json", code=3)
    lines = [json.loads(x) for x in res.output.splitlines()]
    ev = next(x for x in lines if x["event"] == "tracking_review")
    assert ev["paused"] is True and ev["hints"][0]["kind"] == "drift"
    assert lines[-1]["event"] == "result" and lines[-1]["outcome"] == "needs_review"

    out = cli("status", "1", "-p", project.root, "--to", "smooth").output
    assert "pauses after this stage" in out and "fighter B" in out

    out = cli("run", "1", "-p", project.root, "--to", "smooth", "--yes").output
    assert "(going on: --yes)" in out and "smooth" in out
    cli("run", "1", "-p", project.root, "--to", "smooth")  # acknowledged: exit 0


def test_cli_batch_counts_reviews(cli, project, shot, fakes, do_pick, qc):
    do_pick(shot)
    out = cli("batch", "-p", project.root, "--to", "smooth", code=3).output
    assert "1 waiting for a tracking review" in out
    cli("batch", "-p", project.root, "--to", "smooth", "--yes")


def test_cli_status_shows_stale_code(cli, project, shot, fakes, do_pick, monkeypatch):
    from afv.core import codehash
    from afv.core import runner as runner_mod

    do_pick(shot)
    cli("run", "1", "-p", project.root, "--to", "smooth")
    real = runner_mod.stage_code
    monkeypatch.setattr(fakes.Pose, "env", "gemx")
    new = {"hash": "new", "files": {"afv.fake": "2"}, "python": codehash.PYTHON}
    monkeypatch.setattr(runner_mod, "stage_code", lambda cls: new if cls is fakes.Pose else real(cls))
    out = cli("status", "1", "-p", project.root, "--to", "smooth").output
    line = next(x for x in out.splitlines() if x.strip().startswith("pose3d"))
    assert "stale (code)" in line and "afv.fake" in line
    assert line.index("--code-changes rerun") < line.index("--from pose3d")  # the cheaper fix first
    out = cli("run", "1", "-p", project.root, "--to", "smooth").output
    lines = [x for x in out.splitlines() if x.strip().startswith("pose3d")]
    assert len(lines) == 1 and "result kept; code changed:" in lines[0] and "code: code" not in lines[0]
    out = cli("run", "1", "-p", project.root, "--to", "smooth", "--code-changes", "accept").output
    lines = [x for x in out.splitlines() if x.strip().startswith("pose3d")]
    assert len(lines) == 1 and lines[0].startswith("  pose3d        up to date (accepted: code changed: ")
    out = cli("status", "1", "-p", project.root, "--to", "smooth").output
    assert "stale (code)" not in out


def test_ack_is_invalidated_by_new_hints(shot, fakes, do_pick, qc):
    """A QC-only change (new QC code or thresholds) can raise new hints on the same tracking output."""
    do_pick(shot)
    assert run(shot, fakes, pause_on_hints=True, yes=True).outcome == "ok"
    h1 = shot.meta("detect_track")["output_hash"]
    qc["summary"] = SWAP_QC
    over = {"detect_track": {"prompt_frame": 3}, "smooth": {"strength": 0.9}}
    rep = run(shot, fakes, pause_on_hints=True, params=over)
    assert shot.meta("detect_track")["output_hash"] == h1  # same tracks
    assert rep.outcome == "needs_review" and len(rep.review["hints"]) == 2
    assert rep.review["acknowledged"] is False and fakes.calls == ["detect_track"]


def test_cli_export_pauses_like_run(cli, project, shot, fakes, do_pick, qc):
    do_pick(shot)
    res = cli("export", "1", "-p", project.root, "--json", code=3)
    lines = [json.loads(x) for x in res.output.splitlines()]
    assert next(x for x in lines if x["event"] == "tracking_review")["paused"] is True
    assert lines[-1]["outcome"] == "needs_review" and "export" not in fakes.calls
    out = cli("export", "1", "-p", project.root, "--yes").output
    assert "exports in" in out and "export" in fakes.calls


def test_ack_survives_rewording_but_not_new_frames():
    from afv.core.review import hints_hash

    hints = tracking_hints(DRIFT_QC, "001")
    reworded = [{**h, "message": "other words", "fix": "afv pick 001 --frame 82"} for h in hints]
    assert hints_hash(reworded) == hints_hash(hints)
    moved = [{**h, "frames": [80, 91]} for h in hints]
    assert hints_hash(moved) != hints_hash(hints)
