"""Shot states (candidate / queued / skipped), afv shots set/queue/skip/list, afv batch by state, and afv pick
from a stored scan (round 6)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from afv.cli import app, parse_shot_ids
from afv.core.config import ConfigError
from afv.core.project import Project, ProjectError, check_state, scan_paths, shot_hash


@pytest.fixture
def cli():
    r = CliRunner()

    def invoke(*args, code=0):
        res = r.invoke(app, [str(a) for a in args], catch_exceptions=False)
        assert res.exit_code == code, res.output
        return res

    return invoke


@pytest.fixture
def proj(cli, tmp_path, video, fakes):
    root = tmp_path / "p.afv"
    cli("init", video, "--project", root, "--no-probe")
    return root


def rows(cli, proj, *extra):
    return [json.loads(x) for x in cli("shots", "list", "-p", proj, "--json", *extra).output.splitlines()]


def write_scan(proj_root, sid, boxes, scores, frame=15):
    """A scan file as afv shots scan writes it (and its frame image), for the shot as it is now."""
    from afv.media.shots import write_jpeg

    p = Project(proj_root)
    sh = p.shot(sid)
    paths = scan_paths(p, sh.id)
    paths.json.parent.mkdir(parents=True, exist_ok=True)
    write_jpeg(np.full((180, 320, 3), 90, np.uint8), paths.frame)
    write_jpeg(np.full((90, 160, 3), 90, np.uint8), paths.thumb)
    data = {
        "format": "afv_scan", "version": 1, "shot": sh.id,
        "shot_hash": shot_hash(sh.info.start, sh.info.duration, p.info.source.sha256), "frame": frame,
        "prompt": {"frame": frame, "prompt_frame": frame, "image": f"{sh.id}.prompt.jpg", "boxes": boxes,
                   "scores": scores, "threshold": 0.4},
        "people": len(boxes), "fighter_sized": 2, "fighters_likely": True, "size": [320, 180],
    }  # fmt: skip
    paths.json.write_text(json.dumps(data))
    return data


BOXES = [[100.5, 20.0, 180.0, 170.0], [10.0, 30.0, 90.0, 175.0], [200.0, 40.0, 230.0, 100.0]]


def outs(done: dict) -> dict[str, str]:
    """{shot: outcome} of a batch_done event."""
    return {o["shot"]: o["outcome"] for o in done["outcomes"]}


# ---------------------------------------------------------------------------
# project level
# ---------------------------------------------------------------------------


def test_missing_state_is_queued_and_unknown_keys_survive_an_update(project):
    sh = project.add_shot(1.0, 2.0)
    p = sh.dir / "shot.json"
    data = json.loads(p.read_text())
    del data["state"]  # a shot made before round 6
    data["future_key"] = {"x": 1}
    p.write_text(json.dumps(data))
    assert sh.info.state == "queued"
    info, changed = sh.update(state="skipped", notes="crowd")
    assert (info.state, info.notes, changed) == ("skipped", "crowd", ["notes", "state"])
    assert json.loads(p.read_text())["future_key"] == {"x": 1}
    assert sh.update(state="skipped")[1] == []  # no change, no write


def test_update_validates_like_add_shot(project):
    sh = project.add_shot(1.0, 2.0, state="candidate")
    assert sh.info.state == "candidate"
    with pytest.raises(ProjectError, match="bad shot interval"):
        sh.update(duration=-1.0)
    with pytest.raises(ProjectError, match="finite"):
        sh.update(start=float("nan"))
    with pytest.raises(ProjectError, match="speed factor"):
        sh.update(speed_factor=0.0)
    with pytest.raises(ProjectError, match="bad shot state 'done'"):
        sh.update(state="done")
    with pytest.raises(ProjectError, match="bad shot state"):
        project.add_shot(5.0, 1.0, state="chosen")
    assert check_state("queued") == "queued"
    info, changed = sh.update(start=1.5, duration=1.0, speed_factor=0.5)
    assert (info.start, info.duration, info.speed_factor, changed) == (1.5, 1.0, 0.5,
                                                                      ["start", "duration", "speed_factor"])
    bad = sh.dir / "shot.json"
    bad.write_text(bad.read_text().replace('"skipped"', '"x"').replace('"candidate"', '"later"'))
    with pytest.raises(ProjectError, match=r"shot\.json is invalid"):
        _ = sh.info


def test_parse_shot_ids(project):
    for start in range(5):
        project.add_shot(float(start), 1.0)
    project.add_shot(9.0, 1.0, shot_id="a-b")
    assert parse_shot_ids(project, ["2", "4,1"]) == ["002", "004", "001"]
    assert parse_shot_ids(project, ["2-4", "3", "a-b"]) == ["002", "003", "004", "a-b"]
    assert parse_shot_ids(project, ["1 2"]) == ["001", "002"]
    with pytest.raises(ConfigError, match="first > last"):
        parse_shot_ids(project, ["4-2"])
    with pytest.raises(ProjectError, match="no shots in 7-9"):
        parse_shot_ids(project, ["7-9"])
    with pytest.raises(ProjectError, match="no shot '12'"):
        parse_shot_ids(project, ["12"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_add_queue_skip_set_list(cli, proj):
    cli("shots", "add", "--start", "1", "--end", "3", "-p", proj)
    cli("shots", "add", "--start", "4", "--duration", "2", "--state", "candidate", "-p", proj)
    cli("shots", "add", "--start", "7", "--duration", "2", "--state", "candidate", "-p", proj)
    assert [r["state"] for r in rows(cli, proj)] == ["queued", "candidate", "candidate"]
    out = cli("shots", "queue", "2-3", "-p", proj).output
    assert "queued: 002, 003" in out
    assert "(no change)" in cli("shots", "queue", "3", "-p", proj).output
    cli("shots", "skip", "1,3", "-p", proj)
    assert [r["state"] for r in rows(cli, proj)] == ["skipped", "queued", "skipped"]
    assert [r["id"] for r in rows(cli, proj, "--state", "skipped")] == ["001", "003"]
    both = rows(cli, proj, "--state", "queued", "--state", "skipped")
    assert [r["id"] for r in both] == ["001", "002", "003"]
    r = rows(cli, proj)[1]
    assert r["scan"] == {"status": "none", "frame": None, "people": None, "fighter_sized": None,
                         "fighters_likely": None, "thumbnail": None}
    assert (r["picked"], r["pick_stale"], r["exported"], r["stored"]) == (False, False, False, [])
    assert r["end"] == 6.0
    text = cli("shots", "list", "-p", proj).output
    assert "  002  queued" in text and "skipped" in text
    assert "no shot '9'" in cli("shots", "queue", "9", "-p", proj, code=1).output
    assert "bad shot range" in cli("shots", "skip", "3-1", "-p", proj, code=1).output
    cli("shots", "queue", "-p", proj, code=2)  # ids are required

    # set: --start alone keeps the end; --end/--duration keep the start
    out = cli("shots", "set", "2", "--start", "4.5", "-p", proj).output
    assert "start, duration changed" in out and "4.500 - 6.000 s (1.500 s)" in out
    cli("shots", "set", "2", "--end", "0:07", "-p", proj)
    assert (Project(proj).shot("2").info.start, Project(proj).shot("2").info.end) == (4.5, 7.0)
    cli("shots", "set", "2", "--duration", "1", "--speed", "0.25", "--notes", "replay",
        "--state", "candidate", "-p", proj)
    i = Project(proj).shot("2").info
    assert (i.duration, i.speed_factor, i.notes, i.state) == (1.0, 0.25, "replay", "candidate")
    assert "no change" in cli("shots", "set", "2", "--notes", "replay", "-p", proj).output
    out = cli("shots", "set", "2", "--end", "9", "--duration", "1", "-p", proj, code=1).output
    assert "at most one of end and duration" in out
    assert "bad shot interval" in cli("shots", "set", "2", "--end", "3", "-p", proj, code=1).output
    assert "no shot" in cli("shots", "set", "8", "--state", "queued", "-p", proj, code=1).output
    res = json.loads(cli("shots", "set", "2", "--state", "queued", "-p", proj, "--json").output)
    assert res["event"] == "shot_set" and res["changed"] == ["state"] and res["state"] == "queued"


def test_detect_adds_candidates_and_cuts_json(cli, proj, monkeypatch):
    import afv.cli as cli_mod
    from afv.media.cuts import Cut, CutAnalysis, CutParams

    z = np.zeros(10)
    cuts = [Cut(36, 1.2, 0.9, 0.5), Cut(96, 3.2, 0.15, 0.2, "gradual", 90, 3.0)]
    an = CutAnalysis(cuts, np.arange(10) / 30, z, z, z, 30.0, CutParams())

    def detector(video, min_confidence=None):
        segs = [{"start": 0.0, "end": 1.2, "confidence": 1.0}, {"start": 1.2, "end": 3.2, "confidence": 0.9}]
        return segs, an

    monkeypatch.setattr(cli_mod, "_shot_detector", lambda: detector)
    cli("shots", "add", "--start", "1.3", "--end", "3", "-p", proj)  # covers most of 1.2-3.2
    out = cli("shots", "detect", "-p", proj).output
    assert "2 segments, 1 added as candidates, 1 skipped" in out
    assert [(r["id"], r["state"], r["origin"]) for r in rows(cli, proj)] == [
        ("001", "queued", "manual"), ("002", "candidate", "detected")]
    cuts = json.loads((proj / "media" / "cuts.json").read_text())
    assert cuts["format"] == "afv_cuts" and cuts["source_sha256"] == Project(proj).info.source.sha256
    assert [(c["frame"], c["kind"]) for c in cuts["cuts"]] == [(36, "hard"), (96, "gradual")]


def test_init_detect(cli, tmp_path, video, fakes, monkeypatch):
    import afv.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_shot_detector",
                        lambda: lambda video, min_confidence=None: [{"start": 0.0, "duration": 2.0}])
    root = tmp_path / "q.afv"
    out = cli("init", video, "--project", root, "--no-probe", "--detect").output
    assert "1 added as candidates" in out and "afv shots scan" in out
    assert Project(root).shot("1").info.state == "candidate"


def test_batch_runs_queued_shots_only(cli, proj):
    for start, state in ((1, "queued"), (4, "candidate"), (7, "skipped"), (10, "queued")):
        cli("shots", "add", "--start", start, "--duration", "2", "--state", state, "-p", proj)
    for sid in ("1", "4"):
        cli("run", sid, "--to", "detect_track", "-p", proj, code=3)
        cli("pick", sid, "--fighters", "1,0", "-p", proj)
    events = [json.loads(x) for x in cli("batch", "--to", "camera", "-p", proj, "--json").output.splitlines()]
    done = events[-1]
    assert done["event"] == "batch_done" and outs(done) == {"001": "ok", "004": "ok"}
    assert done["outcomes"][0] == {"shot": "001", "outcome": "ok", "message": "", "exports": []}
    assert (done["states"], done["not_selected"]) == (["queued"], 2)
    assert not Project(proj).shot("2").meta("ingest")  # the candidate did not run
    res = cli("batch", "--to", "camera", "--state", "candidate", "-p", proj, code=3)
    assert "waiting for a pick" in res.output and "002: needs_pick" in res.output
    out = cli("batch", "--to", "camera", "--all", "-p", proj, "--json", code=3).output
    done = json.loads(out.splitlines()[-1])
    assert set(outs(done)) == {"001", "002", "003", "004"} and done["states"] is None
    cli("shots", "skip", "1-4", "-p", proj)
    assert "no shots in state queued (4 in other states" in cli("batch", "-p", proj).output


def test_old_projects_batch_as_before(cli, proj):
    """Shots without a state (made before round 6) are queued: afv batch runs all of them, as before."""
    for start in (1, 4):
        cli("shots", "add", "--start", start, "--duration", "2", "-p", proj)
    for sh in Project(proj).shots():
        p = sh.dir / "shot.json"
        d = json.loads(p.read_text())
        del d["state"]
        p.write_text(json.dumps(d))
    done = json.loads(cli("batch", "--to", "camera", "-p", proj, "--json", code=3).output.splitlines()[-1])
    assert outs(done) == {"001": "needs_pick", "002": "needs_pick"}


def test_pick_from_scan_starts_no_detector(cli, proj, monkeypatch):
    from afv.core.runner import Runner

    cli("shots", "add", "--start", "1", "--duration", "2", "--state", "candidate", "-p", proj)
    write_scan(proj, "1", BOXES, [0.9, 0.85, 0.6])

    def no_run(self, *a, **k):
        raise AssertionError("pick from a scan must not run a stage")

    monkeypatch.setattr(Runner, "run", no_run)
    out = cli("pick", "1", "-p", proj).output
    assert "scan:" in out and "001.prompt.jpg (frame 15)" in out and "  1: box [10, 30, 90, 175]" in out
    assert (proj / "media/scan/001.prompt.jpg").is_file()
    ev = json.loads(cli("pick", "1", "-p", proj, "--json").output)
    assert ev["source"] == "scan" and ev["prompt"]["boxes"] == BOXES
    assert "no detection [5]" in cli("pick", "1", "--fighters", "0,5", "-p", proj, code=1).output
    out = cli("pick", "1", "--fighters", "1,0", "--extra", "2", "-p", proj).output
    assert "(numbers from the scan)" in out
    edit = json.loads((proj / "shots/001/edits/fighters.json").read_text())
    assert edit["prompt_frame"] == 15 and edit["fighters"] == [1, 0] and edit["extra"] == [2]
    assert edit["boxes"] == {"1": BOXES[1], "0": BOXES[0], "2": BOXES[2]}
    assert edit["shot"] == {"start": 1.0, "duration": 2.0, "source_sha256": Project(proj).info.source.sha256}
    r = rows(cli, proj)[0]
    assert r["picked"] and not r["pick_stale"] and r["state"] == "candidate"  # picking does not queue
    assert r["scan"]["status"] == "ok" and r["scan"]["thumbnail"] == "media/scan/001.jpg"
    # trimming makes the scan stale (not used) and the picks stale
    out = cli("shots", "set", "1", "--start", "1.5", "-p", proj).output
    assert "picked for the old interval" in out and "afv shots scan 001" in out
    r = rows(cli, proj)[0]
    assert r["pick_stale"] and r["scan"]["status"] == "stale" and r["scan"]["people"] is None
    monkeypatch.undo()
    out = cli("pick", "1", "--fighters", "0,5", "-p", proj).output  # nothing known: unchecked, as before
    assert json.loads((proj / "shots/001/edits/fighters.json").read_text())["prompt_frame"] is None


def test_pick_frame_chooses_between_stage_prompt_and_scan(cli, proj):
    cli("shots", "add", "--start", "1", "--duration", "2", "-p", proj)
    cli("run", "1", "--to", "detect_track", "-p", proj, code=3)  # the fake stage: prompt.json on frame 0
    write_scan(proj, "1", BOXES[:2], [0.9, 0.85], frame=15)
    ev = json.loads(cli("pick", "1", "-p", proj, "--json").output)
    assert ev["source"] == "stage"  # without --frame: the prompt image afv run showed, as before
    cli("pick", "1", "--fighters", "2,1", "-p", proj)
    edit = json.loads((proj / "shots/001/edits/fighters.json").read_text())
    assert (edit["prompt_frame"], edit["boxes"]["2"]) == (0, [110, 20, 130, 80])
    out = cli("pick", "1", "--fighters", "1,0", "--frame", "15", "-p", proj).output
    assert "numbers from the scan" in out
    edit = json.loads((proj / "shots/001/edits/fighters.json").read_text())
    assert (edit["prompt_frame"], edit["boxes"]["1"]) == (15, BOXES[1])
    out = cli("pick", "1", "--fighters", "1,0", "--prompt", "3:3:0", "-p", proj, code=1).output
    assert "pick first" in out
    assert "--extra needs --fighters" in cli("pick", "1", "--extra", "2", "-p", proj, code=1).output


# ---------------------------------------------------------------------------
# review round 1 fixes
# ---------------------------------------------------------------------------


def test_pick_after_trim_uses_the_scan_not_the_old_stage_prompt(cli, proj):
    """(The reviewer's case) run to detect_track, trim, fresh scan, pick: the old interval's prompt.json
    must not number the people any more."""
    cli("shots", "add", "--start", "1", "--duration", "2", "-p", proj)
    cli("run", "1", "--to", "detect_track", "-p", proj, code=3)  # prompt.json on frame 0 for 1.0+2.0
    cli("shots", "set", "1", "--start", "1.5", "-p", proj)
    write_scan(proj, "1", BOXES, [0.9, 0.85, 0.6])  # a fresh scan of the trimmed shot
    show = json.loads(cli("pick", "1", "-p", proj, "--json").output)
    assert show["source"] == "scan" and show["prompt"]["boxes"] == BOXES
    out = cli("pick", "1", "--fighters", "1,0", "-p", proj).output
    assert "numbers from the scan" in out
    edit = json.loads((proj / "shots/001/edits/fighters.json").read_text())
    assert (edit["prompt_frame"], edit["boxes"]["1"]) == (15, BOXES[1])
    assert edit["shot"]["start"] == 1.5 and not rows(cli, proj)[0]["pick_stale"]
    # without a scan the stale prompt is not used either: detect_track runs again for the new interval
    (proj / "media/scan/001.json").unlink()
    res = json.loads(cli("pick", "1", "-p", proj, "--json").output.splitlines()[-1])
    assert res["source"] == "stage"
    assert Project(proj).shot("1").meta("detect_track")["key_parts"]["shot"]["start"] == 1.5


def test_fighters_likely_follows_the_current_rule(cli, proj):
    cli("shots", "add", "--start", "1", "--duration", "2", "-p", proj)
    cli("shots", "add", "--start", "4", "--duration", "2", "-p", proj)
    fighters = [[20.0, 30.0, 70.0, 160.0], [150.0, 35.0, 200.0, 165.0]]  # two full bodies (180 px frame)
    closeups = [[0.0, 0.0, 150.0, 178.0], [160.0, 0.0, 320.0, 179.0]]
    d = write_scan(proj, "1", fighters, [0.9, 0.9])
    d["fighters_likely"], d["fighter_sized"] = False, 0  # what an older rule stored
    (proj / "media/scan/001.json").write_text(json.dumps(d))
    write_scan(proj, "2", closeups, [0.9, 0.9])  # write_scan stores fighters_likely: true
    r = rows(cli, proj)
    assert [(x["scan"]["fighters_likely"], x["scan"]["fighter_sized"]) for x in r] == [(True, 2), (False, 0)]
    assert [x["id"] for x in rows(cli, proj, "--likely")] == ["001"]


def test_scan_goes_stale_when_the_threshold_changes(cli, proj):
    cli("shots", "add", "--start", "1", "--duration", "2", "--state", "candidate", "-p", proj)
    write_scan(proj, "1", BOXES, [0.9, 0.85, 0.6])
    assert rows(cli, proj)[0]["scan"]["status"] == "ok"
    Project(proj).shot("1").modify(lambda i: i.settings.params.update(detect_track={"det_threshold": 0.5}))
    assert rows(cli, proj)[0]["scan"]["status"] == "stale"
    ev = json.loads(cli("pick", "1", "--fighters", "0,7", "-p", proj, "--json").output)
    assert ev["source"] == "" and ev["prompt_frame"] is None  # the stale scan's numbers are not used


def test_scan_threshold_matches_the_runner(project):
    """scan_threshold reads the settings layers as Runner.params_for does (default, project, shot)."""
    from afv.core.project import DET_THRESHOLD_DEFAULT, scan_threshold
    from afv.models.track import TrackParams

    assert DET_THRESHOLD_DEFAULT == TrackParams().det_threshold
    sh = project.add_shot(1.0, 2.0)
    assert scan_threshold(project.info, sh.info) == 0.4
    info = project.info
    info.settings.params["detect_track"] = {"det_threshold": 0.3}
    project.save_info(info)
    assert scan_threshold(project.info, sh.info) == 0.3
    sh.modify(lambda i: i.settings.params.update(detect_track={"det_threshold": 0.7}))
    assert scan_threshold(project.info, sh.info) == 0.7


def test_one_bad_shot_json_does_not_hide_the_others(cli, proj):
    for start in (1, 4, 7):
        cli("shots", "add", "--start", start, "--duration", "2", "-p", proj)
    p = proj / "shots/002/shot.json"
    p.write_text(p.read_text().replace('"queued"', '"chosen"'))
    r = rows(cli, proj)
    assert [(x["id"], x["error"] is None) for x in r] == [("001", True), ("002", False), ("003", True)]
    assert "shot.json is invalid" in r[1]["error"] and r[1]["state"] is None
    assert [x["id"] for x in rows(cli, proj, "--state", "queued")] == ["001", "002", "003"]  # errors listed
    assert "  002  ERROR" in cli("shots", "list", "-p", proj).output
    out = cli("batch", "--to", "camera", "-p", proj, "--json", code=1).output
    done = json.loads(out.splitlines()[-1])
    assert outs(done)["002"] == "failed" and set(outs(done)) == {"001", "002", "003"}


def test_concurrent_adds_get_distinct_ids(project):
    import threading

    errors: list[Exception] = []

    def add(k):
        try:
            for j in range(5):
                project.add_shot(float(k * 10 + j), 0.5)
        except Exception as e:  # pragma: no cover - the failure being tested
            errors.append(e)

    threads = [threading.Thread(target=add, args=(k,)) for k in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors and len(project.shot_ids()) == 30
    starts = sorted(project.shot(i).info.start for i in project.shot_ids())
    assert starts == sorted(float(k * 10 + j) for k in range(6) for j in range(5))  # nothing overwritten


def test_add_shot_skips_a_leftover_folder(project):
    project.add_shot(0.0, 1.0)
    (project.shots_dir / "002").mkdir()  # a half-made shot (no shot.json)
    assert project.add_shot(1.0, 1.0).id == "003"
    with pytest.raises(ProjectError, match="shot 003 already exists"):
        project.add_shot(2.0, 1.0, shot_id="003")


def test_a_state_change_does_not_revert_a_concurrent_trim(project):
    """(The reviewer's case) a state change that read shot.json before a trim must not write the old
    interval back."""
    import afv.core.project as pm

    sh = project.add_shot(1.0, 2.0)
    old = sh.info
    sh.update(start=1.5, duration=1.5)
    orig = pm.Shot.info
    try:
        pm.Shot.info = property(lambda self: old)  # what the state change read before the trim
        sh.update(state="skipped")
    finally:
        pm.Shot.info = orig
    i = project.shot("1").info
    assert (i.start, i.duration, i.state) == (1.5, 1.5, "skipped")


def test_concurrent_updates_of_different_fields_all_survive(project):
    import threading

    sh = project.add_shot(1.0, 2.0)

    def notes():
        for j in range(40):
            project.shot("1").update(notes=f"n{j}")

    def states():
        for j in range(40):
            project.shot("1").update(state=("skipped", "queued")[j % 2])

    def settings():
        for j in range(40):
            project.shot("1").modify(lambda i, j=j: i.settings.params.update(smooth={"strength": j / 100}))

    threads = [threading.Thread(target=f) for f in (notes, states, settings)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    i = sh.info
    assert (i.notes, i.state, i.settings.params["smooth"]["strength"]) == ("n39", "queued", 0.39)


def test_batch_reads_the_queue_after_each_shot_and_logs(cli, proj, monkeypatch):
    from afv.cli import batch_status
    from afv.core.runner import Runner

    for start, state in ((1, "queued"), (4, "queued"), (7, "candidate")):
        cli("shots", "add", "--start", start, "--duration", "2", "--state", state, "-p", proj)
    real = Runner.run

    def run(self, *a, **k):  # while the first shot runs, the user queues 003 and skips 002
        if self.shot.id == "001":
            Project(proj).shot("3").update(state="queued")
            Project(proj).shot("2").update(state="skipped")
        return real(self, *a, **k)

    monkeypatch.setattr(Runner, "run", run)
    out = cli("batch", "--to", "camera", "-p", proj, "--json", code=3).output
    evs = [json.loads(x) for x in out.splitlines()]
    done = evs[-1]
    assert outs(done) == {"001": "needs_pick", "003": "needs_pick"} and done["not_selected"] == 1
    assert [e["event"] for e in evs[:1]] == ["batch_shot"]  # batch_start goes to the log only
    st = batch_status(Project(proj))
    assert not st["running"] and st["finished"] and not st["interrupted"]
    assert st["batch_id"] == done["batch_id"]
    assert [o["shot"] for o in st["outcomes"]] == ["001", "003"]
    assert st["done"]["outcomes"] == done["outcomes"]
    logged = [json.loads(x) for x in (proj / st["log"]).read_text().splitlines()]
    assert logged[0]["event"] == "batch_start"
    assert [e["event"] for e in logged if e["event"] in ("batch_shot", "shot_done")] == [
        "batch_shot", "shot_done", "batch_shot", "shot_done"]
    assert (proj / "media/batch/current.json").is_file()


def test_one_batch_per_project(cli, proj):
    from afv.cli import BatchSession, batch_running, batch_status

    cli("shots", "add", "--start", "1", "--duration", "2", "-p", proj)
    p = Project(proj)
    with BatchSession(p, "held-1", {"to": "export"}):
        assert batch_running(p) and batch_status(p)["running"] and batch_status(p)["batch_id"] == "held-1"
        assert "a batch is already running" in cli("batch", "-p", proj, code=1).output
    assert not batch_running(p)
    st = batch_status(p)
    assert st["interrupted"] and not st["finished"]  # held-1 ended without batch_done


def test_batch_cancel_never_signals_a_stale_pid(proj, monkeypatch):
    """current.json still naming the previous batch (or any process that is not that afv batch): no
    signal is sent."""
    import subprocess
    import sys

    import afv.cli as cli_mod
    from afv.cli import BatchSession, _is_batch_process, batch_cancel, batch_dir

    p = Project(proj)
    opened: list[bool] = []
    real_open = open

    def watch_open(path, *a, **k):  # current.json is written before the log is opened
        if str(path).endswith(".jsonl"):
            opened.append((batch_dir(p) / "current.json").is_file())
        return real_open(path, *a, **k)

    monkeypatch.setattr(cli_mod, "open", watch_open, raising=False)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    named = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "batch", "--batch-id",
                              "b-2"])
    try:
        with BatchSession(p, "b-2", {"to": "export"}):  # this test holds the batch lock
            assert opened == [True]
            cur = batch_dir(p) / "current.json"
            data = json.loads(cur.read_text())
            cur.write_text(json.dumps({**data, "pid": sleeper.pid, "batch_id": "b-1"}))  # the old batch
            res = batch_cancel(p, wait_s=0)
            assert res["cancelled"] is False and "no running batch found" in res["message"]
            assert sleeper.poll() is None  # not signalled
            assert _is_batch_process(p, named.pid, "b-2") and not _is_batch_process(p, named.pid, "b-1")
            assert not _is_batch_process(p, sleeper.pid, "b-1") and not _is_batch_process(p, 999999, "b-1")
    finally:
        for proc in (sleeper, named):
            proc.kill()
            proc.wait()
