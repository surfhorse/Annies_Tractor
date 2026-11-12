from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from afv.cli import app, parse_time
from afv.core.config import ConfigError
from afv.core.project import Project
from afv.core.stage import register
from afv.core.types import LicenceRecord


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
    res = cli("init", video, "--project", root, "--title", "Bout", "--no-probe")
    assert "created" in res.output
    cli("shots", "add", "--start", "0:01", "--end", "3.5", "-p", root)
    return root


def test_init_and_shots(cli, proj):
    p = Project(proj)
    assert p.info.source.title == "Bout"
    s = p.shot("1")
    assert (s.info.start, s.info.duration) == (1.0, 2.5)
    cli("shots", "add", "--start", "5", "--duration", "1", "--speed", "0.25", "--notes", "replay", "-p", proj)
    out = cli("shots", "list", "-p", proj).output
    assert "001" in out and "002" in out and "replay" in out
    lines = [json.loads(x) for x in cli("shots", "list", "-p", proj, "--json").output.splitlines()]
    assert [x["id"] for x in lines] == ["001", "002"] and lines[1]["speed_factor"] == 0.25
    res = cli("shots", "add", "--start", "5", "-p", proj, code=1)
    assert "exactly one of --end or --duration" in res.output


def test_shots_detect(cli, proj, monkeypatch):
    import afv.cli as cli_mod

    class Seg:
        def __init__(self, start, duration, confidence):
            self.start, self.duration, self.confidence = start, duration, confidence

    found = [Seg(0.0, 1.2, 1.0), Seg(1.2, 2.0, 0.8), {"start": 3.2, "end": 9.0, "confidence": 0.6}]
    seen: list[float | None] = []

    def detector(video, min_confidence=None):
        seen.append(min_confidence)
        return found, "analysis"

    monkeypatch.setattr(cli_mod, "_shot_detector", lambda: detector)
    out = cli("shots", "detect", "--dry-run", "-p", proj).output
    assert seen == [0.2]  # the CLI's default minimum cut confidence reaches the detector
    assert "3 segments, 0 added as candidates, 1 skipped" in out  # 1.2-3.2 overlaps shot 001 (1.0-3.5)
    out = cli("shots", "detect", "-p", proj).output
    assert "2 added" in out
    assert Project(proj).shot_ids() == ["001", "002", "003"]
    s3 = Project(proj).shot("3").info
    assert (s3.origin, s3.start, s3.duration, s3.confidence) == ("detected", 3.2, pytest.approx(5.8), 0.6)
    assert s3.state == "candidate" and Project(proj).shot("1").info.state == "queued"  # added by hand
    assert not (proj / "media" / "cuts.json").exists()  # the fake detector gave no analysis

    def broken():
        def fn(video, min_confidence=None):
            raise ValueError("no video stream")

        return fn

    monkeypatch.setattr(cli_mod, "_shot_detector", broken)
    assert "cut detection failed" in cli("shots", "detect", "-p", proj, code=1).output


def test_shots_detect_unavailable(cli, proj, monkeypatch):
    import afv.media

    monkeypatch.delattr(afv.media, "detect_shots")
    assert "afv shots add" in cli("shots", "detect", "-p", proj, code=1).output


def test_pick_flow(cli, proj):
    res = cli("run", "1", "-p", proj, code=3)
    assert "afv pick 001 --fighters" in res.output and "prompt.jpg" in res.output

    res = cli("pick", "1", "-p", proj)
    assert "prompt image:" in res.output and "  2: box [110, 20, 130, 80]" in res.output

    assert "exactly two" in cli("pick", "1", "--fighters", "1", "-p", proj, code=1).output
    assert "no detection [7]" in cli("pick", "1", "--fighters", "1,7", "-p", proj, code=1).output
    assert (
        "must not repeat" in cli("pick", "1", "--fighters", "1,0", "--extra", "0", "-p", proj, code=1).output
    )

    res = cli("pick", "1", "--fighters", "1,0", "--extra", "2", "-p", proj)
    assert "next: afv run 001" in res.output
    edit = json.loads((proj / "shots/001/edits/fighters.json").read_text())
    assert edit == {
        "fighters": [1, 0],
        "extra": [2],
        "prompt_frame": 0,
        "boxes": {"1": [60, 10, 100, 90], "0": [10, 10, 50, 90], "2": [110, 20, 130, 80]},
        "shot": {"start": 1.0, "duration": 2.5, "source_sha256": Project(proj).info.source.sha256},
    }

    res = cli("run", "1", "-p", proj)
    assert "camera" in res.output and "done in" in res.output
    assert "current pick: fighters [1, 0]" in cli("pick", "1", "-p", proj).output

    events = [json.loads(x) for x in cli("run", "1", "-p", proj, "--json").output.splitlines()]
    assert {e["event"] for e in events} == {"plan", "stage_skip", "done", "result"}
    assert events[-1]["outcome"] == "ok"


def test_pick_with_nobody_detected(cli, proj):
    cli("run", "1", "--to", "detect_track", "-p", proj, code=3)
    prompt = next((proj / "shots/001/stages/detect_track").glob("v*/prompt.json"))
    prompt.write_text('{"prompt_frame": 0, "boxes": [], "scores": []}')
    assert "nobody detected on frame 0" in cli("pick", "1", "-p", proj).output
    res = cli("pick", "1", "--fighters", "0,1", "-p", proj, code=1)
    assert "detect_track.prompt_frame=N" in res.output


def test_pick_keeps_reprompts_only_for_the_same_picks(cli, proj):
    p = Project(proj).shot("1")
    reprompts = {
        "prompts": [{"frame": 40, "obj": 2, "detection": 5}],
        "objects": [{"id": 2, "label": "referee"}],
    }
    p.write_edit("fighters", {"fighters": [0, 1], "extra": [2], "prompt_frame": 0, **reprompts})
    cli("run", "1", "--to", "detect_track", "-p", proj)
    res = cli("pick", "1", "--fighters", "0,1", "--extra", "2", "-p", proj)  # same picks: kept
    assert "dropped" not in res.output
    assert {k: p.read_edit("fighters")[k] for k in reprompts} == reprompts

    res = cli("pick", "1", "--fighters", "1,0", "--extra", "2", "-p", proj)  # A and B swapped: dropped
    assert "dropped 1 re-prompt and 1 label" in res.output and "--keep-prompts" in res.output
    data = p.read_edit("fighters")
    assert "prompts" not in data and "objects" not in data and data["fighters"] == [1, 0]

    p.write_edit("fighters", {**data, **reprompts})
    res = cli("pick", "1", "--fighters", "0,1", "--keep-prompts", "-p", proj)
    assert "dropped" not in res.output and p.read_edit("fighters")["prompts"] == reprompts["prompts"]


def test_set_is_validated_and_saved(cli, proj):
    Project(proj).shot("1").write_edit("fighters", {"fighters": [1, 0], "extra": []})
    res = cli("run", "1", "--set", "smooth.strenght=2", "-p", proj, code=1)
    assert "unknown setting(s) for stage smooth: strenght" in res.output
    assert "error:" in res.output
    assert "unknown stage" in cli("run", "1", "--set", "smoth.strength=2", "-p", proj, code=1).output
    cli("run", "1", "--set", "smooth.strength=0.9", "--camera", "static", "-p", proj)
    info = Project(proj).shot("1").info
    assert info.settings.params == {"smooth": {"strength": 0.9}} and info.settings.impl == {
        "camera": "static"
    }
    out = cli("status", "1", "-p", proj).output
    assert "smooth" in out and "fresh" in out and "stale" not in out
    cli("run", "1", "--set", "smooth.strength=null", "-p", proj)
    assert Project(proj).shot("1").info.settings.params == {}
    out = cli("run", "1", "--dry-run", "--from", "pose3d", "-p", proj).output
    assert "forced" in out


def test_export_and_provenance(cli, proj):
    Project(proj).shot("1").write_edit("fighters", {"fighters": [1, 0], "extra": []})
    res = cli("export", "1", "--preset", "unreal", "-p", proj)
    assert "exported A_unreal.bvh" in res.output
    assert (proj / "shots/001/exports/A_unreal.bvh").read_text() == "HIERARCHY fps 30.0"
    prov = json.loads(cli("provenance", "-p", proj, "--json").output)
    s = prov["shots"][0]
    assert prov["source"]["title"] == "Bout" and s["in_s"] == 1.0 and s["out_s"] == 3.5
    assert s["stages"]["pose3d"]["licences"][0]["component"] == "fake-gemx"
    assert s["exports"][0]["path"].endswith("A_unreal.bvh") and len(s["exports"][0]["sha256"]) == 64
    text = cli("provenance", "1", "-p", proj).output
    assert "sha256" in text and "fake-gemx" in text


def test_batch(cli, proj):
    cli("shots", "add", "--start", "5", "--duration", "1", "-p", proj)
    Project(proj).shot("1").write_edit("fighters", {"fighters": [1, 0], "extra": []})
    res = cli("batch", "-p", proj, code=3)
    assert "1 ok, 1 waiting for a pick, 0 failed" in res.output


def test_models_check(cli, proj, fakes):
    res = cli("models", "--check")
    assert "fake-gemx" in res.output and "commercial ok" in res.output

    @register("unknown")
    class Mystery(fakes.Pose):
        licences = (LicenceRecord("mystery-net", "?", "?", "?", None),)

    res = cli("models", "--check", code=1)
    assert "UNKNOWN" in res.output and "problem: pose3d/unknown: mystery-net" in res.output
    data = json.loads(cli("models", "--json").output)
    assert any("mystery-net" in p for p in data["problems"])


def test_missing_stage_message(cli, proj, registry):
    del registry["smooth"]
    res = cli("run", "1", "-p", proj, code=1)
    assert "stage 'smooth' has no implementation yet" in res.output


def test_no_project(cli, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "no afv project" in cli("shots", "list", code=1).output


def test_run_frames_needs_from_and_dry_run_shows_it(cli, proj, fakes, do_pick):
    do_pick(Project(proj).shot("1"))
    cli("run", "1", "--to", "smooth", "-p", proj)
    res = cli("run", "1", "--frames", "2:5", "-p", proj, code=1)
    assert "--frames needs --from" in res.output
    res = cli("run", "1", "--frames", "5:2", "--from", "smooth", "-p", proj, code=1)
    assert "0 <= a < b" in res.output
    args = ("--to", "smooth", "--from", "smooth", "--frames", "2:5", "--dry-run")
    out = cli("run", "1", *args, "-p", proj).output
    assert "smooth" in out and "(in full: no range support)" in out


def test_parse_time():
    assert parse_time("62.5") == 62.5 and parse_time("1:02.5") == 62.5 and parse_time("1:00:00") == 3600
    for bad in ("nan", "inf", "1:nan"):
        with pytest.raises(ConfigError):
            parse_time(bad)


def test_shots_add_rejects_non_finite(cli, proj):
    assert (
        "finite"
        in cli("shots", "add", "--start", "1", "--end", "2", "--speed", "nan", "-p", proj, code=1).output
    )
    assert "bad time" in cli("shots", "add", "--start", "inf", "--duration", "1", "-p", proj, code=1).output


def test_export_fps_and_preset_reach_the_stage_and_are_saved(cli, proj, fakes, do_pick):
    # round 2: retime is wired, so --fps is back; both options are saved to the shot like --set
    do_pick(Project(proj).shot("1"))
    out = cli("export", "1", "--preset", "unreal", "--fps", "60", "-p", proj).output
    assert "A_unreal.bvh" in out
    sh = Project(proj).shot("1")
    assert sh.info.settings.params["export"] == {"preset": "unreal", "fps": 60.0}
    assert (sh.exports_dir / "A_unreal.bvh").read_text() == "HIERARCHY fps 60.0"
    cli("run", "1", "-p", proj)  # a plain run (now through export) keeps that preset: nothing re-runs
    assert fakes.calls.count("export") == 1
    assert (sh.exports_dir / "A_unreal.bvh").is_file()


def test_models_check_fails_on_non_commercial_licence(cli, proj, fakes):
    # NFR-12: export requires commercial-OK licences, so a known non-commercial checkpoint fails --check
    @register("nc")
    class NonCommercial(fakes.Pose):
        licences = (LicenceRecord("nc-net", "MIT", "CC BY-NC 4.0", "?", False, weight_sha256="ab"),)

    res = cli("models", "--check", code=1)
    assert "problem: pose3d/nc: nc-net: outputs not cleared for commercial use" in res.output
    import re

    help_text = re.sub(r"[\s│]+", " ", cli("models", "--help").output)  # rich wraps the help in a box
    assert "not allowed (NFR-12)" in help_text


@pytest.mark.parametrize(
    "args",
    [
        ("export", "1", "--preset", "unreal"),
        ("run", "1", "--set", "smooth.strength=0.7"),
        ("run", "1", "--camera", "static"),
    ],
)
def test_settings_are_not_saved_while_another_process_runs_the_shot(cli, proj, fakes, do_pick, args):
    import fcntl
    import os

    sh = Project(proj).shot("1")
    do_pick(sh)
    before = (sh.dir / "shot.json").read_text()
    sh.work_root.mkdir(parents=True, exist_ok=True)
    fd = os.open(sh.work_root / ".lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # "another afv process" owns the shot
        out = cli(*args, "-p", proj, code=1).output
        assert "already being run" in out
        assert (sh.dir / "shot.json").read_text() == before  # nothing saved
    finally:
        os.close(fd)
    cli(*args, "-p", proj)  # once the shot is free, the same command saves and runs
    assert (sh.dir / "shot.json").read_text() != before


def test_malformed_prompt_json_is_readable(cli, proj):
    cli("run", "1", "--to", "detect_track", "-p", proj, code=3)
    prompt = next((proj / "shots/001/stages/detect_track").glob("v*/prompt.json"))
    prompt.write_text("{broken")
    assert "cannot read" in cli("pick", "1", "--fighters", "1,0", "-p", proj, code=1).output
    prompt.write_text('{"boxes": [[1, 2, "x", 4]]}')
    assert "malformed detections" in cli("pick", "1", "-p", proj, code=1).output


def test_os_errors_are_one_readable_line(cli, proj, monkeypatch):
    def denied(self, *a, **k):
        raise PermissionError(13, "Permission denied", "/mnt/d/x/shot.json")

    monkeypatch.setattr(Project, "add_shot", denied)
    res = cli("shots", "add", "--start", "1", "--end", "2", "-p", proj, code=1)
    assert "error: Permission denied (/mnt/d/x/shot.json)" in res.output and "Traceback" not in res.output


def test_init_rehash(cli, proj, video):
    out = cli("init", video, "--project", proj, "--rehash", "--no-probe").output
    assert "unchanged" in out
    video.write_bytes(b"new content")
    assert "source video changed" in cli("run", "1", "-p", proj, code=1).output
    assert (
        "changed: stages will re-run"
        in cli("init", video, "--project", proj, "--rehash", "--no-probe").output
    )
    assert "already has" in cli("init", video, "--project", proj, "--no-probe", code=1).output


def test_progress_output_is_throttled(capsys):
    from afv.cli import JsonReporter, TextReporter

    for rep in (TextReporter(), JsonReporter()):
        for i in range(200):
            rep({"event": "progress", "shot": "001", "stage": "smooth", "frac": i / 200, "message": f"{i}"})
        assert 1 <= len(capsys.readouterr().out.splitlines()) <= 3
