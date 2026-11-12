"""``afv pick`` writes fighters.json through ``Shot.write_edit``, which validates: a lax old file is never
silently rewritten, and the message says how to get out of it (not "pick again", which would loop)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from afv.cli import app
from afv.core.edits import EditError, validate_edit
from afv.core.project import Project


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
    cli("init", video, "--project", root, "--title", "Bout", "--no-probe")
    cli("shots", "add", "--start", "0:01", "--end", "3.5", "-p", root)
    cli("run", "1", "--to", "detect_track", "-p", root, code=3)  # makes the prompt image, waits for a pick
    return root


def path(proj):
    return proj / "shots/001/edits/fighters.json"


def test_pick_on_a_clean_file_writes_a_valid_one(cli, proj):
    cli("pick", "1", "--fighters", "1,0", "--extra", "2", "-p", proj)
    first = json.loads(path(proj).read_text())
    validate_edit("fighters", first)
    cli("pick", "1", "--fighters", "1,0", "--extra", "2", "-p", proj)  # the same again
    assert json.loads(path(proj).read_text()) == first
    cli("pick", "1", "--fighters", "0,1", "-p", proj)  # another pick replaces the old one
    validate_edit("fighters", json.loads(path(proj).read_text()))


def test_pick_on_a_lax_old_file_is_refused_with_a_way_out(cli, proj):
    cli("pick", "1", "--fighters", "1,0", "--extra", "2", "-p", proj)
    lax = json.loads(path(proj).read_text())
    lax.update({"version": "1", "note": "mine", "objects": [{"id": "1", "role": "A"}]})
    path(proj).write_text(json.dumps(lax))
    before = path(proj).read_bytes()
    for args in (("--fighters", "1,0", "--extra", "2"), ("--fighters", "0,1")):
        res = cli("pick", "1", *args, "-p", proj, code=1)
        assert "edits/fighters.json:" in res.output
        assert "fix or delete edits/fighters.json first" in res.output
        assert "the file on disk is invalid too" in res.output
        assert path(proj).read_bytes() == before  # not rewritten
    path(proj).unlink()  # the way out the message gives
    cli("pick", "1", "--fighters", "1,0", "--extra", "2", "-p", proj)
    validate_edit("fighters", json.loads(path(proj).read_text()))


def test_the_run_message_for_a_lax_fighters_file_does_not_loop(cli, proj):
    cli("pick", "1", "--fighters", "1,0", "--extra", "2", "-p", proj)
    lax = json.loads(path(proj).read_text())
    lax["fighters"] = ["1", "0"]
    path(proj).write_text(json.dumps(lax))
    res = cli("run", "1", "-p", proj, code=1)
    assert "fix or delete edits/fighters.json first" in res.output
    assert "re-pick with afv pick" not in res.output


def test_prompt_and_label_writes_are_validated_too(proj):
    """``afv pick --prompt/--label`` writes through the same ``Shot.write_edit`` (cli.py)."""
    sh = Project(proj).shot("1")
    good = {"fighters": [1, 0], "extra": [], "prompt_frame": 0,
            "prompts": [{"frame": 40, "obj": 2, "box": [0, 0, 10, 10]}],
            "objects": [{"id": 3, "label": "referee"}]}
    sh.write_edit("fighters", {"fighters": [1, 0], "extra": []})
    with pytest.raises(EditError, match=r"prompts\[0\]\.box\[0\]: must be a finite number, got '0'"):
        sh.write_edit("fighters", {**good, "prompts": [{"frame": 40, "obj": 2, "box": ["0", 0, 10, 10]}]})
    with pytest.raises(EditError, match=r"objects\[0\]\.label: must be a string, got 5"):
        sh.write_edit("fighters", {**good, "objects": [{"id": 3, "label": 5}]})
    assert sh.read_edit("fighters") == {"fighters": [1, 0], "extra": []}


def test_lax_edit_file_saves_no_settings_and_exits_1(cli, proj):
    cli("pick", "1", "--fighters", "1,0", "--extra", "2", "-p", proj)
    cli("run", "1", "-p", proj)
    sh = Project(proj).shot("1")
    sh.edit_path("pins").write_text(json.dumps({"pins": [{"role": "A", "frame": "3", "preset": "p",
                                                          "root": [0, 0, 0]}]}))
    before = sh.info.settings.params
    for args in (("run", "1", "--set", "smooth.strength=0.9"), ("export", "1", "--preset", "unreal"),
                 ("batch",)):
        res = cli(*args, "-p", proj, code=1)
        assert "edits/pins.json: pins[0].frame: must be an integer >= 0, got '3'" in res.output
        assert Project(proj).shot("1").info.settings.params == before  # nothing saved
    for args in (("status", "1"), ("run", "1", "--dry-run", "--set", "smooth.strength=0.9")):
        res = cli(*args, "-p", proj, code=1)  # the table is still printed
        assert "ingest" in res.output and "smooth" in res.output and "blocked" in res.output
        assert "edits/pins.json" in res.output
    assert Project(proj).shot("1").info.settings.params == before
    sh.edit_path("pins").unlink()
    cli("status", "1", "-p", proj)


def test_blender_prints_the_bad_edit_file_and_the_detail_lines(cli, proj, monkeypatch):
    """``afv blender`` with a lax pins file: the warning names the file and the detail lines follow, as in
    ``afv take``. (blend_shot is patched: its stale fields are built the way take_inputs builds them.)"""
    from afv import take as tk

    cli("pick", "1", "--fighters", "1,0", "--extra", "2", "-p", proj)
    cli("run", "1", "-p", proj)
    sh = Project(proj).shot("1")
    sh.edit_path("pins").write_text(json.dumps({"pins": [{"role": "A", "frame": "3", "preset": "p",
                                                          "root": [0, 0, 0]}]}))

    def fake_blend(shot, out=None, **kw):
        plan = tk.plan_stale(shot, "smooth")
        return {"take": "t.json", "blend": "take.blend", "summary": None, "check": None,
                "stale": sorted({f"afv run would re-run {s}" for s in plan}),
                "stale_summary": tk.stale_summary(shot.id, [], plan)}

    monkeypatch.setattr(tk, "blend_shot", fake_blend)
    out = cli("blender", "1", "-p", proj).output
    assert "warning: edits/pins.json is invalid: fix or delete that entry (details: afv status 001)" in out
    assert "older than the current settings" not in out
    assert "    afv run would re-run smooth: blocked (edits/pins.json: pins[0].frame" in out
