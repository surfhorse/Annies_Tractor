"""The shot browser in a real Blender window with simulated mouse clicks and keys (``browser_windowed.py``,
``--enable-event-simulate``), against the fake-stage ``afv serve`` of ``test_browser.py``: Pick fighters
through 3 shots with Then pick the next (clicks on boxes, number keys, Enter; the modal goes on from shot
to shot), every shortcut key pressed over the 3D view and over the timeline (and the browser camera left
untouched), and Ctrl+Z (the scene and the fields follow the project; an undo past the opening closes the
browser cleanly).

It opens a Blender window on the desktop for about 20 s, so it is opt-in: ``AFV_WINDOWED_TESTS=1``."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from take_helpers import REPO, TESTS_DIR
from test_browser import worker_argv, write_video

from afv.export import blender as bl

pytestmark = [
    pytest.mark.blender,
    pytest.mark.slow,
    pytest.mark.skipif(bl.find_blender() is None, reason="Blender not installed"),
    pytest.mark.skipif(os.environ.get("AFV_WINDOWED_TESTS") != "1",
                       reason="opens a Blender window: set AFV_WINDOWED_TESTS=1"),
]


@pytest.fixture(scope="module")
def win(tmp_path_factory):
    exe = bl.find_blender()
    tmp = tmp_path_factory.mktemp("browser_win")
    video = write_video(tmp / "bout.mp4")
    result = tmp / "result.json"
    job = {"result_path": bl.host_path(result, exe), "addon_parent": bl.host_path(REPO / "blender", exe),
           "video": bl.host_path(video, exe), "project": bl.host_path(tmp / "bout.afv", exe),
           "worker": worker_argv(tmp)}
    job_file = tmp / "job.json"
    job_file.write_text(json.dumps(job))
    cmd = [str(exe), "--factory-startup", "--enable-event-simulate", "-noaudio", "--window-geometry",
           "0", "0", "1600", "900", "--python", bl.host_path(TESTS_DIR / "browser_windowed.py", exe), "--",
           bl.host_path(job_file, exe)]
    proc = subprocess.run(cmd, capture_output=True, timeout=600, cwd=tmp)
    (tmp / "blender.log").write_bytes(proc.stdout + b"\n--- stderr ---\n" + proc.stderr)
    assert result.is_file(), (proc.stdout + proc.stderr)[-3000:].decode("utf-8", "replace")
    res = json.loads(Path(result).read_text(encoding="utf-8"))
    assert res.get("ok"), res.get("error")
    return res


def test_then_next_keeps_clicks_and_keys_working(win):
    picks = win["picks"]
    assert [p["shot"] for p in picks] == ["001", "002", "003"]
    for p in picks:
        assert p["boxes"] == 2 and p["corners"] and p["selected"] == [1, 0] and p["picked"], p
    assert picks[0]["after"] == {"active": True, "shot": "002"}  # Then next: the same modal goes on
    assert picks[1]["after"] == {"active": True, "shot": "003"}  # (number keys on shot 002)
    assert picks[2]["after"]["active"] is False and win["after_last_pick"] == {"active": False}


def test_every_shortcut_by_real_key_presses(win):
    """Pressed with event_simulate over the 3D view and over the timeline of the AFV Shots scene."""
    for kind in ("VIEW_3D", "DOPESHEET_EDITOR"):
        k = win["keys"][kind]
        assert k["page_down"] == "002" and k["page_up"] == "001", kind  # next / previous shot
        assert k["L"] in ("002", "003"), kind  # next likely shot (002 is trimmed after the 3D pass)
        assert k["I"]["start"] == pytest.approx((k["I"]["frame"] - 0.5) / 25), kind  # set in at the playhead
        assert k["O"]["end"] == pytest.approx((k["O"]["frame"] + 0.5) / 25), kind  # set out
        assert k["S"] == {"state": "skipped", "selected": "003"}, kind  # skip, then the next shot
        assert k["Q"] == {"state": "queued", "selected": "003"}, kind  # queue, then the next shot
        assert k["shift_Q"] == {"state": "queued", "picking": "001"}, kind  # queue & pick
        assert k["esc"] is False and k["enter"] == "003", kind  # Esc leaves the pick; Enter picks
        assert k["N"] == "003", kind  # N stays Blender's (the sidebar): no navigation
    cam = win["keys"]["camera"]  # Blender's own I (keyframe) and S (scale) never reach the camera
    assert not cam["action"] and cam["after"] == cam["before"] and cam["hide_select"] and not cam["selected"]


def test_undo_never_leaves_the_ui_disagreeing_with_the_project(win):
    u = win["undo"]
    a = u["after_one"]
    assert u["ret"] == "{'FINISHED'}" and a["scene"] == "AFV Shots" and a["open"]
    assert a["shot"] == "003" and a["speed_field"] == a["speed_saved"] == 0.5 and a["set_edge_poll"]
    assert u["state_op"] == ["FINISHED"]
    d = u["deep"]  # an undo past the opening: the browser is closed, nothing half-there
    assert d["scene"] is None and not d["open"] and not d["set_edge_poll"]
    assert "undo closed the shot browser" in d["status"]
