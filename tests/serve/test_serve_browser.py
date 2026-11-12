"""``afv serve`` shot-browser commands (round 6): init, shots, shot_add, shot_set, shots_detect, shots_scan,
pick and batch, against a real worker subprocess (fake stages, fake cut detection and a fake track worker:
see worker_main.py)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from afv.core.project import Project, load_scan

SEGMENTS = [{"start": 0.0, "duration": 2.0, "confidence": 1.0}, {"start": 2.0, "end": 4.0, "confidence": 0.7},
            {"start": 4.0, "duration": 2.0, "confidence": 0.5}]


def write_video(path: Path, seconds: float = 6.0, fps: int = 25) -> Path:
    """Three 2 s scenes of noise in different tints (160x96)."""
    import av

    rng = np.random.default_rng(0)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps, options={"crf": "20", "preset": "ultrafast"})
        s.width, s.height, s.pix_fmt = 160, 96, "yuv420p"
        for i in range(int(seconds * fps)):
            tint = np.array([(200, 60, 60), (60, 200, 60), (60, 60, 200)][min(2, i // (2 * fps))])
            img = np.clip(tint + rng.integers(-30, 30, (96, 160, 3)), 0, 255).astype(np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


@pytest.fixture
def video(tmp_path) -> Path:
    (tmp_path / "src").mkdir(exist_ok=True)
    return write_video(tmp_path / "src" / "bout.mp4")


@pytest.fixture
def browser(new_client, worker_env):
    """A worker with fake cut detection and a fake track worker."""
    return new_client({**worker_env, "AFV_TEST_SEGMENTS": json.dumps(SEGMENTS), "AFV_TEST_FAKE_TRACK": "1"})


def kinds(evs: list[dict]) -> list[str]:
    return [e.get("kind") or e["event"] for e in evs]


def wait_for(cond, timeout=20.0, what="condition"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if v := cond():
            return v
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def test_init_list_add_set(browser, tmp_path, video):
    root = tmp_path / "bout.afv"
    res, evs = browser.request("init", {"video": str(video), "project": str(root)})
    assert res["event"] == "result", res
    assert res["created"] == str(root) and res["detect"]["added"] == 3
    assert res["counts"] == {"candidate": 3, "queued": 0, "skipped": 0}
    assert [r["id"] for r in res["shots"]] == ["001", "002", "003"]
    assert res["proxy"] is None  # 96 px tall: "auto" makes no proxy
    assert res["source"]["width"] == 160 and res["source"]["fps"] == 25.0 and res["scan_dir"] == "media/scan"
    assert any(e["event"] == "progress" for e in evs) and "detect_done" in kinds(evs)
    err, _ = browser.request("init", {"video": str(video), "project": str(root)})
    assert err["event"] == "error" and "already an afv project" in err["message"]
    err, _ = browser.request("init", {"video": str(tmp_path / "nope.mp4"), "project": str(tmp_path / "x")})
    assert err["event"] == "error" and "video not found" in err["message"]

    p = {"project": str(root)}
    t0 = time.monotonic()
    res, _ = browser.request("shots", p)
    assert res["event"] == "result" and len(res["shots"]) == 3 and time.monotonic() - t0 < 2
    row = res["shots"][1]
    assert (row["state"], row["start"], row["end"], row["confidence"], row["picked"]) == (
        "candidate", 2.0, 4.0, 0.7, False)
    assert row["scan"]["status"] == "none"

    res, _ = browser.request("shot_add", {**p, "start": "0:04.5", "duration": 1.0, "notes": "by hand"})
    assert res["event"] == "result" and res["shot"] == "004" and res["row"]["state"] == "queued"
    for bad, msg in (({"end": 2.0}, '"start"'), ({"start": 1, "end": 2, "duration": 1}, "exactly one"),
                     ({"start": 1, "end": 2, "state": "done"}, "bad shot state"),
                     ({"start": 3, "end": 2}, "bad shot interval"), ({"start": "x", "end": 2}, "bad time")):
        err, _ = browser.request("shot_add", {**p, **bad})
        assert err["event"] == "error" and msg in err["message"], (bad, err)

    t0 = time.monotonic()
    res, _ = browser.request("shot_set", {**p, "shot": "2", "start": 2.5})  # the end stays
    assert time.monotonic() - t0 < 2
    assert res["changed"] == ["start", "duration"] and (res["row"]["start"], res["row"]["end"]) == (2.5, 4.0)
    res, _ = browser.request("shot_set", {**p, "shot": "2", "state": "queued", "speed": 0.5})
    assert res["changed"] == ["speed_factor", "state"] and res["row"]["state"] == "queued"
    res, _ = browser.request("shots", {**p, "state": ["queued"]})
    assert [r["id"] for r in res["shots"]] == ["002", "004"]
    assert res["counts"] == {"candidate": 2, "queued": 2, "skipped": 0}
    for bad, msg in (({"shot": "2", "state": "done"}, "bad shot state"),
                     ({"shot": "2", "end": 1.0}, "interval"),
                     ({"shot": "9", "state": "queued"}, "no shot"),
                     ({"shot": "2", "speed": "fast"}, "number")):
        err, _ = browser.request("shot_set", {**p, **bad})
        assert err["event"] == "error" and msg in err["message"], (bad, err)

    res, _ = browser.request("shots_detect", p)  # everything is covered already
    assert res["event"] == "result" and res["detect"]["added"] == 0 and res["detect"]["skipped_overlap"] == 3
    assert json.loads((root / "shots/002/shot.json").read_text())["state"] == "queued"  # kept


def test_scan_and_pick(browser, tmp_path, video):
    proj = Project.create(tmp_path / "p.afv", video)
    for k in range(3):
        proj.add_shot(2.0 * k, 2.0, state="candidate")
    p = {"project": str(proj.root)}
    res, evs = browser.request("shots_scan", {**p, "shots": ["1", "2"]}, timeout=120)
    assert res["event"] == "result", res
    assert sorted(res["scanned"]) == ["001", "002"] and res["shots"] == ["001", "002"]
    scanned = [e for e in evs if e.get("kind") == "scanned"]
    assert sorted(e["shot"] for e in scanned) == ["001", "002"]
    assert scanned[0]["row"]["scan"]["status"] == "ok" and scanned[0]["row"]["scan"]["people"] == 2
    assert any(e["event"] == "progress" and e["stage"] == "scan" for e in evs)
    assert load_scan(proj.shot("1"))["frame"] == 15

    t0 = time.monotonic()
    res, _ = browser.request("pick", {**p, "shot": "1", "fighters": [1, 0], "frame": 15})
    assert res["event"] == "result", res
    assert time.monotonic() - t0 < 2
    assert res["source"] == "scan" and res["frame"] == 15 and res["row"]["picked"]
    edit = json.loads((proj.root / "shots/001/edits/fighters.json").read_text())
    assert edit["prompt_frame"] == 15 and edit["boxes"]["1"] == [90.0, 12.0, 125.0, 88.0]

    res, _ = browser.request("pick", {**p, "shot": "2"})  # the prompt to show: the scan's
    assert res["source"] == "scan" and res["frame"] == 15 and len(res["prompt"]["boxes"]) == 2
    assert Path(res["image"]).is_file() and res["current"] is None

    res, evs = browser.request("pick", {**p, "shot": "3"}, timeout=60)  # not scanned: scanned now
    assert res["event"] == "result" and res["source"] == "scan", res
    assert load_scan(proj.shot("3")) is not None and any(e["event"] == "progress" for e in evs)

    for bad, msg in (({"fighters": [0]}, "exactly two"), ({"fighters": [0, 5]}, "no detection [5]"),
                     ({"fighters": [0, 0]}, "repeated"),
                     ({"fighters": [0, 1], "extra": [1]}, "must not repeat"),
                     ({"fighters": "0,1"}, "list"), ({"extra": [1]}, '"extra" needs "fighters"'),
                     ({"frame": -1}, "frame")):
        err, _ = browser.request("pick", {**p, "shot": "2", **bad})
        assert err["event"] == "error" and msg in err["message"], (bad, err)

    res, _ = browser.request("shots_scan", p)  # every candidate: all up to date
    assert res["scanned"] == [] and sorted(res["up_to_date"]) == ["001", "002", "003"]
    res, _ = browser.request("shots", p)
    assert [r["scan"]["thumbnail"] for r in res["shots"]] == [f"media/scan/00{k}.jpg" for k in (1, 2, 3)]


def test_scan_cancel_keeps_what_finished(new_client, worker_env, tmp_path, video):
    c = new_client({**worker_env, "AFV_TEST_FAKE_TRACK": "1", "AFV_TEST_TRACK_SLEEP": "1.5"})
    proj = Project.create(tmp_path / "p.afv", video)
    for k in range(3):
        proj.add_shot(2.0 * k, 2.0, state="candidate")
    p = {"project": str(proj.root)}
    rid = c.send("shots_scan", p)
    deadline = time.monotonic() + 60
    while load_scan(proj.shot("1")) is None:  # the first shot's people are in: cancel the rest
        assert time.monotonic() < deadline, c.by_id.get(rid)
        time.sleep(0.1)
    res, _ = c.request("cancel", {"target_id": rid})
    assert res["state"] == "running"
    err = c.wait(rid, 30)[-1]
    assert err["event"] == "error" and err["cancelled"], err
    assert "001" in err["report"]["scanned"] and len(err["report"]["scanned"]) < 3
    res, _ = c.request("shots_scan", p, timeout=60)
    assert "001" in res["up_to_date"] and sorted(res["scanned"] + res["up_to_date"]) == ["001", "002", "003"]


def test_batch_runs_queued_shots_and_cancels(client, project):
    project.add_shot(4.0, 2.0, state="candidate")
    project.shot("2").update(state="skipped")
    p = {"project": str(project.root)}
    res, evs = client.request("batch", {**p, "to": "smooth"}, timeout=120)
    assert res["event"] == "result", res
    assert [o["shot"] for o in res["outcomes"]] == ["001"] and res["ok"] == 1 and res["exit_code"] == 0
    assert res["attached"] is False and res["log"] == f"media/batch/{res['batch_id']}.jsonl"
    assert kinds(evs)[:2] == ["batch_started", "batch_start"]
    assert [e["shot"] for e in evs if e.get("kind") == "batch_shot"] == ["001"]
    assert [e["outcome"] for e in evs if e.get("kind") == "shot_done"] == ["ok"]
    assert any(e["event"] == "progress" and e.get("shot") == "001" for e in evs)
    assert not project.shot("3").meta("ingest")  # the candidate did not run

    res, _ = client.request("batch", {**p, "to": "smooth", "state": ["candidate", "skipped"]}, timeout=120)
    assert [o["shot"] for o in res["outcomes"]] == ["002", "003"]
    res, _ = client.request("batch", {**p, "to": "smooth", "shots": ["3"]}, timeout=120)
    assert [o["shot"] for o in res["outcomes"]] == ["003"]
    for bad, msg in (({"state": "done"}, "bad shot state"), ({"to": "nowhere"}, "unknown stage"),
                     ({"code_changes": "maybe"}, "code_changes"), ({"shots": ["9"]}, "no shot '9'")):
        err, _ = client.request("batch", {**p, **bad})
        assert err["event"] == "error" and msg in err["message"], (bad, err)

    project.shot("2").update(state="queued")
    rid = client.send("batch", {**p, "to": "smooth", "sets": ["contacts.hold_s=30"]})
    wait_for(lambda: any(e.get("kind") == "batch_shot" for e in client.by_id.get(rid, [])), 60, "batch_shot")
    time.sleep(0.5)
    res, _ = client.request("cancel", {"target_id": rid})
    assert res["state"] == "running"
    err = client.wait(rid, 60)[-1]
    assert err["event"] == "error" and err["cancelled"] and err["report"]["cancelled"], err
    assert len(err["report"]["outcomes"]) < 2
    st, _ = client.request("batch_status", p)
    assert not st["running"] and st["done"]["cancelled"] and st["finished"]


def _hold(project, shot: str, seconds: float) -> None:
    project.shot(shot).modify(lambda i: i.settings.params.update(contacts={"hold_s": seconds}))


def test_batch_survives_the_client(new_client, worker_env, project):
    """Close the client mid-batch: the batch finishes; a new client sees it with batch_status; a shot
    queued while it runs is run too."""
    project.add_shot(4.0, 2.0, state="candidate")
    _hold(project, "1", 8.0)
    p = {"project": str(project.root)}
    c1 = new_client(worker_env)
    rid = c1.send("batch", {**p, "to": "smooth"})
    wait_for(lambda: any(e.get("kind") == "batch_shot" for e in c1.by_id.get(rid, [])), 60, "batch_shot")
    c1.close()  # Blender went away: the worker exits, the batch must go on
    c2 = new_client(worker_env)
    st, _ = c2.request("batch_status", p)
    assert st["running"] and st["current"]["shot"] in ("001", "002"), st
    res, _ = c2.request("shot_set", {**p, "shot": "3", "state": "queued"})  # queued while it runs
    assert res["event"] == "result", res
    res, _ = c2.request("batch", p, timeout=120)  # a second start attaches to the running batch
    assert res["event"] == "result" and res["attached"] and res["batch_id"] == st["batch_id"], res
    assert [o["shot"] for o in res["outcomes"]] == ["001", "002", "003"] and res["ok"] == 3
    st, _ = c2.request("batch_status", p)
    assert not st["running"] and st["finished"] and not st["interrupted"] and len(st["outcomes"]) == 3
    assert st["done"]["batch_id"] == st["batch_id"] and project.shot("3").meta("smooth")


def test_batch_cancel_from_another_client(new_client, worker_env, project):
    _hold(project, "1", 30.0)
    p = {"project": str(project.root)}
    c1 = new_client(worker_env)
    rid = c1.send("batch", {**p, "to": "smooth"})
    wait_for(lambda: any(e.get("kind") == "batch_shot" for e in c1.by_id.get(rid, [])), 60, "batch_shot")
    c2 = new_client(worker_env)
    t0 = time.monotonic()
    res, _ = c2.request("batch_cancel", p, timeout=90)
    assert res["event"] == "result" and res["cancelled"] and not res["running"], res
    assert time.monotonic() - t0 < 30 and res["done"]["cancelled"] and res["outcomes"] == []
    err = c1.wait(rid, 30)[-1]
    assert err["event"] == "error" and err["cancelled"], err
    assert not project.shot("1").meta("contacts")  # the stopped shot wrote no result
    res, _ = c2.request("batch_cancel", p)
    assert res["event"] == "result" and res["cancelled"] is False  # nothing runs


def test_bad_shot_json_is_a_row_with_an_error(client, project):
    p = project.shots_dir / "002" / "shot.json"
    p.write_text(p.read_text().replace('"queued"', '"chosen"'))
    res, _ = client.request("shots", {"project": str(project.root)})
    assert res["event"] == "result", res
    assert [(r["id"], r["error"] is None) for r in res["shots"]] == [("001", True), ("002", False)]
    assert res["counts"] == {"candidate": 0, "queued": 1, "skipped": 0}


def test_cancelled_pick_reports_cancelled(new_client, worker_env, tmp_path, video):
    c = new_client({**worker_env, "AFV_TEST_FAKE_TRACK": "1", "AFV_TEST_TRACK_SLEEP": "3"})
    proj = Project.create(tmp_path / "p.afv", video)
    proj.add_shot(0.0, 2.0, state="candidate")
    rid = c.send("pick", {"project": str(proj.root), "shot": "1"})  # not scanned: scans it first
    wait_for(lambda: any(e["event"] == "progress" for e in c.by_id.get(rid, [])), 60, "the scan")
    time.sleep(1.5)  # the frame is read in well under a second; the fake detector takes 3 s
    c.request("cancel", {"target_id": rid})
    err = c.wait(rid, 30)[-1]
    assert err["event"] == "error" and err["cancelled"] and not err.get("completed"), err
    assert load_scan(proj.shot("1")) is None


def test_code_change_mid_batch_restarts_the_worker(client, project, stages_file):
    """A code change while a batch streams: the stream ends detached, the worker restarts, the queued
    requests are answered, the batch goes on."""
    p = {"project": str(project.root)}
    res, _ = client.request("run", {**p, "shot": "002", "to": "smooth"})
    assert res["event"] == "result" and res["outcome"] == "ok", res
    project.shot("2").update(state="skipped")
    _hold(project, "1", 60.0)
    rid = client.send("batch", {**p, "to": "smooth"})
    wait_for(lambda: any(e.get("kind") == "batch_shot" for e in client.by_id.get(rid, [])), 60, "batch_shot")
    pid = client.request("ping")[0]["pid"]
    src = stages_file.read_text()
    time.sleep(3.2)  # past the worker's stat reuse window (afv.serve.STAT_TTL_S)
    stages_file.write_text(src.replace("* params.strength\n", "* params.strength * 2\n"))
    r = client.send("run", {**p, "shot": "002", "to": "smooth"})  # sees the change: restart
    s = client.send("shots", p)
    err = client.wait(rid, 60)[-1]
    assert err["event"] == "error" and err["detached"] and err["cancelled"] and err["batch_id"], err
    res = client.wait(s, 90)[-1]
    assert res["event"] == "result" and len(res["shots"]) == 2, res
    assert client.wait(r, 90)[-1]["event"] == "result"
    info = client.request("ping")[0]
    assert info["pid"] == pid and info["restarts"] == 1
    st, _ = client.request("batch_status", p)
    assert st["running"] and st["batch_id"] == err["batch_id"] and st["current"]["shot"] == "001", st
    res, _ = client.request("batch_cancel", p, timeout=90)
    assert res["cancelled"] and not res["running"], res


def test_project_commands_need_a_project(client, tmp_path):
    err, _ = client.request("shots", {})
    assert err["event"] == "error" and '"project"' in err["message"]
    err, _ = client.request("shots", {"project": str(tmp_path)})
    assert err["event"] == "error" and "not an afv project" in err["message"]
    err, _ = client.request("frobnicate", {})
    assert "shots_scan" in err["message"] and "batch" in err["message"]
