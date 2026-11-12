"""``afv serve`` end to end: a real worker subprocess with fake CPU stages (conftest.py)."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from afv.core.project import Project, stage_dir
from afv.serve import wsl_path

SPAN = {"role": "A", "start": 3, "end": 9, "state": "plant", "foot": "left"}


def where(project: Project, shot: str = "001") -> dict:
    return {"project": str(project.root), "shot": shot}


def full_run(client, project, shot="001") -> dict:
    res, _ = client.request("run", {**where(project, shot), "to": "smooth"})
    assert res["event"] == "result" and res["outcome"] == "ok", res
    return res


def root(project, stage, shot="001") -> np.ndarray:
    return project.shot(shot).load_result(stage).actors["A"].root


def wait_for(cond, timeout=20.0, what="condition"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if v := cond():
            return v
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as f:  # a zombie is gone for our purposes
        return f.read().split(") ")[-1][0] != "Z"


def test_wsl_path():
    assert str(wsl_path("D:\\Workspace\\x y\\p.afv")) == "/mnt/d/Workspace/x y/p.afv"
    assert str(wsl_path("d:/a/b")) == "/mnt/d/a/b"
    assert str(wsl_path("\\\\wsl.localhost\\Ubuntu\\home\\me\\p")) == "/home/me/p"
    assert str(wsl_path("/mnt/d/p")) == "/mnt/d/p"


def test_ping_and_bad_requests(client, project):
    res, evs = client.request("ping")
    assert res["event"] == "result" and res["pong"] and res["protocol"] == 1 and len(evs) == 1
    client.write(b"this is not json\n")
    assert client.wait(None)[-1]["event"] == "error"
    res, _ = client.request("frobnicate")
    assert res["event"] == "error" and "unknown cmd" in res["message"]
    res, _ = client.request("run", {"shot": "1"})
    assert res["event"] == "error" and "project" in res["message"]
    res, _ = client.request("run", where(project, "042"))
    assert res["event"] == "error" and "no shot" in res["message"]
    res, _ = client.request("run", {**where(project), "frames": "4:2", "from": "contacts"})
    assert res["event"] == "error" and "--frames" in res["message"]
    res, _ = client.request("ping")
    assert res["pong"]  # still alive after all that


def test_run_events_status_and_stdout_hygiene(client, project):
    res = full_run(client, project)
    assert res["exit_code"] == 0 and res["elapsed_s"] > 0
    assert [s["state"] for s in res["stages"]] == ["ran"] * 4
    evs = client.by_id[1]
    kinds = [e.get("kind") for e in evs if e["event"] == "log"]
    assert kinds[0] == "plan" and "stage_start" in kinds and "stage_done" in kinds and kinds[-1] == "done"
    prog = [e for e in evs if e["event"] == "progress"]
    assert prog and {"stage", "frac", "message", "shot"} <= set(prog[0])
    assert not client.bad, client.bad  # the fake ingest's print() went to stderr
    assert "a stray print on stdout" in client.log.read_text()
    st, _ = client.request("status", {**where(project, "1"), "to": "smooth"})
    assert st["outcome"] == "planned" and {s["state"] for s in st["stages"]} == {"fresh"}
    # settings given to run are saved to the shot
    res, _ = client.request("run", {**where(project), "to": "smooth", "sets": {"smooth.strength": 0.25}})
    assert res["outcome"] == "ok"
    assert project.shot("001").info.settings.params["smooth"] == {"strength": 0.25}


def test_edit(client, project):
    sh = project.shot("001")
    edit = {**where(project), "name": "contacts"}
    res, _ = client.request("edit", {**edit, "data": {"spans": [SPAN], "note": 1}})
    assert res["event"] == "result" and res["stages"] == ["contacts"] and not res["removed"]
    assert sh.read_edit("contacts") == {"spans": [SPAN], "note": 1}
    res, _ = client.request("edit", {**edit, "data": {"spans": [{**SPAN, "end": 2}]}})
    assert res["event"] == "error" and "end" in res["message"]
    assert sh.read_edit("contacts") == {"spans": [SPAN], "note": 1}  # unchanged
    res, _ = client.request("edit", {**where(project), "name": "contacts", "data": {"spans": []}})
    assert sh.read_edit("contacts") == {"spans": [], "note": 1}  # unknown top-level key kept
    res, _ = client.request("edit", {**where(project), "name": "nonsense", "data": {}})
    assert res["event"] == "error" and "unknown edit" in res["message"]
    pin = {"role": "B", "frame": 12, "preset": "blender", "bones": {"Hips": [0, 0, 0, 1]}, "root": None}
    res, _ = client.request("edit", {**where(project), "name": "pins", "data": {"pins": [pin]}})
    assert res["event"] == "result" and sh.read_edit("pins")["pins"][0]["frame"] == 12
    res, _ = client.request("edit", {**where(project), "name": "pins", "data": None})
    assert res["removed"] and sh.read_edit("pins") is None


def test_range_rerun(client, project):
    full_run(client, project)
    before = root(project, "smooth")
    client.request("edit", {**where(project), "name": "contacts", "data": {"spans": [{**SPAN, "start": 12,
                                                                                         "end": 18}]}})
    args = {**where(project), "from": "contacts", "to": "smooth", "frames": "10:20"}
    res, _ = client.request("run", args)
    assert res["outcome"] == "ok", res
    st = {s["name"]: s for s in res["stages"]}
    assert st["contacts"]["range"] == [10, 20] and st["smooth"]["range"] == [10, 20]
    after = root(project, "smooth")
    assert np.array_equal(after[:12], before[:12]) and np.array_equal(after[18:], before[18:])
    assert np.allclose(after[12:18], before[12:18] + 0.5)  # the painted span, smoothed by 0.5
    st, _ = client.request("status", {**where(project), "to": "smooth"})
    sm = next(s for s in st["stages"] if s["name"] == "smooth")
    assert sm["state"] == "fresh" and "range re-run of frames 10:20" in sm["reason"]
    # a paint outside the range: contacts runs in full (and smooth after it)
    client.request("edit", {**where(project), "name": "contacts", "data": {"spans": [SPAN]}})
    res, _ = client.request("run", args)
    st = {s["name"]: s for s in res["stages"]}
    assert st["contacts"]["range"] is None and "outside frames 10:20" in st["contacts"]["reason"]


def test_one_shot_in_order_other_shots_in_parallel(client, project):
    full_run(client, project)
    slow = {**where(project), "from": "contacts", "to": "smooth", "sets": ["contacts.hold_s=1.5"]}
    r1 = client.send("run", slow)
    r2 = client.send("edit", {**where(project), "name": "contacts", "data": {"spans": [SPAN]}})
    r3 = client.send("status", {**where(project), "to": "smooth"})
    r4 = client.send("ping")
    r5 = client.send("run", {**where(project, "002"), "to": "smooth"})
    for r in (r1, r2, r3, r4, r5):
        client.wait(r)

    def done_at(rid):
        return next(i for i, ln in enumerate(client.lines) if (e := json.loads(ln))["id"] == rid
                    and e["event"] in ("result", "error"))  # fmt: skip

    assert done_at(r1) < done_at(r2) < done_at(r3)
    assert done_at(r4) < done_at(r1) and done_at(r5) < done_at(r1)
    stale = {s["name"]: s["state"] for s in client.by_id[r3][-1]["stages"]}
    assert stale["contacts"] == "stale"  # the edit landed after the run, the status after the edit


def test_cancel_running_and_queued(client, project):
    full_run(client, project)
    before = project.shot("001").meta("contacts")["output_hash"]
    hold = {**where(project), "from": "contacts", "to": "smooth", "sets": ["contacts.hold_s=30"]}
    r1 = client.send("run", hold)
    r2 = client.send("status", where(project))
    wait_for(lambda: any(e["event"] == "progress" for e in client.by_id.get(r1, [])), what="progress")
    res, _ = client.request("cancel", {"target_id": r2})
    assert res["state"] == "queued"
    err = client.wait(r2)[-1]
    assert err["event"] == "error" and err["cancelled"] and "before it started" in err["message"]
    t0 = time.monotonic()
    res, _ = client.request("cancel", {"target_id": r1})
    assert res["state"] == "running"
    err = client.wait(r1, 10)[-1]
    assert time.monotonic() - t0 < 5
    assert err["event"] == "error" and err["cancelled"] and err["report"]["outcome"] == "cancelled"
    assert project.shot("001").meta("contacts")["output_hash"] == before  # previous result kept
    work = project.shot("001").work_root
    assert not [p for p in work.iterdir() if p.is_dir() and "contacts" in p.name]  # work dropped
    res, _ = client.request("cancel", {"target_id": r1})
    assert res["state"] == "finished"
    res, _ = client.request("cancel", {"target_id": 999})
    assert res["state"] == "unknown"


def test_cancel_stops_a_worker_subprocess(client, project):
    full_run(client, project)
    child = {**where(project), "from": "contacts", "to": "smooth", "sets": ["contacts.child_s=60"]}
    r1 = client.send("run", child)
    work = project.shot("001").work_root
    pid_file = wait_for(lambda: next(iter(work.glob("contacts-*/child.pid")), None), what="child")
    pid = int(wait_for(lambda: pid_file.read_text().strip(), what="pid"))
    assert alive(pid)
    t0 = time.monotonic()
    client.request("cancel", {"target_id": r1})
    err = client.wait(r1, 15)[-1]
    assert err["event"] == "error" and err["cancelled"], err
    assert time.monotonic() - t0 < 10
    wait_for(lambda: not alive(pid), 5, "the child to stop")


def test_stdin_closed_exits_and_cancels(client, project):
    full_run(client, project)
    hold = {**where(project), "from": "contacts", "to": "smooth", "sets": ["contacts.hold_s=30"]}
    r1 = client.send("run", hold)
    wait_for(lambda: any(e["event"] == "progress" for e in client.by_id.get(r1, [])), what="progress")
    assert client.proc.stdin is not None
    client.proc.stdin.close()
    assert client.proc.wait(20) == 0
    assert client.wait(r1, 1)[-1]["cancelled"]
    work = project.shot("001").work_root
    assert not [p for p in work.iterdir() if p.is_dir() and "contacts" in p.name]


def test_shutdown(client, project):
    res, _ = client.request("shutdown")
    assert res["shutdown"]
    assert client.proc.wait(20) == 0


@pytest.mark.slow
def test_code_change_restarts_the_worker_and_reruns(client, project, stages_file):
    full_run(client, project)
    pid = client.request("ping")[0]["pid"]
    before = root(project, "smooth")
    src = stages_file.read_text()
    assert src.count("* params.strength\n") == 1
    time.sleep(3.2)  # past the worker's stat reuse window (afv.serve.STAT_TTL_S)
    stages_file.write_text(src.replace("* params.strength\n", "* params.strength * 2\n"))
    r = client.send("run", {**where(project), "to": "smooth"})
    q = client.send("ping")  # queued behind nothing (answered at once) or by the restarted worker
    evs = client.wait(r, 60)
    assert [e for e in evs if e.get("kind") == "restart"], evs
    res = evs[-1]
    assert res["event"] == "result" and res["outcome"] == "ok", res
    assert {s["name"]: s["state"] for s in res["stages"]}["smooth"] == "ran"
    assert np.allclose(root(project, "smooth"), before * 2)
    client.wait(q)
    info = client.request("ping")[0]
    assert info["pid"] == pid and info["restarts"] == 1
    assert sum(e["event"] in ("result", "error") for e in client.by_id[r]) == 1
    assert stage_dir(project.shot("001").dir, "smooth").is_dir()


def test_lax_edit_file_stops_run_before_settings_are_saved(client, project):
    full_run(client, project)
    sh = project.shot("001")
    sh.edit_path("pins").write_text(json.dumps({"pins": [{"role": "A", "frame": "3", "preset": "p",
                                                          "root": [0, 0, 0]}]}))  # as a hand edit leaves it
    before = sh.info.settings.params
    res, _ = client.request("run", {**where(project), "to": "smooth", "sets": {"smooth.strength": 0.9}})
    assert res["event"] == "error", res
    assert "edits/pins.json: pins[0].frame: must be an integer >= 0, got '3'" in res["message"]
    assert "fix or delete that entry in edits/pins.json" in res["message"]
    assert sh.info.settings.params == before  # nothing was saved
    sh.edit_path("pins").unlink()
    res, _ = client.request("run", {**where(project), "to": "smooth", "sets": {"smooth.strength": 0.9}})
    assert res["event"] == "result" and sh.info.settings.params["smooth"] == {"strength": 0.9}
