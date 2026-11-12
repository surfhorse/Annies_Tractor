"""``afv serve`` regression tests from the round-4 review: malformed lines, cancel races, range chains,
concurrent restarts, path spellings, stage logs."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from afv.core.project import Project, stage_dir
from afv.serve import _json_default

REPO = Path(__file__).resolve().parents[2]
DRVFS = REPO / "tests" / "_out"  # on the Windows drive when the repo is (case-insensitive paths)


def where(project, shot="001"):
    return {"project": str(project.root), "shot": shot}


def full_run(client, project, shot="001"):
    res, _ = client.request("run", {**where(project, shot), "to": "smooth"})
    assert res["outcome"] == "ok", res
    return res


def root(project, stage, shot="001"):
    return project.shot(shot).load_result(stage).actors["A"].root


def wait_for(cond, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if v := cond():
            return v
        time.sleep(0.05)
    raise AssertionError("timed out")


def terminal(client, rid, timeout=30.0):
    return client.wait(rid, timeout)[-1]


@pytest.mark.parametrize(
    "line",
    [
        json.dumps({"id": [1], "cmd": "ping"}),
        json.dumps({"id": 7, "cmd": "cancel", "args": {"target_id": {"a": 1}}}),
        '{"id": 3, "cmd": "ping", "args": {"x": ' + "[" * 100000 + "]" * 100000 + "}}",
        json.dumps({"id": 4, "cmd": "status", "args": {"project": "/tmp/a\u0000b", "shot": "1"}}),
        '{"id": 5, "cmd": "run", "args": {"project": "/x", "shot": "1", "sets": {"smooth.strength": NaN}}}',
        json.dumps({"id": True, "cmd": "ping"}),
    ],
    ids=["list-id", "dict-target", "deep", "nul", "nan", "bool-id"],
)
def test_malformed_lines_get_an_error_and_the_worker_lives(client, project, line):
    client.write(line.encode() + b"\n")
    rid = json.loads(line)["id"] if '"id": 7' in line or '"id": 4' in line else None
    if '"id": 5' in line:
        rid = None  # NaN: not valid JSON for the worker, so no id can be trusted
    err = terminal(client, rid, 10)
    assert err["event"] == "error"
    res, _ = client.request("ping", timeout=10)
    assert res["pong"]


def test_overlong_line_and_nan_in_sets(client, project):
    client.write(b'{"id": 1, "cmd": "ping", "args": {"pad": "' + b"x" * (5 << 20) + b'"}}\n')
    assert "longer than" in terminal(client, None)["message"]
    res, _ = client.request("ping", timeout=10)
    assert res["pong"]
    res, _ = client.request("run", {**where(project), "to": "smooth", "sets": ["smooth.strength=1e999"]})
    assert res["event"] == "error" and "finite" in res["message"]
    args = json.dumps({**where(project), "name": "contacts", "data": {"spans": [], "x": 0}})
    line = '{"id": "e9", "cmd": "edit", "args": ' + args.replace('"x": 0', '"x": 1e999') + "}\n"
    client.write(line.encode())
    err = terminal(client, "e9")  # 1e999 parses as infinity: the edit refuses it
    assert err["event"] == "error" and "infinity" in err["message"]


def test_json_default_makes_numbers():
    assert json.loads(json.dumps({"a": np.float32(0.5), "b": np.int64(3), "c": np.arange(2)},
                                 default=_json_default)) == {"a": 0.5, "b": 3, "c": [0, 1]}  # fmt: skip


def test_cancel_before_the_child_exists(client, project):
    full_run(client, project)
    r1 = client.send("run", {**where(project), "from": "contacts", "to": "smooth",
                             "sets": ["contacts.pre_s=1.0", "contacts.child_s=30"]})  # fmt: skip
    wait_for(lambda: any(e.get("kind") == "stage_start" and e.get("stage") == "contacts"
                         for e in client.by_id.get(r1, [])))  # fmt: skip
    t0 = time.monotonic()
    client.request("cancel", {"target_id": r1})
    err = terminal(client, r1, 20)
    assert err["event"] == "error" and err["cancelled"]
    assert time.monotonic() - t0 < 8  # the child was stopped as it started, not waited for (30 s)


def test_range_chain_with_a_lagging_downstream(client, project):
    full_run(client, project)
    res, _ = client.request("run", {**where(project), "from": "contacts", "to": "contacts",
                                    "sets": {"contacts.offset": 5.0}})  # fmt: skip
    assert res["outcome"] == "ok"
    res, _ = client.request("run", {**where(project), "from": "contacts", "to": "smooth", "frames": "10:20"})
    st = {s["name"]: s for s in res["stages"]}
    assert st["contacts"]["range"] == [10, 20]
    assert st["smooth"]["range"] is None and "upstream contacts changed outside" in st["smooth"]["reason"]
    assert np.allclose(root(project, "smooth"), root(project, "contacts") * 0.5)


def test_stage_logs_of_parallel_shots_stay_apart(client, project):
    r1 = client.send("run", {**where(project, "001"), "to": "smooth", "sets": ["contacts.hold_s=1.0"]})
    r2 = client.send("run", {**where(project, "002"), "to": "smooth", "sets": ["contacts.hold_s=1.0"]})
    assert terminal(client, r1)["outcome"] == "ok" and terminal(client, r2)["outcome"] == "ok"
    for sid in ("001", "002"):
        text = (stage_dir(project.shot(sid).dir, "contacts") / "logs" / "stage.log").read_text()
        assert text.count("stage contacts/default start") == 1, text


@pytest.mark.slow
def test_ping_and_cancel_during_a_pending_restart(client, project, stages_file):
    full_run(client, project)
    full_run(client, project, "002")
    r2 = client.send("run", {**where(project, "002"), "from": "contacts", "to": "contacts",
                             "sets": ["contacts.hold_s=20"]})  # fmt: skip
    wait_for(lambda: any(e["event"] == "progress" for e in client.by_id.get(r2, [])))
    time.sleep(3.2)  # past the worker's stat reuse window
    src = stages_file.read_text()
    stages_file.write_text(src.replace("* params.strength\n", "* params.strength * 2\n"))
    r1 = client.send("run", {**where(project, "001"), "to": "smooth"})
    wait_for(lambda: any(e.get("kind") == "restart" for e in client.by_id.get(r1, [])), 20)
    t0 = time.monotonic()
    res, _ = client.request("ping", timeout=5)
    assert res["pong"] and time.monotonic() - t0 < 2  # answered while the restart waits
    res, _ = client.request("cancel", {"target_id": r2}, timeout=5)
    assert res["state"] == "running"
    assert terminal(client, r2)["cancelled"]
    assert terminal(client, r1, 60)["outcome"] == "ok"
    res, _ = client.request("cancel", {"target_id": r2})
    assert res["state"] == "finished"  # an id answered before the restart is known after it


@pytest.mark.slow
def test_two_shots_hit_a_code_change_together(client, project, stages_file):
    full_run(client, project)
    full_run(client, project, "002")
    time.sleep(3.2)
    src = stages_file.read_text()
    stages_file.write_text(src.replace("* params.strength\n", "* params.strength * 2\n"))
    r1 = client.send("run", {**where(project, "001"), "to": "smooth"})
    r2 = client.send("run", {**where(project, "002"), "to": "smooth"})
    for r in (r1, r2):
        evs = client.wait(r, 60)
        assert evs[-1]["event"] == "result" and evs[-1]["outcome"] == "ok", evs[-1]
        assert sum(e["event"] in ("result", "error") for e in evs) == 1
    assert client.request("ping")[0]["restarts"] == 1


@pytest.fixture
def drvfs_project():
    if not str(DRVFS).startswith("/mnt/"):
        pytest.skip("the repo is not on a Windows drive")
    d = DRVFS / f"serve-{uuid.uuid4().hex[:8]}"
    (d / "src").mkdir(parents=True)
    v = d / "src" / "fight.mp4"
    v.write_bytes(np.random.default_rng(0).bytes(4096))
    proj = Project.create(d / "Fight.afv", v)
    proj.add_shot(0.0, 2.0)
    yield proj
    shutil.rmtree(d, ignore_errors=True)


def test_two_spellings_of_one_shot_share_a_queue(client, drvfs_project):
    p1 = str(drvfs_project.root)
    p2 = p1[:7] + p1[7:].upper()  # /mnt/d/WORKSPACE/...: the same folder on a case-insensitive drive
    assert Path(p2).is_dir()
    res, _ = client.request("run", {"project": p1, "shot": "001", "to": "smooth"})
    assert res["outcome"] == "ok", res
    r1 = client.send("run", {"project": p1, "shot": "001", "from": "contacts", "to": "smooth",
                             "sets": ["contacts.hold_s=2"]})  # fmt: skip
    r2 = client.send("run", {"project": p2, "shot": "001", "from": "contacts", "to": "smooth"})
    e1, e2 = terminal(client, r1), terminal(client, r2)
    assert e1["outcome"] == "ok" and e2["outcome"] == "ok", (e1, e2)  # queued, not "already being run"

    def done_at(rid):
        return next(i for i, ln in enumerate(client.lines) if (e := json.loads(ln))["id"] == rid
                    and e["event"] in ("result", "error"))  # fmt: skip

    assert done_at(r1) < done_at(r2)


def test_plumbing_snapshot_covers_the_worker_itself():
    from afv.serve import _plumbing_files

    files = _plumbing_files()
    assert any(f.endswith("afv/serve.py") for f in files) and any(f.endswith("core/runner.py") for f in files)


@pytest.mark.slow
def test_restart_into_code_that_does_not_import(client, project, stages_file):
    full_run(client, project)
    time.sleep(3.2)  # past the stat reuse window
    good = stages_file.read_text()
    stages_file.write_text(good + "\ndef broken(:\n")
    pid = client.request("ping")[0]["pid"]
    res = terminal(client, client.send("run", {**where(project), "to": "smooth"}), 120)
    assert res["event"] == "error" and res["code_error"]
    assert res["message"].startswith("afv code has an error")
    info = client.request("ping")[0]
    assert info["pid"] == pid and info["restarts"] == 0 and not info["running"]  # still serving
    stages_file.write_text(good.replace("* params.strength\n", "* params.strength * 2\n"))
    time.sleep(3.2)
    res = terminal(client, client.send("run", {**where(project), "to": "smooth"}), 120)
    assert res["event"] == "result" and res["outcome"] == "ok", res
    assert client.request("ping")[0]["restarts"] == 1


@pytest.mark.slow
def test_restart_into_broken_plumbing_keeps_serving(tmp_path, worker_env, project, new_client):
    import afv

    src = tmp_path / "srccopy"
    shutil.copytree(Path(afv.__file__).parent, src / "afv", ignore=shutil.ignore_patterns("__pycache__"))
    c = new_client({**worker_env, "PYTHONPATH": str(src)})
    assert c.request("run", {**where(project), "to": "smooth"})[0]["outcome"] == "ok"
    time.sleep(3.2)
    prov = src / "afv" / "core" / "provenance.py"
    good = prov.read_text()
    prov.write_text(good + "\ndef broken(:\n")
    res = c.wait(c.send("status", where(project)), 120)[-1]
    assert res["event"] == "error" and "afv code has an error" in res["message"]
    assert c.proc.poll() is None and c.request("ping")[0]["pong"]
    prov.write_text(good)
    time.sleep(3.2)
    res = c.wait(c.send("status", where(project)), 120)[-1]
    assert res["event"] == "result", res
