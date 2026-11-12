"""The shot browser (AFV > Shots, round 6 track B) in headless Blender against a real ``afv serve`` worker
with fake cut detection, a fake track worker and fake stages (``tests/serve/worker_main.py``; no GPU), on a
synthetic video. Blender side: ``browser_checks.py`` mode ``flow``.

Checked through the operators the panel's buttons call: New project from video (candidate shots, the
browser scene with the clip behind its camera, the source's fps and frames; a second time refused); select
and jump (preview range = the shot); Set in / Set out from the playhead (afv's half-frame convention) and a
refused out-before-in; speed and notes; Queue / Skip; the filter; Add shot here; Run queue refused while
queued shots have no pick (listed); Scan (rows and thumbnails); Pick fighters from the scan (the exact
frame image shown, numbers chosen, saved, the next queued shot offered), a wrong number and a cancel;
Run queue to the end with per-shot outcomes; each edit's round trip under 1 s; the panels' draw code; and
listing, Set in, Queue, Add shot and Pick again through the one-shot fallback (the real afv CLI)."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from take_helpers import REPO, TESTS_DIR

from afv.export import blender as bl

pytestmark = [
    pytest.mark.blender,
    pytest.mark.slow,
    pytest.mark.skipif(bl.find_blender() is None, reason="Blender not installed"),
]

CHECKS = TESTS_DIR / "browser_checks.py"
SEGMENTS = [{"start": 0.0, "duration": 2.0, "confidence": 1.0}, {"start": 2.0, "end": 4.0, "confidence": 0.7},
            {"start": 4.0, "duration": 2.0, "confidence": 0.5}]
FPS = 25


def run_check(mode: str, job: dict, work: Path, timeout: float = 900.0) -> dict:
    """Run ``browser_checks.py`` in headless Blender; its ``out``."""
    exe = bl.find_blender()
    work.mkdir(parents=True, exist_ok=True)
    result = work / f"{mode}_result.json"
    result.unlink(missing_ok=True)
    job_file = work / f"{mode}_job.json"
    job = {"mode": mode, "addon_parent": bl.host_path(REPO / "blender", exe),
           "tests_dir": bl.host_path(TESTS_DIR, exe), "result_path": bl.host_path(result, exe), **job}
    job_file.write_text(json.dumps(job, indent=1))
    cmd = [str(exe), "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1",
           "--python", bl.host_path(CHECKS, exe), "--", bl.host_path(job_file, exe)]
    proc = None
    for _ in range(2):  # WSL interop sometimes fails to launch a Windows process once
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=work)
        if result.is_file() or b"UtilAcceptVsock" not in proc.stderr + proc.stdout:
            break
    (work / f"{mode}_blender.log").write_bytes(proc.stdout + b"\n--- stderr ---\n" + proc.stderr)
    if not result.is_file():
        tail = (proc.stdout + proc.stderr)[-3000:].decode("utf-8", "replace")
        raise AssertionError(f"Blender wrote no result (exit {proc.returncode}):\n{tail}")
    res = json.loads(result.read_text(encoding="utf-8"))
    assert res["ok"], res.get("error")
    return res["out"]


def write_video(path: Path, seconds: float = 6.0) -> Path:
    """Three 2 s scenes of tinted noise (160x96, 25 fps)."""
    import av

    rng = np.random.default_rng(0)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=FPS, options={"crf": "20", "preset": "ultrafast"})
        s.width, s.height, s.pix_fmt = 160, 96, "yuv420p"
        for i in range(int(seconds * FPS)):
            tint = np.array([(200, 60, 60), (60, 200, 60), (60, 60, 200)][min(2, i // (2 * FPS))])
            img = np.clip(tint + rng.integers(-30, 30, (96, 160, 3)), 0, 255).astype(np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


def worker_argv(tmp: Path) -> list[str]:
    """wsl.exe running the fake-stage ``afv serve`` (as the add-on starts the real one)."""
    stages = tmp / "stages" / "fake_stages_copy.py"
    stages.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / "tests" / "serve" / "fake_stages.py", stages)
    env = {"AFV_TEST_STAGES": str(stages), "AFV_TEST_SEGMENTS": json.dumps(SEGMENTS),
           "AFV_TEST_FAKE_TRACK": "1",
           "AFV_CONFIG_DIR": str(tmp / "config"), "AFV_HOME": str(tmp / "home"), "PYTHONUNBUFFERED": "1"}
    cmd = ("echo AFVPID:$$; exec env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
           + f" {shlex.quote(sys.executable)} "
           + shlex.quote(str(REPO / "tests" / "serve" / "worker_main.py")))
    return ["wsl.exe", "-d", "Ubuntu", "-e", "bash", "-lc", cmd]


@pytest.fixture(scope="module")
def flow(tmp_path_factory):
    exe = bl.find_blender()
    tmp = tmp_path_factory.mktemp("browser")
    video = write_video(tmp / "bout.mp4")
    project = tmp / "bout.afv"
    afv_cli = Path(sys.executable).with_name("afv")
    job = {"video": bl.host_path(video, exe), "project": bl.host_path(project, exe),
           "worker": worker_argv(tmp),
           "oneshot": str(afv_cli)}
    return {"out": run_check("flow", job, tmp / "work"), "project": project}


def test_new_project_makes_the_browser_scene(flow):
    o = flow["out"]
    assert o["new_call"][0] == ["FINISHED"]
    new = o["new"]
    assert sorted(new["rows"]) == ["001", "002", "003"]
    assert {r["state"] for r in new["rows"].values()} == {"candidate"}
    assert new["frames"] == [0, 6 * FPS - 1] and new["fps"] == pytest.approx(FPS)
    assert new["res"] == [160, 96] and new["clip"] == ["bout.mp4"] and new["camera"] == "AFV Shots camera"
    assert "already an afv project" in o["new_again"][1]
    assert o["draw_empty"][:2] == [["operator", "afv.browser_new", ""], ["operator", "afv.browser_open", ""]]


def test_select_jumps_and_sets_the_preview_range(flow):
    sel = flow["out"]["select"]
    assert sel["frame"] == 50 and sel["preview"] == [True, 50, 99] and sel["speed"] == 1.0


def test_set_in_and_out_from_the_playhead(flow):
    o = flow["out"]
    assert o["set_in"]["frame"] == 55
    assert o["set_in"]["start"] == pytest.approx((55 - 0.5) / FPS) and o["set_in"]["end"] == 4.0
    assert o["set_in"]["preview"] == [55, 99]
    assert o["set_out"]["frame"] == 65 and o["set_out"]["end"] == pytest.approx((65 + 0.5) / FPS)
    assert o["set_out"]["preview"] == [55, 65]
    bad = o["set_bad"]  # out before in: afv refuses, the shot is unchanged, the message is shown
    assert bad["status"]["level"] == "error" and "002" in bad["status"]["text"]
    assert bad["end"] == pytest.approx((65 + 0.5) / FPS)
    assert o["fields"] == {"speed": 0.5, "notes": "right hand"}


def test_queue_skip_filter_and_add(flow):
    o = flow["out"]
    assert o["states"] == {"001": "queued", "002": "candidate", "003": "skipped"}
    assert o["counts"] == {"candidate": 1, "queued": 1, "skipped": 1}
    assert o["filter_queued"] == ["001"]
    add = o["add"]
    assert add["selected"] == "004" and add["row"]["state"] == "queued" and add["row"]["origin"] == "manual"
    assert add["row"]["start"] == pytest.approx((10 - 0.5) / FPS)
    assert add["row"]["end"] == pytest.approx((10 + 5 * FPS - 1 + 0.5) / FPS)  # no cut ahead: 5 s


def test_round_trips_under_a_second(flow):
    t = flow["out"]["times"]
    for k in ("set_in", "set_out", "queue", "skip", "add"):
        assert t[k] < 1.0, (k, t[k])
    assert t["speed_notes"] < 2.0
    # dragging the Speed slider sends nothing; once it rests, one shot_set with the last value
    assert flow["out"]["speed_debounce"] == {"while_dragging": 0, "after": ["shot_set"]}
    bt = flow["out"]["browser_times"]
    assert bt["pick_prompt"] < 1.0 and bt["pick_save"] < 1.0


def test_run_queue_needs_picks_first(flow):
    r = flow["out"]["run_unpicked"]
    assert r["result"] == ["FINISHED"] and not r["batch"]
    assert [m[0] for m in r["missing"]] == ["001", "004"]
    labels = [x[1] for x in r["draw"] if x[0] == "label"]
    assert any("2 queued shots need their fighters picked" in x for x in labels)
    ops = [x for x in r["draw"] if x[0] == "operator" and x[1] == "afv.browser_select"]
    assert any(x[2].startswith("001: fighters not picked") for x in ops)


def test_scan_fills_rows_and_thumbnails(flow):
    o = flow["out"]
    rows = o["scan"]["rows"]
    for sid in ("001", "002", "004"):
        assert rows[sid]["scan"] == "ok" and rows[sid]["people"] == 2 and rows[sid]["thumb"]
    assert rows["003"]["scan"] == "none"  # skipped shots are not scanned
    assert o["thumb_files"] == {"001": True, "002": True, "004": True}


def test_pick_from_the_scan(flow):
    o = flow["out"]
    p = o["pick1"]
    assert p["result"] == ["FINISHED"] and p["boxes"] == 2 and p["source"] == "scan" and p["frame"] == 15
    assert p["image"] == "001.frame.jpg" and p["size"] == [160, 96]  # the exact scanned frame
    assert p["backgrounds"] == ["IMAGE"]  # shown instead of the video while picking
    assert p["roles"] == {"1": "A", "0": "B"} and p["picked"] is True
    assert p["shown_s"] < 1.0 and p["save_s"] < 1.0
    assert any(x[0] == "operator" and x[1] == "afv.browser_pick_save" for x in p["draw"])
    assert o["pick_next_started"] == {"active": True, "shot": "004"}  # Then pick the next
    assert o["pick2"]["picked"] is True
    assert "no person #7" in o["pick_bad_box"][1]
    c = o["pick_cancel"]
    assert c["active"] is False and c["backgrounds"] == ["MOVIE_CLIP"] and c["picked"] is False


def test_run_queue_runs_the_picked_shots(flow):
    o = flow["out"]
    assert o["run_call"][0] == ["FINISHED"]
    assert [(x["shot"], x["outcome"]) for x in o["run"]["outcomes"]] == [("001", "ok"), ("004", "ok")]
    assert o["run"]["summary"].startswith("last batch ") and o["run"]["summary"].endswith("2 ran ok")
    assert o["times"]["batch_start"] < 5.0
    # every queued shot, by state (afv reads the queue again after each shot): no fixed shot list
    assert o["run_args"]["state"] == "queued" and "shots" not in o["run_args"]
    assert any(x[0] == "progress" for x in o["run_draw"])  # per-shot progress while it runs
    assert not flow["project"].joinpath("shots", "002", "stages", "ingest").exists()  # not queued: not run
    done = [x for x in o["draw_end"] if x[0] == "operator" and x[1] == "afv.browser_select"]
    assert any(x[2] == "001: ran ok " for x in done)


def test_a_trimmed_shot_is_scanned_in_one_job_before_its_pick(flow):
    o = flow["out"]
    assert o["trimmed"] == {"scan": "stale", "pick_stale": True, "needs_scan": ["001"]}
    w = o["stale_pick_wait"]
    assert w["waiting"] and "scanning this shot first" in w["text"] and "left" in w["help"]
    sp = o["stale_pick"]
    assert sp["source"] == "scan" and sp["boxes"] == 2
    assert sp["requests"][:2] == [["shots_scan", ["001"], None], ["pick", None, None]]  # no detector per shot


def test_a_refused_save_keeps_the_clicks(flow):
    r = flow["out"]["refused_save"]
    assert r["active"] and r["selected"] == [0, 6] and not r["saving"] and not r["picked"]
    assert r["status"]["level"] == "error" and "not saved" in r["status"]["text"]
    assert "no detection [6]" in r["status"]["text"]
    assert flow["out"]["fixed_save"] == {"active": False, "picked": True, "stale": False}


def test_keys(flow):
    k = flow["out"]["keys"]
    assert k["poll"] and k["next"] == "003" and k["prev"] == "002"
    assert k["skipped"] == "skipped" and k["after_skip_selected"] == "003"  # Skip, then the next shot
    assert k["next_likely"] == "001"
    assert k["queue_pick"] == {"state": "queued", "picking": "002"}


def test_import_take_without_results_says_so(flow):
    res, _err, status, scenes = flow["out"]["take_missing"]
    assert res == ["FINISHED"] and "no finished motion yet" in status and "afv " not in status
    assert scenes == ["AFV Shots", "Scene"]  # nothing made for a take that could not be written


def test_closing_the_project_frees_its_thumbnails(flow):
    pv = flow["out"]["previews"]
    assert pv["before"] > 0 and pv["after"] == 0 and pv["jobs"] == 0 and not pv["session"]


def test_one_shot_fallback(flow):
    fb = flow["out"]["fallback"]
    assert "one-shot" in fb["worker"]
    assert fb["open"][0] == ["FINISHED"] and "4 shots" in fb["open"][2]  # Open project needs no batch_status
    assert fb["rows"] == 4 and fb["state"] == "queued" and fb["rows_after_add"] == 5
    assert fb["set_in_start"] > 4.0
    assert fb["pick"]["source"] == "scan" or fb["pick"]["boxes"] == 2
    assert fb["pick"]["picked"] is True


def _row(sid, start, state="queued", picked=True, **kw):
    return {"id": sid, "start": start, "duration": 2.0, "end": start + 2.0, "state": state, "picked": picked,
            "pick_stale": False, "speed_factor": 1.0, "notes": "", "confidence": 0.5,
            "scan": {"status": "none"}, **kw}


def _stub_cfg(tmp: Path, exe, kind: str) -> tuple[dict, Path]:
    proj = tmp / "p.afv"
    proj.mkdir()
    (proj / "project.json").write_text("{}")
    shots = {"project": "/x/p.afv", "root_win": bl.host_path(proj, exe), "proxy": None, "cuts": None,
             "scan_dir": "media/scan", "source": {"fps": 30.0, "width": 160, "height": 96, "num_frames": 300,
                                                  "duration": 10.0, "path_win": None, "path": "/x/gone.mp4"},
             "shots": [_row("001", 0.0), _row("002", 2.0), _row("004", 4.0, picked=False),
                       {"id": "003", "state": None, "error": "shot.json: not valid JSON"}]}
    running = {"running": True, "batch_id": "b1", "pid": 7, "started": "2026-11-11T20:00:00", "log":
               "media/batch/b1.jsonl", "log_path": "/x/p.afv/media/batch/b1.jsonl",
               "current": {"shot": "002", "index": 1, "stage": "pose3d", "frac": 0.5, "message": ""},
               "outcomes": [{"shot": "001", "outcome": "ok", "message": "", "exports": []}], "done": None,
               "finished": None, "interrupted": False}
    after = {**running, "running": False, "current": None, "finished": "2026-11-11T20:05:00",
             "outcomes": [{"shot": "001", "outcome": "ok", "message": "", "exports": []},
                          {"shot": "002", "outcome": "cancelled", "message": "", "exports": []}],
             "done": {"batch_id": "b1", "cancelled": True, "ok": 1}}
    cfg = {"log": bl.host_path(tmp / "requests.jsonl", exe), "shots": shots, "after_cancel": after,
           "statuses": [running, {**running, "current": {"shot": "002", "index": 1, "stage": "smooth"}}]}
    if kind == "stream":  # what serve's batch streams when it attaches (its log from the start)
        cfg["stream"] = [
            {"event": "log", "kind": "batch_start", "batch_id": "b1", "to": "export", "states": ["queued"],
             "shots": ["001", "002"]},
            {"event": "log", "kind": "batch_shot", "shot": "001", "index": 0},
            {"event": "progress", "shot": "001", "stage": "camera", "frac": 0.4},
            {"event": "log", "kind": "shot_done", "shot": "001", "outcome": "ok", "message": "",
             "exports": []},
            {"event": "log", "kind": "batch_shot", "shot": "002", "index": 1},
            {"event": "progress", "shot": "002", "stage": "pose3d", "frac": 0.5}]
    if kind == "finished":  # last night's batch, and a take afv writes on request
        last = {**after, "outcomes": [{"shot": "001", "outcome": "ok", "message": "", "exports": []},
                                      {"shot": "002", "outcome": "needs_review", "message": "hints",
                                       "exports": []}],
                "done": {"batch_id": "b0", "cancelled": False, "ok": 1, "needs_review": 1}, "batch_id": "b0"}
        cfg["statuses"] = [last]
        cfg["take"] = {"src": bl.host_path(synthetic_take(tmp / "take_src"), exe),
                       "root_win": bl.host_path(proj, exe)}
    (tmp / "cfg.json").write_text(json.dumps(cfg))
    job = {"project": bl.host_path(proj, exe), "cfg": bl.host_path(tmp / "cfg.json", exe), "log": cfg["log"],
           "kind": kind}
    return job, tmp


def synthetic_take(out: Path) -> Path:
    """A take package (two synthetic fighters) for Import take."""
    from take_helpers import synthetic_shot

    from afv import take as tk

    bm, cam = synthetic_shot(n_frames=12, seed=3)
    tk.write_take(out, tk.TakeInputs(motion=bm, camera=cam, shot_id="001", project_dir=str(out / "proj")))
    return out


def _stub(tmp_path_factory, kind):
    exe = bl.find_blender()
    job, tmp = _stub_cfg(tmp_path_factory.mktemp(f"browser_{kind}"), exe, kind)
    return run_check("detached", job, tmp / "work")


@pytest.fixture(scope="module")
def detached(tmp_path_factory):
    """A batch started earlier still runs in WSL; serve attaches to it and streams it."""
    return _stub(tmp_path_factory, "stream")


@pytest.fixture(scope="module")
def polled(tmp_path_factory):
    """The same with an afv whose batch cannot attach (unknown command): batch_status every few seconds."""
    return _stub(tmp_path_factory, "poll")


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """Last night's batch has finished; a take is written on request."""
    return _stub(tmp_path_factory, "finished")


def test_open_project_attaches_to_a_running_batch(detached):
    o = detached["open"]
    assert o["streaming"] and not o["detached"] and o["running"] and not o["run_poll"] and o["stop_poll"]
    assert o["batch"]["id"] == "b1" and o["batch"]["log"] == "media/batch/b1.jsonl"
    assert o["current"]["shot"] == "002" and o["current"]["stage"] == "pose3d" and o["current"]["total"] >= 2
    assert o["line"].startswith("Run queue (attached): shot 2/") and "pose3d 50%" in o["line"]
    assert [(x["shot"], x["outcome"]) for x in o["outcomes"]] == [("001", "ok")]
    labels = [x[1] for x in detached["draw_running"] if x[0] == "label"]
    assert any("closing Blender does not stop" in x for x in labels)
    assert any("Shots you queue and pick now are run by this batch" in x for x in labels)
    assert any(x[0] == "progress" for x in detached["draw_running"])


def test_a_new_file_or_disabling_the_add_on_never_stops_the_batch(detached):
    a = detached["after_new_file"]
    assert a["jobs"] == [] and not a["session"]
    assert "cancel" not in a["requests"] and "batch_cancel" not in a["requests"]
    assert detached["reattached"]["streaming"]  # Open project attaches again
    req = detached["requests"]
    assert req.count("batch_cancel") == 1  # only Stop batch's
    assert "cancel" not in req and req[-1] == "shutdown"  # disabling: the worker just shuts down


def test_stop_batch_says_what_ran(detached):
    st = detached["stopped"]
    assert not st["streaming"] and not st["running"]
    assert [(x["shot"], x["outcome"]) for x in st["outcomes"]] == [("001", "ok"), ("002", "cancelled")]
    assert st["summary"] == ("batch b1 stopped: 2 shots ran (1 ran ok, 1 stopped); 1 queued shot did not run "
                             "(they stay queued)")
    assert "stopping" not in st["status"]


def test_a_batch_that_cannot_be_attached_is_polled(polled):
    o = polled["open"]
    assert o["detached"] and not o["streaming"] and not o["run_poll"] and o["stop_poll"]
    assert o["batch"]["running"] and o["batch"]["id"] == "b1" and o["current"]["shot"] == "002"
    labels = [x[1] for x in polled["draw_running"] if x[0] == "label"]
    assert any(x.startswith("Batch b1 running in WSL: shot 002") for x in labels)
    assert any(x[:2] == ["operator", "afv.browser_stop_batch"] for x in polled["draw_running"])
    assert "smooth" in polled["polled"]["summary"]
    st = polled["stopped"]
    assert not st["detached"] and "stopped" in st["summary"]
    assert [(x["shot"], x["outcome"]) for x in st["outcomes"]] == [("001", "ok"), ("002", "cancelled")]
    req = polled["requests"]
    assert req.count("batch") == 1 and "batch_cancel" in req  # one attach attempt, then polling


def test_last_nights_results_on_open(finished):
    o = finished["open"]
    assert not o["running"] and o["run_poll"]
    assert o["summary"] == "last batch b0, finished 11-11 20:05: 1 ran ok, 1 needs a tracking review"
    assert o["row_outcomes"] == {"001": "ok", "002": "needs_review"}
    labels = [x[1] for x in finished["finished_draw"] if x[0] == "label"]
    assert any("last run: needs a tracking review" in x for x in labels)
    ops_ = [x[2] for x in finished["finished_draw"] if x[0] == "operator" and x[1] == "afv.browser_select"]
    assert any(x.startswith("002: needs a tracking review") for x in ops_)


def test_import_take_lands_in_its_own_scene_once(finished):
    t1, t2 = finished["take"]
    assert t1["result"] == ["FINISHED"] and t1["scene"] == "AFV 001" and len(t1["armatures"]) == 2
    assert t2["scene"] == "AFV 001" and t2["scenes"] == t1["scenes"]
    assert "Elsewhere" in t1["scenes"]  # the same scene, no "AFV 001.001"
    assert not any(n.startswith("001_") for n in finished["browser_scene_objects"])
    r = finished["take_ready"]  # written while the owner was in another scene: offered, not switched to
    assert r["ready"] == {"shot": "001", "scene": "AFV 001"} and r["status"].endswith("is ready: Show take")
    assert r["show_poll"] and r["after_show"] == {}


def test_missing_video_and_unreadable_rows(detached, finished):
    v = finished["video"]
    assert "video not found" in v["warning"] and v["clip"] is None and v["backgrounds"] == 0
    labels = [x[1] for x in finished["draw_running"] if x[0] == "label"]
    assert any(x.startswith("video not found") for x in labels)
    b = detached["bad_row"]
    assert "cannot read this shot" in b["line"] and b["selected"] == "003"
    assert b["missing"] == [["004", "fighters not picked"]] and None not in b["counts"]
    assert any("cannot read this shot" in str(x[2]) for x in detached["draw_end"] if x[0] == "operator")
    assert detached["pick_poll_unselected"]  # Run queue's pick arrows work without a selected shot
