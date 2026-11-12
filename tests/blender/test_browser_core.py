"""The shot browser's plain-Python part (``blender/afv_addon/browser_core.py``, no Blender): frames <-> shot
times (as ``afv shots detect`` writes them), the list's filter and order (fighters-likely sorts, never
hides), the queue check, Add shot here's range, cut markers, the one-shot fallback's command lines and
results, and the pick boxes' hit test (``takeio``, shared with Swap IDs)."""

from __future__ import annotations

import json

import pytest
from take_helpers import addon_module

core = addon_module("browser_core")
takeio = addon_module("takeio")
FPS = 30000 / 1001


def row(sid, start, dur, state="candidate", likely=None, people=None, picked=False, stale=False, scan="ok"):
    return {"id": sid, "start": start, "duration": dur, "end": start + dur, "state": state, "picked": picked,
            "pick_stale": stale, "confidence": 0.5, "notes": "", "speed_factor": 1.0,
            "scan": {"status": scan, "people": people, "fighters_likely": likely, "thumbnail": None}}


@pytest.mark.parametrize("typed, want", [
    ("", "/mnt/d/ws/projects/Full Fight"),  # empty: <projects folder>/<video name>
    ("full_fight_test", "/mnt/d/ws/projects/full_fight_test"),  # a new name (the owner's case)
    ("  full_fight_test\\ ", "/mnt/d/ws/projects/full_fight_test"),
    ('"full_fight_test"', "/mnt/d/ws/projects/full_fight_test"),
    ("tests\\ff2", "/mnt/d/ws/projects/tests/ff2"),
    ("//ff3", "/mnt/d/ws/projects/ff3"),  # Blender-relative from an unsaved file
    (r"E:\fights\ff4", "/mnt/e/fights/ff4"),  # a full path as given
    (r"\\wsl.localhost\Ubuntu\home\u\ff5", "/home/u/ff5"),
    ("/mnt/d/elsewhere/ff6", "/mnt/d/elsewhere/ff6"),
])
def test_new_project_folder_accepts_a_plain_name(typed, want):
    wsl = addon_module("wsl")
    got = core.project_folder(typed, r"D:\ws\projects", "Full Fight")
    assert wsl.to_wsl_path(got) == want  # every result maps to WSL (no "cannot map" error)


def test_frames_and_times_match_detected_shots():
    # afv shots detect: a segment starting at frame 529 starts at (529 - 0.5) / fps
    assert core.start_time(529, FPS) == pytest.approx(17.634283, abs=1e-6)
    assert core.first_frame(17.634283, FPS) == 529
    assert core.first_frame(0.0, FPS) == 0 and core.start_time(0, FPS) == 0.0
    for f in (0, 1, 17, 529, 29405):
        assert core.first_frame(core.start_time(f, FPS), FPS) == f
        assert core.last_frame(core.end_time(f, FPS), FPS) == f
        assert core.first_frame(f / FPS, FPS) == f  # a start exactly on a frame keeps it
    # shot 001 of Full Fight: 0 .. 0.583933 s holds frames 0..17
    assert core.shot_frames({"start": 0.0, "end": 0.583933}, FPS) == (0, 17)
    assert core.shot_frames({"start": 5.0, "duration": 0.0}, FPS)[1] >= core.shot_frames(
        {"start": 5.0, "duration": 0.0}, FPS)[0]
    # the scan / prompt frame (30 Hz shot timeline) -> the source frame afv's normaliser takes for it: the
    # shot's first source frame + round(i * fps / 30) (not round((start + i / 30) * fps): one frame early)
    assert core.prompt_source_frame({"start": 17.634283}, 15, FPS) == 529 + 15
    assert core.prompt_source_frame({"start": 17.634283}, 0, FPS) == 529
    assert core.prompt_source_frame({"start": 0.0}, 300, FPS) == round(300 * FPS / 30)
    # Pick on this frame: the playhead's source frame -> the shot-timeline frame (clamped into the shot)
    r = {"start": 17.634283, "duration": 2.0, "end": 19.634283}
    assert core.shot_frame_at(r, 529, FPS) == 0 and core.shot_frame_at(r, 544, FPS) == 15
    assert core.shot_frame_at(r, 100, FPS) == 0 and core.shot_frame_at(r, 10**6, FPS) == 59
    for i in (0, 7, 15, 31, 59):
        assert core.shot_frame_at(r, core.prompt_source_frame(r, i, FPS), FPS) == i


def test_time_text():
    assert core.fmt_time(0) == "0:00.0" and core.fmt_time(62.5) == "1:02.5"
    assert core.fmt_time(59.97) == "1:00.0"
    assert core.fmt_time(3725.2) == "1:02:05.2" and core.fmt_time(None) == "-"
    assert core.fmt_len(6.54) == "6.5 s"


def test_likely_sorts_and_never_hides():
    rows = [row("001", 0, 2, likely=False, people=1), row("002", 2, 2, likely=True, people=3),
            row("003", 4, 2, state="queued", likely=None, scan="none"),
            row("004", 6, 2, likely=True, people=2),
            row("005", 8, 2, state="skipped", likely=False, people=0)]
    by_time = [r["id"] for r in core.visible(rows, "all", "time")]
    by_likely = [r["id"] for r in core.visible(rows, "all", "likely")]
    assert by_time == ["001", "002", "003", "004", "005"]
    assert by_likely == ["002", "004", "001", "003", "005"]  # likely first, all still there
    assert [r["id"] for r in core.visible(rows, "candidate", "likely")] == ["002", "004", "001"]
    assert [r["id"] for r in core.visible(rows, "queued")] == ["003"]
    assert core.counts(rows) == {"candidate": 3, "queued": 1, "skipped": 1}
    nav = [r["id"] for r in core.visible(rows, "all", "likely")]
    assert core.next_in_order(nav, "002") == "004" and core.next_in_order(nav, "002", -1) is None
    assert core.next_in_order(nav, "005") is None and core.next_in_order(nav, None) == "002"
    assert core.next_in_order([], "1") is None
    line = core.row_line(rows[1])
    assert line == "002  0:02.0  2.0 s  50%  *"
    assert core.row_detail(rows[1]) == "candidate, 3 people, fighters likely"
    assert core.row_detail({**rows[0], "picked": True, "notes": "jab"}) == "candidate, 1 person, picked, jab"
    assert core.people_text(rows[2]) == "not scanned" and "fighters likely" in core.people_text(rows[1])


def test_queue_check_and_next_pick():
    rows = [row("001", 0, 2, state="queued", picked=True), row("002", 2, 2, state="queued"),
            row("003", 4, 2, state="queued", picked=True, stale=True), row("004", 6, 2),
            row("005", 8, 2, state="queued")]
    ready, missing = core.queue_check(rows)
    assert ready == ["001"]
    assert [m[0] for m in missing] == ["002", "003", "005"] and "again" in missing[1][1]
    assert core.next_to_pick(rows) == "002" and core.next_to_pick(rows, "002") == "003"
    assert core.next_to_pick(rows, "005") == "002"  # wraps around
    assert core.next_to_pick([row("1", 0, 1)]) is None


def test_add_range():
    fps = 30.0
    # to the next cut when it is 0.5..10 s away
    s, e = core.add_range(100, fps, [50, 220, 400], 1000)
    assert core.first_frame(s, fps) == 100 and core.last_frame(e, fps) == 219
    # a cut too close or too far: 5 s
    s, e = core.add_range(100, fps, [105, 900], 1000)
    assert core.last_frame(e, fps) == 249
    # never past the source's last frame
    s, e = core.add_range(990, fps, [], 1000)
    assert core.last_frame(e, fps) == 999


def test_cut_markers():
    cuts = {"cuts": [{"frame": 30, "start_frame": 18, "confidence": 0.2763, "kind": "gradual"},
                     {"frame": 529, "confidence": 0.4609, "kind": "hard"}, {"frame": "x"}]}
    # at the first frame of the segment after the cut: for a fade where it ends (frame), not where it
    # begins (start_frame)
    assert core.cut_frames(cuts) == [(30, "fade 28%"), (529, "cut 46%")]
    assert core.cut_frames(None) == []


def test_rows_afv_cannot_read_and_batch_status():
    bad = {"id": "009", "state": None, "error": "shot.json: not valid JSON"}
    rows = [row("001", 3, 2, state="queued", picked=True), row("002", 5, 2), bad]
    assert [r["id"] for r in core.visible(rows)] == ["009", "001", "002"]  # shown (first: no start)
    for state in ("candidate", "queued", "skipped"):  # no filter hides an unreadable shot
        assert "009" in [r["id"] for r in core.visible(rows, state)]
    assert core.counts(rows) == {"candidate": 1, "queued": 1, "skipped": 0} and core.errors(rows) == 1
    assert None not in core.counts(rows)
    assert "cannot read" in core.row_line(bad) and not core.has_range(bad) and core.start_of(bad) == 0.0
    assert core.queue_check(rows) == (["001"], [])  # afv runs it as failed; the panel shows its error
    st = {"id": 12, "running": True, "batch_id": "b7", "log": "media/batch/b7.jsonl",
          "current": {"shot": "012", "index": 3, "stage": "pose3d", "frac": 0.4},
          "outcomes": [{"shot": "003", "outcome": "ok", "message": "", "exports": []}], "done": None}
    v = core.batch_view(st)
    assert v["running"] and v["id"] == "b7" and v["shot"] == "012" and v["stage"] == "pose3d"
    assert v["index"] == 3 and v["frac"] == 0.4 and v["log"].endswith(".jsonl")
    assert v["outcomes"] == [{"shot": "003", "outcome": "ok", "message": ""}]
    v = core.batch_view({"id": 99, "running": False, "batch_id": None, "current": None, "outcomes": [],
                         "done": None})
    assert v["id"] is None  # the reply's request id is not a batch id
    done = {"batch_id": "b8", "cancelled": True,
            "outcomes": [{"shot": "004", "outcome": "failed", "message": "x"}]}
    v = core.batch_view({"running": False, "batch_id": "b8", "finished": "2026-11-11T22:41:05-0600",
                         "outcomes": [], "done": done})
    assert v["cancelled"] and v["outcomes"][0]["message"] == "x" and core.when(v["finished"]) == "11-11 22:41"
    assert core.batch_view(None)["running"] is False
    assert core.outcome_counts([{"outcome": "ok"}, {"outcome": "needs_review"}, {"outcome": "ok"}]) == (
        "2 ran ok, 1 needs a tracking review")
    assert core.row_detail(rows[0], "needs_review").endswith("last run: needs a tracking review")


def test_messages_for_the_panel():
    root = "/mnt/c/Users/me/AppData/Local/Temp/x/scratchpad/ff2"
    msg = (f"cannot read {root}/shots/003/shot.json; the fighters were picked for the old interval: pick "
           "again (afv pick 011); its scan is for the old interval: afv shots scan 011")
    out = core.humanize(msg, (root, "C:\\Users\\me\\ff2"))
    assert "shots/003/shot.json" in out and "/mnt/c" not in out
    assert "Pick fighters on shot 011" in out and "Scan shots (shot 011)" in out and "afv " not in out
    out = core.humanize("log in /mnt/d/afv/projects/other/media/batch/abc.stderr")
    assert out == "log in ...batch/abc.stderr"


def test_scan_before_picking():
    rows = [row("001", 0, 2, state="queued"), row("002", 2, 2, state="queued", scan="stale"),
            row("003", 4, 2, state="queued", scan="none"), row("004", 6, 2, scan="none"),
            row("005", 8, 2, state="queued", picked=True, scan="none")]
    assert core.needs_scan(rows) == ["002", "003"]  # queued, needing a pick, without an up-to-date scan
    assert core.needs_scan(rows, ["004", "001"]) == ["004"]
    assert core.scan_estimate(20) < core.scan_estimate(1) * 2  # one job: the GPU start is paid once


def test_session_rows_update():
    s = core.Session()
    assert not s.open and s.fps == 30.0
    payload = {"project": "/p", "root_win": "D:\\p", "source": {"fps": FPS, "num_frames": 29406},
               "shots": [row("002", 2, 2), row("001", 0, 2)]}
    s.load(payload)
    assert s.open and s.num_frames == 29406 and [r["id"] for r in s.visible()] == ["001", "002"]
    v = s.version
    s.update_row({**row("001", 0, 2), "state": "queued"})
    assert s.version == v + 1 and s.counts()["queued"] == 1
    assert [r["id"] for r in s.visible("queued")] == ["001"]  # the cache follows the update
    s.update_row(None)
    assert s.version == v + 1


def test_oneshot_command_lines():
    p = {"project": "/mnt/d/p"}
    assert core.oneshot_argv("shots", p) == ["shots", "list", "--project", "/mnt/d/p"]
    assert core.oneshot_argv("shot_set", {**p, "shot": "012", "start": 3.5, "state": "queued"}) == [
        "shots", "set", "012", "--project", "/mnt/d/p", "--start", "3.5", "--state", "queued"]
    assert core.oneshot_argv("shot_set", {**p, "shot": "1", "notes": ""})[-2:] == ["--notes", ""]
    assert core.oneshot_argv("shot_add", {**p, "start": 1, "end": 2.5, "state": "queued"}) == [
        "shots", "add", "--start", "1.0", "--project", "/mnt/d/p", "--end", "2.5", "--state", "queued"]
    assert core.oneshot_argv("pick", {**p, "shot": "3"}) == ["pick", "3", "--project", "/mnt/d/p"]
    assert core.oneshot_argv("pick", {**p, "shot": "3", "fighters": [1, 0], "extra": [2], "frame": 15}) == [
        "pick", "3", "--project", "/mnt/d/p", "--fighters", "1,0", "--extra", "2", "--frame", "15"]
    assert core.oneshot_argv("shots_scan", {**p, "force": True}) == ["shots", "scan", "--project", "/mnt/d/p",
                                                                     "--force"]
    # Run the N picked ones: exactly those shots (as serve's batch); Run queue: the queued state
    assert core.oneshot_argv("batch", {**p, "to": "detect_track", "shots": ["003", "017"]}) == [
        "batch", "--project", "/mnt/d/p", "--to", "detect_track", "--shot", "003", "--shot", "017"]
    assert core.oneshot_argv("batch", {**p, "to": "export", "state": "queued"}) == [
        "batch", "--project", "/mnt/d/p", "--to", "export", "--state", "queued"]
    assert core.oneshot_argv("init", {"video": "/v.mp4", "project": "/p", "title": "T"}) == [
        "init", "/v.mp4", "--project", "/p", "--detect", "--title", "T"]
    with pytest.raises(ValueError):
        core.oneshot_argv("run", p)


def test_oneshot_results(tmp_path):
    (tmp_path / "media").mkdir()
    (tmp_path / "project.json").write_text(json.dumps({"source": {"path": "/v.mp4", "sha256": "abc",
                                                                  "probe": {
        "avg_fps": FPS, "display_width": 1280, "display_height": 720, "duration": 10.0, "num_frames": 300}}}))
    (tmp_path / "media" / "cuts.json").write_text("{}")
    evs = [{"event": "shot", **row("001", 0, 2)}, {"event": "shot", **row("002", 2, 2, state="queued")}]
    res = core.oneshot_result("shots", {"project": "/p"}, evs, str(tmp_path))
    assert [r["id"] for r in res["shots"]] == ["001", "002"] and res["counts"]["queued"] == 1
    assert res["source"]["width"] == 1280 and res["cuts"] == "media/cuts.json" and res["proxy"] is None
    (tmp_path / "media" / "proxy.mp4").write_bytes(b"x")
    (tmp_path / "media" / "proxy.json").write_text(json.dumps({"source_sha256": "abc"}))
    assert core.oneshot_result("shots", {"project": "/p"}, evs, str(tmp_path))["proxy"] == "media/proxy.mp4"
    ev = {"event": "shot_set", **row("001", 0.5, 1.5), "shot": "001", "changed": ["start"],
          "notes_for_user": []}
    res = core.oneshot_result("shot_set", {}, [ev], None)
    assert res["shot"] == "001" and res["changed"] == ["start"] and res["row"]["start"] == 0.5
    ev = {"event": "pick_prompt", "shot": "1", "source": "scan", "prompt_image": "/x.jpg",
          "prompt": {"prompt_frame": 15, "boxes": [[0, 0, 1, 1]]}}
    res = core.oneshot_result("pick", {"fighters": None}, [ev])
    assert res["frame"] == 15 and res["prompt"]["boxes"] and res["image"] == "/x.jpg"
    res = core.oneshot_result("batch", {}, [{"event": "batch_done", "outcomes": {"001": "ok"}, "ok": 1}])
    assert res["outcomes"] == [{"shot": "001", "outcome": "ok", "message": "", "exports": []}]  # older afv
    out = [{"shot": "001", "outcome": "ok", "message": "", "exports": ["001_A.fbx"]},
           {"shot": "002", "outcome": "needs_review", "message": "hints", "exports": []}]
    res = core.oneshot_result("batch", {}, [{"event": "batch_done", "outcomes": out, "ok": 1}])
    assert res["outcomes"] == out and res["ok"] == 1  # afv now: a list, as serve
    res = core.oneshot_result("take", {}, [{"event": "take", "path": "/p/shots/001/take/take.json"}])
    assert res == {"path": "/p/shots/001/take/take.json"}
    assert core.oneshot_result("shot_set", {}, [], None) is None  # failed: no event


def test_pick_box_hit_test():
    """A prompt box (image pixels) mapped onto a camera frame drawn at an offset and scale is hit where it is
    drawn; the smaller of two overlapping boxes wins (the nearer fighter)."""
    corners = ((100.0, 500.0), (740.0, 500.0), (100.0, 140.0))  # top-left, top-right, bottom-left (region)
    size = (1280, 720)
    boxes = {0: (90.0, 90.0, 940.0, 715.0), 1: (840.0, 360.0, 1100.0, 715.0)}
    region = []
    for k, (x0, y0, x1, y1) in boxes.items():
        a = takeio.image_to_region(corners, size, x0, y0)
        b = takeio.image_to_region(corners, size, x1, y1)
        region.append((k, (a[0], a[1], b[0], b[1])))
    cx, cy = takeio.image_to_region(corners, size, 300.0, 400.0)
    assert takeio.box_at(region, cx, cy) == 0
    cx, cy = takeio.image_to_region(corners, size, 900.0, 600.0)  # inside both: the smaller one
    assert takeio.box_at(region, cx, cy) == 1
    assert takeio.inside_frame(corners, 120.0, 480.0) and not takeio.inside_frame(corners, 50.0, 480.0)
