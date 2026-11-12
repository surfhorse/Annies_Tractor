"""Candidate scan (afv.media.shots): exact prompt frames, fighters_likely, cuts.json and scan_shots with a
fake track worker (synthetic video only; the real-footage checks are in test_scan_real.py and
tests/models/test_scan_gpu.py)."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from media_synth import scene_cuts_frames, write_video

from afv.core.project import Project, load_scan, read_scan, scan_paths, shot_hash
from afv.media import shots as ms
from afv.media.cuts import Cut, CutAnalysis, CutParams


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def scenes_video(tmp_path_factory) -> Path:
    """25 fps, 320x180: scenes of 150, 40 and 60 frames (6 s, 1.6 s, 2.4 s), textured with noise so x264
    has real decisions to make."""
    d = tmp_path_factory.mktemp("scan_video")
    return write_video(d / "scenes.mp4", scene_cuts_frames([150, 40, 60], w=320, h=180), fps=25.0, crf=18)


def fake_worker(boxes: list[list[float]] | None = None, *, sleep: float = 0.0, fail_after: int | None = None,
                calls: list[dict] | None = None, started: threading.Event | None = None):
    """A stand-in for the track worker's ``detect`` mode: the same files and response, boxes made up."""
    boxes = boxes if boxes is not None else [[60, 30, 140, 170], [170, 40, 240, 175], [5, 5, 20, 30]]

    def call(req: dict[str, Any], work_dir: Path, gpu: bool) -> dict[str, Any]:
        assert req["mode"] == "detect" and gpu
        if calls is not None:
            calls.append(req)
        if started is not None:
            started.set()
        fdir, out = Path(req["frames_dir"]), Path(req["out_dir"])
        assert len(list(fdir.glob("*.jpg"))) == req["num_frames"]
        prompts = []
        for k, f in enumerate(req["frames"]):
            if fail_after is not None and k >= fail_after:
                return {"ok": False, "error": "RuntimeError: CUDA error: out of memory"}
            time.sleep(sleep)
            assert (fdir / f"{f:05d}.jpg").is_file()
            info = {"frame": f, "prompt_frame": f, "image": f"prompt_{f:05d}.jpg", "boxes": boxes,
                    "scores": [0.9, 0.8, 0.3][: len(boxes)], "threshold": req["threshold"]}
            (out / f"prompt_{f:05d}.jpg").write_bytes(b"\xff\xd8fake")
            (out / f"prompt_{f:05d}.json").write_text(json.dumps(info))
            prompts.append(info)
        return {"ok": True, "num_frames": req["num_frames"], "prompts": prompts}

    return call


@pytest.fixture
def scan_project(tmp_path, scenes_video) -> Project:
    from afv.cli import _probe

    probe, _ = _probe(scenes_video)
    proj = Project.create(tmp_path / "p.afv", scenes_video, probe=probe)
    proj.add_shot(0.0, 6.0, state="candidate")  # long enough for the shortened encode
    proj.add_shot(6.0, 1.6, state="candidate")
    proj.add_shot(7.6, 0.4, state="queued")  # 12 frames: the last frame is scanned
    return proj


# ---------------------------------------------------------------------------
# exact frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("start", "duration"), [(0.0, 6.0), (0.3, 5.5), (6.0, 1.6), (7.6, 0.4)])
def test_scan_frame_is_byte_identical_to_prepare_frames(tmp_path, scenes_video, start, duration):
    from afv.models.track import prepare_frames

    k = ms.extract_scan_frame(scenes_video, start, duration, tmp_path / "f.jpg", tmp_path / "t.jpg")
    fr = prepare_frames(scenes_video, start, duration, tmp_path / "cache")
    assert k == min(ms.SCAN_FRAME, fr.num_frames - 1)
    assert sha(tmp_path / "f.jpg") == sha(fr.frames_dir / f"{k:05d}.jpg")
    th = ms.read_jpeg(tmp_path / "t.jpg")
    assert th.shape == (180, 320, 3)  # never upscaled (THUMB_WIDTH 480 > 320)


def test_scan_frame_margin_matters(tmp_path, scenes_video):
    """The margin is what makes the frame exact: with none, x264's look-ahead sees a different future."""
    from afv.models.track import prepare_frames

    fr = prepare_frames(scenes_video, 0.0, 6.0, tmp_path / "cache")
    ms.extract_scan_frame(scenes_video, 0.0, 6.0, tmp_path / "f0.jpg", tmp_path / "t.jpg", margin=0)
    ms.extract_scan_frame(scenes_video, 0.0, 6.0, tmp_path / "f.jpg", tmp_path / "t.jpg")
    assert sha(tmp_path / "f.jpg") == sha(fr.frames_dir / "00015.jpg")
    assert sha(tmp_path / "f0.jpg") != sha(fr.frames_dir / "00015.jpg")


def test_thumbnail_width(tmp_path):
    big = write_video(tmp_path / "big.mp4", scene_cuts_frames([10], w=960, h=540), fps=30.0)
    ms.extract_scan_frame(big, 0.0, 0.3, tmp_path / "f.jpg", tmp_path / "t.jpg")
    assert ms.read_jpeg(tmp_path / "t.jpg").shape == (270, 480, 3)
    assert ms.read_jpeg(tmp_path / "f.jpg").shape == (540, 960, 3)


# ---------------------------------------------------------------------------
# fighters_likely, cuts.json, numbered image
# ---------------------------------------------------------------------------


H = 720.0


def box(x: float, top: float, h: float, w: float | None = None) -> list[float]:
    w = w if w is not None else h / 2
    return [x, top * H, x + w, (top + h) * H]


@pytest.mark.parametrize(
    ("boxes", "scores", "likely", "sized"),
    [
        ([box(100, 0.2, 0.7), box(600, 0.25, 0.65)], [0.9, 0.9], True, 2),  # two fighters
        ([box(100, 0.2, 0.7), box(600, 0.25, 0.65), box(900, 0.1, 0.8)], [0.9, 0.9, 0.8], True, 3),  # + ref
        ([box(100, 0.2, 0.7)], [0.9], False, 1),  # one person (or a clinch seen as one box)
        ([box(100, 0.2, 0.7), box(600, 0.6, 0.2)], [0.9, 0.9], False, 1),  # second one far away / small
        ([box(100, 0.2, 0.7), box(600, 0.25, 0.65)], [0.9, 0.4], False, 1),  # second one unsure
        ([box(100, 0.0, 0.99), box(600, 0.0, 0.98)], [0.9, 0.9], False, 0),  # close-ups: heads cut off
        ([box(100, 0.1, 0.8, w=900), box(600, 0.25, 0.65)], [0.9, 0.9], False, 1),  # a face filling the frame
        ([box(100 * k, 0.1, 0.6) for k in range(4)], [0.9] * 4, False, 4),  # a crowd
        ([box(100, 0.2, 0.7), box(600, 0.25, 0.4)], [0.9, 0.9], False, 2),  # 0.4 < 0.6 x 0.7
        ([], [], False, 0),
    ],
)
def test_fighters_likely(boxes, scores, likely, sized):
    assert ms.fighters_likely(boxes, scores, H) == (likely, sized)


def test_cuts_payload_and_read(tmp_path):
    cuts = [Cut(40, 1.3167, 0.9, 0.5), Cut(80, 2.65, 0.4, 0.3, "gradual", 70, 2.3167)]
    z = np.zeros(100)
    an = CutAnalysis(cuts, np.arange(100) / 30.0, z, z, z, 29.97, CutParams())
    d = ms.cuts_payload(an, "abc", min_confidence=0.2)
    assert (d["format"], d["version"], d["source_sha256"], d["fps"]) == ("afv_cuts", 1, "abc", 29.97)
    assert d["params"]["min_confidence"] == 0.2 and d["params"]["detector"]["k_sigma"] == 6.0
    assert d["cuts"][0] == {"frame": 40, "time": 1.3167, "confidence": 0.9, "kind": "hard", "start_frame": 40,
                            "start_time": 1.3167}
    assert d["cuts"][1]["kind"] == "gradual" and d["cuts"][1]["start_frame"] == 70
    p = tmp_path / "cuts.json"
    p.write_text(json.dumps(d))
    assert ms.read_cuts(p) == d
    p.write_text('{"format": "other"}')
    assert ms.read_cuts(p) is None and ms.read_cuts(tmp_path / "missing.json") is None


def test_draw_numbered_and_jpeg_roundtrip(tmp_path):
    img = np.full((180, 320, 3), 40, np.uint8)
    out = ms.draw_numbered(img, [[10, 10, 100, 170], [150, 20, 400, 300]])  # second box past the edge
    assert out.shape == img.shape and (out != img).any() and (img == 40).all()
    assert tuple(out[10, 50]) == ms._COLOURS_RGB[0]  # the first box's top edge
    ms.write_jpeg(out, tmp_path / "n.jpg")
    back = ms.read_jpeg(tmp_path / "n.jpg")
    assert back.shape == out.shape and np.abs(back.astype(int) - out).mean() < 6


# ---------------------------------------------------------------------------
# scan_shots
# ---------------------------------------------------------------------------


def test_scan_shots_writes_scans_and_resumes(scan_project):
    proj = scan_project
    calls: list[dict] = []
    seen: list[str] = []
    events: list[tuple] = []
    rep = ms.scan_shots(proj, proj.shots(), worker=fake_worker(calls=calls),
                        on_scan=lambda s, d: seen.append(s), progress=lambda f, m: events.append((f, m)))
    assert sorted(rep.scanned) == ["001", "002", "003"] and not rep.failed and not rep.error
    assert len(calls) == 1 and calls[0]["frames"] == [0, 1, 2]  # one worker request for every shot
    assert calls[0]["threshold"] == 0.4 and sorted(seen) == ["001", "002", "003"]
    assert events[0][0] == 0.0 and any(f is not None and f > 0.5 for f, _ in events)
    for sid, frame in (("001", 15), ("002", 15), ("003", 11)):  # 0.4 s = 12 frames: the last one
        sh = proj.shot(sid)
        data = load_scan(sh)
        assert data is not None and data["frame"] == frame and data["shot"] == sid
        assert data["prompt"]["frame"] == data["prompt"]["prompt_frame"] == frame
        assert data["prompt"]["image"] == f"{sid}.prompt.jpg" and data["prompt"]["threshold"] == 0.4
        assert data["people"] == 3 and data["size"] == [320, 180]
        assert data["shot_hash"] == shot_hash(sh.info.start, sh.info.duration, proj.info.source.sha256)
        p = scan_paths(proj, sid)
        assert p.thumb.is_file() and p.frame.is_file() and not p.prompt.exists()
    # 001 and 002 are different shots: their frames differ
    assert sha(scan_paths(proj, "001").frame) != sha(scan_paths(proj, "002").frame)
    # resumable: up-to-date scans are kept
    calls.clear()
    rep = ms.scan_shots(proj, proj.shots(), worker=fake_worker(calls=calls))
    assert rep.scanned == [] and sorted(rep.up_to_date) == ["001", "002", "003"] and calls == []
    rep = ms.scan_shots(proj, [proj.shot("2")], worker=fake_worker(calls=calls), force=True)
    assert rep.scanned == ["002"] and len(calls) == 1
    # a trimmed shot's scan is stale and scanned again
    proj.shot("1").update(start=0.5, duration=5.0)
    assert read_scan(proj.shot("1"))[0] == "stale" and load_scan(proj.shot("1")) is None
    rep = ms.scan_shots(proj, proj.shots(), worker=fake_worker())
    assert rep.scanned == ["001"] and read_scan(proj.shot("1"))[0] == "ok"


def test_scan_shots_keeps_what_finished_when_the_worker_fails(scan_project):
    proj = scan_project
    rep = ms.scan_shots(proj, proj.shots(), worker=fake_worker(fail_after=2))
    assert "out of memory" in rep.error and rep.scanned == ["001", "002"]
    assert rep.failed == {"003": "not scanned: tracking (detect) failed: RuntimeError: CUDA error: out of "
                                  "memory"}
    rep = ms.scan_shots(proj, proj.shots(), worker=fake_worker())
    assert rep.scanned == ["003"] and sorted(rep.up_to_date) == ["001", "002"]


def test_scan_shots_bad_shot_does_not_stop_the_others(scan_project):
    proj = scan_project
    proj.add_shot(9.0, 0.5, shot_id="late")
    p = proj.shots_dir / "late" / "shot.json"
    p.write_text(p.read_text().replace('"start": 9.0', '"start": 20.0'))  # a hand edit: past the end
    rep = ms.scan_shots(proj, proj.shots(), worker=fake_worker())
    assert sorted(rep.scanned) == ["001", "002", "003"] and list(rep.failed) == ["late"]
    assert rep.failed["late"].startswith("MediaError: Video start 20.000s must be before the end")


def test_scan_shots_cancel(scan_project):
    proj = scan_project
    cancel = threading.Event()
    cancel.set()
    rep = ms.scan_shots(proj, proj.shots(), worker=fake_worker(), cancel=cancel)
    assert rep.cancelled and rep.scanned == []
    cancel = threading.Event()
    started = threading.Event()
    out: dict = {}

    def run():
        out["rep"] = ms.scan_shots(proj, proj.shots(), worker=fake_worker(sleep=0.4, started=started),
                                   cancel=cancel)

    th = threading.Thread(target=run)
    th.start()
    started.wait(20)
    cancel.set()  # the fake worker is not stopped (the real one is, by serve); the scan ends after it
    th.join(30)
    assert out["rep"].cancelled


def set_threshold(sh, value):
    sh.modify(lambda i: i.settings.params.update(detect_track={"det_threshold": value}))


def test_scan_threshold_follows_the_shot_settings(scan_project):
    set_threshold(scan_project.shot("2"), 0.55)
    calls: list[dict] = []
    ms.scan_shots(scan_project, scan_project.shots(), worker=fake_worker(calls=calls))
    assert sorted((c["threshold"], c["num_frames"]) for c in calls) == [(0.4, 2), (0.55, 1)]
    assert load_scan(scan_project.shot("2"))["prompt"]["threshold"] == 0.55
    # a threshold change makes the scan stale (its numbering would differ): scanned again
    set_threshold(scan_project.shot("2"), 0.6)
    assert read_scan(scan_project.shot("2"))[0] == "stale" and load_scan(scan_project.shot("2")) is None
    rep = ms.scan_shots(scan_project, scan_project.shots(), worker=fake_worker())
    assert rep.scanned == ["002"] and load_scan(scan_project.shot("2"))["prompt"]["threshold"] == 0.6
    project_settings = scan_project.info
    project_settings.settings.params["detect_track"] = {"det_threshold": 0.45}
    scan_project.save_info(project_settings)  # the project's threshold: every shot without its own
    assert [read_scan(sh)[0] for sh in scan_project.shots()] == ["stale", "ok", "stale"]


def test_scan_write_errors_are_the_shots_not_the_detectors(scan_project, monkeypatch):
    real = ms._finish_scan

    def finish(project, it, *a, **k):
        if it.shot.id == "002":
            raise OSError(28, "No space left on device", "/x/002.json")
        return real(project, it, *a, **k)

    monkeypatch.setattr(ms, "_finish_scan", finish)
    rep = ms.scan_shots(scan_project, scan_project.shots(), worker=fake_worker())
    assert not rep.error and sorted(rep.scanned) == ["001", "003"]
    assert rep.failed == {"002": "cannot write the scan: No space left on device (/x/002.json)"}


def test_scan_hints_use_the_current_rule():
    scan = {"prompt": {"boxes": [[100, 150, 300, 650], [600, 160, 800, 640]], "scores": [0.9, 0.9]},
            "size": [1280, 720], "fighters_likely": False, "fighter_sized": 0}  # stored by an older rule
    assert ms.scan_hints(scan) == (True, 2)
    scan["prompt"]["boxes"] = [[100, 0, 600, 715], [600, 0, 1200, 712]]  # two close-ups
    scan["fighters_likely"] = True
    assert ms.scan_hints(scan) == (False, 0)
    assert ms.scan_hints({"fighters_likely": True, "fighter_sized": 2}) == (True, 2)  # no people: as stored


def test_short_partial_encode_falls_back_to_the_whole_shot(tmp_path, scenes_video, monkeypatch):
    from afv.models.track import prepare_frames

    real = ms.normalised_frame_count
    calls: list[int] = []

    def count(p):
        n = real(p)
        calls.append(n)
        return 40 if len(calls) == 1 else n  # the shortened encode "has" too few frames (a gap)

    monkeypatch.setattr(ms, "normalised_frame_count", count)
    k = ms.extract_scan_frame(scenes_video, 0.0, 6.0, tmp_path / "f.jpg", tmp_path / "t.jpg")
    assert len(calls) == 2 and calls[1] == 180  # the second encode is the whole 6 s shot
    fr = prepare_frames(scenes_video, 0.0, 6.0, tmp_path / "cache")
    assert k == 15 and sha(tmp_path / "f.jpg") == sha(fr.frames_dir / "00015.jpg")


def test_prompt_image_written_on_demand(scan_project):
    proj = scan_project
    ms.scan_shots(proj, [proj.shot("1")], worker=fake_worker())
    sh = proj.shot("1")
    p = ms.prompt_image(sh)
    assert p == scan_paths(proj, "001").prompt and p.is_file()
    img = ms.read_jpeg(p)
    assert img.shape == (180, 320, 3)
    t = p.stat().st_mtime_ns
    assert ms.prompt_image(sh) == p and p.stat().st_mtime_ns == t  # kept while the scan is unchanged
    assert ms.prompt_image(proj.shot("2")) is None  # not scanned
    ms.remove_scan(proj, "001")
    assert not any(f.exists() for f in (p, scan_paths(proj, "001").json, scan_paths(proj, "001").thumb))


def test_cli_shots_scan(scan_project, monkeypatch):
    from typer.testing import CliRunner

    from afv.cli import app

    calls: list[dict] = []
    monkeypatch.setitem(ms.scan_shots.__kwdefaults__, "worker", fake_worker(calls=calls))
    root = str(scan_project.root)
    run = CliRunner().invoke
    res = run(app, ["shots", "scan", "-p", root], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    assert "scanned 3 shots (0 already up to date)" in res.output
    assert "fighters likely in 3: 001-003" in res.output
    res = run(app, ["shots", "scan", "2", "--force", "-p", root, "--json"], catch_exceptions=False)
    evs = [json.loads(x) for x in res.output.splitlines()]
    assert [e["shot"] for e in evs if e["event"] == "scanned"] == ["002"]
    assert evs[-1]["event"] == "scan_done" and evs[-1]["scanned"] == ["002"] and evs[-1]["shots"] == ["002"]
    scan_project.shot("3").update(state="skipped")
    res = run(app, ["shots", "scan", "--state", "skipped", "-p", root, "--force"], catch_exceptions=False)
    assert "scanned 1 shots" in res.output and len(calls) == 3
    res = run(app, ["shots", "list", "-p", root], catch_exceptions=False)
    assert "3 people" in res.output
    monkeypatch.setitem(ms.scan_shots.__kwdefaults__, "worker", fake_worker(fail_after=0))
    res = run(app, ["shots", "scan", "--force", "-p", root], catch_exceptions=False)
    assert res.exit_code == 1 and "error: tracking (detect) failed" in res.output
