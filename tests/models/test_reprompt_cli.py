"""Round 2: `afv pick <shot> --frame F` and `--prompt F:OBJ:DET [--label]` (re-prompting from the CLI),
and detect_track's frame-count check (no cache dependency on ingest). Real runner, ingest and frame
normalisation on a synthetic clip; only the tracking worker is faked."""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_track_stage import ENTER, LATE, FakeWorker, cli_pick, clip, run  # noqa: F401  (clip: fixture)

from afv.core.project import Project, stage_dir
from afv.core.runner import Runner
from afv.models import track as T
from afv.models.track import DetectTrackStage, TrackError


@pytest.fixture
def worker(monkeypatch):
    fw = FakeWorker()
    monkeypatch.setattr(DetectTrackStage, "worker", fw)
    monkeypatch.setattr(T, "envs_worker", lambda progress=None: fw)
    return fw


@pytest.fixture
def shot(tmp_path, request):
    return Project.create(tmp_path / "proj", request.getfixturevalue("clip")).add_shot(0.0, 2.0)


def test_detect_track_is_not_keyed_on_ingest(shot, worker):
    assert DetectTrackStage.inputs == ()
    run(shot, "needs_pick")
    key = shot.meta("detect_track")["key"]
    assert shot.meta("detect_track")["inputs"] == {}
    # an ingest-only change (no proxy) re-runs ingest but leaves tracking fresh
    rep = Runner(shot, params={"ingest": {"proxy": False}}).run(to="detect_track")
    states = {s.name: s.state for s in rep.stages}
    assert states["ingest"] == "ran" and states["detect_track"] in ("fresh", "needs_pick")
    assert shot.meta("detect_track")["key"] == key


@pytest.mark.parametrize("override", [False, True])
def test_detect_track_checks_frames_against_a_fresh_gemx_count(shot, worker, monkeypatch, override):
    import afv.media.normalise as norm
    from afv.core.stage import Ctx

    monkeypatch.setattr(norm, "shot_frame_count", lambda video, start=0.0, duration=None, **kw: 3)
    ctx_dir = shot.dir / ".work" / "t"
    ctx_dir.mkdir(parents=True)
    params = T.TrackParams(start=0.0, duration=1.5) if override else T.TrackParams()
    what = "the detect_track params' interval" if override else "the shot"
    with pytest.raises(TrackError, match=f"for {what} .*GEM-X's timeline has 3"):
        DetectTrackStage().run(Ctx(shot_dir=shot.dir, work_dir=ctx_dir), {}, params)


def test_pick_frame_shows_numbered_people_on_any_frame(shot, worker):
    run(shot, "needs_pick")
    out = cli_pick(shot, "--frame", str(ENTER + 2))
    assert f"frame {ENTER + 2}:" in out and "  3: box [560, 60, 630, 330]" in out
    img = shot.dir / ".cache" / "track" / "prompts" / f"prompt_{ENTER + 2:05d}.jpg"
    assert img.is_file() and str(img) in out
    calls = [c for c in worker.calls if c["mode"] == "detect"]
    cli_pick(shot, "--frame", str(ENTER + 2))  # cached: no new detection
    assert len([c for c in worker.calls if c["mode"] == "detect"]) == len(calls)
    assert "outside the shot" in cli_pick(shot, "--frame", "9999", code=1)


def test_pick_prompt_adds_a_late_person_with_a_label_and_tracks_them(shot, worker):
    run(shot, "needs_pick")
    assert "pick the fighters first" in cli_pick(shot, "--prompt", f"{ENTER}:4:3", code=1)
    cli_pick(shot, "--fighters", "1,0", "--extra", "2")
    out = cli_pick(shot, "--prompt", f"{ENTER}:4:3", "--label", "referee")
    assert f"frame {ENTER}, object 4 = detection 3 (referee)" in out
    data = shot.read_edit("fighters")
    assert data["prompts"] == [{"frame": ENTER, "obj": 4, "detection": 3, "box": LATE.tolist()}]
    assert {"id": 4, "label": "referee"} in data["objects"]
    run(shot)
    tr = shot.load_result("detect_track")
    assert tr.track_ids.tolist() == [1, 2, 3, 4] and np.isnan(tr.boxes[:ENTER, 3]).all()
    # the same (frame, obj) replaces the earlier prompt; a detection number that does not exist fails
    cli_pick(shot, "--prompt", f"{ENTER}:4:3")
    assert len(shot.read_edit("fighters")["prompts"]) == 1
    assert "detections 0..3, not 7" in cli_pick(shot, "--prompt", f"{ENTER}:4:7", code=1)
    assert "FRAME:OBJ:DET" in cli_pick(shot, "--prompt", "20:4", code=1)
    assert "cannot be combined" in cli_pick(shot, "--prompt", "20:4:3", "--fighters", "1,0", code=1)


def test_prompt_on_the_base_prompt_frame_uses_the_stored_numbering(shot, worker):
    run(shot, "needs_pick")
    cli_pick(shot, "--fighters", "1,0")
    n_detect = len([c for c in worker.calls if c["mode"] == "detect"])
    cli_pick(shot, "--prompt", "0:3:2", "--label", "referee")
    assert len([c for c in worker.calls if c["mode"] == "detect"]) == n_detect  # prompt.json reused
    p = json.loads((stage_dir(shot.dir, "detect_track") / "prompt.json").read_text())
    assert shot.read_edit("fighters")["prompts"][0]["box"] == [round(v, 1) for v in p["boxes"][2]]
    assert Runner(shot).run(to="detect_track").outcome == "ok"


def test_pick_json_keeps_the_shot_id(shot, worker):
    run(shot, "needs_pick")
    ev = json.loads(cli_pick(shot, "--fighters", "1,0", "--json").splitlines()[-1])
    assert ev["event"] == "picked" and ev["shot"] == shot.id
    sha = shot.project.info.source.sha256
    assert ev["picked_for"] == {"start": 0.0, "duration": 2.0, "source_sha256": sha}
    ev = json.loads(cli_pick(shot, "--prompt", f"{ENTER}:4:3", "--json").splitlines()[-1])
    assert ev["event"] == "reprompted" and ev["shot"] == shot.id and ev["picked_for"]["duration"] == 2.0


def test_label_without_prompt_is_an_error(shot, worker):
    run(shot, "needs_pick")
    assert "--label needs --prompt" in cli_pick(shot, "--frame", "3", "--label", "referee", code=1)
    assert "--label needs --prompt" in cli_pick(shot, "--label", "referee", code=1)


def test_reprompt_waits_for_the_shot_lock(shot, worker):
    import fcntl
    import os

    run(shot, "needs_pick")
    cli_pick(shot, "--fighters", "1,0")
    before = shot.read_edit("fighters")
    fd = os.open(shot.work_root / ".lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert "already being run" in cli_pick(shot, "--prompt", f"{ENTER}:4:3", code=1)
        assert shot.read_edit("fighters") == before
    finally:
        os.close(fd)
