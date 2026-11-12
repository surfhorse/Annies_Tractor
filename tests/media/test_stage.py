from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from media_synth import logo_frames, write_video

import afv.media.stage as media_stage
from afv.core.stage import Ctx, StageError, get_stage
from afv.core.types import load_result, save_result
from afv.media.ffmpeg import ffmpeg_version
from afv.media.normalise import shot_frame_count
from afv.media.png import read_png_gray
from afv.media.probe import probe
from afv.media.result import ShotMedia


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("proj")
    frames, _ = logo_frames(75)
    padded = []
    for f in frames:  # add 20 px letterbox bars top and bottom
        img = np.zeros((220, 320, 3), np.uint8)
        img[20:200] = f
        padded.append(img)
    src = write_video(root / "src.mp4", padded, fps=25, crf=18, audio=True)
    (root / "project.json").write_text(json.dumps({"source": str(src), "sha256": "abc"}))
    shot = root / "shots" / "s001"
    shot.mkdir(parents=True)
    (shot / "shot.json").write_text(json.dumps({"start": 0.5, "duration": 2.0}))
    return root


def test_stage_registered() -> None:
    cls = get_stage("ingest", "ffmpeg")  # registered by importing afv.media.stage, as the runner does
    assert cls is media_stage.IngestStage
    assert cls.env == "cpu" and cls.inputs == () and cls.weight_hash == ""
    assert cls.licences and cls.licences[0].commercial_output_ok


def test_ingest_stage(project: Path) -> None:
    cls = get_stage("ingest", "ffmpeg")
    shot_dir = project / "shots" / "s001"
    work = shot_dir / "stages" / "ingest.tmp"
    progress: list[float] = []
    ctx = Ctx(shot_dir=shot_dir, work_dir=work, progress=lambda f, m: progress.append(f))
    res = cls().run(ctx, {}, cls.Params())
    assert isinstance(res, ShotMedia)
    src = project / "src.mp4"
    assert res.num_frames == 60 == shot_frame_count(src, 0.5, 2.0)
    np.testing.assert_array_equal(res.frames, np.arange(60))
    np.testing.assert_allclose(res.timestamps, 0.5 + np.arange(60) / 30)
    assert (res.width, res.height) == (320, 220) and res.fps == 30.0
    assert res.start == 0.5 and res.duration == 2.0 and res.source_sha256 == "abc"
    assert res.letterbox == [20, 20, 0, 0]
    assert res.probe["cv_frame_count"] == 75 and res.ffmpeg_version == ffmpeg_version()
    assert probe(work / res.video, hash=False).num_frames == 60
    assert probe(work / res.proxy, hash=False).num_frames == 60
    png = read_png_gray(work / res.overlay_png)
    np.testing.assert_array_equal(png, res.overlay_mask * 255)
    assert res.overlay_reliable and len(res.overlay_boxes) == 1
    x0, y0, x1, y1 = res.overlay_boxes[0]  # logo at (16, 12)+(96, 40) in the content -> y + 20
    assert abs(x0 - 16) <= 4 and abs(y0 - 32) <= 4 and abs(x1 - 112) <= 4 and abs(y1 - 72) <= 4
    assert progress[0] == 0.0 and progress[-1] == 1.0

    save_result(res, work / "result.npz")
    back = load_result(work / "result.npz")
    assert isinstance(back, ShotMedia) and back.num_frames == 60
    np.testing.assert_array_equal(back.overlay_mask, res.overlay_mask)
    assert back.letterbox == res.letterbox and back.overlay_boxes == res.overlay_boxes


def test_params_override_and_errors(project: Path, tmp_path: Path) -> None:
    cls = get_stage("ingest", "ffmpeg")
    shot_dir = project / "shots" / "s001"
    res = cls().run(Ctx(shot_dir=shot_dir, work_dir=tmp_path / "w1"), {},
                    cls.Params(start=0.0, duration=3.0, proxy=False))
    assert res.num_frames == 90 and res.proxy == ""  # 75 frames at 25 fps -> 3 s at 30 Hz
    assert not (tmp_path / "w1" / "proxy.mp4").exists()
    with pytest.raises(StageError):
        cls().run(Ctx(shot_dir=shot_dir, work_dir=tmp_path / "w2"), {}, cls.Params(start=10.0))
    with pytest.raises(StageError):
        cls().run(Ctx(shot_dir=tmp_path / "x" / "shots" / "s", work_dir=tmp_path / "w3"), {}, cls.Params())


def test_uses_stored_probe_and_checks_ffmpeg(project: Path, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """No per-shot re-probe when project.json holds a current probe; a wrong ffmpeg fails clearly."""
    cls = get_stage("ingest", "ffmpeg")
    src = project / "src.mp4"
    root = tmp_path / "p2"
    (root / "shots" / "s1").mkdir(parents=True)
    (root / "shots" / "s1" / "shot.json").write_text(json.dumps({"start": 0.2, "duration": 1.0}))
    stored = probe(src, hash=False).to_dict()
    source = {"path": str(src), "sha256": "x", "probe": stored}
    (root / "project.json").write_text(json.dumps({"source": source}))

    def no_probe(*a: object, **k: object) -> None:
        raise AssertionError("re-probed the source")

    monkeypatch.setattr(media_stage, "probe", no_probe)
    shot_dir = root / "shots" / "s1"
    res = cls().run(Ctx(shot_dir=shot_dir, work_dir=tmp_path / "w"), {}, cls.Params(proxy=False))
    assert res.num_frames == 30 and res.source_sha256 == "x"
    monkeypatch.undo()

    stale = dict(stored, probe_version=0)  # an older stored copy: probed again (header only)
    (root / "project.json").write_text(json.dumps({"source": dict(source, probe=stale)}))
    assert cls().run(Ctx(shot_dir=root / "shots" / "s1", work_dir=tmp_path / "w2"), {},
                     cls.Params(proxy=False)).num_frames == 30
    with pytest.raises(StageError, match=r"expected 0\.0\.1"):
        cls().run(Ctx(shot_dir=root / "shots" / "s1", work_dir=tmp_path / "w3"), {},
                  cls.Params(expected_ffmpeg_version="0.0.1"))
