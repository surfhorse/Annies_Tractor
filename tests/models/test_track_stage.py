"""detect_track through the real runner, ingest, frame normalisation and `afv pick` on a tiny synthetic
clip; only the model worker is faked (no GPU). Covers needs-pick -> pick -> suggestion -> re-prompt,
re-picking, stale picks and the stage-folder file layout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_track_gpu import make_video
from typer.testing import CliRunner

from afv.cli import app
from afv.core.project import Project, stage_dir
from afv.core.runner import Runner
from afv.core.stage import Ctx, Stage, StageError, get_stage
from afv.core.types import Tracks
from afv.models import track as T
from afv.models.track import HIST_DIM, DetectTrackStage, TrackParams, pick_data

BOXES = np.array([[60, 60, 130, 330], [300, 60, 370, 330], [450, 80, 500, 330]], np.float32)
LATE = np.array([560, 60, 630, 330], np.float32)  # a fourth person, detected from frame ENTER on
ENTER = 20


def fake_dets(f: int) -> tuple[np.ndarray, np.ndarray]:
    b, s = BOXES, np.array([0.9, 0.8, 0.7], np.float32)
    if f >= ENTER:
        b, s = np.vstack([b, LATE[None]]), np.append(s, 0.85).astype(np.float32)
    return b, s


class FakeWorker:
    """The worker's file protocol with standing people; the 4th is only tracked when prompted."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, req: dict, work_dir: Path, gpu: bool) -> dict:
        self.calls.append({**req, "_gpu": gpu})
        out = Path(req.get("out_dir", work_dir))
        out.mkdir(parents=True, exist_ok=True)
        n = req["num_frames"]
        assert len(list(Path(req["frames_dir"]).glob("*.jpg"))) >= n  # the real normalised frames
        if req["mode"] == "detect":
            prompts = []
            for f in req["frames"]:
                b, s = fake_dets(f)
                info = {"frame": f, "prompt_frame": f, "image": f"prompt_{f:05d}.jpg", "boxes": b.tolist(),
                        "scores": s.tolist()}  # fmt: skip
                (out / f"prompt_{f:05d}.json").write_text(json.dumps(info))
                (out / f"prompt_{f:05d}.jpg").write_bytes(b"jpg")
                prompts.append(info)
            return {"ok": True, "num_frames": n, "prompts": prompts}
        if req["mode"] == "render":
            for it in req["images"]:
                Path(it["stem"] + ".jpg").write_bytes(b"jpg")
                Path(it["stem"] + ".json").write_text("{}")
            if req.get("out"):
                Path(req["out"]).write_bytes(b"mp4")
            return {"ok": True}
        ids = req["obj_ids"]
        allb = np.vstack([BOXES, LATE[None]])
        b = np.repeat(allb[None, : len(ids)], n, 0)
        if len(ids) == 4:
            b[:ENTER, 3] = np.nan
        df, db, ds = [], [], []
        for f in range(n):
            bb, ss = fake_dets(f)
            df += [f] * len(bb)
            db.append(bb)
            ds.append(ss)
        np.savez(out / "raw.npz", obj_ids=np.array(ids), boxes=b, area=np.where(np.isnan(b[..., 0]), 0, 1e4),
                 centroid=(b[..., :2] + b[..., 2:]) / 2, hist=np.ones((n, len(ids), HIST_DIM), np.float32),
                 det_frame=np.array(df), det_boxes=np.vstack(db), det_scores=np.concatenate(ds),
                 width=640, height=360)  # fmt: skip
        (out / "masks").mkdir(exist_ok=True)
        stats = {"sam2_seconds": 1.0, "mem_trace": [[30, 1.0, 2.0]]}
        prompts = req["prompts"]
        return {"ok": True, "num_frames": n, "raw": "raw.npz", "masks_dir": "masks", "prompts": prompts,
                "stats": stats}  # fmt: skip


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    video = tmp_path_factory.mktemp("clip") / "synthetic.mp4"
    make_video(video)  # 640x360, 2.4 s at 25 fps
    return video


@pytest.fixture
def shot(tmp_path, clip):
    proj = Project.create(tmp_path / "proj", clip)
    return proj.add_shot(0.0, 2.0)


@pytest.fixture
def fake(monkeypatch):
    fw = FakeWorker()
    monkeypatch.setattr(DetectTrackStage, "worker", fw)
    return fw


def cli_pick(shot, *args, code=0) -> str:
    res = CliRunner().invoke(
        app, ["pick", shot.id, *args, "-p", str(shot.project.root)], catch_exceptions=False
    )
    assert res.exit_code == code, res.output
    return res.output


def run(shot, outcome="ok"):
    rep = Runner(shot).run(to="detect_track")
    assert rep.outcome == outcome, rep
    return rep


def work_ctx(shot, **kw) -> Ctx:
    work = shot.dir / ".work" / "t"
    work.mkdir(parents=True, exist_ok=True)
    return Ctx(shot_dir=shot.dir, work_dir=work, **kw)


def test_registered_and_protocol():
    cls = get_stage("detect_track", "rfdetr_sam2")
    assert cls is DetectTrackStage and isinstance(cls(), Stage)
    assert cls.env == "track" and len(cls.weight_hash) == 64 and len(cls.licences) == 2
    assert cls.edits == ("fighters", "ids")


def test_full_flow_with_real_ingest_pick_and_reprompt(shot, fake, clip):
    from afv.media.normalise import shot_frame_count

    n = shot_frame_count(clip, 0.0, 2.0)
    run(shot, "needs_pick")
    assert stage_dir(shot.dir, "ingest").is_dir()
    sdir = stage_dir(shot.dir, "detect_track")
    prompt = json.loads((sdir / "prompt.json").read_text())
    assert prompt["prompt_frame"] == 0 and len(prompt["boxes"]) == 3 and prompt["image"] == "prompt.jpg"
    unpicked = shot.load_result("detect_track")
    assert isinstance(unpicked, Tracks) and unpicked.fighters == {} and unpicked.num_frames == n

    cli_pick(shot, "--fighters", "1,0", "--extra", "2")
    edit = shot.read_edit("fighters")
    assert edit["shot"] == {"start": 0.0, "duration": 2.0, "source_sha256": shot.project.info.source.sha256}
    fake.calls.clear()
    run(shot)
    tr = shot.load_result("detect_track")
    assert tr.fighters == {"A": 1, "B": 2} and tr.track_ids.tolist() == [1, 2, 3]
    assert tr.num_frames == n and np.allclose(tr.timestamps, np.arange(n) / 30)
    assert [c["mode"] for c in fake.calls] == ["track", "render"]
    assert fake.calls[0]["_gpu"] and not fake.calls[1]["_gpu"]
    prompts = [(p["frame"], p["obj"], p["detection"]) for p in fake.calls[0]["prompts"]]
    assert prompts == [(0, 1, 1), (0, 2, 0), (0, 3, 2)]

    # review item 9: results live in stages/detect_track/vNNNN/ and qc.json paths are relative to it
    sdir = stage_dir(shot.dir, "detect_track")
    qc = json.loads((sdir / "qc.json").read_text())
    assert qc["files"] == {"details": "qc_frames.json", "masks": "masks", "review": "review.mp4"}
    assert all((sdir / v).exists() for v in qc["files"].values())
    assert T.masks_dir_for(shot.dir) == sdir / "masks"
    assert json.loads((sdir / "meta.json").read_text())["qc"]["frames"] == n  # the runner reads "summary"
    assert "mem_trace" in json.loads((sdir / "qc_frames.json").read_text())
    sug = [s for s in qc["summary"]["suggestions"] if s["kind"] == "untracked_person"]
    assert len(sug) == 1 and sug[0]["frames"] == [ENTER, n - 1] and sug[0]["obj"] == 4
    assert sug[0]["detection"] == 3 and sug[0]["box"] == LATE.tolist() and (sdir / sug[0]["image"]).is_file()

    # apply the suggestion: four people tracked, suggestion gone
    s = sug[0]
    shot.write_edit(
        "fighters", T.add_prompt(edit, s["frame"], s["obj"], detection=s["detection"], box=s["box"])
    )
    run(shot)
    tr = shot.load_result("detect_track")
    assert tr.track_ids.tolist() == [1, 2, 3, 4] and np.isnan(tr.boxes[:ENTER, 3]).all()
    qc = json.loads((stage_dir(shot.dir, "detect_track") / "qc.json").read_text())
    assert not [s for s in qc["summary"]["suggestions"] if s["kind"] == "untracked_person"]

    # review item 1: re-picking with swapped fighters drops the re-prompts made for the old ids
    assert "dropped 1 re-prompt" in cli_pick(shot, "--fighters", "0,1", "--extra", "2")
    run(shot)
    assert shot.load_result("detect_track").track_ids.tolist() == [1, 2, 3]

    # review item 7: picks made for another interval stop the stage with "re-pick"
    data = shot.read_edit("fighters")
    data["shot"]["start"] = 0.5
    shot.write_edit("fighters", data)
    assert "Re-pick" in run(shot, "failed").message


def test_stage_edits_from_ctx_and_params_override(shot, fake, clip):
    data = pick_data(T.PromptInfo(0, Path("p"), BOXES, np.ones(3, np.float32)), 0, 1)
    tr = DetectTrackStage().run(
        work_ctx(shot, edits={"fighters": data}),
        {},
        TrackParams(video=str(clip), start=1.0, review_video=False),
    )
    assert tr.track_ids.tolist() == [1, 2] and tr.timestamps[0] == 1.0
    assert "out" not in fake.calls[-1]  # render only draws the numbered images


def test_invalid_edit_and_worker_failure(shot, fake, monkeypatch):
    with pytest.raises(StageError, match="distinct"):
        DetectTrackStage().run(work_ctx(shot, edits={"fighters": {"fighters": [0, 0]}}), {}, TrackParams())
    shot.write_edit("fighters", {"fighters": [0, 1]})
    failing = staticmethod(lambda req, wd, gpu: {"ok": False, "error": "CUDA OOM"})
    monkeypatch.setattr(DetectTrackStage, "worker", failing)
    with pytest.raises(StageError, match="CUDA OOM"):
        DetectTrackStage().run(work_ctx(shot), {}, TrackParams())


def test_frame_count_mismatch(shot, monkeypatch):
    fw = FakeWorker()

    def short(req, wd, gpu):
        return fw({**req, "num_frames": req["num_frames"] - 1} if req["mode"] == "track" else req, wd, gpu)

    monkeypatch.setattr(DetectTrackStage, "worker", staticmethod(short))
    shot.write_edit("fighters", {"fighters": [0, 1]})
    with pytest.raises(StageError, match="frames but the shot has"):
        DetectTrackStage().run(work_ctx(shot), {}, TrackParams())


def test_prepare_frames_cache_and_safety(tmp_path, clip):
    from afv.media.normalise import shot_frame_count

    fr = T.prepare_frames(clip, 0.0, None, tmp_path / "cache")
    assert fr.num_frames == shot_frame_count(clip) and (fr.width, fr.height) == (640, 360)
    stamp = (fr.frames_dir / "00000.jpg").stat().st_mtime_ns
    assert T.prepare_frames(clip, 0.0, None, tmp_path / "cache") == fr
    assert (fr.frames_dir / "00000.jpg").stat().st_mtime_ns == stamp  # cached
    m = json.loads((tmp_path / "cache" / "frames.json").read_text())
    assert m["key"]["normaliser"].startswith("v2:")
    foreign = tmp_path / "mine"
    foreign.mkdir()
    (foreign / "precious.txt").write_text("x")
    with pytest.raises(StageError, match="not a tracking frame cache"):
        T.prepare_frames(clip, 0.0, None, foreign)
    assert (foreign / "precious.txt").exists()
