"""Identity swaps (``edits/ids.json``) in detect_track: applied to the stored tracking (no re-tracking),
with QC and tracking hints recomputed; ``afv pick --swap``. The model worker is faked (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from afv.cli import app
from afv.core.edits import EditError, IdsFile, Swap, validate_edit, write_edit
from afv.core.project import Project, stage_dir
from afv.core.review import tracking_hints
from afv.core.runner import Runner
from afv.models import track as T
from afv.models.track import HIST_DIM, DetectTrackStage, RawTrack

W, H, FPS, SECONDS = 320, 180, 25, 1.2
BOXES = np.array([[20, 20, 70, 170], [200, 20, 250, 170], [120, 30, 150, 160]], np.float32)
SAM_SWAP = 10  # the fake SAM 2 exchanges objects 1 and 2 from this frame on


def make_video(path: Path) -> None:
    import av

    rng = np.random.default_rng(1)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=FPS)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        for _ in range(int(SECONDS * FPS)):
            img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


class FakeWorker:
    """Standing people; SAM 2's objects 1 and 2 trade places from SAM_SWAP on. Mask files hold
    "<tracked object>:<frame>" so their arrangement can be checked."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, req: dict, work_dir: Path, gpu: bool) -> dict:
        self.calls.append(req["mode"])
        out = Path(req.get("out_dir", work_dir))
        n = req["num_frames"]
        if req["mode"] == "render":
            for it in req["images"]:
                Path(it["stem"] + ".jpg").write_bytes(b"jpg")
            if req.get("out"):
                Path(req["out"]).write_bytes(b"mp4")
            self.rendered = RawTrack.load(Path(req["raw"]))
            return {"ok": True}
        assert req["mode"] == "track"
        ids = req["obj_ids"]
        b = np.repeat(BOXES[None, : len(ids)], n, 0)
        b[SAM_SWAP:, [0, 1]] = b[SAM_SWAP:, [1, 0]]
        df = np.repeat(np.arange(n), 3)
        np.savez(out / "raw.npz", obj_ids=np.array(ids), boxes=b, area=np.full((n, len(ids)), 1e4),
                 centroid=(b[..., :2] + b[..., 2:]) / 2, hist=np.ones((n, len(ids), HIST_DIM), np.float32),
                 det_frame=df, det_boxes=np.tile(BOXES, (n, 1)), det_scores=np.full(3 * n, 0.9, np.float32),
                 width=W, height=H)  # fmt: skip
        for oid in ids:
            (out / "masks" / str(oid)).mkdir(parents=True)
            for f in range(n):
                (out / "masks" / str(oid) / f"{f:05d}.png").write_text(f"{oid}:{f}")
        return {"ok": True, "num_frames": n, "raw": "raw.npz", "masks_dir": "masks",
                "prompts": req["prompts"], "stats": {"sam2_seconds": 1.0}}  # fmt: skip


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("clip") / "c.mp4"
    make_video(p)
    return p


@pytest.fixture
def fake(monkeypatch):
    fw = FakeWorker()
    monkeypatch.setattr(DetectTrackStage, "worker", fw)
    return fw


@pytest.fixture
def shot(tmp_path, clip, fake):
    sh = Project.create(tmp_path / "proj", clip).add_shot(0.0, 1.0)
    sh.write_edit("fighters", {"fighters": [0, 1], "extra": [2]})
    run(sh)
    assert fake.calls == ["track", "render"]
    fake.calls.clear()
    return sh


def run(sh, **kw):
    rep = Runner(sh).run(to="detect_track", **kw)
    assert rep.outcome == "ok", rep.message
    return rep


def mask(sh, oid: int, f: int) -> str:
    return (stage_dir(sh.dir, "detect_track") / "masks" / str(oid) / f"{f:05d}.png").read_text()


def hints(sh) -> list[dict]:
    return tracking_hints(sh.meta("detect_track")["qc"], sh.id)


# -- units -------------------------------------------------------------------------------------------


def test_swap_permutation_composes_in_order():
    sw = [Swap(start=2, end=None, objects=[1, 2]), Swap(start=4, end=6, objects=[2, 3])]
    perm = T.swap_permutation(sw, [1, 2, 3], 8)
    assert perm[:2].tolist() == [[0, 1, 2]] * 2
    assert perm[2:4].tolist() == [[1, 0, 2]] * 2
    assert perm[4:6].tolist() == [[1, 2, 0]] * 2  # the second swap acts on what the first left
    assert perm[6:].tolist() == [[1, 0, 2]] * 2
    with pytest.raises(T.TrackError, match="not tracked"):
        T.swap_permutation([Swap(start=0, objects=[1, 9])], [1, 2, 3], 8)
    with pytest.raises(T.TrackError, match=r"frames 0\.\.7"):
        T.swap_permutation([Swap(start=8, objects=[1, 2])], [1, 2, 3], 8)


def test_apply_swaps_moves_every_per_object_array():
    F, P = 4, 3
    rng = np.random.default_rng(0)
    raw = RawTrack(
        obj_ids=np.arange(1, P + 1), boxes=rng.random((F, P, 4)), area=rng.random((F, P)),
        centroid=rng.random((F, P, 2)), hist=rng.random((F, P, 5)), det_frame=np.zeros(0, int),
        det_boxes=np.zeros((0, 4)), det_scores=np.zeros(0), mask_iou=rng.random((F, P, P)),
    )  # fmt: skip
    perm = T.swap_permutation([Swap(start=1, objects=[1, 3])], [1, 2, 3], F)
    out = T.apply_swaps(raw, perm)
    assert np.array_equal(out.boxes[0], raw.boxes[0])
    assert np.array_equal(out.boxes[2, 0], raw.boxes[2, 2]) and np.array_equal(out.area[3, 2], raw.area[3, 0])
    assert np.array_equal(out.hist[1, 0], raw.hist[1, 2])
    assert out.mask_iou[2, 0, 1] == raw.mask_iou[2, 2, 1] and out.mask_iou[2, 2, 0] == raw.mask_iou[2, 0, 2]
    assert T.apply_swaps(raw, T.swap_permutation([], [1, 2, 3], F)) is raw


def test_arrange_masks_links_through_two_arrangements(tmp_path):
    ids, F = [1, 2], 3
    src = tmp_path / "a"
    for o in ids:
        (src / str(o)).mkdir(parents=True)
        for f in range(F):
            (src / str(o) / f"{f:05d}.png").write_text(f"{o}:{f}")
    p1 = T.swap_permutation([Swap(start=1, objects=[1, 2])], ids, F)
    T.arrange_masks(src, None, tmp_path / "b", p1, ids, move=False)
    assert (tmp_path / "b/1/00001.png").read_text() == "2:1"
    assert (tmp_path / "b/1/00000.png").read_text() == "1:0"
    p2 = T.swap_permutation([Swap(start=2, objects=[1, 2])], ids, F)  # from b (arranged by p1) to p2
    T.arrange_masks(tmp_path / "b", p1, tmp_path / "c", p2, ids, move=True)
    assert [(tmp_path / "c/1" / f"{f:05d}.png").read_text() for f in range(F)] == ["1:0", "1:1", "2:2"]


def test_ids_edit_format():
    validate_edit("ids", {"swaps": [{"start": 3, "end": None, "objects": [1, 2]}], "note": "kept"})
    for bad in (
        {"swaps": [{"start": 3, "objects": [1, 1]}]},
        {"swaps": [{"start": 3, "end": 3, "objects": [1, 2]}]},
        {"swaps": [{"start": -1, "objects": [1, 2]}]},
        {"swaps": [{"objects": [1, 2]}]},
    ):
        with pytest.raises(EditError):
            validate_edit("ids", bad)
    assert IdsFile.model_validate({"swaps": []}).swaps == []


# -- the stage ---------------------------------------------------------------------------------------


def test_swap_reuses_the_tracking_and_clears_the_hint(shot, fake):
    n = shot.load_result("detect_track").num_frames
    base_tracks = shot.load_result("detect_track")
    first = stage_dir(shot.dir, "detect_track").name
    assert any(h["kind"] == "identity" for h in hints(shot)), hints(shot)
    assert any("--swap" in h["fix"] for h in hints(shot) if h["kind"] == "identity")

    shot.write_edit("ids", {"swaps": [{"start": SAM_SWAP, "end": None, "objects": [1, 2]}]})
    rep = run(shot)
    assert fake.calls == ["render"]  # no tracking, no GPU
    assert [s.state for s in rep.stages if s.name == "detect_track"] == ["ran"]
    tr = shot.load_result("detect_track")
    assert np.array_equal(tr.boxes[:, 0], np.repeat(BOXES[None, 0], n, 0))  # fighter A stays put now
    assert np.array_equal(tr.boxes[:, 1], np.repeat(BOXES[None, 1], n, 0))
    assert not hints(shot)  # QC recomputed on the swapped identities
    assert fake.rendered.boxes[SAM_SWAP, 0].tolist() == BOXES[0].tolist()  # the review shows them too
    sdir = stage_dir(shot.dir, "detect_track")
    qc = json.loads((sdir / "qc.json").read_text())
    assert qc["reused_tracking"] == first and qc["ids"]["swaps"][0]["start"] == SAM_SWAP
    assert RawTrack.load(sdir / "raw.npz").boxes[SAM_SWAP, 0].tolist() == BOXES[1].tolist()  # as tracked
    assert mask(shot, 1, SAM_SWAP - 1) == f"1:{SAM_SWAP - 1}" and mask(shot, 1, SAM_SWAP) == f"2:{SAM_SWAP}"
    assert mask(shot, 3, SAM_SWAP) == f"3:{SAM_SWAP}"

    # a second swap on top (reused from the swapped result), then none: back to the tracked identities
    fake.calls.clear()
    data = shot.read_edit("ids")
    data["swaps"].append({"start": 15, "end": 20, "objects": [1, 3]})
    shot.write_edit("ids", data)
    run(shot)
    assert fake.calls == ["render"]
    assert mask(shot, 1, 16) == "3:16" and mask(shot, 3, 16) == "2:16" and mask(shot, 1, 22) == "2:22"
    shot.write_edit("ids", {"swaps": []})
    run(shot)
    assert mask(shot, 1, SAM_SWAP) == f"1:{SAM_SWAP}"
    again = shot.load_result("detect_track")
    assert np.array_equal(again.boxes, base_tracks.boxes, equal_nan=True)

    fake.calls.clear()
    run(shot, from_stage="detect_track")  # forced: tracks again
    assert fake.calls == ["track", "render"]


@pytest.mark.parametrize("code", ["missing", "other"])
def test_no_reuse_when_the_code_record_differs(shot, fake, code):
    sdir = stage_dir(shot.dir, "detect_track")
    meta = json.loads((sdir / "meta.json").read_text())
    if code == "missing":
        meta.pop("code")
    else:
        meta["code"]["hash"] = "0" * 64
    (sdir / "meta.json").write_text(json.dumps(meta))
    shot.write_edit("ids", {"swaps": [{"start": SAM_SWAP, "objects": [1, 2]}]})
    run(shot)
    assert fake.calls == ["track", "render"]  # tracked again: the stored tracking is from other code


def test_no_reuse_when_ids_did_not_change(shot, fake):
    rep = Runner(shot, code_changes="rerun").run(to="detect_track")  # nothing changed: nothing runs
    assert rep.outcome == "ok" and fake.calls == []
    tracks = T.stored_tracking  # a re-run for another reason (here forced by hand) tracks again
    from afv.core.codehash import snapshot_code
    from afv.core.stage import Ctx

    ctx = Ctx(shot_dir=shot.dir, work_dir=shot.dir / ".work" / "x", edits={})
    code = snapshot_code(DetectTrackStage)["hash"]
    p = Runner(shot).params_for("detect_track", DetectTrackStage)
    args = (ctx, p, shot.read_edit("fighters"), DetectTrackStage.version, DetectTrackStage.weight_hash)
    assert tracks(*args, ids=None, code_hash=code) is None  # ids unchanged (absent before and now)
    assert tracks(*args, ids={"swaps": []}, code_hash=code) is not None


def test_swap_on_the_first_tracking_and_bad_swaps(shot, fake, tmp_path):
    shot.write_edit("fighters", {"fighters": [1, 0], "extra": [2]})  # new picks: no reuse
    shot.write_edit("ids", {"swaps": [{"start": SAM_SWAP, "objects": [1, 2]}]})
    run(shot)
    assert fake.calls == ["track", "render"]
    assert mask(shot, 2, SAM_SWAP) == f"1:{SAM_SWAP}" and not hints(shot)
    shot.write_edit("ids", {"swaps": [{"start": 0, "objects": [1, 7]}]})
    rep = Runner(shot).run(to="detect_track")
    assert rep.outcome == "failed" and "not tracked" in rep.message


def test_pick_swap_cli(shot, fake):
    def cli(*args, code=0):
        res = CliRunner().invoke(app, ["pick", shot.id, *args, "-p", str(shot.project.root)])
        assert res.exit_code == code, res.output
        return res.output

    out = cli("--swap", f"{SAM_SWAP}:1:2")
    assert "objects 1 <-> 2" in out
    assert shot.read_edit("ids") == {"swaps": [{"start": SAM_SWAP, "end": None, "objects": [1, 2]}]}
    cli("--swap", "20:2:3")
    assert len(shot.read_edit("ids")["swaps"]) == 2
    assert "not tracked" in cli("--swap", "3:1:9", code=1)
    assert "outside the shot" in cli("--swap", "999:1:2", code=1)
    assert "FRAME:OBJ1:OBJ2" in cli("--swap", "3-1-2", code=1)
    cli("--clear-swaps")
    assert shot.read_edit("ids") == {"swaps": []}
    write_edit(shot, "ids", {"swaps": [], "note": "mine"})
    cli("--swap", "4:1:2")
    assert shot.read_edit("ids")["note"] == "mine"  # unknown keys kept
