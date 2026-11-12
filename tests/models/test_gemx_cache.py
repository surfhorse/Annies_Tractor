"""pose3d's GEM-X raw-output cache (<shot>/.cache/pose3d, ``GemxOutputCache``): the gravity re-level
pass turns the shot world by one rotation, which GEM-X never sees, so its network run is reused (CPU;
fake gemx worker)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_camera_stages import FIX, FakeWorkers, _rx

import afv.models.gemx as gemx_mod
from afv.core.stage import Ctx
from afv.core.types import CameraTrack, Tracks
from afv.models.camera import (
    camera_to_world_cv,
    gemx_network_rotations,
    gemx_provided_camera,
    make_camera_track,
)
from afv.models.gemx import Pose3dGemx, Pose3dParams, SomaFK, camera_from_gemx


def _turned(cam: CameraTrack, deg: float) -> CameraTrack:
    """``cam`` in a world turned about x by ``deg`` (what a gravity re-level does), float32 like a
    stored result."""
    r, c = camera_to_world_cv(cam)
    t = _rx(deg)
    out = make_camera_track(
        np.einsum("ab,fbc->fac", t, r),
        c @ t.T,
        cam.focal_px,
        width=cam.width,
        height=cam.height,
        cx=cam.cx,
        cy=cam.cy,
    )
    out.cam_rot = out.cam_rot.astype(np.float32)
    out.cam_pos = out.cam_pos.astype(np.float32)
    return out


class Shot:
    def __init__(self, tmp: Path, monkeypatch):
        self.tmp, self.n = tmp, 0
        self.video = tmp / "source.mp4"
        self.video.write_bytes(b"a source video")
        self.fk = SomaFK.load(FIX)
        n = self.fk.num_frames
        self.cam = camera_from_gemx(self.fk)
        self.cam.cam_rot = self.cam.cam_rot.astype(np.float32)
        self.boxes = np.stack([np.load(FIX)["boxes_xyxy"]] * 2, axis=1)

        def pose3d(req):
            out = {}
            for f in req["fighters"]:
                d = Path(req["out_dir"]) / f["role"]
                d.mkdir(parents=True)
                shutil.copy(FIX, d / "soma_fk.npz")
                (d / "motion.pt").write_bytes(b"gem-x motion " + f["role"].encode())
                out[f["role"]] = {"soma_fk": str(d / "soma_fk.npz"), "motion_pt": str(d / "motion.pt")}
            return {"frames": n, "fighters": out, "timings": {"motion_s": 1.0}, "ffmpeg": "x"}

        self.fake = FakeWorkers(tmp, {"pose3d": pose3d})
        monkeypatch.setattr(gemx_mod, "run_env_worker", self.fake)

    def tracks(self, boxes=None) -> Tracks:
        b = self.boxes if boxes is None else boxes
        n = b.shape[0]
        return Tracks(
            frames=np.arange(n),
            timestamps=np.arange(n) / 30.0,
            track_ids=np.array([7, 9]),
            boxes=b,
            scores=np.ones((n, 2), np.float32),
            fighters={"A": 7, "B": 9},
        )

    def run(self, cam=None, *, boxes=None, force=False, **params):
        self.n += 1
        work = self.tmp / f"work{self.n}"
        work.mkdir()
        ctx = Ctx(shot_dir=self.tmp / "shot", work_dir=work, force=force)
        before = len(self.fake.calls)
        motion = Pose3dGemx().run(
            ctx,
            {"detect_track": self.tracks(boxes), "camera": self.cam if cam is None else cam},
            Pose3dParams(video=str(self.video), **params),
        )
        qc = json.loads((work / "qc.json").read_text())
        return motion, qc, len(self.fake.calls) - before, work

    def entries(self) -> list[Path]:
        root = self.tmp / "shot" / ".cache" / "pose3d"
        return sorted(p for p in root.glob("*") if not p.name.startswith(".tmp")) if root.exists() else []


@pytest.fixture
def shot(tmp_path, monkeypatch):
    return Shot(tmp_path, monkeypatch)


def test_network_rotations_do_not_see_the_world_orientation():
    fk = SomaFK.load(FIX)
    cam = camera_from_gemx(fk)
    r1, _ = gemx_provided_camera(cam)
    r2, _ = gemx_provided_camera(_turned(cam, 4.0))
    assert np.abs(r1 - r2).max() > 0.05  # the world's R_w2c differ by the turn
    n1, n2 = gemx_network_rotations(r1), gemx_network_rotations(r2)
    assert np.abs(n1 - n2).max() < 1e-6 and np.allclose(n1[0], np.eye(3), atol=1e-6)
    inc = lambda r: np.einsum("tab,tcb->tac", r[1:].astype(np.float64), r[:-1].astype(np.float64))  # noqa: E731
    assert np.abs(inc(n1) - inc(r1)).max() < 1e-6  # GEM-X's cam_angvel is unchanged


def test_relevel_pass_reuses_the_network_run(shot):
    m1, qc1, calls1, _ = shot.run()
    assert calls1 == 1 and qc1["gemx_cache"]["status"] == "miss, stored"
    m2, qc2, calls2, w2 = shot.run(_turned(shot.cam, 4.0))  # the re-level: one world rotation
    assert calls2 == 0 and qc2["gemx_cache"]["status"] == "hit"
    assert qc2["gemx_cache"]["entry"] == shot.entries()[0].name
    for role in ("A", "B"):  # the worker's files are in the work folder, paths point there
        assert (w2 / "gemx" / role / "soma_fk.npz").read_bytes() == FIX.read_bytes()
        assert (w2 / "gemx" / role / "motion.pt").read_bytes() == b"gem-x motion " + role.encode()
    assert qc2["worker"]["timings"] == {"motion_s": 1.0}
    # the result lives in the turned world: same body, re-expressed
    assert not np.allclose(m1.actors["A"].root, m2.actors["A"].root, atol=1e-3)
    assert qc1["fighters"]["A"]["reprojection_px_median"] == pytest.approx(
        qc2["fighters"]["A"]["reprojection_px_median"], abs=1e-3
    )


def test_force_reruns_and_replaces_the_entry(shot):
    shot.run()
    (old,) = shot.entries()
    _, qc, calls, _ = shot.run(force=True)
    assert calls == 1 and qc["gemx_cache"]["status"] == "stored"
    (new,) = shot.entries()
    assert new != old
    _, qc, calls, _ = shot.run()
    assert calls == 0 and qc["gemx_cache"]["entry"] == new.name


def test_every_network_input_is_in_the_key(shot):
    shot.run()
    boxes = shot.boxes.copy()
    boxes[5, 0] += 3.0
    assert shot.run(boxes=boxes)[2] == 1  # boxes
    zoomed = _turned(shot.cam, 0.0)
    zoomed.focal_px = zoomed.focal_px * np.linspace(1.0, 1.02, zoomed.num_frames)
    assert shot.run(zoomed)[2] == 1  # K per frame
    r, c = camera_to_world_cv(shot.cam)
    wobble = np.stack([_rx(0.01 * np.sin(t)) for t in range(len(r))])  # 0.01 deg: a different camera
    other = make_camera_track(
        np.einsum("fab,fbc->fac", r, wobble),
        c,
        shot.cam.focal_px,
        width=shot.cam.width,
        height=shot.cam.height,
        cx=shot.cam.cx,
        cy=shot.cam.cy,
    )
    assert shot.run(other)[2] == 1  # rotations
    assert shot.run(lod="low")[2] == 1  # request fields
    assert shot.run(boxes=boxes)[2] == 0  # each variant keeps its entry
    shot.video.write_bytes(b"another source video")
    assert shot.run()[2] == 1  # source
    assert shot.run(lod="xlo")[2] == 1
    assert len(shot.entries()) == gemx_mod.GEMX_CACHE_KEEP  # pruned to the newest


def test_worker_script_change_misses_and_drops_old_entries(shot, monkeypatch):
    shot.run()
    real = gemx_mod.file_sha256
    monkeypatch.setattr(
        gemx_mod, "file_sha256", lambda p: "changed" if Path(p).name == "gemx_worker.py" else real(p)
    )
    assert shot.run()[2] == 1 and len(shot.entries()) == 1


def test_a_corrupt_entry_is_dropped(shot):
    shot.run()
    (entry,) = shot.entries()
    (entry / "files" / "A" / "soma_fk.npz").write_bytes(b"truncated")
    _, qc, calls, w = shot.run()
    assert calls == 1 and not entry.exists() and qc["gemx_cache"]["status"] == "miss, stored"
    assert (w / "gemx" / "A" / "soma_fk.npz").read_bytes() == FIX.read_bytes()


def test_cache_can_be_switched_off(shot):
    _, qc, calls, _ = shot.run(cache=False)
    assert calls == 1 and qc["gemx_cache"] == {"status": "off"} and shot.entries() == []
    assert shot.run(cache=False)[2] == 1


def test_cache_only_runs_reuse_gemx_or_raise(shot):
    from afv.core.stage import CacheOnlyMiss

    shot.run()
    shot.n += 1
    work = shot.tmp / f"work{shot.n}"
    work.mkdir()
    ctx = Ctx(shot_dir=shot.tmp / "shot", work_dir=work, cache_only=True)
    inputs = {"detect_track": shot.tracks(), "camera": shot.cam}
    Pose3dGemx().run(ctx, inputs, Pose3dParams(video=str(shot.video)))  # a hit: fine
    calls = len(shot.fake.calls)
    with pytest.raises(CacheOnlyMiss):
        Pose3dGemx().run(ctx, inputs, Pose3dParams(video=str(shot.video), lod="low"))  # a miss
    with pytest.raises(CacheOnlyMiss):
        Pose3dGemx().run(ctx, inputs, Pose3dParams(video=str(shot.video), cache=False))
    assert len(shot.fake.calls) == calls
