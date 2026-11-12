"""Fixtures for the pipeline core: an isolated stage registry with fake CPU stages and a project."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from afv.core import runner as runner_mod
from afv.core import stage as stage_mod
from afv.core.project import Project
from afv.core.stage import StageError, register
from afv.core.types import ActorMotion, BodyMotion, CameraTrack, LicenceRecord, Tracks

T = 10  # frames per fake shot


def _timeline() -> dict[str, np.ndarray]:
    return {"frames": np.arange(T), "timestamps": 1.0 + np.arange(T) / 30.0}


def _make_fakes() -> types.SimpleNamespace:
    calls: list[str] = []

    class Ingest:
        name, version, env, inputs, licences, weight_hash = "ingest", "1", "cpu", (), (), ""

        class Params(BaseModel):
            proxy_height: int = 720

        def run(self, ctx, inputs, params):
            calls.append("ingest")
            (ctx.work_dir / "proxy.txt").write_text("proxy")
            (ctx.work_dir / "tmp").mkdir()
            (ctx.work_dir / "tmp" / "scratch.bin").write_bytes(b"x" * 100)
            return CameraTrack(**_timeline(), width=params.proxy_height, height=params.proxy_height)

    class DetectTrack:
        name, version, env, inputs, licences, weight_hash = (
            "detect_track",
            "1",
            "cpu",
            (),
            (),
            "",
        )  # as module D: no inputs

        class Params(BaseModel):
            prompt_frame: int = 0

        def run(self, ctx, inputs, params):
            calls.append("detect_track")
            assert inputs == {}
            boxes = [[10, 10, 50, 90], [60, 10, 100, 90], [110, 20, 130, 80]]
            (ctx.work_dir / "prompt.jpg").write_bytes(b"\xff\xd8fake")
            (ctx.work_dir / "prompt.json").write_text(
                json.dumps({"prompt_frame": params.prompt_frame, "boxes": boxes, "scores": [0.9, 0.8, 0.7]})
            )
            pick = ctx.edits.get("fighters")
            if not pick:
                return Tracks(**_timeline())
            ids = [*pick["fighters"], *pick.get("extra", [])]
            b = np.array([boxes[i] for i in ids], np.float32)[None].repeat(T, 0)
            return Tracks(
                **_timeline(),
                track_ids=np.array(ids),
                boxes=b,
                scores=np.ones((T, len(ids)), np.float32),
                fighters={"A": pick["fighters"][0], "B": pick["fighters"][1]},
            )

    class CameraBase:
        name, version, env, inputs, licences, weight_hash = (
            "camera",
            "1",
            "cpu",
            ("ingest", "detect_track"),
            (),
            "",
        )

        class Params(BaseModel):
            focal: float = 1000.0
            note: str = ""  # does not change the output

        def run(self, ctx, inputs, params):
            calls.append(f"camera:{self.mode}")
            assert inputs["detect_track"].fighters, "camera must not run before the pick"
            return CameraTrack(
                **_timeline(), focal_px=np.full(T, params.focal, np.float32), static=self.mode == "static"
            )

    @register("vipe")
    class CameraVipe(CameraBase):
        mode = "vipe"

    @register("static")
    class CameraStatic(CameraBase):
        mode = "static"

    class Pose:
        name, version, env, inputs, weight_hash = "pose3d", "1", "cpu", ("detect_track", "camera"), ""
        licences = (LicenceRecord("fake-gemx", "Apache-2.0", "Apache-2.0", "n/a", True),)

        class Params(BaseModel):
            pass

        def run(self, ctx, inputs, params):
            calls.append("pose3d")
            focal = float(inputs["camera"].focal_px[0])
            a = inputs["detect_track"].fighters["A"]
            rot = np.zeros((T, 2, 4), np.float32)
            rot[..., 3] = 1
            root = np.full((T, 3), focal / 1000.0 + a, np.float32)
            return BodyMotion(
                **_timeline(),
                skeleton="soma_provisional",
                fps=30.0,
                actors={"A": ActorMotion(rot=rot, root=root)},
            )

    class Smooth:
        name, version, env, inputs, licences, weight_hash = "smooth", "1", "cpu", ("pose3d",), (), ""

        class Params(BaseModel):
            strength: float = 0.5
            speed_factor: float = 1.0
            fail: str = ""  # "", "stage", "crash", "type"

        def run(self, ctx, inputs, params):
            calls.append("smooth")
            ctx.progress(0.5, "halfway")
            (ctx.work_dir / "qc.json").write_text(
                json.dumps({"summary": {"foot_speed": 0.01}, "per_frame": [1, 2]})
            )
            if params.fail == "stage":
                raise StageError("boom: smoothing diverged")
            if params.fail == "crash":
                raise ZeroDivisionError("division by zero")
            if params.fail == "type":
                return {"not": "a result"}
            m = inputs["pose3d"]
            a = m.actors["A"]
            return BodyMotion(
                **_timeline(),
                skeleton=m.skeleton,
                fps=30.0 / params.speed_factor,
                actors={"A": ActorMotion(rot=a.rot, root=a.root * params.strength)},
            )

    class Export:
        name, version, env, inputs, licences, weight_hash = "export", "1", "cpu", ("smooth",), (), ""

        class Params(BaseModel):
            preset: str = "blender"
            fps: float = 30.0
            speed_factor: float = 1.0

        def run(self, ctx, inputs, params):
            calls.append("export")
            out = ctx.work_dir / "exports"
            out.mkdir()
            (out / f"A_{params.preset}.bvh").write_text(f"HIERARCHY fps {params.fps}")
            return None

    classes = {"Ingest": Ingest, "DetectTrack": DetectTrack, "Pose": Pose, "Smooth": Smooth, "Export": Export}
    for c in classes.values():
        register("default")(c)
    return types.SimpleNamespace(calls=calls, Camera=CameraBase, CameraVipe=CameraVipe, **classes)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path_factory, monkeypatch):
    """Keep tests away from the real ~/.config/afv and ~/afv."""
    base = tmp_path_factory.mktemp("afvhome")
    monkeypatch.setenv("AFV_CONFIG_DIR", str(base / "config"))
    monkeypatch.setenv("AFV_HOME", str(base / "home"))
    monkeypatch.delenv("AFV_PROJECT", raising=False)
    monkeypatch.delenv("AFV_DEBUG", raising=False)
    return base


@pytest.fixture
def registry(monkeypatch):
    """Empty stage registry and no real stage modules."""
    monkeypatch.setattr(stage_mod, "_REGISTRY", {})
    monkeypatch.setattr(runner_mod, "STAGE_MODULES", ())
    return stage_mod._REGISTRY


@pytest.fixture
def fakes(registry):
    return _make_fakes()


@pytest.fixture
def video(tmp_path) -> Path:
    p = tmp_path / "src" / "fight.mp4"
    p.parent.mkdir()
    p.write_bytes(np.random.default_rng(0).bytes(4096))
    return p


@pytest.fixture
def project(tmp_path, video) -> Project:
    return Project.create(tmp_path / "fight.afv", video, title="Test fight", url="https://example.com/v")


@pytest.fixture
def shot(project):
    return project.add_shot(1.0, 2.0)


def pick(shot, fighters=(1, 0), extra=(2,)) -> None:
    shot.write_edit("fighters", {"fighters": list(fighters), "extra": list(extra)})


@pytest.fixture
def do_pick():
    return pick
