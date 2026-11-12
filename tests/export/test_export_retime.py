"""Round 2: the export stage retimes (ExportParams.fps, the shot's speed_factor) through
afv.motion.retime, with the contacts stage's ContactParams, and reads the hands_face result.

``speed_factor`` describes the FOOTAGE (speed vs real time: 0.25 = a 4x slow-motion replay); export
plays the motion back at real time, i.e. retimes by ``1 / speed_factor``."""

from __future__ import annotations

import json

import numpy as np
import pytest
from export_helpers import fighter_offsets, synthetic_motion
from pydantic import ValidationError
from typer.testing import CliRunner

from afv.cli import app
from afv.core import runner as runner_mod
from afv.core import stage as stage_mod
from afv.core.project import Project
from afv.core.skeletons import SOMA_PROVISIONAL
from afv.core.stage import Ctx, StageError, get_stage, register
from afv.core.types import ActorMotion, BodyMotion, Result
from afv.export import stage as export_stage
from afv.export.bvh import read_bvh
from afv.export.stage import ExportParams, ExportStage
from afv.motion.contacts import ContactParams


def _motion(n=31):
    sk = SOMA_PROVISIONAL
    rot, root = synthetic_motion(sk, n, 3, start=(0.0, 0.0))
    a = ActorMotion(rot=rot.astype(np.float32), root=root.astype(np.float32),
                    rest_offsets=fighter_offsets(sk, 3).astype(np.float32))
    return BodyMotion(frames=np.arange(n), timestamps=10.0 + np.arange(n) / 30.0, skeleton=sk.name,
                      fps=30.0, actors={"A": a})


def _run(tmp_path, inputs, **kw):
    ctx = Ctx(shot_dir=tmp_path / "s", work_dir=tmp_path / "w")
    res = get_stage("export", "default")().run(ctx, inputs, ExportParams(formats=["bvh"], **kw))
    meta = json.loads((ctx.work_dir / "exports" / "s_A.json").read_text())
    return res, meta, read_bvh(ctx.work_dir / "exports" / "s_A.bvh")


@pytest.mark.parametrize(
    ("n", "fps", "speed", "frames"),
    [
        (31, None, 1.0, 31),
        (31, 60.0, 1.0, 61),
        (31, 24.0, 1.0, 25),
        (31, 30.0, 0.25, 8),  # a 4x slow-motion replay: 1 s of footage is 0.25 s of real time
        (90, 30.0, 0.25, 23),  # 90 frames -> ~4x shorter
        (31, 60.0, 0.5, 31),  # 2x slow motion at 60 fps: 0.5 s
        (31, 30.0, 1.25, 38),  # undercranked (fast) footage plays longer at real time
    ],
)
def test_export_retimes_to_fps_and_real_time(tmp_path, n, fps, speed, frames):
    res, meta, bvh = _run(tmp_path, {"hands_face": _motion(n)}, fps=fps, speed_factor=speed)
    want_fps = fps or 30.0
    assert res.num_frames == frames and res.fps == want_fps
    assert meta["frame_count"] == frames and meta["fps"] == want_fps and meta["speed_factor"] == speed
    assert meta["provenance"]["playback_multiplier"] == pytest.approx(1.0 / speed)
    assert meta["provenance"]["retimed"] is (fps not in (None, 30.0) or speed != 1.0)
    # source in/out times stay in source seconds whatever the playback rate
    t0, t1 = meta["provenance"]["timestamps"]
    assert t0 == pytest.approx(10.0) and 10.0 + (n - 1) / 30 - 4 / 30 <= t1 <= 10.0 + (n - 1) / 30 + 1e-9
    assert bvh.frame_time == pytest.approx(1.0 / want_fps, rel=1e-6)
    assert len(bvh.motion) == frames


def test_export_passes_contact_params_and_playback_to_retime(tmp_path, monkeypatch):
    seen = {}
    real = export_stage.retime

    def spy(bm, out_fps, speed_factor=1.0, **kw):
        seen.update(kw, out_fps=out_fps, speed=speed_factor)
        return real(bm, out_fps, speed_factor, **kw)

    monkeypatch.setattr(export_stage, "retime", spy)
    cp = ContactParams(reach=0.9)
    _run(tmp_path, {"hands_face": _motion()}, fps=60.0, contact=cp, hold_contacts=False, speed_factor=0.25)
    assert seen["contact_params"] == cp and seen["hold_contacts"] is False and seen["out_fps"] == 60.0
    assert seen["speed"] == pytest.approx(4.0)  # retime's playback multiplier = 1 / footage speed


@pytest.mark.parametrize("exc", [FloatingPointError("overflow"), ZeroDivisionError("x"), MemoryError()])
def test_retime_arithmetic_and_memory_errors_are_stage_errors(tmp_path, monkeypatch, exc):
    def boom(*a, **kw):
        raise exc

    monkeypatch.setattr(export_stage, "retime", boom)
    with pytest.raises(StageError, match="retiming failed"):
        _run(tmp_path, {"hands_face": _motion()}, fps=60.0)


def test_export_prefers_hands_face_input(tmp_path):
    a, b = _motion(31), _motion(11)
    res, _, _ = _run(tmp_path, {"hands_face": b, "smooth": a})
    assert res.num_frames == 11


@pytest.mark.parametrize(
    "kw",
    [{"preset": "cryengine"}, {"fps": float("inf")}, {"fps": float("nan")}, {"fps": 1000.0}, {"fps": 0.0},
     {"speed_factor": float("inf")}, {"speed_factor": 0.0}],
)
def test_export_params_reject_bad_values(kw):
    with pytest.raises(ValidationError):
        ExportParams(**kw)


# ---------------------------------------------------------------------------
# Through the CLI: `afv shots add --speed` decides the exported frame count; bad settings are not saved
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_project(tmp_path, monkeypatch):
    """A project whose pipeline is: fake ingest -> fake hands_face (90 frames) -> the REAL export."""
    monkeypatch.setenv("AFV_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("AFV_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(stage_mod, "_REGISTRY", {})
    monkeypatch.setattr(runner_mod, "STAGE_MODULES", ())
    from pydantic import BaseModel

    class P(BaseModel):
        pass

    @register("default")
    class Ingest:
        name, version, env, inputs, licences, weight_hash, Params = "ingest", "1", "cpu", (), (), "", P

        def run(self, ctx, inputs, params):
            return Result(frames=np.arange(90), timestamps=np.arange(90) / 30.0)

    @register("default")
    class HandsFace:
        name, version, env, inputs, licences, weight_hash, Params = "hands_face", "1", "cpu", (), (), "", P

        def run(self, ctx, inputs, params):
            m = _motion(90)
            m.timestamps = np.arange(90) / 30.0
            return m

    register("default")(ExportStage)
    video = tmp_path / "fight.mp4"
    video.write_bytes(b"\0" * 1000)
    root = tmp_path / "p.afv"
    Project.create(root, video)
    return root


def _cli(*args, code=0):
    res = CliRunner().invoke(app, [str(a) for a in args], catch_exceptions=False)
    assert res.exit_code == code, res.output
    return res.output


@pytest.mark.parametrize(("speed", "frames"), [(1.0, 90), (0.25, 23), (0.5, 45)])
def test_shots_add_speed_sets_the_exported_frame_count(cli_project, speed, frames):
    _cli("shots", "add", "--start", "0", "--duration", "3", "--speed", speed, "-p", cli_project)
    _cli("run", "1", "--set", 'export.formats=["bvh"]', "-p", cli_project)
    shot = Project(cli_project).shot("1")
    assert shot.load_result("export").num_frames == frames
    assert len(read_bvh(shot.exports_dir / "001_A.bvh").motion) == frames


def test_bad_export_settings_are_rejected_before_saving(cli_project):
    _cli("shots", "add", "--start", "0", "--duration", "3", "-p", cli_project)
    shot = Project(cli_project).shot("1")
    before = (shot.dir / "shot.json").read_text()
    assert "unknown export preset" in _cli("export", "1", "--preset", "cryengine", "-p", cli_project, code=1)
    _cli("run", "1", "--set", "export.fps=1000", "-p", cli_project, code=1)
    _cli("run", "1", "--set", "export.fps=inf", "-p", cli_project, code=1)
    assert (shot.dir / "shot.json").read_text() == before
    assert shot.meta("export") is None  # nothing ran either
