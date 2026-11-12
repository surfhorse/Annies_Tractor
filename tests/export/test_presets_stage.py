from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from export_helpers import fighter_offsets, synthetic_motion

from afv.core.project import ShotContext
from afv.core.skeletons import SOMA_PROVISIONAL
from afv.core.stage import Ctx, StageError, get_stage
from afv.core.types import ActorMotion, BodyMotion, Take, load_result, save_result
from afv.export import blender as bl
from afv.export import stage as stage_mod
from afv.export.bvh import bvh_fk, read_bvh
from afv.export.exporter import take_motion
from afv.export.presets import PRESETS
from afv.export.retarget import mapped_source_positions
from afv.export.stage import ExportManifest, ExportParams


def test_presets_match_spec_table():
    assert set(PRESETS) == {"unity_humanoid", "unity_generic", "unreal", "godot", "blender"}
    assert not PRESETS["unity_humanoid"].root_bone and not PRESETS["blender"].root_bone
    for n in ("unity_generic", "unreal", "godot"):
        assert PRESETS[n].root_bone and PRESETS[n].root_motion == "extracted"
    assert PRESETS["unreal"].unit_scale == 0.01
    for p in PRESETS.values():
        fbx = p.fbx
        assert fbx["axis_forward"] == "-Z" and fbx["axis_up"] == "Y"
        assert fbx["add_leaf_bones"] is False
        assert fbx["bake_anim_simplify_factor"] == 0.0
        assert fbx["use_armature_deform_only"] is True
        assert fbx["bake_anim_use_all_actions"] is False
        assert fbx["primary_bone_axis"] == "Y" and fbx["secondary_bone_axis"] == "X"
        assert fbx["apply_scale_options"] in ("FBX_SCALE_ALL", "FBX_SCALE_NONE", "FBX_SCALE_UNITS")
        assert p.glb["export_leaf_bone"] is False and p.glb["export_optimize_animation_size"] is False
        assert "extracted" not in p.allowed_root_motion or p.root_bone


def _body_motion(n_frames=24):
    sk = SOMA_PROVISIONAL
    actors = {}
    for role, seed, start in (("A", 1, (0.0, 0.0)), ("B", 2, (1.0, 2.0))):
        rot, root = synthetic_motion(sk, n_frames, seed, start=start)
        actors[role] = ActorMotion(rot=rot.astype(np.float32), root=root.astype(np.float32),
                                   rest_offsets=fighter_offsets(sk, seed).astype(np.float32))
    return BodyMotion(frames=np.arange(n_frames), timestamps=np.arange(n_frames) / 30.0,
                      skeleton=sk.name, fps=30.0, actors=actors)


def test_export_stage_bvh_only(tmp_path):
    cls = get_stage("export", "default")
    stage = cls()
    shot = tmp_path / "shot_007"
    ctx = Ctx(shot_dir=shot, work_dir=tmp_path / "work")
    motion = _body_motion()
    res = stage.run(ctx, {"smooth": motion}, ExportParams(preset="unreal", formats=["bvh"]))
    assert isinstance(res, ExportManifest)
    assert res.files["A"] == {"bvh": "exports/shot_007_A.bvh", "json": "exports/shot_007_A.json"}
    assert sorted(p.name for p in (ctx.work_dir / "exports").iterdir()) == [
        "shot_007_A.bvh", "shot_007_A.json", "shot_007_B.bvh", "shot_007_B.json"]
    assert res.root_motion == "extracted" and set(res.files) == {"A", "B"}
    assert res.rig_bones[0] == "root"
    # each fighter in its own file, sharing one world origin with its own bone lengths
    for role, a in motion.actors.items():
        d = read_bvh(ctx.work_dir / res.files[role]["bvh"])
        pos = bvh_fk(d)
        take = Take(SOMA_PROVISIONAL.name, 30.0, 1.0, a.rot, a.root, root_motion="extracted", role=role,
                    rest_offsets=a.rest_offsets)
        m = take_motion(take, PRESETS["unreal"])
        want = mapped_source_positions(m.rig, np.asarray(a.rot, np.float64), np.asarray(a.root, np.float64))
        assert np.abs(pos[:, 1:] - want[:, 1:]).max() < 1e-5
        assert (ctx.work_dir / res.files[role]["json"]).is_file()
    save_result(res, tmp_path / "r.npz")
    back = load_result(tmp_path / "r.npz")
    assert isinstance(back, ExportManifest) and back.files == res.files


def test_export_stage_errors(tmp_path):
    stage = get_stage("export", "default")()
    ctx = Ctx(shot_dir=tmp_path / "s", work_dir=tmp_path / "w")
    motion = _body_motion(5)
    with pytest.raises(ValueError, match="unknown export preset"):  # rejected when the params are made
        ExportParams(preset="cryengine", formats=["bvh"])
    with pytest.raises(StageError, match="preset"):  # and by the stage itself, for unvalidated params
        stage.run(ctx, {"smooth": motion}, ExportParams.model_construct(preset="cryengine", formats=["bvh"]))
    with pytest.raises(StageError, match="root motion"):
        stage.run(ctx, {"smooth": motion}, ExportParams(preset="unity_humanoid", root_motion="extracted",
                                                        formats=["bvh"]))
    with pytest.raises(StageError, match="roles"):
        stage.run(ctx, {"smooth": motion}, ExportParams(formats=["bvh"], roles=["C"]))
    stage.run(ctx, {"smooth": motion}, ExportParams(formats=["bvh"], fps=30.0))  # = motion rate: fine


def test_export_stage_declares_speed_factor_and_fills_provenance(tmp_path, monkeypatch):
    cls = get_stage("export", "default")
    assert cls.shot_fields == ("speed_factor",)
    sc = ShotContext(shot_id="s1", shot_dir=str(tmp_path / "s1"), project_dir=str(tmp_path),
                     source_path="/videos/fight.mp4", source_sha256="ab" * 32, source_title="Title",
                     source_url="https://example.com/v", start=12.5, duration=3.0, speed_factor=0.5)
    monkeypatch.setattr(stage_mod, "shot_context", lambda shot_dir: sc)
    ctx = Ctx(shot_dir=tmp_path / "s1", work_dir=tmp_path / "w")
    # the runner injects the shot's speed_factor into the params (shot_fields); here it is given directly
    res = cls().run(ctx, {"smooth": _body_motion(6)}, ExportParams(formats=["bvh"], roles=["A"],
                                                                   speed_factor=0.5))
    assert res.num_frames == 3  # 6 frames of 2x slow-motion footage play in half the time: 5/2 -> 2, +1
    meta = json.loads((ctx.work_dir / "exports" / "s1_A.json").read_text())
    prov = meta["provenance"]
    assert prov["source_path"] == "/videos/fight.mp4" and prov["source_sha256"] == "ab" * 32
    assert (prov["source_title"], prov["source_url"], prov["start"], prov["duration"]) == (
        "Title", "https://example.com/v", 12.5, 3.0)
    assert prov["speed_factor"] == 0.5 and prov["speed_factor_applied"] is True and prov["retimed"] is True
    assert meta["speed_factor"] == 0.5 and meta["frame_count"] == 3


def test_export_stage_without_project_still_records_shot(tmp_path):
    ctx = Ctx(shot_dir=tmp_path / "loose", work_dir=tmp_path / "w")
    get_stage("export", "default")().run(ctx, {"smooth": _body_motion(4)}, ExportParams(formats=["bvh"]))
    meta = json.loads((ctx.work_dir / "exports" / "loose_A.json").read_text())
    assert meta["provenance"]["shot_dir"] == str(tmp_path / "loose")


def test_run_blender_retries_interop_launch_failure(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if len(calls) == 1:  # WSL interop could not launch the Windows process
            err = b"<3>WSL ERROR: UtilAcceptVsock:273: accept4 failed 110"
            return subprocess.CompletedProcess(cmd, 1, b"", err)
        (tmp_path / "job" / "afv_export_result.json").write_text('{"ok": true}')
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(bl, "host_path", lambda p, blender: str(p))
    monkeypatch.setattr(bl.subprocess, "run", fake_run)
    assert bl.run_blender("afv_export.py", {}, tmp_path / "job", blender=Path("blender.exe")) == {"ok": True}
    assert len(calls) == 2

    calls.clear()

    def always_fails(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, b"Error: script failed", b"")

    monkeypatch.setattr(bl.subprocess, "run", always_fails)
    with pytest.raises(bl.BlenderError, match="script failed"):
        bl.run_blender("afv_export.py", {}, tmp_path / "job2", blender=Path("blender.exe"))
    assert len(calls) == 1  # a real Blender failure is not retried
