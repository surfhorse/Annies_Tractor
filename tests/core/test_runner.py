from __future__ import annotations

import fcntl
import json
import os
import sys

import numpy as np
import pytest

from afv.core.config import ConfigError
from afv.core.project import ProjectError
from afv.core.runner import MissingStageError, RunError, Runner, load_stage_modules

ALL = ["detect_track", "camera:vipe", "pose3d", "smooth"]


def run(shot, fakes, **kw):
    to = kw.pop("to", "smooth")
    from_stage = kw.pop("from_stage", None)
    fakes.calls.clear()
    events: list[dict] = []
    rep = Runner(shot, report=events.append, **kw).run(to=to, from_stage=from_stage)
    rep.events = events
    return rep


@pytest.fixture
def done(shot, fakes, do_pick):
    """A shot run through smooth once."""
    do_pick(shot)
    rep = run(shot, fakes)
    assert rep.outcome == "ok", rep.message
    assert fakes.calls == ["ingest", *ALL]
    return shot


def test_pick_flow_stops_then_continues(shot, fakes, do_pick):
    rep = run(shot, fakes)
    assert rep.outcome == "needs_pick"
    assert fakes.calls == ["ingest", "detect_track"]
    assert rep.prompt_image and rep.prompt_image.endswith("prompt.jpg")
    assert f"afv pick {shot.id}" in rep.message
    assert any(e["event"] == "needs_pick" for e in rep.events)

    # Running again without a pick does not redo the detection.
    rep = run(shot, fakes)
    assert rep.outcome == "needs_pick" and fakes.calls == []

    do_pick(shot)
    rep = run(shot, fakes)
    assert rep.outcome == "ok", rep.message
    assert fakes.calls == ALL  # ingest reused
    assert shot.load_result("detect_track").fighters == {"A": 1, "B": 0}


def test_second_run_skips_everything(done, fakes):
    rep = run(done, fakes)
    assert rep.outcome == "ok" and fakes.calls == []
    assert [s.state for s in rep.stages] == ["fresh"] * 5


def test_changing_fighters_invalidates_detect_track_onward(done, fakes, do_pick):
    do_pick(done, fighters=(0, 1))
    rep = run(done, fakes)
    assert rep.outcome == "ok"
    assert fakes.calls == ALL
    reason = next(s.reason for s in rep.stages if s.name == "detect_track")
    assert "edits: fighters" in reason


def test_reformatting_an_edit_changes_nothing(done, fakes):
    p = done.edit_path("fighters")
    p.write_text(json.dumps(json.loads(p.read_text()), indent=8))
    assert run(done, fakes).stages[1].state == "fresh" and fakes.calls == []


def test_unrelated_edit_does_not_invalidate(done, fakes):
    done.write_edit("pins", {"frames": [3]})  # read by smooth only
    run(done, fakes)
    assert fakes.calls == ["smooth"]


def test_param_change_reruns_stage_and_downstream(done, fakes):
    run(done, fakes, params={"camera": {"focal": 1200.0}})
    assert fakes.calls == ["camera:vipe", "pose3d", "smooth"]


def test_unchanged_output_stops_invalidation(done, fakes):
    # 'note' changes camera's key but not its output: pose3d and smooth stay cached.
    run(done, fakes, params={"camera": {"note": "retry"}})
    assert fakes.calls == ["camera:vipe"]


def test_implementation_change(done, fakes):
    rep = run(done, fakes, impl={"camera": "static"})
    assert fakes.calls == ["camera:static", "pose3d"]  # pose3d output unchanged: smooth stays fresh
    assert done.meta("camera")["impl"] == "static"
    assert "impl" in next(s.reason for s in rep.stages if s.name == "camera")


def test_version_and_weight_hash_are_in_the_key(done, fakes, monkeypatch):
    monkeypatch.setattr(fakes.Smooth, "version", "2")
    run(done, fakes)
    assert fakes.calls == ["smooth"]
    monkeypatch.setattr(fakes.Pose, "weight_hash", "abc123")
    run(done, fakes)
    assert fakes.calls == ["pose3d"]  # same output, so smooth is still fresh


def test_from_forces_stage_and_later(done, fakes):
    rep = run(done, fakes, from_stage="pose3d")
    assert fakes.calls == ["pose3d", "smooth"]
    assert [s.state for s in rep.stages][-2:] == ["ran", "ran"]
    with pytest.raises(RunError, match="comes after"):
        Runner(done).run(to="camera", from_stage="smooth")


def test_shot_fields_reach_only_the_stages_that_read_them(done, fakes):
    info = done.info
    info.speed_factor = 0.25
    done.save_info(info)
    run(done, fakes)
    assert fakes.calls == ["smooth"]
    assert done.load_result("smooth").fps == pytest.approx(120.0)
    info.start = 1.5  # the shot interval is in every key
    done.save_info(info)
    run(done, fakes)
    assert fakes.calls == ["ingest", *ALL]


def test_stage_error_keeps_previous_results(done, fakes):
    before = done.meta("smooth")
    rep = run(done, fakes, params={"smooth": {"fail": "stage"}})
    assert rep.outcome == "failed"
    assert "boom: smoothing diverged" in rep.message and "smooth" in rep.message
    assert done.meta("smooth") == before
    assert done.load_result("smooth").actors["A"].root.shape == (10, 3)
    failed = list(done.work_root.glob("failed-smooth-*"))
    assert len(failed) == 1
    assert "boom" in (failed[0] / "logs" / "error.txt").read_text()
    assert "smooth failed" in (failed[0] / "logs" / "stage.log").read_text()
    # Back to the old settings: the stored result is still valid, nothing runs.
    assert run(done, fakes).outcome == "ok" and fakes.calls == []


def test_crash_and_bad_return_are_readable(done, fakes):
    rep = run(done, fakes, params={"smooth": {"fail": "crash"}})
    assert "crashed with ZeroDivisionError" in rep.message
    assert "Traceback" in (next(done.work_root.glob("failed-smooth-*")) / "logs" / "error.txt").read_text()
    rep = run(done, fakes, params={"smooth": {"fail": "type"}})
    assert "expected a Result" in rep.message


def test_upstream_failure_does_not_touch_downstream(done, fakes, monkeypatch):
    smooth_before = done.meta("smooth")

    def bad(self, ctx, inputs, params):
        raise RuntimeError("gpu fell over")

    monkeypatch.setattr(fakes.CameraVipe, "run", bad)
    rep = run(done, fakes, params={"camera": {"focal": 900.0}})
    assert rep.outcome == "failed" and [s.state for s in rep.stages][-1] == "failed"
    assert "pose3d" not in fakes.calls
    assert done.meta("smooth") == smooth_before


def test_failed_runs_are_pruned(done, fakes):
    for _ in range(5):
        run(done, fakes, params={"smooth": {"fail": "stage"}})
    assert len(list(done.work_root.glob("failed-smooth-*"))) == 3


def test_missing_result_file_reruns(done, fakes):
    (done.stage_dir("camera") / "result.npz").unlink()
    run(done, fakes)
    assert fakes.calls == ["camera:vipe"]  # same output as before, so pose3d stays fresh


def test_stage_folder_contents(done):
    d = done.stage_dir("ingest")
    assert (d / "proxy.txt").read_text() == "proxy"
    assert not (d / "tmp").exists()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["stage"] == "ingest" and meta["impl"] == "default" and len(meta["key"]) == 64
    assert "proxy.txt" in meta["files"] and "result.npz" in meta["files"]
    smeta = done.meta("smooth")
    assert smeta["qc"] == {"foot_speed": 0.01}
    assert smeta["params"]["strength"] == 0.5
    assert smeta["inputs"] == {"pose3d": done.meta("pose3d")["output_hash"]}
    assert done.meta("pose3d")["licences"][0]["component"] == "fake-gemx"
    for stage in ("ingest", "detect_track", "camera", "pose3d", "smooth"):
        assert [p.name for p in done.stage_root(stage).iterdir()] == ["v0001"]
    assert [p.name for p in done.work_root.iterdir()] == [".lock"]


def test_progress_events(shot, fakes, do_pick):
    do_pick(shot)
    rep = run(shot, fakes)
    prog = [e for e in rep.events if e["event"] == "progress"]
    assert prog == [
        {"event": "progress", "shot": shot.id, "stage": "smooth", "frac": 0.5, "message": "halfway"}
    ]
    kinds = [e["event"] for e in rep.events]
    assert kinds[0] == "plan" and kinds[-1] == "done"


def test_exports_are_published(done, fakes):
    rep = run(done, fakes, to="export", params={"export": {"preset": "unreal"}})
    assert rep.outcome == "ok" and fakes.calls == ["export"]
    assert (done.exports_dir / "A_unreal.bvh").is_file()
    assert rep.exports == ["A_unreal.bvh"]
    assert done.meta("export")["has_result"] is False
    run(done, fakes, to="export", params={"export": {"preset": "unreal"}})
    assert fakes.calls == []


def test_export_stage_files_outside_exports_are_published(done, fakes, monkeypatch):
    def run_export(self, ctx, inputs, params):
        (ctx.work_dir / "001_A.fbx").write_bytes(b"fbx")
        (ctx.work_dir / "notes.txt").write_text("not an export")
        return None

    monkeypatch.setattr(fakes.Export, "run", run_export)
    rep = run(done, fakes, to="export")
    assert rep.exports == ["001_A.fbx"]
    assert (done.exports_dir / "001_A.fbx").read_bytes() == b"fbx"
    assert not (done.exports_dir / "notes.txt").exists()


def test_large_qc_is_trimmed_in_meta(done, fakes, monkeypatch):
    def run_smooth(self, ctx, inputs, params):
        (ctx.work_dir / "qc.json").write_text(json.dumps({"flagged": list(range(50_000)), "n": 3}))
        return inputs["pose3d"]

    monkeypatch.setattr(fakes.Smooth, "run", run_smooth)
    run(done, fakes, params={"smooth": {"strength": 0.7}})
    assert done.meta("smooth")["qc"] == {"n": 3}
    assert (done.stage_dir("smooth") / "qc.json").is_file()


def test_unknown_setting_is_an_error(done, fakes):
    with pytest.raises(ConfigError, match="strenght"):
        Runner(done, params={"smooth": {"strenght": 1.0}}).run()
    with pytest.raises(ConfigError, match="strength"):
        Runner(done, params={"smooth": {"strength": "lots"}}).run()


def test_settings_layers(done, fakes, project):
    pinfo = project.info
    pinfo.settings.params["smooth"] = {"strength": 0.1}
    project.save_info(pinfo)
    run(done, fakes)
    assert done.meta("smooth")["params"]["strength"] == 0.1
    sinfo = done.info
    sinfo.settings.params["smooth"] = {"strength": 0.2}
    sinfo.settings.impl["camera"] = "static"
    done.save_info(sinfo)
    run(done, fakes)
    assert done.meta("smooth")["params"]["strength"] == 0.2 and done.meta("camera")["impl"] == "static"
    run(done, fakes, params={"smooth": {"strength": 0.3}})
    assert done.meta("smooth")["params"]["strength"] == 0.3


def test_missing_stage_is_clear(shot, fakes, registry):
    del registry["smooth"]
    with pytest.raises(MissingStageError) as e:
        Runner(shot).run()
    msg = str(e.value)
    assert "'smooth' has no implementation yet" in msg and "afv.motion.stages" in msg
    assert "implemented so far" in msg
    plan = Runner(shot).plan()
    assert {s.name: s.state for s in plan.stages}["smooth"] == "unavailable"


def test_several_implementations_need_a_choice(shot, fakes, registry):
    registry["camera"]["other"] = registry["camera"].pop("vipe")
    with pytest.raises(MissingStageError, match="several implementations"):
        Runner(shot).run()


def test_plan_reports_reasons(done, fakes):
    plan = Runner(done, params={"camera": {"focal": 1.0}}).plan()
    states = {s.name: (s.state, s.reason) for s in plan.stages}
    assert states["ingest"][0] == "fresh"
    assert states["camera"] == ("stale", "params: focal")
    assert states["pose3d"][0] == "pending"
    assert fakes.calls == ["ingest", *ALL]  # planning ran nothing


def test_plan_before_pick(shot, fakes):
    run(shot, fakes)
    plan = Runner(shot).plan()
    states = [s.state for s in plan.stages]
    assert states[:2] == ["fresh", "needs_pick"] and set(states[2:]) == {"blocked"}


def test_shot_lock(done, fakes):
    fd = os.open(done.work_root / ".lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RunError, match="already being run"):
            Runner(done).run()
    finally:
        os.close(fd)


def test_new_result_never_renames_the_old_one(done, fakes):
    old = done.stage_dir("camera")
    run(done, fakes, params={"camera": {"focal": 1100.0}})
    new = done.stage_dir("camera")
    assert (old.name, new.name) == ("v0001", "v0002") and not old.exists()
    assert done.meta("camera")["params"]["focal"] == 1100.0


def test_unfinished_version_folder_is_ignored(done, fakes):
    (done.stage_root("camera") / "v0009").mkdir()  # e.g. half-deleted, no meta.json
    rep = run(done, fakes)
    assert rep.outcome == "ok" and fakes.calls == []


def test_failed_commit_is_adopted_by_the_next_run(done, fakes, monkeypatch):
    from afv.core import runner as runner_mod

    real = runner_mod.commit_version

    def locked(work, root):
        raise PermissionError(13, "Permission denied (is a file in it open in another program?)")

    monkeypatch.setattr(runner_mod, "commit_version", locked)
    rep = run(done, fakes, params={"smooth": {"strength": 0.9}})
    assert rep.outcome == "failed" and "next 'afv run' adopts it" in rep.message
    assert done.meta("smooth")["params"]["strength"] == 0.5  # old result still current
    monkeypatch.setattr(runner_mod, "commit_version", real)
    rep = run(done, fakes, params={"smooth": {"strength": 0.9}})
    assert rep.outcome == "ok" and fakes.calls == []  # adopted, not re-run
    assert rep.stages[-1].reason == "adopted a finished run"
    assert done.meta("smooth")["params"]["strength"] == 0.9
    assert not [p for p in done.work_root.iterdir() if p.is_dir()]


def test_stale_work_folders_are_cleaned(done, fakes):
    junk = done.work_root / "smooth-20000101-000000-abcdef"
    junk.mkdir()
    kept = done.work_root / "smooth-20000101-000001-abcdef"  # a failure that could not be renamed
    kept.mkdir()
    (kept / ".afv_failed").write_text("boom")
    run(done, fakes)
    assert not junk.exists() and kept.exists()


def test_consumers_get_the_stored_float32_data(shot, fakes, do_pick, monkeypatch):
    seen = []
    pose_run = fakes.Pose.run

    def pose_float64(self, ctx, inputs, params):
        res = pose_run(self, ctx, inputs, params)
        res.actors["A"].root = res.actors["A"].root.astype(np.float64) + 1e-9
        return res

    def smooth_spy(self, ctx, inputs, params):
        root = inputs["pose3d"].actors["A"].root
        seen.append((root.dtype, root.copy()))
        root += 100.0  # a consumer that mutates its input must not affect anyone else
        return inputs["pose3d"]

    monkeypatch.setattr(fakes.Pose, "run", pose_float64)
    monkeypatch.setattr(fakes.Smooth, "run", smooth_spy)
    do_pick(shot)
    run(shot, fakes)
    run(shot, fakes, from_stage="smooth")
    (d1, r1), (d2, r2) = seen
    assert d1 == d2 == np.float32
    np.testing.assert_array_equal(r1, r2)
    stored = shot.load_result("pose3d").actors["A"].root
    np.testing.assert_array_equal(stored, r1)  # what smooth saw is what is stored
    run(shot, fakes)
    assert fakes.calls == []  # same keys in a later run: nothing invalidated


def test_corrupt_result_is_rerun(done, fakes):
    p = done.stage_dir("camera") / "result.npz"
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])  # truncated
    plan = {s.name: (s.state, s.reason) for s in Runner(done).plan().stages}
    assert plan["camera"] == ("stale", "stored result is missing or damaged")
    run(done, fakes)
    assert fakes.calls == ["camera:vipe"]
    p = done.stage_dir("pose3d") / "result.npz"
    raw = bytearray(p.read_bytes())
    raw[-200] ^= 0xFF  # bit flip: still a zip maybe, but the content hash differs
    p.write_bytes(bytes(raw))
    run(done, fakes)
    assert fakes.calls == ["pose3d"]


def test_ingest_always_runs(shot, fakes, do_pick):
    assert fakes.DetectTrack.inputs == ()
    do_pick(shot)
    run(shot, fakes, to="detect_track")
    assert fakes.calls == ["ingest", "detect_track"]


def test_replaced_source_is_refused_until_rehashed(done, fakes, project, video):
    data = bytearray(video.read_bytes())
    data[0] ^= 0xFF  # same size, new content
    video.write_bytes(bytes(data))
    st = video.stat()
    os.utime(video, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    with pytest.raises(ProjectError, match="--rehash"):
        Runner(done).run()
    assert "--rehash" in Runner(done).plan().message
    project.rehash()
    rep = run(project.shot(done.id), fakes)
    assert rep.outcome == "ok" and fakes.calls == ["ingest", *ALL]  # new sha256 in every key


def test_stale_exports_are_removed_and_missing_ones_republished(done, fakes):
    run(done, fakes, to="export", params={"export": {"preset": "unreal"}})
    assert (done.exports_dir / "A_unreal.bvh").is_file()
    run(done, fakes, to="export", params={"export": {"preset": "godot"}})
    assert sorted(p.name for p in done.exports_dir.iterdir()) == ["A_godot.bvh"]
    (done.exports_dir / "A_godot.bvh").unlink()
    rep = run(done, fakes, to="export", params={"export": {"preset": "godot"}})
    assert fakes.calls == [] and rep.outcome == "ok"
    assert (done.exports_dir / "A_godot.bvh").is_file()


def test_export_publish_failure_is_a_stage_failure(done, fakes, monkeypatch):
    from afv.core import runner as runner_mod

    run(done, fakes, to="export", params={"export": {"preset": "unreal"}})
    before = done.meta("export")

    def denied(src, dst):
        raise PermissionError(13, "Permission denied", str(dst))

    monkeypatch.setattr(runner_mod, "atomic_copy", denied)
    rep = run(done, fakes, to="export", params={"export": {"preset": "godot"}})
    assert rep.outcome == "failed" and "could not publish exports" in rep.message
    assert "open in another program" in rep.message
    assert done.meta("export") == before


def test_load_stage_modules_is_tolerant(tmp_path, monkeypatch):
    pkg = tmp_path / "afvfake_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "broken.py").write_text("import afvfake_not_installed_dep\n")
    (pkg / "raises.py").write_text("raise ValueError('bad config at import')\n")
    (pkg / "fine.py").write_text("X = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    st = load_stage_modules(
        [
            "afvfake_pkg.missing",
            "afvfake_pkg.broken",
            "afvfake_pkg.raises",
            "afvfake_pkg.fine",
            "afvfake_nope.stage",
        ]
    )
    assert st["afvfake_pkg.missing"] == "not built yet"
    assert st["afvfake_nope.stage"] == "not built yet"
    assert (
        st["afvfake_pkg.broken"].startswith("import failed")
        and "afvfake_not_installed_dep" in st["afvfake_pkg.broken"]
    )
    assert "ValueError: bad config at import" in st["afvfake_pkg.raises"]
    assert st["afvfake_pkg.fine"] == "ok"
    for m in [m for m in sys.modules if m.startswith("afvfake")]:
        del sys.modules[m]
