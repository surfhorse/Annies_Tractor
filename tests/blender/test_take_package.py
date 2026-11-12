"""Take package (``afv take``) and the add-on's bpy-free parts (reader, issue navigation, WSL bridge)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from take_helpers import addon_module, synthetic_shot

from afv import take as tk
from afv.core.skeletons import get_skeleton
from afv.core.types import Take
from afv.export.exporter import blender_arrays, take_motion
from afv.export.presets import get_preset
from afv.motion.rotations import qconj, qrotate, wxyz_to_xyzw

takeio = addon_module("takeio")
wsl = addon_module("wsl")


@pytest.fixture
def package(tmp_path):
    bm, cam = synthetic_shot()
    inp = tk.TakeInputs(motion=bm, camera=cam, shot_id="007", speed_factor=0.5, project_dir="/x/proj")
    path = tk.write_take(tmp_path / "take", inp, tk.RigSettings(preset="unity_humanoid"))
    return path, bm, cam


def test_package_files_and_schema(package):
    path, bm, _ = package
    meta = json.loads(path.read_text())
    assert meta["format"] == "afv_take" and meta["version"] == tk.TAKE_VERSION
    assert meta["take_id"] == "proj:007"
    assert meta["shot"] == {"id": "007", "fps": 30.0, "frame_start": 0, "frame_end": 23, "frame_count": 24,
                            "speed_factor": 0.5, "timestamps": [2.0, pytest.approx(2.0 + 23 / 30)]}
    assert [f["role"] for f in meta["fighters"]] == ["A", "B"]
    assert {p["name"] for p in meta["presets"]} >= {"unity_humanoid", "unreal", "godot", "blender"}
    assert meta["camera"]["world_shift"] == bm.world_shift
    names = tk.package_files(meta)
    assert len(names) == 5
    assert {n.split(".")[0] for n in names} == {"camera", "timeline", "qc", "007_A", "007_B"}
    assert {p.name for p in path.parent.iterdir()} == names | {"take.json"}
    assert meta["video"] is None
    assert meta["stale"] == []


def test_fighter_arrays_are_the_exporters(package):
    """Same rig, rest and per-frame arrays the FBX export gets (so the take matches the exports)."""
    path, bm, _ = package
    meta = json.loads(path.read_text())
    sk = get_skeleton("soma77")
    for f in meta["fighters"]:
        a = bm.actors[f["role"]]
        take = Take(skeleton="soma77", fps=30.0, speed_factor=1.0, rot=a.rot, root=a.root,
                    rest_offsets=a.rest_offsets)
        m = take_motion(take, get_preset("unity_humanoid"), skeleton=sk)
        want = blender_arrays(m)
        with np.load(path.parent / f["npz"]) as z:
            for k, v in want.items():
                np.testing.assert_allclose(z[k], v, atol=1e-12, err_msg=k)
            np.testing.assert_allclose(z["rot"], m.rot)
            assert z["plant"].shape == (24, len(f["plant_channels"]))
            assert z["plant"][:12, f["plant_channels"].index("LeftFoot")].all()
        assert [b["name"] for b in f["bones"]] == list(m.rig.names)
        assert f["hips"] == "Hips" and f["root_bone"] is None
        assert len(f["bones"]) == 65


def test_camera_in_bodies_world_projects_like_the_pipeline(package):
    """camera.npz (world shift applied, wxyz) through a plain pinhole = afv.qc.camera.project."""
    path, bm, cam = package
    meta = json.loads(path.read_text())
    with np.load(path.parent / meta["camera"]["npz"]) as z:
        pos, rot, focal = z["pos"], wxyz_to_xyzw(z["rot"]), z["focal_px"]
    np.testing.assert_allclose(pos, cam.cam_pos + np.asarray(bm.world_shift), atol=1e-6)
    pts = np.random.default_rng(3).normal(size=(24, 10, 3)) * 0.5 + [0.5, 1.0, 0.0]
    pc = qrotate(qconj(rot)[:, None], pts - pos[:, None])
    u = meta["camera"]["cx"] + focal[:, None] * pc[..., 0] / -pc[..., 2]
    v = meta["camera"]["cy"] - focal[:, None] * pc[..., 1] / -pc[..., 2]
    ref = tk.project_points(cam, pts, bm.frames, bm.world_shift)
    np.testing.assert_allclose(np.stack([u, v], -1), ref, atol=1e-6)


def test_in_place_setting_shows_world(tmp_path):
    bm, cam = synthetic_shot(n_frames=6)
    path = tk.write_take(tmp_path, tk.TakeInputs(motion=bm, camera=cam),
                         tk.RigSettings(preset="unity_humanoid", root_motion="in_place"))
    assert json.loads(path.read_text())["rig"]["root_motion"] == "world"


def test_unknown_preset_is_a_take_error(tmp_path):
    bm, cam = synthetic_shot(n_frames=4)
    with pytest.raises(tk.TakeError, match="unknown export preset"):
        tk.write_take(tmp_path, tk.TakeInputs(motion=bm, camera=cam), tk.RigSettings(preset="nope"))


def test_root_motion_validated_like_the_export_stage(tmp_path):
    bm, cam = synthetic_shot(n_frames=4)
    with pytest.raises(tk.TakeError, match="does not support root motion 'extracted'"):
        tk.write_take(tmp_path, tk.TakeInputs(motion=bm, camera=cam),
                      tk.RigSettings(preset="blender", root_motion="extracted"))
    assert tk.RigSettings(preset="unreal").mode() == "extracted"


def test_reader_loads_package_without_bpy(package):
    path, _, _ = package
    t = takeio.load_take(path.parent.parent / "take")  # a take folder works too
    assert t.take_id == "proj:007" and t.fps == 30.0
    assert t.frame_start == 0 and t.frame_end == 23
    assert [f.role for f in t.fighters] == ["A", "B"]
    assert t.fighters[0].bones[0] == ("Hips", None)
    assert t.camera["pos"].shape == (24, 3) and t.camera["width"] == 640
    assert t.video_path() is None
    rec = t.record()
    assert rec["roles"] == ["A", "B"] and rec["take_json"] == str(path)
    assert takeio.locate(path)["project_wsl"] == "/x/proj"
    assert all(s["kind"] in tk.QC_KINDS for s in t.spans)


def test_reader_rejects_other_versions(package):
    path, _, _ = package
    meta = json.loads(path.read_text())
    meta["version"] = 99
    path.write_text(json.dumps(meta))
    with pytest.raises(takeio.TakeError, match="not supported"):
        takeio.load_take(path)
    with pytest.raises(takeio.TakeError, match="afv take"):
        takeio.find_take_json(path.parent / "missing")


def test_rewrite_commits_and_removes_only_the_previous_package(tmp_path):
    """New NPZ names every write; take.json is the commit; only files the old take.json named go."""
    bm, cam = synthetic_shot(n_frames=4)
    first = tk.write_take(tmp_path, tk.TakeInputs(motion=bm, camera=cam, shot_id="1"))
    old = tk.package_files(json.loads(first.read_text()))
    (tmp_path / "take.blend").write_bytes(b"x")
    (tmp_path / "notes.npz").write_bytes(b"mine")  # not the take's: kept
    second = tk.write_take(tmp_path, tk.TakeInputs(motion=bm, camera=cam, shot_id="1", name="renamed"))
    new = tk.package_files(json.loads(second.read_text()))
    assert not old & new
    assert {p.name for p in tmp_path.iterdir()} == new | {"take.json", "take.blend", "notes.npz"}
    assert sorted(n.split(".")[0] for n in new if "_A." in n) == ["renamed_A"]
    takeio.load_take(second)


def test_non_take_folder_is_refused(tmp_path):
    bm, cam = synthetic_shot(n_frames=4)
    (tmp_path / "data.npz").write_bytes(b"precious")
    with pytest.raises(tk.TakeError, match="not empty and holds no take"):
        tk.write_take(tmp_path, tk.TakeInputs(motion=bm, camera=cam))
    assert (tmp_path / "data.npz").read_bytes() == b"precious"
    (tmp_path / "data.npz").unlink()
    (tmp_path / "take.json").write_text('{"something": "else"}')
    with pytest.raises(tk.TakeError, match="not an afv take"):
        tk.write_take(tmp_path, tk.TakeInputs(motion=bm, camera=cam))


def test_failed_write_keeps_the_previous_package(tmp_path, monkeypatch):
    bm, cam = synthetic_shot(n_frames=4)
    first = tk.write_take(tmp_path, tk.TakeInputs(motion=bm, camera=cam, shot_id="1"))
    before = {p.name for p in tmp_path.iterdir()}

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(tk, "atomic_write_json", boom)
    with pytest.raises(OSError, match="disk full"):
        tk.write_take(tmp_path, tk.TakeInputs(motion=bm, camera=cam, shot_id="1"))
    assert {p.name for p in tmp_path.iterdir()} == before
    takeio.load_take(first)


def test_plant_channels_are_the_contact_channels_only(package):
    """qc metrics also has foot_contact_<foot> (any channel of a foot): not a plant channel."""
    path, _, _ = package
    for f in json.loads(path.read_text())["fighters"]:
        assert f["plant_channels"] == ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]


class _StubShot:
    def __init__(self, metas):
        self.metas = metas

    def meta(self, name):
        return self.metas.get(name)


def test_stale_results_walks_the_input_chain():
    metas = {
        "hands_face": {"inputs": {"smooth": "s1"}, "output_hash": "h1"},
        "smooth": {"inputs": {"contacts": "c1"}, "output_hash": "s2"},  # re-run since hands_face
        "contacts": {"inputs": {"pair": "p1", "camera": "cam0"}, "output_hash": "c1"},
        "pair": {"inputs": {"camera": "cam1"}, "output_hash": "p1"},
        "camera": {"inputs": {}, "output_hash": "cam1"},
    }
    assert tk.stale_results(_StubShot(metas), "hands_face") == [
        "contacts was made from an older camera result", "hands_face was made from an older smooth result"]
    metas["smooth"]["output_hash"] = "s1"
    metas["contacts"]["inputs"]["camera"] = "cam1"
    assert tk.stale_results(_StubShot(metas), "hands_face") == []


def test_reader_flags_sources_newer_than_the_take(package):
    path, _, _ = package
    shot = path.parent.parent / "shots" / "007"
    (shot / "stages" / "hands_face" / "v0003").mkdir(parents=True)
    (shot / "shot.json").write_text("{}")
    (shot / "stages" / "hands_face" / "v0003" / "meta.json").write_text('{"output_hash": "new"}')
    (shot / "stages" / "hands_face" / "v0002").mkdir()
    (shot / "stages" / "hands_face" / "v0002" / "meta.json").write_text('{"output_hash": "old"}')
    take_dir = shot / "take"
    path.parent.rename(take_dir)
    meta = json.loads((take_dir / "take.json").read_text())
    meta["sources"] = {"motion": {"stage": "hands_face", "version_dir": "v0002", "output_hash": "old"}}
    meta["shot_rel"] = ".."
    (take_dir / "take.json").write_text(json.dumps(meta))
    t = takeio.load_take(take_dir)
    loc = takeio.locate(t.path)
    assert loc["shot_dir"] == shot.resolve() and loc["project_dir"] == shot.resolve().parent.parent
    assert takeio.stale_sources(t) == ["hands_face has a newer result (v0003) than the take (v0002)"]
    meta["sources"]["motion"]["output_hash"] = "new"
    meta["stale"] = ["hands_face was made from an older smooth result"]
    (take_dir / "take.json").write_text(json.dumps(meta))
    t = takeio.load_take(take_dir)
    assert takeio.stale_sources(t) == [] and takeio.pipeline_warnings(t) == meta["stale"]


# --- QC spans -------------------------------------------------------------------------------------
def test_qc_spans_merge_severity_and_scaling():
    frames = np.arange(100, 120)
    slide = np.zeros(20)
    slide[[2, 3, 6]] = [0.06, 0.07, 0.2]  # gap of 2 frames: one span, bad peak
    slide[15] = 0.06
    conf = np.full(20, 0.9)
    conf[10:12] = [0.3, 0.1]
    reproj = np.full(20, np.nan)
    reproj[0] = 30.0  # > 25 px at 720p but < 25 * 1440/720 = 50 px at 1440p
    curves = {"A": {"slide": slide, "low_conf": conf, "reproj": reproj}}
    spans = tk.qc_spans(curves, frames, image_height=1440)
    got = [(s["kind"], s["start"], s["end"], s["severity"]) for s in spans]
    assert got == [("slide", 102, 106, "bad"), ("low_conf", 110, 111, "bad"), ("slide", 115, 115, "warn")]
    assert spans[0]["label"] == "A slide 20cm/s !" and spans[0]["peak"] == pytest.approx(20.0)
    spans720 = tk.qc_spans({"A": {"reproj": reproj}}, frames, image_height=720)
    assert [(s["start"], s["severity"]) for s in spans720] == [(100, "warn")]


def test_synthetic_qc_flags_the_low_confidence_frames(package):
    path, _, _ = package
    spans = json.loads(path.read_text())["qc"]["spans"]
    low = [s for s in spans if s["kind"] == "low_conf"]
    assert {(s["role"], s["start"], s["end"]) for s in low} == {("A", 8, 10), ("B", 8, 10)}


def test_next_span_wraps():
    spans = [{"start": 5, "end": 6}, {"start": 20, "end": 22}, {"start": 12, "end": 12}]
    assert takeio.next_span(spans, 0)["start"] == 5
    assert takeio.next_span(spans, 5)["start"] == 12
    assert takeio.next_span(spans, 25)["start"] == 5
    assert takeio.next_span(spans, 12, -1)["start"] == 5
    assert takeio.next_span(spans, 3, -1)["start"] == 20
    assert takeio.next_span([], 3) is None
    assert takeio.spans_at(spans, 21) == [spans[1]]


# --- WSL bridge -----------------------------------------------------------------------------------
@pytest.mark.parametrize(("win", "posix"), [
    (r"D:\afv\projects\x", "/mnt/d/afv/projects/x"),
    ("C:/Users/me/", "/mnt/c/Users/me"),
    ("E:\\", "/mnt/e"),
    (r"\\wsl.localhost\Ubuntu\home\me\p", "/home/me/p"),
    (r"\\wsl$\Ubuntu\home\me", "/home/me"),
    ("/mnt/d/already", "/mnt/d/already"),
])
def test_wsl_paths(win, posix):
    assert wsl.to_wsl_path(win) == posix


def test_windows_paths():
    assert wsl.to_windows_path("/mnt/d/a/b") == "D:\\a\\b"
    assert wsl.to_windows_path("/home/me/x", "Ubuntu") == "\\\\wsl.localhost\\Ubuntu\\home\\me\\x"


def test_afv_argv_quotes_and_refuses_dollars():
    argv = wsl.afv_argv(["export", "001", "--project", "/mnt/d/my proj", "--json"])
    assert argv[:6] == ["wsl.exe", "-d", "Ubuntu", "-e", "bash", "-lc"]
    assert argv[6] == ("echo AFVPID:$$; exec ~/afv/afv-env/bin/afv export 001 "
                       "--project '/mnt/d/my proj' --json")
    with pytest.raises(ValueError, match=r"\$"):
        wsl.afv_argv(["take", "$HOME"])


def test_decode_line_handles_utf16_and_bom():
    assert wsl.decode_line("Wsl/Service: no distro\r\n".encode("utf-16-le")) == "Wsl/Service: no distro"
    assert wsl.decode_line(b"\xef\xbb\xbfok\n") == "ok"


def _wait(job):
    events = []
    for _ in range(400):
        events += job.drain()
        if job.done:
            break
        job.join(0.05)
    return events


def test_job_review_pause_is_parsed():
    lines = [{"event": "tracking_review", "message": "2 hints: id swap at 40", "hints": [{"kind": "swap"}],
              "review_video": "/x/review.mp4"},
             {"event": "result", "outcome": "needs_review", "exports": []}]
    code = "import json,sys\n" + "".join(f"print(json.dumps({e!r}))\n" for e in lines) + "sys.exit(3)"
    job = wsl.Job([sys.executable, "-c", code], "t").start()
    _wait(job)
    assert job.done and not job.ok and job.returncode == 3
    assert job.review() == {"message": "2 hints: id swap at 40", "hints": [{"kind": "swap"}],
                            "review_video": "/x/review.mp4"}


def test_job_runs_in_background_and_parses_json_lines():
    ev = "{'event': 'result', 'outcome': 'ok', 'exports': ['a.fbx']}"
    code = f"import json; print('AFVPID:4242'); print('hello'); print(json.dumps({ev}))"
    job = wsl.Job([sys.executable, "-c", code], "t").start()
    events = []
    for _ in range(200):
        events += job.drain()
        if job.done:
            break
        job._thread.join(0.05)
    assert job.ok, job.failure()
    assert job.finished and job.linux_pid == 4242
    assert events[0] == {"event": "text", "message": "hello"}
    assert job.result()["exports"] == ["a.fbx"]


def test_job_failure_message():
    code = "import json,sys; print(json.dumps({'event':'error','message':'boom'})); sys.exit(1)"
    job = wsl.Job([sys.executable, "-c", code], "t").start()
    job._thread.join(30)
    job.drain()
    assert job.done and not job.ok and job.failure() == "boom"
    missing = wsl.Job(["/nonexistent/afv-binary"], "t").start()
    missing._thread.join(30)
    missing.drain()
    assert not missing.ok and "cannot start" in missing.failure()


def test_failure_before_the_commit_leaves_no_new_files(tmp_path, monkeypatch):
    bm, cam = synthetic_shot(n_frames=4)
    tk.write_take(tmp_path, tk.TakeInputs(motion=bm, camera=cam, shot_id="1"))
    before = {p.name for p in tmp_path.iterdir()}

    def boom(*a, **k):
        raise RuntimeError("camera broke")

    monkeypatch.setattr(tk, "camera_in_world", boom)  # after the fighters' NPZs were written
    with pytest.raises(RuntimeError, match="camera broke"):
        tk.write_take(tmp_path, tk.TakeInputs(motion=bm, camera=cam, shot_id="1"))
    assert {p.name for p in tmp_path.iterdir()} == before


def test_interrupt_just_after_the_commit_keeps_the_new_take(tmp_path, monkeypatch):
    bm, cam = synthetic_shot(n_frames=4)
    tk.write_take(tmp_path, tk.TakeInputs(motion=bm, camera=cam, shot_id="1"))
    real = tk.atomic_write_json

    def commit_then_interrupt(path, obj):
        real(path, obj)
        raise KeyboardInterrupt

    monkeypatch.setattr(tk, "atomic_write_json", commit_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        tk.write_take(tmp_path, tk.TakeInputs(motion=bm, camera=cam, shot_id="1"))
    takeio.load_take(tmp_path)  # every file the new take.json names is there


def test_orphans_of_failed_writes_are_cleaned_after_a_commit(tmp_path):
    bm, cam = synthetic_shot(n_frames=4)
    for n in ("camera.deadbeef.npz", "1_A.deadbeef.npz", "other.deadbeef.npz"):
        (tmp_path / n).write_bytes(b"x")
    (tmp_path / "take.blend").write_bytes(b"x")  # a take folder without take.json: our artefacts only
    path = tk.write_take(tmp_path, tk.TakeInputs(motion=bm, camera=cam, shot_id="1"))
    names = {p.name for p in tmp_path.iterdir()}
    assert "camera.deadbeef.npz" not in names and "1_A.deadbeef.npz" not in names
    assert {"other.deadbeef.npz", "take.blend"} <= names
    assert names - {"other.deadbeef.npz", "take.blend", "take.json"} == tk.package_files(
        json.loads(path.read_text()))


def test_take_ids_match_old_records():
    assert takeio.ids_match("p:001", "p:001")
    assert takeio.ids_match("p:001", "p:001:abcd1234")  # a record made before ids had the project uid
    assert not takeio.ids_match("p:001:abcd1234", "p:001:ffff0000")
    assert not takeio.ids_match("p:001", "q:001:abcd1234")
    assert not takeio.ids_match("p:00", "p:001:abcd1234")
    assert not takeio.ids_match("", "p:001")


def test_stale_summary_is_short_and_actionable():
    plan = ["camera: stale (params: zoom, version 2 -> 3)", "ingest: stale (code changed: afv.media)"]
    chain = ["hands_face was made from an older smooth result"]
    s = tk.stale_summary("001", chain, plan)
    assert s == ("camera, ingest, hands_face results are older than the current settings or code: "
                 "run `afv run 001` (details: afv status 001)")
    assert tk.stale_summary("001", [], []) == ""
