"""The take scene in headless Blender (synthetic shot): alignment, structure, saved file, Re-import,
the add-on's operators and the extension manifest."""

from __future__ import annotations

import itertools
import shutil
import subprocess

import numpy as np
import pytest
from take_helpers import ADDON_DIR, REPO, TESTS_DIR, addon_module, synthetic_shot, write_video

from afv import take as tk
from afv.export import blender as bl
from afv.motion.rotations import yup_to_zup_vec

takeio = addon_module("takeio")

pytestmark = [
    pytest.mark.blender,
    pytest.mark.slow,
    pytest.mark.skipif(bl.find_blender() is None, reason="Blender not installed"),
]

N = 24


def run_check(mode: str, job: dict, work) -> dict:
    exe = bl.find_blender()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bl, "script_path", lambda name: TESTS_DIR / "bl_checks.py")
        job = {"mode": mode, "addon_parent": bl.host_path(REPO / "blender", exe), **job}
        res = bl.run_blender("bl_checks.py", job, work)
    assert res["ok"], res
    return res["out"]


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("bl_take")
    bm, cam = synthetic_shot(n_frames=N)
    video = write_video(out / "proj" / "proxy.mp4", N)
    inp = tk.TakeInputs(motion=bm, camera=cam, shot_id="007", project_dir=str(out / "proj"), video=video)
    tj = tk.write_take(out / "take", inp)
    samples = out / "samples.npz"
    blend = out / "take" / "take.blend"
    res = tk.build_blend(tj, blend, samples_npz=samples)
    return {"out": out, "bm": bm, "cam": cam, "take": tj, "res": res, "samples": samples, "blend": blend}


def test_overlay_and_joints_match_the_pipeline(synthetic):
    """Joints through the Blender camera vs afv's own projection (zoom, off-centre principal point,
    world shift), and the evaluated NLA pose vs the package's FK."""
    rep = tk.check_alignment(synthetic["take"], synthetic["samples"], synthetic["cam"])
    assert rep["frames"] == N
    for role, r in rep["fighters"].items():
        print(f"{role}: joints max {r['max_mm']:.5f} mm, overlay max {r['px_max']:.5f} px")
        assert r["bones"] == 65
        assert r["max_mm"] <= 1.0
        assert r["px_points"] == 65 * N
        assert r["px_max"] <= 1.0


def test_scene_structure(synthetic):
    s = synthetic["res"]["summary"]
    assert s["fps"] == pytest.approx(30.0)
    assert (s["frame_start"], s["frame_end"]) == (0, N - 1)
    assert s["resolution"] == [640, 360]
    assert s["markers"] == s["spans"] > 0
    assert len(s["instance"]) == 8
    for role in ("A", "B"):
        f = s["fighters"][role]
        assert f["active_action"] is None
        assert f["action_blend_type"] == "COMBINE"  # keys made outside tweak mode add on top
        assert [(t["name"], t["strips"][0]["blend"]) for t in f["tracks"]] == [
            ("AFV solved", "REPLACE"), ("AFV tweaks", "COMBINE")]
        assert f["tracks"][0]["strips"][0]["action"] == f"007_{role}_solved"
        assert (f["tracks"][0]["strips"][0]["start"], f["tracks"][0]["strips"][0]["end"]) == (0, N - 1)
    cam = s["camera"]
    assert cam["clip_frames"] == N and cam["clip_frame_start"] == 0 and cam["clip_size"] == [320, 180]
    assert cam["lens0"] == pytest.approx(600.0 * 36.0 / 640.0, rel=1e-6)


def test_saved_blend_reopens_with_relative_video(synthetic):
    r = run_check("reopen", {"blend": bl.host_path(synthetic["blend"], bl.find_blender())},
                  synthetic["out"] / "reopen")
    assert r["has_take"] and r["show_background"]
    assert r["clip_path"].startswith("//"), r["clip_path"]
    assert r["clip_size"] == [320, 180] and r["clip_frames"] == N
    assert r["fps"] == pytest.approx(30.0) and r["frames"] == [0, N - 1]
    assert set(r["view_camera"]) == {"CAMERA"}
    assert [t["name"] for t in r["fighters"]["A"]] == ["AFV solved", "AFV tweaks"]
    spans = tk.load_take_arrays(synthetic["take"])[0]["qc"]["spans"]
    assert [tuple(m) for m in r["markers"]] == sorted((s["start"], s["label"]) for s in spans)


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def test_reimport_replaces_only_the_solved_motion(synthetic):
    """A key on the tweak track survives Re-import; the solved strip gets the new motion, the new bone
    lengths and a longer range (the tweak strip is stretched to it); the final pose is solved (x)
    tweak (Combine)."""
    out = synthetic["out"]
    n2 = N + 10
    bm2, cam2 = synthetic_shot(n_frames=n2, seed=5, scale=1.04)
    inp2 = tk.TakeInputs(motion=bm2, camera=cam2, shot_id="007", project_dir=str(out / "proj"))
    tj2 = tk.write_take(out / "take2", inp2)
    n3 = N + 20
    bm3, cam3 = synthetic_shot(n_frames=n3, seed=7, scale=1.04)
    tj3 = tk.write_take(out / "take3", tk.TakeInputs(motion=bm3, camera=cam3, shot_id="007",
                                                     project_dir=str(out / "proj")))
    blend = out / "reimport.blend"
    shutil.copy(synthetic["blend"], blend)
    exe = bl.find_blender()
    bone, frame = "LeftHand", 7
    r = run_check("reimport", {"blend": bl.host_path(blend, exe), "take2": bl.host_path(tj2, exe),
                               "take3": bl.host_path(tj3, exe), "role": "A", "bone": bone, "frame": frame},
                  out / "reimport")
    # muted, renamed, slid by 3 and blended in, then a longer take: the same strip, only longer
    st = r["strip_after"]
    assert st["strips"] == 1 and st["name"] == "my fixes" and st["mute"] and st["blend_in"] == 2.0
    assert st["before"] == [3, n2 + 2] and st["start"] == 3 and st["end"] == 3 + n3 - 1
    assert st["action_range"] == [0, n3 - 1] and st["action_frames"] == [0, n3 - 1] and st["keys"] == 4
    tracks = {t["name"]: t["strips"] for t in r["nla"]}
    assert list(tracks) == ["AFV solved", "AFV tweaks"]
    assert tracks["AFV tweaks"][0]["action"] == r["tweak_action_before"] == "007_A_tweaks"
    assert tracks["AFV tweaks"][0]["keys"] == 4 and tracks["AFV tweaks"][0]["blend"] == "COMBINE"
    assert (tracks["AFV tweaks"][0]["start"], tracks["AFV tweaks"][0]["end"]) == (0, n2 - 1)
    assert tracks["AFV tweaks"][0]["action_range"] == [0, n2 - 1]
    assert (tracks["AFV solved"][0]["start"], tracks["AFV solved"][0]["end"]) == (0, n2 - 1)
    assert r["summary"]["frame_end"] == n2 - 1
    assert r["solved_action"] != r["before_action"] and not r["old_action_exists"]
    assert len(r["proxies"]) == 2
    want = _qmul(np.array(r["solved_basis"]), np.array(r["tweak_q"]))
    got = np.array(r["final_basis"])
    assert min(np.abs(got - want).max(), np.abs(got + want).max()) < 1e-5

    meta, arrays = tk.load_take_arrays(tj2)
    f = meta["fighters"][0]
    names = [b["name"] for b in f["bones"]]
    parents = {b["name"]: b["parent"] for b in f["bones"]}

    def under(n):
        while n is not None:
            if n == bone:
                return True
            n = parents[n]
        return False

    sel = [r["bones"].index(n) for n in names]
    np.testing.assert_allclose(np.array(r["rest_heads"])[sel], arrays["A"]["rest_head"], atol=1e-5)
    heads = np.array(r["heads"])[sel]
    want_heads = yup_to_zup_vec(arrays["A"]["pose_head"][frame])
    keep = [i for i, n in enumerate(names) if not under(n) or n == bone]
    err = np.linalg.norm(heads[keep] - want_heads[keep], axis=-1)
    assert err.max() <= 1e-3, f"max {err.max() * 1000:.3f} mm"
    moved = [i for i, n in enumerate(names) if under(n) and n != bone]
    assert np.linalg.norm(heads[moved] - want_heads[moved], axis=-1).max() > 0.01  # the tweak shows


def test_addon_registers_and_navigates_issues(synthetic):
    r = run_check("addon", {"take": bl.host_path(synthetic["take"], bl.find_blender())},
                  synthetic["out"] / "addon")
    assert r["registered"] and all(r["operators"])
    assert r["export_preset"] == "unity_humanoid"
    spans = [{"start": s, "end": s} for s in r["span_starts"]]
    assert spans
    frame, want = 0, []
    for _ in range(len(spans) + 1):
        frame = takeio.next_span(spans, frame, 1)["start"]
        want.append(frame)
    assert r["visited"] == want
    assert any(b <= a for a, b in itertools.pairwise(want))  # wrapped around
    assert r["back"] == takeio.next_span(spans, want[-1], -1)["start"]
    assert r["reimport"] == ["FINISHED"]


def test_extension_builds(tmp_path):
    """``blender --command extension build`` accepts the manifest and packs the add-on."""
    exe = bl.find_blender()
    src = tmp_path / "afv_addon"
    shutil.copytree(ADDON_DIR, src, ignore=shutil.ignore_patterns("__pycache__"))
    cmd = [str(exe), "--factory-startup", "--command", "extension", "build",
           "--source-dir", bl.host_path(src, exe), "--output-dir", bl.host_path(tmp_path, exe)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    zips = list(tmp_path.glob("afv-*.zip"))
    assert zips, proc.stdout


def test_job_lifecycle(synthetic):
    """Background jobs: a file opened mid-job (result not applied, nothing stays busy), Cancel,
    a failing callback (reported, not stuck), Export paused for a tracking review and "Export
    anyway" (--yes), unregister with a job running."""
    exe = bl.find_blender()
    bm, cam = synthetic_shot(n_frames=6)
    plan = ["camera: stale (params: zoom, version 2 -> 3)"]
    inp = tk.TakeInputs(motion=bm, camera=cam, shot_id="007", project_dir=str(synthetic["out"] / "proj"),
                        stale=[f"afv run would re-run {p}" for p in plan],
                        stale_summary=tk.stale_summary("007", [], plan))
    stale = tk.write_take(synthetic["out"] / "take_stale", inp)
    job = {"take": bl.host_path(synthetic["take"], exe), "stale_take": bl.host_path(stale, exe)}
    r = run_check("jobs", job, synthetic["out"] / "jobs")
    load = r["load"]
    assert load["timer_started"] and load["timer_after_load"] and load["busy_during"]
    assert load["applied"] == [] and "another file was opened" in load["status"]
    assert not load["busy_after"] and load["job_cleared"]
    c = r["cancel"]
    assert c["op"] == ["FINISHED"] and c["job_cleared"] and not c["busy"] and not c["timer"]
    e = r["callback_error"]
    assert e["error"] and "ZeroDivisionError" in e["status"] and not e["busy"]
    assert r["normal"] == {"applied": True, "busy": False}
    ch = r["chain"]
    assert ch["poll_returned"] == 0.1 and ch["timer"] and ch["applied"]  # a job started by a callback runs
    st = r["stale_import"]
    assert st["level"] == "warning" and not st["error"]
    assert "run `afv run 007`" in st["status"] and len(st["status"]) < 300
    rv = r["review"]
    assert rv["args"][0] == "export" and "--yes" not in rv["args"]
    assert rv["review"]["hints"] == [{"kind": "swap"}] and "id swap at 40" in rv["review"]["message"]
    assert rv["review"]["shot_id"] == "007" and rv["review"]["preset"] == "unity_humanoid"
    assert len(rv["review"]["instance"]) == 8
    a = r["anyway"]
    assert a["args"][-2:] == ["--yes", "--json"]
    assert a["args"][a["args"].index("--preset") + 1] == "unity_humanoid"  # what was reviewed
    assert not a["error"] and a["status"].startswith("exported 1 files")
    u = r["unregister"]
    assert u["job_cleared"] and not u["timer"] and not u["busy"]
    assert r["applied"] == ["t5", "c2"]


def test_duplicate_imports_are_separate_instances(synthetic):
    out = synthetic["out"]
    bm2, cam2 = synthetic_shot(n_frames=N, seed=6)
    tj2 = tk.write_take(out / "take_inst", tk.TakeInputs(motion=bm2, camera=cam2, shot_id="007",
                                                         project_dir=str(out / "proj")))
    exe = bl.find_blender()
    job = {"take": bl.host_path(synthetic["take"], exe), "take2": bl.host_path(tj2, exe)}
    r = run_check("instances", job, out / "instances")
    a, b = r["instances"]
    assert a != b and r["records"] == sorted([a, b])
    assert r["cameras"] == 2 and r["armatures"] == 4
    assert r["record_for"] == r["reimported"] == r["scene_active"] == a
    assert r["after"][a] != r["before"][a] and r["after"][b] == r["before"][b]


def test_moved_project_still_reimports(synthetic, tmp_path):
    """The .blend and take moved together (project folder moved): the take is found next to it."""
    out = synthetic["out"]
    root = out / "move_from"
    bm, cam = synthetic_shot(n_frames=N)
    inp = tk.TakeInputs(motion=bm, camera=cam, shot_id="007", project_dir=str(out / "proj"),
                        video=out / "proj" / "proxy.mp4")
    tj = tk.write_take(root / "take", inp)
    tk.build_blend(tj, root / "take" / "take.blend")
    dest = out / "move_to"
    root.rename(dest)
    r = run_check("moved", {"blend": bl.host_path(dest / "take" / "take.blend", bl.find_blender())},
                  out / "moved")
    assert "move_from" in r["stored"] and r["rel"] == "take.json"
    assert "move_to" in r["resolved"] and "move_to" in r["stored_after"]
    assert r["reimported"] == "proj:007" and r["clip_exists"]


def test_a_blend_next_to_another_projects_take_runs_nothing(synthetic):
    """Re-import / Export only ever use a take.json with the scene's take id."""
    out = synthetic["out"]
    bm, cam = synthetic_shot(n_frames=8)
    a = tk.write_take(out / "mm_a" / "take", tk.TakeInputs(motion=bm, camera=cam, shot_id="007",
                                                            project_dir="/x/projA"))
    tk.build_blend(a, a.parent / "take.blend")
    b = tk.write_take(out / "mm_b" / "take", tk.TakeInputs(motion=bm, camera=cam, shot_id="007",
                                                            project_dir="/x/projB"))
    shutil.copy(a.parent / "take.blend", b.parent / "take.blend")
    shutil.rmtree(out / "mm_a")  # the original take is gone
    before = b.read_bytes()
    r = run_check("mismatch", {"blend": bl.host_path(b.parent / "take.blend", bl.find_blender())}, out / "mm")
    for k in ("resolve", "reimport"):
        assert "cannot find take 'projA:007'" in r[k] and "is take 'projB:007'" in r[k], r[k]
    assert r["export_op"][0] == "CANCELLED" and "projB:007" in r["export_op"][1]
    assert r["calls"] == [] and not r["job"]  # afv never ran
    assert b.read_bytes() == before


def test_save_copy_keeps_the_open_files_paths(synthetic):
    out = synthetic["out"]
    exe = bl.find_blender()
    (out / "sc" / "a").mkdir(parents=True)
    (out / "sc" / "b" / "deeper").mkdir(parents=True)
    r = run_check("savecopy", {"take": bl.host_path(synthetic["take"], exe),
                               "a": bl.host_path(out / "sc" / "a" / "x.blend", exe),
                               "b": bl.host_path(out / "sc" / "b" / "deeper" / "y.blend", exe)},
                  out / "savecopy")
    assert r["rel_a"].replace("\\", "/") == "../../take/take.json"
    assert r["rel_open_after_copy"] == r["rel_a"] and r["open_file"].endswith("x.blend")
    assert r["rel_in_copy"].replace("\\", "/") == "../../../take/take.json"
    assert r["resolved_in_copy"].replace("\\", "/").endswith("take/take.json")


def test_scene_from_before_instances_still_reimports(synthetic):
    """Records of older imports (no instance, take id without the project uid) keep working."""
    out = synthetic["out"]
    bm, cam = synthetic_shot(n_frames=8)
    tj = tk.write_take(out / "legacy" / "take", tk.TakeInputs(motion=bm, camera=cam, shot_id="007",
                                                               project_dir="/x/proj", project_uid="abcd1234"))
    tk.build_blend(tj, tj.parent / "take.blend")
    blend = bl.host_path(tj.parent / "take.blend", bl.find_blender())
    r = run_check("legacy", {"blend": blend}, out / "legacy")
    assert r["old_id"] == "proj:007" and r["new_id"] == r["reimported"] == "proj:007:abcd1234"
    assert len(r["instance"]) == 8 and r["collections_with_record"] == 1
    assert len(r["tagged"]) == 5  # 2 armatures, 2 proxies, the camera
    assert {(i, t) for _, i, t in r["tagged"]} == {(r["instance"], r["new_id"])}
