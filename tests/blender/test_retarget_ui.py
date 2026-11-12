"""The add-on's Retarget sub-panel (track R2) in headless Blender: its operators on a synthetic take and
synthetic characters (FBX written from a 24 fps scene, GLB, a .blend with a script that must not run,
a non-uniformly scaled one). Blender side: ``ui_retarget_checks.py``.

Checked: Import character keeps the take's fps and frame range and picks the character (never a take
fighter); the guessed map fills the bone list and is stored on the armature; a missing required bone,
a bone used twice and an engine refusal are readable errors that leave no action behind; Retarget makes
one new action and shows its report; Save / Load map; Export character gets the presets' FBX settings
from afv (one-shot fallback, then cached) and its FBX re-imports within 1 mm of the retargeted armature
on every frame, with the exact frame range and fps; the .blend import appends only the armature and its
meshes and runs none of the file's scripts; the panel's draw code runs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from take_helpers import REPO, TESTS_DIR, synthetic_shot

from afv import take as tk
from afv.export import blender as bl
from afv.export.fbx import read_fbx
from afv.retarget import presets_payload

pytestmark = [
    pytest.mark.blender,
    pytest.mark.slow,
    pytest.mark.skipif(bl.find_blender() is None, reason="Blender not installed"),
]

CHECKS = TESTS_DIR / "ui_retarget_checks.py"
N = 30


def run_ui_check(mode: str, job: dict, work: Path, timeout: float = 900.0, autoexec: bool = False) -> dict:
    """Run ``ui_retarget_checks.py`` headless (with --disable-autoexec like the engine tests, or with
    ``autoexec`` --enable-autoexec: Python drivers run)."""
    exe = bl.find_blender()
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    result = work / f"{mode}_result.json"
    result.unlink(missing_ok=True)
    job_file = work / f"{mode}_job.json"
    job = {"mode": mode, "addon_parent": bl.host_path(REPO / "blender", exe),
           "tests_dir": bl.host_path(TESTS_DIR, exe), "result_path": bl.host_path(result, exe), **job}
    job_file.write_text(json.dumps(job, indent=1))
    flag = "--enable-autoexec" if autoexec else "--disable-autoexec"
    cmd = [str(exe), "-b", "--factory-startup", flag, "-noaudio", "--python-exit-code", "1",
           "--python", bl.host_path(CHECKS, exe), "--", bl.host_path(job_file, exe)]
    proc = None
    for _ in range(2):  # WSL interop sometimes fails to launch a Windows process once
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=work)
        if result.is_file() or b"UtilAcceptVsock" not in proc.stderr + proc.stdout:
            break
    (work / f"{mode}_blender.log").write_bytes(proc.stdout + b"\n--- stderr ---\n" + proc.stderr)
    if not result.is_file():
        tail = (proc.stdout + proc.stderr)[-3000:].decode("utf-8", "replace")
        raise AssertionError(f"Blender wrote no result (exit {proc.returncode}):\n{tail}")
    res = json.loads(result.read_text(encoding="utf-8"))
    assert res["ok"], res.get("error")
    return res["out"]


def synthetic_take(out: Path, n: int = N) -> Path:
    bm, cam = synthetic_shot(n_frames=n, seed=2)
    inp = tk.TakeInputs(motion=bm, camera=cam, shot_id="007", project_dir=str(out / "proj"))
    return tk.write_take(out / "take", inp)


def make_characters(out: Path) -> dict:
    """The synthetic character files (Blender-side paths) and their expected maps."""
    exe = bl.find_blender()
    out.mkdir(parents=True, exist_ok=True)
    res = run_ui_check("characters", {"out": bl.host_path(out, exe)}, out / "work")
    kinds = ("fbx", "glb", "blend", "squashed", "scripted", "hidden", "broken", "linked")
    res["local"] = {k: out / Path(res[k].replace("\\", "/")).name for k in kinds}
    return res


def fbx_clips(path: Path) -> dict:
    """The animation stack (clip) names and take names in an FBX file (``afv.export.fbx.read_fbx``)."""
    top = {n.name: n for n in read_fbx(path)}
    stacks = [n.props[1].split("\x00")[0] for n in top["Objects"].children if n.name == "AnimationStack"]
    takes = [n.props[0] for n in top["Takes"].children if n.name == "Take"] if "Takes" in top else []
    return {"stacks": stacks, "takes": takes}


@pytest.fixture(scope="module")
def ui(tmp_path_factory):
    exe = bl.find_blender()
    out = tmp_path_factory.mktemp("rt_ui")
    take = synthetic_take(out)
    chars = make_characters(out / "chars")
    line = out / "presets.jsonl"
    line.write_text(json.dumps({"event": "presets", "presets": presets_payload()}) + "\n")
    partial = out / "presets_partial.jsonl"  # an afv that does not know the chosen preset
    only = [p for p in presets_payload() if p["name"] == "blender"]
    partial.write_text(json.dumps({"event": "presets", "presets": only}) + "\n")
    files = {k: chars[k] for k in ("fbx", "glb", "blend", "squashed", "marker", "hidden", "broken", "linked")}
    job = {"take": bl.host_path(take, exe), **files, "presets_partial": bl.host_path(partial, exe),
           "presets_line": bl.host_path(line, exe), "out": bl.host_path(out, exe)}
    return {"chars": chars, "res": run_ui_check("ui", job, out / "ui"), "out": out}


def test_import_character_keeps_the_take_timing(ui):
    r = ui["res"]
    imp = r["import"]
    assert imp["result"] == ["FINISHED"], imp["error"]
    assert imp["fps"] == pytest.approx(30.0) and imp["fps_before"] == pytest.approx(30.0)
    assert imp["range"] == imp["range_before"] == [0, N - 1]
    assert "24 fps" in imp["status"]  # the importer's change was noticed and put back
    assert imp["collection"].startswith("Character syn_unity")
    assert imp["poll_target"] is True and r["poll_fighter"] is False  # a take fighter is never a target
    assert ("operator", "afv.retarget_import") in [tuple(x) for x in r["draw_empty"]]


def test_guess_fills_the_list_and_is_stored_on_the_armature(ui):
    g = ui["res"]["guess"]
    exp = ui["chars"]["expected_unity"]
    assert g["problems"]["missing"] == [] and g["problems"]["twice"] == [] and g["problems"]["absent"] == []
    for c, b in exp.items():
        assert g["rows"].get(c) == b, c
    assert g["stored"] == g["rows"]
    kinds = {x[0] for x in ui["res"]["draw_mapped"]}
    assert {"template_list", "prop", "operator", "label"} <= kinds


def test_missing_and_doubled_bones_are_readable_refusals(ui):
    m, t = ui["res"]["missing"], ui["res"]["twice"]
    assert m["result"] is None and "LeftFoot" in m["error"] and "Traceback" not in m["error"]
    assert "LeftFoot" in m["panel_error"] and m["new_actions"] == 0
    assert m["stored_has_foot"] is False  # the list's edit reached the stored map
    assert any("LeftFoot" in x[1] for x in m["draw"])  # shown in red in the panel
    assert t["result"] is None and "already mapped" in t["error"] and t["new_actions"] == 0
    assert len(t["problems"]["twice"]) == 1


def test_retarget_makes_one_action_and_shows_the_report(ui):
    rt = ui["res"]["retarget"]
    assert rt["result"] == ["FINISHED"], rt["error"]
    assert rt["new_actions"] == 1 and rt["undo_flag"]
    rep = rt["report"]
    assert rep["action"] == rt["active"] == f"007_A_{ui['res']['import']['target']}"
    assert rep["frames"] == [0, N - 1] and rep["role"] == "A"
    labels = " | ".join(x[1] for x in rt["draw"] if x[0] == "label")
    assert "hips scale" in labels and "foot slide" in labels and "bones mapped" in labels
    assert "(IK legs left out)" in labels  # lock feet on: the IK'd legs are not in the direction error
    b = ui["res"]["role_b"]
    assert b["result"] == ["FINISHED"] and b["report"]["role"] == "B"
    assert b["report"]["action"].startswith("007_B_")


def test_save_and_load_map(ui):
    m = ui["res"]["map_io"]
    assert m["save"][0] == ["FINISHED"] and m["load"][0] == ["FINISHED"], m
    assert m["saved"] == m["rows"] == ui["res"]["guess"]["rows"]  # the cleared Spine came back
    assert m["facing"] == "AUTO"  # facing null in the file = detect


def test_export_character_round_trip(ui):
    e = ui["res"]["export"]
    assert e["result"] == ["FINISHED"], e["error"]
    assert e["pending"] and e["calls"] == [["retarget", "--list-presets", "--json"]]
    assert e["exists"] and "unity_humanoid" in e["presets"] and "unreal" in e["presets"]
    c = e["check"]
    print(f"UI export round trip: max {c['max_mm']:.5f} mm over {c['bones']} bones, {c['frames']} frames")
    assert c["bones"] == c["bones_target"] == c["bones_file"] and c["bones"] > 50
    assert c["max_mm"] <= 1.0
    assert c["frames"] == N and c["file_frames"] == [0, N - 1] and c["file_keys"] == N
    assert c["file_fps"] == pytest.approx(30.0) and e["fps_after"] == pytest.approx(30.0)
    assert c["meshes"] == 1
    cached = ui["res"]["export_cached"]
    assert cached["result"] == ["FINISHED"] and cached["exists"]
    assert cached["calls"] == 1 and not cached["job"]  # from the cache: no second request


def test_engine_refusals_are_shown_not_raised(ui):
    r = ui["res"]["refused"]
    assert r["result"] is None and "non-uniform scale" in r["error"] and "Traceback" not in r["error"]
    assert "non-uniform scale" in r["panel_error"] and r["new_actions"] == 0
    s = ui["res"]["squashed"]
    assert s["import"][0] == ["FINISHED"] and s["retarget"][0] is None and "non-uniform" in s["panel_error"]
    bad = ui["res"]["bad_file"]
    assert bad["result"] is None and "no such file" in bad["error"]


def test_blend_import_appends_only_the_character_and_runs_no_script(ui):
    b = ui["res"]["blend"]
    assert b["result"] == ["FINISHED"], b["error"]
    assert b["marker"] is False  # the .blend's registered text block never ran
    assert b["texts"] == []  # ... and was not even loaded
    assert b["new"] == sorted(["SynAnat", "SynAnat_mesh"])  # not the other armature, not the prop
    assert b["fps"] == pytest.approx(30.0)
    exp = ui["chars"]["expected_anat"]
    assert b["missing"] == [] and all(b["rows"].get(c) == v for c, v in exp.items() if "Hand" not in c)
    assert ui["res"]["switch_back"] == ui["res"]["guess"]["rows"]  # rows follow the picked armature
    g = ui["res"]["glb"]
    assert g["result"] == ["FINISHED"] and g["missing"] == []


def test_bone_list_filters(ui):
    lst = ui["res"]["list"]
    assert lst["rows"] == 65
    assert lst["shown"] == 65 - 40 and lst["shown_fingers"] == 65 and lst["shown_foot"] == 2
    assert [x[0] for x in lst["draw"]].count("prop_search") == 3


def test_edit_mode_and_fighter_targets_are_refused(ui):
    e = ui["res"]["edit_mode"]
    assert e["result"] is None and "leave Edit Mode" in e["error"] and e["new_actions"] == 0
    f = ui["res"]["fighter_target"]
    assert f["assigned"] is not True or (f["result"] is None and "take's own fighter" in f["error"])
    x = ui["res"]["export_edit"]
    assert x["result"] is None and "leave Edit Mode" in x["error"] and not x["job"] and x["calls"] == 0
    assert "leave Edit Mode" in x["panel_error"]


def test_a_failed_export_after_fetching_the_presets_is_shown_in_the_panel(ui):
    f = ui["res"]["export_fetch_fail"]
    assert f["result"] == ["FINISHED"]  # the fetch started
    assert "afv has no preset 'unity_humanoid'" in f["panel_error"]
    assert "Export character failed" in f["status"]
    assert not f["exists"] and f["presets"] == ["blender"]


def test_retarget_again_replaces_only_the_untouched_active_action(ui):
    r = ui["res"]["reuse"]
    u = r["unchanged"]
    assert u["result"] == ["FINISHED"] and u["replaced"] and not u["prev_alive"] and u["new_actions"] == 0
    assert u["active"] == u["prev_name"]  # the new action took over its name


@pytest.mark.parametrize("case", ["renamed", "nla", "interpolation", "modifier", "key", "other_object"])
def test_retarget_again_keeps_an_action_the_user_touched(ui, case):
    c = ui["res"]["reuse"][case]
    assert c["result"] == ["FINISHED"], c["error"]
    assert c["prev_alive"] and not c["replaced"] and c["new_actions"] == 1, c


def test_retarget_again_never_deletes_a_copy_or_another_characters_action(ui):
    c = ui["res"]["reuse"]["copy"]
    assert c["copy_alive"]  # the copy carries the tag but is another datablock: kept
    t = ui["res"]["reuse"]["same_name_other_character"]
    assert t["result"] == ["FINISHED"], t["error"]
    assert t["prev_alive"] and t["new_actions"] == 1


def test_rest_matrix_drift_is_warned_and_fixable(ui):
    d = ui["res"]["drift"]
    assert d["result"] == ["FINISHED"] and any("rest matrices carry up to 0.12 %" in n for n in d["notes"])
    assert d["fix_button"] and d["fix"][0] == ["FINISHED"] and d["mode"] == "POSE"  # its mode kept
    assert d["draw_after"] is False


def test_hips_scale_is_saved_rounded(ui):
    m = ui["res"]["map_io"]
    assert m["hips_file"] == 0.61 and m["hips_loaded"] == 0.61


def test_hidden_armature_imports_and_failures_leave_nothing_behind(ui):
    h = ui["res"]["hidden"]
    assert h["result"] == ["FINISHED"], h["error"]
    assert h["hide_viewport"] is False and h["hide_select"] is False and h["hide_get"] is False
    assert h["mode"] == "OBJECT" and "re-derived" in h["status"]
    assert "was hidden in the file; shown" in h["status"]
    f = ui["res"]["import_fails"]
    assert f["result"] is None and "Cannot edit hidden object" in f["error"] and "Traceback" not in f["error"]
    assert f["objects_left"] == 0 and f["collections_left"] == 0 and f["data_left"] == 0
    b = ui["res"]["broken"]
    assert b["result"] is None and "cannot read this .blend" in b["error"]
    assert b["panel_error"].endswith(")") and "\n" not in b["panel_error"]  # Blender's newline stripped
    assert b["objects_left"] == 0 and b["collections_left"] == 0


def test_exported_clip_is_named_after_the_action(ui):
    action = ui["res"]["drift"]["action"]  # the target's action when ui_A.fbx was exported
    assert ui["res"]["export"]["exists"]
    clips = fbx_clips(ui["out"] / "ui_A.fbx")
    assert clips["stacks"] == [action], clips
    assert all(t == action for t in clips["takes"])


def test_blend_python_drivers_never_run_even_with_auto_run_on(tmp_path_factory):
    exe = bl.find_blender()
    out = tmp_path_factory.mktemp("rt_scripted")
    chars = make_characters(out / "chars")
    control = out / "control_ran.txt"
    job = {"blend": chars["scripted"], "markers": chars["driver_markers"],
           "control": bl.host_path(control, exe)}
    res = run_ui_check("scripted", job, out / "run", autoexec=True)
    assert res["control_ran"], "Python drivers do not run in this session: the test proves nothing"
    assert res["ran"] == {"object": False, "data": False, "shape_key": False, "material": False}, res
    assert any("4 Python-expression drivers in the file were muted" in n for n in res["notes"]), res["notes"]
    imported = [d for d in res["drivers"] if not d[0].startswith("Control")]
    assert len(imported) == 4 and all(d[2] for d in imported)


def test_python_drivers_linked_from_a_third_file_are_refused(ui):
    li = ui["res"]["linked"]
    assert li["result"] is None and "links data with Python drivers" in li["error"]
    assert "LibCubeMesh" in li["error"] and "lib_driver.blend" in li["error"] and "Make Local" in li["error"]
    assert li["objects_left"] == 0 and li["collections_left"] == 0 and li["libraries_left"] == 0
    assert li["data_left"] == 0  # no 0-user armature / mesh / material / action data either
    assert not (ui["out"] / "chars" / "driver_linked.txt").exists()


def test_exporter_errors_show_one_line(ui):
    f = ui["res"]["export_folder"]
    assert f["result"] is None and "Traceback" not in f["error"] and "Traceback" not in f["panel_error"]
    assert "\n" not in f["panel_error"].strip() and f["panel_error"]
    assert "Location:" not in f["panel_error"]
    assert "PermissionError" in f["panel_error"] or "IsADirectoryError" in f["panel_error"], f["panel_error"]
