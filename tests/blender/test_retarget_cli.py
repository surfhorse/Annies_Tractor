"""``afv retarget`` (track R2): the WSL-side driver ``afv.retarget`` and the CLI command.

- Always: the preset list (``--list-presets``), names, argument errors.
- Blender (synthetic, always when Blender is installed): ``retarget_shot`` on a synthetic take (the take
  writer patched in, no shot results needed) with synthetic FBX / GLB / .blend characters, both
  fighters: the FBX files re-import within 1 mm of the retarget on every frame with the exact frame
  range and fps, and the clip named after the action; a required bone missing (exit-1 path, the map
  still saved), a map naming bones the character lacks (map still saved), an engine refusal
  (non-uniform scale), an unreadable .blend, a wrong file type, one role without foot locking into a
  relative ``--out``; the CLI text and JSON with ``--check`` and the stale-results warning (the take
  writer patched, a real project).
- Blender + the owner's files (skip when ``Media_Samples/`` or ``projects/ufc_cross`` is missing): the
  real command on a copy of ``projects/ufc_cross`` (shot 001, never the project itself) for the three
  test characters, both fighters, with ``--check``; the saved map fed back with ``--map``; a map with
  a required bone removed exits 1 listing it. Times are printed (``-s``)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from take_helpers import REPO
from test_retarget_ui import fbx_clips, make_characters, run_ui_check, synthetic_take
from typer.testing import CliRunner

from afv import retarget as rg
from afv import take as tk
from afv.cli import app
from afv.export import blender as bl
from afv.export.presets import FBX_ARGS, PRESETS

HAVE_BLENDER = bl.find_blender() is not None
N = 30


# --- no Blender -------------------------------------------------------------------------------------
def test_list_presets_json():
    res = CliRunner().invoke(app, ["retarget", "--list-presets", "--json"])
    assert res.exit_code == 0, res.output
    ev = json.loads(res.output.strip().splitlines()[-1])
    assert ev["event"] == "presets"
    assert [p["name"] for p in ev["presets"]] == list(PRESETS)
    for p in ev["presets"]:
        assert set(p["fbx"]) == set(FBX_ARGS) and p["fbx"]["add_leaf_bones"] is False
    assert ev["presets"] == rg.presets_payload()


def test_list_presets_text_shows_the_fbx_settings():
    res = CliRunner().invoke(app, ["retarget", "--list-presets"])
    assert res.exit_code == 0, res.output
    for p in PRESETS.values():
        assert f"{p.name}: {p.label}" in res.output
    assert "add_leaf_bones=False" in res.output and "apply_scale_options=FBX_SCALE_ALL" in res.output
    assert res.output.count("  fbx: ") == len(PRESETS)
    assert max(len(ln) for ln in res.output.splitlines()) <= 110
    helptext = CliRunner().invoke(app, ["retarget", "--help"]).output
    assert "FBX exporter settings" in helptext and "root bone" in helptext


def test_roles():
    assert rg.parse_roles("both") == ["A", "B"] and rg.parse_roles("b") == ["B"]
    with pytest.raises(rg.RetargetError, match="A, B or both"):
        rg.parse_roles("C")


def test_needs_shot_and_character():
    res = CliRunner().invoke(app, ["retarget", "--json"])
    assert res.exit_code == 1
    assert "needs a shot id and --character" in res.output


def test_character_extensions_match_the_blender_side():
    """``afv.retarget`` checks the file type early with its own copy of the list: it must equal the
    add-on's (``retarget_io.CHARACTER_EXTS``, read without importing bpy)."""
    import ast

    src = (REPO / "blender" / "afv_addon" / "retarget_io.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") == "CHARACTER_EXTS" for t in n.targets))
    assert tuple(ast.literal_eval(node.value)) == rg.CHARACTER_EXTS


def test_character_file_is_checked_first(tmp_path):
    shot = SimpleNamespace(id="007", exports_dir=tmp_path / "exports", work_root=tmp_path / ".work")
    with pytest.raises(rg.RetargetError, match=r"expected a character file \(\.fbx"):
        rg.retarget_shot(shot, tmp_path / "char.obj")  # before the take is built or Blender starts
    with pytest.raises(rg.RetargetError, match="no such file"):
        rg.retarget_shot(shot, tmp_path / "char.fbx")
    (tmp_path / "char.fbx").write_bytes(b"")
    with pytest.raises(rg.RetargetError, match="--map: no such file"):
        rg.retarget_shot(shot, tmp_path / "char.fbx", map_path=tmp_path / "m.afvmap.json")
    assert not (tmp_path / ".work").exists()


# --- Blender, synthetic --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    if not HAVE_BLENDER:
        pytest.skip("Blender not installed")
    out = tmp_path_factory.mktemp("rt_cli")
    take = synthetic_take(out / "src", N)
    chars = make_characters(out / "chars")
    return {"out": out, "take": take, "chars": chars}


@pytest.fixture
def driver(synth, tmp_path, monkeypatch):
    """``retarget_shot`` on the synthetic take: the take writer copies it, a stand-in shot."""
    def make_take(shot, out_dir=None, **kw):
        assert kw.get("check_plan", True) is True  # the stale-results check runs, as in afv take
        shutil.copytree(synth["take"].parent, out_dir)
        return Path(out_dir) / "take.json"

    monkeypatch.setattr(tk, "make_take", make_take)
    monkeypatch.setattr(tk, "shot_rig_settings", lambda shot, preset=None: tk.RigSettings())
    shot = SimpleNamespace(id="007", exports_dir=tmp_path / "exports", work_root=tmp_path / ".work")
    return shot


def _check_roles(rep, roles, name, shot):
    assert sorted(rep["files"]) == sorted(roles)
    for r in roles:
        f = Path(rep["files"][r])
        assert f == shot.exports_dir / "retarget" / f"007_{r}_{name}.fbx" and f.is_file()
        res = rep["roles"][r]
        c = res["check"]
        print(f"{name} {r}: round trip max {c['max_mm']:.5f} mm ({c['bones']} bones, {c['frames']} frames)")
        assert c["bones"] == c["bones_file"] == c["bones_target"] > 40
        assert c["max_mm"] <= 1.0 and c["max_mm_fighter"] <= 1.0
        assert c["frames"] == N and c["file_frames"] == [0, N - 1] and c["file_keys"] == N
        assert c["file_fps"] == pytest.approx(30.0)
        assert res["report"]["frames"] == [0, N - 1] and res["report"]["action"] == f.stem
        assert res["export"]["meshes"] >= 1 and res["export"]["take"] == f.stem
        clips = fbx_clips(f)
        assert clips["stacks"] == [f.stem] and all(t == f.stem for t in clips["takes"]), clips


@pytest.mark.parametrize("kind", ["fbx", "glb", "blend"])
@pytest.mark.blender
@pytest.mark.slow
def test_synthetic_characters_both_fighters(synth, driver, kind):
    path = synth["chars"]["local"][kind]
    rep = rg.retarget_shot(driver, path, check=True, save_map=driver.work_root.parent / "saved.afvmap.json")
    _check_roles(rep, ["A", "B"], path.stem, driver)
    assert rep["map_source"] == "guess" and rep["preset"] == "unity_humanoid"
    saved = json.loads(Path(rep["saved_map"]).read_text())
    assert saved["format"] == "afv_retarget_map" and saved["bones"] == rep["map"]["bones"]
    assert not any(driver.work_root.iterdir())  # the work folder is cleaned up
    if kind == "fbx":
        assert any("24 fps" in n for n in rep["import"]["notes"])
    if kind == "blend":  # only the armature and its mesh were appended
        assert sorted(rep["objects"]) == ["SynAnat", "SynAnat_mesh"]


@pytest.mark.blender
@pytest.mark.slow
def test_one_role_without_foot_lock_and_another_preset(synth, driver, monkeypatch):
    monkeypatch.chdir(driver.work_root.parent)
    rep = rg.retarget_shot(driver, synth["chars"]["local"]["fbx"], roles=["B"], lock_feet=False,
                           preset="unreal", out_dir=Path("elsewhere"))  # relative: reported absolute
    assert list(rep["files"]) == ["B"] and rep["preset"] == "unreal" and rep["lock_feet"] is False
    f = Path(rep["files"]["B"])
    assert f.is_absolute() and f.parent == (driver.work_root.parent / "elsewhere").resolve()
    assert Path(rep["out_dir"]).is_absolute()
    assert "check" not in rep["roles"]["B"]


@pytest.mark.blender
@pytest.mark.slow
def test_missing_required_bone_refuses_and_still_saves_the_map(synth, driver, tmp_path):
    chars = synth["chars"]
    bones = {c: b for c, b in chars["expected_unity"].items() if c != "LeftLeg"}
    m = tmp_path / "partial.afvmap.json"
    m.write_text(json.dumps({"format": "afv_retarget_map", "version": 1, "bones": bones, "facing": None,
                             "hips_scale": None, "target": "x", "mine": 1}))
    saved = tmp_path / "out.afvmap.json"
    with pytest.raises(rg.RetargetError) as ei:
        rg.retarget_shot(driver, chars["local"]["fbx"], map_path=m, save_map=saved)
    assert ei.value.missing == ["LeftLeg"] and "LeftLeg" in str(ei.value)
    assert json.loads(saved.read_text())["mine"] == 1  # the given map, unknown keys kept
    assert not (driver.exports_dir / "retarget").exists()


@pytest.mark.blender
@pytest.mark.slow
def test_map_with_unknown_bone_and_engine_refusal(synth, driver, tmp_path):
    chars = synth["chars"]
    m = tmp_path / "bad.afvmap.json"
    m.write_text(json.dumps({"format": "afv_retarget_map", "version": 1,
                             "bones": {**chars["expected_unity"], "LeftHand": "NoSuchBone"}}))
    saved = tmp_path / "saved.afvmap.json"
    with pytest.raises(rg.RetargetError, match="does not have: LeftHand -> NoSuchBone"):
        rg.retarget_shot(driver, chars["local"]["fbx"], map_path=m, save_map=saved)
    assert json.loads(saved.read_text())["bones"]["LeftHand"] == "NoSuchBone"  # saved to fix it there
    with pytest.raises(rg.RetargetError, match="non-uniform scale"):
        rg.retarget_shot(driver, chars["local"]["squashed"])
    m.write_text("{not json")
    with pytest.raises(rg.RetargetError, match="not valid JSON"):
        rg.retarget_shot(driver, chars["local"]["fbx"], map_path=m)


@pytest.mark.blender
@pytest.mark.slow
def test_unreadable_and_wrong_files_are_refusals(synth, driver, tmp_path):
    with pytest.raises(rg.RetargetError, match=r"cannot read this \.blend") as ei:
        rg.retarget_shot(driver, synth["chars"]["local"]["broken"])
    assert "Traceback" not in str(ei.value)
    obj = tmp_path / "char.obj"
    obj.write_text("o x\n")
    with pytest.raises(rg.RetargetError, match=r"expected a character file \(\.fbx"):
        rg.retarget_shot(driver, obj)
    assert not any(driver.work_root.iterdir())


@pytest.fixture
def cli_project(synth, tmp_path, monkeypatch):
    """A real project (a stand-in video) whose take writer is patched to the synthetic take with a
    stale-results summary, as ``make_take`` writes it for results older than their code."""
    from afv.core.project import Project

    video = tmp_path / "v.mp4"
    video.write_bytes(os.urandom(2048))
    proj = Project.create(tmp_path / "p.afv", video)
    proj.add_shot(0.0, 1.0)
    calls = []

    def make_take(shot, out_dir=None, **kw):
        calls.append(kw)
        shutil.copytree(synth["take"].parent, out_dir)
        tj = Path(out_dir) / "take.json"
        meta = json.loads(tj.read_text())
        meta["stale_summary"] = "pair result is older than the current settings or code: run `afv run 001`"
        tj.write_text(json.dumps(meta))
        return tj

    monkeypatch.setattr(tk, "make_take", make_take)
    monkeypatch.setattr(tk, "shot_rig_settings", lambda shot, preset=None: tk.RigSettings())
    return {"root": proj.root, "calls": calls}


@pytest.mark.blender
@pytest.mark.slow
def test_cli_text_and_json_with_check_and_stale_warning(synth, cli_project, tmp_path):
    fbx = synth["chars"]["local"]["fbx"]
    args = ["retarget", "001", "-p", str(cli_project["root"]), "--character", str(fbx), "--role", "A",
            "--check", "--out", str(tmp_path / "o")]
    res = CliRunner().invoke(app, args)
    assert res.exit_code == 0, res.output
    assert cli_project["calls"] == [{}]  # make_take with its plan check (no check_plan=False)
    assert "warning: pair result is older" in res.output
    assert "mm at fighter scale" in res.output and "mm in the character's world" in res.output
    assert "(IK legs left out)" in res.output  # lock feet on
    assert "clip '001_A_syn_unity'" in res.output
    res = CliRunner().invoke(app, [*args, "--json"])
    assert res.exit_code == 0, res.output
    events = [json.loads(ln) for ln in res.output.splitlines() if ln.startswith("{")]
    result = events[-1]
    assert result["event"] == "result" and result["stale_summary"].startswith("pair result is older")
    check = next(e for e in events if e["event"] == "retarget")["check"]
    assert {"max_mm", "max_mm_fighter"} <= set(check)


@pytest.mark.blender
@pytest.mark.slow
def test_girush_with_a_hidden_armature(synth, driver, tmp_path):
    girush = Path(os.environ.get("AFV_MEDIA_SAMPLES") or REPO / "Media_Samples") / \
        "Girush Young Warrior (CC0)" / "baseMesh_A1_(human_YW)_by_Girush.blend"
    if not girush.is_file():
        pytest.skip(f"test character missing: {girush}")
    exe = bl.find_blender()
    hidden = tmp_path / "girush_hidden.blend"
    job = {"blend": bl.host_path(girush, exe), "out": bl.host_path(hidden, exe)}
    run_ui_check("hide", job, tmp_path / "w")
    rep = rg.retarget_shot(driver, hidden, roles=["A"], check=True)
    assert any("re-derived" in n for n in rep["import"]["notes"])
    c = rep["roles"]["A"]["check"]
    assert c["max_mm_fighter"] <= 1.0 and c["frames"] == N


# --- Blender, the owner's characters on a copy of ufc_cross ----------------------------------------------
MEDIA = Path(os.environ.get("AFV_MEDIA_SAMPLES") or REPO / "Media_Samples")
CHARACTERS = {
    "mixamo": MEDIA / "Mixamo Mech" / "mixamo_mech.fbx",
    "unity": MEDIA / "Unity Humanoid Mech" / "Unity_Humanoid_Mech_rightarm_fixed.fbx",
    "girush": MEDIA / "Girush Young Warrior (CC0)" / "baseMesh_A1_(human_YW)_by_Girush.blend",
}
PROJECT = Path(os.environ.get("AFV_REAL_PROJECTS") or REPO / "projects") / "ufc_cross"


@pytest.fixture(scope="module")
def ufc_copy(tmp_path_factory):
    if not HAVE_BLENDER:
        pytest.skip("Blender not installed")
    if not (PROJECT / "shots" / "001" / "shot.json").is_file():
        pytest.skip(f"sample project missing: {PROJECT}")
    dst = tmp_path_factory.mktemp("ufc") / "ufc_cross"
    shutil.copytree(PROJECT, dst, ignore=shutil.ignore_patterns("take", "exports", ".work", ".addon"))
    return dst


def _afv(args, cwd=None):
    t = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "afv.cli", *args], capture_output=True, text=True, cwd=cwd,
                          timeout=1200)
    events = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.startswith("{")]
    return proc, events, time.perf_counter() - t


@pytest.mark.parametrize("name", list(CHARACTERS))
@pytest.mark.blender
@pytest.mark.slow
def test_owner_characters_on_ufc_cross(ufc_copy, name, tmp_path):
    path = CHARACTERS[name]
    if not path.is_file():
        pytest.skip(f"test character missing: {path}")
    saved = tmp_path / f"{name}.afvmap.json"
    proc, events, wall = _afv(["retarget", "001", "-p", str(ufc_copy), "--character", str(path), "--check",
                               "--save-map", str(saved), "--json"])
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = events[-1]
    assert result["event"] == "result" and result["outcome"] == "ok"
    per = {e["role"]: e for e in events if e["event"] == "retarget"}
    assert sorted(per) == ["A", "B"]
    for r, e in per.items():
        c, rep = e["check"], e["report"]
        f = Path(e["file"])
        print(f"{name} {r}: {wall:.1f} s total (retarget {rep['seconds']:.1f} s), round trip max "
              f"{c['max_mm']:.4f} mm world / {c['max_mm_fighter']:.4f} mm at fighter scale, "
              f"{c['frames']} frames, {c['file_fps']:g} fps")
        assert f.parent == ufc_copy / "shots" / "001" / "exports" / "retarget" and f.is_file()
        assert c["bones"] == c["bones_target"] == c["bones_file"]
        assert c["max_mm_fighter"] <= 1.0
        if rep["hips_scale"] < 10:  # a metre-scale character: its world is metres too
            assert c["max_mm"] <= 1.0
        assert c["frames"] == rep["frames"][1] - rep["frames"][0] + 1 == c["file_keys"]
        assert c["file_frames"] == rep["frames"] and c["file_fps"] == pytest.approx(30.0)
        assert fbx_clips(f)["stacks"] == [f.stem] == [rep["action"]]
    assert result["fps"] == pytest.approx(30.0)
    assert json.loads(saved.read_text())["bones"] == result["map"]["bones"]

    base = ["retarget", "001", "-p", str(ufc_copy), "--character", str(path)]
    proc, events, _ = _afv([*base, "--map", str(saved), "--role", "A", "--no-lock-feet",
                            "--out", str(tmp_path / "out"), "--json"])
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert events[-1]["map_source"] == "file" and sorted(events[-1]["files"]) == ["A"]

    data = json.loads(saved.read_text())
    data["bones"].pop("RightUpLeg")
    saved.write_text(json.dumps(data))
    proc, events, _ = _afv([*base, "--map", str(saved), "--json"])
    assert proc.returncode == 1
    err = events[-1]
    assert err["event"] == "error" and err["missing"][0] == "RightUpLeg" and "RightUpLeg" in proc.stderr
