"""The take on the real shots under ``projects/`` (git-ignored; skipped when absent), or under
``$AFV_REAL_PROJECTS`` (e.g. a copy, while other work rewrites ``projects/``).

Per shot, with the shot's own results (nothing in the project is written; the take goes to tmp):
- overlay: every rig joint on every frame through the Blender camera (``world_to_camera_view``,
  scene resolution) vs the pipeline's own projection of the same 3D joints with the CameraTrack
  (``afv.qc.camera.project``, world shift applied): <= 1 px;
- the take's evaluated joints vs the exported FBX for the same shot, re-imported in Blender and
  matched by bone name: <= 1 mm. The take's rig settings must equal the export stage's resolved
  params of that export, and the export must be of the same motion; otherwise the test fails and
  says why (re-export the shot) instead of skipping.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from take_helpers import REPO

from afv import take as tk
from afv.core.project import Project
from afv.export import blender as bl
from afv.export.presets import get_preset

pytestmark = [
    pytest.mark.blender,
    pytest.mark.slow,
    pytest.mark.skipif(bl.find_blender() is None, reason="Blender not installed"),
]

PROJECTS = Path(os.environ.get("AFV_REAL_PROJECTS") or REPO / "projects")
SHOTS = [("close_boxing", "001"), ("ufc_cross", "001"), ("who_am_i", "001"), ("ref_walk", "001"),
         ("ref_walk", "002")]
AVAILABLE = [s for s in SHOTS if (PROJECTS / s[0] / "shots" / s[1] / "shot.json").is_file()]


def _shot(project, shot_id):
    return Project(PROJECTS / project).shot(shot_id)


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    if not AVAILABLE:
        pytest.skip("no real projects")
    root = tmp_path_factory.mktemp("real_takes")
    exe = bl.find_blender()
    out, inspect_files = {}, []
    for project, shot_id in AVAILABLE:
        sh = _shot(project, shot_id)
        key = f"{project}_{shot_id}"
        tj = tk.make_take(sh, root / key / "take")
        samples = root / key / "samples.npz"
        built = tk.build_blend(tj, None, samples_npz=samples)
        rep = tk.check_alignment(tj, samples, sh.load_result("camera"))
        meta = json.loads(tj.read_text())
        params_problems, export_problems = _export_problems(sh, meta)
        fbx = {}
        for f in meta["fighters"]:
            p = sh.exports_dir / f"{f['name']}.fbx"
            if not export_problems and p.is_file():
                npz = root / key / f"fbx_{f['role']}.npz"
                inspect_files.append({"path": bl.host_path(p, exe), "format": "fbx",
                                      "fps": meta["shot"]["fps"], "npz": bl.host_path(npz, exe)})
                fbx[f["role"]] = npz
        out[key] = {"take": tj, "samples": samples, "check": rep, "fbx": fbx, "meta": meta,
                    "params_problems": params_problems, "export_problems": export_problems,
                    "summary": built["summary"], "camera": meta["camera"],
                    "frames": meta["shot"]["frame_count"]}
    if inspect_files:
        bl.run_blender("afv_inspect.py", {"files": inspect_files}, root / "inspect")
    return out


def _export_problems(sh, meta):
    """Why the shot's current export is not the take's rig and motion: (rig params, everything)."""
    em = sh.meta("export")
    if em is None:
        return ["no export result"], ["no export result"]
    preset = get_preset(em["params"]["preset"])
    p = em["params"]
    mode = p.get("root_motion") or preset.root_motion
    want = {"preset": preset.name, "prefix": preset.prefix if p.get("prefix") is None else p["prefix"],
            "fingers": p.get("fingers", True), "root_motion": "world" if mode == "in_place" else mode}
    rig = meta["rig"]
    params = [f"{k}: take {rig.get(k)!r} vs export {v!r}" for k, v in want.items() if rig.get(k) != v]
    problems = list(params)
    stage = meta["sources"]["motion"]["stage"]
    if (em.get("inputs") or {}).get(stage) != meta["sources"]["motion"]["output_hash"]:
        problems.append(f"the export was made from another {stage} result than the take's")
    files = set(em.get("exports") or [])
    for f in meta["fighters"]:
        if f"{f['name']}.fbx" not in files:
            problems.append(f"no {f['name']}.fbx among the exports {sorted(files)}")
        elif not (sh.exports_dir / f"{f['name']}.fbx").is_file():
            problems.append(f"{f['name']}.fbx is missing from {sh.exports_dir}")
    return params, problems


def _heads_by_name(npz, names, prefix=""):
    """Bone heads [F,B,3] (Blender world) of a sample file, in the order of ``names``."""
    with np.load(npz) as z:
        bones = [str(b) for b in z[prefix + "bones"]]
        return z[prefix + "heads"][:, [bones.index(n) for n in names]]


@pytest.mark.parametrize(("project", "shot_id"), AVAILABLE)
def test_overlay_alignment(results, project, shot_id):
    r = results[f"{project}_{shot_id}"]["check"]
    for role, c in r["fighters"].items():
        print(f"{project}/{shot_id} {role}: {r['frames']} frames, overlay max {c['px_max']:.4f} px "
              f"mean {c['px_mean']:.5f} px, joints vs take max {c['max_mm']:.4f} mm")
        assert c["px_points"] == c["bones"] * r["frames"]
        assert c["px_max"] <= 1.0
        assert c["max_mm"] <= 1.0


@pytest.mark.parametrize(("project", "shot_id"), AVAILABLE)
def test_take_rig_is_the_export_stages(results, project, shot_id):
    """The take's rig settings and file stems equal what the export stage resolved for this shot."""
    res = results[f"{project}_{shot_id}"]
    assert not res["params_problems"], res["params_problems"]
    assert res["meta"]["take_id"].count(":") == 2  # <project>:<shot>:<project uid>
    print(f"{project}/{shot_id}: rig {res['meta']['rig']}, stale: {res['meta']['stale'] or 'none'}")


@pytest.mark.parametrize(("project", "shot_id"), AVAILABLE)
def test_take_matches_exported_fbx(results, project, shot_id):
    res = results[f"{project}_{shot_id}"]
    if res["export_problems"]:
        pytest.fail(f"{project}/{shot_id}: the exports cannot be compared with the take (re-export the "
                    f"shot): {'; '.join(res['export_problems'])}")
    meta, _ = tk.load_take_arrays(res["take"])
    for f in meta["fighters"]:
        names = [b["name"] for b in f["bones"]]
        fbx = _heads_by_name(res["fbx"][f["role"]], names)  # FBX re-imported, Blender world
        want = _heads_by_name(res["samples"], names, f"{f['role']}/")  # the take scene, evaluated
        assert fbx.shape == want.shape, (fbx.shape, want.shape)
        err = np.linalg.norm(fbx - want, axis=-1)
        print(f"{project}/{shot_id} {f['role']}: take vs FBX max {err.max() * 1000:.4f} mm "
              f"mean {err.mean() * 1000:.4f} mm over {err.shape[0]} frames x {err.shape[1]} bones")
        assert err.max() <= 1e-3


@pytest.mark.parametrize(("project", "shot_id"), AVAILABLE)
def test_video_timing_and_markers(results, project, shot_id):
    """The proxy is the camera background frame for frame, the scene is the shot's rate and size."""
    res = results[f"{project}_{shot_id}"]
    s, cam, n = res["summary"], res["camera"], res["frames"]
    assert s["fps"] == pytest.approx(30.0)
    assert (s["frame_start"], s["frame_end"]) == (0, n - 1)
    assert s["resolution"] == [cam["width"], cam["height"]]
    c = s["camera"]
    assert c["video"] and c["clip_frames"] == n and c["clip_frame_start"] == 0
    assert c["clip_size"][0] / c["clip_size"][1] == pytest.approx(cam["width"] / cam["height"], rel=0.01)
    assert s["markers"] == s["spans"]
    assert {f["bones"] for f in s["fighters"].values()} == {65}
