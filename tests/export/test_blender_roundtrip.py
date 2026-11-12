"""Round trip through headless Blender: export FBX + GLB (+ BVH) per preset, re-import, sample.

Checks per file: one armature, no parent/empty above it, exactly the rig's bones (no extra root
node, no leaf bones), exact frame count and fps, and on every frame every bone's world head position
(within 1 mm) and world orientation including leaf bones and twist (within 0.05 degrees) against
afv's FK of the take, which also rules out axis flips and double scaling. The FBX node tree is read
directly: the "Armature" node must carry no transform.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from export_helpers import build_skeleton, fighter_offsets, make_take, provisional_joints

from afv.core.skeletons import SOMA_PROVISIONAL, get_skeleton
from afv.export import blender as bl
from afv.export.exporter import ExportRequest, export_takes
from afv.export.fbx import fbx_summary
from afv.export.glb import accessor, read_glb
from afv.export.mapping import Mapping, get_mapping, register_mapping
from afv.export.presets import PRESETS
from afv.export.retarget import mapped_source_positions, rig_fk
from afv.motion.rotations import from_rotvec, to_matrix, yup_to_zup_quat, zup_to_yup_vec

pytestmark = [
    pytest.mark.blender,
    pytest.mark.slow,
    pytest.mark.skipif(bl.find_blender() is None, reason="Blender not installed"),
]

TOL_M = 1e-3
TOL_DEG = 0.05
N_FRAMES = 45
NTSC = 30000 / 1001
CASES = {  # preset -> (fps, root motion, source skeleton)
    "unity_humanoid": (30.0, "world", "provisional"),
    "unity_generic": (30.0, "extracted", "apose"),
    "unreal": (60.0, "extracted", "soma77"),
    "godot": (30.0, "extracted", "provisional"),
    "blender": (NTSC, "in_place", "soma77"),  # FBX snaps near-standard rates to its time modes
}
SOMA77_FOLDED = ["Neck2"] + [
    f"{s}Hand{f}1" for s in ("Left", "Right") for f in ("Index", "Middle", "Ring", "Pinky")
]


def _skeleton(kind: str):
    if kind == "provisional":
        return SOMA_PROVISIONAL
    if kind == "soma77":
        return get_skeleton("soma77")
    sk = build_skeleton("apose_roundtrip", provisional_joints(), rest_rotations={
        "l_upperarm": from_rotvec([0.0, 0.0, -0.7]), "r_upperarm": from_rotvec([0.0, 0.0, 0.7]),
        "l_forearm": from_rotvec([0.3, 0.4, 0.0]), "spine2": from_rotvec([0.1, 0.2, 0.0]),
        "l_hand": from_rotvec([0.0, 0.5, 0.3]), "r_calf": from_rotvec([0.2, 0.0, 0.0]),
    })
    register_mapping(Mapping(sk.name, dict(get_mapping(SOMA_PROVISIONAL.name).bones)))
    return sk


def _take(sk, seed, fps, role, start, mode):
    take = make_take(sk, N_FRAMES, seed, fps=fps, role=role, start=start, amp=1.0,
                     rest_offsets=fighter_offsets(sk, seed))
    if sk.name == "soma77":  # folded joints at rest: the take then maps exactly (folds: unit tests)
        for n in SOMA77_FOLDED:
            take.rot[:, sk.index(n)] = [0.0, 0.0, 0.0, 1.0]
    take.root_motion = mode
    return take


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory):
    out = tmp_path_factory.mktemp("export_roundtrip")
    exe = bl.find_blender()
    exported, files = {}, []
    for preset, (fps, mode, kind) in CASES.items():
        sk = _skeleton(kind)
        reqs = [ExportRequest(_take(sk, seed, fps, role, start, mode), f"{preset}_{role}", sk)
                for role, seed, start in (("A", 1, (0.0, 0.0)), ("B", 2, (1.5, 2.5)))]
        res = export_takes(reqs, out / preset, preset=preset, formats=["fbx", "glb", "bvh"])
        for req, r in zip(reqs, res, strict=True):
            exported[r.name] = (req, r)
            for fmt in ("fbx", "glb", "bvh"):
                npz = out / "samples" / f"{r.name}_{fmt}.npz"
                files.append({"path": bl.host_path(r.files[fmt], exe), "format": fmt, "fps": fps,
                              "npz": bl.host_path(npz, exe), "_npz": npz, "name": r.name})
    (out / "samples").mkdir()
    job = {"files": [{k: v for k, v in f.items() if not k.startswith("_")} for f in files]}
    result = bl.run_blender("afv_inspect.py", job, out / "inspect")
    by_key = {}
    for f, info in zip(files, result["files"], strict=True):
        by_key[(f["name"], f["format"])] = (info, dict(np.load(f["_npz"])))
    return exported, by_key


def _names():
    return [(f"{p}_{r}", fmt) for p in CASES for r in ("A", "B") for fmt in ("fbx", "glb", "bvh")]


def _sampled(samples, rig):
    names = list(samples["bones"])
    sel = [names.index(n) for n in rig.names]
    return zup_to_yup_vec(samples["heads"])[:, sel], samples["rots"][:, sel]


@pytest.mark.parametrize(("name", "fmt"), _names())
def test_roundtrip_file(roundtrip, name, fmt):
    exported, by_key = roundtrip
    req, r = exported[name]
    info, samples = by_key[(name, fmt)]
    rig = r.motion.rig
    m = r.motion

    # structure: one armature, nothing above it, exactly the rig's bones
    assert len(info["armatures"]) == 1, info["objects"]
    arm = info["armatures"][0]
    assert arm["parent"] is None
    others = [o for o in info["objects"] if o["type"] != "ARMATURE"]
    if fmt == "bvh":
        assert others == []
    else:  # only the skinned proxy mesh, under the armature: no empties, no extra nodes
        assert [(o["type"], o["parent"]) for o in others] == [("MESH", arm["name"])], info["objects"]
    assert sorted(b["name"] for b in arm["bones"]) == sorted(rig.names)
    assert arm["top_bones"] == [rig.names[0]]
    parents = {b["name"]: b["parent"] for b in arm["bones"]}
    for b, p in enumerate(rig.parents):
        assert parents[rig.names[b]] == (rig.names[p] if p >= 0 else None)

    # timing: exact fps and frame count
    if fmt != "glb":  # glTF stores times, not a rate (its key times are checked below)
        assert abs(info["fps"] - m.fps) <= 1e-6 * m.fps  # Blender keeps fps_base as float32
    keys = np.asarray(samples["key_frames"])
    assert keys.shape == (N_FRAMES,)
    if fmt == "fbx":
        assert arm["action"]["keys_min"] == arm["action"]["keys_max"] == N_FRAMES
        # Blender's FBX importer maps key times to frames with the integer scene fps
        times = keys / info["scene_fps_int"]
    elif fmt == "bvh":
        times = (keys - 1.0) / info["fps"]  # Blender's BVH importer clamps the start frame to 1
    else:
        times = keys / m.fps  # imported with fps * fps_base = the real rate (see afv_inspect.py)
    assert np.allclose(times, np.arange(N_FRAMES) / m.fps, atol=1e-4 / m.fps), times

    # positions and orientations on every frame vs afv FK of the export motion and the source take
    heads, rots = _sampled(samples, rig)
    pos, wrot = rig_fk(m)
    err = np.linalg.norm(heads - pos, axis=-1)
    assert err.max() <= TOL_M, f"{name}.{fmt}: max error {err.max() * 1000:.3f} mm"
    want = to_matrix(yup_to_zup_quat(wrot))
    cos = (np.trace(np.swapaxes(rots, -1, -2) @ want, axis1=-2, axis2=-1) - 1.0) / 2.0
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    worst = rig.names[int(np.argmax(ang.max(0)))]
    assert ang.max() <= TOL_DEG, f"{name}.{fmt}: orientation error {ang.max():.4f} deg at {worst}"
    src = mapped_source_positions(rig, np.asarray(req.take.rot), np.asarray(req.take.root))
    if m.root_motion == "in_place":
        h = rig.hips
        src = src + (pos[:, h : h + 1] - src[:, h : h + 1])  # the frozen horizontal hips shift
    ok = ~np.isnan(src[0, :, 0])
    err_src = np.linalg.norm(heads[:, ok] - src[:, ok], axis=-1)
    assert err_src.max() <= TOL_M, f"{name}.{fmt}: max error vs source FK {err_src.max() * 1000:.3f} mm"
    print(f"{name}.{fmt}: pos max {err.max() * 1000:.4f} mm, orient max {ang.max():.4f} deg")


@pytest.mark.parametrize("preset", list(CASES))
def test_glb_timing_and_nodes(roundtrip, preset):
    exported, _ = roundtrip
    _, r = exported[f"{preset}_A"]
    doc, binary = read_glb(r.files["glb"])
    fps = r.motion.fps
    anims = doc["animations"]
    assert len(anims) == 1
    for s in anims[0]["samplers"]:
        t = accessor(doc, binary, s["input"])
        assert t.shape[0] == N_FRAMES
        assert np.allclose(t, np.arange(N_FRAMES) / fps, atol=1e-5)
    skin = doc["skins"][0]
    joint_names = [doc["nodes"][j]["name"] for j in skin["joints"]]
    assert sorted(joint_names) == sorted(r.motion.rig.names)
    roots = [doc["nodes"][n]["name"] for n in doc["scenes"][0]["nodes"]]
    assert "Armature" not in roots and r.motion.rig.names[0] in roots


@pytest.mark.parametrize("preset", list(CASES))
def test_fbx_node_tree(roundtrip, preset):
    """No rotated or scaled node above the skeleton: the 'Armature' Null is an identity transform."""
    exported, _ = roundtrip
    _, r = exported[f"{preset}_A"]
    s = fbx_summary(r.files["fbx"])
    gs = s.global_settings
    assert (gs["UpAxis"][0], gs["FrontAxis"][0], gs["FrontAxisSign"][0]) == (1, 2, 1)  # Y up, +Z front
    unit_cm = gs["UnitScaleFactor"][0]
    assert unit_cm == pytest.approx(1.0 if preset == "unreal" else 100.0, rel=1e-6)  # x100 once
    armature = s.models["Armature"]
    assert armature.kind == "Null" and armature.parent is None
    for m in (armature, s.models["Proxy"]):
        assert np.allclose(m.translation, 0.0, atol=1e-6), m
        assert np.allclose(m.rotation, 0.0, atol=1e-4), m  # degrees; float32 round-off
        assert np.allclose(m.scaling, 1.0, atol=1e-6), m
    top = r.motion.rig.names[0]
    assert s.models[top].kind == "LimbNode" and s.models[top].parent == "Armature"
    for b, p in enumerate(r.motion.rig.parents):
        if p >= 0:
            assert s.models[r.motion.rig.names[b]].parent == r.motion.rig.names[p]


def test_pair_shares_origin(roundtrip):
    """Attacker and defender are separate files; their world spacing equals the source's on every
    frame and joint (world-mode preset, so nothing is re-centred or moved onto a root)."""
    exported, by_key = roundtrip
    gaps = {}
    for role in ("A", "B"):
        req, r = exported[f"unity_humanoid_{role}"]
        heads, _ = _sampled(by_key[(f"unity_humanoid_{role}", "fbx")][1], r.motion.rig)
        src = mapped_source_positions(r.motion.rig, np.asarray(req.take.rot), np.asarray(req.take.root))
        gaps[role] = (heads, src)
    got = gaps["B"][0] - gaps["A"][0]
    want = gaps["B"][1] - gaps["A"][1]
    assert np.abs(got - want).max() <= 2 * TOL_M
    assert np.linalg.norm(want[0, 0, [0, 2]]) > 2.0  # the fighters really are apart


def test_every_preset_is_round_tripped():
    assert set(CASES) == set(PRESETS)


def _blender_processes_with(marker: str) -> list[str]:
    cmd = ["powershell.exe", "-NoProfile", "-Command",
           "Get-CimInstance Win32_Process -Filter \"Name='blender.exe'\" | ForEach-Object { $_.CommandLine }"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("cannot list Windows processes")
    return [line for line in out.splitlines() if marker in line]


def test_timeout_raises_and_stops_blender(tmp_path, monkeypatch):
    """A hung Blender is stopped after the timeout (WSL interop) and reported as BlenderError."""
    script = tmp_path / "afv_sleep_test.py"
    script.write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setattr(bl, "script_path", lambda name: script)
    work = tmp_path / "hung_job"
    t0 = time.monotonic()
    with pytest.raises(bl.BlenderError, match="timed out"):
        bl.run_blender("afv_sleep_test.py", {}, work, timeout=8)
    assert time.monotonic() - t0 < 30
    time.sleep(3)
    assert _blender_processes_with("hung_job") == []


def test_missing_wslpath_is_a_blender_error(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise FileNotFoundError("wslpath")

    monkeypatch.setattr(bl, "_needs_wslpath", lambda blender: True)
    monkeypatch.setattr(bl.subprocess, "run", boom)
    with pytest.raises(bl.BlenderError, match="wslpath"):
        bl.host_path(tmp_path, Path("blender.exe"))
