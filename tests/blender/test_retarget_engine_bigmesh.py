"""The retarget engine with big skinned meshes (~1M and ~3.4M vertices, 30 % of them in dense feet,
at least 150k below the ankles): positions are read in bulk and vertex groups only for the sole band,
so one fighter (300 frames) stays within 10 s, locked."""

from __future__ import annotations

import pytest
from retarget_helpers import dump, run_checks
from take_helpers import synthetic_shot

from afv import take as tk
from afv.export import blender as bl

pytestmark = [
    pytest.mark.blender,
    pytest.mark.slow,
    pytest.mark.skipif(bl.find_blender() is None, reason="Blender not installed"),
]


@pytest.mark.parametrize("verts", [1_000_000, 3_400_000])
def test_big_mesh_dense_feet(tmp_path, verts):
    exe = bl.find_blender()
    bm, cam = synthetic_shot(n_frames=300, seed=3)
    take = tk.write_take(tmp_path / "take", tk.TakeInputs(motion=bm, camera=cam, shot_id="007",
                                                          project_dir=str(tmp_path / "proj")))
    r = run_checks("bigmesh", {"take": bl.host_path(take, exe), "verts": verts, "feet_share": 0.3},
                   tmp_path / "work")
    dump(f"bigmesh_{verts}", r)
    print("bigmesh", r)
    assert r["verts"] >= 0.99 * verts and r["below_ankle"] >= 150_000
    assert r["locked"]["seconds"] <= 10.0 and r["free"]["seconds"] <= 10.0, r
