"""The golden comparison itself (no GPU): tolerances by key pattern, exact keys, nested values."""

from __future__ import annotations

from golden import compare


def test_compare_tolerances_and_exact_keys():
    want = {
        "ingest.frames": 92,
        "ingest.shot_mp4_sha256": "ab",
        "camera.focal_px": 7300.0,
        "detect_track.box_center_median.1": [250.0, 380.0],
        "export.files": {"A": ["a.fbx", "a.glb"]},
    }
    same = {**want, "camera.focal_px": 7300.0 * 1.02, "detect_track.box_center_median.1": [251.5, 378.5]}
    assert compare(same, want) == []
    bad = {**want, "ingest.frames": 93, "camera.focal_px": 7300.0 * 1.05}
    diffs = compare(bad, want)
    assert len(diffs) == 2 and diffs[0].startswith("camera.focal_px") and "exact" in diffs[1]
    assert compare({**want, "export.files": {"A": ["a.fbx"]}}, want)
    assert compare({k: v for k, v in want.items() if k != "ingest.frames"}, want) == [
        "ingest.frames: missing (golden 92)"
    ]
    assert "new key" in compare({**want, "x": 1}, want)[0]
    assert compare({**want, "camera.focal_px": float("nan")}, want)


def test_fk_summary_on_a_rest_pose():
    import numpy as np
    from golden import FK_EFFECTORS, FK_ROTATIONS, fk_summary

    from afv.core.skeletons import get_skeleton
    from afv.core.types import ActorMotion, BodyMotion
    from afv.motion.fk import rest_positions

    sk = get_skeleton("soma77")
    n = 4
    rot = np.tile([0.0, 0.0, 0.0, 1.0], (n, sk.num_joints, 1)).astype(np.float32)
    root = np.tile(sk.rest_offsets[0], (n, 1)).astype(np.float32)
    bm = BodyMotion(frames=np.arange(n), timestamps=np.arange(n) / 30.0, skeleton="soma77", fps=30.0,
                    actors={"A": ActorMotion(rot=rot, root=root)})
    s = fk_summary(bm)
    rest = rest_positions(sk)
    for e in FK_EFFECTORS:
        want = rest[sk.index(e)] - rest[0]
        assert np.allclose(s[f"fk.A.{e}.rel_mean_m"], want, atol=1e-3)
        assert abs(s[f"fk.A.{e}.rel_p95_m"] - np.linalg.norm(want)) < 1e-3
    for j in FK_ROTATIONS:
        assert s[f"fk.A.{j}.world_angle_mean_deg"] == 0.0
    assert compare(s, s) == []
