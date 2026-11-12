"""Camera conventions: GEM-X/OpenCV <-> afv CameraTrack (OpenGL, camera-to-world), proven by
reprojecting converted joints onto GEM-X's own image geometry; plus world levelling and ground."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from afv.core.skeletons import get_skeleton
from afv.core.stage import Stage, get_stage
from afv.core.types import CameraTrack, load_result, save_result
from afv.models import camera as camera_mod
from afv.models.camera import (
    CV_TO_GL,
    average_up,
    camera_to_world_cv,
    gemx_provided_camera,
    level_rotation,
    leveled_track,
    make_camera_track,
    place_on_ground,
    project,
    run_env_worker,
    shifted_camera,
)
from afv.models.gemx import SomaFK, body_motion_from_fk, camera_from_gemx, gemx_up_in_world
from afv.motion.fk import forward_kinematics
from afv.motion.rotations import to_matrix

FIX = Path(__file__).parent / "fixtures"
FIXTURES = sorted(FIX.glob("gemx_*.npz"))


@pytest.fixture(params=FIXTURES, ids=lambda p: p.stem)
def fk(request) -> SomaFK:
    return SomaFK.load(request.param)


def _rot_x(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


# --- reprojection onto GEM-X's image geometry ---------------------------------------------------


def _world_joints(fk: SomaFK, cam: CameraTrack):
    motion, _ = body_motion_from_fk({"A": fk}, cam)
    a = motion.actors["A"]
    sk = get_skeleton("soma77")
    pos, _ = forward_kinematics(sk.with_offsets(a.rest_offsets.astype(np.float64)), a.rot, a.root)
    return motion, pos


def _gemx_overlay(fk: SomaFK, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """GEM-X's overlay pixels: K @ incam joints, identity extrinsics (gem.inference.render)."""
    x = fk.incam_pos
    return np.stack(
        [
            fx[:, None] * x[..., 0] / x[..., 2] + fk.K[:, None, 0, 2],
            fy[:, None] * x[..., 1] / x[..., 2] + fk.K[:, None, 1, 2],
        ],
        axis=-1,
    )


def test_reprojection_matches_gemx_overlay_and_2d(fk):
    """World joints from our FK, projected with the CameraTrack (moved by world_shift), land where
    GEM-X draws them."""
    cam = camera_from_gemx(fk)
    motion, pos = _world_joints(fk, cam)
    uv, depth = project(shifted_camera(cam, motion.world_shift), pos)
    assert (depth > 0).all()
    # exact against the square-pixel K the CameraTrack holds ...
    f = 0.5 * (fk.K[:, 0, 0] + fk.K[:, 1, 1])
    assert np.abs(uv - _gemx_overlay(fk, f, f)).max() < 0.01
    # ... and within the fx/fy rounding against GEM-X's own K (fx, fy separately)
    assert np.abs(uv - _gemx_overlay(fk, fk.K[:, 0, 0], fk.K[:, 1, 1])).max() < 0.5
    # GEM-X's ViTPose77 2D keypoints (same joint order as soma77): model fit error only.
    good = fk.kp2d[..., 2] > 0.5
    err = np.linalg.norm(uv - fk.kp2d[..., :2], axis=-1)[good]
    assert np.median(err) < 25.0
    # Wrong conventions are far off: a mirrored image or an up-down flip.
    mirrored = uv.copy()
    mirrored[..., 0] = 2 * cam.cx - mirrored[..., 0]
    flipped = uv.copy()
    flipped[..., 1] = 2 * cam.cy - flipped[..., 1]
    for wrong in (mirrored, flipped):
        assert np.median(np.linalg.norm(wrong - fk.kp2d[..., :2], axis=-1)[good]) > 4 * np.median(err)


def test_reprojection_independent_of_camera_helpers(fk):
    """Same check written from the CameraTrack contract alone (scipy quaternion, OpenGL pinhole),
    without camera_to_world_cv / world_to_camera_cv / project."""
    from scipy.spatial.transform import Rotation

    cam = camera_from_gemx(fk)
    motion, pos = _world_joints(fk, cam)
    r_gl = Rotation.from_quat(np.asarray(cam.cam_rot, np.float64)).as_matrix()  # camera-to-world
    centre = np.asarray(cam.cam_pos, np.float64) + np.asarray(motion.world_shift)
    x = np.einsum("fba,fnb->fna", r_gl, pos - centre[:, None])  # R^T (p - c), OpenGL camera axes
    z = -x[..., 2]  # the camera looks down -Z
    assert (z > 0).all()
    f = np.asarray(cam.focal_px, np.float64)[:, None]
    u = cam.cx + f * x[..., 0] / z
    v = cam.cy - f * x[..., 1] / z  # +Y up in the camera = up in the image
    ref = _gemx_overlay(fk, fk.K[:, 0, 0], fk.K[:, 1, 1])
    assert np.hypot(u - ref[..., 0], v - ref[..., 1]).max() < 0.5


def test_shifted_camera_keeps_projection():
    cam = leveled_track(
        np.stack([_rot_x(-10)] * 3),
        None,
        np.full(3, 1200.0),
        up_v=np.array([0, -1.0, 0]),
        width=1280,
        height=720,
        cx=640,
        cy=360,
    )
    pts = np.tile([[0.3, -1.2, -6.0], [-0.5, 0.0, -9.0]], (3, 1, 1))
    shift = np.array([0.2, 0.7, -0.1])
    moved = shifted_camera(cam, shift)
    assert np.allclose(project(moved, pts + shift)[0], project(cam, pts)[0])
    assert np.allclose(cam.cam_pos, 0.0)  # the original is not modified
    assert moved.ground_offset == pytest.approx(cam.ground_offset - shift[1])


def test_gemx_provided_camera_preserves_angular_velocity(fk):
    """GEM-X only sees relative camera rotations: ours must equal the ones GEM-X was given."""
    cam = camera_from_gemx(fk)
    r_w2c, k = gemx_provided_camera(cam)
    rel_ours = np.einsum("tab,cb->tac", r_w2c.astype(np.float64), r_w2c[0].astype(np.float64))
    rel_gemx = np.einsum("tab,cb->tac", fk.R_w2c, fk.R_w2c[0])
    assert np.abs(rel_ours - rel_gemx).max() < 1e-5
    f = np.asarray(cam.focal_px, dtype=np.float32)
    assert k.shape == (cam.num_frames, 3, 3)
    assert np.array_equal(k[:, 0, 0], f) and np.array_equal(k[:, 1, 1], f)
    assert np.all(k[:, 0, 2] == np.float32(cam.cx)) and np.all(k[:, 2] == [0.0, 0.0, 1.0])


def test_world_is_gravity_aligned_with_gemx(fk):
    """A world levelled with GEM-X's mean gravity: GEM-X's per-frame up is within a few degrees."""
    cam = camera_from_gemx(fk, up="gemx")
    up = gemx_up_in_world(fk, cam)
    ang = np.degrees(np.arccos(np.clip(up[:, 1], -1, 1)))
    assert np.median(ang) < 6.0
    sk = get_skeleton("soma77")
    motion, _ = body_motion_from_fk({"A": fk}, cam)
    a = motion.actors["A"]
    pos, _ = forward_kinematics(sk.with_offsets(a.rest_offsets.astype(np.float64)), a.rot, a.root)
    head_above_feet = pos[:, sk.index("Head"), 1] - pos[:, sk.index("LeftFoot"), 1]
    assert np.median(head_above_feet) > 1.0  # upright fighters in a Y-up world


# --- CameraTrack convention -----------------------------------------------------------------------


def test_camera_track_looks_down_minus_z_with_y_up():
    r_c2w_cv = _rot_y(30) @ CV_TO_GL  # level camera, yawed 30 degrees
    cam = make_camera_track(
        np.stack([r_c2w_cv]),
        np.array([1.0, 2.0, 3.0]),
        np.array([1000.0]),
        width=1280,
        height=720,
        cx=640,
        cy=360,
    )
    r_gl = to_matrix(cam.cam_rot)[0]
    ahead = cam.cam_pos[0] + r_gl @ np.array([0, 0, -5.0])
    above = ahead + r_gl @ np.array([0, 1.0, 0])
    right = ahead + r_gl @ np.array([1.0, 0, 0])
    uv, depth = project(cam, np.stack([[ahead, above, right]]))
    assert np.allclose(uv[0, 0], [640, 360]) and depth[0, 0] == pytest.approx(5.0)
    assert uv[0, 1, 1] < 360 and uv[0, 2, 0] > 640  # camera +Y is image up, +X image right
    assert np.allclose(r_gl[:, 1], [0, 1, 0])  # level camera: its up is world up


def test_level_rotation_is_right_handed_and_yaw_zero():
    up = np.array([0.1, -0.95, -0.2])  # tilted OpenCV-style world
    fwd = np.array([0.0, 0.2, 1.0])
    r = level_rotation(up, fwd)
    assert np.allclose(r @ r.T, np.eye(3)) and np.linalg.det(r) == pytest.approx(1.0)
    assert np.allclose(r @ (up / np.linalg.norm(up)), [0, 1, 0])
    f = r @ fwd
    assert abs(f[0]) < 1e-9 and f[2] < 0  # camera 0 looks along world -Z


def test_static_level_camera_is_identity_in_world():
    cam = leveled_track(
        np.tile(np.eye(3), (4, 1, 1)),
        None,
        np.full(4, 1500.0),
        up_v=None,
        width=1280,
        height=720,
        cx=640,
        cy=360,
        static=True,
    )
    assert np.allclose(np.abs(cam.cam_rot), [0, 0, 0, 1])
    assert cam.static and np.allclose(cam.timestamps, np.arange(4) / 30)


def test_tilted_camera_is_levelled_by_gravity():
    # true world W: a camera pitched 12 deg down and panning; the solver's world V is W rotated
    # arbitrarily (as a SLAM world is); gravity is observed in each camera frame.
    n = 20
    r_w_cv = np.stack([_rot_y(1.5 * i) @ _rot_x(-12) @ CV_TO_GL for i in range(n)])
    q = _rot_y(70) @ _rot_x(35)
    r_v_cv = np.einsum("ab,nbc->nac", q, r_w_cv)
    up_v = q @ np.array([0, 1.0, 0])
    up_cam = np.einsum("nba,b->na", r_v_cv, up_v)  # R^T up
    est = average_up(r_v_cv, up_cam, np.arange(n))
    assert np.allclose(est, up_v, atol=1e-9)
    cam = leveled_track(r_v_cv, None, np.full(n, 1200.0), up_v=est, width=1280, height=720, cx=640, cy=360)
    r_gl = to_matrix(cam.cam_rot)
    view = -r_gl[:, :, 2]
    pitch = np.degrees(np.arcsin(view[:, 1]))
    assert np.allclose(pitch, -12.0, atol=1e-6)  # recovered tilt on every frame
    assert np.allclose(r_gl[:, 1, 0], 0, atol=1e-9)  # no roll: camera x stays horizontal


def test_place_on_ground_recovers_height_and_origin():
    h, dist, height = 3.0, 10.0, 1.65
    r_v_cv = np.stack([_rot_x(-10)] * 5)  # pitched down, looking along -Z after levelling
    cam = leveled_track(
        r_v_cv, None, np.full(5, 1400.0), up_v=np.array([0, -1.0, 0]), width=1280, height=720, cx=640, cy=360
    )
    cam.cam_pos = np.tile([0.0, h, 0.0], (5, 1))  # ground at y = 0 for the synthetic scene
    feet = np.array([[0.0, 0.0, -dist]])  # on the centre column: the box's vertical is exact
    head = feet + np.array([0.0, height, 0.0])
    uv_f, _ = project(cam, np.repeat(feet[None], 5, 0))
    uv_h, _ = project(cam, np.repeat(head[None], 5, 0))
    boxes = np.concatenate([uv_f[..., :1] - 40, uv_h[..., 1:], uv_f[..., :1] + 40, uv_f[..., 1:]], axis=-1)
    cam.cam_pos = np.zeros((5, 3))  # forget the height; re-estimate it
    cam, info = place_on_ground(cam, boxes, person_height=height)
    assert info["method"] == "boxes_person_height_prior"
    assert cam.cam_pos[0] == pytest.approx([0.0, h, dist], abs=1e-6)  # origin under the person
    assert cam.ground_normal == [0.0, 1.0, 0.0] and cam.ground_offset == 0.0


def test_place_on_ground_without_boxes_uses_default_height():
    cam = leveled_track(
        np.tile(np.eye(3), (2, 1, 1)),
        None,
        np.full(2, 1000.0),
        up_v=None,
        width=640,
        height=480,
        cx=320,
        cy=240,
    )
    cam, info = place_on_ground(cam, np.full((2, 1, 4), np.nan), default_camera_height=1.4)
    assert info["method"] == "default_camera_height" and cam.cam_pos[0, 1] == pytest.approx(1.4)


def test_camera_track_round_trip(tmp_path, fk):
    cam = camera_from_gemx(fk)
    save_result(cam, tmp_path / "c.npz")
    back = load_result(tmp_path / "c.npz")
    assert isinstance(back, CameraTrack)
    r0, c0 = camera_to_world_cv(cam)
    r1, c1 = camera_to_world_cv(back)
    assert np.abs(r0 - r1).max() < 1e-6 and np.abs(c0 - c1).max() < 1e-6


# --- stages and the worker bridge -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "impl"), [("camera", "vipe"), ("camera", "static"), ("camera", "background"), ("pose3d", "gemx")]
)
def test_stages_registered(name, impl):
    import afv.models.gemx  # noqa: F401  (registers pose3d)

    cls = get_stage(name, impl)
    assert cls.name == name and isinstance(cls(), Stage)
    cls.Params()  # defaults are valid
    assert all(lic.commercial_output_ok for lic in cls.licences)


def test_run_env_worker_uses_afv_bridge(tmp_path, monkeypatch):
    """C's stages call workers through module A's bridge: argv request/response, AFV_PROGRESS lines."""
    from afv.models import envs

    script = tmp_path / "echo_worker.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys, pathlib",
                "req = json.loads(pathlib.Path(sys.argv[1]).read_text())",
                "print('AFV_PROGRESS -1 [echo] hello', flush=True)",
                "if req.get('fail'): sys.exit(3)",
                "pathlib.Path(sys.argv[2]).write_text(json.dumps({'x2': req['x'] * 2}))",
            ]
        )
    )
    fake = {"fake": envs.EnvSpec("fake", Path(sys.executable))}
    monkeypatch.setattr(envs, "load_envs", lambda path=None: fake)
    seen = []
    out = run_env_worker(
        "fake",
        str(script),
        {"command": "t", "x": 21},
        tmp_path,
        gpu=False,
        progress=lambda f, m: seen.append((f, m)),
    )
    assert out == {"x2": 42} and seen == [(-1.0, "[echo] hello")]
    assert json.loads((tmp_path / "worker/echo_worker_t.request.json").read_text())["x"] == 21
    from afv.core.stage import StageError

    with pytest.raises(StageError, match="exit code 3"):
        run_env_worker("fake", str(script), {"command": "t", "x": 1, "fail": True}, tmp_path, gpu=False)


def test_workers_do_not_import_afv():
    for f in (Path(camera_mod.__file__).parent / "workers").glob("*_worker.py"):
        if f.stem in ("gemx_worker", "soma_worker", "vipe_worker"):
            src = f.read_text()
            assert "import afv" not in src and "from afv" not in src, f.name
