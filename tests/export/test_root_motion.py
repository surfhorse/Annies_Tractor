from __future__ import annotations

import numpy as np
import pytest
from export_helpers import single_joint_motion, synthetic_motion

from afv.core.skeletons import SOMA_PROVISIONAL
from afv.export.retarget import retarget, rig_fk
from afv.export.rig import build_rig
from afv.export.root_motion import apply_root_motion
from afv.motion.rotations import from_rotvec, qmul


@pytest.fixture
def world_motion():
    rig = build_rig(SOMA_PROVISIONAL, root_bone=True)
    rot, root = synthetic_motion(SOMA_PROVISIONAL, 60, seed=11, turn=np.pi)
    return retarget(rig, rot, root, 30.0)


@pytest.mark.parametrize("smooth_s", [0.0, 0.25])
def test_extracted_keeps_pose_and_puts_track_on_root(world_motion, smooth_s):
    m = apply_root_motion(world_motion, "extracted", smooth_s=smooth_s, yaw_smooth_s=smooth_s)
    pos_w, _ = rig_fk(world_motion)
    pos, _ = rig_fk(m)
    assert np.abs(pos[:, 1:] - pos_w[:, 1:]).max() < 1e-9  # every joint unchanged in world
    root_pos = m.loc[:, 0]
    assert np.allclose(root_pos[:, 1], 0.0)  # on the floor; vertical motion stays on Hips
    assert np.allclose(m.rot[:, 0, [0, 2]], 0.0)  # pure yaw about +Y
    h = m.rig.hips
    if smooth_s == 0.0:
        assert np.allclose(root_pos[:, [0, 2]], pos_w[:, h][:, [0, 2]])
        left = pos_w[:, m.rig.index("LeftUpLeg")] - pos_w[:, m.rig.index("RightUpLeg")]
        d = np.angle(np.exp(1j * (_root_yaw(m) - np.arctan2(-left[:, 2], left[:, 0]))))
        assert np.abs(d).max() < 1e-9  # unsmoothed: exactly the hip line's heading
        assert np.allclose(m.loc[:, h, [0, 2]], 0.0, atol=1e-9)  # hips right above root
    else:
        # smoothed track is smoother than the raw hips track
        acc_raw = np.abs(np.diff(pos_w[:, h, 0], 2)).mean()
        assert np.abs(np.diff(root_pos[:, 0], 2)).mean() < acc_raw
    # the root covers the world displacement (shared origin: nothing re-centred)
    assert np.linalg.norm(root_pos[-1] - root_pos[0]) > 1.0


def test_in_place_freezes_horizontal_hips(world_motion):
    m = apply_root_motion(world_motion, "in_place")
    h = m.rig.hips
    pos_w, _ = rig_fk(world_motion)
    pos, _ = rig_fk(m)
    assert np.allclose(pos[:, h, [0, 2]], pos_w[0, h, [0, 2]])
    assert np.allclose(pos[:, h, 1], pos_w[:, h, 1])  # vertical kept
    rel_w = pos_w - pos_w[:, h : h + 1]
    rel = pos - pos[:, h : h + 1]
    assert np.abs(rel[:, 1:] - rel_w[:, 1:]).max() < 1e-9  # pose unchanged relative to hips


def test_world_mode_and_errors(world_motion):
    m = apply_root_motion(world_motion, "world")
    assert np.array_equal(m.loc, world_motion.loc)
    with pytest.raises(ValueError):
        apply_root_motion(world_motion, "sideways")
    with pytest.raises(ValueError):
        apply_root_motion(m.copy(root_motion="in_place"), "world")
    rig = build_rig(SOMA_PROVISIONAL, root_bone=False)
    rot, root = synthetic_motion(SOMA_PROVISIONAL, 10, seed=1)
    with pytest.raises(ValueError, match="root bone"):
        apply_root_motion(retarget(rig, rot, root, 30.0), "extracted")


# --- heading robustness (review r1) ---------------------------------------------------------------

F = 60
SK = SOMA_PROVISIONAL


def _root_yaw(m) -> np.ndarray:
    return np.unwrap(2 * np.arctan2(m.rot[:, 0, 1], m.rot[:, 0, 3]))


def _extract(pelvis: np.ndarray, root: np.ndarray, **kw) -> np.ndarray:
    rig = build_rig(SK, root_bone=True)
    rot = single_joint_motion(SK, {"pelvis": pelvis}, root)
    world = retarget(rig, rot, root, 30.0)
    m = apply_root_motion(world, "extracted", **kw)
    assert np.abs(rig_fk(m)[0][:, 1:] - rig_fk(world)[0][:, 1:]).max() < 1e-9  # pose untouched
    return np.degrees(_root_yaw(m))


def _rv(x, y, z):
    return from_rotvec(np.stack(np.broadcast_arrays(x, y, z), axis=-1).astype(float))


@pytest.mark.parametrize("yaw_smooth_s", [0.0, 0.5])
def test_heading_forward_flip(yaw_smooth_s):
    ang = np.linspace(0, 2 * np.pi, F)
    root = np.stack([0 * ang, 0.95 + 0.5 * np.sin(ang / 2), np.linspace(0, 2, F)], -1)
    yaw = _extract(_rv(ang, 0, 0), root, yaw_smooth_s=yaw_smooth_s)
    assert np.abs(yaw).max() < 0.5


@pytest.mark.parametrize("pitch_deg", [90.0, -90.0])  # bent over, lying on the back
def test_heading_pitched_with_noise(pitch_deg):
    rng = np.random.default_rng(0)
    pitch = np.radians(pitch_deg + rng.normal(0, 3, F))
    side = np.radians(rng.normal(0, 2, F))
    true_yaw = np.radians(35.0)
    q = qmul(_rv(0, true_yaw, 0), _rv(pitch, 0, side))
    yaw = _extract(q, np.tile([0, 0.9, 0.0], (F, 1)))
    assert np.abs(yaw - 35.0).max() < 3.0
    assert np.abs(np.diff(yaw)).max() < 1.0


def test_heading_lying_on_side_uses_forward_axis():
    roll = np.radians(90.0)  # about the forward axis: the hip line is vertical
    q = qmul(_rv(0, np.radians(-50.0), 0), _rv(0, 0, np.full(F, roll)))
    yaw = _extract(q, np.tile([0, 0.3, 0.0], (F, 1)))
    assert np.abs(yaw + 50.0).max() < 0.5


def test_heading_punch_twist_barely_moves_root():
    """Three hip twists of 25 degrees (0.35 s each) while facing 20 degrees."""
    t = np.arange(90) / 30.0
    twist = np.zeros_like(t)
    for start in (0.5, 1.4, 2.2):
        inside = (t >= start) & (t < start + 0.35)
        twist[inside] = 25.0 * np.sin(np.pi * (t[inside] - start) / 0.35)
    q = _rv(0, np.radians(20.0 + twist), 0)
    yaw = _extract(q, np.tile([0, 0.95, 0.0], (90, 1)))
    assert np.abs(yaw - 20.0).max() < 8.0  # the raw twist is 25 degrees


def test_heading_follows_a_real_turn():
    t = np.linspace(0, 1, 90)
    true = 180.0 * t
    yaw = _extract(_rv(0, np.radians(true), 0), np.tile([0, 0.95, 0.0], (90, 1)))
    assert np.abs(yaw - true)[20:-20].max() < 1.0  # a steady turn passes the filter unchanged
    assert yaw[-1] - yaw[0] > 150.0
