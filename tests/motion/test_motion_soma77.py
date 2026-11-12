"""The real SOMA skeleton (module C's soma77.json): role config and the contacts/smooth chain."""

from __future__ import annotations

import numpy as np
import pytest

from afv.core.skeletons import SKELETONS, get_skeleton
from afv.core.types import ActorMotion, BodyMotion
from afv.motion.contacts import clean_contacts
from afv.motion.fk import forward_kinematics, rest_positions
from afv.motion.ik import SOMA77_ROLES, LegTarget, get_roles, solve_leg_ik
from afv.motion.rotations import IDENTITY
from afv.motion.smooth import smooth_motion
from afv.qc.metrics import per_frame_metrics, summarise

pytestmark = pytest.mark.skipif("soma77" not in SKELETONS, reason="soma77 skeleton not available")

# Module C names BodyMotion contact channels by joint (GEM-X order: ankle, foot per side, wrists).
SOMA_CHANNELS = ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase", "LeftHand", "RightHand"]


def test_soma77_roles():
    sk = get_skeleton("soma77")
    roles = get_roles(sk)
    assert roles is SOMA77_ROLES
    assert [c.channel for c in roles.contacts] == SOMA_CHANNELS[:4]
    assert roles.legs[0].thigh == "LeftLeg" and roles.legs[0].calf == "LeftShin"
    clear = roles.clearances(sk)
    assert clear["LeftFoot"] == pytest.approx(0.070, abs=0.005)
    assert clear["LeftToeBase"] == pytest.approx(0.022, abs=0.005)
    reg = roles.regions(sk)
    assert reg[sk.index("LeftShin")] == "legs" and reg[sk.index("LeftHandIndex2")] == "hands"
    assert reg[sk.index("RightForeArm")] == "arms" and reg[sk.index("Head")] == "head"


def test_soma77_skating_stance_fixed():
    sk = get_skeleton("soma77")
    roles = get_roles(sk)
    rng = np.random.default_rng(0)
    n = 90
    t = np.arange(n) / 30.0
    rest = rest_positions(sk)
    root = np.tile(rest[0] - [0.0, 0.06, 0.0], (n, 1))  # knees a little bent
    rot = np.tile(IDENTITY, (n, sk.num_joints, 1))
    targets = [
        LegTarget(leg, np.tile(rest[sk.index(leg.foot)], (n, 1)), np.ones(n, bool)) for leg in roles.legs
    ]
    rot = solve_leg_ik(sk, rot, root, targets, forward=np.array(roles.forward))
    # GEM-X-like damage: root drifts 4 cm while the feet should stay put, plus a 10 cm floor offset.
    bad_root = (
        root
        + np.column_stack(
            [0.04 * np.sin(2 * np.pi * t / 1.5), np.full(n, 0.1), 0.04 * np.cos(2 * np.pi * t / 1.1)]
        )
        + rng.normal(0, 0.002, (n, 3))
    )
    contact = np.tile([0.9, 0.9, 0.9, 0.9, 0.05, 0.05], (n, 1)).astype(np.float32)
    bm = BodyMotion(
        frames=np.arange(n),
        timestamps=t,
        skeleton="soma77",
        fps=30.0,
        contact_channels=SOMA_CHANNELS,
        actors={"A": ActorMotion(rot, bad_root, contact=contact)},
    )
    out, rep = clean_contacts(bm)
    assert rep.floor_shift == pytest.approx(-0.1, abs=0.01)
    assert out.contact_channels == SOMA_CHANNELS  # foot masks under the joint names, wrists passed through
    assert out.world_shift == pytest.approx([0.0, rep.floor_shift, 0.0])
    sm, _ = smooth_motion(out)
    for res in (out, sm):
        s = summarise(per_frame_metrics(res, floor=rep.floor))["A"]
        assert s["contact_frames"] > 0.9 * 4 * n
        assert s["mean_contact_foot_speed_cm_s"] <= 3.0
        assert s["max_penetration_cm"] <= 0.5
    pos = forward_kinematics(sk, sm.actors["A"].rot, sm.actors["A"].root)[0]
    assert np.abs(pos[:, sk.index("LeftToeBase"), 1] - roles.clearances(sk)["LeftToeBase"]).max() < 0.005


def test_soma77_sides_resolve_through_roles():
    """Painted contacts and labels by foot side work on soma77 (no "l_"/"r_" name prefixes) and a
    label that slides is not a vacuous pass."""
    from afv.motion.contacts import Floor, painted_spans
    from afv.qc.labels import Labels, PlantLabel, slide_within_plants

    sk = get_skeleton("soma77")
    roles = get_roles(sk)
    spans = painted_spans(
        {"A": [{"foot": "left", "start": 0, "end": 10, "state": "free"}]}, "A", roles, np.arange(30)
    )
    assert set(spans) == {"LeftFoot", "LeftToeBase"}
    assert PlantLabel("A", 0, 10, foot="right").channel_names(roles) == ["RightFoot", "RightToeBase"]
    with pytest.raises(ValueError):
        PlantLabel("A", 0, 10, foot="sideways").channel_names(roles)
    n = 30
    rest = rest_positions(sk)
    root = np.tile(rest[0], (n, 1))
    root[15:, 0] += 0.5  # the whole body (planted foot included) jumps 50 cm mid-plant
    bm = BodyMotion(
        frames=np.arange(n),
        timestamps=np.arange(n) / 30.0,
        skeleton="soma77",
        fps=30.0,
        contact_channels=[],
        actors={"A": ActorMotion(np.tile(IDENTITY, (n, sk.num_joints, 1)), root)},
    )
    slide = slide_within_plants(
        bm, Labels([PlantLabel("A", 0, n, foot="left")]), Floor.horizontal(0.0), roles
    )
    assert slide.max() == pytest.approx(0.5, abs=1e-6)
