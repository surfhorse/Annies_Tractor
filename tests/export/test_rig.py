from __future__ import annotations

import numpy as np
import pytest
from export_helpers import build_skeleton, fighter_offsets, provisional_joints

from afv.core.skeletons import SOMA_PROVISIONAL, get_skeleton
from afv.export.mapping import Mapping, get_mapping
from afv.export.rig import HUMANOID, ROOT_BONE, build_rig
from afv.motion.fk import rest_positions
from afv.motion.rotations import from_rotvec


def test_rig_names_hierarchy_and_prefix():
    rig = build_rig(SOMA_PROVISIONAL, prefix="mixamorig:")
    assert rig.names[0] == "mixamorig:Hips"
    # provisional maps every canonical bone except the fingertip end bones (Mixamo's *4)
    assert len(rig.names) == len(HUMANOID) - 10
    assert "LeftHandIndex4" not in rig.canonical
    assert all(n.startswith("mixamorig:") for n in rig.names)
    assert rig.names[rig.parents[rig.index("LeftForeArm")]] == "mixamorig:LeftArm"
    assert rig.parents[rig.index("LeftUpLeg")] == rig.hips
    for b, p in enumerate(rig.parents):
        assert p < b


def test_root_bone_optional_and_unprefixed():
    rig = build_rig(SOMA_PROVISIONAL, prefix="mixamorig:", root_bone=True)
    assert rig.names[0] == ROOT_BONE and rig.has_root
    assert rig.parents[rig.hips] == 0
    assert np.allclose(rig.offsets[0], 0.0)  # root on the floor at the origin
    assert np.allclose(rig.offsets[rig.hips], SOMA_PROVISIONAL.rest_offsets[0])


def test_rig_uses_fighter_bone_lengths():
    off = fighter_offsets(SOMA_PROVISIONAL, seed=3)
    rig = build_rig(SOMA_PROVISIONAL, rest_offsets=off)
    src = SOMA_PROVISIONAL.with_offsets(off)
    for b, p in enumerate(rig.parents):
        if p < 0:
            continue
        j = src.index(rig.source_joint[b])
        assert np.isclose(np.linalg.norm(rig.offsets[b]), np.linalg.norm(off[j]))


def test_tpose_rest_from_a_pose_source():
    """A source with A-pose arms and bent knees (rest rotations) still gives a T-pose rig."""
    down = np.deg2rad(40.0)
    sk = build_skeleton(
        "apose_test",
        provisional_joints(),
        rest_rotations={
            "l_upperarm": from_rotvec([0.0, 0.0, -down]),
            "r_upperarm": from_rotvec([0.0, 0.0, down]),
            "l_calf": from_rotvec([0.3, 0.0, 0.0]),
            "spine2": from_rotvec([0.05, 0.02, 0.0]),
        },
    )
    rp = rest_positions(sk)
    upper, fore = sk.index("l_upperarm"), sk.index("l_forearm")
    assert rp[fore, 1] < rp[upper, 1] - 0.1  # really an A-pose
    m = Mapping("apose_test", dict(get_mapping(SOMA_PROVISIONAL.name).bones))
    rig = build_rig(sk, m)
    rest = rig.rest_positions()

    def direction(a: str, b: str) -> np.ndarray:
        v = rest[rig.index(b)] - rest[rig.index(a)]
        return v / np.linalg.norm(v)

    assert np.allclose(direction("LeftArm", "LeftForeArm"), [1, 0, 0], atol=1e-9)
    assert np.allclose(direction("LeftForeArm", "LeftHand"), [1, 0, 0], atol=1e-9)
    assert np.allclose(direction("RightArm", "RightForeArm"), [-1, 0, 0], atol=1e-9)
    assert np.allclose(direction("LeftUpLeg", "LeftLeg"), [0, -1, 0], atol=1e-9)
    assert np.allclose(direction("LeftLeg", "LeftFoot"), [0, -1, 0], atol=1e-9)
    # bone lengths unchanged by the T-pose swing
    assert np.isclose(np.linalg.norm(rest[rig.index("LeftForeArm")] - rest[rig.index("LeftArm")]),
                      np.linalg.norm(sk.rest_offsets[fore]))


def test_fingers_optional():
    rig = build_rig(SOMA_PROVISIONAL, fingers=False)
    assert not any("Thumb" in n for n in rig.names)
    assert "LeftHand" in rig.canonical


def test_missing_optional_bone_reparents_children():
    bones = dict(get_mapping(SOMA_PROVISIONAL.name).bones)
    del bones["Spine1"]  # spine2 becomes a folded joint between Spine and Spine2
    rig = build_rig(SOMA_PROVISIONAL, Mapping(SOMA_PROVISIONAL.name, bones))
    assert "Spine1" not in rig.canonical
    assert rig.parents[rig.index("Spine2")] == rig.index("Spine")
    assert rig.compressed[rig.index("Spine2")]


def test_mapping_validation():
    bones = dict(get_mapping(SOMA_PROVISIONAL.name).bones)
    with pytest.raises(ValueError, match="required"):
        build_rig(SOMA_PROVISIONAL, Mapping(SOMA_PROVISIONAL.name, {k: v for k, v in bones.items()
                                                                   if k != "LeftHand"}))
    bad = dict(bones, LeftForeArm="r_forearm")
    with pytest.raises(ValueError):
        build_rig(SOMA_PROVISIONAL, Mapping(SOMA_PROVISIONAL.name, bad))
    with pytest.raises(KeyError):
        build_rig(SOMA_PROVISIONAL, Mapping(SOMA_PROVISIONAL.name, dict(bones, Neck="no_such_joint")))


def test_soma77_rig():
    """The shipped soma77 table: every Mixamo bone mapped, metacarpals and Neck2 folded."""
    sk = get_skeleton("soma77")
    rig = build_rig(sk, root_bone=True)
    assert len(rig.names) == len(HUMANOID) + 1
    folded = {n for n, c in zip(rig.canonical, rig.compressed, strict=True) if c}
    fingers = {f"{s}Hand{f}1" for s in ("Left", "Right") for f in ("Index", "Middle", "Ring", "Pinky")}
    assert folded == {"Head"} | fingers
    assert rig.source_joint[rig.index("LeftHandIndex1")] == "LeftHandIndex2"
    assert rig.source_joint[rig.index("LeftHandIndex4")] == "LeftHandIndexEnd"
    rest = rig.rest_positions()
    v = rest[rig.index("LeftForeArm")] - rest[rig.index("LeftArm")]
    assert np.allclose(v / np.linalg.norm(v), [1, 0, 0], atol=1e-6)  # T-pose arms
