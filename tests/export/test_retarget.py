from __future__ import annotations

import numpy as np
from export_helpers import build_skeleton, fighter_offsets, provisional_joints, synthetic_motion

from afv.core.skeletons import SOMA_PROVISIONAL, get_skeleton
from afv.export.mapping import Mapping, get_mapping
from afv.export.retarget import mapped_source_positions, retarget, rig_fk, source_fk
from afv.export.rig import build_rig
from afv.motion.rotations import from_rotvec, qconj, qmul, to_rotvec


def _check_positions(rig, rot, root, tol):
    m = retarget(rig, rot, root, 30.0)
    pos, _ = rig_fk(m)
    want = mapped_source_positions(rig, rot, root)
    ok = ~np.isnan(want[0, :, 0])
    err = np.linalg.norm(pos[:, ok] - want[:, ok], axis=-1)
    assert err.max() <= tol, f"max position error {err.max():.2e} m"
    return m, err


def test_positions_preserved_provisional_fighter():
    off = fighter_offsets(SOMA_PROVISIONAL, seed=1)
    rig = build_rig(SOMA_PROVISIONAL, rest_offsets=off)
    rot, root = synthetic_motion(SOMA_PROVISIONAL, 40, seed=2)
    _check_positions(rig, rot, root, 1e-9)


def test_positions_preserved_with_root_bone():
    rig = build_rig(SOMA_PROVISIONAL, root_bone=True)
    rot, root = synthetic_motion(SOMA_PROVISIONAL, 40, seed=5)
    m, _ = _check_positions(rig, rot, root, 1e-9)
    assert np.allclose(m.loc[:, 0], 0.0)  # world mode: root stays at the origin


def test_rest_pose_maps_to_identity_and_orientations_transfer():
    rig = build_rig(SOMA_PROVISIONAL)
    rot = np.tile([0.0, 0.0, 0.0, 1.0], (3, SOMA_PROVISIONAL.num_joints, 1))
    root = np.tile(SOMA_PROVISIONAL.rest_offsets[0], (3, 1))
    m = retarget(rig, rot, root, 30.0)
    assert np.allclose(np.abs(m.rot[..., 3]), 1.0)  # provisional rest is already the T-pose
    rot2, root2 = synthetic_motion(SOMA_PROVISIONAL, 10, seed=8)
    m2 = retarget(rig, rot2, root2, 30.0)
    _, wq = rig_fk(m2)
    _, sq = source_fk(rig, rot2, root2)
    for b, s in enumerate(rig.source_joint):
        j = rig.source.index(s)
        d = qmul(wq[:, b], qconj(qmul(sq[:, j], qconj(rig.source_tpose_rot[b]))))
        assert np.allclose(np.abs(d[:, 3]), 1.0, atol=1e-9)


def test_a_pose_source_with_rest_rotations():
    sk = build_skeleton(
        "apose_rt",
        provisional_joints(),
        rest_rotations={
            "l_upperarm": from_rotvec([0.0, 0.0, -0.7]),
            "r_upperarm": from_rotvec([0.0, 0.0, 0.7]),
            "l_thigh": from_rotvec([0.0, 0.0, 0.1]),
            "spine1": from_rotvec([0.1, 0.0, 0.05]),
        },
    )
    rig = build_rig(sk, Mapping("apose_rt", dict(get_mapping(SOMA_PROVISIONAL.name).bones)))
    rot, root = synthetic_motion(sk, 30, seed=4)
    _check_positions(rig, rot, root, 1e-9)


def _extra_spine_skeleton():
    """Provisional skeleton with an extra spine joint and a two-joint neck (like a denser source)."""
    joints = []
    for name, parent, off in provisional_joints():
        if name == "spine3":
            joints.append(("spine2b", "spine2", (0.0, 0.06, 0.0)))
            joints.append(("spine3", "spine2b", (0.0, 0.06, 0.0)))
        elif name == "head":
            joints.append(("neck2", "neck", (0.0, 0.05, 0.01)))
            joints.append(("head", "neck2", (0.0, 0.05, 0.0)))
        else:
            joints.append((name, parent, off))
    return build_skeleton("extra_spine", joints)


def test_folded_spine_and_neck_joints():
    sk = _extra_spine_skeleton()
    rig = build_rig(sk, Mapping("extra_spine", dict(get_mapping(SOMA_PROVISIONAL.name).bones)))
    assert rig.compressed[rig.index("Spine2")] and rig.compressed[rig.index("Head")]
    rot, root = synthetic_motion(sk, 40, seed=6, amp=0.3)
    m, err = _check_positions(rig, rot, root, 0.005)
    # direct links stay exact; only joints downstream of a folded link see the chord-length change
    exact = [rig.index(n) for n in ("Hips", "Spine", "Spine1", "LeftUpLeg", "LeftFoot")]
    assert err[:, exact].max() < 1e-9
    # bones that are not re-aimed keep the source orientation exactly (folded rotation included);
    # Spine1 and Neck are the re-aimed parents of the folded links
    _, wq = rig_fk(m)
    _, sq = source_fk(rig, rot, root)
    for name in ("Spine2", "Head", "LeftHand", "RightFoot"):
        b = rig.index(name)
        j = rig.source.index(rig.source_joint[b])
        want = qmul(sq[:, j], qconj(rig.source_tpose_rot[b]))
        d = qmul(wq[:, b], qconj(want))
        assert np.allclose(np.abs(d[:, 3]), 1.0, atol=1e-9)


# --- soma77 (shipped skeleton): folded metacarpals and Neck2 (review r6) ---------------------------

METACARPALS = [f"{s}Hand{f}1" for s in ("Left", "Right") for f in ("Index", "Middle", "Ring", "Pinky")]


def _soma77_case(seed: int, amp: float, fold_amp: float, wrist: float):
    """Random motion on soma77 with metacarpal/Neck2 rotations scaled by ``fold_amp`` and a sharp
    wrist sweep of ``wrist`` radians on both hands."""
    sk = get_skeleton("soma77")
    off = fighter_offsets(sk, seed)
    rot, root = synthetic_motion(sk, 50, seed, amp=amp)
    for name in [*METACARPALS, "Neck2"]:
        j = sk.index(name)
        rv = to_rotvec(rot[:, j]) * (fold_amp / amp)
        rot[:, j] = from_rotvec(rv)
    s = np.linspace(-1.0, 1.0, 50)[:, None]
    for hand, ax in (("LeftHand", [0.3, 1.0, 0.2]), ("RightHand", [1.0, 0.2, -0.4])):
        a = np.asarray(ax) / np.linalg.norm(ax)
        rot[:, sk.index(hand)] = qmul(rot[:, sk.index(hand)], from_rotvec(wrist * s * a))
    rig = build_rig(sk, rest_offsets=off)
    return sk, rig, rot, root


def _orientation_err_deg(rig, m, rot, root, names):
    _, wq = rig_fk(m)
    _, sq = source_fk(rig, rot, root)
    out = {}
    for name in names:
        b = rig.index(name)
        want = qmul(sq[:, rig.source.index(rig.source_joint[b])], qconj(rig.source_tpose_rot[b]))
        d = qmul(wq[:, b], qconj(want))
        out[name] = float(np.degrees(2 * np.arccos(np.clip(np.abs(d[:, 3]), 0, 1))).max())
    return out


def test_soma77_exact_when_folded_joints_rest():
    _, rig, rot, root = _soma77_case(seed=3, amp=0.45, fold_amp=0.0, wrist=1.6)
    m, _ = _check_positions(rig, rot, root, 1e-9)
    errs = _orientation_err_deg(rig, m, rot, root, rig.canonical[1:] if rig.has_root else rig.canonical)
    assert max(errs.values()) < 1e-4  # degrees (round-off of arccos near 1)


def test_soma77_folded_metacarpals_keep_wrist_exact():
    """Rotating metacarpals and Neck2: the wrist (multi-child parent) stays exact; knuckles move by
    at most the chord of the folded rotation; Head (single-child fold) by the chord-length change."""
    sk, rig, rot, root = _soma77_case(seed=4, amp=0.45, fold_amp=0.3, wrist=1.6)
    m = retarget(rig, rot, root, 30.0)
    exact = ["LeftHand", "RightHand", "Spine2", "Head", "LeftHandIndex2", "RightHandPinky3", "LeftHandThumb2"]
    errs = _orientation_err_deg(rig, m, rot, root, exact)
    assert max(errs.values()) < 1e-4, errs
    pos, _ = rig_fk(m)
    want = mapped_source_positions(rig, rot, root)
    err = np.linalg.norm(pos - want, axis=-1)
    for hand in ("LeftHand", "RightHand"):
        assert err[:, rig.index(hand)].max() < 1e-9
    for name in METACARPALS:  # rig Index1 = SOMA Index2, the knuckle after the folded metacarpal
        b = rig.index(name)
        j = sk.index(name)
        theta = np.linalg.norm(to_rotvec(rot[:, j]), axis=-1)
        bound = 2 * np.linalg.norm(rig.source.rest_offsets[sk.index(rig.source_joint[b])]) * np.sin(theta / 2)
        assert np.all(err[:, b] <= bound + 1e-9), name
        tip = rig.index(name[:-1] + "4")  # finger bones keep orientation: the whole finger shifts rigidly
        assert np.allclose(err[:, tip], err[:, b], atol=1e-9)
    assert err[:, rig.index("Head")].max() < 0.01
