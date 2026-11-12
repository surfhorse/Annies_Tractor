"""QC: projection, reprojection error, per-frame metrics and the labelled metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest
from motion_synth import SK, body, look_at_camera, punch, walk

from afv.core.stage import StageError
from afv.core.types import ActorMotion
from afv.motion.contacts import Floor, segments
from afv.motion.fk import forward_kinematics
from afv.qc.camera import camera_index, project, to_camera
from afv.qc.labels import (
    Labels,
    PlantLabel,
    StrikeLabel,
    contact_recall,
    evaluate,
    ground_penetration,
    slide_within_plants,
    strike_sharpness,
)
from afv.qc.metrics import contact_speed, per_frame_metrics, reprojection_error, summarise


def _walk_with_contacts(**kw):
    a, masks = walk(seconds=2.0, **kw)
    chans = ["l_heel", "l_toe", "r_heel", "r_toe"]
    a.contact = np.stack([masks[c] for c in chans], axis=1).astype(np.float32)
    return body({"A": a}, chans), masks


def test_projection_conventions():
    cam = look_at_camera(np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, 0.0]), 1000.0, 3)
    pts = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 20.0]]] * 3)
    pc = to_camera(pts, cam, np.arange(3))
    assert np.allclose(pc[0, 0], [0, 0, -10])  # looks down -Z
    uv = project(pts, cam, np.arange(3))
    assert np.allclose(uv[0, 0], [cam.cx, cam.cy])
    assert np.allclose(uv[0, 1], [cam.cx + 100, cam.cy])  # +X is image right
    assert np.allclose(uv[0, 2], [cam.cx, cam.cy - 100])  # +Y up is image up (smaller v)
    assert np.isnan(uv[0, 3]).all()  # behind the camera
    with pytest.raises(StageError):
        camera_index(cam, np.array([5]))


def test_reprojection_error_known_offset():
    a, _ = walk(seconds=1.0)
    pos = forward_kinematics(SK, a.rot, a.root)[0]
    cam = look_at_camera(np.array([0.0, 1.0, 6.0]), np.array([0.0, 1.0, 0.0]), 1500.0, len(pos))
    names = ["l_foot", "r_foot", "head", "nonexistent"]
    kp = project(pos[:, [SK.index(n) for n in names[:3]]], cam, np.arange(len(pos)))
    kp = np.concatenate([kp + np.array([3.0, 4.0]), np.zeros((len(pos), 1, 2))], axis=1)
    conf = np.ones(kp.shape[:2])
    conf[:, 2] = 0.0  # head invisible
    err = reprojection_error(pos, SK.joint_names, cam, np.arange(len(pos)), kp, conf, names)
    assert np.allclose(err, 5.0)


def test_contact_speed_and_summary():
    p = np.zeros((10, 3))
    p[:, 0] = np.arange(10) * 0.01  # 30 cm/s
    m = np.zeros(10, bool)
    m[2:6] = True
    assert np.allclose(contact_speed(p, m, 30.0), 0.3) and len(contact_speed(p, m, 30.0)) == 3
    bm, _ = _walk_with_contacts()
    met = per_frame_metrics(bm, floor=Floor.horizontal(0.0))
    s = summarise(met)["A"]
    assert s["mean_contact_foot_speed_cm_s"] < 1e-6
    assert s["max_penetration_cm"] < 1e-6
    assert set(met["A"]) >= {"skate", "penetration", "accel", "pops", "speed_r_hand", "contact_l_heel"}
    json.dumps(s)


def test_labelled_metrics_known_values():
    bm, masks = _walk_with_contacts()
    floor = Floor.horizontal(0.0)
    plants = [PlantLabel("A", s, e, foot="left") for s, e in segments(masks["l_heel"])]
    labels = Labels(plants)
    assert contact_recall(bm, labels) == 1.0
    assert slide_within_plants(bm, labels, floor).max() < 1e-9
    assert ground_penetration(bm, labels, floor) < 1e-9
    # Slide a plant by 3 cm and sink everything 1 cm: the metrics see it.
    a = bm.actors["A"]
    s0, e0 = plants[1].start, plants[1].end
    root = a.root.copy()
    root[s0:e0, 0] += np.linspace(0, 0.03, e0 - s0)
    root[:, 1] -= 0.01
    moved = body({"A": ActorMotion(a.rot, root, contact=a.contact)}, bm.contact_channels)
    assert slide_within_plants(moved, labels, floor)[1] == pytest.approx(0.03, abs=1e-6)
    assert ground_penetration(moved, labels, floor) == pytest.approx(0.01, abs=1e-6)
    # Missed plant: recall drops.
    c = a.contact.copy()
    c[s0:e0, :2] = 0
    missed = body({"A": ActorMotion(a.rot, a.root, contact=c)}, bm.contact_channels)
    assert contact_recall(missed, labels) == pytest.approx((len(plants) - 1) / len(plants))
    res = evaluate(moved, labels, floor)
    assert res["slide_cm"]["pass"] is False and res["contact_recall"]["pass"] is True


def test_strike_sharpness_and_labels_file(tmp_path):
    pu = punch(ext_frames=3)
    before = body({"A": pu})
    lab = Labels(strikes=[StrikeLabel("A", "r_hand", 31)])
    assert strike_sharpness(before, before, lab)[0] == pytest.approx(1.0)
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "plants": [{"role": "A", "foot": "left", "start": 0, "end": 10}],
                "strikes": [{"role": "A", "effector": "r_hand", "frame": 31}],
            }
        )
    )
    loaded = Labels.load(path)
    assert loaded.plants[0].foot == "left" and loaded.strikes[0].frame == 31


def test_hover_is_the_feet_lowest_sole_point():
    """Round 3: planted-foot hover is measured like the contacts stage measures it: the lowest sole
    point of the foot (heel or ball, ``afv.motion.contacts.foot_sole_heights``) on frames where any of
    its channels is planted. A heel lifted off a planted ball is not a float; it shows only in the
    per-joint ``max_joint_hover_in_contact_cm``."""
    bm, _ = _walk_with_contacts()
    a = bm.actors["A"]
    met = per_frame_metrics(bm, floor=Floor.horizontal(0.0))["A"]
    for foot, chans in (("l_foot", ("l_heel", "l_toe")), ("r_foot", ("r_heel", "r_toe"))):
        low = np.minimum(*(met[f"sole_height_{c}"] for c in chans))
        assert np.allclose(met[f"foot_sole_height_{foot}"], low)
        planted = np.maximum(*(met[f"contact_{c}"] for c in chans))
        assert np.array_equal(met[f"foot_contact_{foot}"], planted)
    # Raise the whole body 3 cm: every planted foot floats.
    root = np.asarray(a.root, np.float64) + np.array([0.0, 0.03, 0.0])
    lifted = body({"A": ActorMotion(a.rot, root, contact=a.contact)}, bm.contact_channels)
    s = summarise(per_frame_metrics(lifted, floor=Floor.horizontal(0.0)))["A"]
    assert s["max_hover_in_contact_cm"] == pytest.approx(3.0, abs=0.05)  # whole foot up: a float
    assert s["max_joint_hover_in_contact_cm"] >= 3.0 - 0.05
    json.dumps(s)


def test_foot_pops_against_the_stage_input():
    """Review: a foot that lurches against the input's (3-step displacement > 6 cm longer and 1.6 x
    the input's) is counted per frame against the stage's input (``ref``); filling a stall of the
    input is not."""
    bm, _ = _walk_with_contacts()
    a = bm.actors["A"]
    same = summarise(per_frame_metrics(bm, floor=Floor.horizontal(0.0), ref=bm))["A"]
    assert same["foot_pop_frames"] == 0
    root = np.asarray(a.root, np.float64).copy()
    root[30:, 0] += 0.08  # every joint jumps 8 cm between frames 29 and 30
    popped = body({"A": ActorMotion(a.rot, root, contact=a.contact)}, bm.contact_channels)
    met = per_frame_metrics(popped, floor=Floor.horizontal(0.0), ref=bm)["A"]
    hit = np.flatnonzero(met["foot_pops"]).tolist()
    assert hit and set(hit) <= {29, 30, 31} and met["foot_pops"].max() >= 2  # planted feet
    assert summarise({"A": met})["A"]["foot_pop_frames"] == len(hit)
    stall = np.asarray(a.root, np.float64).copy()
    stall[:, 0] += 0.05 * np.arange(len(stall))  # walking forward 5 cm a frame ...
    stall_in = stall.copy()
    stall_in[30:, 0] -= 0.04  # ... except that the input stalls for one frame
    fill = body({"A": ActorMotion(a.rot, stall, contact=a.contact)}, bm.contact_channels)
    ref = body({"A": ActorMotion(a.rot, stall_in, contact=a.contact)}, bm.contact_channels)
    assert per_frame_metrics(fill, floor=Floor.horizontal(0.0), ref=ref)["A"]["foot_pops"].max() == 0
    assert "foot_pops" not in per_frame_metrics(popped, floor=Floor.horizontal(0.0))["A"]
