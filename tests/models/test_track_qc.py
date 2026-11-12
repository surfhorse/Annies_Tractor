"""QC maths, identity check and the Tracks timeline on synthetic tracks (no GPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from afv.core.types import Tracks, load_result, save_result
from afv.models.track import (
    HIST_DIM,
    FightersEdit,
    RawTrack,
    TrackParams,
    TrackPrompt,
    add_prompt,
    appearance_scores,
    bhattacharyya,
    box_iou,
    build_tracks,
    compute_qc,
    keep_runs,
    load_fighters_edit,
    longest_run,
    review_overlay,
    runs,
    shot_timeline,
)

F = 90
EDIT3 = {
    "objects": [{"id": 1, "role": "A"}, {"id": 2, "role": "B"}, {"id": 3, "label": "referee"}],
    "prompts": [
        {"frame": 0, "obj": 1, "detection": 0},
        {"frame": 0, "obj": 2, "detection": 1},
        {"frame": 0, "obj": 3, "detection": 2},
    ],
}


def edit3() -> FightersEdit:
    return FightersEdit.model_validate(EDIT3)


def hist_for(k: int) -> np.ndarray:
    h = np.full(HIST_DIM, 0.002)
    h[k * 20 : k * 20 + 10] = 1.0  # distinct colour bins per person
    return h / h.sum() * 2


def make_raw(n: int = F, dets_from_boxes: bool = True) -> RawTrack:
    """Three people walking right; detections equal to the mask boxes."""
    t = np.arange(n)[:, None]
    x0 = np.array([100, 500, 900])[None] + 2.0 * t
    boxes = np.stack([x0, np.full_like(x0, 100), x0 + 120, np.full_like(x0, 400)], -1).astype(np.float32)
    boxes[:, 2, 1] = 150  # referee shorter than the fighters but > 0.5 x their height
    area = ((boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1]) * 0.6).astype(np.float32)
    cent = np.stack([(boxes[..., 0] + boxes[..., 2]) / 2, (boxes[..., 1] + boxes[..., 3]) / 2], -1)
    hist = np.stack([np.stack([hist_for(k) for k in range(3)])] * n).astype(np.float32)
    raw = RawTrack(
        np.array([1, 2, 3]), boxes, area, cent.astype(np.float32), hist,
        np.zeros(0, np.int64), np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 1600, 900,
    )  # fmt: skip
    if dets_from_boxes:
        set_dets(raw, [[(b, 0.9) for b in boxes[f]] for f in range(n)])
    return raw


def set_dets(raw: RawTrack, per_frame: list[list[tuple[np.ndarray, float]]]) -> None:
    fr, bx, sc = [], [], []
    for f, dets in enumerate(per_frame):
        dets = sorted(dets, key=lambda d: -(d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))
        for b, s in dets:
            fr.append(f)
            bx.append(b)
            sc.append(s)
    raw.det_frame = np.array(fr, np.int64)
    raw.det_boxes = np.array(bx, np.float32).reshape(-1, 4)
    raw.det_scores = np.array(sc, np.float32)


def box_mask_iou(raw: RawTrack) -> np.ndarray:
    """Stand-in for the worker's mask IoU: IoU of the (rectangular) masks."""
    out = np.stack([box_iou(raw.boxes[f], raw.boxes[f]) for f in range(raw.num_frames)]).astype(np.float32)
    out[:, np.arange(3), np.arange(3)] = 0
    return out


def obj(qc, oid):
    return next(o for o in qc.summary["objects"] if o["id"] == oid)


def test_helpers():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [np.nan] * 4])
    assert np.allclose(box_iou(a, b), [[1, 1 / 3, 0, 0]])
    m = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1], bool)
    assert runs(m) == [(1, 2), (4, 6), (8, 8)]
    assert longest_run(m) == 3 and longest_run(np.zeros(4, bool)) == 0
    assert keep_runs(m, 3).nonzero()[0].tolist() == [4, 5, 6]
    p = np.array([1.0, 0, 0, 0])
    assert bhattacharyya(p, p) == pytest.approx(0)
    assert bhattacharyya(p, np.array([0, 1.0, 0, 0])) == pytest.approx(1)


def test_clean_tracks_have_no_flags():
    raw = make_raw()
    raw.mask_iou = box_mask_iou(raw)
    qc = compute_qc(raw, edit3(), TrackParams())
    assert qc.summary["ok"], qc.summary
    assert qc.summary["flagged_frame_count"] == 0 and qc.details["flagged_frames"] == []
    assert qc.summary["suggestions"] == [] and qc.summary["suggestion_count"] == 0
    for o in qc.summary["objects"]:
        assert o["unmatched_frames"] == 0 and o["empty_frames"] == 0 and o["identity_ok"]
        assert o["status"] == "ok" and o["appearance_check"] == "on"
    assert np.allclose(qc.match_score, 0.9)


def test_drift_run_and_suggestion():
    raw = make_raw()
    raw.boxes[30:45, 0] += [300, 0, 300, 0]  # A's mask slides onto the background, detection stays
    raw.centroid[30:45, 0, 0] += 300
    qc = compute_qc(raw, edit3(), TrackParams(jump_thresh=10))
    a = obj(qc, 1)
    assert a["unmatched_frames"] == 15 and a["longest_unmatched_run"] == 15
    assert a["reason_frames"]["drift"] == 15 and a["flagged_runs"][0][:2] == [30, 44]
    assert qc.details["objects"][0]["unmatched_list"] == list(range(30, 45))
    sug = [s for s in qc.summary["suggestions"] if s["kind"] == "drift"]
    assert len(sug) == 1 and sug[0]["obj"] == 1 and sug[0]["frame"] == 30
    db, _ = raw.dets(30)
    assert np.allclose(db[sug[0]["detection"]], make_raw().boxes[30, 0])
    assert f'"box": {sug[0]["box"]}' in sug[0]["hint"]  # item 11: the hint carries the box
    # a short unmatched run is reported but not flagged
    raw = make_raw()
    raw.boxes[30:34, 0] += [300, 0, 300, 0]
    qc = compute_qc(raw, edit3(), TrackParams(jump_thresh=10))
    assert obj(qc, 1)["unmatched_frames"] == 4 and obj(qc, 1)["reason_frames"]["drift"] == 0


def test_empty_frames_flag_only_fighters():
    raw = make_raw()
    raw.boxes[50:56, 1] = np.nan
    raw.area[50:56, 1] = 0
    raw.boxes[10:20, 2] = np.nan
    raw.area[10:20, 2] = 0
    qc = compute_qc(raw, edit3(), TrackParams())
    assert obj(qc, 2)["empty_frames"] == 6 and obj(qc, 2)["reason_frames"]["empty"] == 6
    assert obj(qc, 3)["empty_frames"] == 10 and obj(qc, 3)["reason_frames"]["empty"] == 0


def test_centroid_jump():
    raw = make_raw()
    raw.centroid[40:, 0, 0] += 300  # one body height in one frame
    qc = compute_qc(raw, edit3(), TrackParams())
    assert np.flatnonzero(qc.flags["jump"][:, 0]).tolist() == [40]


def test_swap_detected_by_position_and_appearance():
    raw = make_raw()
    for arr in (raw.boxes, raw.centroid, raw.hist, raw.area):
        arr[45:, [0, 1]] = arr[45:, [1, 0]].copy()  # A and B exchange identities at frame 45
    raw.mask_iou = box_mask_iou(raw)
    qc = compute_qc(raw, edit3(), TrackParams())
    assert qc.flags["swap"][45, 0] and qc.flags["swap"][45, 1] and not qc.flags["swap"][45, 2]
    assert qc.flags["swap"][:, :2].sum() == 2
    app = qc.flags["appearance"]
    assert app[46:, 0].all() and app[46:, 1].all() and not app[:44].any() and not app[:, 2].any()
    assert not obj(qc, 1)["identity_ok"] and obj(qc, 3)["identity_ok"]
    a = obj(qc, 1)
    assert a["appearance_threshold"] == pytest.approx(0.35 * a["appearance_separation"], abs=2e-3)


def test_gradual_drift_onto_the_other_fighter():
    """Review item 4: A's mask slides onto B over 10 frames (no instant exchange, same colours)."""
    raw = make_raw()
    for i, f in enumerate(range(40, F)):
        a = min(1.0, i / 10)
        raw.boxes[f, 0] = (1 - a) * raw.boxes[f, 0] + a * raw.boxes[f, 1]
        raw.centroid[f, 0] = (1 - a) * raw.centroid[f, 0] + a * raw.centroid[f, 1]
    set_dets(raw, [[(b, 0.9) for b in make_raw().boxes[f]] for f in range(F)])  # detectors see 3 people
    for mask_iou in (box_mask_iou(raw), None):  # worker mask IoU, or the box fallback for old raw files
        raw.mask_iou = mask_iou
        qc = compute_qc(raw, edit3(), TrackParams())
        assert qc.flags["merge"][50:, 0].all() and qc.flags["merge"][50:, 1].all()
        assert not qc.flags["merge"][:44].any() and not qc.flags["merge"][:, 2].any()
        assert qc.flags["shared_detection"][50:, :2].all()
        assert not obj(qc, 1)["identity_ok"] and not obj(qc, 2)["identity_ok"] and obj(qc, 3)["identity_ok"]
        assert qc.flags["swap"].sum() == 0  # the old detector alone missed it


def test_appearance_check_off_when_people_look_alike():
    raw = make_raw()
    raw.hist[:] = hist_for(0)  # everybody in the same colours
    for arr in (raw.boxes, raw.centroid, raw.area):
        arr[45:, [0, 1]] = arr[45:, [1, 0]].copy()
    qc = compute_qc(raw, edit3(), TrackParams())
    assert obj(qc, 1)["appearance_check"] == "not discriminative" and not qc.flags["appearance"].any()


def test_correction_prompt_reference_excludes_the_drifted_frames():
    """Review item 13: frames before a correction prompt may show the wrong person."""
    raw = make_raw()
    raw.hist[40:50, 0] = hist_for(1)  # A's mask showed B until the correction at frame 50
    e = edit3()
    e = FightersEdit(
        objects=e.objects, prompts=[*e.prompts, TrackPrompt(frame=50, obj=1, box=[200, 100, 320, 400])]
    )
    _, own, _ = appearance_scores(raw, e.prompt_frames())
    assert own[60, 0] < 1e-3 and own[45, 0] > 0.5
    qc = compute_qc(raw, e, TrackParams())
    assert np.flatnonzero(qc.flags["appearance"][:, 0]).tolist() == list(range(40, 50))


def untracked_raw(extra_dets) -> RawTrack:
    raw = make_raw()
    per = []
    for f in range(F):
        dets = [(b, 0.9) for b in raw.boxes[f]] + [(np.array(b, np.float32), s) for b, s in extra_dets(f)]
        dets.append((np.array([10, 10, 30, 40], np.float32), 0.95))  # a small spectator: ignored
        per.append(dets)
    set_dets(raw, per)
    return raw


def test_untracked_person_suggestion():
    def entering(f):  # someone fighter-sized enters at the right and grows to full size
        if f < 20:
            return []
        w = min(120, 40 + 4 * (f - 20))
        return [([1450 - w, 100, 1450, 400], 0.8)]

    raw = untracked_raw(entering)
    qc = compute_qc(raw, edit3(), TrackParams())
    sug = [s for s in qc.summary["suggestions"] if s["kind"] == "untracked_person"]
    assert len(sug) == 1
    s = sug[0]
    assert s["frames"] == [20, F - 1] and s["frame"] >= 40 and s["obj"] == 4  # full-size frame chosen
    db, _ = raw.dets(s["frame"])
    assert np.allclose(db[s["detection"]], [1330, 100, 1450, 400]) and s["box"] == [1330, 100, 1450, 400]
    assert '"box": [1330.0, 100.0, 1450.0, 400.0]' in s["hint"]
    assert qc.summary["untracked_frames"] == F - 20
    ov = review_overlay(raw, edit3(), qc)
    assert ov["notes"][19] == "" and ov["notes"][20] == "UNTRACKED PERSON" and len(ov["untracked"][30]) == 1
    # a short appearance is not suggested
    raw = untracked_raw(lambda f: entering(f) if f < 25 else [])
    qc = compute_qc(raw, edit3(), TrackParams())
    assert not [s for s in qc.summary["suggestions"] if s["kind"] == "untracked_person"]


@pytest.mark.parametrize("simultaneous", [False, True])
def test_each_untracked_person_gets_its_own_id(simultaneous):
    """Review item 5: two people (one after the other, or at once) used to share one new id."""
    late = (20, 60) if simultaneous else (60, 80)

    def two(f):
        out = []
        if 20 <= f < 60:
            out.append(([1250, 100, 1370, 400], 0.9))
        if late[0] <= f < late[1]:
            out.append(([1450, 100, 1570, 400], 0.9))
        return out

    qc = compute_qc(untracked_raw(two), edit3(), TrackParams())
    sug = [s for s in qc.summary["suggestions"] if s["kind"] == "untracked_person"]
    assert [(s["obj"], s["frames"][0]) for s in sug] == [(4, 20), (5, late[0])]
    assert sorted(s["box"][0] for s in sug) == [1250, 1450]
    data = dict(EDIT3)
    for s in sug:
        data = add_prompt(data, s["frame"], s["obj"], detection=s["detection"], box=s["box"])
    assert load_fighters_edit(data).prompt_frames()[5] == [sug[1]["frame"]]
    assert len(load_fighters_edit(data).objects) == 5


def test_qc_summary_stays_small_for_long_shots():
    """Review item 13: qc.json's summary goes into the stage meta; long lists go to qc_frames.json."""
    raw = make_raw(1800)
    raw.centroid[::4, 0, 0] += 400  # a jump in and out every 4 frames: 450 flagged runs
    raw.boxes[::4, 0] += [600, 0, 600, 0]  # unmatched on those frames
    qc = compute_qc(raw, edit3(), TrackParams())
    a = obj(qc, 1)
    assert a["flagged_runs_total"] >= 450 and len(a["flagged_runs"]) == 20
    assert len(json.dumps(qc.summary)) < 64_000
    assert len(qc.details["objects"][0]["flagged_runs"]) == a["flagged_runs_total"]
    assert len(qc.details["objects"][0]["unmatched_list"]) == 450


def test_review_overlay_uses_match_iou():
    raw = make_raw()
    raw.boxes[10, 0] += [60, 0, 60, 0]  # IoU with its detection 1/3
    qc = compute_qc(raw, edit3(), TrackParams(match_iou=0.3))
    assert "?" not in review_overlay(raw, edit3(), qc, 0.3)["tags"][10][0]
    assert "?" in review_overlay(raw, edit3(), qc, 0.5)["tags"][10][0]


def test_tracks_on_shot_timeline(tmp_path):
    frames, ts = shot_timeline(4, 12.5)
    assert frames.tolist() == [0, 1, 2, 3] and np.allclose(ts, [12.5, 12.5 + 1 / 30, 12.5 + 2 / 30, 12.6])
    raw = make_raw()
    raw.boxes[5, 1] = np.nan
    qc = compute_qc(raw, edit3(), TrackParams())
    tr = build_tracks(raw, edit3(), 2.0, qc)
    assert tr.num_frames == F and tr.boxes.shape == (F, 3, 4) and tr.scores.shape == (F, 3)
    assert tr.fighters == {"A": 1, "B": 2} and tr.track_ids.tolist() == [1, 2, 3]
    assert tr.timestamps[30] == pytest.approx(3.0)
    assert np.isnan(tr.boxes[5, 1]).all() and tr.scores[5, 1] == 0
    save_result(tr, tmp_path / "result.npz")
    back = load_result(tmp_path / "result.npz")
    assert isinstance(back, Tracks) and back.fighters == {"A": 1, "B": 2}
    assert np.array_equal(back.boxes, tr.boxes, equal_nan=True)
    assert back.timestamps.dtype == np.float64 and np.allclose(back.timestamps, tr.timestamps)
