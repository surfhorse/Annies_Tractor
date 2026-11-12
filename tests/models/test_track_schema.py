"""fighters.json schema and pick helpers (no GPU)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from pydantic import ValidationError

from afv.models.track import (
    PromptInfo,
    ShotFrames,
    TrackError,
    TrackParams,
    add_prompt,
    check_prompts_in_image,
    check_shot,
    fighters_json_schema,
    load_fighters_edit,
    next_object_id,
    pick_data,
)

BOXES = {"1": [200.0, 0.0, 290.0, 280.0], "0": [0.0, 0.0, 100.0, 300.0], "2": [400.0, 50.0, 450.0, 200.0]}
GOOD = {
    "fighters": [1, 0],
    "extra": [2],
    "prompt_frame": 0,
    "boxes": BOXES,
    "objects": [{"id": 3, "label": "referee"}],
    "prompts": [
        {"frame": 141, "obj": 4, "detection": 2},
        {"frame": 200, "obj": 1, "box": [20, 0, 120, 300], "points": [[50, 60, 1], [70, 10, 0]]},
    ],
}


def prompt_info(frame: int = 0) -> PromptInfo:
    boxes = np.array([BOXES["0"], BOXES["1"], BOXES["2"]], np.float32)
    return PromptInfo(frame, "p.jpg", boxes, np.array([0.9, 0.8, 0.5], np.float32))


def test_base_picks_plus_reprompts():
    e = load_fighters_edit(GOOD)
    assert e.fighters == {"A": 1, "B": 2}
    assert e.obj_ids == [1, 2, 3, 4] and e.prompt_frame == 0
    assert e.prompt_frames() == {1: [0, 200], 2: [0], 3: [0], 4: [141]}
    assert [o.name for o in e.objects] == ["fighter A", "fighter B", "referee", "person 4"]
    p0 = {(p.frame, p.obj): p for p in e.prompts}
    assert p0[(0, 1)].detection == 1 and p0[(0, 1)].box == BOXES["1"]
    assert p0[(0, 3)].detection == 2 and p0[(141, 4)].box is None


def test_afv_pick_output_alone_is_valid():
    """Exactly what `afv pick 1 --fighters 1,0` writes (module A)."""
    e = load_fighters_edit({
        "fighters": [1, 0], "extra": [], "prompt_frame": None, "boxes": {},
        "shot": {"start": 1.0, "duration": 2.5, "source_sha256": "ab"},
    })  # fmt: skip
    assert e.fighters == {"A": 1, "B": 2} and [p.detection for p in e.prompts] == [1, 0]


def test_full_form_without_base_picks():
    e = load_fighters_edit({
        "objects": [{"id": 7, "role": "A"}, {"id": 9, "role": "B", "label": "blue"}],
        "prompts": [{"frame": 5, "obj": 7, "box": [1, 2, 30, 40]}, {"frame": 5, "obj": 9, "detection": 0}],
    })  # fmt: skip
    assert e.fighters == {"A": 7, "B": 9} and e.prompt_frame == 5


def test_object_relabel_and_prompt_override():
    d = copy.deepcopy(GOOD)
    d["objects"] = [{"id": 1, "label": "red trunks"}]
    d["prompts"] = [{"frame": 0, "obj": 2, "box": [5, 5, 50, 50]}]  # replaces B's base prompt
    e = load_fighters_edit(d)
    assert e.objects[0].name == "red trunks" and e.objects[0].role == "A"
    assert next(p for p in e.prompts if p.obj == 2).box == [5, 5, 50, 50]


@pytest.mark.parametrize(
    ("mutate", "msg"),
    [
        (lambda d: d.__setitem__("fighters", [1]), "two detection numbers"),
        (lambda d: d.__setitem__("fighters", [1, 1]), "distinct"),
        (lambda d: d.__setitem__("extra", [0]), "distinct"),
        (lambda d: (d.pop("fighters"), d.pop("objects")), "extra given without fighters"),
        (lambda d: [d.pop(k) for k in ("fighters", "extra", "objects")], "role A"),
        (lambda d: d["objects"].append({"id": 3, "role": "A"}), "exactly one person with role A"),
        (lambda d: d["prompts"].append({"frame": 141, "obj": 4, "detection": 0}), "same object and frame"),
        (lambda d: d["prompts"].append({"frame": 7, "obj": 1}), "needs a box or a detection"),
        # review item 6: points alone are not a prompt
        (lambda d: d["prompts"].append({"frame": 7, "obj": 1, "points": [[5, 5, 1]]}), "needs a box"),
        (lambda d: d["prompts"].append({"frame": 7, "obj": 1, "box": [10, 10, 5, 20]}), "x1 > x0"),
        (
            lambda d: d["prompts"].append({"frame": 9, "obj": 1, "box": [1, 1, 9, 9], "points": [[1, 2, 3]]}),
            "1|0",
        ),
        (lambda d: d["objects"].append({"id": 5}), "without a prompt"),
        (lambda d: d.__setitem__("typo", 1), "Extra inputs"),
        # review item 10: non-finite numbers
        (lambda d: d["boxes"].__setitem__("0", [0, 0, 1e300, float("inf")]), "finite"),
        (lambda d: d["prompts"].append({"frame": 3, "obj": 1, "box": [0, 0, float("nan"), 9]}), "finite"),
        (
            lambda d: d["prompts"].append(
                {"frame": 3, "obj": 1, "box": [0, 0, 9, 9], "points": [[float("nan"), 5, 1]]}
            ),
            "finite",
        ),
        (lambda d: d.__setitem__("boxes", {**BOXES, "0": [10, 10, 2, 50]}), "x1 > x0"),
        # review item 1: two objects on one frame pointing at one person
        (lambda d: d["prompts"].append({"frame": 0, "obj": 4, "box": [201, 0, 290, 281]}), "same person"),
        (lambda d: d["prompts"].append({"frame": 141, "obj": 3, "detection": 2}), "same person"),
    ],
)
def test_invalid_files(mutate, msg):
    d = copy.deepcopy(GOOD)
    mutate(d)
    with pytest.raises(TrackError, match=msg.replace("|", r"\|")):
        load_fighters_edit(d)


def test_repick_with_swapped_fighters_and_stale_base_frame_prompt_is_rejected():
    """Review item 1 (schema side): a correction made for fighter A on the base frame, left in the
    file after `afv pick` swapped A and B, now targets the other person's base box."""
    fix_a = {"frame": 0, "obj": 1, "box": [205, 2, 288, 280]}  # refines A = detection 1
    stored = {"fighters": [1, 0], "boxes": BOXES, "prompts": [fix_a]}
    assert load_fighters_edit(stored).fighters == {"A": 1, "B": 2}
    with pytest.raises(TrackError, match="objects 1 and 2 on frame 0 point at the same person"):
        load_fighters_edit({**stored, "fighters": [0, 1]})  # what `afv pick --fighters 0,1` used to leave


def test_unreadable_files_are_track_errors(tmp_path):
    p = tmp_path / "fighters.json"
    p.write_text("{not json")
    with pytest.raises(TrackError, match="cannot be read"):
        load_fighters_edit(p)
    with pytest.raises(TrackError, match="cannot be read"):
        load_fighters_edit(tmp_path / "missing.json")
    with pytest.raises(TrackError, match="invalid"):
        add_prompt({"fighters": [0, 1], "prompts": [{"obj": 1, "detection": 0}]}, 3, 1, detection=0)


def test_check_shot():
    d = {**GOOD, "shot": {"start": 4.0, "duration": 1.5, "source_sha256": "abc"}}
    check_shot(d, 4.0, 1.5, "abc")
    check_shot(GOOD, 9.0, 9.0, "zzz")  # no "shot": not checked (hand-written file)
    with pytest.raises(TrackError, match=r"interval 4.000.*Re-pick"):
        check_shot(d, 4.5, 1.5, "abc")
    with pytest.raises(TrackError, match="different source video"):
        check_shot(d, 4.0, 1.5, "other")


def test_prompts_must_lie_in_the_shot():
    e = load_fighters_edit(GOOD)
    fr = ShotFrames("frames", 300, 0.0, 10.0, 640, 360)
    check_prompts_in_image(e, fr)
    with pytest.raises(TrackError, match=r"frames 0..99"):
        check_prompts_in_image(e, ShotFrames("frames", 100, 0.0, 3.3, 640, 360))
    bad = load_fighters_edit({**GOOD, "prompts": [{"frame": 3, "obj": 1, "box": [-500, -500, -400, -400]}]})
    with pytest.raises(TrackError, match="outside the 640x360 image"):
        check_prompts_in_image(bad, fr)
    bad = load_fighters_edit(
        {**GOOD, "prompts": [{"frame": 3, "obj": 1, "box": [1, 1, 50, 50], "points": [[700, 5, 1]]}]}
    )
    with pytest.raises(TrackError, match=r"point \(700.0, 5.0\) is outside"):
        check_prompts_in_image(bad, fr)


def test_prune_window_validation():
    for ok in (0, 16, 20, 100):
        TrackParams(prune_window=ok)
    with pytest.raises(ValidationError, match=">= 16"):
        TrackParams(prune_window=5)


def test_pick_data_matches_afv_pick_format():
    shot = {"start": 1.0, "duration": 2.5, "source_sha256": "ab"}
    d = pick_data(prompt_info(), 1, 0, [2], labels={2: "referee"}, shot=shot)
    assert d == {
        "fighters": [1, 0], "extra": [2], "prompt_frame": 0, "boxes": BOXES, "shot": shot,
        "objects": [{"id": 3, "label": "referee"}],
    }  # fmt: skip
    with pytest.raises(TrackError, match=r"detections 0\.\.2"):
        pick_data(prompt_info(), 0, 5)
    with pytest.raises(TrackError, match=r"not picked: \[5\]"):
        pick_data(prompt_info(), 0, 1, labels={5: "x"})


def test_add_prompt(tmp_path):
    base = pick_data(prompt_info(), 1, 0)
    assert next_object_id(base) == 3
    d = add_prompt(base, 141, 3, detection=2, prompt=prompt_info(141), label="referee")
    assert "prompts" not in base  # a copy
    assert d["fighters"] == [1, 0]
    assert d["prompts"] == [{"frame": 141, "obj": 3, "detection": 2, "box": BOXES["2"]}]  # item 11: box too
    assert d["objects"] == [{"id": 3, "label": "referee"}]
    d2 = add_prompt(d, 141, 3, box=[1, 2, 30, 40])
    assert d2["prompts"] == [{"frame": 141, "obj": 3, "box": [1.0, 2.0, 30.0, 40.0]}]
    d3 = add_prompt(d2, 90, 1, box=[0, 0, 100, 300], points=[[5, 5, 1]])
    assert load_fighters_edit(d3).prompt_frames() == {1: [0, 90], 2: [0], 3: [141]}
    with pytest.raises(TrackError, match="frame 0, not 7"):
        add_prompt(base, 7, 1, detection=0, prompt=prompt_info(0))
    with pytest.raises(TrackError, match="needs a box"):
        add_prompt(base, 7, 1)
    with pytest.raises(TrackError, match="needs a box"):
        add_prompt(base, 7, 1, points=[[5, 5, 1]])
    (tmp_path / "edits").mkdir()
    (tmp_path / "edits" / "fighters.json").write_text(json.dumps(d3))
    assert load_fighters_edit(tmp_path / "edits" / "fighters.json") == load_fighters_edit(d3)


def test_json_schema_exports():
    s = fighters_json_schema()
    assert {"fighters", "extra", "prompt_frame", "boxes", "shot", "objects", "prompts"} <= set(
        s["properties"]
    )
