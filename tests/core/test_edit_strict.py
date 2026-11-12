"""Strict edit validation, afv core side (pydantic): the shared vectors in ``edit_vectors.py`` (the
add-on side runs the same ones in ``tests/blender/test_edit_strict_addon.py``), the message format,
and the writers (``write_edit``) refusing lax files and keeping what they write valid."""

from __future__ import annotations

import json
from decimal import Decimal

import numpy as np
import pytest
from edit_vectors import F_AB, FIGHTERS_INVALID, FIGHTERS_VALID, INVALID, MULTI, VALID

from afv.core.edits import EditError, parse_ids, validate_edit, write_edit


@pytest.mark.parametrize(("vid", "name", "data"), VALID, ids=[v[0] for v in VALID])
def test_valid_vectors_pass(vid, name, data):
    validate_edit(name, data)


@pytest.mark.parametrize(("vid", "name", "data", "error"), INVALID, ids=[v[0] for v in INVALID])
def test_invalid_vectors_fail_with_file_path_and_value(vid, name, data, error):
    with pytest.raises(EditError) as e:
        validate_edit(name, data)
    assert str(e.value) == f"edits/{name}.json: {error}"


@pytest.mark.parametrize(("vid", "name", "data", "first"), MULTI, ids=[v[0] for v in MULTI])
def test_several_errors_start_with_the_first_error(vid, name, data, first):
    with pytest.raises(EditError) as e:
        validate_edit(name, data)
    assert str(e.value).startswith(f"edits/{name}.json: {first}; ")


def test_vectors_cover_every_edit_and_ids_are_unique():
    ids = [v[0] for v in VALID] + [v[0] for v in INVALID]
    assert len(ids) == len(set(ids))
    for name in ("contacts", "pins", "hands", "ids"):
        assert any(v[1] == name for v in VALID) and sum(v[1] == name for v in INVALID) >= 8


def test_unknown_edit_name_and_validation_does_not_change_the_input():
    with pytest.raises(EditError, match="unknown edit 'nope'"):
        validate_edit("nope", {})
    data = {"spans": [{"role": "A", "start": 1, "end": 3, "state": "plant", "foot": "left"}]}
    before = json.dumps(data)
    validate_edit("contacts", data)
    assert json.dumps(data) == before


def test_long_values_are_shortened_in_messages():
    with pytest.raises(EditError) as e:
        validate_edit("pins", {"pins": [{"role": "A" * 500, "frame": 1}]})
    assert len(str(e.value)) < 300 and "..." in str(e.value)


def test_several_errors_are_listed_up_to_three():
    bad = {"spans": [{"role": "C", "start": "1", "end": 2.5, "side": "up"}]}
    with pytest.raises(EditError) as e:
        validate_edit("hands", bad)
    msg = str(e.value)
    assert msg.startswith("edits/hands.json: spans[0].role: ") and "spans[0].start: " in msg
    assert "(and 1 more)" in msg


def test_parse_ids_is_strict_too():
    assert parse_ids(None).swaps == []
    assert parse_ids({"swaps": [{"start": 3, "objects": [1, 2]}]}).swaps[0].end is None
    with pytest.raises(EditError, match=r"swaps\[0\]\.start: must be an integer >= 0, got '3'"):
        parse_ids({"swaps": [{"start": "3", "objects": [1, 2]}]})


# --- writers ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(("vid", "name", "data"), VALID, ids=[v[0] for v in VALID])
def test_write_edit_writes_valid_vectors_and_the_file_passes(shot, vid, name, data):
    path = write_edit(shot, name, data)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(data))  # (a tuple is stored as a list)
    validate_edit(name, stored)


_WRITABLE = [v for v in INVALID if v[2] is not None]  # (writing None removes the file)


@pytest.mark.parametrize(("vid", "name", "data", "error"), _WRITABLE, ids=[v[0] for v in _WRITABLE])
def test_write_edit_refuses_invalid_vectors_and_leaves_the_file(shot, vid, name, data, error):
    write_edit(shot, name, {})
    before = shot.edit_path(name).read_bytes()
    with pytest.raises(EditError):
        write_edit(shot, name, data)
    assert shot.edit_path(name).read_bytes() == before
    assert not [p for p in shot.edit_path(name).parent.iterdir() if p.suffix == ".tmp"]


def test_write_edit_merge_keeps_unknown_top_level_keys_and_validates_the_result(shot):
    write_edit(shot, "contacts", {"spans": [], "note": "mine"})
    span = {"role": "B", "start": 0, "end": 2, "state": "free", "channels": ["x"]}
    write_edit(shot, "contacts", {"spans": [span]})
    assert json.loads(shot.edit_path("contacts").read_text())["note"] == "mine"
    shot.edit_path("contacts").write_text(json.dumps({"spans": [], "note": float("nan")}))  # hand-made
    with pytest.raises(EditError, match=r"contacts\.json: note: NaN or infinity"):
        write_edit(shot, "contacts", {"spans": []})  # the stored NaN survives the merge: refused
    write_edit(shot, "contacts", None)
    assert not shot.edit_path("contacts").exists()


# --- fighters.json: the same strictness (the add-on refuses the name, see its test file) ------------------
@pytest.mark.parametrize(("vid", "data"), FIGHTERS_VALID, ids=[v[0] for v in FIGHTERS_VALID])
def test_valid_fighters_pass_and_are_written(shot, vid, data):
    validate_edit("fighters", data)
    path = write_edit(shot, "fighters", data)
    assert json.loads(path.read_text()) == data


@pytest.mark.parametrize(("vid", "data", "error"), FIGHTERS_INVALID, ids=[v[0] for v in FIGHTERS_INVALID])
def test_invalid_fighters_fail_with_file_path_and_value(shot, vid, data, error):
    with pytest.raises(EditError) as e:
        validate_edit("fighters", data)
    assert str(e.value) == f"edits/fighters.json: {error}"
    with pytest.raises(EditError):
        write_edit(shot, "fighters", data)
    assert not shot.edit_path("fighters").exists()


def test_fighters_json_schema_keeps_its_types():
    from afv.models.track import fighters_json_schema

    s = fighters_json_schema()
    props, defs = s["properties"], s["$defs"]
    ints = {"type": "array", "items": {"type": "integer"}}
    assert props["fighters"]["anyOf"] == [ints, {"type": "null"}]
    assert props["extra"]["items"] == {"type": "integer"} and props["extra"]["type"] == "array"
    assert props["version"]["const"] == 1 and props["version"]["type"] == "integer"
    assert props["prompt_frame"]["anyOf"][0] == {"type": "integer", "minimum": 0}
    assert props["shot"]["anyOf"][0] == {"$ref": "#/$defs/ShotRef"}
    assert props["boxes"]["additionalProperties"] == {"type": "array", "items": {"type": "number"}}
    obj, prompt, ref = defs["TrackObject"]["properties"], defs["TrackPrompt"]["properties"], defs["ShotRef"]
    assert obj["id"]["type"] == "integer" and obj["id"]["minimum"] == 1
    assert obj["role"]["anyOf"][0] == {"type": "string", "enum": ["A", "B"]}
    assert obj["label"]["type"] == "string"
    assert prompt["frame"]["minimum"] == 0 and prompt["obj"]["minimum"] == 1
    assert prompt["detection"]["anyOf"][0] == {"type": "integer", "minimum": 0}
    assert prompt["box"]["anyOf"][0] == {"type": "array", "items": {"type": "number"}}
    assert prompt["points"]["anyOf"][0]["items"] == {"type": "array", "items": {"type": "number"}}
    assert ref["properties"]["source_sha256"]["type"] == "string"
    assert ref["properties"]["start"]["type"] == "number"


def test_stage_loaded_fighters_fail_with_the_same_message(shot):
    from afv.models.track import TrackError, load_fighters_edit

    want = r"edits/fighters\.json is invalid: prompt_frame: must be an integer >= 0"
    with pytest.raises(TrackError, match=want):
        load_fighters_edit({**F_AB, "prompt_frame": "0"})


# --- python-only values: always an EditError, never another exception type -------------------------------
@pytest.mark.parametrize(
    "doc",
    [
        {"spans": [], "note": np.float32("nan")},
        {"spans": [], "note": np.float32(1.5)},
        {"spans": [], "note": np.int64(3)},
        {"spans": [], "note": Decimal("NaN")},
        {"spans": [], "note": complex(float("nan"), 0)},
        {"spans": [], 7: "x"},
        {"spans": [], "note": 10**5000},
        {"spans": [{"role": "A", "start": 10**5000, "end": 3, "state": "plant", "foot": "left"}]},
        {"spans": [{"role": "A", "start": 1, "end": -(10**400), "state": "plant", "foot": "left"}]},
    ],
)
def test_python_only_values_raise_edit_error_from_validate_and_write(shot, doc):
    with pytest.raises(EditError):
        validate_edit("contacts", doc)
    with pytest.raises(EditError):
        write_edit(shot, "contacts", doc)
    assert not shot.edit_path("contacts").exists()


def test_a_message_never_repeats_a_huge_value():
    huge = {"spans": [{"role": "A", "start": 1, "end": -(10**400), "state": "plant", "foot": "left"}]}
    with pytest.raises(EditError) as e:
        validate_edit("contacts", huge)
    assert len(str(e.value)) < 200
