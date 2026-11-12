"""Strict edit validation, Blender add-on side (``blender/afv_addon/edits.py``, plain Python): the
shared vectors in ``tests/core/edit_vectors.py``, the same ones ``tests/core/test_edit_strict.py`` runs
through afv core; a side-by-side check of both validators; the add-on's writers produce files that
pass and refuse lax input."""

from __future__ import annotations

import json
import math
import sys

import pytest
from take_helpers import REPO, addon_module

sys.path.insert(0, str(REPO / "tests" / "core"))
from edit_vectors import INVALID, MULTI, VALID

edits = addon_module("edits")


@pytest.mark.parametrize(("vid", "name", "data"), VALID, ids=[v[0] for v in VALID])
def test_valid_vectors_pass(vid, name, data):
    edits.validate(name, data)


@pytest.mark.parametrize(("vid", "name", "data", "error"), INVALID, ids=[v[0] for v in INVALID])
def test_invalid_vectors_fail_with_file_path_and_value(vid, name, data, error):
    with pytest.raises(edits.EditError) as e:
        edits.validate(name, data)
    assert str(e.value) == f"edits/{name}.json: {error}"


@pytest.mark.parametrize(("vid", "name", "data"), [v[:3] for v in [*VALID, *INVALID]],
                         ids=[v[0] for v in [*VALID, *INVALID]])
def test_addon_and_core_agree(vid, name, data):
    from afv.core import edits as core

    def verdict(fn):
        try:
            fn(name, data)
            return None
        except Exception as e:
            return str(e)

    assert verdict(edits.validate) == verdict(core.validate_edit)


@pytest.mark.parametrize(("vid", "name", "data", "first"), MULTI, ids=[v[0] for v in MULTI])
def test_several_errors_the_addon_reports_the_first(vid, name, data, first):
    from afv.core import edits as core

    with pytest.raises(edits.EditError) as e:
        edits.validate(name, data)
    assert str(e.value) == f"edits/{name}.json: {first}"
    with pytest.raises(core.EditError) as c:
        core.validate_edit(name, data)
    assert str(c.value).startswith(str(e.value) + "; ")


def test_addon_refuses_fighters_and_unknown_names():
    with pytest.raises(edits.EditError, match="afv pick"):
        edits.validate("fighters", {})
    with pytest.raises(edits.EditError, match="unknown edit"):
        edits.validate("bogus", {})


def test_python_only_values_agree_with_core():
    """Values JSON cannot carry (tuples, numpy numbers, non-string keys) are refused the same way."""
    import numpy as np

    from afv.core import edits as core

    span = {"role": "A", "start": 1, "end": 3, "state": "plant", "foot": "left"}
    pin = {"role": "A", "frame": 1, "preset": "p", "bones": {"X": [0, 0, 0, 1]}}
    cases = [
        ("contacts", {"spans": ({**span},)}),
        ("contacts", {"spans": [{**span, "start": np.int64(1)}]}),
        ("contacts", {"spans": [{**span, "foot": ("left",)}]}),
        ("pins", {"pins": [{**pin, "bones": {"X": (0, 0, 0, 1)}}]}),
        ("pins", {"pins": [{**pin, "bones": {"X": np.array([0, 0, 0, 1.0])}}]}),
        ("pins", {"pins": [{**pin, "bones": {"X": [np.float64(0), 0, 0, 1]}}]}),
        ("pins", {"pins": [{**pin, "bones": {1: [0, 0, 0, 1]}}]}),
        ("pins", {"pins": [{**pin, "bones": {"X": [0, 0, 0, math.inf]}}]}),
        ("ids", {"swaps": [{"start": 1, "objects": (1, 2)}]}),
        ("ids", {"swaps": [{"start": 1, "objects": [np.int64(1), 2]}]}),
    ]
    for name, data in cases:
        results = []
        for fn in (edits.validate, core.validate_edit):
            try:
                fn(name, data)
                results.append("ok")
            except Exception:
                results.append("refused")
        assert results[0] == results[1], (name, data, results)


# --- writers ---------------------------------------------------------------------------------------
def _round_trip(tmp_path, name, data):
    from afv.core import edits as core

    edits.write_local(tmp_path, name, data)
    stored = edits.read(tmp_path, name)
    edits.validate(name, stored)
    core.validate_edit(name, stored)  # what afv serve and the stages' side accept
    return stored


def test_writers_produce_files_that_pass_both_validators(tmp_path):
    data = edits.paint_contact({}, "A", "left", 10, 20, "plant")
    data = edits.paint_contact(data, "B", "right", 0, 5, "free")
    data = edits.paint_contact(data, "A", "left", 15, 18, "free")
    assert _round_trip(tmp_path, "contacts", data)["spans"]
    data = edits.mark_hands({}, "A", "both", 5, 30)
    data = edits.mark_hands(data, "A", "left", 10, 12, off=False)
    assert len(_round_trip(tmp_path, "hands", data)["spans"]) == 3
    pin = {"role": "A", "frame": 12, "radius": 8, "preset": "p", "bones": {"LeftArm": [0, 0, 0, 1]},
           "root": [0.0, 1.0, 0.0], "prefix": "", "root_motion": "world"}
    pins = edits.add_pin({}, pin)
    pins = edits.add_pin(pins, {**pin, "frame": 20, "root": None})
    pins = edits.add_pin(pins, {**pin, "frame": 12, "bones": {"Head": [0.0, 0.0, 0.0, 1.0]}})  # merged
    assert len(_round_trip(tmp_path, "pins", pins)["pins"]) == 2
    swaps = edits.add_swap({}, 40, (2, 1))
    swaps = edits.add_swap(swaps, 60, [1, 2], end=80)
    assert len(_round_trip(tmp_path, "ids", swaps)["swaps"]) == 2


def test_writers_keep_unknown_keys_and_refuse_lax_values():
    span = {"role": "A", "start": 1, "end": 5, "state": "plant", "foot": "left", "note": "x"}
    out = edits.paint_contact({"version": 2, "spans": [span]}, "A", "left", 2, 3, "free")
    assert out["version"] == 2 and all(s.get("note", "x") == "x" for s in out["spans"])
    for bad in (True, 2.0, 2.5, "2", None):
        with pytest.raises(edits.EditError, match="start frame must be an integer"):
            edits.paint_contact({}, "A", "left", bad, 9, "plant")
        with pytest.raises(edits.EditError, match="end frame must be an integer"):
            edits.mark_hands({}, "A", "left", 1, bad)
        with pytest.raises(edits.EditError, match="swap start frame must be an integer"):
            edits.add_swap({}, bad, (1, 2))
    for bad in (True, 1.0, "1"):
        with pytest.raises(edits.EditError, match="swap object must be an integer"):
            edits.add_swap({}, 3, (bad, 2))
    with pytest.raises(edits.EditError, match="swap end frame"):
        edits.add_swap({}, 3, (1, 2), end="9")
    with pytest.raises(edits.EditError, match="pin frame must be an integer"):
        edits.add_pin({}, {"role": "A", "frame": "3", "bones": {"X": [0, 0, 0, 1]}, "preset": "p"})


def test_writers_refuse_an_existing_file_with_lax_values_instead_of_rewriting_it():
    lax = {"spans": [{"role": "A", "start": "1", "end": 5, "state": "plant", "foot": "left"}]}
    with pytest.raises(edits.EditError, match=r"edits/contacts\.json: spans\[0\]\.start: must be an integer"):
        edits.paint_contact(lax, "A", "left", 0, 3, "plant")
    with pytest.raises(edits.EditError, match=r"edits/hands\.json: spans\[0\]\.start"):
        edits.mark_hands({"spans": [{"role": "A", "start": 1.0, "end": 5, "side": "left"}]}, "A", "left", 0,
                         3)
    with pytest.raises(edits.EditError, match=r"edits/pins\.json: pins\[0\]\.frame"):
        edits.add_pin({"pins": [{"role": "A", "frame": "4", "preset": "p", "root": [0, 0, 0]}]},
                      {"role": "A", "frame": 5, "bones": {"X": [0, 0, 0, 1]}})
    with pytest.raises(edits.EditError, match=r"edits/ids\.json: swaps\[0\]\.start"):
        edits.add_swap({"swaps": [{"start": "1", "objects": [1, 2]}]}, 3, (1, 2))


def test_write_local_refuses_lax_values_and_leaves_the_file(tmp_path):
    ok = {"spans": [{"role": "A", "start": 1, "end": 3, "state": "plant", "foot": "left"}]}
    edits.write_local(tmp_path, "contacts", ok)
    before = edits.edit_path(tmp_path, "contacts").read_bytes()
    for span in ({**ok["spans"][0], "start": "1"}, {**ok["spans"][0], "end": 3.0},
                 {**ok["spans"][0], "end": True}, {**ok["spans"][0], "start": float("nan")}):
        with pytest.raises(edits.EditError, match=r"edits/contacts\.json: spans\[0\]\."):
            edits.write_local(tmp_path, "contacts", {"spans": [span]})
    assert edits.edit_path(tmp_path, "contacts").read_bytes() == before
    assert json.loads(before) == ok


# --- the first error is the same on both sides, over generated documents -----------------------------
def _outcome(fn, name, data):
    try:
        fn(name, data)
    except Exception as e:
        assert type(e).__name__ == "EditError", (name, data, type(e), e)
        return str(e)
    return None


def _mutations(rng, n):
    """Seeded mutations of the valid vectors: one to three values replaced by odd ones (wrong types,
    NaN, numpy, mappings, huge ints, extra keys, dropped keys)."""
    import collections
    import copy
    import types
    from decimal import Decimal

    import numpy as np

    odd = [True, False, 0, 1, -1, 2, 3, 7, 10**12, 10**400, -(10**400), 10**5000, 1.0, 1.5, 3.0, 1e308,
           math.nan, math.inf, "1", "A", "B", "left", "both", "plant", "free", "", None, [], {}, [1, 2],
           [0, 0, 0, 1], [0, 0, 0, 3], [0, 0, 1], [0.0, 1.0, 0.0], ["LeftFoot"], (0, 0, 0, 1),
           np.float32("nan"), np.float64(1.0), np.int64(1), Decimal("NaN"), complex(1, 0), b"A", {1, 2},
           np.array([0, 0, 0, 1.0]),
           types.MappingProxyType({"role": "A"}), collections.UserDict(), collections.OrderedDict()]
    keys = ["x", "", 1, None, ("a",), "spans", "pins", "swaps"]
    docs = [(name, doc) for _vid, name, doc in VALID if isinstance(doc, dict)]

    def paths(v, p=()):
        yield p
        if isinstance(v, dict):
            for k, x in v.items():
                yield from paths(x, (*p, k))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                yield from paths(x, (*p, i))

    for _ in range(n):
        name, base = rng.choice(docs)
        try:
            doc = copy.deepcopy(base)
        except TypeError:
            doc = base
        for _ in range(rng.choice([1, 1, 2, 3])):
            p = rng.choice(list(paths(doc)))
            if not p:
                doc[rng.choice(keys)] = _pick(rng, odd)
                continue
            parent = doc
            for k in p[:-1]:
                parent = parent[k]
            op = rng.random()
            if op < 0.7:
                parent[p[-1]] = _pick(rng, odd)
            elif op < 0.85 and isinstance(parent, dict):
                del parent[p[-1]]
            elif isinstance(parent[p[-1]], dict):
                parent[p[-1]][rng.choice(keys)] = _pick(rng, odd)
            elif isinstance(parent[p[-1]], list):
                parent[p[-1]].append(_pick(rng, odd))
        yield name, doc


def test_same_verdict_and_same_first_error_over_generated_documents():
    import random

    from afv.core import edits as core

    refused = 0
    for name, doc in _mutations(random.Random(5), 4000):
        a = _outcome(edits.validate, name, doc)
        c = _outcome(core.validate_edit, name, doc)
        assert (a is None) == (c is None), (name, doc, a, c)
        if a is not None:
            refused += 1
            # core lists up to three errors, the add-on stops at the first: the first one is equal
            assert c == a or c.startswith(a + "; "), (name, doc, a, c)
    assert refused > 1000


def test_python_only_values_are_edit_errors_on_both_sides():
    import numpy as np

    from afv.core import edits as core

    for doc in ({"spans": [], "note": np.float32("nan")}, {"spans": [], "note": np.int64(3)},
                {"spans": [], 1: 2}, {"spans": [], "note": 10**5000},
                {"spans": [{"role": "A", "start": 1, "end": -(10**400), "state": "plant", "foot": "left"}]}):
        assert _outcome(edits.validate, "contacts", doc) and _outcome(core.validate_edit, "contacts", doc)


def test_fighters_is_core_only():
    from edit_vectors import F_AB

    with pytest.raises(edits.EditError, match="written by 'afv pick'"):
        edits.validate("fighters", F_AB)


# --- writers: hints, and what the add-on really writes ---------------------------------------------------
def test_lax_existing_file_is_refused_with_a_hint_to_fix_it():
    lax = {"spans": [{"role": "A", "start": "1", "end": 5, "state": "plant", "foot": "left"}]}
    with pytest.raises(edits.EditError) as e:
        edits.paint_contact(lax, "A", "left", 0, 3, "plant")
    assert str(e.value).endswith("(fix or delete that entry in edits/contacts.json)")
    assert "spans[0].start: must be an integer >= 0, got '1'" in str(e.value)


def test_a_pin_as_pose_pin_builds_it_passes_both_validators(tmp_path):
    """The dict ``cleanup.pose_pin`` returns (same code: ints, rounded floats, str keys), through add_pin,
    write_local and both validators. pose_pin itself needs Blender."""
    import numpy as np

    from afv.core import edits as core

    rest = np.array([0.1, 0.2, 0.3, 0.9])
    rest /= np.linalg.norm(rest)
    basis = np.array([0.9, 0.1, 0.0, 0.2])
    basis /= np.linalg.norm(basis)
    local = [round(float(v), 7) for v in edits.basis_to_local(rest, basis)]
    pin = {"role": "B", "frame": 7, "radius": 8, "preset": "blender",
           "bones": {edits.canonical("mixamorig:LeftArm", "mixamorig:"): local},
           "root": [round(float(v), 6) for v in np.array([0.0, 1.0123456789, -0.5])]}
    pin["root_motion"] = "extracted"
    pin["prefix"] = "mixamorig:"
    pin["root_rotation"] = local
    data = edits.add_pin({}, pin)
    data = edits.add_pin(data, {**pin, "frame": 20, "root": None, "root_rotation": None})
    stored = _round_trip(tmp_path, "pins", data)
    core.validate_edit("pins", stored)
    assert len(stored["pins"]) == 2 and stored["pins"][0]["root_rotation"] == local


def _pick(rng, odd):
    import copy

    v = rng.choice(odd)
    try:
        return copy.deepcopy(v)
    except TypeError:
        return v


def test_delete_and_revert_paths_give_the_hint_when_another_entry_is_lax(tmp_path):
    """Remove span / Delete pin / Delete swap / Revert write the whole file through write_local."""
    good = {"role": "A", "frame": 4, "preset": "p", "root": [0, 0, 0]}
    lax = {"role": "A", "frame": "9", "preset": "p", "root": [0, 0, 0]}
    shot = tmp_path
    p = edits.edit_path(shot, "pins")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"pins": [good, lax]}))
    data, _gone = edits.delete_pin(edits.read(shot, "pins"), 0)  # deletes the good one: lax stays
    with pytest.raises(edits.EditError) as e:
        edits.write_local(shot, "pins", data)
    assert "pins[0].frame: must be an integer >= 0, got '9'" in str(e.value)
    assert "(the file on disk is invalid too: fix or delete that entry in edits/pins.json)" in str(e.value)
    data, _gone = edits.delete_pin(edits.read(shot, "pins"), 1)  # deletes the lax one: fine
    edits.write_local(shot, "pins", data)
    assert edits.read(shot, "pins")["pins"] == [good]
    # a file that is not an object
    want = r"must be a JSON object, got \[1\] .fix or delete the file edits/ids\.json"
    with pytest.raises(edits.EditError, match=want):
        edits.validate_stored("ids", [1])
