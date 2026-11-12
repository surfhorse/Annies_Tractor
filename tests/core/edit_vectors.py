"""Shared test vectors for the two edit validators: ``afv.core.edits.validate_edit`` (pydantic) and
``blender/afv_addon/edits.py`` ``validate`` (hand-written, runs inside Blender without afv).

Both must accept exactly the VALID documents and refuse exactly the INVALID ones. The message of a
document with ONE error is the same on both sides, ``edits/<name>.json: <error>``; for a document with
several errors core lists up to three (``e1; e2; e3 (and N more)``) and the add-on only the first, and
the FIRST error is the same (the add-on checks in the order of core's models). Used by
``tests/core/test_edit_strict.py`` (core side) and
``tests/blender/test_edit_strict_addon.py`` (add-on side, and both side by side).

Rules (see the "strict values" comment in ``afv/core/edits.py``): integers are JSON integers (no bool,
no 3.0, no "3"); numbers are ints or floats, finite (no bool, no "0.5"); strings are strings; null only
where a field allows it; NaN/infinity, keys that are not strings, integers over 4096 bits and values
JSON cannot hold (Decimal, complex, sets, other mappings) are refused anywhere, unknown keys included;
unknown keys kept. ``FIGHTERS_VALID``/``FIGHTERS_INVALID`` (fighters.json) are for afv core only: the
add-on refuses to validate that name.
"""

from __future__ import annotations

import math
from collections import OrderedDict, UserDict
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType

NAN, INF = math.nan, math.inf

SPAN = {"role": "A", "start": 1, "end": 3, "state": "plant", "foot": "left"}
HAND = {"role": "A", "start": 1, "end": 3, "side": "left"}
PIN = {"role": "A", "frame": 1, "preset": "p", "bones": {"X": [0, 0, 0, 1]}}
SWAP = {"start": 1, "end": None, "objects": [1, 2]}

# (id, edit name, document)
VALID: list[tuple[str, str, object]] = [
    ("contacts-empty-object", "contacts", {}),
    ("contacts-empty-list", "contacts", {"spans": []}),
    ("contacts-plant-left", "contacts", {"spans": [SPAN]}),
    ("contacts-free-right-B", "contacts", {"spans": [{**SPAN, "role": "B", "state": "free",
                                         "foot": "right"}]}),
    ("contacts-channels", "contacts", {"spans": [{**SPAN, "foot": None, "channels": ["LeftFoot", "x"]}]}),
    ("contacts-channels-no-foot-key", "contacts",
     {"spans": [{k: v for k, v in SPAN.items() if k != "foot"} | {"channels": ["LeftFoot"]}]}),
    ("contacts-foot-and-null-channels", "contacts", {"spans": [{**SPAN, "channels": None}]}),
    ("contacts-start-zero", "contacts", {"spans": [{**SPAN, "start": 0, "end": 1}]}),
    ("contacts-big-frames", "contacts", {"spans": [{**SPAN, "start": 10**12, "end": 10**12 + 1}]}),
    ("contacts-unknown-keys", "contacts",
     {"version": 3, "note": "x", "obj": {"a": [1, True, "s", None, 2.5]}, "spans": [{**SPAN, "note": True}]}),
    ("hands-empty", "hands", {}),
    ("hands-left", "hands", {"spans": [HAND]}),
    ("hands-right-both", "hands",
     {"spans": [{**HAND, "side": "right"}, {**HAND, "role": "B", "side": "both", "x": [1]}]}),
    ("pins-empty", "pins", {}),
    ("pins-bone", "pins", {"pins": [PIN]}),
    ("pins-root-only", "pins", {"pins": [{**PIN, "bones": {}, "root": [0, 1.5, -2]}]}),
    ("pins-no-bones-key-root", "pins",
     {"pins": [{k: v for k, v in PIN.items() if k != "bones"} | {"root": [0, 1, 2]}]}),
    ("pins-null-root", "pins", {"pins": [{**PIN, "root": None}]}),
    ("pins-radius-12", "pins", {"pins": [{**PIN, "radius": 12}]}),
    ("pins-empty-preset", "pins", {"pins": [{**PIN, "preset": ""}]}),
    ("pins-float-quaternion", "pins", {"pins": [{**PIN, "bones": {"X": [0.0, 0.7071068, 0.0, 0.7071068]}}]}),
    ("pins-many-bones", "pins",
     {"pins": [{**PIN, "bones": {"A": [0, 0, 0, 1], "B": [1, 0, 0, 0], "C": [0.5, 0.5, 0.5, 0.5]}}]}),
    ("pins-addon-keys", "pins",
     {"pins": [{**PIN, "root_motion": "world", "prefix": "", "root_rotation": [0, 0, 0, 1]}]}),
    ("ids-empty", "ids", {}),
    ("ids-open-ended", "ids", {"swaps": [SWAP]}),
    ("ids-missing-end", "ids", {"swaps": [{"start": 1, "objects": [1, 2]}]}),
    ("ids-ended", "ids", {"swaps": [{"start": 0, "end": 5, "objects": [3, 2], "x": None}]}),
    ("ids-unknown-keys", "ids", {"swaps": [SWAP], "note": "kept"}),
]

# (id, edit name, document, expected error after "edits/<name>.json: ")
INVALID: list[tuple[str, str, object, str]] = []


def _bad(vid: str, name: str, data: object, error: str) -> None:
    INVALID.append((vid, name, data, error))


def _cut(text: str) -> str:
    """A repr as the messages show it: cut at 80 characters."""
    return text if len(text) <= 80 else text[:77] + "..."


# (id, edit name, document, the FIRST error): documents with several errors. Core lists up to three, the
# add-on stops at the first; the first is the same on both sides.
MULTI: list[tuple[str, str, object, str]] = []


def _multi(vid: str, name: str, data: object, error: str) -> None:
    MULTI.append((vid, name, data, error))


# --- not an object ---
_bad("contacts-list", "contacts", [], "must be a JSON object, got []")
_bad("pins-string", "pins", "x", "must be a JSON object, got 'x'")
_bad("hands-null", "hands", None, "must be a JSON object, got None")
_bad("ids-number", "ids", 5, "must be a JSON object, got 5")

# --- NaN / infinity anywhere ---
_bad("contacts-nan-top", "contacts", {"spans": [], "note": NAN}, "note: NaN or infinity is not allowed")
_bad("contacts-inf-unknown-nested", "contacts", {"spans": [{**SPAN, "extra": {"a": [1, INF]}}]},
     "spans[0].extra.a[1]: NaN or infinity is not allowed")
_bad("contacts-nan-start", "contacts", {"spans": [{**SPAN, "start": NAN}]},
     "spans[0].start: NaN or infinity is not allowed")
_bad("pins-nan-quat", "pins", {"pins": [{**PIN, "bones": {"X": [0, 0, NAN, 1]}}]},
     "pins[0].bones.X[2]: NaN or infinity is not allowed")
_bad("pins-neg-inf-root", "pins", {"pins": [{**PIN, "root": [0, -INF, 0]}]},
     "pins[0].root[1]: NaN or infinity is not allowed")
_bad("hands-nan-unknown", "hands", {"spans": [{**HAND, "x": NAN}]},
     "spans[0].x: NaN or infinity is not allowed")
_bad("ids-inf-end", "ids", {"swaps": [{**SWAP, "end": INF}]}, "swaps[0].end: NaN or infinity is not allowed")

# --- list / object shape ---
_bad("contacts-spans-null", "contacts", {"spans": None}, "spans: must be a list, got None")
_bad("contacts-spans-dict", "contacts", {"spans": {}}, "spans: must be a list, got {}")
_bad("contacts-span-string", "contacts", {"spans": ["x"]}, "spans[0]: must be an object, got 'x'")
_bad("hands-spans-string", "hands", {"spans": "x"}, "spans: must be a list, got 'x'")
_bad("pins-pins-null", "pins", {"pins": None}, "pins: must be a list, got None")
_bad("pins-pin-list", "pins", {"pins": [PIN, []]}, "pins[1]: must be an object, got []")
_bad("ids-swaps-int", "ids", {"swaps": 1}, "swaps: must be a list, got 1")
_bad("ids-swap-null", "ids", {"swaps": [None]}, "swaps[0]: must be an object, got None")

# --- role ---
_bad("contacts-role-C", "contacts", {"spans": [{**SPAN, "role": "C"}]},
     "spans[0].role: must be 'A' or 'B', got 'C'")
_bad("contacts-role-lower", "contacts", {"spans": [{**SPAN, "role": "a"}]},
     "spans[0].role: must be 'A' or 'B', got 'a'")
_bad("contacts-role-int", "contacts", {"spans": [{**SPAN, "role": 1}]},
     "spans[0].role: must be 'A' or 'B', got 1")
_bad("contacts-role-true", "contacts", {"spans": [{**SPAN, "role": True}]},
     "spans[0].role: must be 'A' or 'B', got True")
_bad("contacts-role-missing", "contacts", {"spans": [{k: v for k, v in SPAN.items() if k != "role"}]},
     "spans[0].role: is required")
_bad("hands-role-null", "hands", {"spans": [{**HAND, "role": None}]},
     "spans[0].role: must be 'A' or 'B', got None")
_bad("pins-role-list", "pins", {"pins": [{**PIN, "role": ["A"]}]},
     "pins[0].role: must be 'A' or 'B', got ['A']")

# --- frames: ints only ---
_bad("contacts-start-float", "contacts", {"spans": [{**SPAN, "start": 1.0}]},
     "spans[0].start: must be an integer >= 0, got 1.0")
_bad("contacts-start-string", "contacts", {"spans": [{**SPAN, "start": "1"}]},
     "spans[0].start: must be an integer >= 0, got '1'")
_bad("contacts-start-true", "contacts", {"spans": [{**SPAN, "start": True}]},
     "spans[0].start: must be an integer >= 0, got True")
_bad("contacts-start-negative", "contacts", {"spans": [{**SPAN, "start": -1}]},
     "spans[0].start: must be an integer >= 0, got -1")
_bad("contacts-start-null", "contacts", {"spans": [{**SPAN, "start": None}]},
     "spans[0].start: must be an integer >= 0, got None")
_bad("contacts-start-missing", "contacts", {"spans": [{k: v for k, v in SPAN.items() if k != "start"}]},
     "spans[0].start: is required")
_bad("contacts-end-fraction", "contacts", {"spans": [{**SPAN, "end": 3.5}]},
     "spans[0].end: must be an integer, got 3.5")
_bad("contacts-end-integral-float", "contacts", {"spans": [{**SPAN, "end": 3.0}]},
     "spans[0].end: must be an integer, got 3.0")
_bad("contacts-end-string", "contacts", {"spans": [{**SPAN, "end": " 3 "}]},
     "spans[0].end: must be an integer, got ' 3 '")
_bad("contacts-end-false", "contacts", {"spans": [{**SPAN, "end": False}]},
     "spans[0].end: must be an integer, got False")
_bad("contacts-end-missing", "contacts", {"spans": [{k: v for k, v in SPAN.items() if k != "end"}]},
     "spans[0].end: is required")
_bad("contacts-end-equal", "contacts", {"spans": [{**SPAN, "start": 3, "end": 3}]},
     "spans[0]: end (3) must be after start (3) (end is exclusive)")
_bad("contacts-end-before", "contacts", {"spans": [SPAN, {**SPAN, "start": 5, "end": 2}]},
     "spans[1]: end (2) must be after start (5) (end is exclusive)")
_bad("hands-start-float", "hands", {"spans": [{**HAND, "start": 2.0}]},
     "spans[0].start: must be an integer >= 0, got 2.0")
_bad("hands-end-string", "hands", {"spans": [{**HAND, "end": "9"}]},
     "spans[0].end: must be an integer, got '9'")
_bad("hands-end-equal", "hands", {"spans": [{**HAND, "end": 1}]},
     "spans[0]: end (1) must be after start (1) (end is exclusive)")
_bad("pins-frame-string", "pins", {"pins": [{**PIN, "frame": "5"}]},
     "pins[0].frame: must be an integer >= 0, got '5'")
_bad("pins-frame-float", "pins", {"pins": [{**PIN, "frame": 5.0}]},
     "pins[0].frame: must be an integer >= 0, got 5.0")
_bad("pins-frame-bool", "pins", {"pins": [{**PIN, "frame": True}]},
     "pins[0].frame: must be an integer >= 0, got True")
_bad("pins-frame-negative", "pins", {"pins": [{**PIN, "frame": -1}]},
     "pins[0].frame: must be an integer >= 0, got -1")
_bad("pins-frame-missing", "pins", {"pins": [{k: v for k, v in PIN.items() if k != "frame"}]},
     "pins[0].frame: is required")
_bad("pins-radius-float", "pins", {"pins": [{**PIN, "radius": 2.0}]},
     "pins[0].radius: must be an integer >= 1, got 2.0")
_bad("pins-radius-fraction", "pins", {"pins": [{**PIN, "radius": 2.5}]},
     "pins[0].radius: must be an integer >= 1, got 2.5")
_bad("pins-radius-negative", "pins", {"pins": [{**PIN, "radius": -1}]},
     "pins[0].radius: must be an integer >= 1, got -1")
_bad("pins-radius-string", "pins", {"pins": [{**PIN, "radius": "8"}]},
     "pins[0].radius: must be an integer >= 1, got '8'")
_bad("pins-radius-bool", "pins", {"pins": [{**PIN, "radius": True}]},
     "pins[0].radius: must be an integer >= 1, got True")
_bad("pins-radius-null", "pins", {"pins": [{**PIN, "radius": None}]},
     "pins[0].radius: must be an integer >= 1, got None")
_bad("ids-start-string", "ids", {"swaps": [{**SWAP, "start": "1"}]},
     "swaps[0].start: must be an integer >= 0, got '1'")
_bad("ids-start-float", "ids", {"swaps": [{**SWAP, "start": 1.0}]},
     "swaps[0].start: must be an integer >= 0, got 1.0")
_bad("ids-start-negative", "ids", {"swaps": [{**SWAP, "start": -3}]},
     "swaps[0].start: must be an integer >= 0, got -3")
_bad("ids-start-missing", "ids", {"swaps": [{"end": None, "objects": [1, 2]}]}, "swaps[0].start: is required")
_bad("ids-end-float", "ids", {"swaps": [{**SWAP, "end": 4.0}]}, "swaps[0].end: must be an integer, got 4.0")
_bad("ids-end-string", "ids", {"swaps": [{**SWAP, "end": "4"}]}, "swaps[0].end: must be an integer, got '4'")
_bad("ids-end-bool", "ids", {"swaps": [{**SWAP, "end": True}]}, "swaps[0].end: must be an integer, got True")
_bad("ids-end-equal", "ids", {"swaps": [{"start": 4, "end": 4, "objects": [1, 2]}]},
     "swaps[0]: end (4) must be after start (4) (end is exclusive)")

# --- strings / literals ---
_bad("contacts-state-hover", "contacts", {"spans": [{**SPAN, "state": "hover"}]},
     "spans[0].state: must be 'plant' or 'free', got 'hover'")
_bad("contacts-state-clear", "contacts", {"spans": [{**SPAN, "state": "clear"}]},
     "spans[0].state: must be 'plant' or 'free', got 'clear'")
_bad("contacts-state-null", "contacts", {"spans": [{**SPAN, "state": None}]},
     "spans[0].state: must be 'plant' or 'free', got None")
_bad("contacts-state-missing", "contacts", {"spans": [{k: v for k, v in SPAN.items() if k != "state"}]},
     "spans[0].state: is required")
_bad("contacts-foot-middle", "contacts", {"spans": [{**SPAN, "foot": "middle"}]},
     "spans[0].foot: must be 'left' or 'right', got 'middle'")
_bad("contacts-foot-upper", "contacts", {"spans": [{**SPAN, "foot": "Left"}]},
     "spans[0].foot: must be 'left' or 'right', got 'Left'")
_bad("contacts-foot-int", "contacts", {"spans": [{**SPAN, "foot": 0}]},
     "spans[0].foot: must be 'left' or 'right', got 0")
_bad("contacts-foot-neither", "contacts", {"spans": [{**SPAN, "foot": None}]},
     "spans[0]: a contact span needs exactly one of 'foot' or 'channels'")
_bad("contacts-foot-neither-missing", "contacts", {"spans": [{k: v for k, v in SPAN.items() if k != "foot"}]},
     "spans[0]: a contact span needs exactly one of 'foot' or 'channels'")
_bad("contacts-foot-and-channels", "contacts", {"spans": [{**SPAN, "channels": ["LeftFoot"]}]},
     "spans[0]: a contact span needs exactly one of 'foot' or 'channels'")
_bad("contacts-channels-numbers", "contacts", {"spans": [{**SPAN, "foot": None, "channels": [1, 2]}]},
     "spans[0].channels: must be a non-empty list of channel names, got [1, 2]")
_bad("contacts-channels-empty", "contacts", {"spans": [{**SPAN, "foot": None, "channels": []}]},
     "spans[0].channels: must be a non-empty list of channel names, got []")
_bad("contacts-channels-string", "contacts", {"spans": [{**SPAN, "foot": None, "channels": "LeftFoot"}]},
     "spans[0].channels: must be a non-empty list of channel names, got 'LeftFoot'")
_bad("contacts-channels-mixed", "contacts", {"spans": [{**SPAN, "foot": None, "channels": ["a", None]}]},
     "spans[0].channels: must be a non-empty list of channel names, got ['a', None]")
_bad("hands-side-up", "hands", {"spans": [{**HAND, "side": "up"}]},
     "spans[0].side: must be 'left', 'right' or 'both', got 'up'")
_bad("hands-side-missing", "hands", {"spans": [{"role": "A", "start": 1, "end": 3}]},
     "spans[0].side: is required")
_bad("hands-side-null", "hands", {"spans": [{**HAND, "side": None}]},
     "spans[0].side: must be 'left', 'right' or 'both', got None")
_bad("hands-side-list", "hands", {"spans": [{**HAND, "side": ["left"]}]},
     "spans[0].side: must be 'left', 'right' or 'both', got ['left']")
_bad("pins-preset-null", "pins", {"pins": [{**PIN, "preset": None}]},
     "pins[0].preset: must be a string, got None")
_bad("pins-preset-int", "pins", {"pins": [{**PIN, "preset": 3}]}, "pins[0].preset: must be a string, got 3")
_bad("pins-preset-missing", "pins", {"pins": [{k: v for k, v in PIN.items() if k != "preset"}]},
     "pins[0].preset: is required")

# --- pins: bones and root ---
_bad("pins-bones-null", "pins", {"pins": [{**PIN, "bones": None}]},
     "pins[0].bones: must be an object, got None")
_bad("pins-bones-list", "pins", {"pins": [{**PIN, "bones": [[0, 0, 0, 1]]}]},
     "pins[0].bones: must be an object, got [[0, 0, 0, 1]]")
_bad("pins-nothing", "pins", {"pins": [{**PIN, "bones": {}}]}, "pins[0]: a pin needs 'bones' and/or 'root'")
_bad("pins-nothing-null-root", "pins", {"pins": [{**PIN, "bones": {}, "root": None}]},
     "pins[0]: a pin needs 'bones' and/or 'root'")
_bad("pins-quat-string-elem", "pins", {"pins": [{**PIN, "bones": {"X": [0, 0, 0, "1"]}}]},
     "pins[0].bones.X[3]: must be a finite number, got '1'")
_bad("pins-quat-bool-elem", "pins", {"pins": [{**PIN, "bones": {"X": [0, 0, 0, True]}}]},
     "pins[0].bones.X[3]: must be a finite number, got True")
_bad("pins-quat-null-elem", "pins", {"pins": [{**PIN, "bones": {"X": [None, 0, 0, 1]}}]},
     "pins[0].bones.X[0]: must be a finite number, got None")
_bad("pins-quat-nested", "pins", {"pins": [{**PIN, "bones": {"X": [[0], 0, 0, 1]}}]},
     "pins[0].bones.X[0]: must be a finite number, got [0]")
_bad("pins-quat-huge-int", "pins", {"pins": [{**PIN, "bones": {"X": [0, 0, 0, 10**400]}}]},
     "pins[0].bones.X[3]: must be a finite number, got " + repr(10**400)[:77] + "...")
_bad("pins-quat-three", "pins", {"pins": [{**PIN, "bones": {"X": [0, 0, 1]}}]},
     "pins[0].bones.X: must be 4 numbers [x, y, z, w], got [0.0, 0.0, 1.0]")
_bad("pins-quat-five", "pins", {"pins": [{**PIN, "bones": {"X": [0, 0, 0, 1, 0]}}]},
     "pins[0].bones.X: must be 4 numbers [x, y, z, w], got [0.0, 0.0, 0.0, 1.0, 0.0]")
_bad("pins-quat-string", "pins", {"pins": [{**PIN, "bones": {"X": "0001"}}]},
     "pins[0].bones.X: must be a list, got '0001'")
_bad("pins-quat-null", "pins", {"pins": [{**PIN, "bones": {"X": None}}]},
     "pins[0].bones.X: must be a list, got None")
_bad("pins-quat-not-unit", "pins", {"pins": [{**PIN, "bones": {"X": [0, 0, 0, 3]}}]},
     "pins[0].bones.X: is not a unit quaternion (length 3), got [0.0, 0.0, 0.0, 3.0]")
_bad("pins-quat-zero", "pins", {"pins": [{**PIN, "bones": {"A": [0, 0, 0, 1], "LeftArm": [0, 0, 0, 0]}}]},
     "pins[0].bones.LeftArm: is not a unit quaternion (length 0), got [0.0, 0.0, 0.0, 0.0]")
_bad("pins-second-pin-bad-bone", "pins",
     {"pins": [PIN, PIN, {**PIN, "bones": {"LeftArm": [0, "x", 0, 1]}}]},
     "pins[2].bones.LeftArm[1]: must be a finite number, got 'x'")
_bad("pins-root-two", "pins", {"pins": [{**PIN, "root": [0, 1]}]},
     "pins[0].root: must be 3 numbers [x, y, z], got [0.0, 1.0]")
_bad("pins-root-four", "pins", {"pins": [{**PIN, "root": [0, 1, 2, 3]}]},
     "pins[0].root: must be 3 numbers [x, y, z], got [0.0, 1.0, 2.0, 3.0]")
_bad("pins-root-string-elem", "pins", {"pins": [{**PIN, "root": [0, "1", 2]}]},
     "pins[0].root[1]: must be a finite number, got '1'")
_bad("pins-root-bool-elem", "pins", {"pins": [{**PIN, "root": [True, 1, 2]}]},
     "pins[0].root[0]: must be a finite number, got True")
_bad("pins-root-dict", "pins", {"pins": [{**PIN, "root": {"x": 1}}]},
     "pins[0].root: must be a list, got {'x': 1}")
_bad("pins-root-string", "pins", {"pins": [{**PIN, "root": "0,1,2"}]},
     "pins[0].root: must be a list, got '0,1,2'")

# --- ids: objects ---
_bad("ids-objects-missing", "ids", {"swaps": [{"start": 1}]}, "swaps[0].objects: is required")
_bad("ids-objects-null", "ids", {"swaps": [{**SWAP, "objects": None}]},
     "swaps[0].objects: must be two different tracked object ids (integers >= 1), got None")
_bad("ids-objects-same", "ids", {"swaps": [{**SWAP, "objects": [2, 2]}]},
     "swaps[0].objects: must be two different tracked object ids (integers >= 1), got [2, 2]")
_bad("ids-objects-zero", "ids", {"swaps": [{**SWAP, "objects": [0, 2]}]},
     "swaps[0].objects: must be two different tracked object ids (integers >= 1), got [0, 2]")
_bad("ids-objects-float", "ids", {"swaps": [{**SWAP, "objects": [1.0, 2]}]},
     "swaps[0].objects: must be two different tracked object ids (integers >= 1), got [1.0, 2]")
_bad("ids-objects-string", "ids", {"swaps": [{**SWAP, "objects": ["1", 2]}]},
     "swaps[0].objects: must be two different tracked object ids (integers >= 1), got ['1', 2]")
_bad("ids-objects-bool", "ids", {"swaps": [{**SWAP, "objects": [True, 2]}]},
     "swaps[0].objects: must be two different tracked object ids (integers >= 1), got [True, 2]")
_bad("ids-objects-three", "ids", {"swaps": [{**SWAP, "objects": [1, 2, 3]}]},
     "swaps[0].objects: must be two different tracked object ids (integers >= 1), got [1, 2, 3]")
_bad("ids-objects-one", "ids", {"swaps": [{**SWAP, "objects": [1]}]},
     "swaps[0].objects: must be two different tracked object ids (integers >= 1), got [1]")
_bad("ids-objects-string-whole", "ids", {"swaps": [{**SWAP, "objects": "12"}]},
     "swaps[0].objects: must be two different tracked object ids (integers >= 1), got '12'")

# --- round 5 fix round: radius >= 1, pin keys smooth reads, python-only values, check order ---
_bad("pins-radius-zero", "pins", {"pins": [{**PIN, "radius": 0}]},
     "pins[0].radius: must be an integer >= 1, got 0")
_bad("hands-role-missing", "hands", {"spans": [{k: v for k, v in HAND.items() if k != "role"}]},
     "spans[0].role: is required")
_bad("hands-start-negative", "hands", {"spans": [{**HAND, "start": -2}]},
     "spans[0].start: must be an integer >= 0, got -2")
_bad("pins-role-missing", "pins", {"pins": [{k: v for k, v in PIN.items() if k != "role"}]},
     "pins[0].role: is required")
_bad("pins-root-rotation-not-unit", "pins", {"pins": [{**PIN, "root_rotation": [0, 0, 0, 3]}]},
     "pins[0].root_rotation: is not a unit quaternion (length 3), got [0.0, 0.0, 0.0, 3.0]")
_bad("pins-root-rotation-string-elem", "pins", {"pins": [{**PIN, "root_rotation": [0, 0, 0, "1"]}]},
     "pins[0].root_rotation[3]: must be a finite number, got '1'")
_bad("pins-root-rotation-three", "pins", {"pins": [{**PIN, "root_rotation": [0, 0, 1]}]},
     "pins[0].root_rotation: must be 4 numbers [x, y, z, w], got [0.0, 0.0, 1.0]")
_bad("pins-root-rotation-string", "pins", {"pins": [{**PIN, "root_rotation": "0001"}]},
     "pins[0].root_rotation: must be a list, got '0001'")
_bad("pins-prefix-int", "pins", {"pins": [{**PIN, "prefix": 3}]}, "pins[0].prefix: must be a string, got 3")
_bad("pins-prefix-null", "pins", {"pins": [{**PIN, "prefix": None}]},
     "pins[0].prefix: must be a string, got None")
_bad("pins-root-motion-bool", "pins", {"pins": [{**PIN, "root_motion": True}]},
     "pins[0].root_motion: must be a string, got True")
_bad("pins-root-motion-null", "pins", {"pins": [{**PIN, "root_motion": None}]},
     "pins[0].root_motion: must be a string, got None")
# not JSON: keys that are not strings, huge integers, numbers JSON has no type for, other mappings
_bad("contacts-int-key-top", "contacts", {"spans": [], 1: "x"}, "keys must be strings, got 1")
_bad("pins-none-key-nested", "pins", {"pins": [{**PIN, None: 1}]}, "pins[0]: keys must be strings, got None")
_bad("pins-tuple-key", "pins", {"pins": [PIN], ("a",): 1}, "keys must be strings, got ('a',)")
_bad("pins-bone-int-key", "pins", {"pins": [{**PIN, "bones": {1: [0, 0, 0, 1]}}]},
     "pins[0].bones: keys must be strings, got 1")
_bad("contacts-huge-int-unknown", "contacts", {"spans": [], "note": 10**5000}, "note: integer is too large")
_bad("contacts-huge-int-start", "contacts", {"spans": [{**SPAN, "start": 10**5000}]},
     "spans[0].start: integer is too large")
_bad("ids-huge-negative-end", "ids", {"swaps": [{**SWAP, "end": -(10**5000)}]},
     "swaps[0].end: integer is too large")
_bad("contacts-decimal-nan", "contacts", {"spans": [], "note": Decimal("NaN")},
     "note: not a JSON value, got Decimal('NaN')")
_bad("pins-decimal-quat", "pins", {"pins": [{**PIN, "bones": {"X": [0, 0, 0, Decimal(1)]}}]},
     "pins[0].bones.X[3]: not a JSON value, got Decimal('1')")
_bad("hands-complex-nan", "hands", {"spans": [], "note": complex(NAN, 0)},
     "note: not a JSON value, got (nan+0j)")
_bad("ids-fraction", "ids", {"swaps": [{**SWAP, "start": Fraction(1, 1)}]},
     "swaps[0].start: not a JSON value, got Fraction(1, 1)")
_bad("contacts-set-unknown", "contacts", {"spans": [], "note": {1, 2}}, "note: not a JSON value, got {1, 2}")
_bad("contacts-bytes-unknown", "contacts", {"spans": [], "note": b"x"}, "note: not a JSON value, got b'x'")
_bad("contacts-mappingproxy-item", "contacts", {"spans": [MappingProxyType(SPAN)]},
     "spans[0]: not a JSON value, got " + _cut(repr(MappingProxyType(SPAN))))
_bad("hands-userdict-item", "hands", {"spans": [UserDict(HAND)]},
     "spans[0]: not a JSON value, got " + _cut(repr(UserDict(HAND))))
# long values are cut in the order messages too
_bad("contacts-end-huge-negative", "contacts", {"spans": [{**SPAN, "end": -(10**400)}]},
     "spans[0]: end (" + repr(-(10**400))[:77] + "...) must be after start (1) (end is exclusive)")
_bad("hands-start-huge", "hands", {"spans": [{**HAND, "start": 10**400, "end": 3}]},
     "spans[0]: end (3) must be after start (" + repr(10**400)[:77] + "...) (end is exclusive)")
_bad("ids-end-huge-negative", "ids", {"swaps": [{**SWAP, "end": -(10**400)}]},
     "swaps[0]: end (" + repr(-(10**400))[:77] + "...) must be after start (1) (end is exclusive)")
# the FIRST error is the same on both sides: fields in the order of the model, then the checks that
# need valid fields (end after start; one of foot/channels; a pin needs something)
_bad("hands-order-after-side", "hands", {"spans": [{"role": "A", "start": 1, "end": 1, "side": 7}]},
     "spans[0].side: must be 'left', 'right' or 'both', got 7")
_bad("contacts-order-after-state", "contacts", {"spans": [{**SPAN, "start": 3, "end": 1, "state": "hover"}]},
     "spans[0].state: must be 'plant' or 'free', got 'hover'")
_bad("contacts-order-before-target", "contacts",
     {"spans": [{k: v for k, v in SPAN.items() if k != "foot"} | {"start": 4, "end": 4}]},
     "spans[0]: end (4) must be after start (4) (end is exclusive)")
_multi("pins-unit-before-later-bone-type", "pins",
     {"pins": [{**PIN, "bones": {"A": [0, 0, 0, 3], "B": [0, "x", 0, 1]}}]},
     "pins[0].bones.A: is not a unit quaternion (length 3), got [0.0, 0.0, 0.0, 3.0]")
_multi("pins-bone-type-before-root", "pins",
     {"pins": [{**PIN, "bones": {"A": [0, "x", 0, 1]}, "root": [0, 0]}]},
     "pins[0].bones.A[1]: must be a finite number, got 'x'")
_bad("pins-root-before-nothing-check", "pins", {"pins": [{**PIN, "bones": {}, "root": [0, 0]}]},
     "pins[0].root: must be 3 numbers [x, y, z], got [0.0, 0.0]")
_multi("pins-first-bad-item-wins", "pins",
     {"pins": [{**PIN, "bones": {}}, {**PIN, "role": "Z"}]}, "pins[0]: a pin needs 'bones' and/or 'root'")
_bad("ids-order-after-objects", "ids", {"swaps": [{"start": 5, "end": 2, "objects": [1, 1]}]},
     "swaps[0].objects: must be two different tracked object ids (integers >= 1), got [1, 1]")

VALID += [
    ("pins-radius-1", "pins", {"pins": [{**PIN, "radius": 1}]}),
    ("pins-root-rotation-null", "pins", {"pins": [{**PIN, "root_rotation": None}]}),
    ("pins-root-rotation-quat", "pins", {"pins": [{**PIN, "root_rotation": [0.0, 0.0, 0.0, 1.0]}]}),
    ("pins-prefix-text", "pins", {"pins": [{**PIN, "prefix": "mixamorig:", "root_motion": "extracted"}]}),
    ("contacts-tuple-unknown", "contacts", {"spans": [SPAN], "note": (1, [2.5, None], "s")}),
    ("contacts-huge-int-ok", "contacts", {"spans": [], "note": 10**1200}),
    ("contacts-ordereddict-item", "contacts", {"spans": [OrderedDict(SPAN)]}),
]

# fighters.json: validated by afv core only (the add-on refuses to write it), same strictness
F_AB = {"fighters": [1, 0], "extra": [2], "prompt_frame": 0,
        "boxes": {"1": [135.6, 101.3, 368.1, 668.7], "0": [878.3, 15.1, 1067.6, 714.3]},
        "shot": {"start": 0.0, "duration": 3.036366, "source_sha256": "ab"}}
F_OBJ = {"objects": [{"id": 1, "role": "A"}, {"id": 2, "role": "B", "label": "red"}],
         "prompts": [{"frame": 0, "obj": 1, "box": [0, 0, 10, 10], "points": [[5, 5, 1], [1, 1, 0]]},
                     {"frame": 0, "obj": 2, "box": [20, 0, 30, 10]}]}
FIGHTERS_VALID: list[tuple[str, object]] = [
    ("fighters-detections", F_AB),
    ("fighters-detections-min", {"fighters": [0, 1]}),
    ("fighters-objects-prompts", F_OBJ),
    ("fighters-null-shot", {**F_AB, "shot": None}),
    ("fighters-int-box", {**F_AB, "boxes": {"1": [0, 0, 5, 5], "0": [6, 0, 9, 5]}}),
]
FIGHTERS_INVALID: list[tuple[str, object, str]] = []


def _fbad(vid: str, data: object, error: str) -> None:
    FIGHTERS_INVALID.append((vid, data, error))


_fbad("fighters-list", [], "must be a JSON object, got []")
_fbad("fighters-nan-box", {**F_AB, "boxes": {"1": [0, 0, NAN, 5]}},
      "boxes.1[2]: NaN or infinity is not allowed")
_fbad("fighters-str-picks", {**F_AB, "fighters": ["1", "0"]},
      "fighters[0]: must be an integer, got '1'; fighters[1]: must be an integer, got '0'")
_fbad("fighters-bool-picks", {**F_AB, "fighters": [True, False]},
      "fighters[0]: must be an integer, got True; fighters[1]: must be an integer, got False")
_fbad("fighters-float-pick", {**F_AB, "fighters": [1.0, 0]}, "fighters[0]: must be an integer, got 1.0")
_fbad("fighters-extra-str", {**F_AB, "extra": ["2"]}, "extra[0]: must be an integer, got '2'")
_fbad("fighters-one-pick", {**F_AB, "fighters": [1]},
      "fighters needs two detection numbers (A, B), got [1]")
_fbad("fighters-prompt-frame-float", {**F_AB, "prompt_frame": 0.0},
      "prompt_frame: must be an integer >= 0, got 0.0")
_fbad("fighters-prompt-frame-str", {**F_AB, "prompt_frame": "0"},
      "prompt_frame: must be an integer >= 0, got '0'")
_fbad("fighters-prompt-frame-bool", {**F_AB, "prompt_frame": True},
      "prompt_frame: must be an integer >= 0, got True")
_fbad("fighters-prompt-frame-negative", {**F_AB, "prompt_frame": -1},
      "prompt_frame: must be an integer >= 0, got -1")
_fbad("fighters-box-str", {**F_AB, "boxes": {"1": ["135.6", 101.3, 368.1, 668.7]}},
      "boxes.1[0]: must be a finite number, got '135.6'")
_fbad("fighters-box-bool", {**F_AB, "boxes": {"1": [True, 1, 5, 5]}},
      "boxes.1[0]: must be a finite number, got True")
_fbad("fighters-box-three", {**F_AB, "boxes": {"1": [0, 0, 5]}},
      "boxes: box must be [x0, y0, x1, y1] with x1 > x0 and y1 > y0, got [0.0, 0.0, 5.0]")
_fbad("fighters-box-inverted", {**F_AB, "boxes": {"1": [5, 0, 1, 5]}},
      "boxes: box must be [x0, y0, x1, y1] with x1 > x0 and y1 > y0, got [5.0, 0.0, 1.0, 5.0]")
_fbad("fighters-boxes-list", {**F_AB, "boxes": []}, "boxes: must be an object, got []")
_fbad("fighters-shot-start-str", {**F_AB, "shot": {"start": "0", "duration": 1.0}},
      "shot.start: must be a finite number, got '0'")
_fbad("fighters-shot-start-bool", {**F_AB, "shot": {"start": True, "duration": 1.0}},
      "shot.start: must be a finite number, got True")
_fbad("fighters-shot-missing-duration", {**F_AB, "shot": {"start": 0.0}}, "shot.duration: is required")
_fbad("fighters-shot-sha-int", {**F_AB, "shot": {"start": 0.0, "duration": 1.0, "source_sha256": 5}},
      "shot.source_sha256: must be a string, got 5")
_fbad("fighters-shot-list", {**F_AB, "shot": [0, 1]}, "shot: must be an object, got [0, 1]")
_fbad("fighters-version-str", {**F_AB, "version": "1"}, "version: must be 1, got '1'")
_fbad("fighters-version-true", {**F_AB, "version": True}, "version: must be 1, got True")
_fbad("fighters-version-two", {**F_AB, "version": 2}, "version: must be 1, got 2")
_fbad("fighters-unknown-key", {**F_AB, "note": 1}, "note: Extra inputs are not permitted, got 1")
_fbad("fighters-object-id-str", {"objects": [{"id": "1", "role": "A"}, F_OBJ["objects"][1]],
                                 "prompts": F_OBJ["prompts"]},
      "objects[0].id: must be an integer >= 1, got '1'")
_fbad("fighters-object-id-zero", {"objects": [{"id": 0, "role": "A"}, F_OBJ["objects"][1]],
                                  "prompts": F_OBJ["prompts"]},
      "objects[0].id: must be an integer >= 1, got 0")
_fbad("fighters-object-role-c", {"objects": [{"id": 1, "role": "C"}, F_OBJ["objects"][1]],
                                 "prompts": F_OBJ["prompts"]},
      "objects[0].role: must be 'A' or 'B', got 'C'")
_fbad("fighters-object-label-int", {"objects": [{"id": 1, "role": "A", "label": 3}, F_OBJ["objects"][1]],
                                    "prompts": F_OBJ["prompts"]},
      "objects[0].label: must be a string, got 3")
_fbad("fighters-prompt-frame-obj-float", {"objects": F_OBJ["objects"], "prompts": [
    {"frame": 0.0, "obj": 1, "box": [0, 0, 10, 10]}, F_OBJ["prompts"][1]]},
      "prompts[0].frame: must be an integer >= 0, got 0.0")
_fbad("fighters-prompt-obj-bool", {"objects": F_OBJ["objects"], "prompts": [
    {"frame": 0, "obj": True, "box": [0, 0, 10, 10]}, F_OBJ["prompts"][1]]},
      "prompts[0].obj: must be an integer >= 1, got True")
_fbad("fighters-prompt-detection-str", {"objects": F_OBJ["objects"], "prompts": [
    {"frame": 0, "obj": 1, "detection": "3"}, F_OBJ["prompts"][1]]},
      "prompts[0].detection: must be an integer >= 0, got '3'")
_fbad("fighters-prompt-point-str", {"objects": F_OBJ["objects"], "prompts": [
    {"frame": 0, "obj": 1, "box": [0, 0, 10, 10], "points": [[5, "5", 1]]}, F_OBJ["prompts"][1]]},
      "prompts[0].points[0][1]: must be a finite number, got '5'")
_fbad("fighters-prompt-point-flag", {"objects": F_OBJ["objects"], "prompts": [
    {"frame": 0, "obj": 1, "box": [0, 0, 10, 10], "points": [[5, 5, 2]]}, F_OBJ["prompts"][1]]},
      "prompts[0].points: points are [x, y, 1|0], got [5.0, 5.0, 2.0]")
_fbad("fighters-prompt-item-string", {"objects": F_OBJ["objects"], "prompts": ["x"]},
      "prompts[0]: must be an object, got 'x'")
_fbad("fighters-prompt-needs-target", {"objects": F_OBJ["objects"], "prompts": [
    {"frame": 0, "obj": 1}, F_OBJ["prompts"][1]]},
      "prompts[0]: prompt for object 1 at frame 0 needs a box or a detection (points only refine a box)")
_fbad("fighters-mappingproxy", {**F_AB, "shot": MappingProxyType(F_AB["shot"])},
      "shot: not a JSON value, got " + _cut(repr(MappingProxyType(F_AB["shot"]))))
