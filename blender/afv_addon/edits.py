"""Edit files of a shot (``shots/<id>/edits/<name>.json``) as the cleanup tools change them.

numpy only (no bpy, no afv), so it also runs in plain Python for tests. Formats (PM contract,
``docs/BUILD_PLAN.md`` "Shared formats"): frames are shot-timeline frame numbers, ``end`` exclusive,
``role`` "A"/"B"; every change keeps keys it does not know (top level and per entry).

- ``contacts.json``: ``{"spans": [{"role", "start", "end", "state": "plant"|"free", "foot"}]}``
- ``pins.json``: ``{"pins": [{"role", "frame", "radius", "preset", "root_motion", "prefix",
  "bones": {canonical bone: [x, y, z, w]}, "root": [x, y, z] | null}]}``: export-rig local rotations
  (afv: Y-up, xyzw; the take's ``rot``) of the take's rig (``root_motion``/``prefix`` as shown)
- ``hands.json``: ``{"spans": [{"role", "start", "end", "side": "left"|"right"|"both"}]}``
- ``ids.json``: ``{"swaps": [{"start", "end": f | null, "objects": [o1, o2]}]}``

The functions here return a new dict (the input is not changed). ``write_local`` is the one-shot
fallback writer (``afv serve``'s ``edit`` command validates and writes when the worker runs); it
checks the same structure as ``afv.core.edits`` (``validate``, mirrored here: Blender has no pydantic)
and, like it, keeps top-level keys of the existing file that the new data leaves out.

The history (``shots/<id>/.addon/history.json``) remembers what each cleanup fix changed (the edit
file's content, or the shot-level smoothing settings, before it; a hash of the state after it) so
"Revert" can put it back: Blender's undo does not touch the shot's files.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np

EDITS_DIR = "edits"
NAMES = ("contacts", "pins", "hands", "ids")
DEFAULT_PIN_RADIUS = 8
SIDES = ("left", "right")
_LEG = re.compile(r"(upleg|leg|thigh|calf|shin|knee|foot|ankle|toe|heel)", re.IGNORECASE)
_ARM = re.compile(r"(shoulder|arm|elbow|hand|wrist|thumb|index|middle|ring|pinky|finger)", re.IGNORECASE)


class EditError(Exception):
    """An edit cannot be made or read; the message is fit to show the user."""


# --- files ------------------------------------------------------------------------------------------
def edit_path(shot_dir: str | Path, name: str) -> Path:
    if name not in NAMES and not re.fullmatch(r"[a-z_]+", name):
        raise EditError(f"bad edit name {name!r}")
    return Path(shot_dir) / EDITS_DIR / f"{name}.json"


def read(shot_dir: str | Path, name: str) -> dict:
    """The edit as stored ({} when there is none)."""
    p = edit_path(shot_dir, name)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as e:
        raise EditError(f"cannot read {p}: {e}") from e
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise EditError(f"{p} is not valid JSON ({e}); fix or delete it") from e
    if not isinstance(data, dict):
        raise EditError(f"{p}: expected a JSON object")
    return data


def write_local(shot_dir: str | Path, name: str, data: dict | None) -> Path:
    """Validate and write the edit atomically (temp file in the same folder, then replace); top-level
    keys of the existing file that ``data`` leaves out are kept; ``None`` removes the file."""
    p = edit_path(shot_dir, name)
    if data is None:
        p.unlink(missing_ok=True)
        return p
    old = read(shot_dir, name) if p.is_file() else {}
    data = {**old, **data}
    try:
        validate(name, data)
    except EditError as e:
        if old:  # the data was merged from the file on disk: say so when that is invalid too
            try:
                validate(name, old)
            except EditError:
                raise EditError(
                    f"{e} (the file on disk is invalid too: {fix_hint(name, old)})") from None
        raise
    return _write_json(p, data)


def _write_json(p: Path, data) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.stem}.", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, indent=1)
            fh.write("\n")
        for attempt in range(20):  # Windows refuses while another process still has the file open
            try:
                os.replace(tmp, p)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def _list(data: dict, key: str) -> list:
    v = data.get(key, [])
    if not isinstance(v, list):
        raise EditError(f"'{key}' must be a list")
    return v


# --- bones ------------------------------------------------------------------------------------------
def canonical(bone: str, prefix: str = "") -> str:
    """The canonical export bone name (pins.json): the rig's prefix taken off."""
    return bone[len(prefix):] if prefix and bone.startswith(prefix) else bone


def _plain(bone: str, prefix: str = "") -> str:
    """``mixamorig:LeftFoot`` / ``<prefix>LeftFoot`` -> ``LeftFoot``."""
    n = bone[len(prefix):] if prefix and bone.startswith(prefix) else bone
    return n.rsplit(":", 1)[-1]


def bone_side(bone: str, prefix: str = "") -> str | None:
    n = _plain(bone, prefix).lower()
    if n.startswith("left") or n.endswith((".l", "_l")):
        return "left"
    if n.startswith("right") or n.endswith((".r", "_r")):
        return "right"
    return None


def foot_side(bone: str, prefix: str = "") -> str | None:
    """The leg a bone belongs to (thigh down to the toes), else None."""
    return bone_side(bone, prefix) if _LEG.search(_plain(bone, prefix)) else None


def hand_side(bone: str, prefix: str = "") -> str | None:
    """The arm a bone belongs to (shoulder down to the fingers), else None."""
    n = _plain(bone, prefix)
    return bone_side(bone, prefix) if _ARM.search(n) and not _LEG.search(n) else None


# --- spans (contacts, hands) ------------------------------------------------------------------------
def _strict_int(v, what: str, lo: int | None = None) -> int:
    """``v`` as given when it is an integer (no bool, float or string: nothing is coerced)."""
    try:
        return _as_int(v, lo)
    except _Bad as e:
        raise EditError(f"{what} {e.args[0]}") from None


def _check_range(start: int, end: int) -> tuple[int, int]:
    start, end = _strict_int(start, "start frame"), _strict_int(end, "end frame")
    if end <= start:
        raise EditError(f"empty frame range {_shown(start)}:{_shown(end)} (end is exclusive)")
    return start, end


def _cut(span: dict, start: int, end: int) -> list[dict]:
    """The parts of ``span`` outside [start, end) (its other keys kept)."""
    out = []
    s0, s1 = int(span["start"]), int(span["end"])
    if s0 < start:
        out.append({**span, "start": s0, "end": min(s1, start)})
    if s1 > end:
        out.append({**span, "start": max(s0, end), "end": s1})
    return out


def _overlaps(span: dict, start: int, end: int) -> bool:
    return int(span["start"]) < end and int(span["end"]) > start


def paint_contact(data: dict, role: str, foot: str, start: int, end: int, state: str) -> dict:
    """Mark ``foot`` of ``role`` as ``state`` ("plant", "free"; "clear" = back to the solver) on
    [start, end). Earlier painted spans of that foot are cut where they overlap, so the file never
    holds two answers for one frame; spans by ``channels`` are left as they are."""
    if foot not in SIDES:
        raise EditError(f"foot must be left or right, got {foot!r}")
    if state not in ("plant", "free", "clear"):
        raise EditError(f"state must be plant, free or clear, got {state!r}")
    start, end = _check_range(start, end)
    validate_stored("contacts", data)  # an existing file with lax values is refused, not silently rewritten
    out = copy.deepcopy(data)
    spans = []
    for s in _list(out, "spans"):
        mine = s.get("role") == role and s.get("foot") == foot and "channels" not in s
        if mine and _overlaps(s, start, end):
            spans += _cut(s, start, end)
        else:
            spans.append(s)
    if state != "clear":
        spans.append({"role": role, "start": start, "end": end, "state": state, "foot": foot})
    out["spans"] = spans
    return out


def _sides(side: str) -> set[str]:
    if side == "both":
        return set(SIDES)
    if side in SIDES:
        return {side}
    raise EditError(f"side must be left, right or both, got {side!r}")


def mark_hands(data: dict, role: str, side: str, start: int, end: int, off: bool = True) -> dict:
    """Hands off (``off``) or back on (``off=False``: remove) for ``side`` of ``role`` on [start, end).
    Overlapping spans of that hand are cut; a "both" span keeps the other hand where it overlaps."""
    start, end = _check_range(start, end)
    want = _sides(side)
    validate_stored("hands", data)
    out = copy.deepcopy(data)
    spans = []
    for s in _list(out, "spans"):
        if s.get("role") != role or not _overlaps(s, start, end):
            spans.append(s)
            continue
        have = _sides(str(s.get("side", "both")))
        if not have & want:
            spans.append(s)
            continue
        spans += _cut(s, start, end)
        rest = have - want
        if rest:  # e.g. "both" minus "left": the right hand stays off inside the range
            (other,) = rest
            spans.append({**s, "start": max(int(s["start"]), start), "end": min(int(s["end"]), end),
                          "side": other})
    if off:
        spans.append({"role": role, "start": start, "end": end, "side": side})
    out["spans"] = sorted(spans, key=lambda s: (str(s.get("role")), int(s["start"])))
    return out


def spans_for(data: dict, role: str | None = None) -> list[dict]:
    return [s for s in data.get("spans", []) or [] if role is None or s.get("role") == role]


# --- pins -------------------------------------------------------------------------------------------
def add_pin(data: dict, pin: dict) -> dict:
    """Add ``pin``. A pin of the same fighter at the same frame is merged: its bones stay unless
    ``pin`` sets them again (new values win), its root stays unless ``pin`` has one."""
    for k in ("role", "frame", "bones"):
        if k not in pin:
            raise EditError(f"a pin needs '{k}'")
    if not pin["bones"] and pin.get("root") is None:
        raise EditError("a pin needs at least one bone")
    _strict_int(pin["frame"], "pin frame", 0)
    validate_stored("pins", data)
    out = copy.deepcopy(data)
    pins, merged = [], dict(pin)
    for p in _list(out, "pins"):
        if p.get("role") == pin["role"] and int(p.get("frame", -1)) == int(pin["frame"]):
            merged = {**p, **pin, "bones": {**(p.get("bones") or {}), **pin["bones"]},
                      "root": pin.get("root") if pin.get("root") is not None else p.get("root")}
        else:
            pins.append(p)
    pins.append(merged)
    out["pins"] = sorted(pins, key=lambda p: (str(p.get("role")), int(p.get("frame", 0))))
    return out


def delete_pin(data: dict, index: int) -> tuple[dict, dict]:
    """(new data, the removed pin)."""
    out = copy.deepcopy(data)
    pins = _list(out, "pins")
    if not 0 <= index < len(pins):
        raise EditError(f"no pin {index} (there are {len(pins)})")
    gone = pins.pop(index)
    out["pins"] = pins
    return out, gone


def pin_range(pin: dict, frame_start: int, frame_end: int) -> tuple[int, int]:
    """The frames [a, b) a pin's re-solve touches (its radius, clipped to the take)."""
    f, r = int(pin["frame"]), int(pin.get("radius", DEFAULT_PIN_RADIUS))
    return max(frame_start, f - r), min(frame_end + 1, f + r + 1)


def _qmul(a, b):
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def basis_to_local(rest_q, basis_q) -> np.ndarray:
    """Blender pose basis (wxyz, in the bone's rest frame) -> the take's local rotation ``rot``
    (xyzw, afv armature frame): local = R basis R^-1 with R the bone's rest rotation
    (``bone.matrix_local``), the inverse of the builder's basis = R^-1 local R."""
    r = np.asarray(rest_q, dtype=np.float64)
    r_inv = r * np.array([1.0, -1.0, -1.0, -1.0])
    q = _qmul(_qmul(r, basis_q), r_inv)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., :1] < 0.0, -q, q)  # w >= 0: one of the two equal quaternions
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


# --- ids --------------------------------------------------------------------------------------------
def add_swap(data: dict, start: int, objects: tuple[int, int] | list[int], end: int | None = None) -> dict:
    o = [_strict_int(v, "swap object", 1) for v in objects]
    if len(o) != 2 or o[0] == o[1]:
        raise EditError(f"a swap needs two different objects, got {_shown(objects)}")
    start = _strict_int(start, "swap start frame", 0)
    if end is not None and _strict_int(end, "swap end frame") <= start:
        raise EditError(f"empty swap range {_shown(start)}:{_shown(end)}")
    validate_stored("ids", data)
    out = copy.deepcopy(data)
    swaps = _list(out, "swaps")
    swaps.append({"start": start, "end": end, "objects": o})
    out["swaps"] = swaps
    return out


def delete_swap(data: dict, index: int) -> tuple[dict, dict]:
    out = copy.deepcopy(data)
    swaps = _list(out, "swaps")
    if not 0 <= index < len(swaps):
        raise EditError(f"no swap {index} (there are {len(swaps)})")
    gone = swaps.pop(index)
    out["swaps"] = swaps
    return out, gone


# --- validation (mirrors afv.core.edits, which needs pydantic) --------------------------------------
# Strict, no coercion; the same rules as ``afv.core.edits``:
#   integer  a JSON integer only: not a bool, not 3.0, not "3"
#   number   an int or a float, finite, not a bool, not "0.5" (an int is fine: JSON has one number type)
#   string   a str
#   null     only where a field says so (pin root and root_rotation, swap end, a contact span's foot or
#            channels)
# Unknown keys are kept and not checked, except for ``_scan`` (anywhere in the document): NaN/infinity,
# keys that are not strings, integers over 4096 bits, and values JSON cannot hold are refused.
# Equal on both sides (``tests/core/edit_vectors.py`` and a differential test): which documents are
# accepted, and the FIRST error message, ``edits/<name>.json: <path>: <what is wrong>, got <value>``
# with the path like ``pins[2].bones.LeftArm[1]``. Core lists up to three errors, this side stops at
# the first; the fields are checked in the order of core's models (item by item, field by field, then
# the checks that need valid fields: end after start, one of foot/channels, a pin needs something).
MAX_BITS = 4096


def _shown(v) -> str:
    try:
        text = repr(v)
    except ValueError:  # an integer past Python's digit limit
        return "<value too large to show>"
    return text if len(text) <= 80 else text[:77] + "..."


class _Bad(Exception):
    """A value is wrong; ``args[0]`` is the part of the message after the field path."""


class _Located(Exception):
    def __init__(self, where: str, msg: str):
        super().__init__(where, msg)
        self.where, self.msg = where, msg


def _fail(msg: str, v) -> _Bad:
    return _Bad(f"{msg}, got {_shown(v)}")


def _as_int(v, lo=None) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or (lo is not None and v < lo):
        raise _fail("must be an integer" + (f" >= {lo}" if lo is not None else ""), v)
    return v


def _as_float(v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _fail("must be a finite number", v)
    try:
        f = float(v)
    except OverflowError:
        raise _fail("must be a finite number", v) from None
    if not math.isfinite(f):
        raise _fail("must be a finite number", v)
    return f


def _one_of(v, *options: str) -> str:
    if not isinstance(v, str) or v not in options:
        shown = ", ".join(repr(o) for o in options[:-1]) + f" or {options[-1]!r}"
        raise _fail(f"must be {shown}", v)
    return v


def _field(d: dict, key: str, path: str, check, *, default=None, required=True):
    """``check(d[key])`` with errors at ``path.key``; a missing key is an error unless it has a default."""
    where = f"{path}.{key}" if path else key
    if key not in d:
        if required:
            raise _Located(where, "is required")
        return default
    try:
        return check(d[key])
    except _Bad as e:
        raise _Located(where, e.args[0]) from None


def _at(path: str, text: str) -> str:
    return f"{path}: {text}" if path else text


def _scan(v, path: str = "") -> str | None:
    """The first problem anywhere in ``v`` that is not about a field (None when there is none)."""
    if v is None or isinstance(v, (bool, str)):
        return None
    if isinstance(v, int):
        return _at(path, "integer is too large") if v.bit_length() > MAX_BITS else None
    if isinstance(v, float):
        return None if math.isfinite(v) else _at(path, "NaN or infinity is not allowed")
    if isinstance(v, dict):
        for k, x in v.items():
            if not isinstance(k, str):
                return _at(path, f"keys must be strings, got {_shown(k)}")
            hit = _scan(x, f"{path}.{k}" if path else k)
            if hit is not None:
                return hit
        return None
    if isinstance(v, (list, tuple)):
        for i, x in enumerate(v):
            hit = _scan(x, f"{path}[{i}]")
            if hit is not None:
                return hit
        return None
    return _at(path, f"not a JSON value, got {_shown(v)}")


def _vector(v, n: int, label: str, where: str, unit: bool = False) -> list[float]:
    """``n`` finite numbers (a unit quaternion when ``unit``) at ``where``."""
    if not isinstance(v, list):
        raise _Located(where, str(_fail("must be a list", v)))
    vals = []
    for i, x in enumerate(v):
        try:
            vals.append(_as_float(x))
        except _Bad as e:
            raise _Located(f"{where}[{i}]", e.args[0]) from None
    if len(vals) != n:
        raise _Located(where, str(_fail(f"must be {n} numbers {label}", vals)))
    if unit:
        length = math.sqrt(sum(x * x for x in vals))
        if not 0.5 < length < 1.5:
            raise _Located(where, f"is not a unit quaternion (length {length:.3g}), got {_shown(vals)}")
    return vals


def _items(data: dict, key: str, check) -> None:
    v = data.get(key, [])
    if not isinstance(v, list):
        raise _Located(key, str(_fail("must be a list", v)))
    for i, item in enumerate(v):
        where = f"{key}[{i}]"
        if not isinstance(item, dict):
            raise _Located(where, str(_fail("must be an object", item)))
        check(item, where)


def _span_fields(s: dict, where: str) -> tuple[int, int]:
    _field(s, "role", where, lambda v: _one_of(v, "A", "B"))
    a = _field(s, "start", where, lambda v: _as_int(v, 0))
    b = _field(s, "end", where, _as_int)
    return a, b


def _order(a: int, b: int, where: str) -> None:
    if b <= a:
        raise _Located(where, f"end ({_shown(b)}) must be after start ({_shown(a)}) (end is exclusive)")


def _channels(v):
    if not isinstance(v, list) or not v or not all(isinstance(c, str) for c in v):
        raise _fail("must be a non-empty list of channel names", v)
    return v


def _contact(s: dict, where: str) -> None:
    a, b = _span_fields(s, where)
    _field(s, "state", where, lambda v: _one_of(v, "plant", "free"))
    foot = _field(s, "foot", where, lambda v: v if v is None else _one_of(v, "left", "right"),
                  required=False)
    ch = _field(s, "channels", where, lambda v: v if v is None else _channels(v), required=False)
    _order(a, b, where)
    if (foot is None) == (ch is None):
        raise _Located(where, "a contact span needs exactly one of 'foot' or 'channels'")


def _hand(s: dict, where: str) -> None:
    a, b = _span_fields(s, where)
    _field(s, "side", where, lambda v: _one_of(v, "left", "right", "both"))
    _order(a, b, where)


def _text(v):
    if not isinstance(v, str):
        raise _fail("must be a string", v)
    return v


def _pin(p: dict, where: str) -> None:
    _field(p, "role", where, lambda v: _one_of(v, "A", "B"))
    _field(p, "frame", where, lambda v: _as_int(v, 0))
    _field(p, "radius", where, lambda v: _as_int(v, 1), required=False)
    _field(p, "preset", where, _text)
    bones = p.get("bones", {})
    if not isinstance(bones, dict):
        raise _Located(f"{where}.bones", str(_fail("must be an object", bones)))
    for b, q in bones.items():
        _vector(q, 4, "[x, y, z, w]", f"{where}.bones.{b}", unit=True)
    root = p.get("root")
    if root is not None:
        _vector(root, 3, "[x, y, z]", f"{where}.root")
    if p.get("root_rotation") is not None:
        _vector(p["root_rotation"], 4, "[x, y, z, w]", f"{where}.root_rotation", unit=True)
    _field(p, "prefix", where, _text, required=False)
    _field(p, "root_motion", where, _text, required=False)
    if not bones and root is None:
        raise _Located(where, "a pin needs 'bones' and/or 'root'")


def _objects(v):
    ok = (isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool)
                                                       for x in v) and v[0] != v[1] and min(v) >= 1)
    if not ok:
        raise _fail("must be two different tracked object ids (integers >= 1)", v)


def _swap(sw: dict, where: str) -> None:
    a = _field(sw, "start", where, lambda v: _as_int(v, 0))
    end = _field(sw, "end", where, lambda v: v if v is None else _as_int(v), required=False)
    _field(sw, "objects", where, _objects)
    if end is not None:
        _order(a, end, where)


_CHECKS = {"contacts": ("spans", _contact), "hands": ("spans", _hand), "pins": ("pins", _pin),
           "ids": ("swaps", _swap)}


def validate(name: str, data) -> None:
    """Raise EditError when ``data`` does not have the structure of ``edits/<name>.json`` (the same
    rules as ``afv.core.edits.validate_edit``). ``fighters`` is written by ``afv pick`` only."""
    where = f"edits/{name}.json"
    if name == "fighters":
        raise EditError(f"{where} is written by 'afv pick' (the add-on does not write it)")
    if name not in NAMES:
        raise EditError(f"unknown edit {name!r}")
    if not isinstance(data, dict):
        raise EditError(f"{where}: must be a JSON object, got {_shown(data)}")
    hit = _scan(data)
    if hit is not None:
        raise EditError(f"{where}: {hit}")
    key, check = _CHECKS[name]
    try:
        _items(data, key, check)
    except _Located as e:
        raise EditError(f"{where}: {e.where}: {e.msg}") from None


def fix_hint(name: str, data) -> str:
    """What to do about an invalid ``edits/<name>.json`` whose content is ``data``."""
    if not isinstance(data, dict):
        return f"fix or delete the file edits/{name}.json"
    return f"fix or delete that entry in edits/{name}.json"


def validate_stored(name: str, data) -> None:
    """``validate`` for a file already on disk: the message ends with what to do."""
    try:
        validate(name, data)
    except EditError as e:
        raise EditError(f"{e} ({fix_hint(name, data)})") from None


# --- history (Revert last fix) ----------------------------------------------------------------------
# shots/<id>/.addon/history.json (outside edits/: not an edit, not in the provenance). Each fix keeps
# the state before it (an edit file's content, or the shot-level smoothing settings) and only a hash
# of the state after it, which a revert checks.
HISTORY_DIR = ".addon"
HISTORY_FILE = "history.json"
OLD_HISTORY = ".addon_history.json"  # round 4 first pass: edits/.addon_history.json
HISTORY_MAX = 30


def state_hash(data) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def history_path(shot_dir: str | Path) -> Path:
    return Path(shot_dir) / HISTORY_DIR / HISTORY_FILE


def _migrate(shot_dir: str | Path) -> None:
    old = Path(shot_dir) / EDITS_DIR / OLD_HISTORY
    if not old.is_file():
        return
    try:
        fixes = json.loads(old.read_text(encoding="utf-8")).get("fixes", [])
    except (OSError, ValueError, AttributeError):
        fixes = []
    if not history_path(shot_dir).is_file():
        out = []
        for f in fixes if isinstance(fixes, list) else []:
            f = dict(f)
            if "after" in f:
                f["after_hash"] = state_hash(f.pop("after"))
            out.append(f)
        _write_json(history_path(shot_dir), {"fixes": out[-HISTORY_MAX:]})
    old.unlink(missing_ok=True)


_HISTORY_CACHE: dict[str, tuple[tuple[int, int], list]] = {}


def history(shot_dir: str | Path) -> list[dict]:
    """The recorded fixes, oldest first (cached per file modification: the panel draws it)."""
    _migrate(shot_dir)
    p = history_path(shot_dir)
    try:
        st = p.stat()
        mtime = (st.st_mtime_ns, st.st_size)
    except OSError:
        return []
    hit = _HISTORY_CACHE.get(str(p))
    if hit is None or hit[0] != mtime:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        fixes = data.get("fixes") if isinstance(data, dict) else None
        fixes = fixes if isinstance(fixes, list) else []
        hit = _HISTORY_CACHE[str(p)] = (mtime, fixes)
    return [dict(f) for f in hit[1]]


def push_history(shot_dir: str | Path, entry: dict) -> None:
    """Remember one fix: ``{"id", "label", "time", "resolve", "name" + "before" (edit content or
    None) | "settings_before" (shot-level values or None = unset), "after_hash"}``."""
    fixes = [*history(shot_dir), entry][-HISTORY_MAX:]
    _write_json(history_path(shot_dir), {"fixes": fixes})


def drop_history(shot_dir: str | Path, entry_id: str) -> dict | None:
    """Remove the entry with ``entry_id`` (Revert done, or Forget); returns it."""
    fixes = history(shot_dir)
    keep = [f for f in fixes if f.get("id") != entry_id]
    if len(keep) == len(fixes):
        return None
    _write_json(history_path(shot_dir), {"fixes": keep})
    return next(f for f in fixes if f.get("id") == entry_id)


def current_state(shot_dir: str | Path, entry: dict):
    """What the fix changed, as it is now: the edit file's content (None when absent) or the
    shot-level smoothing settings."""
    if entry.get("name"):
        name = entry["name"]
        return read(shot_dir, name) if edit_path(shot_dir, name).is_file() else None
    return shot_smoothing(shot_dir)


def entry_matches(shot_dir: str | Path, entry: dict) -> bool:
    """Is the state still what the fix left (so it can be reverted)?"""
    try:
        return state_hash(current_state(shot_dir, entry)) == entry.get("after_hash")
    except EditError:
        return False


def revert_check(shot_dir: str | Path, entry: dict) -> None:
    """Refuse a revert when what the fix changed was changed again since (another fix of the same
    file, a hand edit, settings changed elsewhere)."""
    if not entry_matches(shot_dir, entry):
        what = f"edits/{entry['name']}.json" if entry.get("name") else "the shot's smoothing settings"
        raise EditError(f"{what} changed since '{entry.get('label')}'; not reverting it (Forget the entry)")


# --- shot settings (smoothing sliders) --------------------------------------------------------------
SMOOTH_KEYS = ("smooth.strength", *(f"smooth.region_strength.{r}" for r in ("torso", "head", "arms", "hands",
                                                                            "legs", "root")))


def shot_smoothing(shot_dir: str | Path) -> dict:
    """The smoothing values set at shot level (``shot.json`` settings.params), key -> value or None
    (not set: the default)."""
    try:
        info = json.loads((Path(shot_dir) / "shot.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = {}
    node = ((info.get("settings") or {}).get("params") or {})
    out = {}
    for key in SMOOTH_KEYS:
        v = node
        for part in key.split("."):
            v = v.get(part) if isinstance(v, dict) else None
        out[key] = v
    return out


def revert_sets(before: dict) -> list[str]:
    """``--set`` items putting the shot-level smoothing back (``null`` removes a key)."""
    return [f"{k}={'null' if v is None else json.dumps(v)}" for k, v in before.items()]

REGIONS = ("torso", "head", "arms", "hands", "legs", "root")


def smooth_settings(shot_dir: str | Path) -> dict:
    """The shot's saved smoothing settings (``shot.json`` settings.params.smooth): strength and
    region strengths (missing = 1.0)."""
    try:
        info = json.loads((Path(shot_dir) / "shot.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = {}
    sm = (((info.get("settings") or {}).get("params") or {}).get("smooth") or {})
    reg = sm.get("region_strength") or {}
    return {"strength": float(sm.get("strength", 1.0)),
            "region_strength": {r: float(reg.get(r, 1.0)) for r in REGIONS}}


def smooth_sets(strength: float, regions: dict[str, float]) -> list[str]:
    """``--set`` items for the smoothing sliders."""
    out = [f"smooth.strength={float(strength):.4g}"]
    for r in REGIONS:
        out.append(f"smooth.region_strength.{r}={float(regions.get(r, 1.0)):.4g}")
    return out
