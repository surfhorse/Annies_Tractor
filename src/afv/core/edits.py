"""User edit files (``shots/<id>/edits/<name>.json``): structural validation and atomic writes.

The formats are the shared contract in ``docs/BUILD_PLAN.md`` ("Round 4: Shared formats"). Frames are
shot-timeline frame numbers with ``end`` exclusive; ``role`` is ``"A"``/``"B"``. Values are strict (no
coercion). Unknown keys are allowed and kept (at every level): a writer must not drop what a newer
reader understands.

- ``contacts``: ``{"spans": [{"role", "start", "end", "state": "plant"|"free", "foot": "left"|"right"}
  | {..., "channels": [...]}]}`` (read by contacts)
- ``pins``: ``{"pins": [{"role", "frame", "radius" (default 8), "preset", "bones": {bone: [x, y, z, w]},
  "root": [x, y, z] | null}]}`` (read by smooth)
- ``hands``: ``{"spans": [{"role", "start", "end", "side": "left"|"right"|"both"}]}`` (read by hands_face)
- ``ids``: ``{"swaps": [{"start", "end": f | null, "objects": [o1, o2]}]}`` (read by detect_track)
- ``fighters``: the picks (``afv.models.track`` validates it; see ``afv pick``)

This checks structure only (types, ranges, required keys); the stages that read a file check the
rest (e.g. that a pinned bone exists in the preset's rig, that a swapped object is tracked).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    WithJsonSchema,
    WrapValidator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from afv.core.project import ProjectError, Shot


class EditError(ProjectError):
    """An edit file does not match its format; the message is fit to show the user."""


# --- strict values ----------------------------------------------------------------------------------
# One set of rules, mirrored by hand in ``blender/afv_addon/edits.py`` (Blender has no pydantic); the
# shared vectors in ``tests/core/edit_vectors.py`` run through both. Nothing is coerced:
#   integer  a JSON integer only: not a bool, not 3.0, not "3"
#   number   an int or a float, finite, not a bool, not "0.5" (an int is fine: JSON has one number type)
#   string   a str
#   null     only where a field says so (pin root, swap end, a contact span's foot or channels)
# Unknown keys are kept and not checked, except for ``_scan`` (anywhere in the document): NaN/infinity,
# keys that are not strings, integers over 4096 bits, and values JSON cannot hold are refused.
def _shown(v: Any) -> str:
    try:
        text = repr(v)
    except ValueError:  # an integer past Python's digit limit
        return "<value too large to show>"
    return text if len(text) <= 80 else text[:77] + "..."


def bad(msg: str, v: Any) -> PydanticCustomError:
    return PydanticCustomError("edit", "{msg}", {"msg": f"{msg}, got {_shown(v)}"})


def int_ge(lo: int | None) -> Any:
    def check(v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int) or (lo is not None and v < lo):
            raise bad("must be an integer" + (f" >= {lo}" if lo is not None else ""), v)
        return v

    return BeforeValidator(check)


def _number(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise bad("must be a finite number", v)
    try:
        f = float(v)
    except OverflowError:
        raise bad("must be a finite number", v) from None
    if not math.isfinite(f):
        raise bad("must be a finite number", v)
    return f


def one_of(*options: str) -> Any:
    shown = ", ".join(repr(o) for o in options[:-1]) + f" or {options[-1]!r}"

    def check(v: Any) -> str:
        if not isinstance(v, str) or v not in options:
            raise bad(f"must be {shown}", v)
        return v

    return BeforeValidator(check)


def _string(v: Any) -> str:
    if not isinstance(v, str):
        raise bad("must be a string", v)
    return v


def is_list(v: Any) -> Any:
    if not isinstance(v, list):
        raise bad("must be a list", v)
    return v


def is_dict(v: Any) -> Any:
    if not isinstance(v, dict):
        raise bad("must be an object", v)
    return v


def null_ok(v: Any, handler: Any) -> Any:
    return None if v is None else handler(v)


# (the JSON schemas say what the validators accept, for ``fighters_json_schema``)
Int = Annotated[Any, int_ge(None), WithJsonSchema({"type": "integer"})]
Frame = Annotated[Any, int_ge(0), WithJsonSchema({"type": "integer", "minimum": 0})]
Pos = Annotated[Any, int_ge(1), WithJsonSchema({"type": "integer", "minimum": 1})]
Role = Annotated[Any, one_of("A", "B"), WithJsonSchema({"type": "string", "enum": ["A", "B"]})]
Num = Annotated[float, BeforeValidator(_number)]
Str = Annotated[Any, BeforeValidator(_string), WithJsonSchema({"type": "string"})]


def nullable(schema: dict[str, Any]) -> Any:
    """The JSON schema of a field that is ``schema`` or null (with ``WrapValidator(null_ok)``)."""
    return WithJsonSchema({"anyOf": [schema, {"type": "null"}]})


def _vec(n: int, label: str) -> Any:
    def check(v: list[float]) -> list[float]:
        if len(v) != n:
            raise bad(f"must be {n} numbers {label}", v)
        return v

    return AfterValidator(check)


def _unit(q: list[float]) -> list[float]:
    length = math.sqrt(sum(x * x for x in q))
    if not 0.5 < length < 1.5:
        raise PydanticCustomError(
            "edit", "{msg}", {"msg": f"is not a unit quaternion (length {length:.3g}), got {_shown(q)}"}
        )
    return q


Quat = Annotated[list[Num], BeforeValidator(is_list), _vec(4, "[x, y, z, w]"), AfterValidator(_unit)]
Vec3 = Annotated[list[Num], BeforeValidator(is_list), _vec(3, "[x, y, z]")]


class Strict(BaseModel):
    """Base of the models of edit files: only a real dict (or an instance of the model) is an object."""

    @model_validator(mode="before")
    @classmethod
    def _real_dict(cls, v: Any) -> Any:
        if not isinstance(v, dict | cls):
            raise bad("must be an object", v)
        return v


class _Open(Strict):
    model_config = ConfigDict(extra="allow")


class _Span(_Open):
    role: Role
    start: Frame
    end: Int

    @model_validator(mode="after")
    def _order(self) -> _Span:
        if self.end <= self.start:
            raise ValueError(
                f"end ({_shown(self.end)}) must be after start ({_shown(self.start)}) (end is exclusive)"
            )
        return self


def _channels(v: Any) -> list[str]:
    if not isinstance(v, list) or not v or not all(isinstance(c, str) for c in v):
        raise bad("must be a non-empty list of channel names", v)
    return v


class ContactSpan(_Span):
    state: Annotated[Any, one_of("plant", "free")]
    foot: Annotated[Any, one_of("left", "right"), WrapValidator(null_ok)] = None
    channels: Annotated[Any, BeforeValidator(_channels), WrapValidator(null_ok)] = None

    @model_validator(mode="after")
    def _target(self) -> ContactSpan:
        if (self.foot is None) == (self.channels is None):
            raise ValueError("a contact span needs exactly one of 'foot' or 'channels'")
        return self


class ContactsFile(_Open):
    spans: Annotated[list[ContactSpan], BeforeValidator(is_list)] = Field(default_factory=list)


class Pin(_Open):
    role: Role
    frame: Frame
    radius: Annotated[Any, int_ge(1)] = 8
    preset: Str
    bones: Annotated[dict[str, Quat], BeforeValidator(is_dict)] = Field(default_factory=dict)
    root: Annotated[Vec3, WrapValidator(null_ok)] = None
    root_rotation: Annotated[Quat, WrapValidator(null_ok)] = None  # written by the add-on, read by smooth
    prefix: Str = ""
    root_motion: Str = ""

    @model_validator(mode="after")
    def _something(self) -> Pin:
        if not self.bones and self.root is None:
            raise ValueError("a pin needs 'bones' and/or 'root'")
        return self


class PinsFile(_Open):
    pins: Annotated[list[Pin], BeforeValidator(is_list)] = Field(default_factory=list)


class HandSpan(_Span):
    side: Annotated[Any, one_of("left", "right", "both")]


class HandsFile(_Open):
    spans: Annotated[list[HandSpan], BeforeValidator(is_list)] = Field(default_factory=list)


def _objects(v: Any) -> list[int]:
    ok = (
        isinstance(v, list)
        and len(v) == 2
        and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
        and v[0] != v[1]
        and min(v) >= 1
    )
    if not ok:
        raise bad("must be two different tracked object ids (integers >= 1)", v)
    return v


class Swap(_Open):
    start: Frame
    end: Annotated[Any, int_ge(None), WrapValidator(null_ok)] = None
    objects: Annotated[Any, BeforeValidator(_objects)]

    @model_validator(mode="after")
    def _check(self) -> Swap:
        if self.end is not None and self.end <= self.start:
            raise ValueError(
                f"end ({_shown(self.end)}) must be after start ({_shown(self.start)}) (end is exclusive)"
            )
        return self


class IdsFile(_Open):
    swaps: Annotated[list[Swap], BeforeValidator(is_list)] = Field(default_factory=list)


MODELS: dict[str, type[BaseModel]] = {
    "contacts": ContactsFile,
    "pins": PinsFile,
    "hands": HandsFile,
    "ids": IdsFile,
}
NAMES = (*MODELS, "fighters")


def _path(loc: tuple[Any, ...]) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def _one_error(d: Any) -> str:
    where = _path(d["loc"])
    kind = d["type"]
    if kind == "missing":
        text = "is required"
    elif kind in ("model_type", "dict_type", "model_attributes_type"):
        text = f"must be an object, got {_shown(d['input'])}"
    elif kind == "edit":
        text = d["msg"]
    elif kind == "value_error":
        text = d["msg"].removeprefix("Value error, ")
    else:
        text = f"{d['msg']}, got {_shown(d['input'])}"
    return f"{where}: {text}" if where else text


def describe_errors(e: ValidationError, limit: int = 3) -> str:
    """The first ``limit`` errors of ``e`` as ``path: what is wrong, got value; ...``."""
    errs = [_one_error(d) for d in e.errors()[:limit]]
    more = f" (and {len(e.errors()) - limit} more)" if len(e.errors()) > limit else ""
    return "; ".join(errs) + more


def _message(name: str, e: ValidationError) -> str:
    return f"edits/{name}.json: " + describe_errors(e)


MAX_BITS = 4096  # larger integers are refused (and could not be printed or written)


def _at(path: str, text: str) -> str:
    return f"{path}: {text}" if path else text


def _scan(v: Any, path: str = "") -> str | None:
    """The first problem anywhere in ``v`` that is not about a field: NaN/infinity, a key that is not a
    string, an integer too large, a value JSON cannot hold (numpy numbers, Decimal, Mapping...).
    None when there is none. Mirrored by the add-on."""
    if v is None or isinstance(v, bool | str):
        return None
    if isinstance(v, int):
        return _at(path, "integer is too large") if v.bit_length() > MAX_BITS else None
    if isinstance(v, float):
        return None if math.isfinite(v) else _at(path, "NaN or infinity is not allowed")
    if isinstance(v, dict):
        for k, x in v.items():
            if not isinstance(k, str):
                return _at(path, f"keys must be strings, got {_shown(k)}")
            if (hit := _scan(x, f"{path}.{k}" if path else k)) is not None:
                return hit
        return None
    if isinstance(v, list | tuple):
        for i, x in enumerate(v):
            if (hit := _scan(x, f"{path}[{i}]")) is not None:
                return hit
        return None
    return _at(path, f"not a JSON value, got {_shown(v)}")


def validate_edit(name: str, data: Any) -> None:
    """Raise EditError when ``data`` does not match the format of ``edits/<name>.json``. Values are
    strict (see "strict values" above): no coercion, and NaN/infinity, non-string keys and values JSON
    cannot hold are refused anywhere. The message names the file, the field path
    (``pins[2].bones.LeftArm[1]``) and the bad value."""
    if name not in NAMES:
        raise EditError(f"unknown edit {name!r}; edits: {', '.join(NAMES)}")
    if not isinstance(data, dict):
        raise EditError(f"edits/{name}.json: must be a JSON object, got {_shown(data)}")
    if (hit := _scan(data)) is not None:
        raise EditError(f"edits/{name}.json: {hit}")
    if name == "fighters":
        from afv.models.track import TrackError, load_fighters_edit

        try:
            load_fighters_edit(data)
        except TrackError as e:
            text = str(e).replace("edits/fighters.json is invalid: ", "edits/fighters.json: ", 1)
            if not text.startswith("edits/fighters.json"):
                text = f"edits/fighters.json: {text}"
            raise EditError(text) from None
        return
    try:
        MODELS[name].model_validate(data, strict=False)
    except ValidationError as e:
        raise EditError(_message(name, e)) from None


def fix_hint(name: str, data: Any) -> str:
    """What to do about an invalid ``edits/<name>.json`` whose content is ``data``."""
    if name == "fighters":
        return "fix or delete edits/fighters.json first, then pick again with afv pick"
    if not isinstance(data, dict):
        return f"fix or delete the file edits/{name}.json"
    return f"fix or delete that entry in edits/{name}.json"


def check_stored_edit(name: str, data: Any) -> None:
    """``validate_edit`` for a file already on disk (what stages load): the same rules, and the message
    ends with what to do. Raises EditError."""
    try:
        validate_edit(name, data)
    except EditError as e:
        raise EditError(f"{e} ({fix_hint(name, data)})") from None


def parse_ids(data: Any) -> IdsFile:
    """``edits/ids.json`` as a model (EditError when invalid); ``None`` gives no swaps."""
    if data is None:
        return IdsFile()
    validate_edit("ids", data)
    return IdsFile.model_validate(data)


REPLACED_WHOLE = ("fighters",)  # edits a write replaces entirely (their keys depend on each other)


def write_edit(shot: Shot, name: str, data: dict[str, Any] | None) -> Path:
    """Validate and write ``edits/<name>.json`` atomically; ``data=None`` removes the file.

    contacts, pins, hands, ids: top-level keys of the existing file that ``data`` does not mention are
    kept (unknown keys survive a writer that does not know them; a mentioned key is replaced whole).
    fighters (``REPLACED_WHOLE``): ``data`` is the whole file (its picks, prompts and boxes belong
    together, so keeping old keys could mix two picks)."""
    path = shot.edit_path(name)
    if name not in NAMES:
        raise EditError(f"unknown edit {name!r}; edits: {', '.join(NAMES)}")
    if data is None:
        path.unlink(missing_ok=True)
        return path
    old = shot.read_edit(name) if path.is_file() and name not in REPLACED_WHOLE else None
    merged = {**old, **data} if isinstance(old, dict) and isinstance(data, dict) else data
    return shot.write_edit(name, merged)  # validates
