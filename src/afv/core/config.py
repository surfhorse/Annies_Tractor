"""User configuration and stage settings.

Two kinds of configuration:

- **Machine config** (``~/.config/afv/config.toml``, optional): paths and limits that depend on the
  machine, e.g. ``ffmpeg``, ``gpu_lock``, ``worker_timeout_s``. Model environments live in a separate
  ``envs.toml`` (see ``afv.models.envs``).
- **Stage settings**: per-stage parameter overrides, layered project < shot < command line, and
  validated against the stage's pydantic ``Params`` model. Unknown keys are an error (typos must not
  silently fall back to defaults).

Settings dicts look like ``{"smooth": {"strength": 0.6}}``; on the command line that is
``--set smooth.strength=0.6``.
"""

from __future__ import annotations

import json
import os
import tomllib
import typing
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from afv.core.stage import STAGE_ORDER


class ConfigError(ValueError):
    """Bad configuration or settings, with a message fit to show the user."""


def afv_home() -> Path:
    """Root of the model environments and shared state (``$AFV_HOME`` or ``~/afv``)."""
    return Path(os.environ.get("AFV_HOME", "~/afv")).expanduser()


def config_dir() -> Path:
    """Folder holding ``config.toml`` and ``envs.toml`` (``$AFV_CONFIG_DIR`` or ``~/.config/afv``)."""
    return Path(os.environ.get("AFV_CONFIG_DIR", "~/.config/afv")).expanduser()


def load_toml(path: Path) -> dict[str, Any]:
    """Parse a TOML file; a missing file is an empty dict, a broken one a ConfigError."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise ConfigError(f"{path}: {e}") from None


class MachineConfig(BaseModel):
    """Machine-level settings (``config.toml``)."""

    ffmpeg: str = "~/.local/bin/ffmpeg"
    gpu_lock: str = ""  # default: <afv_home>/.gpu.lock
    worker_timeout_s: float = 6 * 3600.0
    gpu_lock_timeout_s: float | None = None  # None = wait for the GPU as long as it takes

    model_config = {"extra": "forbid"}

    def gpu_lock_path(self) -> Path:
        return Path(self.gpu_lock).expanduser() if self.gpu_lock else afv_home() / ".gpu.lock"


def load_machine_config(path: Path | None = None) -> MachineConfig:
    data = load_toml(path or config_dir() / "config.toml")
    try:
        return MachineConfig.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"config.toml: {_pydantic_message(e)}") from None


# ---------------------------------------------------------------------------
# Stage settings
# ---------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge; ``over`` wins. A value of None in ``over`` removes the key."""
    out = dict(base)
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_value(text: str) -> Any:
    """Command-line value: JSON if it parses (numbers, true/false, null, lists), else the raw string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_set(items: list[str] | tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """``["smooth.strength=0.6", "smooth.region.arms=0.3"]`` -> nested per-stage settings.

    ``key=null`` removes an earlier override (back to the stage default).
    """
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"--set {item!r}: expected <stage>.<setting>=<value>")
        path, _, raw = item.partition("=")
        parts = [p.strip() for p in path.strip().split(".")]
        if len(parts) < 2 or not all(parts):
            raise ConfigError(f"--set {item!r}: expected <stage>.<setting>=<value>")
        stage, *keys = parts
        if stage not in STAGE_ORDER:
            raise ConfigError(f"--set {item!r}: unknown stage {stage!r}; stages: {', '.join(STAGE_ORDER)}")
        node = out.setdefault(stage, {})
        for k in keys[:-1]:
            nxt = node.setdefault(k, {})
            if not isinstance(nxt, dict):
                raise ConfigError(f"--set {item!r}: {k!r} is set both as a value and as a group")
            node = nxt
        node[keys[-1]] = parse_value(raw)
    return out


def resolve_params(stage: str, params_cls: type[BaseModel], *layers: dict[str, Any]) -> BaseModel:
    """Merge settings layers (lowest first) and validate them into ``params_cls``."""
    merged: dict[str, Any] = {}
    for layer in layers:
        if layer:
            merged = deep_merge(merged, layer)
    _check_keys(stage, params_cls, merged, "")
    try:
        return params_cls.model_validate(merged)
    except ValidationError as e:
        raise ConfigError(f"settings for stage {stage}: {_pydantic_message(e)}") from None


def _check_keys(stage: str, cls: type[BaseModel], values: dict[str, Any], prefix: str) -> None:
    """Unknown keys are an error, also inside nested settings models (typos must not be ignored)."""
    if (getattr(cls, "model_config", {}) or {}).get("extra") == "allow":
        return
    unknown = sorted(set(values) - set(cls.model_fields))
    if unknown:
        known = ", ".join(prefix + k for k in sorted(cls.model_fields)) or "(none)"
        raise ConfigError(
            f"unknown setting(s) for stage {stage}: {', '.join(prefix + k for k in unknown)}; known: {known}"
        )
    for k, v in values.items():
        sub = _model_type(cls.model_fields[k].annotation)
        if sub is not None and isinstance(v, dict):
            _check_keys(stage, sub, v, f"{prefix}{k}.")


def _model_type(ann: Any) -> type[BaseModel] | None:
    """The BaseModel in an annotation such as ``Model`` or ``Model | None``."""
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return ann
    models = [a for a in typing.get_args(ann) if isinstance(a, type) and issubclass(a, BaseModel)]
    return models[0] if len(models) == 1 else None


def params_to_json(params: BaseModel) -> dict[str, Any]:
    return params.model_dump(mode="json")


def _pydantic_message(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)
