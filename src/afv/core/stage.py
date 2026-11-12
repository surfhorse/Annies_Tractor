"""Stage plugin interface (spec: Plugin interface).

A stage is a class with class-level metadata and a ``run`` method. Heavy model stages run their
model in a pinned environment (``env``) as a subprocess; CPU stages (contacts, smooth, retarget,
export prep) set ``env = "cpu"`` and must never import torch.

Stage names and order are fixed by ``STAGE_ORDER``. ``inputs`` names upstream stages (or
"ingest") whose results are passed in by name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel

from afv.core.types import LicenceRecord, Result

# P0 outcome: GEM-X runs its own matched 2D keypoints inside pose3d, so there is no separate kp2d
# stage; "pair" puts both fighters in one consistent world (ground plane, depth, penetration).
STAGE_ORDER: tuple[str, ...] = (
    "ingest",
    "detect_track",
    "camera",
    "pose3d",
    "pair",
    "contacts",
    "smooth",
    "hands_face",
    "export",
)


class StageError(RuntimeError):
    """A stage failed with a message fit to show the user. Earlier results are kept."""


class CacheOnlyMiss(StageError):
    """Raised by a stage run with ``Ctx.cache_only`` when it would have to launch model work (a
    worker-cache miss). The runner then keeps the stored result (reported as ``stale_code``) instead
    of failing; nothing the stage wrote is kept."""


@dataclass
class Ctx:
    """Everything a stage needs besides its inputs and params."""

    shot_dir: Path  # <project>/shots/<shot_id>
    work_dir: Path  # temp folder for this run; renamed into place by the runner on success
    # Range re-runs (``afv run --from <stage> --frames a:b``; only for stages with
    # ``supports_range = True``): ``frame_range`` = (a, b), shot-timeline frames, ``b`` exclusive;
    # None = a full run. ``previous`` = the stage's previous stored Result (a fresh copy, as
    # downstream stages load it); the stage re-solves [a, b) plus its own margin, blends into
    # ``previous`` at the edges and returns a new full-length Result (identical outside the range +
    # margin). Both are None in a full run, so a range stage must handle that too.
    # ``previous_dir`` = the result folder ``previous`` was loaded from (``stages/<stage>/v<NNNN>/``),
    # for state a stage kept beside its result (e.g. contacts' ``range_state.npz``). Read it during
    # ``run`` only: once the new result is committed, older versions are deleted. Every file a stage
    # writes to ``ctx.work_dir`` (except ``tmp/``) is kept in its result folder.
    frame_range: tuple[int, int] | None = None
    previous: Result | None = None
    previous_dir: Path | None = None
    seed: int = 0
    deterministic: bool = False  # golden runs: fixed seeds, deterministic algorithms, TF32 off
    progress: Callable[[float, str], None] = field(default=lambda frac, msg: None)  # frac < 0: message only
    edits: dict[str, Any] = field(default_factory=dict)  # parsed edits/*.json relevant to this stage
    force: bool = False  # user forced a re-run (e.g. `afv run --from`): bypass any stage-internal caches
    # re-run triggered only by a code change (code_policy "cached"): use stage-internal caches only and
    # raise CacheOnlyMiss instead of launching model work on a miss
    cache_only: bool = False


@runtime_checkable
class Stage(Protocol):
    name: ClassVar[str]  # one of STAGE_ORDER
    version: ClassVar[str]  # bump when output semantics change; part of the cache key
    env: ClassVar[str]  # "cpu" or a pinned model environment, e.g. "gemx", "vipe"
    inputs: ClassVar[tuple[str, ...]]
    licences: ClassVar[tuple[LicenceRecord, ...]]  # every checkpoint used, incl. transitive
    Params: ClassVar[type[BaseModel]]
    weight_hash: ClassVar[str]  # sha256 of the weights used ("" for CPU stages)
    # Optional (read with getattr by the runner):
    # edits: ClassVar[tuple[str, ...]]        names of edits/<name>.json the stage reads (part of its key)
    # shot_fields: ClassVar[tuple[str, ...]]  shot.json fields it depends on, e.g. ("speed_factor",)
    # code_deps: ClassVar[tuple[str, ...]]    extra modules/files in the stage's code hash (afv.core.codehash)
    # worker_scripts: ClassVar[tuple[str, ...]]  worker scripts it launches (replaces the "*.py" string rule)
    # code_policy: ClassVar[str]  on a code change: "auto" (re-run; CPU default), "report" (keep, show
    #                             "stale (code)"; model default) or "cached" (re-run from caches only)
    # supports_range: ClassVar[bool]  True: the stage can re-solve a frame range (``ctx.frame_range``,
    #                             ``ctx.previous``; see Ctx). The runner gives it a range only when
    #                             its previous result was made from the same upstream results (or
    #                             from upstream stages range-run in the same run) by the same
    #                             implementation, version and code; otherwise it runs in full.
    # run() may return None; files written to ctx.work_dir are kept (tmp/ dropped, exports/ published),
    # ctx.work_dir/qc.json becomes the QC summary in meta.json.

    def run(self, ctx: Ctx, inputs: dict[str, Result], params: BaseModel) -> Result: ...


_REGISTRY: dict[str, dict[str, type]] = {}


def register(impl: str) -> Callable[[type], type]:
    """Class decorator: register a stage implementation under ``(stage.name, impl)``."""

    def deco(cls: type) -> type:
        if cls.name not in STAGE_ORDER:
            raise ValueError(f"unknown stage name {cls.name!r}")
        _REGISTRY.setdefault(cls.name, {})[impl] = cls
        return cls

    return deco


def get_stage(name: str, impl: str) -> type:
    try:
        return _REGISTRY[name][impl]
    except KeyError:
        known = sorted(_REGISTRY.get(name, {}))
        raise KeyError(f"no implementation {impl!r} for stage {name!r}; known: {known}") from None


def implementations(name: str) -> list[str]:
    return sorted(_REGISTRY.get(name, {}))
