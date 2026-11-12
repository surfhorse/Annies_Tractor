"""Pipeline runner: resolve stages, decide what is stale, run, store results safely.

Every run includes ``ingest`` (the first stage) plus the stages the target needs (the closure of
``inputs``), in ``STAGE_ORDER``. For each:

1. resolve the implementation (command line > shot settings > project settings > ``DEFAULT_IMPL`` >
   the only one registered > ``"default"``) and its params (project < shot < command line, plus
   shot fields such as ``speed_factor`` when the Params model declares them);
2. compute the cache key (see ``afv.core.cache``); a stored result with the same key whose file still
   matches its recorded hash is reused unless ``--from`` forces the stage;
3. otherwise run it in a fresh work folder ``shots/<id>/.work/<stage>-...``. On success the result
   (``result.npz``), ``meta.json`` and every file the stage wrote are fsynced and renamed to a new
   ``stages/<stage>/v<NNNN>/`` (the newest version with a meta.json is current; old versions are
   deleted best-effort). ``work_dir/tmp/`` is dropped; ``work_dir/exports/`` (and, for the export
   stage, top-level .fbx/.glb/.gltf/.bvh files) are published to the shot's ``exports/`` before the
   result is committed, replacing what the previous result had published. On failure the previous
   result is untouched and the work folder is kept as ``.work/failed-<stage>-<time>/`` (last
   ``KEEP_FAILED`` per stage) with ``logs/error.txt``. A finished result that could not be moved into
   place is kept with a ``commit_pending`` marker and adopted by the next run with the same key.

Downstream stages always receive their inputs as loaded from the stored ``result.npz`` (float32, a
fresh copy per consumer), so the first run and later runs see identical data.

Fighter pick: a ``Tracks`` result with no ``fighters`` means the user still has to pick. The runner
stops there with outcome ``needs_pick`` and the prompt image (``prompt.jpg``/``prompt.png`` that
the detect_track stage wrote into its work folder). ``afv pick`` writes ``edits/fighters.json``,
which is part of detect_track's key, so the next run re-runs detect_track and continues.

Code changes: each result's meta records the code that produced it (``afv.core.codehash``: the
normalised source of the stage module, the afv modules it imports and the worker scripts it
launches), snapshotted once per process when a run starts. It is not part of the key; the runner
compares it with the code on disk now, so the two kinds of stage can be treated differently. A
stage's ``code_policy`` (class attribute, else ``DEFAULT_CODE_POLICY``, else by env) decides:

- ``auto`` (CPU stages): any code change, an unrecorded record (results from before code hashing)
  or a failed hash makes the result stale; it re-runs on its own, cheaply. ``accept`` does not
  apply.
- ``report`` (model stages, minutes per run): the result is kept and shown as ``stale_code`` with
  the changed files; it re-runs with ``code_changes="rerun"`` (stage caches reused) or ``--from``
  (forced). ``code_changes="accept"`` records the current code on the stored result instead (a
  refactor known not to change outputs). A result without a record stays fresh with a note.
- ``cached`` (camera): when only afv modules changed, it re-runs on its own with ``Ctx.cache_only``
  (worker outputs from the stage's cache only); if that would launch model work, the stage raises
  ``CacheOnlyMiss``, the stored result is kept and shown as ``stale_code`` ("needs a model
  re-run"). A worker script change is treated as ``report``. ``rerun`` and ``--from`` run it fully.

A stage whose code on disk differs from the code this process loaded (saved since afv started,
e.g. during a batch) is not run: the run stops with outcome ``code_changed`` ("run again").

A Python version change and a failed hash count as changed and are named as such.

Tracking review: detect_track's QC hints (``afv.core.review.tracking_hints``: likely ID swap, lost
or drifting fighter, untracked person) are reported after detect_track. With ``pause_on_hints``
(``afv run``) the run stops with outcome ``needs_review`` before the first later stage that would
run and depends on tracking, unless the hints were acknowledged (``yes=True`` records them in
``<shot>/review.json`` for that detect_track output) or a new tracking result has none.

Range re-runs: ``run(from_stage=s, frames=(a, b))`` (``afv run --from s --frames a:b``). Every
forced stage whose class sets ``supports_range = True`` gets ``ctx.frame_range = (a, b)`` and
``ctx.previous`` (its previous stored Result) and returns a new full-length result, *if* that previous
result was made by the same implementation, version, code, shot, weights and seed from the same
upstream results (an upstream stage range-run in this same run counts as the same only if the
previous result was made from the very result that range run replaced), with the *same params*, and
with edits that differ *only inside the range* (every added/removed/changed span or pin of the stage's
edit files lies within [a, b); the edits a result was made with are kept in its ``edits_used.json``).
Any other forced stage runs in full (cheap for CPU stages), and the stages after it then run in full
too (their upstream changed everywhere). A stage that gets a range but solves the whole shot anyway
(its qc.json has no ``"range"``, e.g. its previous result did not match) counts as a full run. A range
result's key adds ``frame_range`` and ``previous`` (the previous output hash) to the parts; its meta
also records ``base_key`` (the key a full run with the same inputs has: under the rules above the
range result stands for that full run, so a later plain run keeps it) and ``range`` (frames, the
previous result, the full run the chain started from, every range since, and the stage's own qc.json
``"range"``). ``afv status`` and provenance show it.

Cancelling: ``Runner(cancel=event)``; once the event is set the run stops before the next stage, at
the running stage's next ``ctx.progress`` call (CPU stages) or when its worker subprocess ends (the
caller, e.g. ``afv serve``, kills it); the stage's work folder is dropped, its previous result kept,
and the report's outcome is ``cancelled``.

Stage conventions beyond ``afv.core.stage`` (all optional class attributes):
``edits`` (names of ``edits/<name>.json`` the stage reads; default ``DEFAULT_EDITS``),
``shot_fields`` (shot.json fields the stage depends on; default ``DEFAULT_SHOT_FIELDS``),
``inherit_params`` ({params field: upstream stage}: the field defaults to that stage's resolved params,
e.g. smooth's ``contact`` = the contacts stage's ContactParams, so a ``--set contacts.x=...`` reaches
every stage that re-pins feet; settings of the field itself still override). A stage may
return None (nothing for other stages to load). ``work_dir/qc.json`` goes into meta as the QC
summary (its ``"summary"`` key if present). ``ctx.progress(frac, msg)`` with ``frac < 0`` is a
message without a fraction.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import logging
import os
import platform
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

import afv
from afv.core import stage as stage_mod
from afv.core.cache import (
    PART_LABELS,
    atomic_copy,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    commit_version,
    diff_parts,
    fsync_tree,
    hash_obj,
    npz_hash,
    prune_versions,
    read_json,
    stage_key,
)
from afv.core.codehash import changed_files, snapshot_code, stage_code, stat_scope
from afv.core.config import ConfigError, afv_home, deep_merge, params_to_json, resolve_params
from afv.core.edits import NAMES as EDIT_NAMES
from afv.core.edits import EditError, check_stored_edit
from afv.core.project import (
    META_FILE,
    QC_FILE,
    RESULT_FILE,
    ProjectError,
    Shot,
    folder_identity,
    now_iso,
)
from afv.core.review import acknowledge, acknowledged, format_review, tracking_hints
from afv.core.stage import STAGE_ORDER, CacheOnlyMiss, Ctx, StageError
from afv.core.types import Result, Tracks, save_result

log = logging.getLogger("afv.runner")
logging.getLogger("afv").addHandler(logging.NullHandler())  # no logging.lastResort double prints

# Modules that register stage implementations, imported lazily; missing ones are tolerated.
STAGE_MODULES: tuple[str, ...] = (
    "afv.media.stage",
    "afv.models.track",
    "afv.models.camera",
    "afv.models.gemx",
    "afv.motion.stages",
    "afv.motion.hands_face",
    "afv.export.stage",
)
# Where each stage's implementation is expected (for error messages).
STAGE_HOME: dict[str, str] = {
    "ingest": "afv.media.stage",
    "detect_track": "afv.models.track",
    "camera": "afv.models.camera",
    "pose3d": "afv.models.gemx",
    "pair": "afv.motion.stages",
    "contacts": "afv.motion.stages",
    "smooth": "afv.motion.stages",
    "hands_face": "afv.motion.hands_face",
    "export": "afv.export.stage",
}
ALWAYS = ("ingest",)  # stages every run includes (probe, proxy, masks for the review tools)
DEFAULT_IMPL: dict[str, str] = {"camera": "vipe"}
DEFAULT_EDITS: dict[str, tuple[str, ...]] = {
    "detect_track": ("fighters", "ids"),
    "contacts": ("contacts",),
    "smooth": ("pins",),
}
DEFAULT_SHOT_FIELDS: dict[str, tuple[str, ...]] = {"smooth": ("speed_factor",), "export": ("speed_factor",)}
SHOT_FIELDS_ALL = ("start", "duration")  # in every key, with the source sha256
DEFAULT_TARGET = "export"
KEEP_FAILED = 3
PROMPT_NAMES = ("prompt.jpg", "prompt.png", "prompt.jpeg")
EXPORT_SUFFIXES = (".fbx", ".glb", ".gltf", ".bvh")  # export-stage files published even outside exports/
QC_META_MAX_BYTES = 64_000
FAILED_MARKER = ".afv_failed"  # inside a kept failed folder that could not be renamed to failed-*
COMMIT_PENDING = "commit_pending.json"  # inside a finished result that could not be committed
# Gravity re-level pass: after pose3d, ``afv.models.camera.relevel_override(pose3d qc)`` may return camera
# params (e.g. ``up_cam0`` from GEM-X's people-based gravity when GeoCalib's telephoto pitch is weak).
# The runner records them in ``<shot>/relevel.json`` with the camera key they were made for, re-runs
# camera (CPU only: its model outputs are cached) and pose3d once, and later runs apply the same
# override (cache-correct: nothing is redone). Delete the file to drop the re-level.
RELEVEL_FILE = "relevel.json"
CODE_CHANGES = ("report", "rerun", "accept")  # what to do with model-stage results whose code changed
CODE_POLICIES = ("auto", "report", "cached")  # see "Code changes"
# camera keeps its worker outputs in <shot>/.cache/camera keyed on the worker script and request, so
# re-running it for a change to its afv code is a cache hit (seconds); a worker change is reported
DEFAULT_CODE_POLICY: dict[str, str] = {"camera": "cached"}
CACHED_NOTE = "; re-runs from its caches only"  # code_policy "cached" (see "Code changes")
REVIEW_STAGE = "detect_track"  # its QC hints can pause the run (see "Tracking review" above)

Reporter = Callable[[dict[str, Any]], None]


def _relevel_rule() -> str:
    """[Z, round 3] ``afv.models.camera.RELEVEL_RULE``: the re-level decision rule's version."""
    try:
        from afv.models.camera import RELEVEL_RULE
    except ImportError:
        return ""
    return str(RELEVEL_RULE)


class RunError(RuntimeError):
    """A run could not complete; the message is fit to show the user."""

    def __init__(self, message: str, *, stage: str | None = None, log_path: Path | None = None):
        super().__init__(message)
        self.stage = stage
        self.log_path = log_path


class MissingStageError(RunError):
    """A needed stage has no (importable) implementation."""


class RunCancelled(BaseException):
    """Raised inside a run once its cancel event is set (a BaseException, so stage code that catches
    ``Exception`` does not swallow it); ``Runner.run`` turns it into the outcome ``cancelled``."""


RANGE_HISTORY_MAX = 50  # range re-runs listed in a result's meta ("range" -> "history")
# key parts a range re-run's previous result must share with the run (see "Range re-runs")
RANGE_SAME_PARTS = ("stage", "impl", "version", "weight_hash", "shot", "seed", "deterministic")


def parse_frames(text: str | list[int] | tuple[int, int] | None) -> tuple[int, int] | None:
    """``"a:b"`` (or ``[a, b]``) -> (a, b) with 0 <= a < b; None stays None."""
    if text is None:
        return None
    try:
        if isinstance(text, str):
            a_s, sep, b_s = text.partition(":")
            if not sep:
                raise ValueError
            a, b = int(a_s), int(b_s)
        else:
            a, b = (int(v) for v in text)
    except (TypeError, ValueError):
        raise RunError(
            f"--frames {text!r}: expected a:b (shot-timeline frames, b exclusive), e.g. 120:180"
        ) from None
    if a < 0 or b <= a:
        raise RunError(f"--frames {a}:{b}: need 0 <= a < b (b is exclusive)")
    return a, b


# ---------------------------------------------------------------------------
# Stage modules
# ---------------------------------------------------------------------------


def load_stage_modules(modules: tuple[str, ...] | list[str] | None = None) -> dict[str, str]:
    """Import the stage modules; returns {module: "ok" | "not built yet" | "import failed: ..."}."""
    status: dict[str, str] = {}
    for m in STAGE_MODULES if modules is None else modules:
        try:
            importlib.import_module(m)
            status[m] = "ok"
        except ModuleNotFoundError as e:
            if e.name and (m == e.name or m.startswith(e.name + ".")):
                status[m] = "not built yet"
            else:
                status[m] = f"import failed: {e}"
        except Exception as e:
            status[m] = f"import failed: {type(e).__name__}: {e}"
    return status


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class StageStatus:
    name: str
    impl: str | None
    # fresh | stale | stale_code (model result kept although its code changed) | forced | pending |
    # unavailable | ran | failed | needs_pick | blocked | skipped (code changed since afv started)
    state: str
    reason: str = ""
    key: str | None = None
    runtime_s: float | None = None
    range: list[int] | None = None  # [a, b]: the result is (or was just made by) a range re-run of a:b


@dataclass
class RunReport:
    shot: str
    outcome: str  # ok | needs_pick | needs_review | code_changed | failed | cancelled | planned
    stages: list[StageStatus] = field(default_factory=list)
    message: str = ""
    log_path: str | None = None
    prompt_image: str | None = None
    exports: list[str] = field(default_factory=list)
    relevel: dict[str, Any] | None = None  # the camera re-level record in effect (see RELEVEL_FILE)
    review: dict[str, Any] | None = None  # tracking QC hints of detect_track (see "Tracking review")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def find_prompt_image(stage_dir: Path) -> Path | None:
    for n in PROMPT_NAMES:
        if (stage_dir / n).is_file():
            return stage_dir / n
    hits = sorted(p for p in stage_dir.glob("prompt*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Runner:
    def __init__(
        self,
        shot: Shot,
        *,
        impl: dict[str, str] | None = None,
        params: dict[str, dict[str, Any]] | None = None,
        seed: int = 0,
        deterministic: bool = False,
        report: Reporter | None = None,
        stage_modules: tuple[str, ...] | list[str] | None = None,
        code_changes: str = "report",
        pause_on_hints: bool = False,
        yes: bool = False,
        cancel: threading.Event | None = None,
    ):
        if code_changes not in CODE_CHANGES:
            raise RunError(f"code_changes must be one of {', '.join(CODE_CHANGES)}, not {code_changes!r}")
        self.code_changes = code_changes
        self.pause_on_hints = pause_on_hints
        self.yes = yes
        self._code: dict[tuple[str, str], dict[str, Any] | None] = {}
        self._pending_review: dict[str, Any] | None = None
        self.shot = shot
        self.impl_over = dict(impl or {})
        self.params_over = dict(params or {})
        self.seed = seed
        self.deterministic = deterministic
        self.report = report or (lambda ev: None)
        self.module_status = load_stage_modules(stage_modules)
        pinfo, sinfo = shot.project.info, shot.info
        self._pinfo, self._sinfo = pinfo, sinfo
        self._verified: dict[str, str] = {}  # stage -> output hash verified against its file this run
        self._lock_depth = 0  # shot_lock() is re-entrant within this Runner
        self.cancel = cancel
        self._frames: tuple[int, int] | None = None  # range re-run frames of the current run
        self._ranged: set[str] = set()  # stages range-run in the current run
        self._replaced: dict[str, str] = {}  # range-run stage -> output hash of the result it replaced
        self._alt_keys: dict[str, str] = {}  # key -> its legacy form (see _legacy_key)
        self._run_thread: threading.Thread | None = None

    def cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def _check_cancel(self) -> None:
        if self.cancelled():
            raise RunCancelled()

    # -- resolution ----------------------------------------------------------

    def _emit(self, event: str, **kw: Any) -> None:
        self.report({"event": event, "shot": self.shot.id, **kw})

    def resolve(self, name: str) -> tuple[type, str]:
        """(stage class, implementation name) for ``name``; MissingStageError if none usable."""
        if name not in STAGE_ORDER:
            raise RunError(f"unknown stage {name!r}; stages: {', '.join(STAGE_ORDER)}")
        known = stage_mod.implementations(name)
        wanted = (
            self.impl_over.get(name)
            or self._sinfo.settings.impl.get(name)
            or self._pinfo.settings.impl.get(name)
        )
        if wanted is None:
            if name in DEFAULT_IMPL and DEFAULT_IMPL[name] in known:
                wanted = DEFAULT_IMPL[name]
            elif len(known) == 1:
                wanted = known[0]
            elif "default" in known:
                wanted = "default"
        if not known:
            home = STAGE_HOME.get(name, "?")
            why = self.module_status.get(home)
            detail = f"{home}: {why}" if why else home
            raise MissingStageError(f"stage {name!r} has no implementation yet ({detail})", stage=name)
        if wanted is None:
            raise MissingStageError(
                f"stage {name!r} has several implementations ({', '.join(known)}); choose one in settings",
                stage=name,
            )
        if wanted not in known:
            raise MissingStageError(
                f"stage {name!r} has no implementation {wanted!r}; available: {', '.join(known)}", stage=name
            )
        cls = stage_mod.get_stage(name, wanted)
        missing = [a for a in ("version", "env", "inputs", "Params", "run") if not hasattr(cls, a)]
        if missing:
            raise RunError(
                f"stage {name}/{wanted} ({cls.__qualname__}) lacks {', '.join(missing)}", stage=name
            )
        for inp in cls.inputs:
            if inp not in STAGE_ORDER or STAGE_ORDER.index(inp) >= STAGE_ORDER.index(name):
                raise RunError(
                    f"stage {name}/{wanted} declares input {inp!r}, not an earlier stage", stage=name
                )
        return cls, wanted

    def needed(self, target: str) -> list[str]:
        """Stages needed to produce ``target`` (always with ``ALWAYS``), in pipeline order.
        Raises if any is unavailable."""
        if target not in STAGE_ORDER:
            raise RunError(f"unknown stage {target!r}; stages: {', '.join(STAGE_ORDER)}")
        need: set[str] = set()
        missing: list[str] = []
        stack = [target, *ALWAYS]
        while stack:
            n = stack.pop()
            if n in need:
                continue
            need.add(n)
            try:
                cls, _ = self.resolve(n)
            except MissingStageError as e:
                missing.append(str(e))
                continue
            stack.extend(cls.inputs)
        if missing:
            avail = [s for s in STAGE_ORDER if stage_mod.implementations(s)]
            hint = f"; implemented so far: {', '.join(avail)}" if avail else ""
            raise MissingStageError("cannot run to " + target + ": " + "; ".join(sorted(missing)) + hint)
        return [s for s in STAGE_ORDER if s in need]

    def params_for(self, name: str, cls: type, extra: dict[str, Any] | None = None) -> BaseModel:
        """Resolved params: shot fields < ``extra`` (e.g. the re-level override) < project < shot <
        command line; inherited fields (``inherit_params``) start from the upstream stage's params."""
        return self._params_for(name, cls, (), extra)

    def _shot_injected(self, name: str, cls: type) -> dict[str, Any]:
        fields = self._shot_fields(name, cls)
        return {f: getattr(self._sinfo, f) for f in fields if f in cls.Params.model_fields}

    def _inherited(self, name: str, cls: type, _seen: tuple[str, ...]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for field_name, src in dict(getattr(cls, "inherit_params", {}) or {}).items():
            if field_name not in cls.Params.model_fields:
                raise RunError(f"stage {name} inherits params into unknown field {field_name!r}", stage=name)
            if src in _seen or src == name:
                raise RunError(f"stage {name}: circular inherit_params via {src!r}", stage=name)
            try:
                src_cls, _ = self.resolve(src)
            except MissingStageError:
                continue  # no implementation: the field keeps its default
            out[field_name] = params_to_json(self._params_for(src, src_cls, (*_seen, name)))
        return out

    def _user_layer(
        self, name: str, cls: type, layers: list[dict[str, Any]], _seen: tuple[str, ...]
    ) -> dict[str, Any]:
        """Merge settings layers (lowest first); an inherited field starts from the upstream params, so
        ``<field>=null`` (or no setting) means "as the upstream stage", key by key otherwise."""
        user: dict[str, Any] = {}
        for layer in layers:
            if layer:
                user = deep_merge(user, layer)
        for field_name, inherited in self._inherited(name, cls, _seen).items():
            mine = user.get(field_name)
            user[field_name] = deep_merge(inherited, mine) if isinstance(mine, dict) else (
                inherited if mine is None else mine
            )
        return user

    def _params_for(
        self, name: str, cls: type, _seen: tuple[str, ...], extra: dict[str, Any] | None = None
    ) -> BaseModel:
        layers = [
            extra or {},
            self._pinfo.settings.params.get(name, {}),
            self._sinfo.settings.params.get(name, {}),
            self.params_over.get(name, {}),
        ]
        return resolve_params(
            name, cls.Params, self._shot_injected(name, cls), self._user_layer(name, cls, layers, _seen)
        )

    @staticmethod
    def _shot_fields(name: str, cls: type) -> tuple[str, ...]:
        return tuple(getattr(cls, "shot_fields", DEFAULT_SHOT_FIELDS.get(name, ())))

    @staticmethod
    def edit_names(name: str, cls: type) -> tuple[str, ...]:
        return tuple(getattr(cls, "edits", DEFAULT_EDITS.get(name, ())))

    def _edits(self, name: str, cls: type) -> dict[str, Any]:
        out = {}
        for e in self.edit_names(name, cls):
            data = self.shot.read_edit(e)
            if data is not None:
                if e in EDIT_NAMES:  # a file with lax values stops the run, whoever wrote it
                    check_stored_edit(e, data)
                out[e] = data
        return out

    def key_parts(
        self,
        name: str,
        cls: type,
        impl: str,
        params: BaseModel,
        inputs: dict[str, str],
        edits: dict[str, Any],
    ) -> dict[str, Any]:
        shot = {"source_sha256": self._pinfo.source.sha256}
        shot.update({f: getattr(self._sinfo, f) for f in SHOT_FIELDS_ALL})
        shot.update({f: getattr(self._sinfo, f) for f in self._shot_fields(name, cls)})
        return {
            "stage": name,
            "impl": impl,
            "version": str(cls.version),
            "weight_hash": str(getattr(cls, "weight_hash", "")),
            "params": params_to_json(params),
            "inputs": dict(inputs),
            "edits": {e: hash_obj(edits[e]) for e in self.edit_names(name, cls) if e in edits},
            "shot": shot,
            "frame_range": None,
            "seed": self.seed,
            "deterministic": self.deterministic,
        }

    def _legacy_key(self, name: str, cls: type, parts: dict[str, Any]) -> str:
        """The key the same parts had before absent edits were left out of them (absent = None): a
        result stored then still matches."""
        legacy = {e: parts["edits"].get(e) for e in self.edit_names(name, cls)}
        return stage_key({**parts, "edits": legacy})

    def validate_settings(self, settings: dict[str, dict[str, Any]]) -> None:
        """Check overrides against each stage's Params (stages without an implementation are skipped)."""
        for name, values in settings.items():
            try:
                cls, _ = self.resolve(name)
            except MissingStageError:
                continue
            user = self._user_layer(name, cls, [values], ())
            resolve_params(name, cls.Params, self._shot_injected(name, cls), user)

    # -- planning ------------------------------------------------------------

    def _forced(self, needed: list[str], from_stage: str | None, target: str) -> set[str]:
        if from_stage is None:
            return set()
        if from_stage not in STAGE_ORDER:
            raise RunError(f"unknown stage {from_stage!r} for --from; stages: {', '.join(STAGE_ORDER)}")
        if STAGE_ORDER.index(from_stage) > STAGE_ORDER.index(target):
            raise RunError(f"--from {from_stage} comes after the target stage {target}")
        i = STAGE_ORDER.index(from_stage)
        return {s for s in needed if STAGE_ORDER.index(s) >= i}

    def check_edits(self, to: str | None = None) -> None:
        """Raise EditError when a stored edit file that a stage up to ``to`` reads is invalid (strict
        validation, with a hint how to fix it). ``run`` does this before anything runs; callers that
        save settings before running (``--set``) should call it first."""
        for name in self.needed(to or DEFAULT_TARGET):
            self._edits(name, self.resolve(name)[0])

    def plan(
        self, to: str | None = None, from_stage: str | None = None, frames: tuple[int, int] | None = None
    ) -> RunReport:
        """What ``run`` would do, without running anything."""
        with stat_scope():
            return self._plan(to, from_stage, frames)

    def _plan(self, to: str | None, from_stage: str | None, frames: tuple[int, int] | None) -> RunReport:
        target = to or DEFAULT_TARGET
        if frames is not None and from_stage is None:
            raise RunError("--frames needs --from <stage> (the first stage to re-run for those frames)")
        report = RunReport(self.shot.id, "planned")
        try:
            self.shot.project.check_source()
        except ProjectError as e:
            report.message = str(e)
        try:
            needed = self.needed(target)
        except MissingStageError as e:
            report.message = str(e)
            needed = []
            for s in STAGE_ORDER[: STAGE_ORDER.index(target) + 1]:
                try:
                    self.resolve(s)
                    needed.append(s)
                except MissingStageError as me:
                    report.stages.append(StageStatus(s, None, "unavailable", str(me)))
        forced = self._forced(needed, from_stage, target)
        hashes: dict[str, str] = {}
        blocked: str | None = None
        for name in needed:
            cls, impl = self.resolve(name)
            try:
                edits = self._edits(name, cls)
            except EditError as e:  # a lax edit file: this stage and the ones after it cannot run
                report.stages.append(StageStatus(name, impl, "blocked", str(e)))
                report.message = _join(report.message, str(e))
                report.outcome = "failed"
                blocked = blocked or f"after {name}: its edit file must be fixed first"
                continue
            if blocked:
                report.stages.append(StageStatus(name, impl, "blocked", blocked))
                continue
            if waiting := [i for i in cls.inputs if i not in hashes]:
                why = f"after {', '.join(waiting)} re-runs"
                report.stages.append(StageStatus(name, impl, "pending", why))
                continue
            params = self.params_for(name, cls)
            inputs = {i: hashes[i] for i in cls.inputs}
            if name == "camera":  # with the re-level override a previous run recorded, if it applies
                params = self._camera_relevel(cls, impl, params, inputs, edits)[0]
            parts = self.key_parts(name, cls, impl, params, inputs, edits)
            key = self._stage_key(name, cls, parts)
            meta = self.shot.meta(name)
            verdict, code_why = "stale", ""
            if name not in forced and self._fresh(name, meta, key):
                assert meta is not None
                verdict, code_why = self._code_verdict(name, cls, meta)
            if name in forced:
                why = "--from"
                if frames is not None:
                    a, b = frames
                    why += (
                        f", frames {a}:{b} (if its previous result allows)"
                        if getattr(cls, "supports_range", False)
                        else " (in full: no range support)"
                    )
                report.stages.append(StageStatus(name, impl, "forced", why, key))
            elif verdict not in ("stale", "stale_cached"):
                assert meta is not None
                hashes[name] = meta["output_hash"]
                if meta.get("needs_input") == "pick":
                    report.stages.append(StageStatus(name, impl, "needs_pick", "fighters not picked", key))
                    blocked = f"waiting for 'afv pick {self.shot.id}'"
                    continue
                if verdict == "accept":
                    verdict, code_why = "fresh", f"{code_why} (accepted when run)"
                why = _join(range_note(meta), self._with_advice(name, verdict, code_why))
                st = StageStatus(name, impl, verdict, why, key, range=_range_frames(meta))
                report.stages.append(st)
                if name == REVIEW_STAGE and (rv := self._tracking_review(meta)) is not None:
                    report.review = rv
                    if not rv["acknowledged"]:
                        n = len(rv["hints"])
                        st.reason = "; ".join(x for x in (st.reason, (
                            f"QC: {n} tracking hint{'s' if n != 1 else ''}, 'afv run' pauses after this "
                            "stage (--yes to go on)")) if x)  # fmt: skip
            else:
                why = code_why or self._stale_reason(name, meta, parts, cls)
                if verdict == "stale_cached":
                    why += CACHED_NOTE
                report.stages.append(StageStatus(name, impl, "stale", why, key))
        report.stages.sort(key=lambda s: STAGE_ORDER.index(s.name))
        return report

    def _key_matches(self, meta: dict[str, Any], key: str) -> bool:
        """The stored result was made with ``key`` (or its legacy form, see ``_legacy_key``), or is a
        range re-run whose full-run key is ``key``."""
        keys = {key, self._alt_keys.get(key)}
        return meta.get("key") in keys or (bool(meta.get("range")) and meta.get("base_key") in keys)

    def _stage_key(self, name: str, cls: type, parts: dict[str, Any]) -> str:
        key = stage_key(parts)
        legacy = self._legacy_key(name, cls, parts)
        if legacy != key:
            self._alt_keys[key] = legacy
        return key

    def _fresh(self, name: str, meta: dict[str, Any] | None, key: str) -> bool:
        if not meta or not self._key_matches(meta, key) or "output_hash" not in meta:
            return False
        if meta.get("has_result") is False:
            return True
        return self._result_intact(name, meta)

    def _result_intact(self, name: str, meta: dict[str, Any]) -> bool:
        """The stored result.npz still matches the hash recorded when it was written."""
        if self._verified.get(name) == meta["output_hash"]:
            return True
        p = self.shot.stage_dir(name) / RESULT_FILE
        try:
            ok = p.is_file() and npz_hash(p) == meta["output_hash"]
        except Exception:  # truncated or corrupt zip
            ok = False
        if ok:
            self._verified[name] = meta["output_hash"]
        return ok

    def _stale_reason(
        self, name: str, meta: dict[str, Any] | None, parts: dict[str, Any], cls: type | None = None
    ) -> str:
        """Why a stored result does not match: the key parts that changed (params, upstream, edits,
        version, ...), plus the code when it changed too."""
        if meta and cls is not None and self._key_matches(meta, self._stage_key(name, cls, parts)):
            return "stored result is missing or damaged"
        old = meta.get("key_parts") if meta else None
        if meta and meta.get("range") and isinstance(old, dict):  # compare with its full-run parts
            old = {k: v for k, v in old.items() if k != "previous"} | {"frame_range": None}
        if isinstance(old, dict) and isinstance(old.get("edits"), dict):  # legacy: absent edits as None
            old = {**old, "edits": {k: v for k, v in old["edits"].items() if v is not None}}
        why = diff_parts(old, parts)
        code = self._code_diff(name, cls, meta) if meta and cls is not None else None
        if code is not None and code[0] != "unrecorded":
            why.append(code[1])
        return ", ".join(why)

    # -- code changes ----------------------------------------------------------

    def code_record(self, name: str, cls: type) -> dict[str, Any] | None:
        """The stage's code record as it is on disk now (compared with the stored one), once per
        Runner; None when it cannot be hashed."""
        k = (name, f"{cls.__module__}.{cls.__qualname__}")
        if k not in self._code:
            try:
                self._code[k] = stage_code(cls)
            except (OSError, RecursionError, ValueError) as e:
                log.warning("cannot hash the code of stage %s: %s", name, e)
                self._code[k] = None
        return self._code[k]

    def _snapshot(self, name: str, cls: type) -> dict[str, Any] | None:
        """The record of the code this process runs (taken once per process when a run starts);
        this is what a result's meta stores."""
        try:
            return snapshot_code(cls)
        except (OSError, RecursionError, ValueError) as e:
            log.warning("cannot hash the code of stage %s: %s", name, e)
            return None

    @staticmethod
    def code_policy(name: str, cls: type) -> str:
        """What a code change does to a stored result (see "Code changes")."""
        pol = getattr(cls, "code_policy", None)
        if pol is None and getattr(cls, "env", "cpu") != "cpu":
            pol = DEFAULT_CODE_POLICY.get(name)
        if pol in CODE_POLICIES:
            return str(pol)
        return "auto" if getattr(cls, "env", "cpu") == "cpu" else "report"

    def _code_diff(self, name: str, cls: type, meta: dict[str, Any]) -> tuple[str, str] | None:
        """(kind, reason) when the stored code differs from the current one, else None; kind is
        changed | workers (a worker script changed) | python | failed | unrecorded."""
        now = self.code_record(name, cls)
        old = meta.get("code")
        if now is None:
            return "failed", "code could not be hashed (counted as changed)"
        if not isinstance(old, dict) or not old.get("hash"):
            return "unrecorded", "code not recorded (result predates code hashing)"
        if old.get("python") and now.get("python") and old["python"] != now["python"]:
            return "python", f"Python version changed ({old['python']} -> {now['python']})"
        if old["hash"] == now["hash"]:
            return None
        files = changed_files(old, now)
        more = f" and {len(files) - 4} more" if len(files) > 4 else ""
        kind = "workers" if any(f.endswith(".py") and "/" in f for f in files) else "changed"
        return kind, f"code changed: {', '.join(files[:4])}{more}"

    def _code_verdict(self, name: str, cls: type, meta: dict[str, Any]) -> tuple[str, str]:
        """For a stored result whose key matches: ("fresh" | "stale" | "stale_code" | "accept", note)."""
        diff = self._code_diff(name, cls, meta)
        if diff is None:
            return "fresh", ""
        kind, why = diff
        policy = self.code_policy(name, cls)
        if policy == "auto":
            return "stale", why  # cheap: re-run on its own
        if self.code_changes == "accept" and kind != "failed":
            return "accept", why
        if policy == "cached" and kind == "changed" and self.code_changes != "rerun":
            return "stale_cached", why  # re-run from the stage's caches only (Ctx.cache_only)
        if kind == "unrecorded":
            return "fresh", why
        if self.code_changes == "rerun":
            return "stale", why
        return "stale_code", why

    @staticmethod
    def _with_advice(name: str, verdict: str, why: str) -> str:
        if verdict != "stale_code":
            return why
        return f"{why}; result kept: --code-changes rerun (reuses worker caches) or --from {name}"

    def _accept_code(self, name: str, cls: type, meta: dict[str, Any]) -> dict[str, Any]:
        """Record the current code on the stored result (meta.json rewritten in place)."""
        previous = (meta.get("code") or {}).get("hash")
        accepted = {"previous": previous, "at": now_iso()}
        new = {**meta, "code": self.code_record(name, cls), "code_accepted": accepted}
        atomic_write_json(self.shot.stage_dir(name) / META_FILE, new)
        return new

    def _code_moved(self, name: str, cls: type) -> str | None:
        """Why ``name`` must not run in this process: its code on disk differs from the code loaded
        (saved since afv started, e.g. during a batch); None when they agree or cannot be compared."""
        snap, disk = self._snapshot(name, cls), self.code_record(name, cls)
        if snap is None or disk is None or snap.get("hash") == disk.get("hash"):
            return None
        files = changed_files(snap, disk)
        return ", ".join(files[:4]) + (f" and {len(files) - 4} more" if len(files) > 4 else "")

    def _stop_code_moved(self, name: str, impl: str, key: str, report: RunReport, files: str) -> str:
        why = f"code changed since afv started ({files}); run again"
        report.stages.append(StageStatus(name, impl, "skipped", why, key))
        report.outcome = "code_changed"
        report.message = f"stage {name} not run: its {why} (this process still has the old code loaded)"
        self._emit("code_changed", stage=name, message=report.message)
        return "stop"

    # -- tracking review -------------------------------------------------------

    def _tracking_review(self, meta: dict[str, Any]) -> dict[str, Any] | None:
        """detect_track's tracking hints and whether they were acknowledged; None without hints."""
        hints = tracking_hints(meta.get("qc"), self.shot.id)
        if not hints:
            return None
        sdir = self.shot.stage_dir(REVIEW_STAGE)
        video = sdir / "review.mp4"
        return {
            "stage": REVIEW_STAGE,
            "hints": hints,
            "acknowledged": acknowledged(self.shot.dir, REVIEW_STAGE, meta["output_hash"], hints),
            "output_hash": meta["output_hash"],
            "review_video": str(video) if video.is_file() else None,
            "qc_file": str(sdir / QC_FILE),
        }

    def _review_after(self, meta: dict[str, Any], report: RunReport) -> None:
        """After detect_track (run or reused): acknowledge its hints (``yes``) or hold them pending."""
        rv = self._tracking_review(meta)
        self._pending_review = None
        if rv is None:
            return
        report.review = rv
        if rv["acknowledged"]:
            return
        if self.yes:
            acknowledge(self.shot.dir, REVIEW_STAGE, rv["output_hash"], rv["hints"], now_iso())
            rv["acknowledged"] = True
            self._emit_review(rv, paused=False)
            return
        self._pending_review = rv

    def _emit_review(self, rv: dict[str, Any], *, paused: bool) -> None:
        self._emit(
            "tracking_review", stage=REVIEW_STAGE, paused=paused, acknowledged=rv["acknowledged"],
            hints=rv["hints"], review_video=rv["review_video"], qc_file=rv["qc_file"],
            message=format_review(rv, self.shot.id, paused=paused),
        )  # fmt: skip

    @staticmethod
    def _depends_on(name: str, target: str, resolved: dict[str, tuple[type, str, BaseModel]]) -> bool:
        stack, seen = [name], set()
        while stack:
            n = stack.pop()
            if n == target and n != name:
                return True
            if n in seen or n not in resolved:
                continue
            seen.add(n)
            stack.extend(resolved[n][0].inputs)
        return False

    # -- running -------------------------------------------------------------

    def run(
        self, to: str | None = None, from_stage: str | None = None, frames: tuple[int, int] | None = None
    ) -> RunReport:
        """Run up to ``to``; ``from_stage`` forces it and the stages after it; ``frames`` (a, b) makes
        that a range re-run (see "Range re-runs")."""
        with stat_scope():
            return self._run(to, from_stage, frames)

    def _run(self, to: str | None, from_stage: str | None, frames: tuple[int, int] | None) -> RunReport:
        target = to or DEFAULT_TARGET
        if frames is not None and from_stage is None:
            raise RunError("--frames needs --from <stage> (the first stage to re-run for those frames)")
        needed = self.needed(target)
        forced = self._forced(needed, from_stage, target)
        self._frames, self._ranged, self._replaced = frames, set(), {}
        self._run_thread = threading.current_thread()
        if frames is not None:  # inside the shot timeline (the first forced stage's stored frame numbers)
            for s in sorted(forced, key=STAGE_ORDER.index):
                if not self.shot.has_result(s) or (self.shot.meta(s) or {}).get("has_result") is False:
                    continue
                with contextlib.suppress(ProjectError):
                    lo, hi = _frame_bounds(self.shot.load_result(s))
                    if not (lo <= frames[0] and frames[1] <= hi):
                        raise RunError(
                            f"--frames {frames[0]}:{frames[1]}: shot {self.shot.id} has frames "
                            f"{lo}..{hi - 1} (b is exclusive)"
                        )
                    break
        self.shot.project.check_source()
        self.check_edits(target)  # a lax edit file stops the run before any stage runs
        # Validate every stage's settings before running anything (fail fast on typos).
        resolved = {}
        for name in needed:
            cls, impl = self.resolve(name)
            resolved[name] = (cls, impl, self.params_for(name, cls))
        for name in needed:  # the code this run executes, before anything runs or is imported lazily
            self._snapshot(name, resolved[name][0])
        report = RunReport(self.shot.id, "ok")
        extra = {"frames": list(frames)} if frames is not None else {}
        forced_list = sorted(forced, key=STAGE_ORDER.index)
        self._emit("plan", target=target, stages=needed, forced=forced_list, **extra)
        with self.shot_lock():
            self._tidy_work()
            done: set[str] = set()
            for attempt in (0, 1):
                # the re-level pass: stages already run (re)run only when their key changed
                outcome = self._run_pass(needed, forced - done, resolved, report, attempt, done)
                if outcome != "relevel":
                    break
            if outcome == "stop":
                return report
        if (rv := self._pending_review) is not None:  # nothing after tracking needed to run: a warning
            self._emit_review(rv, paused=False)
        self._emit("done", outcome="ok")
        return report

    def _pause_for_review(self, name: str, impl: str, key: str, report: RunReport) -> str:
        """Stop before ``name`` (it needs tracking) until the tracking hints are looked at."""
        rv = self._pending_review
        assert rv is not None
        report.outcome = "needs_review"
        report.review = rv
        report.message = format_review(rv, self.shot.id, paused=True)
        report.stages.append(StageStatus(name, impl, "blocked", "waiting for the tracking review", key))
        self._emit_review(rv, paused=True)
        return "stop"

    def _stop_cancelled(self, name: str, impl: str, report: RunReport, *, running: bool = False) -> str:
        why = "cancelled while running: work dropped, previous result kept" if running else "cancelled"
        report.stages.append(StageStatus(name, impl, "cancelled", why))
        report.outcome = "cancelled"
        report.message = f"cancelled ({'during' if running else 'before'} {name})"
        self._emit("cancelled", stage=name, message=report.message)
        return "stop"

    def _range_for(
        self, name: str, cls: type, impl: str, parts: dict[str, Any], meta: dict[str, Any] | None,
        edits: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str]:  # fmt: skip
        """(range plan, "") when ``name`` can be range-run on its stored result, else (None, why not)."""
        assert self._frames is not None
        if not getattr(cls, "supports_range", False):
            return None, "no range support"
        if not meta or meta.get("has_result") is False or "output_hash" not in meta:
            return None, "no previous result"
        if not self._result_intact(name, meta):
            return None, "previous result missing or damaged"
        old = meta.get("key_parts") or {}
        diff = [k for k in RANGE_SAME_PARTS if canonical_json(old.get(k)) != canonical_json(parts.get(k))]
        if diff:
            return None, f"previous result differs in {', '.join(PART_LABELS.get(k, k) for k in diff)}"
        moved = []
        for i, h in parts["inputs"].items():
            was = (old.get("inputs") or {}).get(i)
            if was == h:
                continue
            if i in self._ranged and was == self._replaced.get(i):
                continue  # made from the result that the range re-run of i just replaced
            moved.append(i)
        if moved:
            return None, f"upstream {', '.join(moved)} changed outside the range"
        snap, prev_code = self._snapshot(name, cls), meta.get("code")
        if not (snap and isinstance(prev_code, dict) and prev_code.get("hash") == snap.get("hash")):
            return None, "previous result made by other code"
        if canonical_json(old.get("params")) != canonical_json(parts.get("params")):
            return None, "params differ from the previous result's"
        a, b = self._frames
        try:
            prev = self._load_input(name, name)
        except RunError as e:
            return None, str(e)
        lo, hi = _frame_bounds(prev)
        if not (lo <= a and b <= hi):
            return None, f"frames {a}:{b} outside its frames {lo}..{hi - 1}"
        if why := self._edits_outside(name, meta, old, parts, edits, (a, b), (lo, hi)):
            return None, why
        return {"frames": [a, b], "previous": meta, "result": prev}, ""

    def _edits_outside(
        self, name: str, meta: dict[str, Any], old: dict[str, Any], parts: dict[str, Any],
        edits: dict[str, Any], rng: tuple[int, int], shot: tuple[int, int],
    ) -> str:  # fmt: skip
        """Why the edits changed outside [a, b) = ``rng`` since the previous result ('' when only
        inside), compared frame by frame on the shot's frames ``shot`` (see ``_changes_inside``)."""
        a, b = rng
        old_h = {k: v for k, v in (old.get("edits") or {}).items() if v is not None}
        new_h = parts.get("edits") or {}
        changed = [e for e in sorted(set(old_h) | set(new_h)) if old_h.get(e) != new_h.get(e)]
        if not changed:
            return ""
        try:
            used = read_json(self.shot.stage_dir(name) / EDITS_USED)
        except (FileNotFoundError, ValueError):
            used = None
        for e in changed:
            if e in old_h and not (isinstance(used, dict) and e in used):
                return f"edits/{e}.json changed and the previous result's copy of it is missing"
            if not _changes_inside(used.get(e) if isinstance(used, dict) else None, edits.get(e), rng, shot):
                return f"edits/{e}.json changed outside frames {a}:{b}"
        return ""

    def _run_pass(
        self,
        needed: list[str],
        forced: set[str],
        resolved: dict[str, tuple[type, str, BaseModel]],
        report: RunReport,
        attempt: int,
        done: set[str],
    ) -> str:
        """One pass over the stages: "ok", "stop" (failed / needs a pick; ``report`` says which) or
        "relevel" (pose3d asked for a camera re-level: run again, once)."""
        hashes: dict[str, str] = {}
        self._pending_review = None
        camera_relevel: dict[str, Any] | None = None
        camera_base_key: str | None = None
        for name in needed:
            cls, impl, params = resolved[name]
            if self.cancelled():
                return self._stop_cancelled(name, impl, report)
            edits = self._edits(name, cls)
            inputs = {i: hashes[i] for i in cls.inputs}
            relevel_meta: dict[str, Any] | None = None
            if name == "camera":
                params, camera_base_key, camera_relevel = self._camera_relevel(
                    cls, impl, params, inputs, edits
                )
                relevel_meta = camera_relevel
            parts = self.key_parts(name, cls, impl, params, inputs, edits)
            key = self._stage_key(name, cls, parts)
            meta = self.shot.meta(name)
            verdict, code_why = "stale", ""
            if name not in forced and self._fresh(name, meta, key):
                assert meta is not None
                verdict, code_why = self._code_verdict(name, cls, meta)
                accepted = verdict == "accept"
                if accepted:
                    meta = self._accept_code(name, cls, meta)
                    verdict = "fresh"
            if verdict not in ("stale", "stale_cached"):
                assert meta is not None
                try:
                    self._republish_missing(name, meta)
                except OSError as e:
                    report.stages.append(StageStatus(name, impl, "failed", str(e), key))
                    report.outcome, report.message = "failed", f"stage {name}: {e}"
                    self._emit("stage_failed", stage=name, message=report.message, log=None)
                    return "stop"
                extra = {"code_changed": code_why} if verdict == "stale_code" else {}
                if accepted:
                    extra = {"code_accepted": code_why}
                if (fr := _range_frames(meta)) is not None:
                    extra["range"] = fr
                self._emit("stage_skip", stage=name, impl=impl, **extra)
                why = _join(range_note(meta), self._with_advice(name, verdict, code_why))
                report.stages.append(StageStatus(name, impl, verdict, why, key, range=fr))
            elif not code_why and name not in forced and (adopted := self._adopt(name, key)) is not None:
                meta = adopted
                self._emit("stage_skip", stage=name, impl=impl, adopted=True)
                report.stages.append(StageStatus(name, impl, "fresh", "adopted a finished run", key))
            else:
                if (
                    self._pending_review is not None
                    and self.pause_on_hints
                    and self._depends_on(name, REVIEW_STAGE, resolved)
                ):
                    return self._pause_for_review(name, impl, key, report)
                if (moved := self._code_moved(name, cls)) is not None:
                    return self._stop_code_moved(name, impl, key, report, moved)
                cache_only = verdict == "stale_cached"
                rng: dict[str, Any] | None = None
                if name in forced:
                    reason = "forced (--from)"
                    if self._frames is not None:
                        rng, full_why = self._range_for(name, cls, impl, parts, meta, edits)
                        a, b = self._frames
                        reason += f", frames {a}:{b}" if rng else f", in full ({full_why})"
                else:
                    reason = code_why or self._stale_reason(name, meta, parts, cls)
                    reason += CACHED_NOTE if cache_only else ""
                if attempt:
                    reason = f"re-level pass; {reason}"
                start_extra = {"range": list(self._frames)} if rng and self._frames else {}
                self._emit("stage_start", stage=name, impl=impl, reason=reason, **start_extra)
                t0 = time.monotonic()
                try:
                    meta = self._execute(
                        name, cls, impl, params, edits, parts, key, reason, meta, relevel=relevel_meta,
                        force=name in forced,  # never for the re-level pass (it reuses worker caches)
                        cache_only=cache_only, rng=rng,
                    )
                except RunCancelled:
                    return self._stop_cancelled(name, impl, report, running=True)
                except CacheOnlyMiss as e:  # the code change needs model work: keep the stored result
                    assert meta is not None
                    why = f"{code_why}; needs a model re-run ({e}): --code-changes rerun or --from {name}"
                    report.stages.append(StageStatus(name, impl, "stale_code", why, key))
                    note = f"{code_why}; needs a model re-run"
                    self._emit("stage_skip", stage=name, impl=impl, code_changed=note)
                    hashes[name] = meta["output_hash"]
                    continue
                except RunError as e:
                    dt = time.monotonic() - t0
                    report.stages.append(StageStatus(name, impl, "failed", str(e), key, dt))
                    report.outcome, report.message = "failed", str(e)
                    report.log_path = str(e.log_path) if e.log_path else None
                    self._emit("stage_failed", stage=name, message=str(e), log=report.log_path)
                    return "stop"
                fr = _range_frames(meta) if rng else None
                if fr is not None:
                    self._ranged.add(name)
                    self._replaced[name] = rng["previous"]["output_hash"]  # type: ignore[index]
                elif rng is not None:
                    reason += " (the stage solved the whole shot)"
                report.stages.append(
                    StageStatus(name, impl, "ran", reason, meta["key"], meta["runtime_s"], range=fr)
                )
                report.exports.extend(meta.get("exports", []))
                done_extra = {"range": fr} if fr is not None else {}
                self._emit(
                    "stage_done", stage=name, runtime_s=meta["runtime_s"], qc=meta.get("qc"), **done_extra
                )
                done.add(name)
            hashes[name] = meta["output_hash"]
            if meta.get("needs_input") == "pick":
                img = find_prompt_image(self.shot.stage_dir(name))
                report.outcome = "needs_pick"
                report.prompt_image = str(img) if img else None
                report.stages[-1].state = "needs_pick"
                report.message = (
                    f"pick the fighters for shot {self.shot.id}: open "
                    f"{img or self.shot.stage_dir(name)}, then run "
                    f"'afv pick {self.shot.id} --fighters A,B [--extra N,...]' "
                    f"and 'afv run {self.shot.id}'"
                )
                self._emit("needs_pick", stage=name, prompt_image=report.prompt_image, message=report.message)
                return "stop"
            if name == REVIEW_STAGE:
                self._review_after(meta, report)
            if name == "pose3d" and camera_base_key is not None:
                if camera_relevel is not None:
                    report.relevel = camera_relevel  # applied (now or by an earlier run): never twice
                elif attempt == 0 and self._decide_relevel(
                    camera_base_key, key, report, resolved["camera"][2]
                ):
                    return "relevel"
        return "ok"

    # -- gravity re-level pass -------------------------------------------------

    def _read_relevel(self) -> dict[str, Any] | None:
        try:
            data = read_json(self.shot.dir / RELEVEL_FILE)
        except (FileNotFoundError, ValueError):
            return None
        return data if isinstance(data, dict) and isinstance(data.get("override"), dict) else None

    def _camera_relevel(
        self, cls: type, impl: str, params: BaseModel, inputs: dict[str, str], edits: dict[str, Any]
    ) -> tuple[BaseModel, str, dict[str, Any] | None]:
        """(camera params, base key without any re-level, the re-level record applied or None). A record
        applies only to the camera inputs/settings it was made for (``base_camera_key``)."""
        base_key = stage_key(self.key_parts("camera", cls, impl, params, inputs, edits))
        rec = self._read_relevel()
        if rec is None or rec.get("base_camera_key") != base_key:
            return params, base_key, None
        if rec.get("rule") != _relevel_rule():  # [Z, round 3] made by another decision rule: decide again
            return params, base_key, None
        if any(getattr(params, k, None) is not None for k in rec["override"]):  # the user's own wins
            return params, base_key, None
        if not all(k in cls.Params.model_fields for k in rec["override"]):
            return params, base_key, None
        return self.params_for("camera", cls, extra=rec["override"]), base_key, rec

    def _decide_relevel(
        self, camera_base_key: str, pose3d_key: str, report: RunReport, camera_params: BaseModel
    ) -> bool:
        """After a pose3d without re-level: ask the camera module whether its gravity needs one. Writes
        (or clears) ``relevel.json``; True = run camera and pose3d again with the override."""
        path = self.shot.dir / RELEVEL_FILE
        try:
            from afv.models.camera import relevel_override
        except ImportError:
            return False
        try:
            qc = read_json(self.shot.stage_dir("pose3d") / QC_FILE)
        except (FileNotFoundError, ValueError):
            qc = None
        override = relevel_override(qc if isinstance(qc, dict) else None)
        fields = type(camera_params).model_fields
        if override is not None and (
            not set(override) <= set(fields) or any(getattr(camera_params, k) is not None for k in override)
        ):
            override = None  # this camera implementation cannot take it, or the user set it
        if override is None:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()  # a record for other camera inputs/settings is stale
            return False
        grav = (qc or {}).get("gravity", {}) if isinstance(qc, dict) else {}
        rec = {
            "base_camera_key": camera_base_key,
            "rule": _relevel_rule(),  # [Z, round 3] its decision rule (other rules' records are dropped)
            "override": override,
            "trigger": {
                "pose3d_key": pose3d_key,
                "flags": ((qc or {}).get("summary") or {}).get("flags", []),
                "gravity": {k: v for k, v in grav.items() if isinstance(v, str | int | float | bool)},
            },
            "created": now_iso(),
        }
        atomic_write_json(path, rec)
        report.relevel = rec
        self._emit("relevel", override=override, flags=rec["trigger"]["flags"])
        log.info("camera re-level from pose3d gravity: %s", override)
        return True

    def _load_input(self, name: str, inp: str) -> Result:
        """A fresh copy of ``inp``'s stored result (never a shared in-memory object)."""
        meta = self.shot.meta(inp) or {}
        if meta.get("has_result") is False:
            raise RunError(f"stage {name} needs the result of {inp}, but {inp} produced none", stage=name)
        try:
            return self.shot.load_result(inp)
        except ProjectError as e:
            raise RunError(f"stage {name}: cannot load input {inp}: {e}", stage=name) from None

    def _execute(
        self,
        name: str,
        cls: type,
        impl: str,
        params: BaseModel,
        edits: dict[str, Any],
        parts: dict[str, Any],
        key: str,
        reason: str,
        old_meta: dict[str, Any] | None,
        relevel: dict[str, Any] | None = None,
        force: bool = False,
        cache_only: bool = False,
        rng: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one stage in a work folder and commit its result; ``rng`` (from ``_range_for``) makes it
        a range re-run: the key gets ``frame_range`` and ``previous``, ``key`` becomes its base key."""
        base_key: str | None = None
        full_parts, full_key = parts, key
        if rng is not None:
            base_key = key
            parts = {**parts, "frame_range": list(rng["frames"]), "previous": rng["previous"]["output_hash"]}
            key = stage_key(parts)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        work = self.shot.work_root / f"{name}-{stamp}-{uuid.uuid4().hex[:6]}"
        (work / "logs").mkdir(parents=True)
        started = now_iso()
        t0 = time.monotonic()

        def progress(frac: float, msg: str = "") -> None:
            if self.cancelled() and threading.current_thread() is self._run_thread:
                raise RunCancelled()  # a stage thread: stop the stage here (worker threads just report)
            self._emit(
                "progress", stage=name, frac=None if frac is None or frac < 0 else float(frac), message=msg
            )

        ctx = Ctx(
            shot_dir=self.shot.dir,
            work_dir=work,
            frame_range=tuple(rng["frames"]) if rng is not None else None,
            seed=self.seed,
            deterministic=self.deterministic,
            progress=progress,
            edits=edits,
            force=force,
            cache_only=cache_only,
        )
        result: Result | None = None
        out_hash = hash_obj({"no_result": key})
        error: BaseException | None = None
        tb = ""
        with _stage_log(work / "logs" / "stage.log"):
            try:
                inputs = {i: self._load_input(name, i) for i in cls.inputs}
                if rng is not None:  # the result being replaced (still the current version until commit)
                    ctx.previous_dir = self.shot.stage_dir(name)
                    ctx.previous = rng.pop("result", None) or self._load_input(name, name)
                log.info("stage %s/%s start (%s)", name, impl, reason)
                result = cls().run(ctx, inputs, params)
                if result is not None and not isinstance(result, Result):
                    raise StageError(f"returned {type(result).__name__}, expected a Result or None")
                if result is not None:
                    save_result(result, work / RESULT_FILE)
                    out_hash = npz_hash(work / RESULT_FILE)  # what consumers will load
            except BaseException as e:  # handled below, after the stage log is closed
                error, tb = e, traceback.format_exc()
                log.error("stage %s failed: %s", name, e, extra={"afv_reported": True})
        if isinstance(error, RunCancelled) or self.cancelled():  # also a worker killed by the canceller
            shutil.rmtree(work, ignore_errors=True)
            raise RunCancelled()
        if isinstance(error, KeyboardInterrupt | SystemExit):
            self._keep_failed(work, name, "interrupted\n")
            raise error
        if isinstance(error, CacheOnlyMiss):  # not a failure: nothing of this run is kept
            shutil.rmtree(work, ignore_errors=True)
            raise error
        if error is not None:
            if isinstance(error, StageError | RunError | ConfigError | ProjectError):
                msg = str(error)
            else:
                msg = f"crashed with {type(error).__name__}: {error}"
            self._fail(work, name, impl, msg, tb, getattr(error, "log_path", None))
        runtime = time.monotonic() - t0
        if self.cancelled():  # finished, but cancelled meanwhile: keep the previous result
            shutil.rmtree(work, ignore_errors=True)
            raise RunCancelled()
        shutil.rmtree(work / "tmp", ignore_errors=True)
        if rng is not None and _qc_range(work / QC_FILE) is None:  # it solved the whole shot: a full result
            rng, parts, key, base_key = None, full_parts, full_key, None
        if edits:  # what a later range re-run compares its edits with
            atomic_write_json(work / EDITS_USED, edits)
        publish = _export_files(name, work)  # {path in stage folder: path under exports/}
        try:
            self._publish(work, publish, old_meta)
        except OSError as e:
            self._fail(work, name, impl, f"could not publish exports: {e}", "", None)
        meta = {
            "stage": name,
            "impl": impl,
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "version": str(cls.version),
            "env": getattr(cls, "env", ""),
            "weight_hash": str(getattr(cls, "weight_hash", "")),
            "key": key,
            "key_parts": parts,
            **({"base_key": base_key, "range": self._range_record(name, rng, started)} if rng else {}),
            "code": self._snapshot(name, cls),  # the code that ran (see "Code changes")
            "code_policy": self.code_policy(name, cls),
            "params": params_to_json(params),
            "inputs": parts["inputs"],
            "edits": parts["edits"],
            "output_hash": out_hash,
            "has_result": result is not None,
            "result_kind": result.kind if result is not None else None,
            "num_frames": result.num_frames if result is not None else None,
            "needs_input": "pick" if isinstance(result, Tracks) and not result.fighters else None,
            "qc": _read_qc(work / QC_FILE),
            "licences": [dataclasses.asdict(r) for r in getattr(cls, "licences", ())],
            "exports": sorted(publish.values()),
            "published_from": dict(sorted(publish.items())),
            "reason": reason,
            "relevel": relevel,
            "started": started,
            "finished": now_iso(),
            "runtime_s": round(runtime, 3),
            "afv_version": afv.__version__,
            "python": sys.version.split()[0],
            "host": platform.node(),
        }
        if rng is not None:  # what the stage did with the range (its qc.json "range")
            meta["range"]["stage"] = _qc_range(work / QC_FILE)
        if relevel and isinstance(meta["qc"], dict):  # visible in the stage's QC summary too
            meta["qc"] = {**meta["qc"], "relevel": {"override": relevel.get("override"),
                                                    "trigger_flags": relevel.get("trigger", {}).get("flags")}}
        meta["files"] = [*sorted(str(p.relative_to(work)) for p in work.rglob("*") if p.is_file()), META_FILE]
        fsync_tree(work)
        atomic_write_json(work / META_FILE, meta)
        self._commit(work, name, meta)
        return meta

    def _range_record(self, name: str, rng: dict[str, Any], started: str) -> dict[str, Any]:
        """meta["range"] of a range re-run: its frames, the result it replaced, the full run the chain
        of range re-runs started from, and every range since (oldest first)."""
        prev = rng["previous"]
        prev_rg = prev.get("range") if isinstance(prev.get("range"), dict) else None
        version = self.shot.stage_dir(name).name
        base = (prev_rg or {}).get("base") or {
            "version": version, "key": prev.get("key"), "output_hash": prev.get("output_hash"),
            "finished": prev.get("finished"),
        }  # fmt: skip
        history = [*((prev_rg or {}).get("history") or [])]
        history.append({"frames": list(rng["frames"]), "started": started})
        return {
            "frames": list(rng["frames"]),
            "previous_output_hash": prev.get("output_hash"),
            "previous_key": prev.get("key"),
            "previous_version": version,
            "base": base,
            "history": history[-RANGE_HISTORY_MAX:],
        }

    def _fail(self, work: Path, name: str, impl: str, msg: str, tb: str, worker_log: Any) -> None:
        kept = self._keep_failed(work, name, f"{msg}\n\n{tb}" if tb else f"{msg}\n")
        if worker_log:  # a worker log inside the work folder moved with it
            with contextlib.suppress(ValueError):
                worker_log = kept / Path(worker_log).relative_to(work)
        raise RunError(
            f"stage {name} ({impl}) failed: {msg}\n(logs: {kept / 'logs'})",
            stage=name,
            log_path=Path(worker_log) if worker_log else kept / "logs",
        ) from None

    def _commit(self, work: Path, name: str, meta: dict[str, Any]) -> Path:
        root = self.shot.stage_root(name)
        try:
            dst = commit_version(work, root)
        except OSError as e:
            with contextlib.suppress(OSError):
                atomic_write_json(work / COMMIT_PENDING, {"key": meta["key"], "error": str(e)})
            kept = self._keep_failed(work, name, f"finished, but could not store the result: {e}\n")
            raise RunError(
                f"stage {name}: finished, but could not store its result in {root}: {e}\n"
                f"(kept in {kept}; the next 'afv run' adopts it without re-running)",
                stage=name,
                log_path=kept / "logs",
            ) from None
        left = prune_versions(root, dst)
        if left:
            log.warning("could not delete old results (in use?): %s", ", ".join(str(p) for p in left))
        log.debug("stage %s stored in %s", name, dst)
        return dst

    def _adopt(self, name: str, key: str) -> dict[str, Any] | None:
        """Commit a kept run of ``name`` that finished with ``key`` but could not be stored."""
        root = self.shot.work_root
        if not root.is_dir():
            return None
        for d in sorted(root.iterdir(), reverse=True):
            pending = d / COMMIT_PENDING
            if not (d.is_dir() and pending.is_file()):
                continue
            try:
                meta = read_json(d / META_FILE)
            except (OSError, ValueError):
                continue
            if meta.get("stage") != name or not self._key_matches(meta, key):
                continue
            pending.unlink()
            with contextlib.suppress(FileNotFoundError):
                (d / FAILED_MARKER).unlink()
            with contextlib.suppress(OSError):
                self._publish(d, meta.get("published_from", {}), self.shot.meta(name))
            self._commit(d, name, meta)
            return meta
        return None

    def _publish(self, src_dir: Path, publish: dict[str, str], old_meta: dict[str, Any] | None) -> None:
        """Copy export files to the shot's exports/ and remove what the previous result published."""
        exports = self.shot.exports_dir
        for src_rel, dst_rel in publish.items():
            try:
                atomic_copy(src_dir / src_rel, exports / dst_rel)
            except OSError as e:
                raise OSError(
                    e.errno, f"cannot write {exports / dst_rel}: {e.strerror or e} (open in another program?)"
                ) from e
        for old in (old_meta or {}).get("exports", []):
            if old not in publish.values():
                try:
                    (exports / old).unlink(missing_ok=True)
                except OSError as e:
                    log.warning("could not remove stale export %s: %s", exports / old, e)

    def _republish_missing(self, name: str, meta: dict[str, Any]) -> None:
        sdir = self.shot.stage_dir(name)
        for src_rel, dst_rel in meta.get("published_from", {}).items():
            if not (self.shot.exports_dir / dst_rel).is_file() and (sdir / src_rel).is_file():
                atomic_copy(sdir / src_rel, self.shot.exports_dir / dst_rel)

    # -- housekeeping ----------------------------------------------------------

    def _keep_failed(self, work: Path, name: str, text: str) -> Path:
        with contextlib.suppress(OSError):
            (work / "logs").mkdir(parents=True, exist_ok=True)
            atomic_write_text(work / "logs" / "error.txt", text)
        dst = self.shot.work_root / f"failed-{work.name}"
        try:
            os.rename(work, dst)
        except OSError:
            dst = work
            with contextlib.suppress(OSError):
                (work / FAILED_MARKER).write_text(text[:2000])
        old = sorted(self.shot.work_root.glob(f"failed-{name}-*"), key=lambda p: p.name)
        for p in old[:-KEEP_FAILED]:
            if not (p / COMMIT_PENDING).exists():
                shutil.rmtree(p, ignore_errors=True)
        return dst

    def _tidy_work(self) -> None:
        """Remove work folders left by crashed runs (we hold the shot lock, so none is live).
        Kept failures (renamed or marked) and results waiting to be adopted stay."""
        root = self.shot.work_root
        if not root.is_dir():
            return
        for p in root.iterdir():
            if not p.is_dir() or p.name.startswith("failed-"):
                continue
            if (p / FAILED_MARKER).exists() or (p / COMMIT_PENDING).exists():
                continue
            shutil.rmtree(p, ignore_errors=True)

    @contextlib.contextmanager
    def shot_lock(self) -> Iterator[None]:
        """Exclusive lock on the shot, re-entrant within this Runner. Commands that change shot.json
        or edits and then run take it first, so nothing is saved while another afv process (or another
        request of ``afv serve``) is running the shot. The lock file is ``<afv home>/locks/shot-<folder
        identity>.lock`` on the Linux side (``afv.core.project.folder_identity``: the same for every
        spelling of the shot's path, unlike a lock inside the shot on a Windows drive), plus
        ``.work/.lock`` as before. CLI and ``afv serve`` must see the same afv home (``$AFV_HOME``,
        default ``~/afv``) to exclude each other; a lock file names its shot folder, and lock files of
        shots that no longer exist are removed (once per process, when a lock is taken)."""
        if self._lock_depth:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return
        try:
            import fcntl
        except ImportError:  # not POSIX: no cross-process lock
            yield
            return
        self.shot.work_root.mkdir(parents=True, exist_ok=True)
        paths = [self.shot.work_root / ".lock"]
        with contextlib.suppress(OSError):
            ldir = afv_home() / "locks"
            ldir.mkdir(parents=True, exist_ok=True)
            paths.insert(0, ldir / f"shot-{folder_identity(self.shot.dir)}.lock")
        fds: list[int] = []
        try:
            for p in paths:
                fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o666)
                fds.append(fd)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    msg = f"shot {self.shot.id} is already being run by another afv process"
                    raise RunError(msg) from None
                except OSError:
                    continue  # filesystem without flock: carry on with the other lock
                if p.name.startswith("shot-"):  # which shot this lock is for (stale-lock clean-up)
                    with contextlib.suppress(OSError):
                        os.ftruncate(fd, 0)
                        os.pwrite(fd, f"{self.shot.dir}\n".encode(), 0)
                    _clean_stale_locks(p.parent, keep=p)
            self._lock_depth = 1
            try:
                yield
            finally:
                self._lock_depth = 0
        finally:
            for fd in fds:
                os.close(fd)


_LOCKS_CLEANED: list[bool] = []
STALE_UNNAMED_LOCK_S = 7 * 24 * 3600  # a lock file naming no shot is removed after this long


def _clean_stale_locks(ldir: Path, keep: Path) -> None:
    """Remove lock files whose shot folder is gone (or is another folder now), once per process. A
    file is removed only while this process holds its lock, so nobody is running that shot."""
    if _LOCKS_CLEANED:
        return
    _LOCKS_CLEANED.append(True)
    import fcntl

    for f in ldir.glob("shot-*.lock"):
        if f == keep:
            continue
        try:
            named = f.read_text(encoding="utf-8", errors="replace").strip()
            if named:
                try:
                    if f.name == f"shot-{folder_identity(Path(named))}.lock":
                        continue  # still that shot's lock
                except OSError:
                    pass  # the shot folder is gone
            elif time.time() - f.stat().st_mtime < STALE_UNNAMED_LOCK_S:
                continue
            fd = os.open(f, os.O_RDWR)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            f.unlink()
        except OSError:
            pass  # held (in use) or already gone
        finally:
            os.close(fd)


def _export_files(stage: str, work: Path) -> dict[str, str]:
    """Files to publish to the shot's exports/: {path in the stage folder: path under exports/}."""
    out = {
        str(p.relative_to(work)): str(p.relative_to(work / "exports"))
        for p in (work / "exports").rglob("*")
        if p.is_file()
    }
    if stage == "export":
        for p in work.iterdir():
            if p.is_file() and p.suffix.lower() in EXPORT_SUFFIXES:
                out[p.name] = p.name
    return out


EDITS_USED = "edits_used.json"  # in a result folder: the parsed edits the stage ran with


def _frame_bounds(res: Result) -> tuple[int, int]:
    """(first frame, last frame + 1) of a result on the shot timeline."""
    f = getattr(res, "frames", None)
    if f is None or len(f) == 0:
        return 0, int(getattr(res, "num_frames", 0) or 0)
    return int(f[0]), int(f[-1]) + 1


def _pin_radius() -> int | None:
    try:
        from afv.motion.pins import DEFAULT_RADIUS
    except ImportError:
        return None
    return int(DEFAULT_RADIUS)


_SPAN_PLACE = ("role", "start", "end", "foot", "channels", "side")  # where a span is, not what it says


def _span_targets(item: dict[str, Any]) -> list[tuple[str, str]] | None:
    """What a span of an edit file covers: a foot, contact channels (contacts) or a hand side (hands)."""
    if "foot" in item:
        return [("foot", str(item["foot"]))]
    if "channels" in item and isinstance(item["channels"], list):
        return [("channel", str(c)) for c in item["channels"]]
    if "side" in item:
        side = str(item["side"])
        return [("side", "left"), ("side", "right")] if side == "both" else [("side", side)]
    return None


def _coverage(data: Any, lo: int, hi: int) -> tuple[dict[tuple[str, ...], list[Any]], dict[str, Any]] | None:
    """An edit file rasterised on the shot's frames [lo, hi): ({(kind, role, target...): value per
    frame}, other top-level keys), or None when its form is not understood. Spans (contacts: per role
    and foot/channel; hands: per role and side) mark their frames with what they say (``state`` and
    any other key; later spans win); a pin
    marks its frame +- its radius (default ``afv.motion.pins.DEFAULT_RADIUS``) per role, clipped to the
    shot. Two versions that rasterise alike on a frame mean the same there."""
    n = max(hi - lo, 0)
    cov: dict[tuple[str, ...], list[Any]] = {}
    other: dict[str, Any] = {}
    if not data:
        return cov, other
    if not isinstance(data, dict):
        return None
    items: list[tuple[str, dict[str, Any]]] = []
    for k, v in data.items():
        if k in ("spans", "pins") and isinstance(v, list):
            items += [(k, it) for it in v]
        elif k in ("A", "B") and isinstance(v, list):  # contacts' older {"A": [spans], ...} form
            items += [("spans", {**it, "role": k} if isinstance(it, dict) else it) for it in v]
        else:
            other[k] = v
    radius = _pin_radius()
    try:
        for kind, it in items:
            if not isinstance(it, dict):
                return None
            role = str(it.get("role", ""))
            if kind == "pins":
                r = it.get("radius")
                r = radius if r is None else int(r)
                if r is None:
                    return None
                f = int(it["frame"])
                row = cov.setdefault(("pin", role), [()] * n)
                pin = canonical_json(it)
                for i in range(max(f - r, lo), min(f + r + 1, hi)):
                    row[i - lo] = (*row[i - lo], pin)
                continue
            targets = _span_targets(it)
            if targets is None or it.get("end") is None:
                return None
            s0, s1 = max(int(it["start"]), lo), min(int(it["end"]), hi)
            value = canonical_json({k: v for k, v in it.items() if k not in _SPAN_PLACE})  # e.g. state
            for t in targets:
                row = cov.setdefault(("span", role, *t), [None] * n)
                for i in range(s0, s1):
                    row[i - lo] = value
    except (KeyError, TypeError, ValueError):
        return None
    for key, row in cov.items():
        if key[0] == "pin":
            cov[key] = [tuple(sorted(x)) for x in row]
    return cov, other


def _changes_inside(old: Any, new: Any, rng: tuple[int, int], shot: tuple[int, int]) -> bool:
    """Two versions of an edit file differ only on frames inside [a, b) = ``rng``: compared frame by
    frame (``_coverage``) on the shot's frames ``shot``, so painting over or clearing part of a span
    counts only where the meaning changed. Other top-level keys must be equal."""
    a, b = rng
    lo, hi = shot
    x, y = _coverage(old, lo, hi), _coverage(new, lo, hi)
    if x is None or y is None:
        return canonical_json(old) == canonical_json(new)
    (cx, ox), (cy, oy) = x, y
    if canonical_json(ox) != canonical_json(oy):
        return False
    n = max(hi - lo, 0)
    for key in set(cx) | set(cy):
        empty: list[Any] = [() if key[0] == "pin" else None] * n
        rx, ry = cx.get(key, empty), cy.get(key, empty)
        for i in range(n):
            if rx[i] != ry[i] and not a <= lo + i < b:
                return False
    return True


def _join(*parts: str) -> str:
    return "; ".join(p for p in parts if p)


def _range_frames(meta: dict[str, Any] | None) -> list[int] | None:
    rg = (meta or {}).get("range")
    fr = rg.get("frames") if isinstance(rg, dict) else None
    return [int(fr[0]), int(fr[1])] if isinstance(fr, list | tuple) and len(fr) == 2 else None


def range_note(meta: dict[str, Any] | None) -> str:
    """'range re-run of frames a:b (...)' for a result made by a range re-run, else ''."""
    fr = _range_frames(meta)
    if fr is None:
        return ""
    assert meta is not None
    rg = meta["range"]
    n = len(rg.get("history") or []) or 1
    base = rg.get("base") or {}
    since = f" since the full run of {base['finished']}" if base.get("finished") else ""
    return f"range re-run of frames {fr[0]}:{fr[1]} ({n} range re-run{'s' if n != 1 else ''}{since})"


def _qc_range(path: Path) -> dict[str, Any] | None:
    """A range stage's own record of a range re-run: qc.json ``"range"`` (top level or in its summary,
    e.g. ``{"frames", "solved", "changed"}``); None when it has none (e.g. it fell back to a full solve
    because the previous result did not match)."""
    try:
        qc = read_json(path)
    except (FileNotFoundError, ValueError):
        return None
    if not isinstance(qc, dict):
        return None
    summary = qc.get("summary") if isinstance(qc.get("summary"), dict) else {}
    rg = qc.get("range", summary.get("range"))
    return rg if isinstance(rg, dict) else None


def _read_qc(path: Path) -> Any:
    """QC summary for meta.json: qc.json's "summary" if present, else the whole file when small,
    else its top-level scalars."""
    try:
        qc = read_json(path)
    except (FileNotFoundError, ValueError):
        return None
    if isinstance(qc, dict) and "summary" in qc:
        return qc["summary"]
    if len(json.dumps(qc, default=str)) <= QC_META_MAX_BYTES:
        return qc
    if isinstance(qc, dict):
        return {k: v for k, v in qc.items() if isinstance(v, str | int | float | bool) or v is None}
    return None


_LOG_LEVEL_LOCK = threading.Lock()


class _ThisThread(logging.Filter):
    """Only records logged by one thread (runs of other shots may log meanwhile, e.g. in afv serve)."""

    def __init__(self) -> None:
        super().__init__()
        self.ident = threading.get_ident()

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread == self.ident


@contextlib.contextmanager
def _stage_log(path: Path) -> Iterator[None]:
    """The stage's INFO+ records of this thread into ``path``. The "afv" logger is set to INFO once
    (never restored per stage: parallel runs share it); its other handlers keep their own levels."""
    logger = logging.getLogger("afv")
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.addFilter(_ThisThread())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    with _LOG_LEVEL_LOCK:
        if logger.getEffectiveLevel() > logging.INFO:
            logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
