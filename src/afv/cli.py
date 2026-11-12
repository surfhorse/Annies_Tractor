"""``afv`` command line.

A long video (shot browser workflow; the Blender add-on drives the same steps through ``afv serve``)::

    afv init fight.mp4 --project fight.afv --detect # hash + probe, detect cuts: one candidate shot each
    cd fight.afv
    afv shots scan                                  # numbered people + thumbnail per shot (one GPU job)
    afv shots list --likely                         # candidates where two fighters are likely visible
    afv shots set 12 --start 3:02.5 --end 3:09      # trim (in/out points); --speed 0.25 for a replay
    afv shots queue 12,14-16                        # choose shots (afv shots skip for the others)
    afv pick 12                                     # the scan's numbered people (no GPU)
    afv pick 12 --fighters 1,0                      # numbers from the scan
    afv batch                                       # every queued shot, unattended

One shot::

    afv init fight.mp4 --project fight.afv          # hash + probe the source
    cd fight.afv
    afv shots add --start 31.2 --end 36.8           # a hand-marked shot is queued
    afv run 1                                       # ingest + detection, then stops: pick needed
    afv pick 1 --fighters 1,0 --extra 2             # numbers from the prompt image
    afv run 1                                       # tracking through export (cached stages skipped)
    afv pick 1 --frame 140                          # numbered people on any frame (QC hints name them)
    afv pick 1 --prompt 140:4:2 --label referee     # re-prompt: object 4 = detection 2 on frame 140
    afv export 1 --preset unreal --fps 60           # another engine / frame rate (saved to the shot)

Every command takes ``--project DIR`` (default: ``$AFV_PROJECT`` or the project containing the current
folder) and ``--json`` (one JSON object per line instead of text). Exit codes: 0 ok, 1 error,
3 waiting for the user (fighters not picked, tracking QC hints to review: fix or --yes, or a stage's
code changed since afv started: run again).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import logging
import math
import os
import re
import signal
import sys
import threading
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer

from afv.core import stage as stage_mod
from afv.core.config import ConfigError, deep_merge, parse_set
from afv.core.project import Project, ProjectError, Shot, now_iso
from afv.core.provenance import format_provenance, licence_problems, project_provenance
from afv.core.review import format_review
from afv.core.runner import (
    DEFAULT_TARGET,
    RunError,
    Runner,
    RunReport,
    find_prompt_image,
    load_stage_modules,
    parse_frames,
)
from afv.core.stage import STAGE_ORDER, StageError

EXIT_ERROR = 1
EXIT_NEEDS_INPUT = 3

app = typer.Typer(
    help="afv: fight footage to game-ready 3D animation.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
shots_app = typer.Typer(
    help="Shot list: detect cuts (candidate shots), scan them, trim, queue or skip, add by hand, list.\n\n"
    "States: candidate (found by detect, not chosen), queued (afv batch runs it), skipped. A shot's state "
    "changes only when you change it.",
    no_args_is_help=True,
)
app.add_typer(shots_app, name="shots")


class StateChoice(StrEnum):
    candidate = "candidate"
    queued = "queued"
    skipped = "skipped"


StatesOpt = Annotated[
    list[StateChoice] | None,
    typer.Option("--state", help="Only shots in this state (repeatable)."),
]


def install_signal_handlers() -> None:
    """SIGTERM/SIGHUP raise SystemExit so the normal clean-up runs: worker process groups are
    stopped (with the GPU lock held until they are gone) and the failed work folder is kept."""
    if threading.current_thread() is not threading.main_thread():
        return

    def handler(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, handler)


class _NotReported(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "afv_reported", False)  # the CLI prints those itself


def _setup_logging() -> None:
    logger = logging.getLogger("afv")
    if any(getattr(h, "_afv_cli", False) for h in logger.handlers):
        return
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.WARNING)
    h.setFormatter(logging.Formatter("warning: %(message)s"))
    h.addFilter(_NotReported())
    h._afv_cli = True  # type: ignore[attr-defined]
    logger.addHandler(h)


@app.callback()
def _main() -> None:
    """afv: fight footage to game-ready 3D animation."""
    install_signal_handlers()
    _setup_logging()


ProjectOpt = Annotated[
    Path | None,
    typer.Option(
        "--project", "-p", envvar="AFV_PROJECT", help="Project folder (default: the one containing cwd)."
    ),
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable output: one JSON object per line.")]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", help="Stage setting, e.g. smooth.strength=0.6 (saved to the shot; =null resets)."),
]


class CameraMode(StrEnum):
    vipe = "vipe"
    background = "background"
    static = "static"


class CodeChanges(StrEnum):
    report = "report"
    rerun = "rerun"
    accept = "accept"


CodeChangesOpt = Annotated[
    CodeChanges,
    typer.Option(
        "--code-changes",
        help="Model stages (GPU) whose code changed since their result: report = keep the result and show "
        "'stale (code)'; rerun = run them again (stage caches reused); accept = record the current code as "
        "theirs (a refactor that changes nothing). CPU stages always re-run on a code change (accept "
        "does not apply to them).",
    ),
]
YesOpt = Annotated[
    bool,
    typer.Option("--yes", "-y", help="Go on past tracking QC hints (recorded as reviewed for this result)."),
]


F = TypeVar("F", bound=Callable[..., Any])


def _handled(fn: F) -> F:
    """Turn expected errors into one readable line and exit code 1 (AFV_DEBUG=1 shows tracebacks)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (ProjectError, ConfigError, RunError, StageError, OSError) as e:
            if os.environ.get("AFV_DEBUG"):
                raise
            msg = str(e)
            if isinstance(e, OSError) and not isinstance(e, BrokenPipeError):
                where = f" ({e.filename})" if e.filename else ""
                msg = f"{e.strerror or e}{where}" if e.strerror else msg
            elif isinstance(e, BrokenPipeError):
                raise typer.Exit(EXIT_ERROR) from None
            with contextlib.suppress(OSError):
                if kwargs.get("json_out"):
                    typer.echo(json.dumps({"event": "error", "message": msg}))
                typer.echo(f"error: {msg}", err=True)
            raise typer.Exit(EXIT_ERROR) from None

    return wrapper  # type: ignore[return-value]


def _project(project: Path | None) -> Project:
    return Project(project) if project else Project.find()


def _emit(json_out: bool, obj: dict[str, Any], text: str | None = None) -> None:
    if json_out:
        typer.echo(json.dumps(obj, default=str))
    elif text is not None:
        typer.echo(text)


def parse_time(text: str) -> float:
    """Seconds from '62.5', '1:02.5' or '0:01:02.5'."""
    try:
        parts = [float(p) for p in text.strip().split(":")]
    except ValueError:
        raise ConfigError(f"bad time {text!r}; use seconds (62.5) or m:ss(.s) (1:02.5)") from None
    if not 1 <= len(parts) <= 3 or any(p < 0 or not math.isfinite(p) for p in parts):
        raise ConfigError(f"bad time {text!r}")
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return total


def _int_list(text: str | None, what: str) -> list[int]:
    if text is None or not text.strip():
        return []
    try:
        vals = [int(x) for x in text.replace(" ", "").split(",") if x != ""]
    except ValueError:
        raise ConfigError(
            f"{what}: expected comma-separated numbers from the prompt image, got {text!r}"
        ) from None
    if any(v < 0 for v in vals):
        raise ConfigError(f"{what}: numbers must be >= 0")
    if len(set(vals)) != len(vals):
        raise ConfigError(f"{what}: repeated number in {text!r}")
    return vals


# ---------------------------------------------------------------------------
# Progress output
# ---------------------------------------------------------------------------


class _Throttle:
    """At most ``rate`` progress lines per second per stage; a repeated message waits ``repeat_s``."""

    def __init__(self, rate: float = 4.0, repeat_s: float = 2.0):
        self.gap, self.repeat_s = 1.0 / rate, repeat_s
        self._last: dict[str, tuple[float, str]] = {}

    def allow(self, ev: dict[str, Any]) -> bool:
        stage, msg, now = ev.get("stage", ""), ev.get("message") or "", time.monotonic()
        last = self._last.get(stage)
        if last is not None:
            dt = now - last[0]
            if dt < self.gap or (msg == last[1] and dt < self.repeat_s):
                return False
        self._last[stage] = (now, msg)
        return True


class _Out:
    """stdout that stops writing (instead of failing the run) once the reader has gone away."""

    dead = False

    @classmethod
    def echo(cls, text: str) -> None:
        if cls.dead:
            return
        try:
            typer.echo(text)
        except BrokenPipeError:
            cls.dead = True
            with contextlib.suppress(OSError):
                os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())


class TextReporter:
    """Human progress lines; progress updates are throttled."""

    def __init__(self) -> None:
        self._throttle = _Throttle()

    def __call__(self, ev: dict[str, Any]) -> None:
        kind, stage = ev["event"], ev.get("stage", "")
        echo = _Out.echo
        if kind == "plan":
            forced = f" (forced from {ev['forced'][0]})" if ev.get("forced") else ""
            echo(f"shot {ev['shot']}: {' > '.join(ev['stages'])}{forced}")
        elif kind == "stage_skip":
            note = " (adopted a finished run)" if ev.get("adopted") else ""
            if ev.get("code_changed"):
                note = f" (result kept; {ev['code_changed']}; --code-changes rerun to re-run it)"
            elif ev.get("code_accepted"):
                note = f" (accepted: {ev['code_accepted']})"
            echo(f"  {stage:<13} up to date{note}")
        elif kind == "tracking_review":
            echo(ev.get("message") or "")
        elif kind == "stage_start":
            echo(f"  {stage:<13} running ({ev.get('impl')}; {ev.get('reason', '')})")
        elif kind == "progress":
            if not self._throttle.allow(ev):
                return
            frac, msg = ev.get("frac"), ev.get("message") or ""
            pct = f"{100 * frac:3.0f}% " if frac is not None else ""
            echo(f"  {stage:<13} {pct}{msg}")
        elif kind == "stage_done":
            echo(f"  {stage:<13} done in {ev.get('runtime_s', 0):.1f} s")
        elif kind == "stage_failed":
            echo(f"  {stage:<13} FAILED")
        elif kind == "relevel":
            echo(
                "  re-level      pose3d's gravity disagrees with the camera "
                f"({', '.join(ev.get('flags') or [])}): "
                "camera and pose3d run again, once"
            )


class JsonReporter:
    """One JSON object per line; progress events throttled like the text output."""

    def __init__(self) -> None:
        self._throttle = _Throttle()

    def __call__(self, ev: dict[str, Any]) -> None:
        if ev["event"] == "progress" and not self._throttle.allow(ev):
            return
        _Out.echo(json.dumps(ev, default=str))


def _reporter(json_out: bool) -> Callable[[dict[str, Any]], None]:
    return JsonReporter() if json_out else TextReporter()


def _finish(report: RunReport, json_out: bool) -> int:
    """Print the outcome of a run; return its exit code."""
    _emit(json_out, {"event": "result", **report.to_dict()})
    if not json_out:
        if report.outcome == "failed":
            typer.echo(f"error: {report.message}", err=True)
        elif report.outcome in ("needs_pick", "code_changed"):
            typer.echo(report.message)
        elif report.outcome != "needs_review":  # the tracking_review event printed its message
            for e in report.exports:
                typer.echo(f"  exported {e}")
    waiting = dict.fromkeys(("needs_pick", "needs_review", "code_changed"), EXIT_NEEDS_INPUT)
    return {"ok": 0, **waiting}.get(
        report.outcome, EXIT_ERROR
    )


def _settings_overrides(
    shot: Shot, camera: CameraMode | None, sets: list[str] | None, runner: Runner, persist: bool
) -> None:
    """Validate --set/--camera and save them in shot.json (so later runs keep them)."""
    params = parse_set(sets or [])
    runner.validate_settings(params)
    if camera is not None:
        runner.resolve("camera")  # fails clearly if that camera mode is not implemented
    if not persist or (not params and camera is None):
        return
    def change(info: Any) -> None:
        info.settings.params = deep_merge(info.settings.params, params)
        info.settings.params = {k: v for k, v in info.settings.params.items() if v}
        if camera is not None:
            info.settings.impl["camera"] = camera.value

    shot.modify(change)  # locked: a concurrent state change or trim is not lost


def _make_runner(
    shot: Shot,
    json_out: bool,
    *,
    camera: CameraMode | None = None,
    sets: list[str] | None = None,
    extra_params: dict[str, dict[str, Any]] | None = None,
    seed: int = 0,
    deterministic: bool = False,
    code_changes: CodeChanges = CodeChanges.report,
    pause_on_hints: bool = False,
    yes: bool = False,
    report: Callable[[dict[str, Any]], None] | None = None,
) -> Runner:
    params = parse_set(sets or [])
    if extra_params:
        params = deep_merge(params, extra_params)
    impl = {"camera": camera.value} if camera is not None else {}
    return Runner(
        shot, impl=impl, params=params, seed=seed, deterministic=deterministic,
        report=report or _reporter(json_out),
        code_changes=code_changes.value, pause_on_hints=pause_on_hints, yes=yes,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
@_handled
def init(
    video: Annotated[Path, typer.Argument(help="Source video.")],
    project: Annotated[
        Path | None, typer.Option("--project", "-p", help="New project folder (default: ./<video name>.afv).")
    ] = None,
    title: Annotated[str, typer.Option(help="Source title, for provenance.")] = "",
    url: Annotated[str, typer.Option(help="Source URL, for provenance.")] = "",
    no_probe: Annotated[bool, typer.Option("--no-probe", help="Skip probing the video.")] = False,
    rehash: Annotated[
        bool,
        typer.Option(
            "--rehash",
            help="Existing project: hash (and probe) the source again, e.g. "
            "after replacing or moving it. Every stage whose input changed re-runs.",
        ),
    ] = False,
    detect: Annotated[
        bool,
        typer.Option("--detect", help="Also detect cuts: a candidate shot per segment (afv shots detect)."),
    ] = False,
    min_confidence: Annotated[
        float, typer.Option(help="With --detect: ignore cuts weaker than this (0..1).", min=0, max=1)
    ] = 0.2,
    proxy: Annotated[
        bool,
        typer.Option(
            "--proxy",
            help="Also write media/proxy.mp4: the source at most 720 px tall with a short GOP, for scrubbing "
            "in Blender (minutes for a long video; made while cuts are detected).",
        ),
    ] = False,
    json_out: JsonOpt = False,
) -> None:
    """Create a project for one source video (hashes and probes it); --detect also finds the candidate
    shots (then: afv shots scan)."""
    if not video.is_file():
        raise ProjectError(f"video not found: {video}")
    root = project or Path.cwd() / f"{video.stem}.afv"
    probe_info: dict[str, Any] | None = None
    warning = ""
    if not no_probe:
        probe_info, warning = _probe(video)
    if rehash:
        proj = Project(root)
        old = proj.info.source.sha256
        src = proj.rehash(video, probe=probe_info)
        if warning:
            typer.echo(f"warning: {warning}", err=True)
        same = src.sha256 == old
        _emit(
            json_out,
            {"event": "rehash", "project": str(proj.root), "sha256": src.sha256, "changed": not same},
            f"re-hashed {src.path}\n  sha256 {src.sha256} "
            + ("(unchanged: results stay valid)" if same else "(changed: stages will re-run)"),
        )
        return
    proj = Project.create(root, video, title=title, url=url, probe=probe_info)
    if warning:
        typer.echo(f"warning: {warning}", err=True)
    info = proj.info
    nxt = (f"next: cd {proj.root} && afv shots scan" if detect
           else f"next: cd {proj.root} && afv shots detect   (or afv shots add --start S --end E)")
    _emit(
        json_out,
        {
            "event": "init",
            "project": str(proj.root),
            "sha256": info.source.sha256,
            "probe": info.source.probe,
        },
        f"created {proj.root}\n  source {info.source.path}\n  sha256 {info.source.sha256}"
        + ("" if detect or proxy else f"\n{nxt}"),
    )  # fmt: skip
    if detect or proxy:
        project_media(proj, detect=detect, proxy=proxy, min_confidence=min_confidence,
                      emit=lambda obj, text: _emit(json_out, obj, text))
        if not json_out:
            typer.echo(nxt)


def project_media(
    proj: Project,
    *,
    detect: bool,
    proxy: bool | str,
    min_confidence: float = 0.2,
    emit: Callable[[dict[str, Any], str | None], None] | None = None,
) -> dict[str, Any]:
    """Cut detection (candidate shots, media/cuts.json) and/or the 720p proxy, side by side. ``proxy``:
    True, False or "auto" (only for a source taller than 720 px). Returns {"detect": summary | None,
    "proxy": path | None}."""
    from afv.media.shots import make_source_proxy

    out: dict[str, Any] = {"detect": None, "proxy": None}
    if proxy == "auto":
        probe_info = proj.info.source.probe or {}
        proxy = int(probe_info.get("display_height") or probe_info.get("height") or 0) > 720
    err: list[BaseException] = []
    th = None
    if proxy:
        src = proj.source_path()

        def make() -> None:
            try:
                out["proxy"] = str(make_source_proxy(proj.media_dir, src, proj.info.source.sha256))
            except BaseException as e:  # reported after detection
                err.append(e)

        th = threading.Thread(target=make, daemon=True, name="proxy")
        th.start()
    try:
        if detect:
            out["detect"] = detect_into_project(proj, min_confidence, emit=emit)
    finally:
        if th is not None:
            th.join()
    if err:
        e = err[0]
        raise e if isinstance(e, ProjectError | OSError) else ProjectError(f"proxy failed: {e}")
    if out["proxy"] and emit is not None:
        emit({"event": "proxy", "path": out["proxy"]}, f"proxy {out['proxy']}")
    return out


def _probe(video: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        from afv.media import probe  # type: ignore[attr-defined]
    except ImportError:
        return None, "afv.media.probe is not available yet; video not probed"
    probe = getattr(probe, "probe", probe)  # a probe.py submodule exporting probe()
    if not callable(probe):
        return None, "afv.media.probe is not callable; video not probed"
    from afv.core.cache import to_jsonable

    try:
        try:
            info = probe(video, hash=False)  # afv hashes the file itself
        except TypeError:
            info = probe(video)
    except Exception as e:
        raise ProjectError(f"cannot read {video}: {type(e).__name__}: {e}") from None
    data = to_jsonable(info)
    if not isinstance(data, dict):
        data = to_jsonable(vars(info)) if hasattr(info, "__dict__") else {"value": data}
    return data, ""


@shots_app.command("add")
@_handled
def shots_add(
    start: Annotated[str, typer.Option(help="Start time in the source: seconds or m:ss.s.")],
    end: Annotated[str | None, typer.Option(help="End time (or give --duration).")] = None,
    duration: Annotated[str | None, typer.Option(help="Duration in seconds (or give --end).")] = None,
    speed: Annotated[
        float,
        typer.Option(
            help="Footage speed vs real time: 0.25 = a 4x slow-motion replay (exported at real time, ~4x "
            "shorter), 1.25 = fast/undercranked footage."
        ),
    ] = 1.0,
    notes: Annotated[str, typer.Option(help="Free text.")] = "",
    shot_id: Annotated[str | None, typer.Option("--id", help="Shot id (default: next number).")] = None,
    state: Annotated[
        StateChoice, typer.Option("--state", help="State of the new shot (hand-marked = chosen: queued).")
    ] = StateChoice.queued,
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Add a shot by hand (queued: afv batch runs it)."""
    proj = _project(project)
    t0 = parse_time(start)
    if (end is None) == (duration is None):
        raise ConfigError("give exactly one of --end or --duration")
    dur = parse_time(end) - t0 if end is not None else parse_time(duration or "0")
    shot = proj.add_shot(t0, dur, shot_id=shot_id, speed_factor=speed, notes=notes, state=state.value)
    i = shot.info
    _emit(
        json_out,
        {"event": "shot_added", **i.model_dump(mode="json")},
        f"added shot {i.id}: {i.start:.3f}-{i.end:.3f} s ({i.duration:.3f} s), {i.state}",
    )


@shots_app.command("detect")
@_handled
def shots_detect(
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Only print the shots found.")] = False,
    min_confidence: Annotated[float, typer.Option(help="Ignore cuts weaker than this (0..1).")] = 0.2,
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Detect cuts in the source and add one candidate shot per segment (existing shots are kept; a segment
    mostly covered by one is not added). Writes media/cuts.json (every cut found, for the shot browser).

    Next: afv shots scan (numbered people per shot), then afv shots queue IDS."""
    proj = _project(project)
    detect_into_project(proj, min_confidence, dry_run=dry_run, emit=lambda o, t: _emit(json_out, o, t))


def detect_into_project(
    proj: Project,
    min_confidence: float = 0.2,
    *,
    dry_run: bool = False,
    emit: Callable[[dict[str, Any], str | None], None] | None = None,
) -> dict[str, Any]:
    """``afv shots detect``: candidate shots for the segments between cuts, and ``media/cuts.json``;
    returns the ``detect_done`` summary. ``emit(event, text)`` gets each event."""
    say = emit or (lambda obj, text: None)
    t0 = time.monotonic()
    segs, analysis = _detect_segments(proj.source_path(), min_confidence)
    cuts_path: Path | None = None
    if analysis is not None and not dry_run:
        from afv.core.cache import atomic_write_json
        from afv.media.shots import CUTS_FILE, cuts_payload

        cuts_path = proj.media_dir / CUTS_FILE
        proj.media_dir.mkdir(parents=True, exist_ok=True)
        payload = cuts_payload(analysis, proj.info.source.sha256, min_confidence=min_confidence)
        atomic_write_json(cuts_path, payload)
    existing = [s.info for s in proj.shots()]
    added: list[str] = []
    skipped = 0
    for start, dur, conf in segs:
        overlap = max((min(start + dur, e.end) - max(start, e.start) for e in existing), default=0.0)
        if overlap > 0.5 * dur:
            skipped += 1
            continue
        if dry_run:
            say(
                {"event": "shot_found", "start": start, "duration": dur, "confidence": conf},
                f"  {start:9.3f} - {start + dur:9.3f} s  ({dur:.2f} s, confidence "
                f"{conf if conf is not None else '?'})",
            )
            continue
        s = proj.add_shot(start, dur, origin="detected", confidence=conf, state="candidate").info
        added.append(s.id)
        say({"event": "shot_added", **s.model_dump(mode="json")},
            f"  shot {s.id}: {s.start:9.3f} - {s.end:9.3f} s")
    summary = {
        "event": "detect_done", "found": len(segs), "added": len(added), "skipped_overlap": skipped,
        "shots": added, "cuts": str(cuts_path) if cuts_path else None,
        "cut_count": len(analysis.cuts) if analysis is not None else None,
        "seconds": round(time.monotonic() - t0, 2),
    }  # fmt: skip
    say(
        summary,
        f"{len(segs)} segments, {len(added)} added as candidates, {skipped} skipped (overlap existing shots)"
        + (f"; cuts in {cuts_path}" if cuts_path else ""),
    )
    return summary


def _shot_detector() -> Callable[..., Any]:
    try:
        import afv.media as media
    except ImportError:
        media = None  # type: ignore[assignment]
    fn = getattr(media, "detect_shots", None) if media is not None else None
    if fn is None:
        raise ProjectError(
            "cut detection (afv.media.detect_shots) is not available yet; "
            "add shots by hand: afv shots add --start S --end E"
        )
    return fn


def _detect_segments(
    video: Path, min_confidence: float = 0.0
) -> tuple[list[tuple[float, float, float | None]], Any]:
    """([(start, duration, confidence)], the cut analysis or None) from afv.media.detect_shots (shots, or
    (shots, analysis))."""
    fn = _shot_detector()
    try:
        found = fn(video, min_confidence=min_confidence) if min_confidence else fn(video)
    except Exception as e:
        raise ProjectError(f"cut detection failed on {video}: {type(e).__name__}: {e}") from None
    analysis = None
    if isinstance(found, tuple) and found and isinstance(found[0], list):
        analysis = found[1] if len(found) > 1 and hasattr(found[1], "cuts") else None
        found = found[0]
    out = []
    for seg in found:
        get = (lambda k, s=seg: s.get(k)) if isinstance(seg, dict) else (lambda k, s=seg: getattr(s, k, None))
        start = get("start")
        dur = get("duration")
        if dur is None and get("end") is not None:
            dur = float(get("end")) - float(start)
        if start is None or dur is None:
            raise ProjectError(f"afv.media.detect_shots returned a segment without start/duration: {seg!r}")
        conf = get("confidence")
        out.append((float(start), float(dur), None if conf is None else float(conf)))
    return out, analysis


def parse_shot_ids(proj: Project, tokens: list[str] | None) -> list[str]:
    """Shot ids from ``3 5``, ``3,5`` or ranges of numbered shots ``7-9`` (an existing id with a dash is
    that id); in the given order, without repeats."""
    ids = proj.shot_ids()
    out: list[str] = []
    for tok in tokens or []:
        for part in tok.replace(" ", ",").split(","):
            if not part:
                continue
            if part in ids:
                out.append(part)
                continue
            m = re.fullmatch(r"(\d+)-(\d+)", part)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if a > b:
                    raise ConfigError(f"bad shot range {part!r} (first > last)")
                hits = [i for i in ids if i.isdigit() and a <= int(i) <= b]
                if not hits:
                    raise ProjectError(f"no shots in {part}")
                out += hits
                continue
            out.append(proj.resolve_shot_id(part))
    return list(dict.fromkeys(out))


def select_shots(
    proj: Project, ids: list[str] | None, states: list[str] | set[str] | None, default: tuple[str, ...] | None
) -> list[Shot]:
    """Shots by id (``parse_shot_ids``) or all of them, keeping those in ``states`` (default ``default``;
    None = every state). States filter given ids too only when ``states`` is given. A shot whose shot.json
    cannot be read is left out with a warning (the others go on)."""
    if ids:
        shots = [proj.shot(i) for i in parse_shot_ids(proj, ids)]
        keep = set(states) if states else None
    else:
        shots = proj.shots()
        keep = set(states) if states else (set(default) if default else None)
    out = []
    for sh in shots:
        try:
            state = sh.info.state
        except ProjectError as e:
            logging.getLogger("afv").warning("shot %s left out: %s", sh.id, e)
            continue
        if keep is None or state in keep:
            out.append(sh)
    return out


def shot_row(
    sh: Shot, *, info: Any = None, project_info: Any = None, stages: bool = False,
    scan_listing: set[str] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """A shot as ``afv shots list --json`` and serve ``shots`` show it: shot.json fields, ``end``, pick
    status (``picked``; ``pick_stale``: picked for another interval or source), the scan (``status`` ok |
    stale | none | bad, frame, people, fighter_sized, fighters_likely, ``thumbnail`` relative to the
    project), ``exported`` (has an export result) and, with ``stages``, the stages with stored results.
    ``fighters_likely``/``fighter_sized`` follow the current rule (``afv.media.shots.scan_hints``), not the
    values stored when the shot was scanned. ``scan_listing``: the file names in media/scan (saves a file
    test per shot)."""
    from afv.core.project import read_scan, scan_paths
    from afv.media.shots import scan_hints

    i = info or sh.info
    pinfo = project_info or sh.project.info
    sha = pinfo.source.sha256
    status, scan = read_scan(sh, info=i, project_info=pinfo, listing=scan_listing)
    thumb = scan_paths(sh.project, sh.id).thumb
    has_thumb = thumb.name in scan_listing if scan_listing is not None else thumb.is_file()
    ok = status == "ok" and scan is not None
    likely, sized = scan_hints(scan) if ok and scan is not None else (None, None)
    present: set[str] = set()
    with contextlib.suppress(OSError):
        present = set(os.listdir(sh.dir / "stages"))  # one listing per shot: most have no results
    row: dict[str, Any] = {
        **i.model_dump(mode="json"), "end": i.end, **_pick_status(sh, i, sha), "error": None,
        "scan": {
            "status": status,
            "frame": scan.get("frame") if ok else None,
            "people": scan.get("people") if ok else None,
            "fighter_sized": sized,
            "fighters_likely": likely,
            "thumbnail": str(thumb.relative_to(sh.project.root)) if has_thumb else None,
        },
        "exported": "export" in present and sh.meta("export") is not None,
    }  # fmt: skip
    if stages:
        row["stored"] = [s for s in STAGE_ORDER if s in present and sh.meta(s)]
    return row


def error_row(shot_id: str, message: str) -> dict[str, Any]:
    """The row of a shot whose shot.json cannot be read: ``id`` and ``error``; ``state`` is null."""
    return {"id": shot_id, "error": message, "state": None, "start": None, "end": None, "duration": None,
            "picked": False, "pick_stale": False, "scan": None, "exported": False}  # fmt: skip


def _pick_status(sh: Shot, info: Any, sha: str) -> dict[str, Any]:
    none = {"picked": False, "pick_stale": False}
    try:
        data = json.loads(sh.edit_path("fighters").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return none
    if not isinstance(data, dict):
        return none
    picked = data.get("fighters") is not None or bool(data.get("objects"))
    ref = data.get("shot") if isinstance(data.get("shot"), dict) else None
    stale = False
    if ref is not None:
        with contextlib.suppress(TypeError, ValueError):
            stale = (abs(float(ref.get("start")) - info.start) > 1e-6
                     or abs(float(ref.get("duration")) - info.duration) > 1e-6
                     or bool(ref.get("source_sha256") and ref.get("source_sha256") != sha))  # fmt: skip
    return {"picked": picked, "pick_stale": stale}


def shot_rows(
    proj: Project, states: list[str] | set[str] | None = None, *, stages: bool = False, jobs: int = 16
) -> list[dict[str, Any]]:
    """``shot_row`` of every shot (in ``states``), in shot order; a shot whose shot.json cannot be read is
    an ``error_row`` (always listed) and does not stop the others. The files are read by ``jobs`` threads:
    on the Windows drives each read is a round trip to Windows, so a long list is latency-bound."""
    from concurrent.futures import ThreadPoolExecutor

    from afv.core.project import scan_dir, shot_sort_key

    pinfo = proj.info
    try:
        listing: set[str] | None = set(os.listdir(scan_dir(proj)))
    except FileNotFoundError:
        listing = set()
    except OSError:
        listing = None
    try:
        names = sorted(os.listdir(proj.shots_dir), key=shot_sort_key)
    except FileNotFoundError:
        names = []

    def one(name: str) -> dict[str, Any] | None:
        sh = Shot(proj, name, check=False)
        if not (sh.dir / "shot.json").exists():  # not a shot folder (e.g. half made): as shot_ids()
            return None
        try:
            i = sh.info
        except ProjectError as e:
            return error_row(name, str(e))
        if states and i.state not in states:
            return None
        try:
            return shot_row(sh, info=i, project_info=pinfo, stages=stages, scan_listing=listing)
        except (ProjectError, OSError) as e:
            return error_row(name, str(e))

    with ThreadPoolExecutor(max(1, min(jobs, len(names) or 1))) as ex:
        return [r for r in ex.map(one, names) if r is not None]


@shots_app.command("list")
@_handled
def shots_list(
    state: StatesOpt = None,
    likely: Annotated[
        bool, typer.Option("--likely", help="Only shots whose scan says two fighters are likely visible.")
    ] = False,
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """List shots: state, range, speed, pick status, scan (people found, fighters likely) and which stages
    have stored results."""
    proj = _project(project)
    rows = shot_rows(proj, [s.value for s in state] if state else None, stages=True)
    if likely:
        rows = [r for r in rows if r["error"] or r["scan"]["fighters_likely"]]
    if not rows and not proj.shot_ids() and not json_out:
        typer.echo("no shots yet: afv shots detect, or afv shots add --start S --end E")
    for r in rows:
        if r["error"]:
            _emit(json_out, {"event": "shot", **r}, f"{r['id']:>5}  ERROR     {r['error']}")
            continue
        sc = r["scan"]
        if sc["status"] == "ok":
            scan = f"{sc['people']} people{', LIKELY' if sc['fighters_likely'] else ''}"
        else:
            scan = {"none": "-", "stale": "scan stale", "bad": "scan unreadable"}.get(sc["status"], "-")
        pick = "re-pick" if r["pick_stale"] else ("picked" if r["picked"] else "")
        _emit(
            json_out,
            {"event": "shot", **r},
            f"{r['id']:>5}  {r['state']:<9} {r['start']:9.3f} - {r['end']:9.3f} s  x{r['speed_factor']:<5g} "
            f"{pick:<7}  {scan:<18} {','.join(r['stored']) or '-'}"
            + (f"  # {r['notes']}" if r["notes"] else ""),
        )


@shots_app.command("set")
@_handled
def shots_set(
    shot: Annotated[str, typer.Argument(help="Shot id.")],
    start: Annotated[
        str | None,
        typer.Option(help="New start (in point): seconds or m:ss.s. Alone, it keeps the end where it is."),
    ] = None,
    end: Annotated[str | None, typer.Option(help="New end (out point).")] = None,
    duration: Annotated[str | None, typer.Option(help="New duration in seconds (instead of --end).")] = None,
    speed: Annotated[
        float | None, typer.Option(help="Footage speed vs real time, e.g. 0.25 for a 4x slow-motion replay.")
    ] = None,
    notes: Annotated[str | None, typer.Option(help="Free text (replaces the notes).")] = None,
    state: Annotated[StateChoice | None, typer.Option("--state", help="candidate, queued, skipped.")] = None,
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Change a shot: trim it (in/out points), its speed factor, notes or state.

    A new interval makes the shot's scan stale (afv shots scan ID scans it again) and its fighter picks (afv
    pick ID shows the people again); its stages re-run. Changing the interval or speed waits for no run:
    it fails while afv runs the shot."""
    sh = _project(project).shot(shot)
    def t(v: str | None) -> float | None:
        return parse_time(v) if v is not None else None

    res = set_shot(sh, start=t(start), end=t(end), duration=t(duration), speed=speed, notes=notes,
                   state=state.value if state is not None else None)  # fmt: skip
    text = f"shot {sh.id}: " + (f"{', '.join(res['changed'])} changed" if res["changed"] else "no change")
    text += (f"\n  {res['state']}  {res['start']:.3f} - {res['end']:.3f} s ({res['duration']:.3f} s) "
             f"x{res['speed_factor']:g}")
    for n in res["notes_for_user"]:
        text += f"\n  note: {n}"
    _emit(json_out, {"event": "shot_set", **res}, text)


def set_shot(
    sh: Shot,
    *,
    start: float | None = None,
    end: float | None = None,
    duration: float | None = None,
    speed: float | None = None,
    notes: str | None = None,
    state: str | None = None,
) -> dict[str, Any]:
    """``afv shots set`` (and serve ``shot_set``): returns the shot's row (``shot_row``) plus ``changed``
    (field names) and ``notes_for_user``. ``start`` alone keeps the end; ``end``/``duration`` keep the start.
    Interval or speed changes take the shot lock (a running shot is refused); notes/state changes do not."""
    if end is not None and duration is not None:
        raise ConfigError("give at most one of end and duration")
    info = sh.info
    new_start = info.start if start is None else start
    dur: float | None = None
    if end is not None:
        dur = end - new_start
    elif duration is not None:
        dur = duration
    elif start is not None:
        dur = info.end - new_start

    def apply() -> tuple[Any, list[str]]:
        return sh.update(start=start, duration=dur, speed_factor=speed, notes=notes, state=state)

    if dur is not None or speed is not None:
        with Runner(sh).shot_lock():
            new, changed = apply()
    else:
        new, changed = apply()
    row = shot_row(sh, info=new)
    msgs = []
    if {"start", "duration"} & set(changed):
        if row["picked"] and row["pick_stale"]:
            msgs.append(f"the fighters were picked for the old interval: pick again (afv pick {sh.id})")
        if row["scan"]["status"] == "stale":
            msgs.append(f"its scan is for the old interval: afv shots scan {sh.id}")
    return {**row, "shot": sh.id, "changed": changed, "notes_for_user": msgs}


def _set_states(proj: Project, tokens: list[str], state: str, json_out: bool) -> None:
    ids = parse_shot_ids(proj, tokens)
    if not ids:
        raise ConfigError("give shot ids, e.g. 3 5 or 3,5 or 7-9")
    changed = [sid for sid in ids if proj.shot(sid).update(state=state)[1]]
    _emit(
        json_out,
        {"event": "shots_state", "state": state, "shots": ids, "changed": changed},
        f"{state}: {', '.join(ids)}" + ("" if changed else " (no change)"),
    )


@shots_app.command("queue")
@_handled
def shots_queue(
    ids: Annotated[list[str], typer.Argument(help="Shot ids: 3 5, 3,5 or 7-9.")],
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Queue shots: afv batch runs every queued shot."""
    _set_states(_project(project), ids, "queued", json_out)


@shots_app.command("skip")
@_handled
def shots_skip(
    ids: Annotated[list[str], typer.Argument(help="Shot ids: 3 5, 3,5 or 7-9.")],
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Skip shots: afv batch leaves them out (afv shots queue IDS takes them back)."""
    _set_states(_project(project), ids, "skipped", json_out)


@shots_app.command("scan")
@_handled
def shots_scan(
    ids: Annotated[
        list[str] | None, typer.Argument(help="Shot ids (default: every candidate and queued shot).")
    ] = None,
    state: StatesOpt = None,
    force: Annotated[bool, typer.Option("--force", help="Also scan shots whose scan is up to date.")] = False,
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Find the people on one frame of each shot (RF-DETR in one GPU job for all shots): a numbered prompt
    (what afv pick uses, so picking needs no GPU), a thumbnail, and whether two fighters are likely visible.

    Files: media/scan/<id>.json, <id>.jpg (thumbnail), <id>.frame.jpg. Shots with an up-to-date scan are
    kept (--force scans them again), so an interrupted scan resumes where it stopped. A trimmed shot's scan
    is stale until it is scanned again."""
    from afv.core.project import read_scan
    from afv.media.shots import scan_hints, scan_shots

    proj = _project(project)
    shots = select_shots(proj, ids, [s.value for s in state] if state else None, ("candidate", "queued"))
    throttle = _Throttle()

    def progress(frac: float | None, msg: str) -> None:
        ev = {"event": "progress", "stage": "scan", "frac": frac, "message": msg}
        if throttle.allow(ev):
            if json_out:
                _Out.echo(json.dumps(ev))
            else:
                pct = f"{100 * frac:3.0f}% " if frac is not None and frac >= 0 else ""
                _Out.echo(f"  scan {pct}{msg}")

    def scanned(sid: str, data: dict[str, Any]) -> None:
        if json_out:
            _Out.echo(json.dumps({"event": "scanned", "shot": sid, "frame": data["frame"],
                                  "people": data["people"], "fighters_likely": scan_hints(data)[0]}))

    rep = scan_shots(proj, shots, force=force, progress=progress, on_scan=scanned)
    likely = []
    for sh in shots:
        st, sc = read_scan(sh)
        if st == "ok" and sc and scan_hints(sc)[0]:
            likely.append(sh.id)
    _emit(
        json_out,
        {"event": "scan_done", **rep.to_dict(), "shots": [sh.id for sh in shots], "fighters_likely": likely},
        f"scanned {len(rep.scanned)} shots ({len(rep.up_to_date)} already up to date"
        + (f", {len(rep.failed)} failed" if rep.failed else "") + ")"
        + (f" in {rep.seconds.get('total', 0):.0f} s" if rep.scanned else "")
        + f"; fighters likely in {len(likely)}: {_id_ranges(likely) or '-'}"
        + "".join(f"\n  {k}: {v}" for k, v in rep.failed.items())
        + (f"\nerror: {rep.error}" if rep.error else ""),
    )
    if rep.error or rep.failed:
        raise typer.Exit(EXIT_ERROR)


def _id_ranges(ids: list[str]) -> str:
    """'3,5,7-9' for numbered ids (others listed as they are)."""
    out: list[str] = []
    nums = [int(i) for i in ids if i.isdigit()]
    width = max((len(i) for i in ids if i.isdigit()), default=3)
    k = 0
    while k < len(nums):
        j = k
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        a, b = f"{nums[k]:0{width}d}", f"{nums[j]:0{width}d}"
        out.append(a if j == k else f"{a}-{b}")
        k = j + 1
    return ",".join(out + [i for i in ids if not i.isdigit()])



@app.command()
@_handled
def pick(
    shot: Annotated[str, typer.Argument(help="Shot id.")],
    fighters: Annotated[
        str | None,
        typer.Option(help="The two fighters' numbers from the prompt image: fighter A first, e.g. 1,0."),
    ] = None,
    extra: Annotated[
        str | None, typer.Option(help="Other people to track (referee, corner), e.g. 2.")
    ] = None,
    keep_prompts: Annotated[
        bool,
        typer.Option(
            "--keep-prompts",
            help="Keep earlier re-prompts and labels even though the picks change (their object ids "
            "then refer to the new picks).",
        ),
    ] = False,
    frame: Annotated[
        int | None,
        typer.Option("--frame", help="Show the numbered people on this frame (30 Hz shot timeline); with "
                     "--fighters: the numbers are from this frame.", min=0),
    ] = None,
    prompts: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt",
            help="Re-prompt F:OBJ:DET: on frame F, object OBJ (1 = fighter A, 2 = B, 3.. = extras; a new "
            "number adds a person) is detection DET of that frame's numbered image. Repeatable.",
        ),
    ] = None,
    label: Annotated[
        str, typer.Option("--label", help="Label for the re-prompted object, e.g. referee (one --prompt).")
    ] = "",
    swaps: Annotated[
        list[str] | None,
        typer.Option(
            "--swap",
            help="Identity swap F:O1:O2: from frame F to the shot end, tracked objects O1 and O2 trade "
            "identities (1 = fighter A, 2 = B, 3.. = extras), e.g. after a mask jumped from one fighter to "
            "the other. Saved to edits/ids.json; afv run then re-checks the tracking without tracking "
            "again. Repeatable.",
        ),
    ] = None,
    clear_swaps: Annotated[
        bool, typer.Option("--clear-swaps", help="Remove every identity swap (edits/ids.json).")
    ] = False,
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Choose the fighters. Without --fighters: show the numbered people (no GPU when the shot has a prompt
    image or an up-to-date scan from afv shots scan; else detecting first).

    The numbers refer to: with --frame F, frame F's people (the scan's when it was made on F, else the
    shot's prompt image when it is on F, else a detection on F); without --frame, the shot's prompt image
    (afv run, when nothing was picked), else its scan. Picking from a scan starts no GPU work.

    Changing the picks drops earlier re-prompts and labels (edits/fighters.json "prompts"/"objects"),
    because object ids follow the pick order; --keep-prompts keeps them.

    Re-prompting (after the first pick): --frame F shows the numbered people on any frame, --prompt
    F:OBJ:DET adds a correction or a new person (e.g. a referee entering late) to edits/fighters.json;
    then afv run tracks again."""
    proj = _project(project)
    sh = proj.shot(shot)
    sdir = sh.stage_dir("detect_track")
    if swaps or clear_swaps:
        if fighters is not None or extra is not None or frame is not None or prompts:
            raise ConfigError("--swap/--clear-swaps cannot be combined with other pick options")
        with Runner(sh).shot_lock():
            _swap_ids(sh, swaps or [], clear_swaps, json_out)
        return
    if label and not prompts:
        raise ConfigError("--label needs --prompt")
    if prompts and (fighters is not None or extra is not None):
        raise ConfigError("--prompt cannot be combined with --fighters/--extra; pick first")
    if extra is not None and fighters is None:
        raise ConfigError("--extra needs --fighters")
    if (frame is not None and fighters is None) or prompts:
        with Runner(sh).shot_lock():  # no edit while another afv process runs this shot
            _reprompt(sh, frame, prompts or [], label, json_out)
        return
    if fighters is None:
        prompt, source, img = pick_prompt(sh)
        if not source or img is None:
            report = _make_runner(sh, json_out).run(to="detect_track")
            if report.outcome == "failed":
                raise typer.Exit(_finish(report, json_out))
            prompt, source, img = pick_prompt(sh)
        if img is None:
            raise RunError(f"detect_track wrote no prompt image in {sdir}")
        current = sh.read_edit("fighters")
        _emit(
            json_out,
            {
                "event": "pick_prompt",
                "shot": sh.id,
                "prompt_image": str(img),
                "prompt": prompt,
                "source": source,
                "current": current,
            },
        )
        if not json_out:
            what = "scan" if source == "scan" else "prompt image"
            typer.echo(f"{what}: {img} (frame {prompt.get('prompt_frame', prompt.get('frame', '?'))})")
            dets = _detections(prompt)
            for k, (box, score) in enumerate(dets):
                b = ", ".join(f"{v:.0f}" for v in box)
                typer.echo(f"  {k}: box [{b}]" + (f"  score {score:.2f}" if score is not None else ""))
            if not dets and prompt:
                typer.echo(
                    f"nobody detected on frame {prompt.get('prompt_frame', '?')}; "
                    + _NO_PEOPLE_HINT.format(id=sh.id)
                )
            if current:
                typer.echo(
                    f"current pick: fighters {current.get('fighters')} extra {current.get('extra', [])}"
                )
            typer.echo(f"next: afv pick {sh.id} --fighters A,B [--extra N,...]   then: afv run {sh.id}")
        return

    fl = _int_list(fighters, "--fighters")
    el = _int_list(extra, "--extra")
    res = pick_fighters(sh, fl, el, frame=frame, keep_prompts=keep_prompts)
    data, path, dropped = res["data"], res["edit"], res["dropped"]
    note = ""
    if dropped:
        what = " and ".join(
            f"{n} {'re-prompt' if k == 'prompts' else 'label'}{'' if n == 1 else 's'}"
            for k, n in dropped.items()
        )
        note = (
            f"\ndropped {what} of the earlier pick (their ids would now mean other people; "
            "--keep-prompts keeps them)"
        )
    _emit(
        json_out,
        {
            "event": "picked",
            **data,
            "shot": sh.id,
            "picked_for": data.get("shot"),
            "edit": str(path),
            "dropped": dropped,
            "source": res["source"],
        },
        f"saved {path}: fighter A = {fl[0]}, fighter B = {fl[1]}"
        + (f", also tracking {el}" if el else "")
        + (" (numbers from the scan)" if res["source"] == "scan" else "")
        + note
        + f"\nnext: afv run {sh.id}",
    )


def pick_prompt(
    sh: Shot, frame: int | None = None, *, image: bool = True
) -> tuple[dict[str, Any], str, Path | None]:
    """(prompt, source, image) the pick numbers refer to (see ``afv pick --help``). ``source``: ``stage``
    (detect_track's prompt.json), ``scan`` (media/scan), ``detected`` (a detection on ``frame``, GPU,
    cached like ``afv pick --frame``) or "" (nothing yet: only without ``frame``). The scan's numbered
    image is written on first use (``image=False``: not looked for, the image is None)."""
    from afv.core.project import load_scan

    def scan_image() -> Path | None:
        if not image:
            return None
        from afv.media.shots import prompt_image  # imports the media stack (PyAV): only when needed

        return prompt_image(sh)

    sdir = sh.stage_dir("detect_track")
    scan = load_scan(sh)
    if frame is not None and scan is not None and int(scan["frame"]) == frame:
        return dict(scan["prompt"]), "scan", scan_image()
    stage = _read_prompt(sdir)
    if stage and not stage_prompt_current(sh, stage):
        stage = {}  # made for another interval, source or threshold: its numbers mean nothing now
    if frame is None:
        if stage:
            return stage, "stage", find_prompt_image(sdir)
        if scan is not None:
            return dict(scan["prompt"]), "scan", scan_image()
        return {}, "", None
    stage_frame = stage.get("prompt_frame", stage.get("frame")) if stage else None
    if stage and isinstance(stage_frame, int) and stage_frame == frame:
        return stage, "stage", find_prompt_image(sdir)
    info, img = _frame_prompt(sh, frame)
    d = {"frame": info.frame, "prompt_frame": info.frame, "boxes": info.boxes.tolist(),
         "scores": info.scores.tolist()}
    return d, "detected", img


def stage_prompt_current(sh: Shot, prompt: dict[str, Any]) -> bool:
    """A detect_track prompt (prompt.json, suggest_<f>.json) belongs to the shot as it is now: its result was
    made for the shot's interval and source, and the prompt (when it records one) at the shot's detection
    threshold (``afv.core.project.scan_threshold``)."""
    from afv.core.project import scan_threshold

    meta = sh.meta("detect_track") or {}
    ref = (meta.get("key_parts") or {}).get("shot") or {}
    pinfo, i = sh.project.info, sh.info
    try:
        if abs(float(ref["start"]) - i.start) > 1e-6 or abs(float(ref["duration"]) - i.duration) > 1e-6:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    if ref.get("source_sha256") != pinfo.source.sha256:
        return False
    thr = prompt.get("threshold")
    return not isinstance(thr, int | float) or abs(float(thr) - scan_threshold(pinfo, i)) <= 1e-9


def pick_fighters(
    sh: Shot, fl: list[int], el: list[int], *, frame: int | None = None, keep_prompts: bool = False
) -> dict[str, Any]:
    """Write the base picks (``afv pick --fighters``): fighters ``fl`` (A, B) and extras ``el`` are numbers
    of the prompt ``pick_prompt(sh, frame)`` gives. Returns {"data", "edit" (path), "dropped", "source",
    "frame"}."""
    if len(fl) != 2:
        raise ConfigError(f"--fighters needs exactly two numbers (fighter A, fighter B), got {fl}")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in fl + el):
        raise ConfigError(f"fighters/extra must be numbers >= 0, got {fl} / {el}")
    if len(set(fl)) != 2 or len(set(el)) != len(el):
        raise ConfigError(f"repeated number in fighters {fl} / extra {el}")
    if set(fl) & set(el):
        raise ConfigError("--extra must not repeat a fighter")
    prompt, source, _ = pick_prompt(sh, frame, image=False)
    dets = _detections(prompt)
    if "boxes" in prompt or "detections" in prompt:  # the prompt is known: check the numbers
        if not dets:
            raise ConfigError(f"nobody was detected on the prompt frame; {_NO_PEOPLE_HINT.format(id=sh.id)}")
        bad = [v for v in fl + el if v >= len(dets)]
        if bad:
            raise ConfigError(f"no detection {bad} in the prompt image (it has 0..{len(dets) - 1})")
    data = sh.read_edit("fighters")
    data = dict(data) if isinstance(data, dict) else {}
    prompt_frame = prompt.get("prompt_frame") if prompt else None
    if prompt_frame is None and prompt:
        prompt_frame = prompt.get("frame")
    old = (data.get("fighters"), data.get("extra", []), data.get("prompt_frame"))
    dropped: dict[str, int] = {}
    if data.get("fighters") is not None and old != (fl, el, prompt_frame) and not keep_prompts:
        for key in ("prompts", "objects"):  # their object ids follow the old pick order
            if data.get(key):
                dropped[key] = len(data[key])
            data.pop(key, None)
    info = sh.info
    data.update(
        {
            "fighters": fl,
            "extra": el,
            "prompt_frame": prompt_frame,
            "boxes": {str(v): list(dets[v][0]) for v in fl + el} if dets else {},
            "shot": {
                "start": info.start,
                "duration": info.duration,
                "source_sha256": sh.project.info.source.sha256,
            },
        }
    )
    path = sh.write_edit("fighters", data)
    return {"data": data, "edit": path, "dropped": dropped, "source": source, "frame": prompt_frame}


def _swap_ids(sh: Shot, texts: list[str], clear: bool, json_out: bool) -> None:
    """Append identity swaps to (or clear) edits/ids.json, checked against the stored tracking."""
    from afv.core.edits import write_edit

    added = []
    for text in texts:
        try:
            f, o1, o2 = (int(v) for v in text.split(":"))
        except ValueError:
            raise ConfigError(f"--swap {text!r}: expected FRAME:OBJ1:OBJ2, e.g. 141:1:2") from None
        added.append({"start": f, "end": None, "objects": [o1, o2]})
    tracks = sh.load_result("detect_track") if sh.has_result("detect_track") else None
    ids = [int(i) for i in getattr(tracks, "track_ids", [])]
    n = int(getattr(tracks, "num_frames", 0) or 0)
    for sw in added:
        if ids and (bad := [o for o in sw["objects"] if o not in ids]):
            raise ConfigError(f"--swap: object(s) {bad} are not tracked in shot {sh.id} (tracked: {ids})")
        if n and sw["start"] >= n:
            raise ConfigError(f"--swap: frame {sw['start']} is outside the shot (0..{n - 1})")
    data = sh.read_edit("ids")
    data = dict(data) if isinstance(data, dict) and not clear else {}
    data["swaps"] = [*([] if clear else list(data.get("swaps") or [])), *added]
    path = write_edit(sh, "ids", data)
    n = len(data["swaps"])
    lines = [f"swap: from frame {a['start']}, objects {a['objects'][0]} <-> {a['objects'][1]}" for a in added]
    if clear:
        lines.insert(0, "cleared the identity swaps")
    _emit(
        json_out,
        {"event": "swapped", "shot": sh.id, "edit": str(path), "swaps": data["swaps"], "added": added,
         "cleared": clear},
        "\n".join(lines) + f"\nsaved {path} ({n} swap{'s' if n != 1 else ''})"
        f"\nnext: afv run {sh.id} (re-checks the tracking with the swaps; no re-tracking)",
    )  # fmt: skip


def _parse_prompt(text: str) -> tuple[int, int, int]:
    try:
        f, o, d = (int(v) for v in text.split(":"))
    except ValueError:
        raise ConfigError(f"--prompt {text!r}: expected FRAME:OBJ:DET, e.g. 141:4:2") from None
    if f < 0 or o < 1 or d < 0:
        raise ConfigError(f"--prompt {text!r}: frame and detection must be >= 0, the object id >= 1")
    return f, o, d


def _frame_prompt(sh: Shot, frame: int) -> tuple[Any, Path]:
    """(PromptInfo, image) of the numbered RF-DETR people on ``frame``: the shot's scan (afv shots scan),
    the stored base prompt or a QC suggestion when it was made on that frame at the current detection
    threshold, else a fresh detection cached in ``<shot>/.cache/track/prompts`` (dropped with the frame
    cache when the shot changes)."""
    from afv.core.project import load_scan
    from afv.media.shots import prompt_image
    from afv.models import track

    runner = Runner(sh)
    cls, _ = runner.resolve("detect_track")
    threshold = float(getattr(runner.params_for("detect_track", cls), "det_threshold", 0.4))
    scan = load_scan(sh)  # the scan of this frame at this threshold: no frames to prepare, no GPU
    if (scan is not None and int(scan["frame"]) == frame
            and abs(float(scan["prompt"].get("threshold", threshold)) - threshold) <= 1e-9):
        img = prompt_image(sh)
        if img is not None:
            return track.PromptInfo.from_json({**scan["prompt"], "image": str(img)}), img
    video = sh.project.source_path()
    cache = sh.dir / track.CACHE_SUBDIR
    fr = track.prepare_frames(video, sh.info.start, sh.info.duration, cache)
    if not 0 <= frame < fr.num_frames:
        raise ConfigError(f"frame {frame} is outside the shot (0..{fr.num_frames - 1} on the 30 Hz timeline)")
    shot_key = {"start": fr.start, "duration": fr.duration, "source_sha256": sh.project.info.source.sha256}
    sdir = sh.stage_dir("detect_track")
    pdir = cache / "prompts"
    for c in (pdir / f"prompt_{frame:05d}.json", sdir / f"suggest_{frame:05d}.json", sdir / "prompt.json"):
        try:
            d = json.loads(c.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if int(d.get("frame", d.get("prompt_frame", -1))) != frame:
            continue
        if abs(float(d.get("threshold", threshold)) - threshold) > 1e-9:
            continue
        if c.parent == pdir and d.get("shot") != shot_key:
            continue
        if c.parent == sdir and not stage_prompt_current(sh, d):
            continue  # detect_track's result is for another interval or source
        info = track.PromptInfo.from_json(d)
        return info, c.parent / info.image
    pdir.mkdir(parents=True, exist_ok=True)
    track.detect_prompt(video, pdir, [frame], threshold=threshold, worker=track.envs_worker(), shot_frames=fr)
    pj = pdir / f"prompt_{frame:05d}.json"
    d = {**json.loads(pj.read_text(encoding="utf-8")), "threshold": threshold, "shot": shot_key}
    pj.write_text(json.dumps(d, indent=1), encoding="utf-8")
    info = track.PromptInfo.from_json(d)
    return info, pdir / info.image


def _reprompt(sh: Shot, frame: int | None, prompts: list[str], label: str, json_out: bool) -> None:
    from afv.models import track

    if frame is not None:
        info, img = _frame_prompt(sh, frame)
        boxes = [[float(v) for v in b] for b in info.boxes]
        scores = [float(v) for v in info.scores]
        _emit(json_out, {"event": "frame_prompt", "shot": sh.id, "frame": frame, "image": str(img),
                         "boxes": boxes, "scores": scores})  # fmt: skip
        if not json_out:
            typer.echo(f"frame {frame}: {img}")
            for k, (box, score) in enumerate(zip(boxes, scores, strict=True)):
                typer.echo(f"  {k}: box [{', '.join(f'{v:.0f}' for v in box)}]  score {score:.2f}")
            if not prompts:
                typer.echo(
                    f"next: afv pick {sh.id} --prompt {frame}:OBJ:DET [--label NAME]   then: afv run {sh.id}"
                )
    if not prompts:
        return
    if label and len(prompts) != 1:
        raise ConfigError("--label applies to one --prompt at a time")
    data = sh.read_edit("fighters")
    if not isinstance(data, dict) or (data.get("fighters") is None and not data.get("objects")):
        raise ConfigError(f"pick the fighters first: afv pick {sh.id} --fighters A,B [--extra N,...]")
    added = []
    for text in prompts:
        f, obj, det = _parse_prompt(text)
        info, _ = _frame_prompt(sh, f)
        data = track.add_prompt(data, f, obj, detection=det, prompt=info, label=label)
        added.append({"frame": f, "obj": obj, "detection": det, "box": info.box(det)})
    path = sh.write_edit("fighters", data)
    lines = [f"prompt: frame {a['frame']}, object {a['obj']} = detection {a['detection']}" for a in added]
    _emit(
        json_out,
        {
            "event": "reprompted",
            **data,
            "shot": sh.id,
            "picked_for": data.get("shot"),
            "edit": str(path),
            "added": added,
        },
        "\n".join(lines) + (f" ({label})" if label else "") + f"\nsaved {path}\nnext: afv run {sh.id}",
    )


_NO_PEOPLE_HINT = "try another prompt frame: afv run {id} --to detect_track --set detect_track.prompt_frame=N"


def _read_prompt(sdir: Path) -> dict[str, Any]:
    p = sdir / "prompt.json"
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ProjectError(
            f"cannot read {p}: {e}; re-run detection: afv run <shot> --from detect_track --to detect_track"
        ) from None
    if not isinstance(data, dict):
        raise ProjectError(f"{p} is not a JSON object")
    return data


def _detections(prompt: dict[str, Any]) -> list[tuple[list[float], float | None]]:
    """[(box, score)] from prompt.json: {"boxes": [[x0,y0,x1,y1]...], "scores": [...]} or
    {"detections": [{"box": [...], "score": s}, ...]}."""
    try:
        if isinstance(prompt.get("detections"), list):
            dets = [(d.get("box", []), d.get("score")) for d in prompt["detections"]]
        else:
            boxes = prompt.get("boxes") or []
            scores = prompt.get("scores") or [None] * len(boxes)
            dets = list(zip(boxes, scores, strict=False))
        out = []
        for box, score in dets:
            b = [float(v) for v in box]
            if len(b) != 4:
                raise ValueError(f"box {box!r} is not [x0, y0, x1, y1]")
            out.append((b, None if score is None else float(score)))
        return out
    except (TypeError, ValueError, AttributeError) as e:
        raise ProjectError(f"prompt.json has malformed detections: {e}") from None


@app.command()
@_handled
def run(
    shot: Annotated[str, typer.Argument(help="Shot id.")],
    from_stage: Annotated[
        str | None, typer.Option("--from", help="Re-run this stage and all after it.")
    ] = None,
    to: Annotated[str, typer.Option(help="Last stage to run.")] = DEFAULT_TARGET,
    frames: Annotated[
        str | None,
        typer.Option(
            "--frames",
            help="With --from: re-solve only frames a:b (shot timeline, b exclusive) in the stages that "
            "support it (contacts, smooth, hands_face), blended into their previous result.",
        ),
    ] = None,
    camera: Annotated[CameraMode | None, typer.Option(help="Camera solve (saved to the shot).")] = None,
    sets: SetOpt = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    deterministic: Annotated[bool, typer.Option(help="Golden run: deterministic algorithms.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would run.")] = False,
    yes: YesOpt = False,
    code_changes: CodeChangesOpt = CodeChanges.report,
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Run the pipeline for one shot; up-to-date stages are skipped.

    Stops after detect_track (exit code 3) when its QC finds tracking problems (likely ID swap, lost
    fighter, ...): fix them with afv pick --frame/--prompt/--swap, or go on with --yes.

    Range re-run: --from contacts --frames 120:180 re-solves those frames (plus a margin) in contacts and
    the range-capable stages after it; other stages run in full."""
    sh = _project(project).shot(shot)
    fr = parse_frames(frames)
    if fr is not None and from_stage is None:
        raise ConfigError("--frames needs --from <stage>, e.g. --from contacts --frames 120:180")
    runner = _make_runner(sh, json_out, camera=camera, sets=sets, seed=seed, deterministic=deterministic,
                          code_changes=code_changes, pause_on_hints=True, yes=yes)  # fmt: skip
    if dry_run:
        _settings_overrides(sh, camera, sets, runner, persist=False)
        plan = runner.plan(to=to, from_stage=from_stage, frames=fr)
        _print_plan(plan, json_out)
        if plan.outcome == "failed":  # an invalid edit file: the table is shown, the exit code is 1
            raise typer.Exit(EXIT_ERROR)
        return
    with runner.shot_lock():  # settings are saved only once this process owns the shot
        runner.check_edits(to)  # an invalid edit file stops the run before settings are saved
        _settings_overrides(sh, camera, sets, runner, persist=True)
        report = runner.run(to=to, from_stage=from_stage, frames=fr)
    raise typer.Exit(_finish(report, json_out))


@app.command()
@_handled
def status(
    shot: Annotated[str, typer.Argument(help="Shot id.")],
    to: Annotated[str, typer.Option(help="Target stage.")] = DEFAULT_TARGET,
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Which stages are up to date, stale (and why), or waiting."""
    sh = _project(project).shot(shot)
    plan = _make_runner(sh, json_out).plan(to=to)
    _print_plan(plan, json_out)
    if plan.outcome == "failed":  # an invalid edit file: the table is shown, the exit code is 1
        raise typer.Exit(EXIT_ERROR)


def _print_plan(report: RunReport, json_out: bool) -> None:
    _emit(json_out, {"event": "plan", **report.to_dict()})
    if json_out:
        return
    typer.echo(f"shot {report.shot}:")
    for s in report.stages:
        state = "stale (code)" if s.state == "stale_code" else s.state
        typer.echo(f"  {s.name:<13} {state:<12} {s.impl or '':<12} {s.reason}")
    if report.message:
        typer.echo(f"note: {report.message}")
    if report.review and not report.review.get("acknowledged"):
        typer.echo(format_review(report.review, report.shot, paused=False))


@app.command()
@_handled
def batch(
    to: Annotated[str, typer.Option(help="Last stage to run.")] = DEFAULT_TARGET,
    camera: Annotated[CameraMode | None, typer.Option(help="Camera solve (saved to each shot).")] = None,
    sets: SetOpt = None,
    stop_on_error: Annotated[
        bool, typer.Option("--stop-on-error", help="Stop at the first failure.")
    ] = False,
    yes: YesOpt = False,
    code_changes: CodeChangesOpt = CodeChanges.report,
    state: Annotated[
        list[StateChoice] | None,
        typer.Option("--state", help="Run the shots in this state instead of the queued ones (repeatable)."),
    ] = None,
    all_shots: Annotated[bool, typer.Option("--all", help="Run every shot, whatever its state.")] = False,
    shot_ids: Annotated[
        list[str] | None,
        typer.Option("--shot", help="Run only these shots, in this order (repeatable; their state is not "
                     "checked unless --state is given)."),
    ] = None,
    batch_id: Annotated[
        str | None, typer.Option("--batch-id", hidden=True, help="Id of the batch log (set by afv serve).")
    ] = None,
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Run every queued shot (afv shots queue; shots made before shot states are queued). Shots waiting for
    a pick or a tracking review are reported and skipped.

    The queued shots are read again after each shot: a shot queued while the batch runs is run too; it
    stops when no queued shot is left. One batch per project at a time; every event is also written to
    media/batch/<batch id>.jsonl (afv serve's batch_status reads it, also for a batch started here).
    SIGTERM (afv serve's batch_cancel) stops it cleanly: finished shots are kept, the running shot keeps
    its previous results."""
    proj = _project(project)
    states = None if all_shots else ({s.value for s in state} if state else {"queued"})
    fixed = parse_shot_ids(proj, shot_ids) if shot_ids else None
    if fixed is not None and not state:
        states = None
    with BatchSession(proj, batch_id, {"to": to, "states": sorted(states) if states else None,
                                       "shots": fixed}) as log:  # fmt: skip
        reporter = _reporter(json_out)

        def report(ev: dict[str, Any]) -> None:
            reporter(ev)
            log.write(ev)

        def say(obj: dict[str, Any], text: str | None = None) -> None:
            _emit(json_out, obj, text)
            log.write(obj)

        outcomes: list[dict[str, Any]] = []
        done: set[str] = set()
        left = 0
        cancelled = False
        while not cancelled:
            try:
                sh, left = _next_batch_shot(proj, states, done, fixed, outcomes, say)
            except (SystemExit, KeyboardInterrupt):
                cancelled = True
                break
            if sh is None:
                break
            done.add(sh.id)
            say({"event": "batch_shot", "shot": sh.id, "index": len(done) - 1})
            try:
                runner = _make_runner(sh, json_out, camera=camera, sets=sets, pause_on_hints=True, yes=yes,
                                      code_changes=code_changes, report=report)  # fmt: skip
                with runner.shot_lock():
                    runner.check_edits(to)
                    _settings_overrides(sh, camera, sets, runner, persist=True)
                    rep = runner.run(to=to)
            except (RunError, ConfigError, ProjectError, StageError) as e:
                rep = RunReport(sh.id, "failed", message=str(e))
            except (SystemExit, KeyboardInterrupt):  # batch_cancel (SIGTERM) or Ctrl-C: stop cleanly
                rep = RunReport(sh.id, "cancelled", message="the batch was stopped")
                cancelled = True
            _finish(rep, json_out)
            log.write({"event": "result", **rep.to_dict()})
            entry = {"shot": sh.id, "outcome": rep.outcome, "message": rep.message,
                     "exports": list(rep.exports)}
            if not cancelled:
                outcomes.append(entry)
            say({"event": "shot_done", **entry})
            if cancelled or (rep.outcome == "failed" and stop_on_error):
                break
        kinds = ("ok", "needs_pick", "needs_review", "code_changed", "failed")
        counts = {k: sum(1 for o in outcomes if o["outcome"] == k) for k in kinds}
        if not outcomes and not cancelled and not json_out:
            typer.echo(f"no shots in state {', '.join(sorted(states or []))}"
                       + (f" ({left} in other states; afv shots queue IDS)" if left else ""))
        say(
            {"event": "batch_done", "batch_id": log.batch_id, "outcomes": outcomes, **counts,
             "states": sorted(states) if states else None, "not_selected": left, "cancelled": cancelled,
             "log": str(log.path)},
            ("batch stopped: " if cancelled else "")
            + f"batch: {counts['ok']} ok, {counts['needs_pick']} waiting for a pick, "
            + (f"{counts['needs_review']} waiting for a tracking review, " if counts["needs_review"] else "")
            + (f"{counts['code_changed']} to run again (code changed), " if counts["code_changed"] else "")
            + f"{counts['failed']} failed"
            + "".join(f"\n  {o['shot']}: {o['outcome']}" for o in outcomes if o["outcome"] != "ok"),
        )  # fmt: skip
    if cancelled:
        raise typer.Exit(130)
    if counts["failed"]:
        raise typer.Exit(EXIT_ERROR)
    if counts["needs_pick"] or counts["needs_review"] or counts["code_changed"]:
        raise typer.Exit(EXIT_NEEDS_INPUT)


def _next_batch_shot(
    proj: Project, states: set[str] | None, done: set[str], fixed: list[str] | None,
    outcomes: list[dict[str, Any]], say: Callable[..., None],
) -> tuple[Shot | None, int]:  # fmt: skip
    """The next shot to run (the shot list is read again each time) and how many shots are left out by
    state. A shot whose shot.json cannot be read gets a ``failed`` outcome once."""
    left = 0
    nxt: Shot | None = None
    for sid in fixed if fixed is not None else proj.shot_ids():
        if sid in done:
            continue
        try:
            sh = proj.shot(sid)
            st = sh.info.state
        except ProjectError as e:
            done.add(sid)
            entry = {"shot": sid, "outcome": "failed", "message": str(e), "exports": []}
            outcomes.append(entry)
            say({"event": "shot_done", **entry})
            continue
        if states is not None and st not in states:
            left += 1
        elif nxt is None:
            nxt = sh
    return nxt, left


# ---------------------------------------------------------------------------
# Batch log and lock (media/batch; afv serve's batch, batch_status, batch_cancel)
# ---------------------------------------------------------------------------

BATCH_SUBDIR = "batch"
BATCH_CURRENT = "current.json"
_BATCH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def batch_dir(proj: Project) -> Path:
    return proj.media_dir / BATCH_SUBDIR


def _batch_lock_path(proj: Project) -> Path:
    from afv.core.config import afv_home
    from afv.core.project import folder_identity

    d = afv_home() / "locks"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"batch-{folder_identity(proj.root)}.lock"


def new_batch_id() -> str:
    """<date>-<time>-<4 random hex digits>: unique also for two batches started in one second."""
    return time.strftime("%Y%m%d-%H%M%S") + f"-{os.urandom(2).hex()}"


class BatchSession:
    """The project's batch lock (one batch per project) and its event log ``media/batch/<id>.jsonl``;
    ``media/batch/current.json`` names the running (or last) batch: {batch_id, pid, started, log, ...}."""

    def __init__(self, proj: Project, batch_id: str | None, args: dict[str, Any]):
        self.proj = proj
        self.batch_id = batch_id or new_batch_id()
        if not _BATCH_ID.match(self.batch_id):
            raise ConfigError(f"bad batch id {self.batch_id!r}")
        self.args = args
        self.path = batch_dir(proj) / f"{self.batch_id}.jsonl"
        self._fd: int | None = None
        self._f: Any = None
        self._throttle = _Throttle(rate=1.0, repeat_s=5.0)

    def __enter__(self) -> BatchSession:
        import fcntl

        self._fd = os.open(_batch_lock_path(self.proj), os.O_RDWR | os.O_CREAT, 0o666)
        end = time.monotonic() + 2.0  # a batch_status may hold it for a moment
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > end:
                    os.close(self._fd)
                    cur = read_batch_current(self.proj) or {}
                    raise RunError(
                        f"a batch is already running for this project (batch {cur.get('batch_id', '?')}, pid "
                        f"{cur.get('pid', '?')}): wait for it, or stop it (afv serve batch_cancel)"
                    ) from None
                time.sleep(0.05)
        batch_dir(self.proj).mkdir(parents=True, exist_ok=True)
        from afv.core.cache import atomic_write_json

        now = now_iso()
        atomic_write_json(batch_dir(self.proj) / BATCH_CURRENT, {  # before the log: serve waits for the log
            "batch_id": self.batch_id, "pid": os.getpid(), "started": now,
            "log": str(self.path.relative_to(self.proj.root)), **self.args,
        })  # fmt: skip
        self._f = open(self.path, "a", encoding="utf-8")  # closed in __exit__
        self.write({"event": "batch_start", "batch_id": self.batch_id, "pid": os.getpid(), "started": now,
                    **self.args})  # fmt: skip
        return self

    def write(self, ev: dict[str, Any]) -> None:
        """Append one event (progress at most once a second per stage); never fails the batch."""
        if self._f is None or (ev.get("event") == "progress" and not self._throttle.allow(ev)):
            return
        with contextlib.suppress(OSError, ValueError):
            self._f.write(json.dumps(ev, default=str) + "\n")
            self._f.flush()

    def __exit__(self, *exc: Any) -> None:
        with contextlib.suppress(OSError, ValueError):
            if self._f is not None:
                self._f.close()
        if self._fd is not None:
            os.close(self._fd)  # releases the lock


def read_batch_current(proj: Project) -> dict[str, Any] | None:
    try:
        d = json.loads((batch_dir(proj) / BATCH_CURRENT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def batch_running(proj: Project) -> bool:
    """True while an ``afv batch`` holds the project's batch lock."""
    try:
        import fcntl
    except ImportError:
        return False
    try:
        fd = os.open(_batch_lock_path(proj), os.O_RDWR | os.O_CREAT, 0o666)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)  # shared: two status calls do not block each other
        return False
    except BlockingIOError:
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def batch_status(proj: Project) -> dict[str, Any]:
    """The running or last batch of the project, from its log: {running, batch_id, pid, started, log
    (relative), log_path, current: {shot, index, stage, frac, message} | None, outcomes: [{shot, outcome,
    message, exports}], done (the batch_done event) | None, finished, interrupted (stopped without
    batch_done, e.g. killed)}."""
    running = batch_running(proj)
    cur = read_batch_current(proj)
    out: dict[str, Any] = {"running": running, "batch_id": None, "pid": None, "started": None, "log": None,
                           "log_path": None, "current": None, "outcomes": [], "done": None, "finished": False,
                           "interrupted": False}  # fmt: skip
    if cur is None:
        return out
    path = proj.root / str(cur.get("log", ""))
    out.update(batch_id=cur.get("batch_id"), pid=cur.get("pid"), started=cur.get("started"),
               log=cur.get("log"), log_path=str(path))  # fmt: skip
    current: dict[str, Any] | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue  # a line being written
        kind = ev.get("event")
        if kind == "batch_shot":
            current = {"shot": ev.get("shot"), "index": ev.get("index"), "stage": None, "frac": None,
                       "message": None}  # fmt: skip
        elif kind in ("progress", "stage_start") and current and ev.get("shot") == current["shot"]:
            current.update(stage=ev.get("stage"), frac=ev.get("frac"), message=ev.get("message"))
        elif kind == "shot_done":
            entry = {k: ev.get(k) for k in ("shot", "outcome", "message", "exports")}
            if entry["outcome"] != "cancelled":
                out["outcomes"].append(entry)
            if current is not None and current["shot"] == ev.get("shot"):
                current = None
        elif kind == "batch_done":
            out["done"] = ev
    out["current"] = current if running else None
    out["finished"] = out["done"] is not None
    out["interrupted"] = not running and out["done"] is None
    return out


def _is_batch_process(proj: Project, pid: int, batch_id: str) -> bool:
    """``pid`` is an ``afv batch`` of this batch: its command line has ``batch`` and, when it was started
    with ``--batch-id`` (by afv serve), that id; started without one (from a terminal), it must hold the
    project's batch lock (``/proc/locks``). Without /proc (not Linux) the pid is trusted."""
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace").split("\0")
    except FileNotFoundError:
        return False
    except OSError:
        return True
    if "batch" not in args:
        return False
    if "--batch-id" in args:
        k = args.index("--batch-id")
        return k + 1 < len(args) and args[k + 1] == batch_id
    try:
        st = os.stat(_batch_lock_path(proj))
        dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
        for line in Path("/proc/locks").read_text().splitlines():
            f = line.split()
            if len(f) >= 6 and f[1] == "FLOCK" and f[4] == str(pid) and f[5] == dev:
                return True
        return False
    except OSError:
        return True


def batch_cancel(proj: Project, wait_s: float = 60.0) -> dict[str, Any]:
    """Stop the project's running batch (SIGTERM: a clean stop, finished shots kept); waits up to
    ``wait_s`` for it to end. Returns ``batch_status`` plus ``cancelled`` (a running batch was stopped)."""
    st = batch_status(proj)
    if not st["running"] or not isinstance(st["pid"], int):
        return {**st, "cancelled": False}
    if not _is_batch_process(proj, st["pid"], str(st["batch_id"])):  # current.json is an old batch's?
        msg = f"no running batch found (pid {st['pid']} is not afv batch {st['batch_id']}); none stopped"
        return {**st, "cancelled": False, "message": msg}
    try:
        os.kill(st["pid"], signal.SIGTERM)
    except ProcessLookupError:
        return {**batch_status(proj), "cancelled": False}
    end = time.monotonic() + wait_s
    while batch_running(proj) and time.monotonic() < end:
        time.sleep(0.2)
    return {**batch_status(proj), "cancelled": True}


@app.command()
@_handled
def export(
    shot: Annotated[str, typer.Argument(help="Shot id.")],
    preset: Annotated[
        str | None,
        typer.Option(help="Engine preset, e.g. unity_humanoid, unreal, godot, blender (saved to the shot)."),
    ] = None,
    fps: Annotated[
        float | None,
        typer.Option(help="Export frame rate (saved to the shot); default: the 30 Hz shot timeline.", min=1,
                     max=240),
    ] = None,
    sets: SetOpt = None,
    yes: YesOpt = False,
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Run the pipeline through export with an engine preset; files land in shots/<id>/exports/.

    The shot's speed factor (footage speed, afv shots add --speed) is undone while retiming to --fps, so
    a slow-motion replay exports at real time. --preset and
    --fps are saved to the shot like --set, so a later 'afv run' exports the same way.
    Like afv run, stops after detect_track (exit code 3) on unreviewed tracking QC hints unless --yes."""
    sh = _project(project).shot(shot)
    sets = list(sets or [])
    if preset is not None:
        sets.append(f"export.preset={json.dumps(preset)}")
    if fps is not None:
        sets.append(f"export.fps={fps!r}")
    runner = _make_runner(sh, json_out, sets=sets, pause_on_hints=True, yes=yes)
    with runner.shot_lock():  # validated, then saved only once this process owns the shot
        runner.check_edits("export")
        _settings_overrides(sh, None, sets, runner, persist=True)
        report = runner.run(to="export")
    if report.outcome == "ok" and not report.exports:
        meta = sh.meta("export") or {}
        report.exports = list(meta.get("exports", []))
    if report.outcome == "ok" and not json_out:
        typer.echo(f"exports in {sh.exports_dir}")
    raise typer.Exit(_finish(report, json_out))


def _take_failed(json_out: bool, e: Exception) -> typer.Exit:
    with contextlib.suppress(OSError):
        if json_out:
            typer.echo(json.dumps({"event": "error", "message": str(e)}))
        typer.echo(f"error: {e}", err=True)
    return typer.Exit(EXIT_ERROR)


TakePresetOpt = Annotated[
    str | None, typer.Option(help="Export preset whose rig the take uses (default: the shot's export one).")
]


@app.command()
@_handled
def take(
    shot: Annotated[str, typer.Argument(help="Shot id.")],
    preset: TakePresetOpt = None,
    out: Annotated[Path | None, typer.Option("--out", help="Take folder (default: shots/<id>/take).")] = None,
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Write the shot's take package for Blender (shots/<id>/take/: JSON + NPZ, read by the AFV add-on)."""
    from afv import take as tk

    sh = _project(project).shot(shot)
    try:
        with Runner(sh).shot_lock():  # no afv run writes the shot's results meanwhile
            path = tk.make_take(sh, out, preset=preset)
    except tk.TakeError as e:
        raise _take_failed(json_out, e) from None
    s = tk.take_summary(path)
    _emit(json_out, {"event": "take", "path": str(path), **s},
          f"take {s['take_id']}: {len(s['fighters'])} fighters, {s['frames']} frames, rig {s['preset']}, "
          f"{s['spans']} QC spans ({s['bad_spans']} bad)\n  {path}"
          + "".join(f"\n  note: {n}" for n in s["notes"])
          + (f"\n  warning: {s['stale_summary']}" if s["stale_summary"] else "")
          + "".join(f"\n    {n}" for n in s["stale"]))


@app.command()
@_handled
def blender(
    shot: Annotated[str, typer.Argument(help="Shot id.")],
    out: Annotated[
        Path | None, typer.Option("--out", help="The .blend to write (default: shots/<id>/take/take.blend).")
    ] = None,
    preset: TakePresetOpt = None,
    check: Annotated[
        bool, typer.Option("--check", help="Also verify the scene: joints vs the take (mm), overlay (px).")
    ] = False,
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Build the take as a .blend in headless Blender (open it with File > Open; no add-on needed)."""
    from afv import take as tk
    from afv.export.blender import BlenderError

    sh = _project(project).shot(shot)
    try:
        with Runner(sh).shot_lock():
            rep = tk.blend_shot(sh, out, preset=preset, check=check)
    except (tk.TakeError, BlenderError) as e:
        raise _take_failed(json_out, e) from None
    text = f"wrote {rep['blend']}" + (f"\n  warning: {rep['stale_summary']}" if rep["stale_summary"] else "")
    text += "".join(f"\n    {n}" for n in rep["stale"])
    for role, c in (rep.get("check") or {}).get("fighters", {}).items():
        px = f", overlay max {c['px_max']:.3f} px" if c.get("px_max") is not None else ""
        text += f"\n  check {role}: joints max {c['max_mm']:.4f} mm{px}"
    _emit(json_out, {"event": "blend", "path": rep["blend"], "take": rep["take"],
                     "summary": rep["summary"], "check": rep.get("check"), "stale": rep["stale"],
                     "stale_summary": rep["stale_summary"]}, text)


@app.command()
@_handled
def retarget(
    shot: Annotated[str | None, typer.Argument(help="Shot id.")] = None,
    character: Annotated[
        Path | None, typer.Option("--character", help="The character: an .fbx, .glb/.gltf or .blend file.")
    ] = None,
    map_file: Annotated[
        Path | None,
        typer.Option("--map", help="Bone map (*.afvmap.json); default: guessed from the bone names."),
    ] = None,
    role: Annotated[str, typer.Option("--role", help="Fighter: A, B or both.")] = "both",
    preset: Annotated[
        str | None,
        typer.Option(help="Engine preset whose FBX exporter settings are used (only those: no root "
                     "bone, root-motion mode, armature rename or Unreal cm scale; default: the shot's export "
                     "one)."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Output folder (default: shots/<id>/exports/retarget).")
    ] = None,
    no_lock_feet: Annotated[
        bool, typer.Option("--no-lock-feet", help="Don't hold planted feet in place (no foot IK).")
    ] = False,
    save_map: Annotated[
        Path | None,
        typer.Option("--save-map", help="Write the bone map used (guessed or --map) to this *.afvmap.json."),
    ] = None,
    check: Annotated[
        bool,
        typer.Option("--check", help="Re-import each FBX and compare it with the retarget (bone heads, mm)."),
    ] = False,
    list_presets: Annotated[
        bool,
        typer.Option("--list-presets", help="Only print the engine presets and their FBX exporter settings."),
    ] = False,
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Retarget the shot's fighters onto your own character and export one FBX per fighter.

    Builds the take headless (the shot's take/ folder is not touched), imports the character, maps
    its bones (--map, else a guess from the bone names), retargets each fighter (--role) and writes
    <shot>_<role>_<character>.fbx (its clip has the same name) with the preset's FBX exporter
    settings only: unlike afv export, no root bone is added, no root-motion mode applies, the armature
    keeps its name and the scene is not rescaled for Unreal. The character keeps its own size. Both
    fighters share the take's world origin. Exits 1 listing the bones to map when a required bone is
    unmapped (--save-map writes the guess so you can complete it and pass it with --map). Warns, like
    afv take, when the shot's results are older than its settings or code."""
    from afv import retarget as rg
    from afv import take as tk
    from afv.export.blender import BlenderError

    if list_presets:
        _emit(json_out, {"event": "presets", "presets": rg.presets_payload()}, rg.format_presets())
        return
    if shot is None or character is None:
        raise ConfigError("afv retarget needs a shot id and --character (or --list-presets)")
    sh = _project(project).shot(shot)
    try:
        roles = rg.parse_roles(role)
        with Runner(sh).shot_lock():
            rep = rg.retarget_shot(sh, character, map_path=map_file, roles=roles, preset=preset, out_dir=out,
                                   lock_feet=not no_lock_feet, save_map=save_map, check=check)
    except rg.RetargetError as e:
        with contextlib.suppress(OSError):
            saved = e.result.get("saved_map")
            if json_out:
                typer.echo(json.dumps({"event": "error", "message": str(e), "missing": e.missing,
                                       "saved_map": saved}, default=str))
            typer.echo(f"error: {e}" + (f"\n  the bone map was written to {saved}: map those bones there and "
                                         f"pass it with --map" if saved and e.missing else ""), err=True)
        raise typer.Exit(EXIT_ERROR) from None
    except (tk.TakeError, BlenderError) as e:
        raise _take_failed(json_out, e) from None
    lines = [f"retargeted {', '.join(rep['files'])} onto {rep['target']} ({rep['bones']} bones, map: "
             f"{rep['map_source']}, preset {rep['preset']}) in {rep['seconds']:.1f} s"]
    for n in rep["import"]["notes"]:
        lines.append(f"  note: {n}")
    for r, res in rep["roles"].items():
        _emit(json_out, {"event": "retarget", "role": r, "file": rep["files"][r], **res})
        lines += ["  " + ln for ln in rg.format_report(r, res["report"])]
        lines.append(f"   wrote {rep['files'][r]}")
        c = res.get("check")
        if c:
            lines.append(f"   check: re-imported bone heads max {c['max_mm_fighter']:.4f} mm at fighter "
                         "scale "
                         f"({c['max_mm']:.4f} mm in the character's world), {c['bones']} bones, "
                         f"{c['frames']} frames, file {c['file_fps']:g} fps, frames {c['file_frames'][0]}-"
                         f"{c['file_frames'][1]}, clip {res['export'].get('take')!r}")
    if rep.get("saved_map"):
        lines.append(f"  bone map written to {rep['saved_map']}")
    if rep.get("stale_summary"):
        lines.append(f"  warning: {rep['stale_summary']}")
    _emit(json_out, {"event": "result", "outcome": "ok", **{k: v for k, v in rep.items() if k != "roles"}},
          "\n".join(lines))


@app.command()
@_handled
def models(
    check: Annotated[
        bool,
        typer.Option(
            "--check",
            help="Exit 1 on a licence whose commercial use is unknown or not allowed (NFR-12), a stage "
            "without licence records, or a broken environment.",
        ),
    ] = False,
    json_out: JsonOpt = False,
) -> None:
    """List model environments, stage implementations and their licences."""
    from afv.models.envs import check_env, load_envs

    modules = load_stage_modules()
    envs = load_envs()
    problems: list[str] = []
    warnings: list[str] = []
    for m, st in modules.items():
        if st.startswith("import failed"):
            problems.append(f"{m}: {st}")
        elif st != "ok":
            warnings.append(f"{m}: {st}")
    used_envs: set[str] = set()
    impls = []
    for name in STAGE_ORDER:
        for impl in stage_mod.implementations(name):
            cls = stage_mod.get_stage(name, impl)
            env = str(getattr(cls, "env", "?"))
            used_envs.add(env)
            if env not in envs:
                problems.append(f"{name}/{impl}: unknown environment {env!r}")
            recs = list(getattr(cls, "licences", ()))
            p, w = licence_problems(f"{name}/{impl}", recs, needs_licence=env != "cpu")
            problems += p
            warnings += w
            impls.append(
                {
                    "stage": name,
                    "impl": impl,
                    "version": getattr(cls, "version", "?"),
                    "env": env,
                    "weight_hash": getattr(cls, "weight_hash", ""),
                    "licences": [dataclasses.asdict(r) for r in recs],
                }
            )
    env_rows = []
    for name, spec in envs.items():
        issues = check_env(spec)
        (problems if name in used_envs else warnings).extend(issues)
        env_rows.append({"env": name, "executable": str(spec.executable), "gpu": spec.gpu, "ok": not issues})
    if json_out:
        typer.echo(
            json.dumps(
                {
                    "event": "models",
                    "modules": modules,
                    "envs": env_rows,
                    "stages": impls,
                    "problems": problems,
                    "warnings": warnings,
                },
                default=str,
            )
        )
    else:
        typer.echo("environments:")
        for r in env_rows:
            typer.echo(
                f"  {r['env']:<8} {'ok ' if r['ok'] else 'MISSING'} {'gpu' if r['gpu'] else '   '} "
                f"{r['executable']}"
            )
        typer.echo("stage modules:")
        for m, st in modules.items():
            typer.echo(f"  {m:<20} {st}")
        typer.echo("stage implementations:")
        for r in impls:
            typer.echo(f"  {r['stage']:<13} {r['impl']:<12} v{r['version']:<6} env {r['env']}")
            for lic in r["licences"]:
                ok = {True: "commercial ok", False: "NOT commercial", None: "UNKNOWN"}[
                    lic["commercial_output_ok"]
                ]
                typer.echo(
                    f"      {lic['component']}: code {lic['code_licence']}, weights "
                    f"{lic['weight_licence']} ({ok})"
                )
        for w in warnings:
            typer.echo(f"warning: {w}")
        for p in problems:
            typer.echo(f"problem: {p}")
    if check and problems:
        raise typer.Exit(EXIT_ERROR)


@app.command()
@_handled
def provenance(
    shot: Annotated[str | None, typer.Argument(help="Shot id (default: all).")] = None,
    no_hash: Annotated[bool, typer.Option("--no-hash", help="Do not hash export files.")] = False,
    project: ProjectOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Sources, times and models behind each shot and export."""
    proj = _project(project)
    ids = [proj.resolve_shot_id(shot)] if shot else None
    prov = project_provenance(proj, ids, hash_exports=not no_hash)
    _emit(json_out, {"event": "provenance", **prov}, format_provenance(prov))


@app.command()
def serve() -> None:
    """Long-lived worker for the Blender add-on: JSON-line requests on stdin, replies on stdout.

    One worker per Blender session, for any project (see afv.serve for the protocol). It exits when
    stdin closes."""
    from afv.serve import serve as serve_main

    raise typer.Exit(serve_main())


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
