"""``afv serve``: one long-lived afv worker for a Blender session (JSON lines on stdin/stdout).

The add-on starts one worker (``afv serve``, in WSL) and talks to it for any project. afv modules stay
imported, CPU stages run in this process, GPU stages through their usual worker subprocesses.

Protocol (docs/BUILD_PLAN.md, round 4 "Shared formats")
-------------------------------------------------------
One JSON object per line each way, UTF-8, at most ``MAX_LINE`` bytes (4 MiB) per request line. Request::

    {"id": n, "cmd": "<cmd>", "args": {...}}

``id`` is a string, an integer or null. Replies carry the request's ``id`` and an ``event``:
``progress`` | ``log`` | ``tracking_review`` | ``result`` | ``error``. Every request ends with exactly
one ``result`` or ``error``. A line that is not such a request (not JSON, too long, too deeply nested,
``NaN``/``Infinity`` anywhere, a bad ``id``, no ``cmd``) gets an ``error`` with its ``id`` when it has
a valid one, else ``null``; the worker carries on. Numbers in replies are plain JSON numbers.

Commands and their ``args`` (``project`` = the project folder, a WSL path or a Windows path such as
``D:\\...`` or ``\\\\wsl.localhost\\Ubuntu\\...``; ``shot`` = the shot id):

- ``ping`` {}: result ``{"pong": true, "protocol": 1, "pid", "afv_version", "started", "restarts",
  "running": [ids], "queued": [ids]}``. Answered at once.
- ``status`` {project, shot, to?, from?, frames?}: result = ``afv status --json``'s plan
  (``RunReport.to_dict()``: ``shot``, ``outcome: "planned"``, ``stages: [{name, impl, state, reason,
  key, runtime_s, range}]``, ``message``, ``review``, ...).
- ``run`` {project, shot, from?, to? (default export), frames? ("a:b" or [a, b]), sets? (list of
  "stage.key=value" strings as ``--set``, or {"stage.key": value}), yes?, code_changes? (report |
  rerun | accept), camera?, seed?, deterministic?}: as ``afv run`` (settings are saved to the shot;
  pauses on tracking hints unless ``yes``). Result = the CLI's final ``result`` event
  (``RunReport.to_dict()``: ``outcome`` ok | needs_pick | needs_review | code_changed | failed,
  ``stages``, ``message``, ``exports``, ``review``, ...) plus ``exit_code`` (as the CLI: 0, 3 waiting
  for the user, 1) and ``elapsed_s``. A run that fails inside a stage is a ``result`` with outcome
  ``failed`` (like the CLI), not an ``error``.
- ``take`` {project, shot, preset?, out?, check_plan? (default true)}: as ``afv take``; result
  ``{"path", **take_summary}``. ``check_plan: false`` skips asking the runner which stages ``afv run``
  would redo (a code-hash scan of every stage's sources): for a take right after a ``run`` the client
  just made.
- ``export`` {project, shot, preset?, fps?, sets?, yes?}: as ``afv export``; result like ``run``.
- ``edit`` {project, shot, name, data}: validate ``data`` against the edit format
  (``afv.core.edits``: contacts, pins, hands, ids, fighters) and write ``edits/<name>.json``
  atomically. contacts/pins/hands/ids: top-level keys of the existing file that ``data`` leaves out
  are kept (send ``"spans": []`` to clear the spans); fighters: ``data`` replaces the whole file.
  ``data: null`` deletes the file. Result ``{"path", "name", "removed", "stages": [stages that read
  it]}``.
- ``presets`` {}: result ``{"presets": [every engine preset of afv.export.presets, as
  ``EnginePreset.to_dict()``: name, label, formats, ..., "fbx": {FBX exporter arguments}, "glb": {...}]}``
  (the add-on's Export character uses the FBX arguments). Answered at once, like ``ping``.

Shot browser (round 6; ``docs/BUILD_PLAN.md`` "Round 6 assignments"). Times are seconds (a number) or a
time string ("62.5", "1:02.5"); ``state`` is candidate | queued | skipped. A *row* is a shot as
``afv shots list --json`` shows it (``afv.cli.shot_row``: shot.json fields, ``end``, ``picked``,
``pick_stale``, ``scan`` {status ok | stale | none | bad, frame, people, fighter_sized, fighters_likely,
thumbnail}, ``exported``); paths in it are relative to the project folder. The *shot list* is
``{"project", "root_win", "source": {path, path_win, sha256, title, fps, width, height, duration,
num_frames}, "proxy" (media/proxy.mp4 or null), "cuts" (media/cuts.json or null), "scan_dir", "counts":
{state: n}, "shots": [rows]}``.

- ``init`` {video, project (the new folder), title?, url?, detect? (default true), proxy? (true | false |
  "auto", default: only for a source taller than 720 px), min_confidence? (0.2)}: as ``afv init --detect``
  (probe, hash, cut detection: candidate shots and media/cuts.json; the proxy alongside). ``progress``
  events (stage ``init``, ``frac`` null, every second), a ``log`` ``kind: "detect_done"`` (the summary
  below) and, when a proxy was made, ``kind: "proxy"`` {path}; result ``{"created",
  "sha256", "warning", "detect": {found, added, skipped_overlap, shots, cuts, cut_count, seconds},
  "elapsed_s", **shot list}``. An existing project is an error (open it with ``shots``).
- ``shots`` {project, state? (a state or a list), stages? (false)}: result = the shot list (``counts`` over
  every shot; ``stages: true`` adds ``stored`` per row). Reads run in parallel threads: well under 1 s for
  126 shots on a Windows drive. A shot whose shot.json cannot be read is a row ``{"id", "error":
  message, "state": null, ...}`` (always listed, whatever ``state``); every other row has ``"error":
  null``. ``fighters_likely`` and ``fighter_sized`` are computed from the scan's people with the current
  rule (a hint: sort and highlight by it, never hide shots because of it). A scan is ``stale`` when the
  shot was trimmed or its ``detect_track.det_threshold`` changed since it was scanned.
- ``shot_add`` {project, start, end | duration, speed?, notes?, state? (queued), shot_id?}: as ``afv shots
  add``; result ``{"shot", "row"}``.
- ``shot_set`` {project, shot, start?, end?, duration?, speed?, notes?, state?}: as ``afv shots set``
  (``start`` alone keeps the end); result ``{"shot", "changed": [fields], "notes_for_user": [...],
  "row"}``. An interval or speed change takes the shot lock: refused while the shot runs.
- ``shots_detect`` {project, min_confidence?, dry_run?}: as ``afv shots detect``; heartbeat ``progress``
  (stage ``detect``), ``log`` ``kind: "detect_done"`` and, with ``dry_run``, one ``kind: "shot_found"``
  {start, duration, confidence} per segment that would be added; result ``{"detect": summary, **shot
  list}``.
- ``shots_scan`` {project, shots? [ids], state? (default candidate and queued), force?}: as ``afv shots
  scan``: ``progress`` (stage ``scan``: frames 0..0.5, people 0.5..1) and a ``log`` ``kind: "scanned"``
  ``{"shot", "row"}`` as each shot's scan is written; result ``ScanReport.to_dict()`` (scanned,
  up_to_date, failed {shot: error}, error, seconds) plus ``shots`` and ``elapsed_s``. Cancellable: the
  worker is stopped, the scans written so far are kept (the error carries the ``report``); scanning
  again does the rest. A worker failure is an ``error`` with the ``report``.
- ``pick`` {project, shot, fighters? [a, b], extra? [..], frame?, keep_prompts?}: with ``fighters``, as
  ``afv pick --fighters [--frame]``: the numbers are the stored scan's when ``frame`` is its frame (or no
  frame is given and the shot has no detect_track prompt): no GPU; result ``{"shot", "source": scan |
  stage | detected, "frame", "fighters", "extra", "edit", "dropped", "data", "row"}``. Without
  ``fighters``: the prompt to show, ``{"shot", "source", "frame", "prompt": {frame, boxes, scores, ...},
  "image", "image_win", "current"}``; a shot with no prompt yet is scanned first (one frame, GPU,
  ``progress`` events), a ``frame`` that is neither the scan's nor the prompt image's is detected (GPU).
  detect_track's prompt image counts only while its result is for the shot's interval, source and
  threshold (after a trim the up-to-date scan is used). Cancellable while it scans or detects: the error
  then has ``"cancelled": true`` and nothing was written.
- ``batch`` {project, state? (a state or a list; default queued), all?, shots? [ids: only these, in this
  order], to? (export), yes?, code_changes?, sets?, camera?, stop_on_error?}: the batch **survives the
  client**. It starts a detached ``afv batch --json --batch-id ID ...`` (its own session, not a child of
  this worker: closing Blender, the worker or this request does not stop it) unless the project already
  has a running batch, which it then attaches to (``attached: true``; the arguments are not used). The
  batch reads the shot list again after each shot (a shot queued meanwhile runs too; it stops when no
  shot in ``state`` is left) and writes every event to ``media/batch/<ID>.jsonl``; this request streams
  that log while the client is connected (from its start, also when attaching): ``log`` ``kind:
  "batch_started"`` {batch_id, attached, log}, ``kind: "batch_start"`` {batch_id, pid, started, to,
  states, shots}, per shot ``kind: "batch_shot"`` {shot, index}, the runner's events of the shot
  (``progress`` with ``shot`` and ``stage``, at most once a second per stage in the log;
  ``tracking_review``; the others as ``log`` with their ``kind``), ``kind: "result"`` (the shot's
  ``RunReport``), ``kind: "shot_done"`` {shot, outcome, message, exports}; a shot.json that cannot be read
  gives a ``shot_done`` with outcome ``failed``. Result ``{"batch_id", "attached", "outcomes": [{shot,
  outcome, message, exports}], "ok", "needs_pick", "needs_review", "code_changed", "failed", "shots",
  "exit_code" (as the CLI: 0, 3 waiting for the user, 1), "cancelled": false, "log", "elapsed_s"}``.
  Cancelling this request stops the batch like ``batch_cancel`` (error ``cancelled`` with the
  ``report``); a worker shutting down (the client went away) or about to restart for a code change
  ends the request with an error ``{"cancelled": true, "detached": true, "batch_id"}`` and the batch goes
  on (send ``batch`` again to attach). A batch that dies without
  finishing (killed) ends the request with an error and its ``batch_status`` as ``report``. A batch that
  cannot start is an error with the last lines of ``media/batch/<ID>.stderr``. ``afv batch`` started
  from a terminal takes the same lock and writes the same log: ``batch`` attaches to it, ``batch_status``
  and ``batch_cancel`` see it.
- ``batch_status`` {project}: result ``{"running", "batch_id", "pid", "started", "log" (relative),
  "log_path", "current": {shot, index, stage, frac, message} | null, "outcomes": [{shot, outcome,
  message, exports}] so far, "done": the ``batch_done`` event | null, "finished", "interrupted" (it
  stopped without finishing)}`` of the running or last batch (all null / empty when there was none).
  The ``batch_done`` event: {batch_id, outcomes, ok, needs_pick, needs_review, code_changed, failed,
  states, not_selected, cancelled, log}.
- ``batch_cancel`` {project, wait_s? (60)}: stops the running batch cleanly (SIGTERM: the running shot's
  stage stops, its previous results are kept, finished shots are kept; it writes ``batch_done`` with
  ``cancelled: true``) and waits up to ``wait_s`` for it to end; result = ``batch_status`` plus
  ``"cancelled"`` (false when no batch was running).

``init`` runs on a queue of its folder; the other project commands on queues of their project: ``shots``
alone, ``shot_add``/``shots_detect`` together, ``shots_scan`` alone, ``batch_status`` alone,
``batch_cancel`` alone, each ``batch`` stream on its own, so a listing or a new shot never waits behind a
scan or a batch. ``shot_set`` and ``pick`` use the shot's queue. Shot changes are safe against each other
and against CLI commands: ``add_shot`` holds a project lock while it chooses the id and writes the shot,
``shot_set`` a per-shot lock while it reads and writes shot.json (only the changed keys are written).

- ``cancel`` {target_id}: result ``{"target_id", "state": "queued" | "running" | "finished" |
  "unknown"}``, answered at once. The target then ends with an ``error`` carrying ``"cancelled":
  true``: a queued one at once; a running run/export cleanly (before its next stage, at a CPU
  stage's next progress call, or by stopping its worker subprocesses, also ones it starts after the
  cancel or while waiting for the GPU: the stage's work folder is dropped, its previous result kept),
  with the partial ``report``; a running batch stream by stopping the batch (``batch_cancel``); a
  running shots_scan or pick by stopping its worker; a running
  take/edit/status (and the other new commands) when it finishes (it is not interrupted, nor are its
  helper processes such as Blender; its error says ``"completed": true``, i.e. its effect stays).
  ``finished``: the target had already ended (also for requests answered before a code-change
  restart).
- ``shutdown`` {}: cancels everything (queued requests get their error at once), replies ``result
  {"shutdown": true}``, waits up to ``SHUTDOWN_WAIT_S`` for running requests (each ends with its
  error; one still running then gets an ``error`` "the worker shut down") and exits 0.

Events while a request runs (run/export): the runner's ``--json`` events with the request ``id``
added. ``progress`` (``{"stage", "frac": 0..1 | null, "message"}``, at most 4 per second per stage)
and ``tracking_review`` keep their names; every other runner event (``plan``, ``stage_start``,
``stage_skip``, ``stage_done``, ``stage_failed``, ``relevel``, ``needs_pick``, ``code_changed``,
``cancelled``, ``done``) is sent as ``event: "log"`` with its name in ``kind`` and its payload
unchanged. Warnings logged while a request runs are ``log`` events with ``kind: "warning"``.

Ordering: requests for one shot run one at a time, in arrival order (``status`` and ``edit`` too, so
an edit sent after a run applies after it); different shots run in parallel. A shot is identified by
its folder (``afv.core.project.folder_identity``), so two spellings of one path (the Windows drives
are case-insensitive) share a queue, and the shot lock (``Runner.shot_lock``) works across spellings
and against CLI runs. ``ping``, ``presets``, ``cancel`` and ``shutdown`` are answered at once, always.

Speed: afv modules stay imported, and file stats for the code-hash checks are reused for
``STAT_TTL_S`` seconds across requests (``afv.core.codehash.set_stat_ttl``), so a request right after
another does not stat every source file again on the Windows drive.

Robustness: the worker exits when stdin closes (the add-on or Blender went away) or on SIGTERM,
cancelling what runs. Nothing but protocol lines is written to stdout (stray prints go to stderr).
When afv code changed since this worker started (a run's outcome ``code_changed``: stage code; or,
checked before every request, a change to any afv module the worker has loaded: runner, serve, cli,
take, export helpers, ...), the request gets a ``log`` event ``kind: "restart"``; the worker stops
starting requests (it keeps reading stdin and answering ping/cancel/shutdown), waits until none is
running, checks in a subprocess that the new code imports (``AFV_SERVE_CHECK``), re-executes itself
with the same stdin/stdout (fresh code) and runs every such request again, then the queued ones, in
arrival order: the client sees each request end once, and the worker's ``pid`` stays the same
(``restarts`` counts). If the new code does not import, those requests end with an ``error`` "afv code
has an error: ...; fix it and retry" and the worker carries on with the code it has; the next request
checks again.

``status`` takes the shot lock when it is free (a consistent view); while another request or process
runs the shot it plans without it and asks once more when a result looked missing (a commit or prune
in progress).

Warnings at start: stage modules that do not import; no ``setpriv`` (workers would not die with a
killed worker, see ``afv.models.envs``).

Additions to the plan's protocol (no deviation): ``exit_code``/``elapsed_s`` in run results, the
``kind`` field on ``log`` events, ``cancelled``/``completed``/``report`` on cancel errors, Windows
project paths, ``sets`` as a dict, the ``restart`` log event, ``check_plan`` on ``take``; round 5: the
``presets`` command; round 6: the ``shot_add`` command (the browser's Add shot here), ``pick`` without
``fighters`` (the prompt to show), ``shots`` on ``batch``, ``stages`` on ``shots``, ``wait_s`` on
``batch_cancel``, the ``batch_started`` log event.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import math
import os
import re
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import weakref
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import afv
from afv.core.cache import to_jsonable
from afv.core.config import ConfigError
from afv.core.project import Project, ProjectError, Shot, folder_identity, now_iso
from afv.core.runner import DEFAULT_TARGET, RunError, Runner, RunReport, load_stage_modules, parse_frames
from afv.core.stage import STAGE_ORDER, StageError

PROTOCOL = 1
SHOT_CMDS = ("status", "run", "take", "export", "edit", "shot_set", "pick")
# project commands -> their queue ("lane") per project: requests of one lane run one at a time, lanes in
# parallel, so a listing or a new shot never waits behind a scan or a batch
PROJECT_CMDS = {"init": "init", "shots": "list", "shot_add": "edit", "shots_detect": "edit",
                "shots_scan": "scan", "batch": "batch", "batch_status": "batch_status",
                "batch_cancel": "batch_cancel"}  # fmt: skip
RESUME_ENV = "AFV_SERVE_RESUME"  # file with requests to replay after a code-change restart
RESTARTS_ENV = "AFV_SERVE_RESTARTS"
CHECK_ENV = "AFV_SERVE_CHECK"  # set: import everything a worker would, then exit (0 = the code imports)
CHECK_TIMEOUT_S = 300.0
CANCELLABLE = ("run", "export", "shots_scan", "batch", "pick")  # their stages and workers stop on cancel
STAT_TTL_S = 3.0  # code-hash file stats reused across requests for this long (see "Speed")
SHUTDOWN_WAIT_S = 60.0  # how long shutdown waits for running requests to stop
MAX_LINE = 4 << 20  # bytes per request line
EXIT_CODES = {"ok": 0, "needs_pick": 3, "needs_review": 3, "code_changed": 3}
NL = b"\n"
EXPECTED = (ProjectError, ConfigError, RunError, StageError, OSError, ValueError)
RESTART_MESSAGE = "afv code changed since this worker started: restarting it and running the request again"

log = logging.getLogger("afv.serve")


class RequestError(Exception):
    """A request that cannot be served; the message goes into its ``error`` reply."""


@dataclass(eq=False)
class Job:
    id: Any
    cmd: str
    args: dict[str, Any]
    seq: int
    line: bytes  # the request as received (replayed after a code-change restart)
    key: str = ""  # the shot folder's identity (queue key)
    where: tuple[str, str] = ("", "")  # (project root, shot id)
    replayed: bool = False
    state: str = "queued"  # queued | running | restart (waiting to be replayed) | done
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: int | None = None


# ---------------------------------------------------------------------------
# Child processes (so cancel can stop a request's worker subprocesses)
# ---------------------------------------------------------------------------

_CHILDREN: dict[int, weakref.WeakSet[subprocess.Popen[Any]]] = {}
_THREAD_CANCEL: dict[int, threading.Event] = {}  # thread -> cancel event of the request it runs
_CHILDREN_LOCK = threading.Lock()
_ORIGINAL_POPEN = subprocess.Popen


class _TrackedPopen(_ORIGINAL_POPEN):  # type: ignore[misc, valid-type]
    """``subprocess.Popen`` that remembers which thread started it (installed by ``Server``) and
    stops the process at once when that thread's request is already cancelled."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        ident = threading.get_ident()
        with _CHILDREN_LOCK:
            _CHILDREN.setdefault(ident, weakref.WeakSet()).add(self)
            ev = _THREAD_CANCEL.get(ident)
        if ev is not None and ev.is_set():
            _stop(self)


def _stop(p: subprocess.Popen[Any]) -> bool:
    if p.poll() is not None:
        return False
    with contextlib.suppress(OSError):
        if os.getpgid(p.pid) == p.pid:  # a session/group leader (afv's workers): the whole group
            os.killpg(p.pid, signal.SIGTERM)
        else:
            p.terminate()
        return True
    return False


def _stop_children(thread: int | None) -> int:
    """SIGTERM the process groups (or processes) started by ``thread`` that still run."""
    if thread is None:
        return 0
    with _CHILDREN_LOCK:
        procs = list(_CHILDREN.get(thread, ()))
    return sum(_stop(p) for p in procs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WIN_DRIVE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_WSL_UNC = re.compile(r"^[\\/]{2}wsl(?:\.localhost|\$)[\\/][^\\/]+[\\/]?(.*)$", re.IGNORECASE)


def wsl_path(text: str) -> Path:
    """A project path as given by the add-on: a WSL path, ``D:\\...`` (-> /mnt/d/...) or
    ``\\\\wsl.localhost\\<distro>\\...`` (-> /...)."""
    if "\0" in text:
        raise RequestError("a path must not contain NUL characters")
    if m := _WIN_DRIVE.match(text):
        return Path(f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/"))
    if m := _WSL_UNC.match(text):
        return Path("/" + m.group(1).replace("\\", "/"))
    return Path(text)


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not allowed (finite numbers only)")


def parse_request(text: str) -> Any:
    """JSON without NaN/Infinity; RecursionError (absurd nesting) becomes ValueError."""
    try:
        return json.loads(text, parse_constant=_no_constant)
    except RecursionError:
        raise ValueError("nested too deeply") from None


def _valid_id(rid: Any) -> bool:
    return rid is None or isinstance(rid, str) or (isinstance(rid, int) and not isinstance(rid, bool))


def _json_default(obj: Any) -> Any:
    out = to_jsonable(obj)  # numpy scalars and arrays -> numbers and lists, Paths -> str, ...
    return str(out) if out is obj else out


def _finite(v: Any, where: str) -> None:
    if isinstance(v, float) and not math.isfinite(v):
        raise RequestError(f"{where}: {v} is not allowed (finite numbers only)")
    if isinstance(v, dict):
        for k, x in v.items():
            _finite(x, f"{where}.{k}")
    elif isinstance(v, list):
        for x in v:
            _finite(x, where)


def _sets(value: Any) -> list[str]:
    """``sets`` as ``--set`` strings: a list of "stage.key=value" or a {"stage.key": value} dict."""
    from afv.core.config import parse_set

    if value is None:
        return []
    if isinstance(value, dict):
        out = [f"{k}={json.dumps(v)}" for k, v in value.items()]
    elif isinstance(value, list) and all(isinstance(v, str) for v in value):
        out = list(value)
    else:
        raise RequestError('"sets" must be a list of "stage.key=value" strings or a {"key": value} dict')
    _finite(parse_set(out), "sets")
    return out


def _arg(args: dict[str, Any], name: str, kind: type | tuple[type, ...], default: Any = None) -> Any:
    v = args.get(name, default)
    if v is not None and not isinstance(v, kind):
        raise RequestError(f'"{name}" has the wrong type ({type(v).__name__})')
    return v


def _report_payload(report: RunReport, t0: float) -> dict[str, Any]:
    return {
        **report.to_dict(),
        "exit_code": EXIT_CODES.get(report.outcome, 1),
        "elapsed_s": round(time.monotonic() - t0, 3),
    }


WORKER_MODULES = ("afv.take", "afv.export.blender")  # what take/export use beyond the stages


def _import_worker_modules() -> list[str]:
    """Import the plumbing modules and what take/export use; returns the import errors."""
    import importlib

    from afv.core.codehash import PLUMBING

    errors = []
    for name in (*sorted(PLUMBING), *WORKER_MODULES):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as e:
            if not (e.name and (name == e.name or name.startswith(e.name + "."))):
                errors.append(f"{name}: {e}")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    return errors


def _plumbing_files(keys: Any = None) -> dict[str, tuple[int, int]]:
    """(mtime_ns, size) of every afv module this process has loaded (with ``keys``: of those files):
    a change to one needs a restart of the long-lived worker. Stats are reused for a few seconds
    (``afv.core.codehash.cached_stat``)."""
    from afv.core.codehash import cached_stat

    if keys is None:
        _import_worker_modules()
        mods = [m for n, m in list(sys.modules.items()) if n == "afv" or n.startswith("afv.")]
        keys = sorted({f for m in mods if (f := getattr(m, "__file__", None))})
    out = {}
    for f in keys:
        with contextlib.suppress(OSError):
            st = cached_stat(Path(f))
            out[f] = (st.st_mtime_ns, st.st_size)
    return out


def check_new_code() -> str:
    """'' when a fresh worker would start (same command, ``CHECK_ENV`` set), else its error."""
    argv = list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
    env = {**os.environ, CHECK_ENV: "1"}
    env.pop(RESUME_ENV, None)
    try:
        r = subprocess.run(
            [sys.executable, *argv[1:]], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, env=env, timeout=CHECK_TIMEOUT_S, check=False,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"{type(e).__name__}: {e}"
    if r.returncode == 0:
        return ""
    lines = [ln for ln in r.stderr.decode("utf-8", "replace").splitlines() if ln.strip()]
    return " | ".join(lines[-3:]) or f"exit code {r.returncode}"


class _JobLog(logging.Handler):
    """Warnings logged by a request's thread become ``log`` events of that request."""

    def __init__(self, server: Server):
        super().__init__(logging.WARNING)
        self.server = server

    def emit(self, record: logging.LogRecord) -> None:
        job = self.server.job_of_thread(threading.get_ident())
        if job is None or getattr(record, "afv_reported", False):
            return
        with contextlib.suppress(Exception):
            msg = record.getMessage()
            self.server.send({"id": job.id, "event": "log", "kind": "warning", "message": msg})


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class Server:
    def __init__(self, in_fd: int = 0, out_fd: int = 1, *, redirect_stdout: bool = True):
        self.in_fd = in_fd
        if redirect_stdout and out_fd == 1:  # protocol on a private copy; fd 1 (prints) -> stderr
            self.out_fd = os.dup(1)
            os.dup2(2, 1)
        else:
            self.out_fd = out_fd
        self._out_lock = threading.Lock()
        self._cond = threading.Condition()
        self._queues: dict[str, deque[Job]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._jobs: dict[Any, Job] = {}
        self._running: set[Job] = set()
        self._by_thread: dict[int, Job] = {}
        self._seq = 0
        self._closing = False
        self._dead = False  # stdout closed: the client is gone
        self._restarts: list[Job] = []  # requests to run again after a code-change restart
        self._ended: set[Any] = set()  # ids whose request ended (cancel answers "finished")
        self._silenced: set[Any] = set()  # ids already answered at shutdown: drop later replies
        self.started = now_iso()
        self.restarts = int(os.environ.pop(RESTARTS_ENV, "0") or 0)
        self._resume: list[bytes] = []
        self._rest = b""
        if path := os.environ.pop(RESUME_ENV, None):
            with contextlib.suppress(OSError, ValueError):
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                self._resume = [ln.encode("utf-8") for ln in data.get("lines", [])]
                self._rest = base64.b64decode(data.get("rest", ""))
                self._ended = {i for i in data.get("ended", []) if _valid_id(i)}
            with contextlib.suppress(OSError):
                os.unlink(path)
        self._plumbing = _plumbing_files()
        subprocess.Popen = _TrackedPopen  # type: ignore[misc]
        logging.getLogger("afv").addHandler(_JobLog(self))

    # -- output ----------------------------------------------------------------

    def send(self, obj: dict[str, Any]) -> None:
        if obj.get("id") in self._silenced and obj.get("id") is not None:
            return
        data = (json.dumps(obj, default=_json_default, ensure_ascii=False) + "\n").encode("utf-8")
        with self._out_lock:
            if self._dead:
                return
            try:
                view = memoryview(data)
                while view:
                    n = os.write(self.out_fd, view)
                    view = view[n:]
            except OSError:
                self._dead = True
        if self._dead:
            self._begin_shutdown()

    def _error(self, rid: Any, message: str, **extra: Any) -> None:
        self.send({"id": rid, "event": "error", "message": message, **extra})

    def _result(self, rid: Any, payload: dict[str, Any]) -> None:
        self.send({"id": rid, "event": "result", **payload})

    def job_of_thread(self, ident: int) -> Job | None:
        return self._by_thread.get(ident)

    # -- input -----------------------------------------------------------------

    def serve_forever(self) -> int:
        """Read requests until stdin closes or ``shutdown``; returns the exit code."""
        try:
            for line in self._resume:
                self.dispatch(line)
            buf, self._rest = self._rest, b""
            skipping = False  # inside an over-long line: drop it up to its newline
            while not self._closing:
                if self._restarts and self._idle():
                    if not (err := check_new_code()):
                        self._exec(buf)  # does not return
                    self._restart_failed(err)
                if NL in buf:
                    line, buf = buf.split(NL, 1)
                    if skipping:
                        skipping = False
                    elif len(line) > MAX_LINE:
                        self._error(None, f"request line longer than {MAX_LINE} bytes")
                    else:
                        self.dispatch(line)
                    continue
                if len(buf) > MAX_LINE:
                    if not skipping:
                        self._error(None, f"request line longer than {MAX_LINE} bytes")
                    skipping, buf = True, b""
                ready, _, _ = select.select([self.in_fd], [], [], 0.2)
                if not ready:
                    continue
                chunk = os.read(self.in_fd, 1 << 16)
                if not chunk:  # the client closed stdin (or went away)
                    if buf.strip() and not skipping:
                        self.dispatch(buf)
                    self._begin_shutdown()
                    break
                buf += chunk
        except (KeyboardInterrupt, SystemExit):
            self._begin_shutdown()
        self._wait_idle(SHUTDOWN_WAIT_S)
        with self._cond:
            left = list(self._running)
        for j in left:
            self._error(j.id, "the worker shut down while this request was running", cancelled=True)
            self._silenced.add(j.id)
        return 0

    def dispatch(self, line: bytes) -> None:
        """Handle one request line; never raises (a bad line gets an ``error`` reply)."""
        rid = None
        try:
            text = line.decode("utf-8", "replace").strip()
            if not text:
                return
            try:
                req = parse_request(text)
            except ValueError as e:
                self._error(None, f"not a valid JSON request: {e}")
                return
            if not isinstance(req, dict):
                self._error(None, "a request must be a JSON object")
                return
            if not _valid_id(req.get("id")):
                self._error(None, '"id" must be a string, an integer or null')
                return
            rid, cmd, args = req.get("id"), req.get("cmd"), req.get("args", {})
            if args is None:
                args = {}
            if not isinstance(cmd, str) or not isinstance(args, dict):
                self._error(rid, 'a request needs "cmd" (a string) and "args" (an object)')
                return
            if cmd == "ping":
                self._result(rid, self._ping())
            elif cmd == "cancel":
                target = args.get("target_id")
                if not _valid_id(target):
                    self._error(rid, '"target_id" must be a string, an integer or null')
                    return
                self._result(rid, self.cancel(target))
            elif cmd == "shutdown":
                self._begin_shutdown()
                self._result(rid, {"shutdown": True})
            elif cmd == "presets":
                from afv.retarget import presets_payload

                self._result(rid, {"presets": presets_payload()})
            elif cmd in SHOT_CMDS:
                try:
                    where, key = self._shot_key(args)
                except (RequestError, ProjectError, OSError, ValueError) as e:
                    self._error(rid, str(e))
                    return
                job = Job(rid, cmd, args, 0, line, key, where, replayed=bool(req.get("_replay")))
                self._enqueue(job)
            elif cmd in PROJECT_CMDS:
                try:
                    where, key = self._project_key(cmd, args)
                except (RequestError, ProjectError, OSError, ValueError) as e:
                    self._error(rid, str(e))
                    return
                job = Job(rid, cmd, args, 0, line, key, where, replayed=bool(req.get("_replay")))
                self._enqueue(job)
            else:
                cmds = ", ".join(("ping", "presets", *SHOT_CMDS, *PROJECT_CMDS, "cancel", "shutdown"))
                self._error(rid, f"unknown cmd {cmd!r}; cmds: {cmds}")
        except Exception as e:  # never let one line take the worker down
            log.debug("bad request line", exc_info=True)
            with contextlib.suppress(Exception):
                self._error(rid if _valid_id(rid) else None, f"bad request: {type(e).__name__}: {e}")

    def _ping(self) -> dict[str, Any]:
        with self._cond:
            running = [j.id for j in self._running]
            queued = [j.id for q in self._queues.values() for j in q] + [j.id for j in self._restarts]
        return {
            "pong": True, "protocol": PROTOCOL, "pid": os.getpid(), "afv_version": afv.__version__,
            "python": sys.version.split()[0], "started": self.started, "restarts": self.restarts,
            "running": running, "queued": queued,
        }  # fmt: skip

    @staticmethod
    def _shot_key(args: dict[str, Any]) -> tuple[tuple[str, str], str]:
        """((project root, shot id), the shot folder's identity)."""
        project, shot = args.get("project"), args.get("shot")
        if not isinstance(project, str) or not project:
            raise RequestError('"project" (the project folder) is required')
        if not isinstance(shot, str | int) or isinstance(shot, bool) or str(shot) == "":
            raise RequestError('"shot" (the shot id) is required')
        proj = Project(wsl_path(project))
        sid = proj.resolve_shot_id(str(shot))
        return (str(proj.root), sid), folder_identity(proj.shots_dir / sid)

    @staticmethod
    def _project_key(cmd: str, args: dict[str, Any]) -> tuple[tuple[str, str], str]:
        """((project root, ""), "<lane>:<project folder identity>"); ``init``: the folder to create."""
        project = args.get("project")
        if not isinstance(project, str) or not project:
            raise RequestError('"project" (the project folder) is required')
        root = wsl_path(project)
        if cmd == "init":
            root = Path(os.path.abspath(root))
            return (str(root), ""), f"init:{str(root).lower()}"
        proj = Project(root)
        key = f"{PROJECT_CMDS[cmd]}:{folder_identity(proj.root)}"
        if cmd == "batch":  # a stream of a detached batch: never waits behind another one
            key += f":{os.urandom(6).hex()}"
        return (str(proj.root), ""), key

    # -- queues ----------------------------------------------------------------

    def _enqueue(self, job: Job) -> None:
        with self._cond:
            self._seq += 1
            job.seq = self._seq
            self._jobs[job.id] = job
            self._ended.discard(job.id)
            self._queues.setdefault(job.key, deque()).append(job)
            if job.key not in self._threads:
                t = threading.Thread(
                    target=self._shot_loop, args=(job.key,), daemon=True, name=job.key[:40]
                )
                self._threads[job.key] = t
                t.start()
            self._cond.notify_all()

    def _shot_loop(self, key: str) -> None:
        q = self._queues[key]
        while True:
            with self._cond:
                while self._closing or not q or self._restarts:
                    if self._closing:
                        return  # queued requests were answered by _begin_shutdown
                    self._cond.wait(1.0)
                job = q.popleft()
                job.state, job.thread = "running", threading.get_ident()
                self._running.add(job)
                self._by_thread[job.thread] = job
            if job.cmd in CANCELLABLE:
                with _CHILDREN_LOCK:
                    _THREAD_CANCEL[job.thread] = job.cancel
            try:
                self._execute(job)
            finally:
                with _CHILDREN_LOCK:
                    _THREAD_CANCEL.pop(job.thread, None)
                with self._cond:
                    if job.state != "restart":
                        job.state = "done"
                        self._ended.add(job.id)
                    self._running.discard(job)
                    self._by_thread.pop(threading.get_ident(), None)
                    self._cond.notify_all()

    def _idle(self) -> bool:
        with self._cond:
            return not self._running

    def _wait_idle(self, timeout: float) -> None:
        end = time.monotonic() + timeout
        with self._cond:
            while self._running and time.monotonic() < end:
                self._cond.wait(0.2)

    # -- cancel / shutdown -----------------------------------------------------

    def cancel(self, target: Any) -> dict[str, Any]:
        with self._cond:
            job = self._jobs.get(target)
            state = "unknown" if job is None else job.state
            if job is not None and state == "queued":
                with contextlib.suppress(ValueError):
                    self._queues[job.key].remove(job)
                job.state = "done"
                self._ended.add(job.id)
            elif job is not None and state == "restart":
                with contextlib.suppress(ValueError):
                    self._restarts.remove(job)
                job.state, state = "done", "queued"
                self._ended.add(job.id)
            elif job is not None and state == "running":
                job.cancel.set()
            if job is None and target in self._ended:
                state = "done"
        if job is not None and state == "queued":
            self._error(job.id, "cancelled before it started", cancelled=True)
        elif job is not None and state == "running" and job.cmd in CANCELLABLE:
            _stop_children(job.thread)
        return {"target_id": target, "state": "finished" if state == "done" else state}

    def _begin_shutdown(self) -> None:
        with self._cond:
            if self._closing:
                return
            self._closing = True
            queued = [j for q in self._queues.values() for j in q] + list(self._restarts)
            for q in self._queues.values():
                q.clear()
            self._restarts.clear()
            running = list(self._running)
            for j in running:
                j.cancel.set()
            self._cond.notify_all()
        for j in queued:
            j.state = "done"
            self._error(j.id, "cancelled: the worker is shutting down", cancelled=True)
        for j in running:
            if j.cmd in CANCELLABLE:
                _stop_children(j.thread)

    # -- restart after a code change -------------------------------------------

    def _request_restart(self, job: Job) -> None:
        with self._cond:
            job.state = "restart"
            self._restarts.append(job)
            self._cond.notify_all()
        self.send({"id": job.id, "event": "log", "kind": "restart", "message": RESTART_MESSAGE})

    def _restart_failed(self, err: str) -> None:
        """The new code does not import: end the restart requests with the error, keep serving."""
        with self._cond:
            jobs, self._restarts = list(self._restarts), []
            for j in jobs:
                j.state = "done"
                self._ended.add(j.id)
            self._cond.notify_all()
        log.warning("afv code has an error, not restarting: %s", err)
        for j in jobs:
            self._error(j.id, f"afv code has an error: {err}; fix it and retry", code_error=True)

    def _exec(self, buf: bytes) -> None:
        """Re-execute this process (same stdin/stdout) with every restart request and every queued one."""
        with self._cond:
            restart = list(self._restarts)
            jobs = [*restart, *(j for q in self._queues.values() for j in q)]
            ended = [i for i in self._ended if _valid_id(i)]
        jobs.sort(key=lambda j: j.seq)
        lines = []
        for j in jobs:
            text = j.line.decode("utf-8", "replace")
            if j in restart:
                req = json.loads(text)
                req["_replay"] = True
                text = json.dumps(req)
            lines.append(text)
        fd, path = tempfile.mkstemp(prefix="afv-serve-resume-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"lines": lines, "rest": base64.b64encode(buf).decode(), "ended": ended[-10000:]}, f)
        os.environ[RESUME_ENV] = path
        os.environ[RESTARTS_ENV] = str(self.restarts + 1)
        with contextlib.suppress(Exception):
            sys.stdout.flush()
            sys.stderr.flush()
        os.dup2(self.out_fd, 1)
        argv = list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
        os.execv(sys.executable, [sys.executable, *argv[1:]])

    # -- requests --------------------------------------------------------------

    def _execute(self, job: Job) -> None:
        from afv.models.envs import cancel_scope

        if not job.replayed and _plumbing_files(self._plumbing) != self._plumbing:
            self._request_restart(job)  # the worker's own code changed: run it with the new code
            return
        handler = getattr(self, f"_cmd_{job.cmd}")
        try:
            with cancel_scope(job.cancel if job.cmd in CANCELLABLE else None):
                payload = handler(job)
        except RequestError as e:
            self._error(job.id, str(e))
        except EXPECTED as e:
            if job.cancel.is_set() and job.cmd in CANCELLABLE:  # e.g. its worker was stopped by the cancel
                self._error(job.id, f"cancelled ({str(e).splitlines()[0]})", cancelled=True)
            else:
                self._error(job.id, str(e))
        except Exception as e:  # never let a request take the worker down
            from afv.take import TakeError

            if isinstance(e, TakeError):
                self._error(job.id, str(e))
            else:
                log.debug("request %s crashed", job.id, exc_info=True)
                tb = traceback.format_exc()
                self._error(job.id, f"internal error: {type(e).__name__}: {e}", traceback=tb)
        else:
            if payload is None:
                return
            if job.cancel.is_set():  # a take/edit/status cancelled while it ran: it completed anyway
                self._error(job.id, f"cancelled, but the {job.cmd} had already completed (its effect stays)",
                            cancelled=True, completed=True, result=payload)  # fmt: skip
                return
            self._result(job.id, payload)

    def _open(self, job: Job) -> Shot:
        return Project(Path(job.where[0])).shot(job.where[1])

    def _reporter(self, job: Job) -> Any:
        from afv.cli import _Throttle

        throttle = _Throttle()

        def report(ev: dict[str, Any]) -> None:
            kind = ev.get("event")
            if kind == "progress":
                if throttle.allow(ev):
                    self.send({**ev, "id": job.id})
            elif kind == "tracking_review":
                self.send({**ev, "id": job.id})
            else:
                self.send({**{k: v for k, v in ev.items() if k != "event"}, "id": job.id, "event": "log",
                           "kind": kind})  # fmt: skip

        return report

    def _runner(self, job: Job, sh: Shot, sets: list[str], **kw: Any) -> Runner:
        from afv.core.config import parse_set

        a = job.args
        cc = _arg(a, "code_changes", str, "report")
        return Runner(
            sh, params=parse_set(sets), report=self._reporter(job), code_changes=cc, pause_on_hints=True,
            yes=bool(a.get("yes")), cancel=job.cancel, **kw,
        )  # fmt: skip

    def _finish_run(self, job: Job, report: RunReport, t0: float) -> dict[str, Any] | None:
        if report.outcome == "code_changed" and not job.replayed:
            self._request_restart(job)
            return None  # replied by the restarted worker
        if report.outcome == "cancelled":
            self._error(job.id, report.message or "cancelled", cancelled=True, report=report.to_dict())
            return None
        return _report_payload(report, t0)

    def _cmd_run(self, job: Job) -> dict[str, Any] | None:
        from afv.cli import CameraMode, _settings_overrides

        t0 = time.monotonic()
        a = job.args
        sh = self._open(job)
        frames = parse_frames(a.get("frames"))
        from_stage = _arg(a, "from", str)
        to = _arg(a, "to", str) or DEFAULT_TARGET
        if frames is not None and from_stage is None:
            raise RequestError('"frames" needs "from" (the first stage to re-run for those frames)')
        sets = _sets(a.get("sets"))
        camera = _arg(a, "camera", str)
        cam = CameraMode(camera) if camera else None
        runner = self._runner(
            job, sh, sets, impl={"camera": camera} if camera else {}, seed=int(a.get("seed") or 0),
            deterministic=bool(a.get("deterministic")),
        )  # fmt: skip
        with runner.shot_lock():
            runner.check_edits(to)
            _settings_overrides(sh, cam, sets, runner, persist=True)
            report = runner.run(to=to, from_stage=from_stage, frames=frames)
        return self._finish_run(job, report, t0)

    def _cmd_export(self, job: Job) -> dict[str, Any] | None:
        from afv.cli import _settings_overrides

        t0 = time.monotonic()
        a = job.args
        sh = self._open(job)
        sets = _sets(a.get("sets"))
        if (preset := _arg(a, "preset", str)) is not None:
            sets.append(f"export.preset={json.dumps(preset)}")
        if (fps := _arg(a, "fps", int | float)) is not None:
            sets.append(f"export.fps={float(fps)!r}")
        runner = self._runner(job, sh, sets)
        with runner.shot_lock():
            runner.check_edits("export")
            _settings_overrides(sh, None, sets, runner, persist=True)
            report = runner.run(to="export")
        if report.outcome == "ok" and not report.exports:
            report.exports = list((sh.meta("export") or {}).get("exports", []))
        return self._finish_run(job, report, t0)

    def _cmd_status(self, job: Job) -> dict[str, Any]:
        a = job.args
        sh = self._open(job)
        frames = parse_frames(a.get("frames"))
        kw = {"to": _arg(a, "to", str) or DEFAULT_TARGET, "from_stage": _arg(a, "from", str)}
        kw["frames"] = frames
        runner = Runner(sh)
        try:
            with runner.shot_lock():  # free: nobody is committing meanwhile
                return runner.plan(**kw).to_dict()
        except RunError as e:
            if "already being run" not in str(e):
                raise
        report = runner.plan(**kw)  # another process runs the shot: plan without the lock
        if any("missing or damaged" in st.reason for st in report.stages):
            time.sleep(0.5)  # caught a commit or prune half way: ask once more
            report = Runner(sh).plan(**kw)
        return report.to_dict()

    def _cmd_take(self, job: Job) -> dict[str, Any]:
        from afv import take as tk

        a = job.args
        sh = self._open(job)
        out = _arg(a, "out", str)
        kw = {} if a.get("check_plan", True) else {"check_plan": False}
        with Runner(sh).shot_lock():
            path = tk.make_take(sh, wsl_path(out) if out else None, preset=_arg(a, "preset", str), **kw)
        return {"path": str(path), **tk.take_summary(path)}

    def _cmd_edit(self, job: Job) -> dict[str, Any]:
        from afv.core.edits import write_edit

        a = job.args
        name = _arg(a, "name", str)
        if not name:
            raise RequestError('"name" (the edit, e.g. contacts) is required')
        if "data" not in a:
            raise RequestError('"data" is required (null deletes the edit)')
        data = a["data"]
        if data is not None and not isinstance(data, dict):
            raise RequestError('"data" must be a JSON object (or null to delete the edit)')
        sh = self._open(job)
        runner = Runner(sh)
        with runner.shot_lock():
            path = write_edit(sh, name, data)
        stages = []
        for s in STAGE_ORDER:
            with contextlib.suppress(RunError):
                cls, _ = runner.resolve(s)
                if name in runner.edit_names(s, cls):
                    stages.append(s)
        return {"path": str(path), "name": name, "removed": data is None, "stages": stages}


    # -- round 6: shot browser (project commands, shot_set, pick) ---------------

    def _progress(self, job: Job, stage: str) -> Any:
        """``progress(frac, message)`` sending throttled ``progress`` events for ``stage``."""
        from afv.cli import _Throttle

        throttle = _Throttle()

        def progress(frac: float | None, msg: str) -> None:
            ev = {"id": job.id, "event": "progress", "stage": stage,
                  "frac": None if frac is None or frac < 0 else round(float(frac), 4), "message": msg}
            if throttle.allow(ev):
                self.send(ev)

        return progress

    @contextlib.contextmanager
    def _heartbeat(self, job: Job, stage: str, what: str) -> Any:
        """A ``progress`` event (``frac`` null) every second while a step without progress of its own runs."""
        stop = threading.Event()
        t0 = time.monotonic()

        def beat() -> None:
            while not stop.wait(1.0):
                self.send({"id": job.id, "event": "progress", "stage": stage, "frac": None,
                           "message": f"{what} ({time.monotonic() - t0:.0f} s)"})

        th = threading.Thread(target=beat, daemon=True)
        th.start()
        try:
            yield
        finally:
            stop.set()
            th.join()

    def _project_of(self, job: Job) -> Project:
        return Project(Path(job.where[0]))

    def _log_events(self, job: Job, skip: tuple[str, ...] = ("shot_added", "shot_found")) -> Any:
        def emit(obj: dict[str, Any], text: str | None) -> None:
            kind = obj.get("event", "")
            if kind not in skip:
                self.send({**{k: v for k, v in obj.items() if k != "event"}, "id": job.id, "event": "log",
                           "kind": kind})  # fmt: skip

        return emit

    def _cmd_init(self, job: Job) -> dict[str, Any]:
        from afv.cli import _probe, project_media

        a = job.args
        video_s = _arg(a, "video", str)
        if not video_s:
            raise RequestError('"video" (the source video) is required')
        video = wsl_path(video_s)
        detect = _arg(a, "detect", bool, True)
        proxy = a.get("proxy", "auto")
        if proxy not in (True, False, "auto"):
            raise RequestError('"proxy" must be true, false or "auto" (only for a source taller than 720 px)')
        mc = _min_confidence(a)
        if not video.is_file():
            raise RequestError(f"video not found: {video}")
        root = Path(job.where[0])
        if (root / "project.json").exists():
            raise RequestError(f"{root} is already an afv project: open it (shots) instead")
        t0 = time.monotonic()
        say = self._progress(job, "init")
        say(None, "probing the video")
        probe_info, warning = _probe(video)
        say(None, "hashing the video")
        proj = Project.create(root, video, title=_arg(a, "title", str, ""), url=_arg(a, "url", str, ""),
                              probe=probe_info)  # fmt: skip
        if warning:
            self.send({"id": job.id, "event": "log", "kind": "warning", "message": warning})
        media: dict[str, Any] = {"detect": None, "proxy": None}
        if detect or proxy:
            with self._heartbeat(job, "init", "detecting cuts" if detect else "making the proxy"):
                media = project_media(proj, detect=detect, proxy=proxy, min_confidence=mc,
                                      emit=self._log_events(job))  # fmt: skip
        info = proj.info
        return {"created": str(proj.root), "sha256": info.source.sha256, "warning": warning or None,
                "detect": media["detect"], "elapsed_s": round(time.monotonic() - t0, 3),
                **_shots_payload(proj)}  # fmt: skip

    def _cmd_shots(self, job: Job) -> dict[str, Any]:
        a = job.args
        return _shots_payload(self._project_of(job), _states_arg(a), stages=_arg(a, "stages", bool, False))

    def _cmd_shot_add(self, job: Job) -> dict[str, Any]:
        from afv.cli import shot_row

        a = job.args
        proj = self._project_of(job)
        start = _time_arg(a, "start")
        if start is None:
            raise RequestError('"start" (seconds or "m:ss.s") is required')
        end, dur = _time_arg(a, "end"), _time_arg(a, "duration")
        if (end is None) == (dur is None):
            raise RequestError('give exactly one of "end" and "duration"')
        dur = end - start if end is not None else dur
        speed = _num_arg(a, "speed", 1.0)
        state = _arg(a, "state", str, "queued")
        sid = a.get("shot_id")
        if sid is not None and (isinstance(sid, bool) or not isinstance(sid, str | int)):
            raise RequestError('"shot_id" must be a string')
        sh = proj.add_shot(start, float(dur or 0), shot_id=None if sid is None else str(sid),
                           speed_factor=float(speed if speed is not None else 1.0),
                           notes=_arg(a, "notes", str, ""), state=state)  # fmt: skip
        return {"shot": sh.id, "row": shot_row(sh)}

    def _cmd_shot_set(self, job: Job) -> dict[str, Any]:
        from afv.cli import set_shot

        a = job.args
        sh = self._open(job)
        res = set_shot(sh, start=_time_arg(a, "start"), end=_time_arg(a, "end"),
                       duration=_time_arg(a, "duration"), speed=_num_arg(a, "speed"),
                       notes=_arg(a, "notes", str), state=_arg(a, "state", str))  # fmt: skip
        return {"shot": sh.id, "changed": res.pop("changed"), "notes_for_user": res.pop("notes_for_user"),
                "row": {k: v for k, v in res.items() if k != "shot"}}  # fmt: skip

    def _cmd_shots_detect(self, job: Job) -> dict[str, Any]:
        from afv.cli import detect_into_project

        a = job.args
        proj = self._project_of(job)
        with self._heartbeat(job, "detect", "detecting cuts"):
            summary = detect_into_project(proj, _min_confidence(a), dry_run=_arg(a, "dry_run", bool, False),
                                          emit=self._log_events(job, skip=("shot_added",)))  # fmt: skip
        return {"detect": summary, **_shots_payload(proj)}

    def _cmd_shots_scan(self, job: Job) -> dict[str, Any] | None:
        from afv.cli import select_shots, shot_row
        from afv.media.shots import scan_shots

        a = job.args
        proj = self._project_of(job)
        ids = _ids_arg(a)
        shots = select_shots(proj, ids, _states_arg(a), ("candidate", "queued"))
        pinfo = proj.info

        def scanned(sid: str, data: dict[str, Any]) -> None:
            with contextlib.suppress(Exception):
                row = shot_row(proj.shot(sid), project_info=pinfo)
                self.send({"id": job.id, "event": "log", "kind": "scanned", "shot": sid, "row": row})

        t0 = time.monotonic()
        rep = scan_shots(proj, shots, force=_arg(a, "force", bool, False),
                         progress=self._progress(job, "scan"), cancel=job.cancel, on_scan=scanned)
        payload = {**rep.to_dict(), "shots": [sh.id for sh in shots],
                   "elapsed_s": round(time.monotonic() - t0, 3)}  # fmt: skip
        if rep.cancelled or job.cancel.is_set():
            self._error(job.id, f"cancelled; {len(rep.scanned)} shots were scanned and kept (scan again to "
                        "finish)", cancelled=True, report=payload)  # fmt: skip
            return None
        if rep.error:
            self._error(job.id, f"the scan failed ({len(rep.scanned)} shots were scanned and kept): "
                        f"{rep.error}", report=payload)  # fmt: skip
            return None
        return payload

    def _cmd_pick(self, job: Job) -> dict[str, Any] | None:
        from afv.cli import pick_fighters, pick_prompt, shot_row
        from afv.media.shots import scan_shots

        a = job.args
        sh = self._open(job)
        frame = _arg(a, "frame", int)
        if isinstance(frame, bool) or (frame is not None and frame < 0):
            raise RequestError('"frame" must be a shot-timeline frame number >= 0')
        fighters = a.get("fighters")
        if fighters is None:
            if a.get("extra") is not None:
                raise RequestError('"extra" needs "fighters"')
            prompt, source, img = pick_prompt(sh, frame)
            if not source or img is None:  # nothing on this shot yet: scan it (one frame, one GPU job)
                rep = scan_shots(sh.project, [sh], progress=self._progress(job, "scan"), cancel=job.cancel)
                if rep.cancelled or job.cancel.is_set():
                    self._error(job.id, f"cancelled while finding the people on shot {sh.id}", cancelled=True)
                    return None
                if rep.error or rep.failed:
                    raise RequestError(f"cannot find the people on shot {sh.id}: "
                                       f"{rep.error or rep.failed.get(sh.id)}")  # fmt: skip
                prompt, source, img = pick_prompt(sh, frame)
            pf = prompt.get("prompt_frame", prompt.get("frame"))
            return {"shot": sh.id, "source": source, "frame": pf, "prompt": prompt,
                    "image": str(img) if img else None, "image_win": _win(img) if img else None,
                    "current": sh.read_edit("fighters")}  # fmt: skip
        fl, el = _int_list_arg(a, "fighters"), _int_list_arg(a, "extra") or []
        with Runner(sh).shot_lock():
            res = pick_fighters(sh, fl, el, frame=frame, keep_prompts=_arg(a, "keep_prompts", bool, False))
        return {"shot": sh.id, "source": res["source"], "frame": res["frame"], "fighters": fl, "extra": el,
                "edit": str(res["edit"]), "dropped": res["dropped"], "data": res["data"], "row": shot_row(sh)}

    def _cmd_batch(self, job: Job) -> dict[str, Any] | None:
        from afv.cli import CameraMode, batch_dir, batch_running, batch_status, new_batch_id, parse_shot_ids

        t0 = time.monotonic()
        a = job.args
        proj = self._project_of(job)
        to = _arg(a, "to", str) or DEFAULT_TARGET
        if to not in STAGE_ORDER:
            raise RequestError(f'"to": unknown stage {to!r}; stages: {", ".join(STAGE_ORDER)}')
        sets = _sets(a.get("sets"))
        camera = _arg(a, "camera", str)
        if camera:
            CameraMode(camera)  # ValueError: a readable error
        cc = _arg(a, "code_changes", str, "report")
        if cc not in ("report", "rerun", "accept"):
            raise RequestError('"code_changes" must be report, rerun or accept')
        states = _states_arg(a)
        ids = parse_shot_ids(proj, _ids_arg(a)) if _ids_arg(a) else None
        attached = batch_running(proj)
        if attached and batch_status(proj)["done"] is not None:  # it has finished and is exiting
            end = time.monotonic() + 30
            while batch_running(proj) and time.monotonic() < end:
                time.sleep(0.1)
            attached = batch_running(proj)
        if attached:
            bid = str(batch_status(proj)["batch_id"])
        else:
            bid = new_batch_id()
            args = ["batch", "--json", "--batch-id", bid, "-p", str(proj.root), "--to", to,
                    "--code-changes", cc]
            for s in sets:
                args += ["--set", s]
            for flag, name in (("--yes", "yes"), ("--stop-on-error", "stop_on_error"), ("--all", "all")):
                if _arg(a, name, bool, False):
                    args.append(flag)
            for st in states or []:
                args += ["--state", st]
            for sid in ids or []:
                args += ["--shot", sid]
            if camera:
                args += ["--camera", camera]
            self._start_detached(proj, bid, args)
        log = batch_dir(proj) / f"{bid}.jsonl"
        self.send({"id": job.id, "event": "log", "kind": "batch_started", "batch_id": bid,
                   "attached": attached, "log": str(log.relative_to(proj.root))})  # fmt: skip
        return self._stream_batch(job, proj, bid, attached, t0)

    def _start_detached(self, proj: Project, bid: str, args: list[str]) -> None:
        """Start ``afv batch`` in its own session, not tracked as a child of a request: it survives the
        client, this worker and a cancel of the streaming request (``batch_cancel`` stops it)."""
        from afv.cli import batch_dir

        d = batch_dir(proj)
        d.mkdir(parents=True, exist_ok=True)
        err_path = d / f"{bid}.stderr"
        with open(err_path, "wb") as err:
            proc = _ORIGINAL_POPEN(
                [*_self_command(), *args], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err,
                start_new_session=True, close_fds=True,
            )  # fmt: skip
        threading.Thread(target=proc.wait, daemon=True, name=f"reap batch {bid}").start()
        log = d / f"{bid}.jsonl"
        end = time.monotonic() + 120
        while not log.exists():  # it takes the project's batch lock, then opens its log
            if proc.poll() is not None and not log.exists():
                tail = ""
                with contextlib.suppress(OSError):
                    lines = [ln for ln in err_path.read_text("utf-8", "replace").splitlines() if ln.strip()]
                    tail = " | ".join(lines[-3:])
                raise RequestError(f"the batch did not start: {tail or f'exit code {proc.returncode}'}")
            if time.monotonic() > end:
                raise RequestError(f"the batch did not start within 2 minutes (see {err_path})")
            time.sleep(0.05)

    def _stream_batch(
        self, job: Job, proj: Project, bid: str, attached: bool, t0: float
    ) -> dict[str, Any] | None:
        """Send the batch log's events while the batch runs; the result when it ends."""
        from afv.cli import _Throttle, batch_cancel, batch_dir, batch_running, batch_status

        path = batch_dir(proj) / f"{bid}.jsonl"
        throttle = _Throttle()
        pos, rest = 0, b""
        done: dict[str, Any] | None = None
        cancel_sent = False
        while done is None:
            try:
                with open(path, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
            except OSError:
                chunk = b""
            pos += len(chunk)
            rest += chunk
            *lines, rest = rest.split(b"\n")
            for line in lines:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                kind = ev.get("event")
                if kind == "batch_done":
                    done = ev
                elif kind == "progress":
                    if throttle.allow(ev):
                        self.send({**ev, "id": job.id})
                elif kind == "tracking_review":
                    self.send({**ev, "id": job.id})
                else:
                    self.send({**{k: v for k, v in ev.items() if k != "event"}, "id": job.id, "event": "log",
                               "kind": kind})  # fmt: skip
            if done is not None:
                break
            if job.cancel.is_set() and self._closing:  # the client went away: the batch goes on
                self._error(job.id, "the worker is shutting down; the batch goes on (see batch_status)",
                            cancelled=True, detached=True, batch_id=bid)  # fmt: skip
                return None
            if self._restarts:  # a code-change restart waits for running requests: let it (batch goes on)
                self._error(job.id, "afv code changed: the worker restarts; the batch goes on (send batch "
                            "again to follow it)", cancelled=True, detached=True, batch_id=bid)  # fmt: skip
                return None
            if job.cancel.is_set():
                if not cancel_sent:  # cancel of this request = batch_cancel
                    cancel_sent = True
                    threading.Thread(target=batch_cancel, args=(proj, 0.0), daemon=True).start()
            if not chunk and not batch_running(proj):
                time.sleep(0.3)  # its last lines may still be on their way
                with contextlib.suppress(OSError):
                    if path.stat().st_size > pos:
                        continue
                st = batch_status(proj)
                self._error(job.id, f"batch {bid} stopped without finishing (killed?); finished shots are "
                            "kept", report=st)  # fmt: skip
                return None
            time.sleep(0.25)
        assert done is not None
        kinds = ("ok", "needs_pick", "needs_review", "code_changed", "failed")
        counts = {k: int(done.get(k) or 0) for k in kinds}
        code = 1 if counts["failed"] else (3 if counts["needs_pick"] or counts["needs_review"]
                                           or counts["code_changed"] else 0)  # fmt: skip
        payload = {"batch_id": bid, "attached": attached, "outcomes": done.get("outcomes") or [], **counts,
                   "shots": [o["shot"] for o in done.get("outcomes") or []], "exit_code": code,
                   "cancelled": bool(done.get("cancelled")), "log": str(path.relative_to(proj.root)),
                   "elapsed_s": round(time.monotonic() - t0, 3)}  # fmt: skip
        if payload["cancelled"]:
            self._error(job.id, f"cancelled after {len(payload['outcomes'])} shots; they are kept",
                        cancelled=True, report=payload)  # fmt: skip
            return None
        return payload

    def _cmd_batch_status(self, job: Job) -> dict[str, Any]:
        from afv.cli import batch_status

        return batch_status(self._project_of(job))

    def _cmd_batch_cancel(self, job: Job) -> dict[str, Any]:
        from afv.cli import batch_cancel

        wait = _num_arg(job.args, "wait_s", 60.0)
        return batch_cancel(self._project_of(job), float(wait if wait is not None else 60.0))


def _self_command() -> list[str]:
    """How this worker was started, without the trailing ``serve``: the command that runs ``afv <args>``
    with the same Python and the same stage modules (a test worker passes its own script)."""
    argv = list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
    if argv and argv[-1] == "serve":
        argv = argv[:-1]
    return [sys.executable, *argv[1:]]


# ---------------------------------------------------------------------------
# Argument helpers and the shot list (round 6)
# ---------------------------------------------------------------------------


def _time_arg(args: dict[str, Any], name: str) -> float | None:
    """Seconds as a number, or a time string ("62.5", "1:02.5")."""
    from afv.cli import parse_time

    v = args.get(name)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int | float | str):
        raise RequestError(f'"{name}" must be seconds (a number) or a time string like "1:02.5"')
    t = parse_time(v) if isinstance(v, str) else float(v)
    if not math.isfinite(t) or t < 0:
        raise RequestError(f'"{name}" must be a time >= 0, got {v!r}')
    return t


def _num_arg(args: dict[str, Any], name: str, default: float | None = None) -> float | None:
    v = args.get(name, default)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise RequestError(f'"{name}" must be a number')
    return float(v)


def _min_confidence(args: dict[str, Any]) -> float:
    mc = _num_arg(args, "min_confidence", 0.2)
    if mc is None or not 0 <= mc <= 1:
        raise RequestError('"min_confidence" must be a number from 0 to 1')
    return mc


def _states_arg(args: dict[str, Any]) -> list[str] | None:
    from afv.core.project import check_state

    v = args.get("state")
    if v is None:
        return None
    vals = [v] if isinstance(v, str) else v
    if not isinstance(vals, list) or not all(isinstance(x, str) for x in vals):
        raise RequestError('"state" must be a shot state or a list of them (candidate, queued, skipped)')
    return [check_state(x) for x in vals]


def _ids_arg(args: dict[str, Any]) -> list[str] | None:
    v = args.get("shots")
    if v is None:
        return None
    if not isinstance(v, list) or not all(isinstance(x, str | int) and not isinstance(x, bool) for x in v):
        raise RequestError('"shots" must be a list of shot ids')
    return [str(x) for x in v]


def _int_list_arg(args: dict[str, Any], name: str) -> list[int] | None:
    v = args.get(name)
    if v is None:
        return None
    if not isinstance(v, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        raise RequestError(f'"{name}" must be a list of detection numbers, e.g. [1, 0]')
    return list(v)


_WIN: dict[str, str | None] = {}


def _win(p: Path | str) -> str | None:
    """Windows form of a path (cached; None when it cannot be converted)."""
    key = str(p)
    if key not in _WIN:
        from afv.take import windows_path

        try:
            _WIN[key] = windows_path(Path(p))
        except Exception:
            _WIN[key] = None
    return _WIN[key]


def _shots_payload(proj: Project, states: list[str] | None = None, *, stages: bool = False) -> dict[str, Any]:
    """The serve ``shots`` result (also part of ``init`` and ``shots_detect``)."""
    from afv.cli import shot_rows
    from afv.core.project import SCAN_SUBDIR, SHOT_STATES
    from afv.media.shots import CUTS_FILE, source_proxy

    info = proj.info
    probe = info.source.probe or {}
    every = shot_rows(proj, None, stages=stages)
    rows = [r for r in every if not states or r["state"] in states]
    proxy = source_proxy(proj.media_dir, info.source.sha256)
    cuts = proj.media_dir / CUTS_FILE
    return {
        "project": str(proj.root), "root_win": _win(proj.root),
        "source": {
            "path": info.source.path, "path_win": _win(info.source.path), "sha256": info.source.sha256,
            "title": info.source.title, "fps": probe.get("avg_fps"), "width": probe.get("display_width"),
            "height": probe.get("display_height"), "duration": probe.get("duration"),
            "num_frames": probe.get("num_frames"),
        },
        "proxy": str(proxy.relative_to(proj.root)) if proxy else None,
        "cuts": f"media/{CUTS_FILE}" if cuts.is_file() else None,
        "scan_dir": f"media/{SCAN_SUBDIR}",
        "counts": {s: sum(1 for r in every if r["state"] == s) for s in SHOT_STATES},
        "shots": rows,
    }  # fmt: skip


def serve() -> int:
    """Entry point of ``afv serve`` (with ``CHECK_ENV`` set: import everything, exit 0 or 1)."""
    status = load_stage_modules()
    if os.environ.get(CHECK_ENV):
        errors = [f"{m}: {st}" for m, st in status.items() if st.startswith("import failed")]
        errors += _import_worker_modules()
        for e in errors:
            print(e, file=sys.stderr)
        return 1 if errors else 0
    from afv.core import stage as stage_mod
    from afv.core.codehash import set_stat_ttl, snapshot_code

    for name in STAGE_ORDER:  # record the code loaded now (see afv.core.codehash.snapshot_code)
        for impl in stage_mod.implementations(name):
            with contextlib.suppress(Exception):
                snapshot_code(stage_mod.get_stage(name, impl))
    for m, st in status.items():
        if st != "ok":
            log.warning("stage module %s: %s", m, st)
    from afv.models.envs import death_signal_note

    if note := death_signal_note():
        log.warning("%s", note)
    set_stat_ttl(STAT_TTL_S)
    return Server().serve_forever()
