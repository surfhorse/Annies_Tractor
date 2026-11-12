"""Background requests of the shot browser.

Why not ``ops.submit``: the take tools run one job at a time (``ops.busy()`` refuses a second one, and
``ops.STATE`` holds one job and one set of callbacks), but the browser must keep several requests in
flight on the session's one ``afv serve`` worker: a scan or an overnight batch streams for minutes or
hours while shots are listed, trimmed, queued and picked (the worker runs those on separate lanes), and
the take tools must stay usable meanwhile. So this module keeps its own list of jobs on the same worker
client (``ops.serve_client``) and the same one-shot runner (``wsl.Job``). Each ``Job`` has a lane
(``scan``, ``batch``, ``bstatus``, ``pick``, ``init``, ``list``, ``edit``, ``take``); ``EXCLUSIVE`` lanes
take one job at a time. A persistent timer drains every job's events on the main thread and calls its
callbacks, so Blender never waits.

Opening another file or disabling the add-on never cancels anything in afv (``forget``): the batch and
its status are dropped here and go on in WSL (Open project attaches again); a scan or a new project keeps
running and reports its end in the status line; edits have already been sent.

When the worker is off in the preferences or cannot start, a request runs as the equivalent one-shot
``afv ... --json`` command (``browser_core.oneshot_argv``; ``take`` through ``serve.cli_args``, as the
take tools), and its events are turned into the serve result (``browser_core.oneshot_result``).
``batch_status`` and ``batch_cancel`` have no one-shot form: callers skip them without the worker.
"""

from __future__ import annotations

import time
import traceback

import bpy

from . import browser_core as core
from . import ops, serve, wsl

POLL_S = 0.05
EXCLUSIVE = ("scan", "batch", "pick", "init", "list", "bstatus", "take")
KEEP_ON_LOAD = ("scan", "init")  # they finish in afv and say so; everything else is just dropped here


class Job:
    """One browser request: a ``serve.Request`` or a one-shot ``wsl.Job`` plus callbacks."""

    def __init__(self, lane, cmd, args, label, on_done=None, on_fail=None, on_event=None, root_win=None):
        self.lane, self.cmd, self.args, self.label = lane, cmd, args, label
        self.on_done, self.on_fail, self.on_event = on_done, on_fail, on_event
        self.root_win = root_win  # the project folder as Blender sees it (one-shot results read files)
        self.inner = None  # serve.Request | wsl.Job
        self.oneshot = False
        self.started = time.monotonic()
        self.frac: float | None = None
        self.message = ""
        self.cancelled = False
        self.orphan = False  # its file was closed: it only reports its end in the status line

    # --- what the panel shows ---------------------------------------------------------------------
    def line(self) -> str:
        return f"{self.label}: {self.message}" if self.message else f"{self.label}..."

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


JOBS: list[Job] = []
STATUS = {"text": "", "level": "info"}  # the browser's last message (shown in its panel)


def set_status(text, level="info"):
    STATUS["text"], STATUS["level"] = text, level
    redraw()


def redraw():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type in ("VIEW_3D", "DOPESHEET_EDITOR", "SEQUENCE_EDITOR"):
                area.tag_redraw()


def busy(lane=None) -> bool:
    return any(lane is None or j.lane == lane for j in JOBS)


def job(lane) -> Job | None:
    return next((j for j in JOBS if j.lane == lane), None)


def _start_inner(j: Job) -> None:
    if ops.use_serve():
        try:
            j.inner = ops.serve_client().submit(j.cmd, j.args, j.label)
            return
        except serve.ServeError as e:
            ops.STATE.serve_off = f"afv serve not available ({e}); using one-shot afv commands"
    _start_oneshot(j)


def oneshot_args(cmd: str, args: dict) -> list[str]:
    """The one-shot ``afv`` arguments for a request (RuntimeError: it needs the worker)."""
    try:
        return core.oneshot_argv(cmd, args)
    except ValueError:
        pass
    try:
        return serve.cli_args(cmd, args)  # take (and run / export), as the take tools do
    except (ValueError, KeyError):
        raise RuntimeError(f"{cmd} needs the afv serve worker (on in the add-on's preferences)") from None


def _start_oneshot(j: Job) -> None:
    distro, afv = ops._wsl_settings()
    argv = wsl.afv_argv([*oneshot_args(j.cmd, j.args), "--json"], distro=distro, afv=afv)
    j.inner = wsl.Job(argv, j.label).start()
    j.oneshot = True


def submit(lane, cmd, args, label, on_done=None, on_fail=None, on_event=None, root_win=None) -> Job:
    """Start a request on ``lane``. ``on_done(job, result)``, ``on_fail(job, message)`` and
    ``on_event(job, event)`` run on the main thread. Raises RuntimeError when an exclusive lane is busy."""
    if lane in EXCLUSIVE and busy(lane):
        raise RuntimeError(f"{job(lane).label} is still running")
    j = Job(lane, cmd, args, label, on_done, on_fail, on_event, root_win)
    _start_inner(j)
    JOBS.append(j)
    if not bpy.app.timers.is_registered(poll):
        bpy.app.timers.register(poll, first_interval=POLL_S, persistent=True)
    redraw()
    return j


def cancel(lane) -> bool:
    j = job(lane)
    if j is None:
        return False
    j.cancelled = True
    j.message = "cancelling..."
    if j.inner is not None:
        j.inner.cancel()
    redraw()
    return True


def forget(keep=KEEP_ON_LOAD) -> None:
    """Drop the jobs here without cancelling anything in afv (another file was opened, or the add-on is
    being disabled): the batch goes on in WSL; ``keep`` lanes (scan, new project) keep running and only
    report their end in the status line."""
    for j in list(JOBS):
        if j.lane in keep:
            j.orphan = True
            j.on_event = None
            j.on_done = _orphan_done
            j.on_fail = _orphan_fail
        else:
            JOBS.remove(j)


def _orphan_done(j: Job, res) -> None:
    what = {"scan": "the scan", "init": "the new project"}.get(j.lane, j.label)
    where = j.root_win or ""
    set_status(f"{what} finished ({where}): Open project to see it" if where else f"{what} finished")


def _orphan_fail(j: Job, message: str) -> None:
    set_status(f"{j.label} (from the file you left) ended: {message}", "warning")


def _progress(j: Job, ev: dict) -> None:
    if ev.get("event") == "progress":
        f = ev.get("frac")
        j.frac = float(f) if isinstance(f, int | float) and f >= 0 else j.frac
        j.message = wsl.status_line(ev)
    elif ev.get("event") == "error" and not ev.get("id"):
        j.message = str(ev.get("message", ""))


def _finish(j: Job) -> None:
    inner = j.inner
    if getattr(inner, "unavailable", False) and not j.cancelled:  # no worker: the same as a one-shot
        ops.STATE.serve_off = f"afv serve not available ({inner.error}); using one-shot afv commands"
        print(f"AFV: {ops.STATE.serve_off}")
        _start_oneshot(j)
        return
    JOBS.remove(j)
    try:
        if j.oneshot:
            res = None if j.cancelled else core.oneshot_result(j.cmd, j.args, inner.events, j.root_win)
            if res is not None:
                if j.on_done is not None:
                    j.on_done(j, res)
            else:
                _fail(j, inner.failure() if not j.cancelled else "cancelled")
        elif inner.ok:
            if j.on_done is not None:
                j.on_done(j, inner.result())
        else:
            _fail(j, inner.failure())
    except Exception as e:  # a callback bug must not stop the timer
        traceback.print_exc()
        set_status(f"{j.label}: {type(e).__name__}: {e}", "error")


def _fail(j: Job, message: str) -> None:
    if j.on_fail is not None:
        j.on_fail(j, message)
    else:
        set_status(f"{j.label} failed: {message}", "error")


def terminal_event(j: Job) -> dict | None:
    """The request's error event (serve), for details such as a cancelled scan's report."""
    t = getattr(j.inner, "terminal", None)
    return t if isinstance(t, dict) else None


def poll():
    for j in list(JOBS):
        try:
            for ev in j.inner.drain():
                _progress(j, ev)
                if j.on_event is not None:
                    j.on_event(j, ev)
        except Exception as e:
            traceback.print_exc()
            j.message = f"{type(e).__name__}: {e}"
        if j.inner.done:
            _finish(j)
    redraw()
    return POLL_S if JOBS else None


def pump(until=None, timeout=60.0) -> bool:
    """Run ``poll`` until ``until()`` is true (default: no jobs left) or ``timeout`` s (tests and
    background mode, where timers do not run). Returns whether the condition was met."""
    end = time.monotonic() + timeout
    cond = until or (lambda: not JOBS)
    while time.monotonic() < end:
        poll()
        if cond():
            return True
        time.sleep(0.01)
    return cond()
