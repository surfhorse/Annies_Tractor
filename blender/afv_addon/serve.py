"""Client for one ``afv serve`` worker (JSON lines over stdin/stdout), no bpy here.

Protocol (docs/BUILD_PLAN.md, "Shared formats"): requests ``{"id": n, "cmd": ..., "args": {...}}``
(``ping``, ``status``, ``run``, ``take``, ``export``, ``edit``, ``cancel``, ``shutdown``); replies
``{"id": n, "event": "progress"|"log"|"tracking_review"|"result"|"error", ...}``, exactly one
``result`` or ``error`` per request. The worker runs requests one at a time per shot, in order.

``ServeClient`` starts the worker on first use (``wsl.exe ... bash -lc 'echo AFVPID:$$; exec <afv>
serve'``, no console window), pings it, and restarts it when it died (at most ``MAX_RESTARTS`` in
``RESTART_WINDOW_S``). Two daemon threads read stdout (replies, routed to their request by id) and
stderr (kept as a short tail for error messages). ``Request`` has the same polling interface as
``wsl.Job`` (``drain``, ``done``, ``ok``, ``result``, ``review``, ``failure``, ``cancel``), so the
add-on's timer handles both alike.

A worker that exits before it answered its first ping is "unavailable" (e.g. an afv without
``serve``): its requests fail with ``unavailable`` set, and the add-on falls back to one-shot
``afv`` commands (``cli_args``).
"""

from __future__ import annotations

import collections
import json
import os
import queue
import re
import subprocess
import threading
import time
from collections.abc import Callable

try:  # the add-on imports this as part of its package; tests load the file on its own
    from .wsl import _NO_WINDOW, PID_PREFIX, decode_line
except ImportError:  # pragma: no cover - exercised by the plain-Python tests
    import importlib.util as _ilu
    import sys as _sys
    from pathlib import Path as _Path

    _spec = _ilu.spec_from_file_location("_afv_serve_wsl", _Path(__file__).with_name("wsl.py"))
    _wsl = _ilu.module_from_spec(_spec)
    _sys.modules[_spec.name] = _wsl
    _spec.loader.exec_module(_wsl)
    _NO_WINDOW, PID_PREFIX, decode_line = _wsl._NO_WINDOW, _wsl.PID_PREFIX, _wsl.decode_line

TERMINAL = ("result", "error")
MAX_RESTARTS = 3
RESTART_WINDOW_S = 60.0
PING_TIMEOUT_S = 60.0  # a cold WSL start plus importing afv can take a while
_PID_LINE = re.compile(r"^AFVPID:(\d+)$")


def serve_argv(distro: str, afv: str) -> list[str]:
    """The wsl.exe command line that starts ``afv serve`` (``afv`` may start with ``~``)."""
    if "$" in afv:
        raise ValueError("'$' is not allowed in the afv command")
    return ["wsl.exe", "-d", distro, "-e", "bash", "-lc", PID_PREFIX + afv + " serve"]


class ServeError(RuntimeError):
    """The worker cannot be (re)started; the message is fit to show the user."""


class Request:
    """One request to the worker; poll ``drain()`` from the main thread."""

    def __init__(self, client: ServeClient | None, rid: int, cmd: str, args: dict, label: str):
        self.client = client
        self.id = rid
        self.cmd = cmd
        self.args = args
        self.label = label
        self.events: list[dict] = []
        self.text: list[str] = []
        self.queue: queue.Queue[dict] = queue.Queue()
        self.terminal: dict | None = None
        self.finished = False  # the terminal event arrived (or the worker died)
        self.done = False  # ... and was drained
        self.cancelled = False
        self.unavailable = False  # the worker never answered: fall back to one-shot afv
        self.error: str | None = None
        self.started = time.monotonic()
        self.ended: float | None = None
        self._event = threading.Event()

    # called by the reader thread
    def _put(self, ev: dict) -> None:
        if self.finished:
            return
        self.queue.put(ev)
        if ev.get("event") in TERMINAL:
            self.terminal = ev
            self._finish()

    def _fail(self, message: str, unavailable: bool = False) -> None:
        if self.finished:
            return
        self.error = message
        self.unavailable = unavailable
        self.queue.put({"event": "error", "message": message, "id": self.id})
        self._finish()

    def _finish(self) -> None:
        self.ended = time.monotonic()
        self.finished = True
        self._event.set()

    # the wsl.Job interface
    def join(self, timeout: float | None = None) -> None:
        self._event.wait(timeout)

    def drain(self) -> list[dict]:
        new = []
        while True:
            try:
                ev = self.queue.get_nowait()
            except queue.Empty:
                break
            self.events.append(ev)
            if ev.get("event") == "log" and ev.get("message"):
                self.text.append(str(ev["message"]))
            new.append(ev)
        if self.finished and self.queue.empty():
            self.done = True
        return new

    def cancel(self) -> None:
        self.cancelled = True
        if self.finished or self.client is None:
            return
        try:
            self.client.send("cancel", {"target_id": self.id}, "cancel", track=False)
        except ServeError:
            pass

    @property
    def returncode(self) -> int | None:
        return None

    @property
    def elapsed(self) -> float:
        return (self.ended or time.monotonic()) - self.started

    @property
    def ok(self) -> bool:
        t = self.terminal
        return (self.done and not self.cancelled and self.error is None and t is not None
                and t.get("event") == "result" and t.get("outcome") in (None, "ok"))

    def result(self) -> dict | None:
        t = self.terminal
        return t if t is not None and t.get("event") == "result" else None

    def review(self) -> dict | None:
        res = self.result() or {}
        if res.get("outcome") != "needs_review":
            return None
        ev = next((e for e in reversed(self.events) if e.get("event") == "tracking_review"), {})
        return {"message": ev.get("message") or res.get("message") or "tracking QC hints to review",
                "hints": ev.get("hints") or (res.get("review") or {}).get("hints") or [],
                "review_video": ev.get("review_video")}

    def failure(self) -> str:
        if self.cancelled:
            return "cancelled"
        if self.error:
            return self.error
        t = self.terminal or {}
        if t.get("event") == "error":
            return str(t.get("message") or "afv serve reported an error")
        if t.get("event") == "result" and t.get("outcome") not in (None, "ok"):
            return str(t.get("message") or t.get("outcome"))
        return "no reply from afv serve"


class ServeClient:
    """One worker process; ``submit`` starts (or restarts) it as needed. Thread-safe."""

    def __init__(self, argv: Callable[[], list[str]], env: dict | None = None,
                 ping_timeout: float = PING_TIMEOUT_S):
        self._argv = argv
        self._env = env
        self.ping_timeout = ping_timeout
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._pending: dict[int, Request] = {}
        self._next_id = 1
        self._starts: list[float] = []
        self._confirmed = False  # the current process answered its ping
        self._ping: Request | None = None
        self.stderr_tail: collections.deque[str] = collections.deque(maxlen=40)
        self.linux_pid: int | None = None
        self.last_exit: int | None = None
        self.restarts = 0

    # --- process ----------------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def confirmed(self) -> bool:
        return self.running and self._confirmed

    def status(self) -> str:
        if self.running:
            return "running" + ("" if self._confirmed else " (starting)") + (
                f", pid {self.linux_pid}" if self.linux_pid else "")
        if self._proc is None and not self._starts:
            return "not started"
        return f"stopped (exit {self.last_exit})"

    def _start(self) -> None:
        now = time.monotonic()
        self._starts = [t for t in self._starts if now - t < RESTART_WINDOW_S]
        if len(self._starts) > MAX_RESTARTS:
            raise ServeError(f"afv serve stopped {len(self._starts)} times within {RESTART_WINDOW_S:.0f} s: "
                             + self._tail())
        if self._starts:
            self.restarts += 1
        self._starts.append(now)
        env = {**(self._env if self._env is not None else os.environ), "WSL_UTF8": "1"}
        argv = self._argv()
        try:
            pipe = subprocess.PIPE
            proc = subprocess.Popen(argv, stdin=pipe, stdout=pipe, stderr=pipe, creationflags=_NO_WINDOW,
                                    env=env, bufsize=0)
        except OSError as e:
            raise ServeError(f"cannot start {argv[0]}: {e}") from e
        self._proc = proc
        self._confirmed = False
        self.linux_pid = None
        self.stderr_tail.clear()
        threading.Thread(target=self._read_out, args=(proc,), name="afv-serve-out", daemon=True).start()
        threading.Thread(target=self._read_err, args=(proc,), name="afv-serve-err", daemon=True).start()
        self._ping = self._send_locked("ping", {}, "ping", track=True)

    def ensure(self) -> None:
        with self._lock:
            if not self.running:
                self._start()

    def _tail(self) -> str:
        lines = [ln for ln in self.stderr_tail if ln.strip()]
        return " | ".join(lines[-3:]) if lines else "no output"

    def _read_out(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = decode_line(raw)
                if not line.strip():
                    continue
                m = _PID_LINE.match(line.strip())
                if m:
                    self.linux_pid = int(m.group(1))
                    continue
                try:
                    ev = json.loads(line)
                    if not isinstance(ev, dict):
                        raise ValueError
                except ValueError:
                    self.stderr_tail.append(line)  # stray print from a library
                    continue
                self._route(proc, ev)
        except (OSError, ValueError):
            pass
        code = proc.wait()
        self._died(proc, code)

    def _read_err(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        try:
            for raw in proc.stderr:
                line = decode_line(raw)
                if line.strip():
                    self.stderr_tail.append(line)
        except (OSError, ValueError):
            pass

    def _route(self, proc: subprocess.Popen, ev: dict) -> None:
        with self._lock:
            if proc is not self._proc:
                return
            rid = ev.get("id")
            req = self._pending.get(rid) if isinstance(rid, int) else None
            if req is None:
                return  # e.g. the reply to a cancelled request nobody waits for
            if req is self._ping and ev.get("event") in TERMINAL:
                self._confirmed = True  # any reply: the worker speaks the protocol
            if ev.get("event") in TERMINAL:
                self._pending.pop(rid, None)
        req._put(ev)

    def _died(self, proc: subprocess.Popen, code: int) -> None:
        time.sleep(0.05)  # let the stderr reader catch the last lines
        with self._lock:
            if proc is not self._proc:
                return
            self.last_exit = code
            unavailable = not self._confirmed
            pending = list(self._pending.values())
            self._pending.clear()
            self._proc = None
        why = (f"afv serve could not start (exit {code}): {self._tail()}" if unavailable
               else f"afv serve stopped (exit {code}): {self._tail()}")
        for req in pending:
            req._fail(why, unavailable=unavailable)

    # --- requests ---------------------------------------------------------------------------------
    def _send_locked(self, cmd: str, args: dict, label: str, track: bool) -> Request:
        rid = self._next_id
        self._next_id += 1
        req = Request(self, rid, cmd, args, label)
        if track:
            self._pending[rid] = req
        line = (json.dumps({"id": rid, "cmd": cmd, "args": args}) + "\n").encode("utf-8")
        proc = self._proc
        try:
            assert proc is not None and proc.stdin is not None
            proc.stdin.write(line)
            proc.stdin.flush()
        except (OSError, ValueError, AssertionError):
            self._pending.pop(rid, None)
            req._fail(f"afv serve is not accepting requests (exit {proc.poll() if proc else None}): "
                      f"{self._tail()}", unavailable=not self._confirmed)
        return req

    def send(self, cmd: str, args: dict, label: str, track: bool = True) -> Request:
        """Send one request to the running worker (no start)."""
        with self._lock:
            if not self.running:
                raise ServeError("afv serve is not running")
            return self._send_locked(cmd, args, label, track)

    def submit(self, cmd: str, args: dict, label: str) -> Request:
        """Send a request, starting or restarting the worker first if needed."""
        with self._lock:
            if not self.running:
                self._start()
            return self._send_locked(cmd, args, label, True)

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until the worker answered its ping (tests; the add-on never blocks on this)."""
        ping = self._ping
        if ping is None:
            return False
        ping.join(self.ping_timeout if timeout is None else timeout)
        return self._confirmed

    def shutdown(self, wait: float = 0.0) -> None:
        """Ask the worker to stop and close its stdin (it exits when stdin closes); kill it after
        ``wait`` seconds (in the background when ``wait`` is 0)."""
        with self._lock:
            proc = self._proc
            pending = list(self._pending.values())
            self._pending.clear()
            self._proc = None
            if proc is None:
                return
            try:
                assert proc.stdin is not None
                bye = {"id": self._next_id, "cmd": "shutdown", "args": {}}
                proc.stdin.write((json.dumps(bye) + "\n").encode())
                proc.stdin.flush()
                proc.stdin.close()
            except (OSError, ValueError, AssertionError):
                pass
        for req in pending:
            req._fail("afv serve was shut down")

        def reap():
            try:
                proc.wait(timeout=wait or 10.0)
            except subprocess.TimeoutExpired:
                proc.kill()

        if wait:
            reap()
        else:
            threading.Thread(target=reap, name="afv-serve-reap", daemon=True).start()


# --- the one-shot fallback ------------------------------------------------------------------------
def cli_args(cmd: str, args: dict) -> list[str]:
    """The ``afv`` command line (without ``--json``) equivalent to a serve request."""
    shot, project = str(args["shot"]), str(args["project"])
    if cmd == "run":
        out = ["run", shot, "--project", project]
        if args.get("from"):
            out += ["--from", str(args["from"])]
        if args.get("to"):
            out += ["--to", str(args["to"])]
        if args.get("frames"):
            out += ["--frames", str(args["frames"])]
        for s in args.get("sets") or []:
            out += ["--set", str(s)]
        if args.get("yes"):
            out.append("--yes")
        if args.get("code_changes"):
            out += ["--code-changes", str(args["code_changes"])]
        return out
    if cmd == "take":
        preset = ["--preset", str(args["preset"])] if args.get("preset") else []
        return ["take", shot, "--project", project, *preset]
    if cmd == "export":
        return (["export", shot, "--project", project, "--preset", str(args["preset"])]
                + (["--yes"] if args.get("yes") else []))
    raise ValueError(f"no one-shot afv command for {cmd!r}")


def frames_arg(start: int, end: int) -> str:
    """``--frames`` / ``frames`` for [start, end) (end exclusive)."""
    return f"{int(start)}:{int(end)}"
