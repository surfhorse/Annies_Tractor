"""Run ``afv`` in WSL from Windows Blender, off the main thread (no bpy here).

A ``Job`` runs one ``afv ... --json`` command through ``wsl.exe -d <distro> -e bash -lc '<cmd>'``
(``-e``: no extra shell re-parses the line; the user's arguments may not contain ``$``) with no
console window, reads its JSON lines on a background thread and queues them; the add-on drains the
queue from a ``bpy.app.timers`` callback, so the UI never waits.

The command first prints ``AFVPID:<pid>`` and then ``exec``s afv, so ``cancel`` can stop the Linux
process itself (ending wsl.exe alone leaves it running). wsl.exe's own messages are UTF-16 unless
``WSL_UTF8=1`` (set here); stray NULs are dropped anyway.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import PureWindowsPath

DEFAULT_DISTRO = "Ubuntu"
DEFAULT_AFV = "~/afv/afv-env/bin/afv"
PID_PREFIX = "echo AFVPID:$$; exec "  # $$ is bash's pid, which exec hands on to afv
_PID_LINE = re.compile(r"^AFVPID:(\d+)$")
_DRIVE = re.compile(r"^([A-Za-z]):[\\/]?(.*)$")
_UNC_WSL = re.compile(r"^[\\/]{2}wsl(?:\.localhost|\$)[\\/]([^\\/]+)[\\/]?(.*)$", re.IGNORECASE)
_MNT = re.compile(r"^/mnt/([a-z])(?:/(.*))?$")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def to_wsl_path(path: str) -> str:
    r"""``D:\x\y`` -> ``/mnt/d/x/y``; ``\\wsl.localhost\Ubuntu\home\u`` -> ``/home/u``; POSIX
    paths unchanged."""
    p = str(path)
    if p.startswith("/"):
        return p
    m = _UNC_WSL.match(p)
    if m:
        return "/" + m.group(2).replace("\\", "/").strip("/")
    m = _DRIVE.match(p)
    if m:
        rest = m.group(2).replace("\\", "/").strip("/")
        return f"/mnt/{m.group(1).lower()}" + (f"/{rest}" if rest else "")
    raise ValueError(f"cannot map {path!r} to a WSL path")


def to_windows_path(path: str, distro: str = DEFAULT_DISTRO) -> str:
    """``/mnt/d/x`` -> ``D:\\x``; other WSL paths -> ``\\\\wsl.localhost\\<distro>\\...``."""
    p = str(path)
    m = _MNT.match(p)
    if m:
        rest = (m.group(2) or "").replace("/", "\\")
        return f"{m.group(1).upper()}:\\{rest}"
    if p.startswith("/"):
        return f"\\\\wsl.localhost\\{distro}" + p.replace("/", "\\")
    return str(PureWindowsPath(p))


def afv_argv(args: list[str], distro: str = DEFAULT_DISTRO, afv: str = DEFAULT_AFV) -> list[str]:
    """The wsl.exe command line running ``afv <args>`` (``afv`` may start with ``~``)."""
    if "$" in afv or any("$" in a for a in args):
        raise ValueError("'$' is not allowed in afv commands run through WSL")
    cmd = PID_PREFIX + afv + " " + " ".join(shlex.quote(a) for a in args)
    return ["wsl.exe", "-d", distro, "-e", "bash", "-lc", cmd]


def decode_line(raw: bytes) -> str:
    """One output line: UTF-8, with the NULs of UTF-16 text (wsl.exe's own errors) dropped."""
    if b"\x00" in raw:
        raw = raw.replace(b"\x00", b"")
    return raw.decode("utf-8", "replace").lstrip("\ufeff").rstrip("\r\n")


class Job:
    """One background ``afv`` run. Poll ``drain()`` from the main thread; ``done`` once everything
    was drained after the process ended (``finished`` is set by the thread itself)."""

    def __init__(self, argv: list[str], label: str):
        self.argv = argv
        self.label = label
        self.distro = argv[argv.index("-d") + 1] if argv[:1] == ["wsl.exe"] and "-d" in argv else None
        self.linux_pid: int | None = None
        self.lines: queue.Queue[str] = queue.Queue()
        self.events: list[dict] = []
        self.text: list[str] = []
        self.returncode: int | None = None
        self.error: str | None = None
        self.finished = False
        self.done = False
        self.cancelled = False
        self.started = time.monotonic()
        self._proc: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._run, name=f"afv:{label}", daemon=True)

    def start(self) -> Job:
        self._thread.start()
        return self

    def _run(self) -> None:
        env = {**os.environ, "WSL_UTF8": "1"}
        try:
            self._proc = subprocess.Popen(
                self.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=_NO_WINDOW, env=env,
            )
            assert self._proc.stdout is not None
            for raw in self._proc.stdout:
                line = decode_line(raw)
                m = _PID_LINE.match(line.strip())
                if m:
                    self.linux_pid = int(m.group(1))
                    continue
                self.lines.put(line)
            self.returncode = self._proc.wait()
        except OSError as e:
            self.error = f"cannot start {self.argv[0]}: {e}"
            self.returncode = -1
        finally:
            self.finished = True
            self.lines.put("\0done")

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def drain(self) -> list[dict]:
        """New events since the last call (non-JSON lines become {"event": "text", "message": ..})."""
        new = []
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                break
            if line == "\0done":
                self.done = True
                continue
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
                if not isinstance(ev, dict):
                    raise ValueError
            except ValueError:
                ev = {"event": "text", "message": line}
                self.text.append(line)
            self.events.append(ev)
            new.append(ev)
        return new

    def cancel(self) -> None:
        """Stop afv in WSL (SIGTERM to its pid; afv stops its own workers), then wsl.exe."""
        self.cancelled = True
        if self.finished:
            return
        if self.linux_pid is not None and self.distro is not None:
            try:
                subprocess.run(["wsl.exe", "-d", self.distro, "-e", "kill", "-TERM", str(self.linux_pid)],
                               capture_output=True, timeout=15, creationflags=_NO_WINDOW)
            except (OSError, subprocess.SubprocessError):
                pass
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()

    @property
    def ok(self) -> bool:
        return self.done and self.returncode == 0 and self.error is None and not self.cancelled

    def result(self) -> dict | None:
        """The last ``{"event": "result"}`` or ``take``/``blend`` event, if any."""
        for ev in reversed(self.events):
            if ev.get("event") in ("result", "take", "blend"):
                return ev
        return None

    def review(self) -> dict | None:
        """The tracking review that paused the run (``afv export`` without ``--yes``), if any."""
        res = self.result() or {}
        if res.get("outcome") != "needs_review":
            return None
        ev = next((e for e in reversed(self.events) if e.get("event") == "tracking_review"), {})
        return {"message": ev.get("message") or res.get("message") or "tracking QC hints to review",
                "hints": ev.get("hints") or (res.get("review") or {}).get("hints") or [],
                "review_video": ev.get("review_video")}

    def failure(self) -> str:
        if self.error:
            return self.error
        if self.cancelled:
            return "cancelled"
        for ev in reversed(self.events):
            if ev.get("event") == "error":
                return str(ev.get("message"))
            if ev.get("event") == "result" and ev.get("outcome") not in (None, "ok"):
                return str(ev.get("message") or ev.get("outcome"))
        tail = " | ".join(self.text[-3:])
        return f"afv exited with {self.returncode}" + (f": {tail}" if tail else "")


def status_line(ev: dict) -> str:
    """A short human line for a progress event."""
    kind = ev.get("event")
    stage = ev.get("stage", "")
    if kind == "progress":
        frac = ev.get("frac")
        pct = f"{100 * frac:.0f}% " if isinstance(frac, (int, float)) and frac >= 0 else ""
        return f"{stage} {pct}{ev.get('message') or ''}".strip()
    if kind == "stage_start":
        return f"{stage} running"
    if kind == "stage_skip":
        return f"{stage} up to date"
    if kind == "stage_done":
        return f"{stage} done"
    if kind == "tracking_review":
        return "paused: tracking QC hints to review"
    if kind == "error":
        return f"error: {ev.get('message')}"
    return str(ev.get("message") or kind or "")
