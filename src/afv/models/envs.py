"""Model environments and the subprocess bridge to worker scripts.

Environments
------------
Heavy stages run their model in a pinned environment as a subprocess. Built-in defaults (paths in
WSL, ``~`` = the user's home, ``$AFV_HOME`` = ``~/afv``):

=========  ======================================================  ====  ==========================
name       interpreter                                             GPU   notes
=========  ======================================================  ====  ==========================
cpu        the afv interpreter itself (``sys.executable``)         no
gemx       ``~/afv/gem-x/.venv/bin/python``                        yes   cwd ``~/afv/gem-x``
track      ``~/afv/track/.venv/bin/python``                        yes   cwd ``~/afv/track``
vipe       ``~/afv/vipe/.venv/bin/python``                         yes   conda env ``cu128`` first
blender    ``/mnt/c/Program Files/.../Blender 5.2/blender.exe``   no    ``$AFV_BLENDER`` overrides
=========  ======================================================  ====  ==========================

Override or add environments in ``~/.config/afv/envs.toml``::

    [gemx]
    python = "/home/me/afv/gem-x/.venv/bin/python"   # or: executable = ...
    cwd = "/home/me/afv/gem-x"
    setup = "source ~/miniforge/etc/profile.d/conda.sh && conda activate cu128"
    gpu = true
    timeout_s = 7200
    [gemx.env]
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE = "0"

Worker protocol
---------------
``run_worker(env, script, request, work_dir)``:

1. writes ``request`` to ``<work_dir>/worker/<name>.request.json``;
2. runs ``<python> <script> <request.json> <response.json>`` in the environment (for Blender:
   ``blender.exe --background --factory-startup --python-exit-code 1 --python <script> -- <request>
   <response>`` with Windows paths). The same paths are in ``$AFV_REQUEST``, ``$AFV_RESPONSE`` and
   ``$AFV_WORK_DIR`` (not for Blender). Relative paths inside the request are the caller's business;
   pass absolute paths;
3. streams stdout+stderr to ``<work_dir>/logs/<name>.log``. A line ``AFV_PROGRESS <0..1> <message>``
   is reported to the ``progress`` callback;
4. the worker writes a JSON object to the response path and exits 0. Any non-zero exit, a timeout
   or a missing/invalid response raises ``WorkerError`` (a ``StageError``) with the log tail. A worker
   that fails on purpose may write ``{"error": "<message>"}`` before exiting non-zero; that message
   leads the error.

GPU jobs (``env.gpu``) hold an exclusive ``flock`` on ``~/afv/.gpu.lock`` (the same lock as the shell's
``flock ~/afv/.gpu.lock ...``) for the whole subprocess, so only one GPU job runs at a time.

Cancelling (``afv serve``): inside ``cancel_scope(event)`` a set event stops the wait for the GPU lock
and refuses to start a worker (``WorkerCancelled``); the caller stops a running worker's process group.

Workers run in their own session (``start_new_session``) and die with the afv thread that started
them (``PR_SET_PDEATHSIG``), set by exec'ing through ``setpriv --pdeathsig KILL`` (util-linux) rather
than a ``preexec_fn`` (unsafe in a process with threads, e.g. ``afv serve``). Without ``setpriv`` the
``preexec_fn`` is used only while afv has a single thread.
"""

from __future__ import annotations

import codecs
import contextlib
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from afv.core.cache import atomic_write_json, read_json
from afv.core.config import ConfigError, afv_home, config_dir, load_machine_config, load_toml
from afv.core.stage import StageError

WORKERS_DIR = Path(__file__).parent / "workers"
DEFAULT_BLENDER = "/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"
PROGRESS_PREFIX = "AFV_PROGRESS"
TAIL_LINES = 40
SETUP_FAILED = 97  # bash wrapper's exit code when an environment's setup snippet fails
_DROP_VARS = ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "__PYVENV_LAUNCHER__")

Progress = Callable[[float, str], None]
_log = logging.getLogger("afv.envs")


class WorkerError(StageError):
    """A worker subprocess failed; ``log_path`` holds its full output."""

    def __init__(self, message: str, *, log_path: Path | None = None, returncode: int | None = None):
        super().__init__(message)
        self.log_path = log_path
        self.returncode = returncode


class WorkerCancelled(WorkerError):
    """The caller cancelled (``cancel_scope``) before or while waiting for the worker to start."""


_cancel = threading.local()


@contextlib.contextmanager
def cancel_scope(event: threading.Event | None) -> Iterator[None]:
    """Worker launches in this thread give up once ``event`` is set (see the module docstring)."""
    old = getattr(_cancel, "event", None)
    _cancel.event = event
    try:
        yield
    finally:
        _cancel.event = old


def cancel_requested() -> bool:
    ev = getattr(_cancel, "event", None)
    return ev is not None and ev.is_set()


class GpuBusyError(StageError):
    """The GPU lock could not be taken within the allowed time."""


@dataclass(frozen=True)
class EnvSpec:
    name: str
    executable: Path  # python interpreter, or blender for kind == "blender"
    kind: Literal["python", "blender"] = "python"
    cwd: Path | None = None
    setup: str = ""  # bash snippet run before exec (e.g. conda activate)
    env: Mapping[str, str] = field(default_factory=dict)
    gpu: bool = False
    timeout_s: float | None = None  # None: machine default (config.toml worker_timeout_s)

    def argv(self, script: Path, request: Path, response: Path) -> list[str]:
        if self.kind == "blender":
            conv = to_windows_path if str(self.executable).lower().endswith(".exe") else str
            core = [
                str(self.executable),
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                conv(script),
                "--",
                conv(request),
                conv(response),
            ]
        else:
            core = [str(self.executable), "-u", str(script), str(request), str(response)]
        if self.setup:
            wrapper = f'{{\n{self.setup}\n}} || exit {SETUP_FAILED}\nexec "$@"'
            return ["bash", "-c", wrapper, "afv-worker", *core]
        return core

    def environ(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _DROP_VARS}
        path = [str(Path("~/.local/bin").expanduser())]
        if self.kind == "python":
            path.insert(0, str(Path(self.executable).parent))
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONNOUSERSITE"] = "1"
        env["PATH"] = os.pathsep.join([*path, env.get("PATH", "")])
        env.update({k: os.path.expandvars(os.path.expanduser(str(v))) for k, v in self.env.items()})
        if extra:
            env.update(extra)
        return env


def _home(*parts: str) -> Path:
    return afv_home().joinpath(*parts)


def default_envs() -> dict[str, EnvSpec]:
    return {
        "cpu": EnvSpec("cpu", Path(sys.executable)),
        "gemx": EnvSpec("gemx", _home("gem-x", ".venv", "bin", "python"), cwd=_home("gem-x"), gpu=True),
        "track": EnvSpec("track", _home("track", ".venv", "bin", "python"), cwd=_home("track"), gpu=True),
        "vipe": EnvSpec(
            "vipe",
            _home("vipe", ".venv", "bin", "python"),
            cwd=_home("vipe"),
            setup=f'source "{_home("miniforge", "etc", "profile.d", "conda.sh")}" && conda activate cu128',
            gpu=True,
        ),
        "blender": EnvSpec("blender", Path(os.environ.get("AFV_BLENDER", DEFAULT_BLENDER)), kind="blender"),
    }


_ENV_KEYS = {"python", "executable", "kind", "cwd", "setup", "env", "gpu", "timeout_s"}


def load_envs(path: Path | None = None) -> dict[str, EnvSpec]:
    """Built-in environments with ``envs.toml`` overrides applied."""
    path = path or config_dir() / "envs.toml"
    envs = default_envs()
    for name, over in load_toml(path).items():
        if not isinstance(over, dict):
            raise ConfigError(f"{path}: [{name}] must be a table")
        bad = set(over) - _ENV_KEYS
        if bad:
            raise ConfigError(f"{path}: [{name}] unknown keys {sorted(bad)}; allowed: {sorted(_ENV_KEYS)}")
        exe = over.get("executable", over.get("python"))
        base = envs.get(name)
        if base is None:
            if exe is None:
                raise ConfigError(f"{path}: new environment [{name}] needs 'python' or 'executable'")
            base = EnvSpec(name, Path(exe))
        kw: dict[str, Any] = {}
        if exe is not None:
            kw["executable"] = Path(str(exe)).expanduser()
        if "kind" in over:
            if over["kind"] not in ("python", "blender"):
                raise ConfigError(f"{path}: [{name}] kind must be 'python' or 'blender'")
            kw["kind"] = over["kind"]
        if "cwd" in over:
            kw["cwd"] = Path(str(over["cwd"])).expanduser() if over["cwd"] else None
        if "setup" in over:
            kw["setup"] = str(over["setup"])
        if "env" in over:
            kw["env"] = {**base.env, **{str(k): str(v) for k, v in dict(over["env"]).items()}}
        if "gpu" in over:
            kw["gpu"] = bool(over["gpu"])
        if "timeout_s" in over:
            kw["timeout_s"] = float(over["timeout_s"]) if over["timeout_s"] else None
        envs[name] = replace(base, **kw)
    return envs


def get_env(name: str, envs: Mapping[str, EnvSpec] | None = None) -> EnvSpec:
    envs = load_envs() if envs is None else envs
    try:
        return envs[name]
    except KeyError:
        raise StageError(f"unknown model environment {name!r}; known: {sorted(envs)}") from None


def check_env(spec: EnvSpec) -> list[str]:
    """Problems that would stop a worker from starting (empty list = looks runnable)."""
    problems = []
    exe = Path(spec.executable)
    if not exe.exists():
        problems.append(f"{spec.name}: executable not found: {exe}")
    elif not os.access(exe, os.X_OK):
        problems.append(f"{spec.name}: not executable: {exe}")
    if spec.cwd is not None and not Path(spec.cwd).is_dir():
        problems.append(f"{spec.name}: working folder not found: {spec.cwd}")
    return problems


def to_windows_path(p: Path | str) -> str:
    """WSL path -> Windows path (for Windows executables such as blender.exe)."""
    try:
        out = subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        raise StageError(f"cannot convert {p} to a Windows path (wslpath): {e}") from None
    return out.stdout.strip()


def worker_script(script: str | Path) -> Path:
    """Resolve a worker: a path, or a name in ``afv/models/workers`` (``"gemx_worker"``)."""
    p = Path(script)
    if not p.is_absolute() and len(p.parts) == 1:
        p = WORKERS_DIR / (p.name if p.suffix == ".py" else f"{p.name}.py")
    if not p.is_file():
        raise StageError(f"worker script not found: {p}")
    return p.resolve()


# ---------------------------------------------------------------------------
# GPU lock
# ---------------------------------------------------------------------------

_held = threading.local()


@contextlib.contextmanager
def gpu_lock(
    path: Path | None = None,
    *,
    timeout: float | None = None,
    on_wait: Callable[[str], None] | None = None,
    poll_s: float = 0.5,
) -> Iterator[int]:
    """Exclusive ``flock`` on the GPU lock file; yields the lock's file descriptor.

    Re-entrant within a thread (nested use yields the same fd). The lock belongs to the open file,
    so a worker given a copy of the fd (``run_worker`` passes it) keeps the GPU locked until it and
    anything holding that copy have exited, even if afv itself dies.
    """
    import fcntl  # POSIX only: afv runs inside WSL

    path = Path(path).expanduser() if path else load_machine_config().gpu_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = os.path.realpath(path)
    held: dict[str, int] = getattr(_held, "fds", None) or {}
    _held.fds = held
    if key in held:
        yield held[key]
        return
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        t0 = time.monotonic()
        announced = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if cancel_requested():
                    raise WorkerCancelled("cancelled while waiting for the GPU") from None
                holder = _read_holder(path)
                if not announced and on_wait is not None:
                    on_wait(f"waiting for the GPU (lock {path} held{' by ' + holder if holder else ''})")
                    announced = True
                if timeout is not None and time.monotonic() - t0 >= timeout:
                    raise GpuBusyError(
                        f"GPU busy: could not take {path} within {timeout:.0f} s"
                        + (f" (held by {holder})" if holder else "")
                    ) from None
                time.sleep(poll_s)
        with contextlib.suppress(OSError):
            os.ftruncate(fd, 0)
            os.pwrite(fd, f"pid {os.getpid()} {' '.join(sys.argv[:3])}\n".encode(), 0)
        held[key] = fd
        try:
            yield fd
        finally:
            held.pop(key, None)
    finally:
        os.close(fd)  # no LOCK_UN: a worker still holding a copy of the fd keeps the lock


def _read_holder(path: Path) -> str:
    try:
        return path.read_text(errors="replace").strip()[:200]
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Subprocess bridge
# ---------------------------------------------------------------------------

_PROGRESS_RE = re.compile(rf"^{PROGRESS_PREFIX}\s+([0-9.eE+-]+)\s*(.*)$")


def _unique_name(base: str, folder: Path) -> str:
    name, k = base, 2
    while (folder / f"{name}.request.json").exists():
        name, k = f"{base}-{k}", k + 1
    return name


def run_worker(
    env: str | EnvSpec,
    script: str | Path,
    request: dict[str, Any],
    work_dir: Path,
    *,
    timeout: float | None = None,
    gpu: bool | None = None,
    name: str | None = None,
    progress: Progress | None = None,
    lock_timeout: float | None = None,
    lock_path: Path | None = None,
    envs: Mapping[str, EnvSpec] | None = None,
) -> dict[str, Any]:
    """Run ``script`` in ``env`` with ``request``; return the worker's JSON response.

    ``timeout`` (seconds) bounds the subprocess, not the wait for the GPU lock (``lock_timeout``).
    ``name`` names the request/response/log files (default: script name; made unique per work dir).
    """
    spec = env if isinstance(env, EnvSpec) else get_env(env, envs)
    problems = check_env(spec)
    if problems:
        raise WorkerError("; ".join(problems) + " (see `afv models --check`, or ~/.config/afv/envs.toml)")
    script_path = worker_script(script)
    work_dir = Path(work_dir).resolve()
    wdir, logs = work_dir / "worker", work_dir / "logs"
    wdir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    name = _unique_name(name or script_path.stem, wdir)
    req, resp, log = wdir / f"{name}.request.json", wdir / f"{name}.response.json", logs / f"{name}.log"
    atomic_write_json(req, request)
    with contextlib.suppress(FileNotFoundError):
        resp.unlink()

    machine = load_machine_config()
    if timeout is None:
        timeout = spec.timeout_s if spec.timeout_s is not None else machine.worker_timeout_s
    use_gpu = spec.gpu if gpu is None else gpu
    argv = spec.argv(script_path, req, resp)
    extra = (
        {}
        if spec.kind == "blender"
        else {
            "AFV_REQUEST": str(req),
            "AFV_RESPONSE": str(resp),
            "AFV_WORK_DIR": str(work_dir),
        }
    )
    environ = spec.environ(extra)

    safe_progress = _SafeProgress(progress)

    def note(msg: str) -> None:
        safe_progress(-1.0, msg)

    if use_gpu:
        lock_cm: contextlib.AbstractContextManager[int | None] = gpu_lock(
            lock_path or machine.gpu_lock_path(),
            timeout=lock_timeout if lock_timeout is not None else machine.gpu_lock_timeout_s,
            on_wait=note,
        )
    else:
        lock_cm = contextlib.nullcontext()
    t0 = time.monotonic()
    with lock_cm as lock_fd:
        if cancel_requested():
            raise WorkerCancelled(f"cancelled before {script_path.name} started")
        rc, tail, timed_out, sink = _run_streaming(
            argv,
            spec.cwd,
            environ,
            log,
            timeout,
            safe_progress,
            spec.name,
            pass_fds=(lock_fd,) if lock_fd is not None else (),
        )
    elapsed = time.monotonic() - t0
    if safe_progress.error is not None:
        _log.warning(
            "progress display failed (%s: %s); the worker ran on regardless",
            type(safe_progress.error).__name__,
            safe_progress.error,
        )
    if sink.error is not None:
        _log.warning("could not write the worker log %s: %s", log, sink.error)

    def fail(headline: str) -> WorkerError:
        detail = _response_error(resp)
        text = headline + (f": {detail}" if detail else "")
        text += f"\n--- last {len(tail)} lines of {log} ---\n" + "\n".join(tail)
        return WorkerError(text, log_path=log, returncode=rc)

    label = f"worker {script_path.name} (env {spec.name})"
    if timed_out:
        raise fail(f"{label} timed out after {timeout:.0f} s and was killed")
    if rc == SETUP_FAILED and spec.setup:
        raise fail(f"{label}: the environment setup failed ({spec.setup!r}; see envs.toml)")
    if rc == SETPRIV_EXEC_FAILED and _setpriv() is not None and any("failed to execute" in t for t in tail):
        raise WorkerError(f"cannot start worker ({argv[0]}): {tail[-1].strip()}", log_path=log)
    if rc != 0:
        how = f"was killed by signal {-rc}" if rc < 0 else f"failed with exit code {rc}"
        raise fail(f"{label} {how} after {elapsed:.1f} s")
    try:
        out = read_json(resp)
    except FileNotFoundError:
        raise fail(f"{label} exited 0 but wrote no response ({resp})") from None
    except ValueError as e:
        raise fail(f"{label} wrote an invalid response {resp}: {e}") from None
    if not isinstance(out, dict):
        raise fail(f"{label} response must be a JSON object, got {type(out).__name__}")
    return out


def _response_error(resp: Path) -> str:
    try:
        data = read_json(resp)
    except (OSError, ValueError):
        return ""
    return str(data.get("error", "")) if isinstance(data, dict) else ""


class _LogSink:
    """Worker log file shared by the pump thread and the caller; write errors are kept, not raised."""

    def __init__(self, path: Path):
        self._f: Any = open(path, "a", encoding="utf-8", errors="replace")
        self._lock = threading.Lock()
        self.error: Exception | None = None

    def write(self, text: str) -> None:
        with self._lock:
            if self._f is None or self.error is not None:
                return
            try:
                self._f.write(text)
                self._f.flush()
            except (OSError, ValueError) as e:
                self.error = e

    def close(self, trailer: str = "") -> None:
        with self._lock:
            if self._f is None:
                return
            with contextlib.suppress(OSError):
                if trailer and self.error is None:
                    self._f.write(trailer)
                self._f.close()
            self._f = None


class _SafeProgress:
    """Progress callback that can never break the worker: the first error disables it."""

    def __init__(self, fn: Progress | None):
        self.fn = fn
        self.error: BaseException | None = None

    def __call__(self, frac: float, msg: str) -> None:
        if self.fn is None or self.error is not None:
            return
        try:
            self.fn(frac, msg)
        except Exception as e:  # e.g. BrokenPipeError from `afv run --json | head`
            self.error = e


_SETPRIV: list[str | None] = []
SETPRIV_EXEC_FAILED = 127  # setpriv's exit code when it cannot execute the worker


def death_signal_note() -> str:
    """'' when workers get PR_SET_PDEATHSIG through setpriv, else why not (for a one-time warning)."""
    if _setpriv() is not None or not sys.platform.startswith("linux"):
        return ""
    return ("setpriv (util-linux) not found: workers started by threads (afv serve) will not die with "
            "afv if it is killed (install util-linux)")


def _setpriv() -> str | None:
    if not _SETPRIV:
        import shutil

        _SETPRIV.append(shutil.which("setpriv") if sys.platform.startswith("linux") else None)
    return _SETPRIV[0]


def _death_signal(argv: list[str]) -> tuple[list[str], Callable[[], None] | None]:
    """(argv, preexec_fn) so the worker dies with the afv thread that starts it (module docstring)."""
    if (sp := _setpriv()) is not None:
        return [sp, "--pdeathsig", "KILL", "--", *argv], None
    if threading.active_count() == 1:
        return argv, _child_init(os.getpid())
    return argv, None  # threads and no setpriv: the caller stops its workers itself (afv serve does)


def _child_init(parent_pid: int) -> Callable[[], None]:
    """In the child before exec: die with SIGKILL when the afv process that started it dies."""

    def init() -> None:
        if _LIBC is not None:
            _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        if os.getppid() != parent_pid:  # afv already died between fork and prctl
            os._exit(1)

    return init


def _load_libc() -> Any:
    try:
        import ctypes

        return ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        return None


_LIBC = _load_libc() if sys.platform.startswith("linux") else None
_PR_SET_PDEATHSIG = 1


def _run_streaming(
    argv: list[str],
    cwd: Path | None,
    environ: dict[str, str],
    log: Path,
    timeout: float | None,
    progress: _SafeProgress,
    env_name: str,
    pass_fds: tuple[int, ...] = (),
) -> tuple[int, list[str], bool, _LogSink]:
    """Run ``argv`` in its own process group, streaming output to ``log``.

    Whatever happens (normal exit, timeout, Ctrl-C, SIGTERM turned into SystemExit by the CLI),
    every process left in the worker's group is stopped before this returns, so a GPU lock held by
    the caller is never released while part of the job still runs.
    """
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    sink = _LogSink(log)
    sink.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')} env={env_name} cwd={cwd or os.getcwd()}\n")
    sink.write(f"# {shlex.join(argv)}\n")
    argv, preexec = _death_signal(list(argv))
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=environ,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=pass_fds,
            preexec_fn=preexec,
        )
    except OSError as e:
        sink.close()
        raise WorkerError(f"cannot start worker ({argv[0]}): {e}", log_path=log) from None
    reader = threading.Thread(target=_pump, args=(proc, sink, tail, progress), daemon=True)
    reader.start()
    timed_out = False
    note = ""
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:  # also on Ctrl-C / SystemExit: never leave part of a GPU job running behind us
        if proc.poll() is None or _group_alive(proc.pid):
            if proc.poll() is not None:
                note = "# the worker exited but left processes running; they were stopped\n"
            _kill_group(proc)
        reader.join(timeout=10)
        sink.close(f"{note}# exit {proc.returncode}{' (timeout)' if timed_out else ''}\n")
    return proc.returncode if proc.returncode is not None else -9, list(tail), timed_out, sink


def _pump(proc: subprocess.Popen[bytes], sink: _LogSink, tail: deque[str], progress: _SafeProgress) -> None:
    """Drain the worker's output until EOF, whatever the log file or the progress callback do."""
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    partial = ""
    while True:
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        text = dec.decode(chunk)
        sink.write(text)
        partial += text
        *lines, partial = re.split(r"\r\n|\r|\n", partial)
        for line in lines:
            _line(line, tail, progress)
    rest = partial + dec.decode(b"", final=True)
    if rest:
        _line(rest, tail, progress)
    with contextlib.suppress(OSError):
        proc.stdout.close()


def _line(line: str, tail: deque[str], progress: _SafeProgress) -> None:
    if not line.strip():
        return
    m = _PROGRESS_RE.match(line.strip())
    if m:
        try:
            frac = float(m.group(1))
        except ValueError:
            frac = -1.0
        progress(frac, m.group(2))
        return
    tail.append(line)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM the worker's process group, then SIGKILL survivors; wait until it is gone."""
    for sig, wait in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            proc.poll()
            return
        except PermissionError:
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            proc.poll()  # reap the leader, or its zombie keeps the group alive
            if proc.returncode is not None and not _group_alive(proc.pid):
                return
            time.sleep(0.05)
