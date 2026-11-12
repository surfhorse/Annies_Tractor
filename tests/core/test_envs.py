from __future__ import annotations

import fcntl
import itertools
import json
import os
import signal
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from afv.core.config import ConfigError
from afv.core.stage import StageError
from afv.models import envs as envs_mod
from afv.models.envs import EnvSpec, GpuBusyError, WorkerError, gpu_lock, load_envs, run_worker

CPU = EnvSpec("cpu", Path(sys.executable))


def worker(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / "workers" / f"{name}.py"
    p.parent.mkdir(exist_ok=True)
    p.write_text(
        "import json, os, sys, time\nreq_path, resp_path = sys.argv[1], sys.argv[2]\n"
        "req = json.load(open(req_path))\n" + textwrap.dedent(body)
    )
    return p


def test_round_trip_progress_and_log(tmp_path):
    w = worker(
        tmp_path,
        "echo",
        """
        print("loading model", flush=True)
        print("AFV_PROGRESS 0.5 half way", flush=True)
        sys.stderr.write("tqdm 10%\\rtqdm 100%\\n")
        assert os.environ["AFV_REQUEST"] == req_path
        json.dump({"echo": req["x"], "cwd": os.getcwd(), "exe": sys.executable}, open(resp_path, "w"))
    """,
    )
    got = []
    out = run_worker(CPU, w, {"x": [1, 2]}, tmp_path / "work", progress=lambda f, m: got.append((f, m)))
    assert out["echo"] == [1, 2]
    assert got == [(0.5, "half way")]
    log = (tmp_path / "work" / "logs" / "echo.log").read_text()
    assert "loading model" in log and "tqdm 100%" in log and "# exit 0" in log
    assert json.loads((tmp_path / "work" / "worker" / "echo.request.json").read_text()) == {"x": [1, 2]}
    # A second call in the same work dir gets its own files.
    run_worker(CPU, w, {"x": 3}, tmp_path / "work")
    assert (tmp_path / "work" / "worker" / "echo-2.response.json").is_file()


def test_failure_has_exit_code_message_and_tail(tmp_path):
    w = worker(
        tmp_path,
        "bad",
        """
        for i in range(100):
            print(f"line {i}")
        json.dump({"error": "CUDA out of memory"}, open(resp_path, "w"))
        sys.exit(3)
    """,
    )
    with pytest.raises(WorkerError) as e:
        run_worker(CPU, w, {}, tmp_path / "work")
    msg = str(e.value)
    assert isinstance(e.value, StageError)
    assert "exit code 3" in msg and "CUDA out of memory" in msg
    assert "line 99" in msg and "line 10\n" not in msg  # only the tail
    assert e.value.log_path.read_text().count("line ") == 100


def test_traceback_reaches_the_error(tmp_path):
    w = worker(tmp_path, "tb", "raise KeyError('missing weights')\n")
    with pytest.raises(WorkerError, match="missing weights"):
        run_worker(CPU, w, {}, tmp_path / "work")


def test_missing_or_bad_response(tmp_path):
    with pytest.raises(WorkerError, match="wrote no response"):
        run_worker(CPU, worker(tmp_path, "none", "pass\n"), {}, tmp_path / "w1")
    with pytest.raises(WorkerError, match="must be a JSON object"):
        run_worker(
            CPU, worker(tmp_path, "list", "json.dump([1], open(resp_path, 'w'))\n"), {}, tmp_path / "w2"
        )


def test_stale_response_is_not_reused(tmp_path):
    w = worker(tmp_path, "w", "pass\n")
    resp = tmp_path / "work" / "worker" / "x.response.json"
    resp.parent.mkdir(parents=True)
    resp.write_text("{}")
    with pytest.raises(WorkerError, match="wrote no response"):
        run_worker(CPU, w, {}, tmp_path / "work", name="x")


def test_timeout_kills_the_process_group(tmp_path):
    w = worker(
        tmp_path,
        "slow",
        """
        import subprocess
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open(req["pidfile"], "w").write(str(child.pid))
        time.sleep(60)
    """,
    )
    pidfile = tmp_path / "child.pid"
    t0 = time.monotonic()
    with pytest.raises(WorkerError, match="timed out after 1 s"):
        run_worker(CPU, w, {"pidfile": str(pidfile)}, tmp_path / "work", timeout=1.0)
    assert time.monotonic() - t0 < 15
    child = int(pidfile.read_text())
    for _ in range(50):  # the grandchild was killed too (and reaped by init)
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("grandchild still alive after the timeout")


def test_missing_executable_and_script(tmp_path):
    bad = EnvSpec("gemx", tmp_path / "nope" / "python")
    with pytest.raises(WorkerError, match="executable not found"):
        run_worker(bad, worker(tmp_path, "w", "pass\n"), {}, tmp_path / "work")
    with pytest.raises(StageError, match="worker script not found"):
        run_worker(CPU, "no_such_worker", {}, tmp_path / "work")


def test_setup_snippet_and_env_vars(tmp_path):
    spec = EnvSpec(
        "x", Path(sys.executable), setup="export AFV_TEST_SETUP=yes", env={"AFV_TEST_VAR": "v"}, cwd=tmp_path
    )
    w = worker(
        tmp_path,
        "envs",
        """
        json.dump({k: os.environ.get(k) for k in ("AFV_TEST_SETUP", "AFV_TEST_VAR", "VIRTUAL_ENV")}
                  | {"cwd": os.getcwd()}, open(resp_path, "w"))
    """,
    )
    out = run_worker(spec, w, {}, tmp_path / "work")
    assert out == {"AFV_TEST_SETUP": "yes", "AFV_TEST_VAR": "v", "VIRTUAL_ENV": None, "cwd": str(tmp_path)}


def test_gpu_lock_times_out_when_held(tmp_path):
    lock = tmp_path / "gpu.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)  # someone else (e.g. `flock ~/afv/.gpu.lock ...`) holds it
    try:
        notes = []
        with pytest.raises(GpuBusyError, match="GPU busy"):
            run_worker(
                CPU,
                worker(tmp_path, "w", "pass\n"),
                {},
                tmp_path / "work",
                gpu=True,
                lock_path=lock,
                lock_timeout=0.3,
                progress=lambda f, m: notes.append(m),
            )
        assert notes and "waiting for the GPU" in notes[0]
    finally:
        os.close(fd)


def test_gpu_jobs_do_not_overlap(tmp_path):
    lock = tmp_path / "gpu.lock"
    w = worker(
        tmp_path,
        "job",
        """
        t0 = time.time(); time.sleep(0.4); t1 = time.time()
        json.dump({"t0": t0, "t1": t1}, open(resp_path, "w"))
    """,
    )
    spans = []

    def go(i):
        spans.append(run_worker(CPU, w, {}, tmp_path / f"work{i}", gpu=True, lock_path=lock))

    threads = [threading.Thread(target=go, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    spans.sort(key=lambda s: s["t0"])
    assert len(spans) == 3
    for a, b in itertools.pairwise(spans):
        assert b["t0"] >= a["t1"] - 0.01


def test_gpu_lock_is_reentrant(tmp_path):
    lock = tmp_path / "gpu.lock"
    with gpu_lock(lock, timeout=0.2), gpu_lock(lock, timeout=0.2):
        pass
    with gpu_lock(lock, timeout=0.2):
        pass


def test_envs_toml_overrides(tmp_path):
    p = tmp_path / "envs.toml"
    p.write_text(
        textwrap.dedent("""
        [gemx]
        python = "/opt/gemx/bin/python"
        timeout_s = 60
        [gemx.env]
        FOO = "bar"
        [mine]
        executable = "/opt/mine/bin/python"
        gpu = true
    """)
    )
    envs = load_envs(p)
    assert envs["gemx"].executable == Path("/opt/gemx/bin/python")
    assert envs["gemx"].gpu and envs["gemx"].timeout_s == 60 and envs["gemx"].env == {"FOO": "bar"}
    assert envs["mine"].gpu and envs["vipe"].setup and envs["cpu"].executable == Path(sys.executable)
    p.write_text("[gemx]\npyton = 'x'\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_envs(p)
    p.write_text("[gemx\n")
    with pytest.raises(ConfigError):
        load_envs(p)


def test_default_envs_follow_afv_home(isolated_env):
    envs = load_envs()
    home = isolated_env / "home"
    assert envs["gemx"].executable == home / "gem-x" / ".venv" / "bin" / "python"
    assert envs["gemx"].cwd == home / "gem-x" and envs["track"].gpu and not envs["cpu"].gpu


def test_blender_argv(tmp_path):
    spec = EnvSpec("blender", Path("/usr/bin/blender"), kind="blender")
    argv = spec.argv(Path("/s.py"), Path("/req.json"), Path("/resp.json"))
    assert argv == [
        "/usr/bin/blender",
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        "/s.py",
        "--",
        "/req.json",
        "/resp.json",
    ]
    assert envs_mod.worker_script.__doc__


# --- process lifetime and the GPU lock ------------------------------------------------------

DRIVER = """
import sys
from pathlib import Path
sys.path.insert(0, {src!r})
from afv.models.envs import EnvSpec, run_worker
if {handlers}:
    from afv.cli import install_signal_handlers
    install_signal_handlers()
w, work, lock = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
run_worker(EnvSpec("cpu", Path(sys.executable)), w, {{"pidfile": sys.argv[4]}}, work, gpu=True,
           lock_path=lock)
"""


def _lock_free(lock: Path) -> bool:
    fd = os.open(lock, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as f:  # a zombie is dead for our purposes
        return f.read().split(") ")[-1][0] != "Z"


def _wait(cond, timeout=10.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


def _start_driver(tmp_path, handlers: bool):
    import subprocess

    src = str(Path(envs_mod.__file__).resolve().parents[2])
    drv = tmp_path / "driver.py"
    drv.write_text(DRIVER.format(src=src, handlers=handlers))
    w = worker(tmp_path, "sleeper", 'open(req["pidfile"], "w").write(str(os.getpid()))\ntime.sleep(60)\n')
    pidfile, lock = tmp_path / "worker.pid", tmp_path / "gpu.lock"
    proc = subprocess.Popen(
        [sys.executable, str(drv), str(w), str(tmp_path / "work"), str(lock), str(pidfile)]
    )
    assert _wait(lambda: pidfile.is_file() and pidfile.read_text()), "worker did not start"
    return proc, int(pidfile.read_text()), lock


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_sigterm_to_afv_stops_the_worker_and_keeps_the_gpu_locked_until_then(tmp_path, sig):
    proc, wpid, lock = _start_driver(tmp_path, handlers=True)
    assert not _lock_free(lock)
    proc.send_signal(sig)
    assert proc.wait(timeout=20) == 128 + sig
    assert not _alive(wpid)
    assert _lock_free(lock)
    assert "# exit" in (tmp_path / "work" / "logs" / "sleeper.log").read_text()


def test_killed_afv_takes_the_worker_with_it(tmp_path):
    proc, wpid, lock = _start_driver(tmp_path, handlers=False)
    proc.kill()  # SIGKILL: no clean-up code runs in afv at all
    proc.wait(timeout=10)
    assert _wait(lambda: not _alive(wpid)), "worker survived its parent (PR_SET_PDEATHSIG)"
    assert _wait(lambda: _lock_free(lock))


def test_leftover_processes_are_stopped_before_the_lock_is_released(tmp_path):
    w = worker(
        tmp_path,
        "leaky",
        """
        import subprocess
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open(req["pidfile"], "w").write(str(child.pid))
        json.dump({"ok": True}, open(resp_path, "w"))
    """,
    )  # exits 0 at once, leaving its child running
    pidfile, lock = tmp_path / "child.pid", tmp_path / "gpu.lock"
    out = run_worker(CPU, w, {"pidfile": str(pidfile)}, tmp_path / "work", gpu=True, lock_path=lock)
    assert out == {"ok": True}
    assert not _alive(int(pidfile.read_text()))
    assert _lock_free(lock)
    log = (tmp_path / "work" / "logs" / "leaky.log").read_text()
    assert "left processes running" in log and log.rstrip().endswith("# exit 0")


def test_raising_progress_callback_does_not_block_the_worker(tmp_path, caplog):
    w = worker(
        tmp_path,
        "chatty",
        """
        for i in range(3000):
            print(f"AFV_PROGRESS {i / 3000:.3f} step {i}")
            print("x" * 200)
        json.dump({"done": True}, open(resp_path, "w"))
    """,
    )  # ~600 kB of output: far more than a pipe buffer

    def broken(frac, msg):
        raise BrokenPipeError(32, "Broken pipe")

    t0 = time.monotonic()
    with caplog.at_level("WARNING", logger="afv.envs"):
        out = run_worker(CPU, w, {}, tmp_path / "work", progress=broken, timeout=60)
    assert out == {"done": True} and time.monotonic() - t0 < 30
    assert [r.getMessage() for r in caplog.records if "progress display failed" in r.getMessage()]
    assert (tmp_path / "work" / "logs" / "chatty.log").read_text().count("x" * 200) == 3000


def test_failing_setup_snippet_is_reported(tmp_path):
    spec = EnvSpec("vipe", Path(sys.executable), setup="source /nonexistent/conda.sh && conda activate x")
    with pytest.raises(WorkerError, match="environment setup failed"):
        run_worker(
            spec, worker(tmp_path, "w", "json.dump({}, open(resp_path, 'w'))\n"), {}, tmp_path / "work"
        )


def test_gpu_lock_reentrancy_follows_the_real_path(tmp_path):
    real = tmp_path / "real.lock"
    real.touch()
    link = tmp_path / "link.lock"
    link.symlink_to(real)
    with gpu_lock(real, timeout=0.2) as a, gpu_lock(link, timeout=0.2) as b:
        assert a == b


def test_cancel_stops_the_wait_for_the_gpu_and_the_start(tmp_path):
    import fcntl
    import threading

    from afv.models.envs import WorkerCancelled, cancel_scope

    lock = tmp_path / "gpu.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)  # someone else holds the GPU
    ev = threading.Event()
    ev.set()
    try:
        with cancel_scope(ev), pytest.raises(WorkerCancelled, match="waiting for the GPU"):
            with gpu_lock(lock, poll_s=0.01):
                pass
    finally:
        os.close(fd)
