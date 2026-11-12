"""A real ``afv serve`` subprocess (fake CPU stages, see fake_stages.py) and a small client."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from afv.core.project import Project

HERE = Path(__file__).resolve().parent
TERMINAL = ("result", "error")


class Client:
    def __init__(self, env: dict[str, str], log: Path):
        self.log = log
        self._logf = open(log, "ab")
        self.proc = subprocess.Popen(
            [sys.executable, str(HERE / "worker_main.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._logf, env=env,
        )  # fmt: skip
        self.lines: list[str] = []  # every line received, in order
        self.by_id: dict[Any, list[dict]] = {}
        self.bad: list[str] = []  # lines that were not JSON objects
        self._cond = threading.Condition()
        self._n = 0
        self.eof = False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            with self._cond:
                self.lines.append(line)
                try:
                    ev = json.loads(line)
                    assert isinstance(ev, dict)
                except (ValueError, AssertionError):
                    self.bad.append(line)
                else:
                    self.by_id.setdefault(ev.get("id"), []).append(ev)
                self._cond.notify_all()
        with self._cond:
            self.eof = True
            self._cond.notify_all()

    def send(self, cmd: str, args: dict | None = None, rid: Any = None) -> Any:
        if rid is None:
            self._n += 1
            rid = self._n
        self.write(json.dumps({"id": rid, "cmd": cmd, "args": args or {}}).encode() + b"\n")
        return rid

    def write(self, data: bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def wait(self, rid: Any, timeout: float = 60.0) -> list[dict]:
        """Events of request ``rid`` up to and including its terminal one."""
        end = time.monotonic() + timeout
        with self._cond:
            while True:
                evs = self.by_id.get(rid, [])
                if any(e.get("event") in TERMINAL for e in evs):
                    return list(evs)
                left = end - time.monotonic()
                if left <= 0 or self.eof:
                    log = self.log.read_text()[-3000:]
                    raise AssertionError(f"no reply to {rid} (got {evs}); log:\n{log}")
                self._cond.wait(min(left, 0.5))

    def request(self, cmd: str, args: dict | None = None, timeout: float = 60.0) -> tuple[dict, list[dict]]:
        evs = self.wait(self.send(cmd, args), timeout)
        assert sum(e.get("event") in TERMINAL for e in evs) == 1, evs
        return evs[-1], evs

    def close(self) -> None:
        if self.proc.poll() is None:
            with contextlib.suppress(OSError):
                assert self.proc.stdin is not None
                self.proc.stdin.close()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._logf.close()


@pytest.fixture
def stages_file(tmp_path) -> Path:
    p = tmp_path / "stages" / "fake_stages_copy.py"
    p.parent.mkdir()
    shutil.copyfile(HERE / "fake_stages.py", p)
    return p


@pytest.fixture
def project(tmp_path) -> Project:
    video = tmp_path / "src" / "fight.mp4"
    video.parent.mkdir()
    video.write_bytes(np.random.default_rng(0).bytes(4096))
    proj = Project.create(tmp_path / "fight.afv", video)
    proj.add_shot(0.0, 2.0)
    proj.add_shot(2.0, 2.0)
    return proj


@pytest.fixture
def worker_env(tmp_path, stages_file) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "AFV_TEST_STAGES": str(stages_file), "AFV_CONFIG_DIR": str(tmp_path / "config"),
        "AFV_HOME": str(tmp_path / "home"), "PYTHONUNBUFFERED": "1",
    })  # fmt: skip
    env.pop("AFV_PROJECT", None)
    return env


@pytest.fixture
def client(tmp_path, worker_env):
    c = Client(worker_env, tmp_path / "worker.log")
    yield c
    c.close()


@pytest.fixture
def new_client(tmp_path):
    """Start a worker with a given environment: ``new_client(env)``; stopped after the test."""
    made: list[Client] = []

    def make(env: dict[str, str]) -> Client:
        c = Client(env, tmp_path / f"worker{len(made)}.log")
        made.append(c)
        return c

    yield make
    for c in made:
        c.close()
