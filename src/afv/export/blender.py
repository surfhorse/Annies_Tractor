"""Headless Blender bridge: find Blender, translate paths, run a script from ``blender/``.

afv runs in WSL2 and Blender is the Windows build, reached through WSL interop: every path handed
to Blender is converted with ``wslpath -w``. Files Blender reads or writes should live on the
Windows drive (``/mnt/<drive>/...``); WSL-only paths become ``\\\\wsl.localhost\\...`` UNC paths,
which work but are slow. On Windows (or a native Linux Blender) paths pass through unchanged.

Override the executable with ``AFV_BLENDER``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_BLENDER_DIR = Path(__file__).resolve().parents[3] / "blender"
DEFAULT_BLENDER_WSL = "/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"
DEFAULT_BLENDER_WIN = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"


class BlenderError(RuntimeError):
    """Blender (or the WSL path bridge) failed; the message is fit to show the user."""


def find_blender() -> Path | None:
    env = os.environ.get("AFV_BLENDER")
    cands = [env] if env else []
    cands += [DEFAULT_BLENDER_WIN] if sys.platform == "win32" else [DEFAULT_BLENDER_WSL]
    found = shutil.which("blender")
    if found:
        cands.append(found)
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    return None


def _needs_wslpath(blender: Path) -> bool:
    return sys.platform.startswith("linux") and blender.suffix.lower() == ".exe"


def host_path(p: Path, blender: Path) -> str:
    """Path string as Blender's process sees it."""
    p = Path(p).resolve()
    if not _needs_wslpath(blender):
        return str(p)
    try:
        out = subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        detail = getattr(e, "stderr", "") or str(e)
        msg = f"cannot convert {p} to a Windows path for Blender (wslpath): {detail}"
        raise BlenderError(msg.strip()) from e
    return out.stdout.strip()


def script_path(name: str) -> Path:
    p = REPO_BLENDER_DIR / name
    if not p.is_file():
        raise BlenderError(f"Blender script not found: {p}")
    return p


# WSL interop sometimes fails to launch a Windows process under load ("UtilAcceptVsock: accept4
# failed 110"); that launch failure (no result written) is retried once.
_INTEROP_LAUNCH_FAILURE = "UtilAcceptVsock"


def run_blender(script: str, job: dict, work_dir: Path, *, blender: Path | None = None,
                timeout: float = 900.0, attempts: int = 2) -> dict:
    """Run ``blender/<script>`` headless with ``job`` (JSON) and return the JSON it writes back.

    The job dict gets ``result_path`` added; paths inside ``job`` must already be host paths
    (see ``host_path``). Errors (start, timeout, failure, unreadable result) raise BlenderError.
    """
    blender = blender or find_blender()
    if blender is None:
        raise BlenderError("Blender not found; set AFV_BLENDER or install Blender 5.2")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    job_file = work_dir / f"{Path(script).stem}_job.json"
    result_file = work_dir / f"{Path(script).stem}_result.json"
    result_file.unlink(missing_ok=True)
    job = {**job, "result_path": host_path(result_file, blender)}
    job_file.write_text(json.dumps(job, indent=1))
    cmd = [
        str(blender), "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1",
        "--python", host_path(script_path(script), blender), "--", host_path(job_file, blender),
    ]
    for attempt in range(1, max(1, attempts) + 1):
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=work_dir)
        except subprocess.TimeoutExpired as e:
            raise BlenderError(f"Blender timed out after {timeout:.0f} s running {script}") from e
        except OSError as e:
            raise BlenderError(f"cannot start Blender ({blender}): {e}") from e
        log = b"".join([proc.stdout or b"", proc.stderr or b""]).decode("utf-8", "replace")
        (work_dir / f"{Path(script).stem}.log").write_text(log)
        launch_failed = not result_file.is_file() and _INTEROP_LAUNCH_FAILURE in log
        if not (launch_failed and attempt < attempts):
            break
    if proc.returncode != 0 or not result_file.is_file():
        tail = "\n".join(log.strip().splitlines()[-25:])
        raise BlenderError(f"Blender failed running {script} (exit {proc.returncode}):\n{tail}")
    try:
        return json.loads(result_file.read_text())
    except (OSError, ValueError) as e:
        raise BlenderError(f"cannot read Blender's result {result_file}: {e}") from e
