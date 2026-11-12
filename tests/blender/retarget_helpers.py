"""Run ``retarget_checks.py`` in headless Blender for the retarget engine tests, and the pytest-side
metrics on its per-frame results (penetration against the source's own feet, acceleration at lock
edges).

The runner is like ``afv.export.blender.run_blender`` but adds ``--disable-autoexec`` (the Girush
test character is a .blend that ships its own scripts; they must never run)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
from take_helpers import REPO, TESTS_DIR

from afv.export import blender as bl

CHECKS = TESTS_DIR / "retarget_checks.py"
MEDIA = Path(os.environ.get("AFV_MEDIA_SAMPLES") or REPO / "Media_Samples")
CHARACTERS = {
    "mixamo": MEDIA / "Mixamo Mech" / "mixamo_mech.fbx",
    "unity": MEDIA / "Unity Humanoid Mech" / "Unity_Humanoid_Mech_rightarm_fixed.fbx",
    "girush": MEDIA / "Girush Young Warrior (CC0)" / "baseMesh_A1_(human_YW)_by_Girush.blend",
}
FACING = {"mixamo": "-Y", "unity": "-Y", "girush": "-Y"}  # their forward in their armature axes
FPS = 30.0
EDGE_WINDOW = 5
EDGE_STILL_LIMIT = 1800.0  # fighter cm/s^2: the bound where the free foot is nearly still


def run_checks(mode: str, job: dict, work: Path, timeout: float = 1200.0) -> dict:
    exe = bl.find_blender()
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    result = work / f"{mode}_result.json"
    result.unlink(missing_ok=True)
    job_file = work / f"{mode}_job.json"
    job = {"mode": mode, "addon_parent": bl.host_path(REPO / "blender", exe),
           "result_path": bl.host_path(result, exe), **job}
    job_file.write_text(json.dumps(job, indent=1))
    cmd = [str(exe), "-b", "--factory-startup", "--disable-autoexec", "-noaudio", "--python-exit-code", "1",
           "--python", bl.host_path(CHECKS, exe), "--", bl.host_path(job_file, exe)]
    proc = None
    for _ in range(2):  # WSL interop sometimes fails to launch a Windows process once
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=work)
        if result.is_file() or b"UtilAcceptVsock" not in proc.stderr + proc.stdout:
            break
    (work / f"{mode}_blender.log").write_bytes(proc.stdout + b"\n--- stderr ---\n" + proc.stderr)
    if not result.is_file():
        tail = (proc.stdout + proc.stderr)[-3000:].decode("utf-8", "replace")
        raise AssertionError(f"Blender wrote no result (exit {proc.returncode}):\n{tail}")
    res = json.loads(result.read_text(encoding="utf-8"))
    assert res["ok"], res.get("error")
    return res["out"]


def dump(name: str, data) -> Path | None:
    """Also write results to ``$AFV_RETARGET_DUMP/<name>.json`` (pytest's tmp folders get rotated)."""
    d = os.environ.get("AFV_RETARGET_DUMP")
    if not d:
        return None
    p = Path(d) / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1))
    return p


def planted(side: dict) -> np.ndarray:
    return np.array(side["foot"], bool) | np.array(side["toe"], bool)


def penetration(measure: dict) -> dict:
    """Per foot: worst penetration (fighter cm) on its planted frames, and the worst excess over the
    source's own lowest foot point on the other frames (both from the evaluated scene)."""
    out = {}
    for s, side in measure["sides"].items():
        pen, src = np.array(side["pen"]), np.array(side["src_depth"])
        pl = planted(side)
        out[s] = {"planted_frames": int(pl.sum()),
                  "planted_max": float(pen[pl].max()) if pl.any() else 0.0,
                  "other_excess_max": float((pen - src)[~pl].max()) if (~pl).any() else 0.0}
    return out


def _accel(p: np.ndarray) -> np.ndarray:
    a = np.zeros(len(p))
    a[1:-1] = np.linalg.norm(p[2:] - 2.0 * p[1:-1] + p[:-2], axis=-1) * FPS * FPS
    return a


def _near_edges(side: dict) -> np.ndarray:
    pl = planted(side)
    near = np.zeros(len(pl), bool)
    for e in np.flatnonzero(np.diff(pl.astype(np.int8))):
        near[max(0, e - EDGE_WINDOW + 1): e + EDGE_WINDOW + 1] = True
    near[[0, -1]] = False
    return near


def edge_correction_accel(locked: dict, free: dict) -> dict:
    """Per foot: the peak acceleration (fighter cm/s^2) of the lock's correction (locked ankle - free
    ankle, both evaluated) within +-EDGE_WINDOW frames of the plant edges."""
    out = {}
    for s, side in locked["sides"].items():
        near = _near_edges(side)
        corr = np.array(side["ankle"]) - np.array(free["sides"][s]["ankle"])
        out[s] = float(_accel(corr)[near].max()) if near.any() else None
    return out


def edge_accel(locked: dict, free: dict) -> dict:
    """Per foot, per plant edge: the peak ankle acceleration (fighter cm/s^2) within +-EDGE_WINDOW
    frames of that edge, locked and free (same window), and whether it passes acceptance 2: locked <=
    1.5 x free, or locked <= EDGE_STILL_LIMIT (where the free foot is nearly still)."""
    out = {}
    for s, side in locked["sides"].items():
        pl = planted(side)
        a_l = _accel(np.array(side["ankle"]))
        a_f = _accel(np.array(free["sides"][s]["ankle"]))
        edges = []
        for e in np.flatnonzero(np.diff(pl.astype(np.int8))).tolist():
            lo, hi = max(1, e - EDGE_WINDOW + 1), min(len(pl) - 1, e + EDGE_WINDOW + 1)
            pl_, pf_ = float(a_l[lo:hi].max()), float(a_f[lo:hi].max())
            edges.append({"edge": e, "locked": pl_, "free": pf_, "ratio": pl_ / max(pf_, 1e-9),
                          "ok": pl_ <= 1.5 * pf_ or pl_ <= EDGE_STILL_LIMIT})
        out[s] = edges
    return out
