"""The shot browser's scripted end-to-end run on the owner's long video (round 6 track B acceptance), in
headless Blender with the real ``afv serve``: a scratch copy of ``projects/full_fight`` (never the project
itself), then open, (re)scan, queue 20 fighters-likely candidates, pick fighters in each (the two largest
people stand in for the owner's choice), start the queue and cancel it. Blender side:
``browser_checks.py`` mode ``e2e``.

GPU (the scan, and the batch's first stage before the cancel), so opt-in with ``AFV_GPU_TESTS=1``; skipped
without the project. ``AFV_BROWSER_E2E_SCAN=force`` scans every shot again (as a fresh project would;
about 75 s), else the copy's up-to-date scans are kept. ``AFV_BROWSER_E2E_DIR``: where the copy goes
(default: the test's tmp folder)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from take_helpers import REPO
from test_browser import run_check

from afv.export import blender as bl

PROJECT = REPO / "projects" / "full_fight"
pytestmark = [
    pytest.mark.blender,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(bl.find_blender() is None, reason="Blender not installed"),
    pytest.mark.skipif(not (PROJECT / "project.json").is_file(), reason="projects/full_fight not present"),
]


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    exe = bl.find_blender()
    base = Path(os.environ["AFV_BROWSER_E2E_DIR"]) if os.environ.get("AFV_BROWSER_E2E_DIR") else (
        tmp_path_factory.mktemp("browser_e2e"))
    copy = base / "full_fight_copy"
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(PROJECT, copy, ignore=shutil.ignore_patterns("stages", "take", "exports", ".addon"))
    for f in (copy / "shots").glob("*/edits/fighters.json"):  # the owner's picks stay with the owner
        f.unlink()
    job = {"project": bl.host_path(copy, exe), "queue": 20,
           "scan": os.environ.get("AFV_BROWSER_E2E_SCAN", "")}
    return run_check("e2e", job, base / "work", timeout=1800)


def test_end_to_end_machine_time(e2e):
    steps = dict(e2e["steps"])
    assert e2e["open"]["rows"] >= 100 and e2e["open"]["markers"] > 0 and e2e["open"]["clip"]
    assert steps["open"] <= 15.0
    assert len(e2e["queue"]["shots"]) == 20 and e2e["queue"]["max_s"] <= 1.0
    picked = [p for p in e2e["pick"] if "save_s" in p]
    assert len(picked) >= 18, e2e["pick"]
    assert all(p["source"] == "scan" and p["shown_s"] <= 1.0 and p["save_s"] <= 1.0 for p in picked)
    assert e2e["run"]["started"] is not None and e2e["run"]["first_stage"].get("stage")
    assert e2e["total_s"] <= 180.0, e2e["steps"]
