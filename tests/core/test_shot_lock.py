"""The shot lock holds across spellings of the shot's path (Windows drives are case-insensitive)."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from afv.core.project import Project, folder_identity
from afv.core.runner import RunError, Runner

DRVFS = Path(__file__).resolve().parents[2] / "tests" / "_out"


def test_same_folder_same_identity(tmp_path):
    (tmp_path / "a").mkdir()
    assert folder_identity(tmp_path / "a") == folder_identity(tmp_path / "a" / ".." / "a")
    assert folder_identity(tmp_path / "a") != folder_identity(tmp_path)


def test_lock_across_spellings_on_a_windows_drive(fakes):
    if not str(DRVFS).startswith("/mnt/"):
        pytest.skip("the repo is not on a Windows drive")
    d = DRVFS / f"lock-{uuid.uuid4().hex[:8]}"
    try:
        (d / "src").mkdir(parents=True)
        (d / "src" / "v.mp4").write_bytes(np.random.default_rng(0).bytes(1024))
        proj = Project.create(d / "Fight.afv", d / "src" / "v.mp4")
        proj.add_shot(0.0, 1.0)
        other = str(proj.root)
        other = other[:7] + other[7:].upper()  # /mnt/d/...: another spelling of the same folder
        assert Path(other).is_dir()
        a, b = Runner(proj.shot("001")), Runner(Project(Path(other)).shot("001"))
        with a.shot_lock(), pytest.raises(RunError, match="already being run"):
            with b.shot_lock():
                pass
        with b.shot_lock():  # free again
            pass
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_stale_lock_files_are_removed(tmp_path):
    from afv.core import runner as runner_mod

    ldir = tmp_path / "locks"
    ldir.mkdir()
    gone = ldir / "shot-1-2.lock"
    gone.write_text(str(tmp_path / "no-such-shot") + "\n")
    (tmp_path / "shot").mkdir()
    live = ldir / f"shot-{folder_identity(tmp_path / 'shot')}.lock"
    live.write_text(str(tmp_path / "shot") + "\n")
    unnamed = ldir / "shot-3-4.lock"
    unnamed.write_text("")
    runner_mod._LOCKS_CLEANED.clear()
    runner_mod._clean_stale_locks(ldir, keep=ldir / "shot-x.lock")
    assert not gone.exists() and live.exists() and unnamed.exists()  # unnamed: kept until a week old
