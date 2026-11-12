"""Windows-drive (drvfs/9p) behaviour: projects live on /mnt/d so Blender can open them."""

from __future__ import annotations

import glob
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from afv.core.project import Project
from afv.core.runner import Runner


def _drvfs_tmp() -> Path | None:
    cands = [os.environ.get("AFV_DRVFS_TMP", ""), *glob.glob("/mnt/c/Users/*/AppData/Local/Temp")]
    for c in cands:
        if c and os.path.isdir(c) and os.access(c, os.W_OK):
            return Path(c)
    return None


@pytest.mark.slow
def test_open_file_in_the_current_result_does_not_block_the_next_commit(fakes, do_pick):
    base = _drvfs_tmp()
    if base is None:
        pytest.skip("no writable Windows drive folder (set AFV_DRVFS_TMP)")
    root = Path(tempfile.mkdtemp(prefix="afv-drvfs-", dir=base))
    try:
        video = root / "clip.mp4"
        video.write_bytes(os.urandom(2048))
        shot = Project.create(root / "p.afv", video).add_shot(1.0, 2.0)
        do_pick(shot)
        assert Runner(shot).run().outcome == "ok"
        old = shot.stage_dir("camera")
        with open(old / "result.npz", "rb"):  # e.g. a viewer has a file of the result open
            try:  # what the old swap-based commit did; drvfs refuses it while the file is open
                os.rename(old, old.parent / "probe-rename")
                os.rename(old.parent / "probe-rename", old)
                renamed = True
            except PermissionError:
                renamed = False
            rep = Runner(shot, params={"camera": {"focal": 1234.0}}).run()
            assert rep.outcome == "ok", rep.message
            assert shot.meta("camera")["params"]["focal"] == 1234.0
            assert shot.stage_dir("camera").name == "v0002"
        rep = Runner(shot, params={"camera": {"focal": 1234.0}}).run()  # prunes what is left
        assert rep.outcome == "ok"
        assert [p.name for p in shot.stage_root("camera").iterdir()] == ["v0002"]
        print("rename of a folder with an open file:", "allowed" if renamed else "refused (EACCES)")
    finally:
        shutil.rmtree(root, ignore_errors=True)
