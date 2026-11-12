from __future__ import annotations

import errno
import os
from pathlib import Path

import numpy as np
import pytest

from afv.core import cache
from afv.core.types import CameraTrack, save_result


def test_canonical_json_is_order_and_type_insensitive():
    a = {"b": 1, "a": [1, 2, {"y": np.float32(0.5), "x": Path("/p")}], "t": (1, 2)}
    b = {"t": [1, 2], "a": [1, 2, {"x": "/p", "y": 0.5}], "b": 1}
    assert cache.hash_obj(a) == cache.hash_obj(b)
    assert cache.hash_obj({"v": float("nan")}) == cache.hash_obj({"v": float("nan")})
    assert cache.hash_obj({"v": 1}) != cache.hash_obj({"v": 2})


def test_result_hash_matches_stored_file_and_ignores_zip_timestamps(tmp_path):
    res = CameraTrack(frames=np.arange(5), timestamps=np.arange(5) / 30, focal_px=np.full(5, 1000.0))
    save_result(res, tmp_path / "a.npz")
    save_result(res, tmp_path / "b.npz")
    h = cache.result_hash(res)
    assert cache.npz_hash(tmp_path / "a.npz") == h == cache.npz_hash(tmp_path / "b.npz")
    res.focal_px = res.focal_px + 1
    assert cache.result_hash(res) != h


def test_stage_key_and_diff():
    p1 = {"stage": "smooth", "params": {"strength": 0.5}, "inputs": {"pose3d": "aa"}}
    p2 = {"stage": "smooth", "params": {"strength": 0.6}, "inputs": {"pose3d": "aa"}}
    assert cache.stage_key(p1) != cache.stage_key(p2)
    assert cache.stage_key(p1) == cache.stage_key(dict(reversed(list(p1.items()))))
    assert cache.diff_parts(p1, p2) == ["params: strength"]
    assert cache.diff_parts(None, p2) == ["no earlier result"]


def test_atomic_write_leaves_no_temp_files(tmp_path):
    p = tmp_path / "sub" / "x.json"
    cache.atomic_write_json(p, {"a": 1})
    cache.atomic_write_json(p, {"a": 2})
    assert cache.read_json(p) == {"a": 2}
    assert [f.name for f in p.parent.iterdir()] == ["x.json"]


def test_atomic_write_failure_keeps_old_file(tmp_path, monkeypatch):
    p = tmp_path / "x.txt"
    cache.atomic_write_text(p, "old")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        cache.atomic_write_text(p, "new")
    assert p.read_text() == "old"
    assert [f.name for f in tmp_path.iterdir()] == ["x.txt"]


def _work(tmp_path, name, text):
    w = tmp_path / name
    w.mkdir()
    (w / "meta.json").write_text(text)
    return w


def test_commit_version_and_current(tmp_path):
    root = tmp_path / "stages" / "camera"
    assert cache.current_version(root) is None
    v1 = cache.commit_version(_work(tmp_path, "w1", "one"), root)
    v2 = cache.commit_version(_work(tmp_path, "w2", "two"), root)
    assert (v1.name, v2.name) == ("v0001", "v0002")
    assert cache.current_version(root) == v2
    (root / "v0003").mkdir()  # an incomplete folder (no meta.json) is never current
    assert cache.current_version(root) == v2
    assert cache.prune_versions(root, v2) == []
    assert [p.name for _, p in cache.versions(root)] == ["v0002"]


def test_commit_version_retries_in_use_errors(tmp_path, monkeypatch):
    real, calls = os.rename, []

    def flaky(src, dst):
        calls.append(src)
        if len(calls) < 3:
            raise PermissionError(errno.EACCES, "Permission denied")
        return real(src, dst)

    monkeypatch.setattr(os, "rename", flaky)
    monkeypatch.setattr(cache.time, "sleep", lambda s: None)
    v = cache.commit_version(_work(tmp_path, "w", "x"), tmp_path / "root")
    assert len(calls) == 3 and (v / "meta.json").read_text() == "x"


def test_commit_version_names_the_likely_cause(tmp_path, monkeypatch):
    def denied(src, dst):
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(os, "rename", denied)
    monkeypatch.setattr(cache.time, "sleep", lambda s: None)
    w = _work(tmp_path, "w", "x")
    with pytest.raises(OSError, match="open in another program"):
        cache.commit_version(w, tmp_path / "root")
    assert w.is_dir()  # nothing lost


def test_prune_reports_versions_it_cannot_delete(tmp_path, monkeypatch):
    root = tmp_path / "root"
    v1 = cache.commit_version(_work(tmp_path, "w1", "1"), root)
    v2 = cache.commit_version(_work(tmp_path, "w2", "2"), root)
    monkeypatch.setattr(cache.shutil, "rmtree", lambda p, ignore_errors=False: None)  # "in use"
    assert cache.prune_versions(root, v2) == [v1]
