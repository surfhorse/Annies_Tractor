"""Cache keys, content hashes and safe (atomic) writes.

Cache key
---------
A stage result is reused only when its key is unchanged. The key is the sha256 of the canonical JSON
of ``key_parts``::

    {"schema": KEY_SCHEMA, "stage", "impl", "version", "weight_hash", "params",
     "inputs": {upstream stage: upstream *output* hash},
     "edits":  {edit name: hash of its parsed JSON, or None when absent},
     "shot":   {source_sha256, start, duration, plus any shot fields the stage reads},
     "frame_range", "seed", "deterministic"}

Upstream results enter by their **output hash** (a hash of the stored arrays), not their key: a
forced or non-deterministic re-run of an upstream stage that changes its output invalidates
everything downstream, while one that reproduces the same arrays does not. Edits are hashed after
parsing, so reformatting a JSON file does not invalidate anything. The stage's source code is not in
the key: the runner stores a code record (``afv.core.codehash``) next to it and compares it apart,
so CPU and model stages can treat code changes differently (see ``afv.core.runner``).

Safe writes
-----------
Files are written to a temp name in the same folder, fsynced and ``os.replace``d. Stage results are
built in a work folder, fsynced, and renamed to a new ``stages/<stage>/v<NNNN>/``; the newest version
holding a ``meta.json`` is current. Old versions are deleted best-effort: an in-use folder on a
Windows drive (which refuses renames while a file inside is open) is never renamed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from afv.core.types import Result, result_to_arrays

KEY_SCHEMA = 1


# ---------------------------------------------------------------------------
# Canonical JSON and hashing
# ---------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays, Paths, tuples, sets and dataclasses to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set | frozenset):
        return sorted(to_jsonable(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if hasattr(obj, "model_dump"):
        return to_jsonable(obj.model_dump(mode="json"))
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, NaN/inf as strings (stable and valid)."""
    return json.dumps(_finite(to_jsonable(obj)), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _finite(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return repr(obj)
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


def hash_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def hash_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def hash_arrays(arrays: dict[str, np.ndarray]) -> str:
    """Hash of named arrays (names, dtypes, shapes, bytes): independent of NPZ/zip timestamps."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode("utf-8"))
        h.update(b"\0" + a.dtype.str.encode() + b"\0" + repr(a.shape).encode() + b"\0")
        h.update(a.tobytes())
    return h.hexdigest()


def result_hash(res: Result) -> str:
    """Content hash of a result exactly as it is stored (float32 etc.); equals ``npz_hash`` of its file."""
    return hash_arrays(result_to_arrays(res))


def npz_hash(path: Path) -> str:
    with np.load(path, allow_pickle=False) as z:
        return hash_arrays({k: z[k] for k in z.files})


def stage_key(parts: dict[str, Any]) -> str:
    return hash_obj({"schema": KEY_SCHEMA, **parts})


PART_LABELS = {"inputs": "upstream", "weight_hash": "weights"}  # key part -> word shown to the user


def diff_parts(old: dict[str, Any] | None, new: dict[str, Any]) -> list[str]:
    """Human-readable list of what changed between two key-part dicts (for 'why is this stale'):
    e.g. ``params: strength``, ``upstream: pose3d`` (an input's output changed), ``edits: fighters``,
    ``version 3 -> 4``."""
    if not old:
        return ["no earlier result"]
    out: list[str] = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if canonical_json(a) == canonical_json(b):
            continue
        label = PART_LABELS.get(k, k)
        if k == "version" and isinstance(a, str | int) and isinstance(b, str | int):
            out.append(f"version {a} -> {b}")
        elif isinstance(a, dict) and isinstance(b, dict):
            # a key present on one side only counts even when its value is None (a new param = None)
            sub = [
                s
                for s in sorted(set(a) | set(b))
                if (s in a) != (s in b) or canonical_json(a.get(s)) != canonical_json(b.get(s))
            ]
            out.append(f"{label}: {', '.join(sub)}")
        else:
            out.append(label)
    return out or ["unchanged"]


# ---------------------------------------------------------------------------
# Safe writes
# ---------------------------------------------------------------------------


def _fsync_dir(path: Path) -> None:
    with contextlib.suppress(OSError, AttributeError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _fsync_dir(path.parent)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False) + "\n")


def atomic_copy(src: Path, dst: Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent)
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Versioned stage folders
# ---------------------------------------------------------------------------

_VERSION = re.compile(r"^v(\d{4,})$")
RETRY_ERRNOS = (errno.EACCES, errno.EBUSY, errno.EPERM, errno.ETXTBSY)


def versions(root: Path) -> list[tuple[int, Path]]:
    """``v<NNNN>`` folders under ``root``, oldest first."""
    if not Path(root).is_dir():
        return []
    out = []
    for p in Path(root).iterdir():
        m = _VERSION.match(p.name)
        if m and p.is_dir():
            out.append((int(m.group(1)), p))
    return sorted(out)


def current_version(root: Path, marker: str = "meta.json") -> Path | None:
    """The newest complete version (one containing ``marker``), or None.

    A version folder only ever appears by renaming a finished work folder, so the newest one with
    its marker file is the current result; there is no pointer file to get out of step.
    """
    for _, p in reversed(versions(root)):
        if (p / marker).is_file():
            return p
    return None


def retry_os(fn: Any, what: str, *, attempts: int = 6, first_wait_s: float = 0.2) -> Any:
    """Call ``fn`` and retry on 'file in use' errors (Windows drives under WSL), with backoff."""
    wait = first_wait_s
    for i in range(attempts):
        try:
            return fn()
        except OSError as e:
            if e.errno not in RETRY_ERRNOS or i == attempts - 1:
                raise OSError(
                    e.errno,
                    f"{what}: {e.strerror or e} (is a file in it open in another program, e.g. a "
                    "video player, Blender or Explorer preview?)",
                ) from e
            time.sleep(wait)
            wait *= 2
    raise AssertionError("unreachable")


def commit_version(work: Path, root: Path) -> Path:
    """Move the finished folder ``work`` to the next ``root/v<NNNN>`` (same filesystem).

    The previous version is never renamed, so a file held open in it (on a Windows drive that
    blocks renames) cannot stop the commit; ``prune_versions`` deletes old versions best-effort.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    have = versions(root)
    dst = root / f"v{(have[-1][0] + 1) if have else 1:04d}"
    retry_os(lambda: os.rename(work, dst), f"cannot move {work} to {dst}")
    _fsync_dir(root)
    return dst


def prune_versions(root: Path, keep: Path | None) -> list[Path]:
    """Delete every version except ``keep``; returns the ones that could not be deleted (in use)."""
    left = []
    for _, p in versions(root):
        if keep is not None and p == keep:
            continue
        shutil.rmtree(p, ignore_errors=True)
        if p.exists():
            left.append(p)
    return left


def fsync_tree(path: Path) -> None:
    """fsync every file under ``path`` (before it is committed)."""
    for p in Path(path).rglob("*"):
        if p.is_file():
            with contextlib.suppress(OSError):
                fd = os.open(p, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
