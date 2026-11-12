"""Stage source hashing: a code change to a stage marks its stored results as stale.

A stage's code record (``stage_code(cls)``) covers, all found from the stage class:

- the module that defines the class and every afv module it imports (directly or through other afv
  modules, including imports inside functions). The ``__init__.py`` of an imported module's parent
  packages is hashed but not followed (importing ``afv.media.ffmpeg`` runs ``afv/media/__init__.py``
  but uses nothing it imports); a package is followed when a name is taken from it
  (``from afv.media import detect_shots``). ``PLUMBING`` modules (runner, stage, cache, project,
  config, edits, provenance, review, envs, cli, serve) are left out and not followed: by contract they
  do not change what a stage computes;
- the worker and Blender scripts the stage launches. Default rule: a ``"*.py"`` string constant
  (e.g. ``"gemx_worker.py"``) found in the module's folder, its ``workers/`` subfolder,
  ``afv/models/workers/`` or the repo's ``blender/`` folder, named in the stage's own module or in
  a helper module it imports. A string in *another stage module* (one that registers stages, e.g.
  ``camera.py`` imported by ``gemx.py``) belongs to that module's stages only, so pose3d does not
  hash ViPE's worker. A stage can instead declare ``worker_scripts`` (script names), which then
  replace the strings of its own module; ``DEFAULT_WORKER_SCRIPTS`` adds scripts for stages whose
  modules (owned elsewhere) do not declare them. Scripts' sibling scripts imported at module level
  (``import soma_worker``) count too; imports inside a worker's functions are mode-specific and do
  not;
- the module's ``data/`` folder when the module names ``"data"`` (skeleton and mapping tables);
- whatever the optional stage attribute ``code_deps`` lists (afv module names or file paths).

Python files are hashed as their normalised AST, so comments, docstrings (and any other bare
string statement), blank lines, quoting and line wrapping do not count; JSON files are hashed after
parsing. The AST layout belongs to the running Python version, so the record notes it (``python``)
and a different version is reported as such rather than as every file changing. The record is
``{"hash": h, "files": {label: h}, "python": "3.12"}``, where a label is a module name or a path
relative to the repo.

``snapshot_code(cls)`` is the record of the code this process runs: taken once per process and
stage class (the runner takes it when a run starts), with the afv modules it covers imported right
then, so a file saved during a long run or between the shots of a batch is not recorded as the
code that ran.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import importlib
import json
import re
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import afv

AFV_ROOT = Path(afv.__file__).resolve().parent  # src/afv
REPO_ROOT = AFV_ROOT.parent.parent
PLUMBING = frozenset(
    {
        "afv",
        "afv.cli",
        "afv.core",
        "afv.core.cache",
        "afv.core.codehash",
        "afv.core.config",
        "afv.core.edits",
        "afv.core.project",
        "afv.core.provenance",
        "afv.core.review",
        "afv.core.runner",
        "afv.core.stage",
        "afv.models.envs",
        "afv.serve",
    }
)
# Scripts a stage launches that the string rule may miss (e.g. soma_worker is imported inside
# gemx_worker's pose code): added to what the rule finds, unless the stage declares worker_scripts.
DEFAULT_WORKER_SCRIPTS: dict[str, tuple[str, ...]] = {"pose3d": ("gemx_worker.py", "soma_worker.py")}
_SCRIPT = re.compile(r"^[A-Za-z0-9_\-]+\.py$")
_DATA_SUFFIXES = (".json", ".toml", ".txt", ".csv", ".npz", ".npy", ".bvh")
PYTHON = f"{sys.version_info[0]}.{sys.version_info[1]}"


@dataclass(frozen=True)
class _Scan:
    hash: str
    imports: frozenset[str]  # every imported name (afv modules resolved, relative imports too)
    from_imports: frozenset[tuple[str, tuple[str, ...]]]  # (resolved base, names) of each "from" import
    top_imports: frozenset[str]  # plain ``import x`` at module level (sibling worker scripts)
    strings: frozenset[str]
    defines_stage: bool  # a class decorated with ``register(...)``


# (path, module, mtime_ns, size) -> scan: runs, plans and status parse the same files many times
_FILE_CACHE: dict[tuple[str, str | None, int, int], _Scan] = {}
_SNAPSHOT: dict[type, dict[str, Any] | None] = {}


_SCOPE = threading.local()  # per-thread stat memo of an open stat_scope()
_RECENT: dict[str, tuple[float, Any]] = {}  # path -> (monotonic time, stat) for set_stat_ttl()
_RECENT_LOCK = threading.Lock()
_STAT_TTL = [0.0]


def set_stat_ttl(seconds: float) -> None:
    """A long-lived process (``afv serve``) can reuse a file's stat for ``seconds`` across runs, so a
    request right after another (e.g. ``take`` after ``run``) does not stat every source file again.
    A file saved within that window is noticed by the next check after it (results still record the
    code that ran). 0 (the default) turns it off."""
    _STAT_TTL[0] = max(0.0, float(seconds))
    with _RECENT_LOCK:
        _RECENT.clear()


@contextlib.contextmanager
def stat_scope() -> Iterator[None]:
    """Inside: each file is stat'ed once (re-entrant, per thread). A run checks the code of every
    stage, and most stages share most modules: on the Windows drive each stat costs ~1 ms, so a run
    used to spend over a second on them. Files changed while the scope is open are seen by the next
    scope (a result still records the code that ran: the process snapshot)."""
    outer = getattr(_SCOPE, "stats", None)
    if outer is None:
        _SCOPE.stats = {}
    try:
        yield
    finally:
        if outer is None:
            _SCOPE.stats = None


def _stat(path: Path) -> Any:
    memo = getattr(_SCOPE, "stats", None)
    k = str(path)
    if memo is not None and (st := memo.get(k)) is not None:
        return st
    st = _recent_stat(path, k)
    if memo is not None:
        memo[k] = st
    return st


def cached_stat(path: Path) -> Any:
    """``path.stat()``, reused within the ``set_stat_ttl`` window (a plain stat when it is off)."""
    return _recent_stat(Path(path), str(path))


def _recent_stat(path: Path, k: str) -> Any:
    ttl = _STAT_TTL[0]
    if ttl <= 0:
        return path.stat()
    now = time.monotonic()
    with _RECENT_LOCK:
        hit = _RECENT.get(k)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    st = path.stat()
    with _RECENT_LOCK:
        _RECENT[k] = (now, st)
    return st


def clear_cache() -> None:
    """Forget file scans, layout lookups and the per-process snapshots (tests, or a new 'process')."""
    _FILE_CACHE.clear()
    _SNAPSHOT.clear()
    with _RECENT_LOCK:
        _RECENT.clear()
    afv_module_file.cache_clear()
    _label.cache_clear()
    _find_script.cache_clear()
    _data_files.cache_clear()


class _Strip(ast.NodeTransformer):
    """Drop bare string statements (docstrings, attribute docs): they never change behaviour (code
    reading ``__doc__`` is the exception, and does not exist in afv)."""

    def generic_visit(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        for name in ("body", "orelse", "finalbody"):
            body = getattr(node, name, None)
            if not isinstance(body, list) or not body or not isinstance(body[0], ast.stmt):
                continue
            kept = [
                s for s in body
                if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                        and isinstance(s.value.value, str))
            ]  # fmt: skip
            # a block must keep a statement; a module may be empty (docstring-only == empty)
            setattr(node, name, kept or ([] if isinstance(node, ast.Module) else [ast.Pass()]))
        return node


def normalised_source_hash(text: str) -> str:
    """sha256 of the AST of ``text`` without docstrings/comments/formatting."""
    tree = _Strip().visit(ast.parse(text))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()


def json_hash(raw: bytes) -> str:
    """sha256 of parsed JSON (key order and whitespace do not count); raw bytes if it does not parse."""
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return hashlib.sha256(raw).hexdigest()
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _is_register(dec: ast.expr) -> bool:
    f = dec.func if isinstance(dec, ast.Call) else dec
    return (isinstance(f, ast.Name) and f.id == "register") or (
        isinstance(f, ast.Attribute) and f.attr == "register"
    )


def _scan(path: Path, module: str | None) -> _Scan:
    st = _stat(path)
    ck = (str(path), module, st.st_mtime_ns, st.st_size)
    hit = _FILE_CACHE.get(ck)
    if hit is not None:
        return hit
    raw = path.read_bytes()
    imports: set[str] = set()
    froms: set[tuple[str, tuple[str, ...]]] = set()
    top: set[str] = set()
    strings: set[str] = set()
    stage = False
    if path.suffix == ".py":
        try:
            text = raw.decode("utf-8")
            tree: ast.Module | None = ast.parse(text)
            h = normalised_source_hash(text)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            h, tree = hashlib.sha256(raw).hexdigest(), None
        if tree is not None:
            pkg = _package_of(module, path)
            top = {a.name for s in tree.body if isinstance(s, ast.Import) for a in s.names}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    base = _resolve_from(node, pkg)
                    if base:
                        froms.add((base, tuple(a.name for a in node.names)))
                        imports.add(base)
                        imports.update(f"{base}.{a.name}" for a in node.names if a.name != "*")
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 200:
                    strings.add(node.value)
                elif isinstance(node, ast.ClassDef) and any(_is_register(d) for d in node.decorator_list):
                    stage = True
    elif path.suffix == ".json":
        h = json_hash(raw)
    else:
        h = hashlib.sha256(raw).hexdigest()
    out = _Scan(h, frozenset(imports), frozenset(froms), frozenset(top), frozenset(strings), stage)
    _FILE_CACHE[ck] = out
    return out


def _package_of(module: str | None, path: Path) -> str | None:
    if not module:
        return None
    return module if path.name == "__init__.py" else module.rpartition(".")[0] or None


def _resolve_from(node: ast.ImportFrom, pkg: str | None) -> str | None:
    """Absolute module of ``from X import ...`` (relative imports resolved against ``pkg``)."""
    if node.level == 0:
        return node.module
    if pkg is None:
        return None
    parts = pkg.split(".")
    if node.level - 1 > len(parts) - 1:
        return None
    base = ".".join(parts[: len(parts) - (node.level - 1)])
    return f"{base}.{node.module}" if node.module else base


@functools.lru_cache(maxsize=4096)
def afv_module_file(name: str) -> Path | None:
    """Source file of an afv module (without importing it), or None if ``name`` is not a module."""
    parts = name.split(".")
    if parts[0] != "afv" or len(parts) < 2:
        return (AFV_ROOT / "__init__.py") if name == "afv" else None
    p = AFV_ROOT.joinpath(*parts[1:])
    if (f := p.with_name(p.name + ".py")).is_file():
        return f
    if (f := p / "__init__.py").is_file():
        return f
    return None


@functools.lru_cache(maxsize=4096)
def _label(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


# Folder layout lookups (not file contents) are memoised for the process: they do not change while
# afv runs, and stat calls are slow on the Windows drive.
@functools.lru_cache(maxsize=4096)
def _find_script(here: Path, name: str, local: bool = False) -> Path | None:
    shared = (AFV_ROOT / "models" / "workers", REPO_ROOT / "blender")
    dirs = (here,) if local else (here, here / "workers", *shared)
    for d in dirs:
        if (d / name).is_file():
            return d / name
    return None


@functools.lru_cache(maxsize=256)
def _data_files(here: Path) -> tuple[Path, ...]:
    d = here / "data"
    return tuple(sorted(p for p in d.rglob("*") if p.suffix in _DATA_SUFFIXES)) if d.is_dir() else ()


def _parents(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(2, len(parts))]


def stage_code(cls: type) -> dict[str, Any] | None:
    """Code record ``{"hash", "files", "python"}`` of a stage class as it is on disk now, or None
    when its source cannot be found."""
    mod = sys.modules.get(cls.__module__)
    src = getattr(mod, "__file__", None)
    if not src or not Path(src).is_file():
        return None
    own = cls.__module__
    declared = getattr(cls, "worker_scripts", None)
    files: dict[str, str] = {}
    followed: set[str] = set()
    queue: list[tuple[str, Path, bool]] = [(own, Path(src), True)]

    def add(name: str, follow: bool) -> None:
        """Queue an afv module (followed, or only hashed) and hash its parent packages."""
        if name in PLUMBING or (f := afv_module_file(name)) is None:
            return
        queue.append((name, f, follow))
        queue.extend((p, pf, False) for p in _parents(name) if (pf := afv_module_file(p)) is not None)

    for p in _parents(own):
        add(p, False)
    scripts: list[Path] = []
    for dep in getattr(cls, "code_deps", ()) or ():
        dep = str(dep)
        if dep.startswith("afv.") and afv_module_file(dep) is not None:
            add(dep, True)
        else:
            scripts.append(Path(dep) if Path(dep).is_absolute() else REPO_ROOT / dep)
    default = DEFAULT_WORKER_SCRIPTS.get(str(getattr(cls, "name", "")), ())
    wanted = declared if declared is not None else default
    for s in wanted:
        f = _find_script(Path(src).parent, str(s))
        scripts.append(f if f is not None else REPO_ROOT / str(s))
    while queue:
        name, path, follow = queue.pop()
        if name in PLUMBING:
            continue
        sc = _scan(path, name)
        files[name] = sc.hash
        if not follow or name in followed:
            continue
        followed.add(name)
        plain = (
            sc.imports
            - {b for b, _ in sc.from_imports}
            - {f"{b}.{n}" for b, names in sc.from_imports for n in names}
        )
        for imp in plain:
            if imp.startswith("afv."):
                add(imp, True)
        for base, names in sc.from_imports:
            if not base.startswith("afv"):
                continue
            for n in names:
                sub = f"{base}.{n}"
                if n != "*" and afv_module_file(sub) is not None:
                    add(sub, True)  # a submodule: its package is only hashed
                else:
                    add(base, True)  # a name defined in the package itself
        here = path.parent
        # whose worker scripts: the stage's own module (unless declared), or helper modules
        take_scripts = (declared is None) if name == own else not sc.defines_stage
        for s in sc.strings:
            if take_scripts and _SCRIPT.match(s):
                if (f := _find_script(here, s)) is not None:
                    scripts.append(f)
            elif s == "data":
                scripts += _data_files(here)
    while scripts:  # scripts run in other environments, and the sibling scripts they import
        p = scripts.pop()
        label = _label(p)
        if label in files:
            continue
        try:
            sc = _scan(p, None)
        except FileNotFoundError:
            files[label] = "missing"
            continue
        files[label] = sc.hash
        if p.suffix == ".py":
            siblings = (_find_script(p.parent, f"{i}.py", local=True) for i in sc.top_imports if "." not in i)
            scripts += [f for f in siblings if f is not None]
    total = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"hash": total, "files": dict(sorted(files.items())), "python": PYTHON}


def script_hash(path: Path) -> tuple[str, str]:
    """(label, hash) of one file exactly as a code record lists it (e.g. a worker script), so a stage
    can check whether a stored result was made by the same worker."""
    return _label(Path(path)), _scan(Path(path), None).hash


def snapshot_code(cls: type) -> dict[str, Any] | None:
    """The record of the code this process runs for ``cls``: hashed once per process, and the afv
    modules it covers imported at once so later disk edits cannot slip into this process."""
    if cls in _SNAPSHOT:
        return _SNAPSHOT[cls]
    rec = stage_code(cls)
    if rec is not None:
        for label in rec["files"]:
            if label.startswith("afv.") and label not in sys.modules:
                with contextlib.suppress(Exception):  # a broken helper fails when the stage uses it
                    importlib.import_module(label)
    _SNAPSHOT[cls] = rec
    return rec


def changed_files(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Labels whose hash differs (or that appear on one side only) between two code records."""
    a, b = old.get("files") or {}, new.get("files") or {}
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
