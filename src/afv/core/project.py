"""Project folder: source video, shots, stage results, edits and exports.

Layout (one project per source video)::

    <project>/
      project.json                 source path, sha256, probe info, title/url, settings
      media/                       made from the whole source (safe to delete; made again by afv shots ...)
        cuts.json                  every cut afv shots detect found (afv.media.shots.cuts_payload)
        scan/<id>.json, .jpg ...   candidate scans: numbered people + thumbnail per shot (afv shots scan)
        proxy.mp4, proxy.json      optional 720p short-GOP copy of the source (afv init --proxy)
      shots/<id>/
        shot.json                  start, duration, speed factor, notes, state (candidate | queued |
                                   skipped; missing = queued), settings
        stages/<stage>/v<NNNN>/    result.npz + meta.json + any files the stage wrote; the newest
                                   version with a meta.json is current (see Shot.stage_dir)
        edits/<name>.json          user edits (fighters.json, ids.json, contacts.json, pins.json)
        exports/                   published export files
        .work/                     in-progress and failed stage runs (safe to delete)
        .cache/track/              detect_track's 30 Hz frames, detections and `afv pick --frame`
                                   prompt images (safe to delete; rebuilt when the shot changes)
        .cache/camera/<key>/       the camera stage's worker outputs (normalised video, ViPE +
                                   GeoCalib, background flow tracks), content-keyed, so a re-run
                                   that changes only CPU-side params (e.g. the gravity re-level
                                   pass) skips the GPU solves (safe to delete; ``Ctx.force``
                                   recomputes; see ``afv.models.camera.WorkerCache``)

``settings`` (project and shot) is ``{"impl": {stage: implementation}, "params": {stage: {...}}}``;
shot settings override project settings.

Stages get only ``ctx.shot_dir``; they can call ``shot_context(ctx.shot_dir)`` for the source path
and shot interval, and ``stage_dir(ctx.shot_dir, "ingest")`` for another stage's files.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from afv.core.cache import atomic_write_json, current_version, hash_file, read_json, to_jsonable
from afv.core.types import Result, load_result

PROJECT_FILE = "project.json"
SHOT_FILE = "shot.json"
RESULT_FILE = "result.npz"
META_FILE = "meta.json"
QC_FILE = "qc.json"
PROJECT_VERSION = 1

_SHOT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_EDIT_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class ProjectError(RuntimeError):
    """Problem with the project folder, with a message fit to show the user."""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Settings(BaseModel):
    impl: dict[str, str] = Field(default_factory=dict)
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SourceInfo(BaseModel):
    path: str
    sha256: str
    size: int
    mtime_ns: int = 0  # modification time when hashed; size + mtime are checked before each run
    title: str = ""
    url: str = ""
    probe: dict[str, Any] | None = None


class ProjectInfo(BaseModel):
    version: int = PROJECT_VERSION
    created: str = Field(default_factory=now_iso)
    source: SourceInfo
    settings: Settings = Field(default_factory=Settings)


ShotState = Literal["candidate", "queued", "skipped"]
SHOT_STATES: tuple[str, ...] = ("candidate", "queued", "skipped")


class ShotInfo(BaseModel):
    id: str
    start: float  # seconds in the source
    duration: float  # seconds in the source
    # The footage's speed relative to real time: 0.25 = a 4x slow-motion replay, 1.25 = undercranked
    # (fast) footage. Export plays the motion back at real time (retime by 1 / speed_factor).
    speed_factor: float = 1.0
    notes: str = ""
    origin: str = "manual"  # "manual" or "detected"
    confidence: float | None = None  # cut-detection confidence for detected shots
    # candidate (found by `afv shots detect`, not chosen yet) | queued (runs in `afv batch`) | skipped.
    # Missing = queued: every shot made before round 6 keeps its behaviour. Changed only by the user.
    state: ShotState = "queued"
    created: str = Field(default_factory=now_iso)
    settings: Settings = Field(default_factory=Settings)

    @property
    def end(self) -> float:
        return self.start + self.duration


def _load_model(cls: type[BaseModel], path: Path) -> Any:
    return _read_model(cls, path)[1]


def _read_model(cls: type[BaseModel], path: Path) -> tuple[dict[str, Any], Any]:
    """(the file's JSON object, the validated model)."""
    try:
        raw = read_json(path)
        if not isinstance(raw, dict):
            raise ValueError(f"not a JSON object ({type(raw).__name__})")
        return raw, cls.model_validate(raw)
    except FileNotFoundError:
        raise ProjectError(f"missing {path}") from None
    except ValueError as e:  # JSONDecodeError and ValidationError are ValueErrors
        msg = e.errors()[0]["msg"] if isinstance(e, ValidationError) else str(e)
        raise ProjectError(f"{path} is invalid: {msg}") from None


@contextlib.contextmanager
def short_lock(kind: str, folder: Path, what: str, timeout: float = 30.0) -> Iterator[None]:
    """Exclusive ``flock`` for a short read-modify-write: ``<afv home>/locks/<kind>-<folder identity>.lock``
    (like the shot lock: the same for every spelling of the folder's path). Waits up to ``timeout``;
    without flock (not POSIX, or a filesystem without it) it does not lock."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    from afv.core.config import afv_home

    try:
        ldir = afv_home() / "locks"
        ldir.mkdir(parents=True, exist_ok=True)
        fd = os.open(ldir / f"{kind}-{folder_identity(folder)}.lock", os.O_RDWR | os.O_CREAT, 0o666)
    except OSError:
        yield
        return
    try:
        end = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > end:
                    raise ProjectError(f"{what} is being changed by another afv process; try again") from None
                time.sleep(0.01)
            except OSError:
                break  # no flock here: carry on unlocked
        yield
    finally:
        os.close(fd)  # releases the lock


class Project:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        if not (self.root / PROJECT_FILE).is_file():
            raise ProjectError(f"{self.root} is not an afv project (no {PROJECT_FILE}); run 'afv init' first")

    # -- creation / discovery ------------------------------------------------

    @classmethod
    def create(
        cls,
        root: Path,
        video: Path,
        *,
        title: str = "",
        url: str = "",
        probe: dict[str, Any] | None = None,
    ) -> Project:
        root, video = Path(root), Path(video)
        if not video.is_file():
            raise ProjectError(f"video not found: {video}")
        if (root / PROJECT_FILE).exists():
            raise ProjectError(f"{root} already has a {PROJECT_FILE}")
        root.mkdir(parents=True, exist_ok=True)
        (root / "shots").mkdir(exist_ok=True)
        src = _source_info(video, title=title, url=url, probe=probe)
        atomic_write_json(root / PROJECT_FILE, ProjectInfo(source=src).model_dump(mode="json"))
        return cls(root)

    def rehash(self, video: Path | None = None, *, probe: dict[str, Any] | None = None) -> SourceInfo:
        """Hash the source again (optionally a new path); every stage key changes if the content did."""
        info = self.info
        old = info.source
        video = Path(video) if video is not None else Path(old.path)
        if not video.is_file():
            raise ProjectError(f"video not found: {video}")
        info.source = _source_info(
            video, title=old.title, url=old.url, probe=probe if probe is not None else old.probe
        )
        self.save_info(info)
        return info.source

    @classmethod
    def find(cls, start: Path | None = None) -> Project:
        """The project containing ``start`` (default: cwd), searching parent folders."""
        here = Path(start or Path.cwd()).resolve()
        for d in (here, *here.parents):
            if (d / PROJECT_FILE).is_file():
                return cls(d)
        raise ProjectError(f"no afv project at or above {here}; pass --project DIR or set AFV_PROJECT")

    # -- project info ----------------------------------------------------------

    @property
    def info(self) -> ProjectInfo:
        return _load_model(ProjectInfo, self.root / PROJECT_FILE)

    def save_info(self, info: ProjectInfo) -> None:
        atomic_write_json(self.root / PROJECT_FILE, info.model_dump(mode="json"))

    def source_path(self, *, check: bool = True) -> Path:
        src = self.info.source
        p = Path(src.path)
        if check:
            self.check_source()
        return p

    def check_source(self) -> None:
        """Refuse to run on a source that is missing or was replaced since it was hashed."""
        src = self.info.source
        p = Path(src.path)
        hint = f"if this is the video you want, run: afv init {p} --project {self.root} --rehash"
        if not p.is_file():
            raise ProjectError(
                f"source video missing: {p} (moved? afv init <new path> --project {self.root} --rehash)"
            )
        st = p.stat()
        if st.st_size != src.size or (src.mtime_ns and st.st_mtime_ns != src.mtime_ns):
            raise ProjectError(f"source video changed since it was hashed: {p}; {hint}")

    # -- shots -----------------------------------------------------------------

    @property
    def shots_dir(self) -> Path:
        return self.root / "shots"

    def shot_ids(self) -> list[str]:
        if not self.shots_dir.is_dir():
            return []
        ids = [d.name for d in self.shots_dir.iterdir() if (d / SHOT_FILE).is_file()]
        return sorted(ids, key=shot_sort_key)

    def shots(self) -> list[Shot]:
        return [Shot(self, i) for i in self.shot_ids()]

    def resolve_shot_id(self, given: str) -> str:
        try:  # one listing and one test instead of testing every shot (slow on the Windows drives)
            if given in os.listdir(self.shots_dir) and (self.shots_dir / given / SHOT_FILE).is_file():
                return given
        except OSError:
            pass
        ids = self.shot_ids()
        if given in ids:
            return given
        if given.isdigit():
            hits = [i for i in ids if i.isdigit() and int(i) == int(given)]
            if len(hits) == 1:
                return hits[0]
        known = ", ".join(ids) if ids else "none yet (afv shots detect / afv shots add)"
        raise ProjectError(f"no shot {given!r}; shots: {known}")

    def shot(self, given: str) -> Shot:
        return Shot(self, self.resolve_shot_id(given))

    def next_shot_id(self) -> str:
        nums = [int(i) for i in self.shot_ids() if i.isdigit()]
        return f"{(max(nums) + 1) if nums else 1:03d}"

    @property
    def media_dir(self) -> Path:
        """``media/``: project-level files made from the source (``cuts.json``, ``scan/``, ``proxy.mp4``)."""
        return self.root / "media"

    def check_interval(self, start: float, duration: float, speed_factor: float = 1.0) -> None:
        """The checks ``add_shot`` and ``Shot.update`` apply to a shot's interval and speed."""
        values = (start, duration, speed_factor)
        if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
            raise ProjectError(
                f"shot times must be numbers: start {start!r}, duration {duration!r}, speed {speed_factor!r}"
            )
        if not all(math.isfinite(v) for v in (start, duration, speed_factor)):
            raise ProjectError(
                f"shot times must be finite numbers: start {start}, duration {duration}, speed {speed_factor}"
            )
        if start < 0 or duration <= 0:
            raise ProjectError(f"bad shot interval: start {start}, duration {duration}")
        if speed_factor <= 0:
            raise ProjectError(f"speed factor must be > 0, got {speed_factor}")
        probe = self.info.source.probe or {}
        total = probe.get("duration")
        if isinstance(total, int | float) and total > 0 and start >= total:
            raise ProjectError(f"shot starts at {start:.3f} s, after the end of the video ({total:.3f} s)")

    def add_shot(
        self,
        start: float,
        duration: float,
        *,
        shot_id: str | None = None,
        speed_factor: float = 1.0,
        notes: str = "",
        origin: str = "manual",
        confidence: float | None = None,
        state: str = "queued",
    ) -> Shot:
        self.check_interval(start, duration, speed_factor)
        check_state(state)
        if shot_id is not None and not _SHOT_ID.match(shot_id):
            raise ProjectError(f"bad shot id {shot_id!r} (letters, digits, _ . -)")
        self.shots_dir.mkdir(exist_ok=True)
        with short_lock("shots", self.shots_dir, "the shot list"):  # id choice + folder + shot.json
            return self._add_shot_locked(start, duration, shot_id, speed_factor, notes, origin, confidence,
                                         state)  # fmt: skip

    def _add_shot_locked(
        self, start: float, duration: float, shot_id: str | None, speed_factor: float, notes: str,
        origin: str, confidence: float | None, state: str,
    ) -> Shot:  # fmt: skip
        if shot_id is None:
            n = int(self.next_shot_id())
            while (self.shots_dir / f"{n:03d}").exists():  # a leftover folder without shot.json
                n += 1
            sid = f"{n:03d}"
            d = self.shots_dir / sid
            d.mkdir(exist_ok=False)  # a second process that got the same id fails here, not overwrites
        else:
            sid = shot_id
            d = self.shots_dir / sid
            if (d / SHOT_FILE).exists():
                raise ProjectError(f"shot {sid} already exists")
        info = ShotInfo(
            id=sid,
            start=float(start),
            duration=float(duration),
            speed_factor=float(speed_factor),
            notes=notes,
            origin=origin,
            confidence=confidence,
            state=state,  # type: ignore[arg-type]
        )
        for sub in ("stages", "edits", "exports"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        atomic_write_json(d / SHOT_FILE, info.model_dump(mode="json"))
        return Shot(self, sid)


class Shot:
    def __init__(self, project: Project, shot_id: str, *, check: bool = True):
        """``check=False`` skips testing that shot.json exists (reading ``info`` then fails clearly)."""
        self.project = project
        self.id = shot_id
        self.dir = project.shots_dir / shot_id
        if check and not (self.dir / SHOT_FILE).is_file():
            raise ProjectError(f"no shot {shot_id!r} in {project.root}")

    @property
    def info(self) -> ShotInfo:
        return _load_model(ShotInfo, self.dir / SHOT_FILE)

    def save_info(self, info: ShotInfo) -> None:
        atomic_write_json(self.dir / SHOT_FILE, info.model_dump(mode="json"))

    def update(
        self,
        *,
        start: float | None = None,
        duration: float | None = None,
        speed_factor: float | None = None,
        notes: str | None = None,
        state: str | None = None,
    ) -> tuple[ShotInfo, list[str]]:
        """Change the given fields (validated like ``Project.add_shot``) and save; returns the new info
        and the names of the fields that changed. Read, check and write happen under the shot's info lock
        (``short_lock``) and only the changed keys are written over the file as it is, so concurrent
        changes of other fields (a trim and a state change) both survive; unknown keys are kept."""
        if state is not None:
            check_state(state)
        if notes is not None and not isinstance(notes, str):
            raise ProjectError(f"notes must be text, got {notes!r}")
        path = self.dir / SHOT_FILE
        with short_lock("shotinfo", self.dir, f"shot {self.id}"):
            raw, old = _read_model(ShotInfo, path)
            new_start = old.start if start is None else start
            new_dur = old.duration if duration is None else duration
            new_speed = old.speed_factor if speed_factor is None else speed_factor
            self.project.check_interval(new_start, new_dur, new_speed)
            values = {
                "start": float(new_start), "duration": float(new_dur), "speed_factor": float(new_speed),
                "notes": old.notes if notes is None else notes,
                "state": old.state if state is None else state,
            }  # fmt: skip
            changed = [k for k, v in values.items() if getattr(old, k) != v]
            if changed:
                atomic_write_json(path, {**raw, **{k: values[k] for k in changed}})
        return self.info, changed

    def modify(self, fn: Callable[[ShotInfo], None]) -> ShotInfo:
        """Locked read-modify-write of shot.json: ``fn`` changes the fresh ShotInfo in place; the keys it
        changed are written over the file (as ``update``)."""
        path = self.dir / SHOT_FILE
        with short_lock("shotinfo", self.dir, f"shot {self.id}"):
            raw, info = _read_model(ShotInfo, path)
            before = info.model_dump(mode="json")
            fn(info)
            after = info.model_dump(mode="json")
            changed = {k: v for k, v in after.items() if before.get(k) != v}
            if changed:
                atomic_write_json(path, {**raw, **changed})
        return info

    # -- folders ---------------------------------------------------------------

    def stage_root(self, stage: str) -> Path:
        return self.dir / "stages" / stage

    def stage_dir(self, stage: str) -> Path:
        """Folder of the stage's current result (``stages/<stage>/v<NNNN>``); a non-existent
        ``stages/<stage>/none`` when there is none."""
        return stage_dir(self.dir, stage)

    @property
    def edits_dir(self) -> Path:
        return self.dir / "edits"

    @property
    def exports_dir(self) -> Path:
        return self.dir / "exports"

    @property
    def work_root(self) -> Path:
        return self.dir / ".work"

    # -- results -----------------------------------------------------------------

    def meta(self, stage: str) -> dict[str, Any] | None:
        p = self.stage_dir(stage) / META_FILE
        try:
            m = read_json(p)
        except (FileNotFoundError, ValueError):
            return None
        return m if isinstance(m, dict) else None

    def has_result(self, stage: str) -> bool:
        m = self.meta(stage)
        if m is None:
            return False
        return (not m.get("has_result", True)) or (self.stage_dir(stage) / RESULT_FILE).is_file()

    def load_result(self, stage: str) -> Result:
        p = self.stage_dir(stage) / RESULT_FILE
        if not p.is_file():
            raise ProjectError(f"shot {self.id}: stage {stage} has no stored result ({p})")
        try:
            return load_result(p)
        except Exception as e:
            raise ProjectError(f"shot {self.id}: cannot read {p}: {type(e).__name__}: {e}") from None

    # -- edits -----------------------------------------------------------------

    def edit_path(self, name: str) -> Path:
        if not _EDIT_NAME.match(name):
            raise ProjectError(f"bad edit name {name!r}")
        return self.edits_dir / f"{name}.json"

    def read_edit(self, name: str) -> Any | None:
        p = self.edit_path(name)
        if not p.is_file():
            return None
        try:
            return read_json(p)
        except ValueError as e:
            raise ProjectError(f"edit file {p} is not valid JSON: {e}") from None

    def write_edit(self, name: str, data: Any) -> Path:
        """Write ``edits/<name>.json`` atomically. contacts, pins, hands, ids and fighters are validated
        first (``afv.core.edits.validate_edit``, strict): an EditError leaves the file as it was. When
        the file on disk is invalid too (so the data, merged from it, carries its bad values), the
        message says to fix or delete it first."""
        from afv.core.edits import NAMES, EditError, fix_hint, validate_edit

        p = self.edit_path(name)
        if name in NAMES:
            try:
                validate_edit(name, data)
            except EditError as e:
                old = self.read_edit(name) if p.is_file() else None
                if old is not None:
                    try:
                        validate_edit(name, old)
                    except EditError:
                        hint = fix_hint(name, old)
                        raise EditError(f"{e} (the file on disk is invalid too: {hint})") from None
                raise
        atomic_write_json(p, data)
        return p


# ---------------------------------------------------------------------------
# Helpers for stages (which only get ctx.shot_dir)
# ---------------------------------------------------------------------------


class ShotContext(BaseModel):
    """What a stage needs to know about its shot and source."""

    shot_id: str
    shot_dir: str
    project_dir: str
    source_path: str
    source_sha256: str
    source_title: str = ""
    source_url: str = ""
    start: float
    duration: float
    speed_factor: float = 1.0
    probe: dict[str, Any] | None = None


def open_shot(shot_dir: Path) -> Shot:
    shot_dir = Path(shot_dir).resolve()
    return Shot(Project(shot_dir.parent.parent), shot_dir.name)


def shot_context(shot_dir: Path) -> ShotContext:
    shot = open_shot(shot_dir)
    pinfo, sinfo = shot.project.info, shot.info
    return ShotContext(
        shot_id=shot.id,
        shot_dir=str(shot.dir),
        project_dir=str(shot.project.root),
        source_path=pinfo.source.path,
        source_sha256=pinfo.source.sha256,
        source_title=pinfo.source.title,
        source_url=pinfo.source.url,
        start=sinfo.start,
        duration=sinfo.duration,
        speed_factor=sinfo.speed_factor,
        probe=pinfo.source.probe,
    )


def stage_dir(shot_dir: Path, stage: str) -> Path:
    """Folder of a stage's current committed result for this shot (non-existent when none)."""
    root = Path(shot_dir) / "stages" / stage
    return current_version(root, META_FILE) or root / "none"


def _source_info(video: Path, *, title: str, url: str, probe: dict[str, Any] | None) -> SourceInfo:
    st = video.stat()
    return SourceInfo(
        path=str(video.resolve()),
        sha256=hash_file(video),
        size=st.st_size,
        mtime_ns=st.st_mtime_ns,
        title=title,
        url=url,
        probe=to_jsonable(probe) if probe is not None else None,
    )


# ---------------------------------------------------------------------------
# Candidate scans: media/scan/<id>.json (written by afv.media.shots.scan_shots; format there)
# ---------------------------------------------------------------------------

SCAN_SUBDIR = "scan"
SCAN_FORMAT = "afv_scan"
SCAN_VERSION = 1


@dataclass(frozen=True)
class ScanPaths:
    json: Path
    thumb: Path
    frame: Path
    prompt: Path


def scan_dir(project: Project) -> Path:
    return project.media_dir / SCAN_SUBDIR


def scan_paths(project: Project, shot_id: str) -> ScanPaths:
    d = scan_dir(project)
    return ScanPaths(d / f"{shot_id}.json", d / f"{shot_id}.jpg", d / f"{shot_id}.frame.jpg",
                     d / f"{shot_id}.prompt.jpg")


def shot_hash(start: float, duration: float, source_sha256: str) -> str:
    """Identity of a shot's frames: its start, duration and source (a trimmed shot gets another hash)."""
    from afv.core.cache import hash_obj

    return hash_obj({"start": float(start), "duration": float(duration), "source_sha256": source_sha256})[:16]


def current_shot_hash(shot: Shot, source_sha256: str | None = None, info: Any = None) -> str:
    i = info or shot.info
    return shot_hash(i.start, i.duration, source_sha256 or shot.project.info.source.sha256)


DET_THRESHOLD_DEFAULT = 0.4  # afv.models.track.TrackParams.det_threshold (tests keep the two equal)


def scan_threshold(project_info: ProjectInfo, shot_info: ShotInfo) -> float:
    """The shot's ``detect_track.det_threshold`` as ``Runner.params_for`` resolves it (default < project
    settings < shot settings), without loading the stage modules: the threshold a scan must have been
    made at to number people as ``afv pick`` and detect_track do."""
    v = DET_THRESHOLD_DEFAULT
    for settings in (project_info.settings, shot_info.settings):
        layer = settings.params.get("detect_track") or {}
        x = layer.get("det_threshold") if isinstance(layer, dict) else None
        if isinstance(x, int | float) and not isinstance(x, bool):
            v = float(x)
    return v


def read_scan(
    shot: Shot, *, info: Any = None, listing: set[str] | None = None, project_info: ProjectInfo | None = None,
) -> tuple[str, dict[str, Any] | None]:  # fmt: skip
    """(status, scan): ``ok`` (made for the shot as it is now), ``stale`` (made for another interval or
    source, or at another detection threshold than the shot's ``detect_track.det_threshold`` now: the scan
    is returned but must not be used), ``none`` (no scan) or ``bad`` (unreadable). ``info`` (the shot's
    ShotInfo), ``project_info`` and ``listing`` (the file names in the scan folder) save file reads."""
    p = scan_paths(shot.project, shot.id).json
    if listing is not None and p.name not in listing:
        return "none", None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "none", None
    except (OSError, ValueError):
        return "bad", None
    if (not isinstance(data, dict) or data.get("format") != SCAN_FORMAT or data.get("version") != SCAN_VERSION
            or not isinstance(data.get("prompt"), dict)):
        return "bad", None
    pinfo = project_info or shot.project.info
    sinfo = info or shot.info
    if data.get("shot_hash") != current_shot_hash(shot, pinfo.source.sha256, sinfo):
        return "stale", data
    thr = data["prompt"].get("threshold")
    if not isinstance(thr, int | float) or abs(float(thr) - scan_threshold(pinfo, sinfo)) > 1e-9:
        return "stale", data
    return "ok", data


def load_scan(shot: Shot) -> dict[str, Any] | None:
    """The shot's scan when it is valid for the shot as it is now (``read_scan``), else None."""
    status, data = read_scan(shot)
    return data if status == "ok" else None


def shot_sort_key(shot_id: str) -> tuple[int, int, str]:
    """Numbered shots first, by number; then the others by name."""
    return (0, int(shot_id), shot_id) if shot_id.isdigit() else (1, 0, shot_id)


def check_state(state: Any) -> str:
    """``state`` when it is a shot state (candidate, queued, skipped), else a ProjectError."""
    if not isinstance(state, str) or state not in SHOT_STATES:
        raise ProjectError(f"bad shot state {state!r}; states: {', '.join(SHOT_STATES)}")
    return state


def folder_identity(path: Path) -> str:
    """One name for a folder however its path is spelled: ``<st_dev>-<st_ino>``. The Windows drives
    under WSL are case-insensitive (``/mnt/d/Fight.afv`` = ``/mnt/d/fight.afv``), and ``flock`` on two
    spellings of one file does not conflict there, so locks and queues key on this instead."""
    st = os.stat(path)
    return f"{st.st_dev:x}-{st.st_ino:x}"


def default_project_dir() -> Path | None:
    env = os.environ.get("AFV_PROJECT")
    return Path(env) if env else None
