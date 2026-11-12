"""Read an afv take package (``afv take`` writes it; see ``src/afv/take.py``).

numpy only (no bpy, no afv), so it also runs in plain Python for tests.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TAKE_FORMAT = "afv_take"
SUPPORTED_VERSIONS = (1,)
TAKE_DIR = "take"
TAKE_FILE = "take.json"
FIGHTER_ARRAYS = ("rest_head", "rest_tail", "rest_zaxis", "pose_head", "pose_delta")


class TakeError(Exception):
    """The take package is missing or unreadable; the message is fit to show the user."""


@dataclass
class Fighter:
    role: str
    name: str
    bones: list[tuple[str, str | None]]  # (name, parent), parents before children
    translating: list[str]
    arrays: dict[str, np.ndarray]
    color: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)
    plant_channels: list[str] = field(default_factory=list)
    contact_channels: list[str] = field(default_factory=list)
    hips: str = ""

    @property
    def bone_names(self) -> list[str]:
        return [b for b, _ in self.bones]


@dataclass
class TakeData:
    path: Path  # take.json
    meta: dict
    fighters: list[Fighter]
    frames: np.ndarray  # [F] int scene frames (the shot timeline)
    timestamps: np.ndarray  # [F] source seconds
    camera: dict | None = None  # take.json "camera" + arrays pos/rot/focal_px
    spans: list[dict] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return self.path.parent

    @property
    def take_id(self) -> str:
        return str(self.meta.get("take_id", ""))

    @property
    def shot_id(self) -> str:
        return str(self.meta["shot"]["id"])

    @property
    def fps(self) -> float:
        return float(self.meta["shot"]["fps"])

    @property
    def frame_start(self) -> int:
        return int(self.frames[0]) if self.frames.size else 0

    @property
    def frame_end(self) -> int:
        return int(self.frames[-1]) if self.frames.size else 0

    def video_path(self) -> str | None:
        """The proxy video: relative to the take folder first (survives a moved project), then absolute."""
        v = self.meta.get("video") or {}
        rel = v.get("path_rel") or ""
        cands = [self.dir / rel] if rel else []
        if v.get("path_win"):
            cands.append(Path(v["path_win"]))
        for p in cands:
            try:  # e.g. \\wsl.localhost\<distro>\mnt\... is refused by Windows
                if p.is_file():
                    return str(p.resolve())
            except OSError:
                continue
        return None

    def presets(self) -> list[dict]:
        return list(self.meta.get("presets") or [])

    def record(self) -> dict:
        """What a scene remembers about the take it holds (add-on Re-import / Export). Folders are
        not stored: they are derived from the take's location (``locate``) when needed."""
        return {
            "take_id": self.take_id,
            "take_json": str(self.path),
            "shot_id": self.shot_id,
            "fps": self.fps,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "preset": (self.meta.get("rig") or {}).get("preset", ""),
            "prefix": (self.meta.get("rig") or {}).get("prefix", "") or "",
            "root_motion": (self.meta.get("rig") or {}).get("root_motion", "") or "",
            "presets": self.presets(),
            "roles": [f.role for f in self.fighters],
        }


_JSON_CACHE: dict[str, tuple[tuple[int, int], object]] = {}


def read_json_cached(path: str | Path):
    """A JSON file, parsed once per modification (the sidebar and the overlays read take.json and
    stage meta.json on every redraw). Raises OSError / ValueError like a plain read."""
    key = str(path)
    st = Path(path).stat()
    mtime = (st.st_mtime_ns, st.st_size)
    hit = _JSON_CACHE.get(key)
    if hit is None or hit[0] != mtime:
        hit = (mtime, json.loads(Path(path).read_text(encoding="utf-8")))
        if len(_JSON_CACHE) > 64:
            _JSON_CACHE.clear()
        _JSON_CACHE[key] = hit
    return hit[1]


def locate(take_json: str | Path, meta: dict | None = None) -> dict:
    """Shot and project folders of a take, from where the take is now (``shot_rel`` in take.json,
    else the folders above it); ``project_wsl`` is take.json's WSL path, a fallback when the project
    folder cannot be found."""
    jp = Path(take_json)
    if meta is None:
        try:
            meta = read_json_cached(jp)
        except (OSError, ValueError):
            meta = {}
    shot = None
    rel = meta.get("shot_rel") or ""
    if rel:
        cand = jp.parent / rel
        if (cand / "shot.json").is_file():
            shot = cand.resolve()
    if shot is None:
        shot = shot_dir_of(jp)
    return {
        "shot_dir": shot,
        "project_dir": shot.parent.parent if shot is not None else None,
        "shot_id": (meta.get("shot") or {}).get("id") or (shot.name if shot is not None else ""),
        "project_wsl": meta.get("project_dir", ""),
    }


def _current_meta(stage_root: Path) -> tuple[str, dict] | None:
    """(version folder, meta.json) of a stage's current result: the newest v#### with a meta.json."""
    try:
        dirs = [d for d in stage_root.iterdir() if d.is_dir() and re.fullmatch(r"v\d+", d.name)]
    except OSError:
        return None
    for d in sorted(dirs, key=lambda d: int(d.name[1:]), reverse=True):
        try:
            return d.name, read_json_cached(d / "meta.json")
        except (OSError, ValueError):
            continue
    return None


def ids_match(recorded: str, package: str) -> bool:
    """Is ``package`` the take a scene recorded as ``recorded``? Records made before take ids carried
    the project uid (``<project>:<shot>``) match the same take with one (``<project>:<shot>:<uid>``)."""
    if not recorded or not package:
        return False
    if recorded == package:
        return True
    return recorded.count(":") == 1 and package.count(":") == 2 and package.startswith(recorded + ":")


def stale_summary(take: TakeData) -> str:
    """One short, actionable line about results ``afv run`` would redo ("" when none)."""
    s = take.meta.get("stale_summary") or ""
    if s or not take.meta.get("stale"):
        return s
    return f"some results are older than the current settings or code: run `afv run {take.shot_id}`"


def pipeline_warnings(take: TakeData) -> list[str]:
    """What ``afv take`` flagged when it wrote the package: results that ``afv run`` would redo."""
    return list(take.meta.get("stale") or [])


def stale_sources(take: TakeData) -> list[str]:
    """Why the package is older than the shot's results: a stage it was made from has a newer
    result now (``afv take`` fixes that; ``pipeline_warnings`` are for ``afv run``)."""
    out = []
    shot = locate(take.path, take.meta)["shot_dir"]
    if shot is None:
        return out
    for key, src in (take.meta.get("sources") or {}).items():
        stage = src.get("stage", key)
        cur = _current_meta(shot / "stages" / stage)
        if cur is not None and src.get("output_hash") and cur[1].get("output_hash") != src["output_hash"]:
            out.append(f"{stage} has a newer result ({cur[0]}) than the take ({src.get('version_dir')})")
    return out


def find_take_json(path: str | Path) -> Path:
    """take.json from a take.json path, a take folder or a shot folder."""
    p = Path(path)
    if p.is_file():
        return p
    for cand in (p / TAKE_FILE, p / TAKE_DIR / TAKE_FILE):
        if cand.is_file():
            return cand
    raise TakeError(f"no take in {p}: run 'afv take <shot>' first (it writes <shot>/{TAKE_DIR}/{TAKE_FILE})")


def shot_dir_of(path: str | Path) -> Path | None:
    """The shot folder (the one holding shot.json) for a shot, take folder or take.json path."""
    p = Path(path)
    for cand in (p, p.parent, p.parent.parent):
        if (cand / "shot.json").is_file():
            return cand
    return None


def _npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError) as e:
        raise TakeError(f"cannot read {path}: {e}") from e


def load_take(path: str | Path) -> TakeData:
    jp = find_take_json(path)
    try:
        meta = json.loads(jp.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise TakeError(f"cannot read {jp}: {e}") from e
    if meta.get("format") != TAKE_FORMAT:
        raise TakeError(f"{jp} is not an afv take")
    if meta.get("version") not in SUPPORTED_VERSIONS:
        raise TakeError(f"{jp}: take version {meta.get('version')} not supported by this add-on "
                        f"(supports {SUPPORTED_VERSIONS}); update the add-on")
    base = jp.parent
    fighters = []
    for f in meta["fighters"]:
        arrays = _npz(base / f["npz"])
        missing = [k for k in FIGHTER_ARRAYS if k not in arrays]
        if missing:
            raise TakeError(f"{f['npz']}: missing arrays {missing}")
        bones = [(b["name"], b["parent"]) for b in f["bones"]]
        if arrays["rest_head"].shape[0] != len(bones):
            n = arrays["rest_head"].shape[0]
            raise TakeError(f"{f['npz']}: {n} bones in the arrays, {len(bones)} listed")
        fighters.append(Fighter(
            role=f["role"], name=f["name"], bones=bones, translating=list(f.get("translating", [])),
            arrays=arrays, color=tuple(f.get("color") or (0.6, 0.6, 0.6, 1.0)),
            plant_channels=list(f.get("plant_channels", [])),
            contact_channels=list(f.get("contact_channels", [])), hips=f.get("hips", ""),
        ))
    tl = _npz(base / meta.get("timeline", "timeline.npz"))
    frames = np.asarray(tl["frames"], dtype=np.int64)
    for f in fighters:
        if f.arrays["pose_head"].shape[0] != frames.size:
            n = f.arrays["pose_head"].shape[0]
            raise TakeError(f"fighter {f.role}: {n} frames, the timeline has {frames.size}")
    camera = None
    if meta.get("camera"):
        camera = dict(meta["camera"])
        camera.update(_npz(base / camera["npz"]))
        if camera["pos"].shape[0] != frames.size:
            raise TakeError(f"camera has {camera['pos'].shape[0]} frames, timeline has {frames.size}")
    return TakeData(
        path=jp, meta=meta, fighters=fighters, frames=frames,
        timestamps=np.asarray(tl.get("timestamps", np.zeros(frames.size)), dtype=np.float64),
        camera=camera, spans=list((meta.get("qc") or {}).get("spans", [])),
    )


# --- issue navigation (pure; the add-on's Next/Previous issue) -----------------------------------
def next_span(spans: list[dict], frame: int, direction: int = 1) -> dict | None:
    """The first span starting after (direction 1) / before (-1) ``frame``; wraps around."""
    if not spans:
        return None
    starts = sorted(spans, key=lambda s: (s["start"], s.get("role", ""), s.get("kind", "")))
    if direction >= 0:
        later = [s for s in starts if s["start"] > frame]
        return later[0] if later else starts[0]
    earlier = [s for s in starts if s["start"] < frame]
    return earlier[-1] if earlier else starts[-1]


def spans_at(spans: list[dict], frame: int) -> list[dict]:
    return [s for s in spans if s["start"] <= frame <= s["end"]]


# --- detect_track boxes (the box overlay; pure) ----------------------------------------------------
@dataclass
class TrackData:
    ids: list[int]
    boxes: np.ndarray  # [F,T,4] x0, y0, x1, y1, camera image pixels (y down); NaN = absent
    fighters: dict[str, int]  # role -> object id
    labels: dict[int, str]
    image_size: tuple[int, int]
    frames: np.ndarray  # [F] scene frames

    def role_of(self, obj: int) -> str | None:
        return next((r for r, o in self.fighters.items() if o == obj), None)

    def boxes_at(self, frame: int) -> list[tuple[int, tuple[float, float, float, float]]]:
        """(object id, box) of the objects present at scene ``frame``."""
        if not self.frames.size:
            return []
        i = int(np.searchsorted(self.frames, frame))
        if i >= self.frames.size or self.frames[i] != frame:
            return []
        out = []
        for t, obj in enumerate(self.ids):
            b = self.boxes[i, t]
            if np.all(np.isfinite(b)):
                out.append((obj, tuple(float(v) for v in b)))
        return out


def load_tracks(take_json: str | Path) -> TrackData | None:
    """The take's tracking boxes (None for a take without them, e.g. written before round 4)."""
    jp = find_take_json(take_json)
    try:
        meta = json.loads(jp.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise TakeError(f"cannot read {jp}: {e}") from e
    tr = meta.get("tracks")
    if not tr:
        return None
    z = _npz(jp.parent / tr["npz"])
    tl = _npz(jp.parent / meta.get("timeline", "timeline.npz"))
    boxes = np.asarray(z["boxes"], dtype=np.float64)
    ids = [int(i) for i in tr.get("ids", z.get("ids", []))]
    if boxes.ndim != 3 or boxes.shape[1] != len(ids):
        raise TakeError(f"{tr['npz']}: boxes {boxes.shape} do not match {len(ids)} objects")
    fighters = {str(k): int(v) for k, v in (tr.get("fighters") or {}).items()}
    return TrackData(ids=ids, boxes=boxes, fighters=fighters,
                     labels={int(k): str(v) for k, v in (tr.get("labels") or {}).items()},
                     image_size=tuple(int(v) for v in tr.get("image_size", (0, 0))),
                     frames=np.asarray(tl["frames"], dtype=np.int64))


def image_to_region(corners, size, u: float, v: float) -> tuple[float, float]:
    """Image pixel (u right, v down) -> region pixel, given the camera frame's region corners
    ``corners`` = (top-left, top-right, bottom-left) and the image ``size`` (w, h)."""
    (tlx, tly), (trx, try_), (blx, bly) = corners
    w, h = size
    a, b = u / w, v / h
    return (tlx + a * (trx - tlx) + b * (blx - tlx), tly + a * (try_ - tly) + b * (bly - tly))


def inside_frame(corners, x: float, y: float) -> bool:
    """Is region point (x, y) inside the camera frame given by ``corners`` (see ``image_to_region``)?"""
    (tlx, tly), (trx, try_), (blx, bly) = corners
    ux, uy = trx - tlx, try_ - tly
    vx, vy = blx - tlx, bly - tly
    px, py = x - tlx, y - tly
    a = (px * ux + py * uy) / max(ux * ux + uy * uy, 1e-12)
    b = (px * vx + py * vy) / max(vx * vx + vy * vy, 1e-12)
    return 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0


def box_at(boxes, x: float, y: float) -> int | None:
    """The object whose region box (``[(obj, (x0, y0, x1, y1))]``, any corner order) contains the
    point; the smallest one when boxes overlap (the nearer, partly hidden fighter)."""
    best, best_area = None, None
    for obj, (x0, y0, x1, y1) in boxes:
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        if lo_x <= x <= hi_x and lo_y <= y <= hi_y:
            area = (hi_x - lo_x) * (hi_y - lo_y)
            if best_area is None or area < best_area:
                best, best_area = obj, area
    return best


# --- confidence strip (pure) -----------------------------------------------------------------------
SEVERITY_COLOR = {"warn": (1.0, 0.68, 0.1, 0.85), "bad": (0.95, 0.15, 0.12, 0.95)}


def strip_bars(spans: list[dict], roles: list[str]) -> list[tuple[int, float, float, str]]:
    """Bars of the confidence strip: (row, frame from, frame to, severity), one row per fighter
    (in ``roles`` order); a span covers its frames start..end inclusive, so the bar ends at end + 1.
    Bad spans come last so they are drawn on top."""
    rows = {r: i for i, r in enumerate(roles)}
    out = []
    for s in spans:
        row = rows.get(s.get("role"), len(roles))
        out.append((row, float(s["start"]), float(s["end"]) + 1.0, s.get("severity", "warn")))
    out.sort(key=lambda b: (b[3] == "bad", b[1]))
    return out
