"""Shot list from detected cuts, the project's ``media/cuts.json`` and the candidate scan.

Shot intervals are chosen so GEM-X's normalisation of each shot contains only that shot's frames:

- a shot starts halfway between its first source frame and the previous one (ffmpeg's input
  ``-ss`` drops every frame before it, so the last frame of the previous shot is excluded);
- a shot ends 1/60 s before the first frame of the next shot (or of a gradual transition). With
  ``fps=30`` rounding to the nearest tick, the last kept 30 Hz tick then always shows a frame of
  this shot, also for 50/60 fps sources where a plain midpoint could leak one frame.
- the last shot ends at GEM-X's source duration (OpenCV frame count / fps); shots longer than
  GEM-X's 60 s limit are split into equal parts of a whole number of 1/30 s ticks.

``media/cuts.json`` (``cuts_payload``, written by ``afv shots detect``, read by the shot browser)::

    {"format": "afv_cuts", "version": 1, "source_sha256", "fps", "params": {CutParams...,
     "min_confidence", "include_gradual"}, "cuts": [{"frame", "time", "confidence", "kind",
     "start_frame", "start_time"}]}

``cuts`` are every cut the detector found (also those below ``params.min_confidence``, which made no
shot boundary); times are seek times in source seconds, frames source frame indices; ``start_*`` mark
the first frame of a gradual transition (equal to ``frame``/``time`` for a hard cut).

Candidate scan (``scan_shots``; ``afv shots scan``)
---------------------------------------------------
One numbered-people prompt per shot, so the fighters can be picked without starting a detector. Per
shot ``<id>``, in ``media/scan/``:

- ``<id>.json``: ``{"format": "afv_scan", "version": 1, "shot": id, "shot_hash", "frame", "prompt",
  "people", "fighter_sized", "fighters_likely", "size": [w, h], "scanned"}``. ``prompt`` is exactly what
  ``afv.models.track.detect_prompt`` returns for that shot and frame (``frame``/``prompt_frame`` = the
  shot-timeline frame, ``boxes`` largest first, ``scores``, ``threshold`` = the shot's
  ``detect_track.det_threshold``), with ``image`` = ``<id>.prompt.jpg``. ``shot_hash`` hashes the
  shot's start, duration and source; a scan whose hash differs from the shot's (it was trimmed) is
  stale and ignored.
- ``<id>.jpg``: thumbnail of the scanned frame, at most ``THUMB_WIDTH`` px wide.
- ``<id>.frame.jpg``: the scanned frame itself, byte-identical to ``prepare_frames``' ``<frame>.jpg``.
- ``<id>.prompt.jpg``: the frame with the numbered boxes, written on demand (``prompt_image``).

How the scan gets exactly ``detect_prompt``'s prompt without preparing every shot's frames: the frame
is made by the same two ffmpeg steps as ``afv.media.normalise_shot`` (GEM-X's x264 command, then
JPEG ``-q:v 2``), but the first step encodes only ``scan frame + ENCODE_MARGIN + 1`` frames: x264's
decisions for a frame depend on at most its look-ahead (40 frames at its default preset) of later
frames, so the frame's JPEG is byte-identical (measured: identical from a margin of 60 frames on;
the tests check it). All the shots' frames then go to one ``detect`` request of the track worker
(``detect_prompt`` over a folder holding one frame per shot): one process, RF-DETR loaded once, one
image per call as for a single prompt, under the GPU lock.

``fighters_likely`` (``fighters_likely``): the frame shows two to ``LIKELY_MAX_PEOPLE`` "fighter-sized"
people, the second tallest at least ``LIKELY_PAIR_RATIO`` of the tallest one's height (two fighters at a
similar distance, maybe a referee). Fighter-sized: score >= ``LIKELY_MIN_SCORE``, box at least
``LIKELY_MIN_HEIGHT`` of the frame height tall, its top at least ``LIKELY_MIN_TOP`` of the frame height
below the frame's top edge (the head is in the picture: close-ups cut it off) and at least
``LIKELY_MIN_ASPECT`` times as tall as wide (a body, not a face filling the frame). Tuned on the 126
candidates of a 16-minute boxing broadcast (pre-fight, fight, post-fight): 47 marked, 32 of them fight
action; 9 of its 41 fight-action shots missed, mostly clinches seen as one box.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from afv.core.project import (  # the scan files' layout and reading live with the project (no media imports)
    SCAN_FORMAT,
    SCAN_VERSION,
    load_scan,
    read_scan,
    scan_dir,
    scan_paths,
    scan_threshold,
    shot_hash,
)
from afv.media.cuts import Cut, CutAnalysis, CutParams, detect_cuts
from afv.media.ffmpeg import ffmpeg_path, run_ffmpeg
from afv.media.normalise import (
    GEMX_FPS,
    GEMX_MAX_DURATION,
    estimate_frame_count,
    gemx_ffmpeg_args,
    normalised_frame_count,
    select_interval,
)
from afv.media.probe import ClipInfo, PacketTable, PathLike, packet_table, probe

if TYPE_CHECKING:
    from afv.core.project import Project
    from afv.core.project import Shot as ProjectShot

_END_GUARD = 0.5 / GEMX_FPS
DEFAULT_MIN_CONFIDENCE = 0.2  # also the CLI's `afv shots detect` default


@dataclass(frozen=True)
class Shot:
    index: int
    start: float  # seek time, pass to normalise_shot
    duration: float  # pass to normalise_shot
    start_frame: int  # first source frame
    end_frame: int  # one past the last source frame
    num_frames: int  # GEM-X's T predicted from timestamps (estimate_frame_count); the ingest
    # stage's ShotMedia.num_frames is the authoritative count
    cut_in: Cut | None = None
    cut_out: Cut | None = None
    split: bool = False  # True when produced by splitting a > 60 s shot

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def confidence(self) -> float:
        """Weakest of the two bounding cuts (1.0 at the clip's ends)."""
        cs = [c.confidence for c in (self.cut_in, self.cut_out) if c is not None]
        return min(cs) if cs else 1.0

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["end"] = self.end
        return d


def shots_from_cuts(
    cuts: Sequence[Cut],
    times: np.ndarray,
    info: ClipInfo,
    *,
    min_confidence: float = 0.0,
    include_gradual: bool = True,
    max_duration: float = GEMX_MAX_DURATION,
    table: PacketTable | None = None,
) -> list[Shot]:
    """Turn cuts into shot intervals covering the clip (see module docstring)."""
    use = sorted((c for c in cuts if c.confidence >= min_confidence
                  and (include_gradual or c.kind == "hard")), key=lambda c: c.frame)
    src_end = info.cv_duration
    n = len(times)
    bounds: list[tuple[float, float, int, int, Cut | None, Cut | None]] = []
    start, first, cut_in = 0.0, 0, None
    for c in use:
        if not 0 < c.start_frame <= c.frame < n:
            continue
        end = float(times[c.start_frame]) - _END_GUARD
        if end > start:
            bounds.append((start, end, first, c.start_frame, cut_in, c))
        start, first, cut_in = c.time, c.frame, c
    if src_end > start:
        bounds.append((start, src_end, first, n, cut_in, None))

    shots: list[Shot] = []
    for s, e, f0, f1, cin, cout in bounds:
        parts = max(1, math.ceil((e - s) / max_duration - 1e-9))
        # equal parts whose length is a whole number of 30 Hz ticks, so consecutive parts' shot
        # timelines (start + i/30) join without a gap or overlap
        step = math.ceil((e - s) * GEMX_FPS / parts - 1e-6) / GEMX_FPS if parts > 1 else e - s
        for j in range(parts):
            ps = s + step * j
            pe = e if j == parts - 1 else s + step * (j + 1)
            if pe <= ps:
                break
            pf0 = f0 if j == 0 else int(np.searchsorted(times, ps))
            pf1 = f1 if j == parts - 1 else int(np.searchsorted(times, pe))
            start_r = round(ps, 6)  # microseconds (what ffmpeg parses); the end never moves later
            dur = _floor6(pe - start_r)
            try:
                t = (estimate_frame_count(None, start_r, dur, info=info, table=table)
                     if table is not None else 0)
            except ValueError:
                t = 0
            shots.append(Shot(len(shots), start_r, _floor6(pe - start_r), pf0, pf1, t,
                              cin if j == 0 else None, cout if j == parts - 1 else None, parts > 1))
    return shots


def _floor6(x: float) -> float:
    """Round down to microseconds (ffmpeg parses -t to microseconds) so the end never moves later."""
    return math.floor(x * 1e6) / 1e6


def detect_shots(
    path: PathLike,
    params: CutParams | None = None,
    *,
    info: ClipInfo | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    include_gradual: bool = True,
) -> tuple[list[Shot], CutAnalysis]:
    """Detect cuts in ``path`` and return the shot list (cuts below ``min_confidence`` ignored)
    plus the full analysis (all cuts, per-frame scores)."""
    info = info or probe(path, hash=False, packets=False)
    analysis = detect_cuts(path, params)
    table = packet_table(path)
    shots = shots_from_cuts(analysis.cuts, analysis.times, info, min_confidence=min_confidence,
                            include_gradual=include_gradual, table=table)
    return shots, analysis


# ---------------------------------------------------------------------------
# media/cuts.json
# ---------------------------------------------------------------------------

CUTS_FILE = "cuts.json"
CUTS_FORMAT = "afv_cuts"
CUTS_VERSION = 1


def cuts_payload(
    analysis: CutAnalysis, source_sha256: str, *, min_confidence: float, include_gradual: bool = True
) -> dict[str, Any]:
    """``media/cuts.json`` for a cut analysis (see the module docstring)."""
    return {
        "format": CUTS_FORMAT,
        "version": CUTS_VERSION,
        "source_sha256": source_sha256,
        "fps": float(analysis.fps),
        "params": {
            "detector": analysis.params.model_dump(mode="json"),
            "min_confidence": float(min_confidence),
            "include_gradual": bool(include_gradual),
        },
        "cuts": [
            {
                "frame": int(c.frame), "time": round(float(c.time), 6), "confidence": float(c.confidence),
                "kind": c.kind, "start_frame": int(c.start_frame),
                "start_time": round(float(c.start_time), 6),
            }
            for c in analysis.cuts
        ],
    }  # fmt: skip


def read_cuts(path: Path) -> dict[str, Any] | None:
    """The cuts file at ``path``, or None when it is missing or not an ``afv_cuts`` file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("format") != CUTS_FORMAT:
        return None
    return data


# ---------------------------------------------------------------------------
# Candidate scan
# ---------------------------------------------------------------------------

SCAN_FRAME = 15  # shot-timeline frame scanned (0.5 s in, past a cut's first frames); last frame if shorter
ENCODE_MARGIN = 90  # frames encoded after the scan frame (see the module docstring)
MIN_EXACT_MARGIN = 60  # the smallest margin measured to give the exact frame
THUMB_WIDTH = 480
LIKELY_MIN_SCORE = 0.5
LIKELY_MIN_HEIGHT = 0.3  # of the frame height
LIKELY_MIN_TOP = 0.03  # box top below the frame top, of the frame height (head in the picture)
LIKELY_MIN_ASPECT = 0.8  # box height / width
LIKELY_MAX_PEOPLE = 3  # two fighters and a referee; more fighter-sized people is a crowd or a corner
LIKELY_PAIR_RATIO = 0.6  # second tallest fighter-sized person's height / the tallest one's
SCAN_JOBS = 4  # frames extracted in parallel (each ffmpeg/x264 runs several threads itself)


def scan_hints(scan: Mapping[str, Any]) -> tuple[bool, int]:
    """(fighters_likely, fighter_sized) of a scan by the current rule, from its stored people (the values
    stored in the file are what the rule said when it was scanned)."""
    try:
        p = scan["prompt"]
        return fighters_likely(p["boxes"], p["scores"], float(scan["size"][1]))
    except (KeyError, TypeError, ValueError, IndexError):
        return bool(scan.get("fighters_likely")), int(scan.get("fighter_sized") or 0)


def fighters_likely(boxes: Any, scores: Any, height: float) -> tuple[bool, int]:
    """(likely, fighter-sized people): see the module docstring."""
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    s = np.asarray(scores, np.float64).reshape(-1)
    h, w, hh = b[:, 3] - b[:, 1], b[:, 2] - b[:, 0], float(height)
    sized = ((s >= LIKELY_MIN_SCORE) & (h >= LIKELY_MIN_HEIGHT * hh) & (b[:, 1] >= LIKELY_MIN_TOP * hh)
             & (h >= LIKELY_MIN_ASPECT * w))  # fmt: skip
    tall = np.sort(h[sized])[::-1]
    likely = 2 <= len(tall) <= LIKELY_MAX_PEOPLE and tall[1] >= LIKELY_PAIR_RATIO * tall[0]
    return bool(likely), len(tall)


def extract_scan_frame(
    video: PathLike,
    start: float,
    duration: float | None,
    frame_jpg: Path,
    thumb_jpg: Path,
    *,
    info: ClipInfo | None = None,
    scan_frame: int = SCAN_FRAME,
    margin: int = ENCODE_MARGIN,
    thumb_width: int = THUMB_WIDTH,
    ffmpeg: str | None = None,
) -> int:
    """Write shot frame ``min(scan_frame, T-1)`` as ``frame_jpg`` (byte-identical to the frame
    ``afv.models.track.prepare_frames`` writes) and a thumbnail at most ``thumb_width`` wide; returns
    the frame index. Only the first ``scan_frame + margin + 1`` frames are encoded (module docstring);
    when that yields too few frames for an exact one (``MIN_EXACT_MARGIN``), the whole shot is encoded."""
    info = info or probe(video, hash=False, packets=False)
    selected = select_interval(info, start, duration)
    part = (scan_frame + margin + 1) / GEMX_FPS
    whole = part >= selected
    with tempfile.TemporaryDirectory(prefix="afv-scan-frame-") as tmp:
        mp4 = Path(tmp) / "shot.mp4"
        run_ffmpeg(gemx_ffmpeg_args(video, mp4, start, selected if whole else part), ffmpeg)
        n = normalised_frame_count(mp4)
        if not whole and n <= scan_frame + min(margin, MIN_EXACT_MARGIN):  # too few frames (a gap?):
            whole = True  # encode the whole shot, which is exact by definition
            run_ffmpeg(gemx_ffmpeg_args(video, mp4, start, selected), ffmpeg)
            n = normalised_frame_count(mp4)
        if n <= 0:
            raise ValueError(f"the interval {start:.6f}s+{selected:.6f}s contains no 30 Hz frame")
        k = min(scan_frame, n - 1)
        out, th = Path(tmp) / "frame.jpg", Path(tmp) / "thumb.jpg"
        last = ["-frames:v", str(k + 1), "-update", "1"]
        run_ffmpeg(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(mp4),
                    "-map", "0:v:0", "-fps_mode", "passthrough", "-q:v", "2", *last, str(out),
                    "-map", "0:v:0", "-fps_mode", "passthrough", "-vf", f"scale='min({thumb_width},iw)':-2",
                    "-q:v", "4", *last, str(th)], ffmpeg)  # fmt: skip
        _copy(out, Path(frame_jpg))
        _copy(th, Path(thumb_jpg))
    return k


def _copy(src: Path, dst: Path) -> None:
    from afv.core.cache import atomic_copy

    atomic_copy(src, dst)


@dataclass
class _Item:
    shot: ProjectShot
    hash: str
    threshold: float
    start: float
    duration: float
    frame: int = -1
    error: str = ""
    done: bool = False


@dataclass
class ScanReport:
    scanned: list[str] = field(default_factory=list)  # shots scanned now
    up_to_date: list[str] = field(default_factory=list)  # shots whose scan was kept
    failed: dict[str, str] = field(default_factory=dict)  # shot -> error
    cancelled: bool = False
    error: str = ""  # the worker's error, when it failed (scans finished before it are kept)
    seconds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Progress = Callable[[float | None, str], None]


def scan_shots(
    project: Project,
    shots: Sequence[ProjectShot],
    *,
    force: bool = False,
    worker: Any = None,
    progress: Progress | None = None,
    cancel: threading.Event | None = None,
    jobs: int = SCAN_JOBS,
    on_scan: Callable[[str, dict[str, Any]], None] | None = None,
) -> ScanReport:
    """Scan ``shots`` (module docstring): their frames in parallel, then one track-worker request for all
    (GPU lock held by ``afv.models.envs.run_worker``). Shots with an up-to-date scan are kept unless
    ``force``. Each finished scan is written at once (``on_scan(shot id, scan)`` is called), so a
    cancelled or failed scan keeps what it finished; running it again does the rest."""
    from afv.models import track

    rep = ScanReport()
    say = progress or (lambda frac, msg: None)
    stopped = (lambda: cancel.is_set()) if cancel is not None else (lambda: False)
    t0 = time.monotonic()
    pinfo = project.info
    sha = pinfo.source.sha256
    video = project.source_path()
    todo: list[_Item] = []
    for sh in shots:
        si = sh.info
        h = shot_hash(si.start, si.duration, sha)
        if not force:
            status, data = read_scan(sh, info=si, project_info=pinfo)
            if status == "ok" and data is not None:
                rep.up_to_date.append(sh.id)
                continue
        todo.append(_Item(sh, h, _det_threshold(sh, pinfo, si), si.start, si.duration))
    if not todo:
        return rep
    sdir = scan_dir(project)
    sdir.mkdir(parents=True, exist_ok=True)
    info = probe(video, hash=False, packets=False)
    with tempfile.TemporaryDirectory(prefix="afv-scan-") as tmp_s:
        tmp = Path(tmp_s)
        xdir = tmp / "frames"
        xdir.mkdir()
        lock = threading.Lock()
        count = [0]

        def extract(it: _Item) -> None:
            if stopped():
                return
            try:
                it.frame = extract_scan_frame(video, it.start, it.duration, xdir / f"{it.shot.id}.jpg",
                                              scan_paths(project, it.shot.id).thumb, info=info)
            except Exception as e:  # one bad shot must not stop the others
                it.error = f"{type(e).__name__}: {e}".splitlines()[0]
            with lock:
                count[0] += 1
                say(0.5 * count[0] / len(todo), f"frame {count[0]}/{len(todo)} (shot {it.shot.id})")

        say(0.0, f"reading the frames of {len(todo)} shot{'s' if len(todo) != 1 else ''}")
        with ThreadPoolExecutor(max(1, jobs)) as ex:
            list(ex.map(extract, todo))
        rep.seconds["frames"] = round(time.monotonic() - t0, 2)
        for it in todo:
            if it.error:
                rep.failed[it.shot.id] = it.error
        if stopped():
            rep.cancelled = True
            return rep
        ready = [it for it in todo if not it.error and it.frame >= 0]
        groups: dict[float, list[_Item]] = {}
        for it in ready:
            groups.setdefault(it.threshold, []).append(it)
        t1 = time.monotonic()
        for g, (thr, items) in enumerate(groups.items()):
            fdir, out = tmp / f"group{g}" / "frames", tmp / f"group{g}" / "out"
            fdir.mkdir(parents=True)
            for i, it in enumerate(items):
                os.replace(xdir / f"{it.shot.id}.jpg", fdir / f"{i:05d}.jpg")
            try:
                _detect_group(project, items, fdir, out, thr, sha, worker or track.envs_worker(), say, rep,
                              len(ready), on_scan)  # fmt: skip
            except Exception as e:
                if stopped():
                    rep.cancelled = True
                    return rep
                rep.error = str(e)
                first = str(e).splitlines()[0] if str(e) else type(e).__name__
                for it in items:
                    if not it.done and it.shot.id not in rep.failed:
                        rep.failed[it.shot.id] = f"not scanned: {first}"
                return rep
            if stopped():
                rep.cancelled = True
                return rep
        rep.seconds["detect"] = round(time.monotonic() - t1, 2)
    rep.seconds["total"] = round(time.monotonic() - t0, 2)
    return rep


def _det_threshold(shot: ProjectShot, project_info: Any = None, info: Any = None) -> float:
    """The shot's ``detect_track.det_threshold`` (``afv.core.project.scan_threshold``: what ``afv pick``
    and detect_track number people with, and what ``read_scan`` checks a scan against)."""
    return scan_threshold(project_info or shot.project.info, info or shot.info)


def _detect_group(
    project: Project, items: list[_Item], fdir: Path, out: Path, threshold: float, sha: str, worker: Any,
    say: Progress, rep: ScanReport, total: int, on_scan: Callable[[str, dict[str, Any]], None] | None,
) -> None:  # fmt: skip
    """One ``detect_prompt`` request for ``items`` (frame i of ``fdir`` = items[i]); a watcher thread
    writes each scan as soon as the worker has written its prompt."""
    from afv.models import track

    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    lock = threading.Lock()

    def harvest() -> None:
        with lock:
            for i, it in enumerate(items):
                if it.done:
                    continue
                try:
                    d = json.loads((out / f"prompt_{i:05d}.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue  # not written yet (or half written)
                try:
                    _finish_scan(project, it, d, fdir / f"{i:05d}.jpg", on_scan)
                except OSError as e:  # disk full, permissions: an error of this shot, not of the detector
                    it.done = True
                    where = f" ({e.filename})" if e.filename else ""
                    rep.failed[it.shot.id] = f"cannot write the scan: {e.strerror or e}{where}"
                    continue
                rep.scanned.append(it.shot.id)
                n = len(rep.scanned)
                say(0.5 + 0.5 * n / max(total, 1), f"people on {n}/{total} frames (shot {it.shot.id})")

    def watch() -> None:
        while not stop.wait(0.5):
            with contextlib.suppress(Exception):
                harvest()

    say(0.5, f"detecting people on {len(items)} frames (loading RF-DETR)")
    th = threading.Thread(target=watch, daemon=True, name="scan-watch")
    th.start()
    frames = track.ShotFrames(fdir, len(items), 0.0, 0.0)
    try:
        track.detect_prompt(project.source_path(check=False), out, list(range(len(items))),
                            threshold=threshold, worker=worker, shot_frames=frames)  # fmt: skip
    finally:
        stop.set()
        th.join()
        harvest()


def _finish_scan(
    project: Project, it: _Item, det: Mapping[str, Any], frame_jpg: Path,
    on_scan: Callable[[str, dict[str, Any]], None] | None,
) -> None:  # fmt: skip
    """Write ``<id>.frame.jpg`` and ``<id>.json`` from the worker's prompt for the item's frame."""
    from afv.core.cache import atomic_write_json

    p = scan_paths(project, it.shot.id)
    w, h = _jpeg_size(frame_jpg)
    prompt = {**dict(det), "frame": it.frame, "prompt_frame": it.frame, "image": p.prompt.name}
    likely, tall = fighters_likely(prompt["boxes"], prompt["scores"], h)  # stored; readers use scan_hints
    data = {
        "format": SCAN_FORMAT, "version": SCAN_VERSION, "shot": it.shot.id, "shot_hash": it.hash,
        "frame": it.frame, "prompt": prompt, "people": len(prompt["boxes"]), "fighter_sized": tall,
        "fighters_likely": likely, "size": [w, h], "scanned": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }  # fmt: skip
    _copy(frame_jpg, p.frame)
    with contextlib.suppress(OSError):
        p.prompt.unlink()  # numbered image of an earlier scan
    atomic_write_json(p.json, data)
    it.done = True
    if on_scan is not None:
        on_scan(it.shot.id, data)


def _jpeg_size(path: Path) -> tuple[int, int]:
    import av

    with av.open(str(path)) as c:
        cc = c.streams.video[0].codec_context
        return int(cc.width), int(cc.height)


def remove_scan(project: Project, shot_id: str) -> None:
    p = scan_paths(project, shot_id)
    for f in (p.json, p.thumb, p.frame, p.prompt):
        with contextlib.suppress(FileNotFoundError):
            f.unlink()


# ---------------------------------------------------------------------------
# Numbered prompt image (written on demand; there is no OpenCV in the afv env)
# ---------------------------------------------------------------------------

_COLOURS_RGB = [(66, 135, 245), (245, 90, 66), (80, 200, 120), (230, 200, 40), (200, 80, 220), (40, 220, 220)]
_GLYPHS = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
}


def draw_numbered(rgb: np.ndarray, boxes: Any) -> np.ndarray:
    """``rgb`` with each box outlined and its number (0 = first box) written in its top-left corner."""
    img = np.ascontiguousarray(rgb).copy()
    hh, ww = img.shape[:2]
    t = max(2, hh // 240)  # line width
    s = max(3, hh // 120)  # glyph pixel size
    for k, b in enumerate(np.asarray(boxes, np.float64).reshape(-1, 4)):
        c = np.array(_COLOURS_RGB[k % len(_COLOURS_RGB)], np.uint8)
        x0, y0 = int(np.clip(b[0], 0, ww - 1)), int(np.clip(b[1], 0, hh - 1))
        x1, y1 = int(np.clip(b[2], x0 + 1, ww)), int(np.clip(b[3], y0 + 1, hh))
        img[y0:min(y0 + t, y1), x0:x1] = c
        img[max(y1 - t, y0):y1, x0:x1] = c
        img[y0:y1, x0:min(x0 + t, x1)] = c
        img[y0:y1, max(x1 - t, x0):x1] = c
        text = str(k)
        gw, gh = (6 * len(text) + 1) * s, 9 * s
        lx, ly = min(x0 + t, max(ww - gw, 0)), min(y0 + t, max(hh - gh, 0))
        img[ly:ly + gh, lx:lx + gw] = 0
        for j, ch in enumerate(text):
            for r, row in enumerate(_GLYPHS[ch]):
                for q, bit in enumerate(row):
                    if bit == "1":
                        y, x = ly + (r + 1) * s, lx + (j * 6 + q + 1) * s
                        img[y:y + s, x:x + s] = c
    return img


def read_jpeg(path: Path) -> np.ndarray:
    import av

    with av.open(str(path)) as c:
        return next(c.decode(video=0)).to_ndarray(format="rgb24")


def write_jpeg(rgb: np.ndarray, path: Path, quality: int = 3) -> None:
    """Encode ``rgb`` [H,W,3] uint8 as a JPEG with the pinned ffmpeg (``-q:v quality``), atomically."""
    h, w = rgb.shape[:2]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".jpg", dir=path.parent)
    os.close(fd)
    try:
        proc = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-", "-frames:v", "1", "-update", "1",
             "-q:v", str(quality), tmp],
            input=np.ascontiguousarray(rgb, np.uint8).tobytes(), capture_output=True, check=False,
        )  # fmt: skip
        if proc.returncode != 0:
            raise OSError(f"ffmpeg could not write {path}: {proc.stderr.decode('utf-8', 'replace')[-300:]}")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def prompt_image(shot: ProjectShot) -> Path | None:
    """``media/scan/<id>.prompt.jpg``: the scanned frame with its numbered people (written when missing
    or older than the scan); None without a usable scan."""
    scan = load_scan(shot)
    p = scan_paths(shot.project, shot.id)
    if scan is None or not p.frame.is_file():
        return None
    with contextlib.suppress(OSError):
        if p.prompt.stat().st_mtime_ns >= p.json.stat().st_mtime_ns:
            return p.prompt
    write_jpeg(draw_numbered(read_jpeg(p.frame), scan["prompt"]["boxes"]), p.prompt)
    return p.prompt


# ---------------------------------------------------------------------------
# Source proxy (media/proxy.mp4)
# ---------------------------------------------------------------------------

PROXY_FILE = "proxy.mp4"
PROXY_INFO = "proxy.json"


def make_source_proxy(media_dir: Path, video: PathLike, source_sha256: str, *, height: int = 720) -> Path:
    """``media/proxy.mp4``: the whole source at most ``height`` px tall with a short GOP (for scrubbing in
    Blender), and ``proxy.json`` naming the source it was made from."""
    from afv.core.cache import atomic_write_json
    from afv.media.proxy import make_proxy

    media_dir = Path(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    out, part = media_dir / PROXY_FILE, media_dir / f".{PROXY_FILE}.part.mp4"
    try:
        make_proxy(video, part, height=height)
        os.replace(part, out)
    finally:
        with contextlib.suppress(OSError):
            part.unlink()
    atomic_write_json(media_dir / PROXY_INFO, {"file": PROXY_FILE, "source_sha256": source_sha256,
                                               "height": height})
    return out


def source_proxy(media_dir: Path, source_sha256: str) -> Path | None:
    """The proxy when it was made from this source, else None."""
    try:
        meta = json.loads((Path(media_dir) / PROXY_INFO).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    p = Path(media_dir) / PROXY_FILE
    ok = isinstance(meta, dict) and meta.get("source_sha256") == source_sha256 and p.is_file()
    return p if ok else None
