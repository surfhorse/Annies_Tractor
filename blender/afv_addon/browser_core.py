"""The shot browser's plain-Python part (no bpy, no afv): source frames <-> shot times, the shot list's
filter and order, display text, the queue check, the range of **Add shot here**, cut markers, and the
one-shot ``afv`` command lines (with their results turned into the ``afv serve`` result shapes) used when
the worker is off or unavailable.

Frames are **source frames** (the browser scene's frame f shows source frame f, at ``f / fps`` s). Shot
times follow ``afv shots detect``: a shot that starts at frame f starts half a frame earlier, at
``(f - 0.5) / fps``, and one whose last frame is l ends at ``(l + 0.5) / fps``, so the 30 Hz shot timeline
picks exactly those frames whatever the rounding. A row is a shot as ``afv serve`` ``shots`` lists it
(``afv.cli.shot_row``).
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

STATES = ("candidate", "queued", "skipped")
EPS = 1e-6
SHOT_HZ = 30.0  # the shot timeline (scan and prompt frames)

_ABS_WIN = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]{2})")  # D:\... or \\server\... (UNC, \\wsl.localhost\...)


# --- paths ----------------------------------------------------------------------------------------
def project_folder(text: str, base: str, default_name: str) -> str:
    r"""The new project's folder from the dialog's Project folder field: empty -> ``base/default_name``;
    a full Windows path (``D:\...``, ``\\wsl.localhost\...``) as given; anything else (a plain new name,
    ``sub\name``, a Blender-relative ``//name`` from an unsaved file) is taken relative to ``base`` (the
    projects folder, else the video's folder). Trailing separators are dropped."""
    t = (text or "").strip().strip('"')
    if not t:
        return os.path.join(base, default_name).rstrip("\\/")
    if t.startswith("/") and not t.startswith("//"):
        return t.rstrip("/")  # a WSL path (/mnt/d/..., /home/...)
    if not _ABS_WIN.match(t) or t.startswith("//"):
        t = os.path.join(base, t.lstrip("\\/"))
    return t.rstrip("\\/")


# --- time <-> frames ------------------------------------------------------------------------------
def start_time(frame: int, fps: float) -> float:
    """Shot start (s) for a shot whose first frame is source frame ``frame``."""
    return max(0.0, (int(frame) - 0.5) / fps)


def end_time(frame: int, fps: float) -> float:
    """Shot end (s, exclusive) for a shot whose last frame is source frame ``frame``."""
    return (int(frame) + 0.5) / fps


def first_frame(start: float, fps: float) -> int:
    """The first source frame inside a shot starting at ``start`` s."""
    return max(0, math.ceil(float(start) * fps - EPS))


def last_frame(end: float, fps: float) -> int:
    """The last source frame inside a shot ending (exclusive) at ``end`` s."""
    return max(0, math.ceil(float(end) * fps - EPS) - 1)


def shot_frames(row: dict, fps: float) -> tuple[int, int]:
    """(first, last) source frame of a row (last >= first)."""
    a = first_frame(row["start"], fps)
    return a, max(a, last_frame(row_end(row), fps))


def start_of(row: dict) -> float:
    """A row's start (0 for a row afv could not read: it has ``error`` instead of the shot's fields)."""
    try:
        return float(row.get("start") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def has_range(row: dict) -> bool:
    return not row.get("error") and isinstance(row.get("start"), int | float)


def row_end(row: dict) -> float:
    end = row.get("end")
    return float(end) if end is not None else float(row["start"]) + float(row.get("duration") or 0.0)


def prompt_source_frame(row: dict, prompt_frame: int, fps: float) -> int:
    """The source frame that shot-timeline frame ``prompt_frame`` (30 Hz) of a row shows: the shot's first
    source frame plus ``round(i * fps / 30)`` (as afv's normaliser picks frames)."""
    return first_frame(row["start"], fps) + round(int(prompt_frame) * fps / SHOT_HZ)


def shot_frame_at(row: dict, source_frame: int, fps: float) -> int:
    """The shot-timeline frame (30 Hz) showing source frame ``source_frame`` (clamped into the shot): what
    **Pick on this frame** sends as ``frame``."""
    a, b = shot_frames(row, fps)
    f = min(max(int(source_frame), a), b)
    n = max(1, math.floor(float(row.get("duration") or (b - a + 1) / fps) * SHOT_HZ + EPS))
    return min(max(0, round((f - a) * SHOT_HZ / fps)), n - 1)


def fmt_time(t: float | None) -> str:
    """``m:ss.s`` (``h:mm:ss.s`` past an hour)."""
    if t is None:
        return "-"
    t = max(0.0, float(t))
    h, rest = divmod(t, 3600.0)
    m, s = divmod(rest, 60.0)
    if s >= 59.95:  # rounding would print 60.0
        m, s = m + 1, 0.0
    return f"{int(h)}:{int(m):02d}:{s:04.1f}" if h >= 1 else f"{int(m)}:{s:04.1f}"


def fmt_len(d: float | None) -> str:
    return "-" if d is None else f"{float(d):.1f} s"


# --- the list ---------------------------------------------------------------------------------------
def counts(rows: list[dict]) -> dict[str, int]:
    """Shots per state (rows afv could not read have no state and are not counted: ``errors``)."""
    out = {s: 0 for s in STATES}
    for r in rows:
        st = r.get("state", "queued")
        if not r.get("error") and st in out:
            out[st] += 1
    return out


def errors(rows: list[dict]) -> int:
    return sum(1 for r in rows if r.get("error"))


def likely(row: dict) -> bool:
    return bool((row.get("scan") or {}).get("fighters_likely"))


def visible(rows: list[dict], state: str = "all", order: str = "time") -> list[dict]:
    """The rows to show: ``state`` "all" or one state; ``order`` "time" (by start) or "likely" (fighters
    likely first, then by start). Fighters-likely is a hint: it sorts and highlights, it never hides."""
    out = [r for r in rows if state in ("all", "", None) or r.get("error")  # never hide an unreadable shot
           or r.get("state", "queued") == state]
    if order == "likely":
        out.sort(key=lambda r: (not likely(r), start_of(r), str(r.get("id"))))
    else:
        out.sort(key=lambda r: (start_of(r), str(r.get("id"))))
    return out


def pick_status(row: dict) -> str:
    """"picked", "re-pick" (picked for another interval) or "" (not picked)."""
    if row.get("picked"):
        return "re-pick" if row.get("pick_stale") else "picked"
    return ""


def people_text(row: dict) -> str:
    sc = row.get("scan") or {}
    status = sc.get("status")
    if status == "ok":
        n = sc.get("people")
        return f"{n} {'person' if n == 1 else 'people'}" + (", fighters likely" if sc.get("fighters_likely")
                                                            else "")
    return {"stale": "scan stale (trimmed)", "bad": "scan unreadable", "none": "not scanned"}.get(
        status or "none", "not scanned")


def conf_text(row: dict) -> str:
    c = row.get("confidence")
    return "-" if c is None else f"{100 * float(c):.0f}%"


def row_line(row: dict) -> str:
    """The list's first line (its button): id, start, length, cut confidence, ``*`` when fighters are
    likely."""
    if row.get("error"):
        return f"{row.get('id')}  cannot read this shot: {row['error']}"
    star = "  *" if likely(row) else ""
    return (f"{row.get('id')}  {fmt_time(row.get('start'))}  {fmt_len(row.get('duration'))}  "
            f"{conf_text(row)}{star}")


OUTCOME_TEXT = {"ok": "ran ok", "needs_pick": "needs a pick", "needs_review": "needs a tracking review",
                "code_changed": "run again (code changed)", "failed": "failed", "cancelled": "stopped"}


def row_detail(row: dict, outcome: str | None = None) -> str:
    """The list's second line: state, people found, pick status, the last batch's outcome, notes."""
    if row.get("error"):
        return "fix or delete its shot.json, then Refresh"
    parts = [row.get("state", "queued"), people_text(row)]
    pick = pick_status(row)
    if pick:
        parts.append(pick)
    if outcome:
        parts.append(f"last run: {OUTCOME_TEXT.get(outcome, outcome)}")
    if row.get("notes"):
        parts.append(str(row["notes"]))
    return ", ".join(parts)


def queue_check(rows: list[dict]) -> tuple[list[str], list[tuple[str, str]]]:
    """(queued shots ready to run, [(queued shot, why not)]): a queued shot needs its fighters picked for
    its current interval."""
    ready, missing = [], []
    for r in sorted(rows, key=lambda r: (start_of(r), str(r.get("id")))):
        if r.get("state", "queued") != "queued" or r.get("error"):
            continue
        if not r.get("picked"):
            missing.append((r["id"], "fighters not picked"))
        elif r.get("pick_stale"):
            missing.append((r["id"], "picked for another interval: pick again"))
        else:
            ready.append(r["id"])
    return ready, missing


def needs_scan(rows: list[dict], ids=None) -> list[str]:
    """Shots (``ids``, default: the queued ones without a valid pick) whose scan is missing or stale:
    picking them would start the detector one shot at a time; one scan job does them all."""
    want = set(ids) if ids is not None else {sid for sid, _ in queue_check(rows)[1]}
    return [r["id"] for r in sorted(rows, key=lambda r: (start_of(r), str(r.get("id"))))
            if r["id"] in want and not r.get("error") and (r.get("scan") or {}).get("status") != "ok"]


def scan_estimate(n: int) -> float:
    """Seconds a scan job of ``n`` shots takes (measured on the 16-minute sample: about 15 s to start the
    GPU worker, then about 0.6 s a shot)."""
    return 15.0 + 0.6 * max(0, n)


def next_in_order(ids: list[str], current: str | None, step: int = 1) -> str | None:
    """The shot ``step`` places after ``current`` in ``ids`` (the list's order), or None at the end."""
    if not ids:
        return None
    if current not in ids:
        return ids[0] if step > 0 else ids[-1]
    k = ids.index(current) + step
    return ids[k] if 0 <= k < len(ids) else None


def next_to_pick(rows: list[dict], after: str | None = None) -> str | None:
    """The next queued shot (in time order, after shot ``after``) whose fighters still need picking."""
    _ready, missing = queue_check(rows)
    ids = [sid for sid, _ in missing]
    if after in ids:
        ids = ids[ids.index(after) + 1:] + ids[:ids.index(after)]
    return ids[0] if ids else None


class Session:
    """The open project's shot list as afv last sent it (``shots`` payload) plus the rows changed since,
    by shot id. Plain data: the panel, the timeline strip and the operators read it."""

    def __init__(self):
        self.clear()

    def clear(self) -> None:
        self.payload: dict | None = None
        self.rows: dict[str, dict] = {}
        self.cuts: list[tuple[int, str]] = []
        self.selected = ""  # the selected shot (the scene's copy is put back from this after an undo)
        self.video_warning = ""  # why the timeline shows no video (kept in the panel)
        self.outcomes: dict[str, str] = {}  # shot -> its outcome in the last batch (shown in the list)
        self.version = 0
        self._cache: dict = {}

    @property
    def open(self) -> bool:
        return self.payload is not None

    @property
    def project(self) -> str:
        return (self.payload or {}).get("project") or ""

    @property
    def root_win(self) -> str:
        return (self.payload or {}).get("root_win") or ""

    @property
    def source(self) -> dict:
        return (self.payload or {}).get("source") or {}

    @property
    def fps(self) -> float:
        return float(self.source.get("fps") or 30.0)

    @property
    def num_frames(self) -> int:
        n = self.source.get("num_frames")
        if n:
            return int(n)
        return max(1, round(float(self.source.get("duration") or 0.0) * self.fps))

    def load(self, payload: dict) -> None:
        self.payload = payload
        self.rows = {str(r["id"]): r for r in payload.get("shots") or []}
        self._changed()

    def update_row(self, row: dict | None) -> None:
        if row and row.get("id") is not None:
            self.rows[str(row["id"])] = row
            self._changed()

    def _changed(self) -> None:
        self.version += 1
        self._cache.clear()

    def all_rows(self) -> list[dict]:
        return list(self.rows.values())

    def counts(self) -> dict[str, int]:
        return counts(self.all_rows())

    def errors(self) -> int:
        return errors(self.all_rows())

    def visible(self, state: str = "all", order: str = "time") -> list[dict]:
        key = (state, order)
        if key not in self._cache:
            self._cache[key] = visible(self.all_rows(), state, order)
        return self._cache[key]


def batch_view(status: dict | None) -> dict:
    """The parts of a serve ``batch_status`` (or ``batch_cancel``) result the panel shows: {running, id,
    shot, stage, frac, index, outcomes [{shot, outcome, message}], log, started, finished, interrupted,
    cancelled}. The batch's id is ``batch_id`` (never the reply's request ``id``)."""
    st = status or {}
    cur = st.get("current") if isinstance(st.get("current"), dict) else {}
    done = st.get("done") if isinstance(st.get("done"), dict) else {}
    outs = st.get("outcomes") or done.get("outcomes") or []
    if isinstance(outs, dict):
        outs = [{"shot": k, "outcome": v} for k, v in outs.items()]
    outcomes = [{"shot": str(o.get("shot")), "outcome": o.get("outcome"), "message": o.get("message") or ""}
                for o in outs if isinstance(o, dict)]
    return {"running": bool(st.get("running")), "id": st.get("batch_id") or done.get("batch_id"),
            "shot": cur.get("shot"), "stage": cur.get("stage") or "", "frac": cur.get("frac"),
            "index": cur.get("index"), "outcomes": outcomes, "log": st.get("log") or st.get("log_path"),
            "started": st.get("started"), "finished": st.get("finished"),
            "interrupted": bool(st.get("interrupted")),
            "cancelled": bool(done.get("cancelled") or (st.get("cancelled") and not st.get("running")))}


def outcome_counts(outcomes: list[dict]) -> str:
    """"5 ran ok, 1 needs a tracking review" (in a fixed order)."""
    n: dict = {}
    for o in outcomes:
        n[o.get("outcome")] = n.get(o.get("outcome"), 0) + 1
    order = [*OUTCOME_TEXT, *sorted(str(k) for k in n if k not in OUTCOME_TEXT)]
    return ", ".join(f"{n[k]} {OUTCOME_TEXT.get(k, k)}" for k in order if n.get(k)) or "no shots ran"


def when(stamp: str | None) -> str:
    """``2026-11-11T22:41:05...`` -> ``11-11 22:41`` (as stored; empty when unknown)."""
    m = re.match(r"\d{4}-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})", str(stamp or ""))
    return f"{m.group(1)}-{m.group(2)} {m.group(3)}:{m.group(4)}" if m else ""


_CLI_ADVICE = (
    (re.compile(r"`?afv shots scan ([\w,-]+)`?"), r"Scan shots (shot \1)"),
    (re.compile(r"`?afv pick ([\w-]+)(?: --fighters[^;)`]*)?`?"), r"Pick fighters on shot \1"),
    (re.compile(r"`?afv run ([\w-]+)(?: --yes)?`?"), r"Run queue (shot \1)"),
    (re.compile(r"`?afv shots queue ([\w,-]+)`?"), r"Queue (shot \1)"),
)
_LONG_PATH = re.compile(r"(?:/mnt/[a-z]|[A-Za-z]:|\\\\wsl[^\\/\s]*)[\\/][^\s'\":;,()]{30,}")


def humanize(text: str, roots: tuple[str, ...] = ()) -> str:
    """A message from afv for the panel: the project folder dropped from paths, other long paths shortened
    to their last two parts, and CLI advice ("afv pick 042") as the browser's buttons."""
    out = str(text)
    for root in roots:
        if root:
            for r in {root, root.replace("\\", "/")}:
                out = out.replace(r.rstrip("\\/") + "/", "").replace(r.rstrip("\\/") + "\\", "")
    out = _LONG_PATH.sub(lambda m: "..." + "/".join(re.split(r"[\\/]", m.group(0))[-2:]), out)
    for pat, rep_ in _CLI_ADVICE:
        out = pat.sub(rep_, out)
    return out


# --- cuts and Add shot here -------------------------------------------------------------------------
def cut_frames(cuts: dict | None) -> list[tuple[int, str]]:
    """[(source frame, marker name)] of ``media/cuts.json``: each cut at the first frame of the segment
    after it (``frame``; for a fade, where the fade ends: ``start_frame`` is where it begins)."""
    out = []
    for c in (cuts or {}).get("cuts") or []:
        try:
            f = int(c.get("frame", c.get("start_frame")))
        except (TypeError, ValueError):
            continue
        conf = c.get("confidence")
        kind = "fade" if c.get("kind") == "gradual" else "cut"
        out.append((f, f"{kind} {100 * float(conf):.0f}%" if conf is not None else kind))
    out.sort()
    return out


def add_range(frame: int, fps: float, cut_starts: list[int], num_frames: int | None,
              default_s: float = 5.0, max_s: float = 10.0) -> tuple[float, float]:
    """(start, end) s for **Add shot here** at source frame ``frame``: from the playhead to the next cut
    when it is 0.5..``max_s`` s away, else ``default_s`` s; never past the source's end."""
    last = (num_frames - 1) if num_frames else None
    nxt = next((c for c in sorted(cut_starts) if c > frame), None)
    if nxt is not None and 0.5 * fps <= nxt - frame <= max_s * fps:
        end_f = nxt - 1
    else:
        end_f = frame + round(default_s * fps) - 1
    if last is not None:
        end_f = min(end_f, last)
    end_f = max(end_f, frame)
    return start_time(frame, fps), end_time(end_f, fps)


# --- paths --------------------------------------------------------------------------------------------
def scan_file(root: str | os.PathLike, scan_dir: str, shot: str, suffix: str) -> Path:
    """``<project>/<scan_dir>/<shot><suffix>`` (e.g. ``.frame.jpg``); ``scan_dir`` as afv lists it."""
    return Path(root) / Path(scan_dir.replace("\\", "/")) / f"{shot}{suffix}"


def project_file(root: str | os.PathLike, rel: str | None) -> Path | None:
    return Path(root) / Path(rel.replace("\\", "/")) if rel else None


# --- one-shot fallback ----------------------------------------------------------------------------------
def _num(v) -> str:
    return repr(float(v)) if isinstance(v, int | float) and not isinstance(v, bool) else str(v)


def oneshot_argv(cmd: str, args: dict) -> list[str]:
    """The ``afv`` command line (without ``--json``) that does what serve request ``cmd`` does."""
    p = ["--project", str(args["project"])]
    if cmd == "shots":
        return ["shots", "list", *p]
    if cmd == "shot_set":
        out = ["shots", "set", str(args["shot"]), *p]
        for k in ("start", "end", "duration", "speed", "notes", "state"):
            if args.get(k) is not None:
                out += [f"--{k}", _num(args[k])]
        return out
    if cmd == "shot_add":
        out = ["shots", "add", "--start", _num(args["start"]), *p]
        out += ["--end", _num(args["end"])] if args.get("end") is not None else ["--duration",
                                                                                  _num(args["duration"])]
        for k in ("speed", "notes", "state"):
            if args.get(k) is not None:
                out += [f"--{k}", _num(args[k])]
        return out
    if cmd == "shots_scan":
        return ["shots", "scan", *[str(s) for s in args.get("shots") or []], *p,
                *(["--force"] if args.get("force") else [])]
    if cmd == "pick":
        out = ["pick", str(args["shot"]), *p]
        if args.get("fighters") is not None:
            out += ["--fighters", ",".join(str(int(v)) for v in args["fighters"])]
            if args.get("extra"):
                out += ["--extra", ",".join(str(int(v)) for v in args["extra"])]
        if args.get("frame") is not None and args.get("fighters") is not None:
            out += ["--frame", str(int(args["frame"]))]
        return out
    if cmd == "batch":  # as serve's batch: exactly ``shots`` (--shot), else the shots in ``state``
        out = ["batch", *p]
        if args.get("to"):
            out += ["--to", str(args["to"])]
        for sid in args.get("shots") or []:
            out += ["--shot", str(sid)]
        st = args.get("state")
        for s in ([st] if isinstance(st, str) else st or []):
            out += ["--state", str(s)]
        if args.get("yes"):
            out.append("--yes")
        return out
    if cmd == "init":
        out = ["init", str(args["video"]), "--project", str(args["project"]), "--detect"]
        if args.get("title"):
            out += ["--title", str(args["title"])]
        if args.get("proxy") is True:
            out.append("--proxy")
        return out
    raise ValueError(f"no one-shot afv command for {cmd!r}")


def _event(events: list[dict], name: str) -> dict | None:
    return next((e for e in reversed(events) if e.get("event") == name), None)


def oneshot_result(cmd: str, args: dict, events: list[dict], root_win: str | None = None) -> dict | None:
    """The serve-shaped result of a finished one-shot command from its ``--json`` events (None: it
    failed). ``shots`` and ``init`` read project.json and media/ under ``root_win`` (the project folder as
    this process sees it) for the list's source part."""
    if cmd == "shots":
        rows = [{k: v for k, v in e.items() if k != "event"} for e in events if e.get("event") == "shot"]
        if root_win is None:
            return None
        return list_payload(root_win, rows, str(args["project"]))
    if cmd == "shot_set":
        ev = _event(events, "shot_set")
        if ev is None:
            return None
        row = {k: v for k, v in ev.items() if k not in ("event", "shot", "changed", "notes_for_user")}
        return {"shot": ev["shot"], "changed": ev.get("changed", []), "notes_for_user":
                ev.get("notes_for_user", []), "row": row}
    if cmd == "shot_add":
        ev = _event(events, "shot_added")
        return None if ev is None else {"shot": ev["id"], "row": None}
    if cmd == "shots_scan":
        ev = _event(events, "scan_done")
        return None if ev is None else {k: v for k, v in ev.items() if k != "event"}
    if cmd == "pick":
        if args.get("fighters") is None:
            ev = _event(events, "pick_prompt")
            if ev is None:
                return None
            prompt = ev.get("prompt") or {}
            return {"shot": ev["shot"], "source": ev.get("source"), "frame": prompt.get(
                "prompt_frame", prompt.get("frame")), "prompt": prompt, "image": ev.get("prompt_image"),
                "image_win": None, "current": ev.get("current")}
        ev = _event(events, "picked")
        return None if ev is None else {"shot": ev["shot"], "source": ev.get("source"),
                                        "fighters": ev.get("fighters"), "extra": ev.get("extra"), "row": None}
    if cmd == "batch":
        ev = _event(events, "batch_done")
        if ev is None:
            return None
        outs = ev.get("outcomes") or []
        if isinstance(outs, dict):  # an afv from before round 6's fix round: {shot: outcome}
            outs = [{"shot": k, "outcome": v} for k, v in outs.items()]
        outcomes = [{"shot": str(o.get("shot")), "outcome": o.get("outcome"),
                     "message": o.get("message") or "", "exports": list(o.get("exports") or [])}
                    for o in outs if isinstance(o, dict)]
        return {**{k: v for k, v in ev.items() if k not in ("event", "outcomes")}, "outcomes": outcomes}
    if cmd == "take":
        ev = next((e for e in reversed(events) if e.get("event") in ("take", "blend", "result")
                   and e.get("path")), None)
        return None if ev is None else {"path": ev["path"]}
    if cmd == "init":
        ev = _event(events, "init")
        if ev is None or root_win is None:
            return None
        return {"created": ev.get("project"), "detect": _event(events, "detect_done"), **list_payload(
            root_win, [], str(args["project"]))}
    return None


def list_payload(root_win: str | os.PathLike, rows: list[dict], project: str) -> dict:
    """A serve ``shots``-shaped shot list from rows and the project's files (read here)."""
    root = Path(root_win)
    try:
        info = json.loads((root / "project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = {}
    src = info.get("source") or {}
    probe = src.get("probe") or {}
    proxy = None
    try:
        meta = json.loads((root / "media" / "proxy.json").read_text(encoding="utf-8"))
        if meta.get("source_sha256") == src.get("sha256") and (root / "media" / "proxy.mp4").is_file():
            proxy = "media/proxy.mp4"
    except (OSError, ValueError, AttributeError):
        pass
    return {
        "project": project, "root_win": str(root),
        "source": {"path": src.get("path"), "path_win": None, "sha256": src.get("sha256"),
                   "title": src.get("title"), "fps": probe.get("avg_fps"),
                   "width": probe.get("display_width"),
                   "height": probe.get("display_height"), "duration": probe.get("duration"),
                   "num_frames": probe.get("num_frames")},
        "proxy": proxy, "cuts": "media/cuts.json" if (root / "media" / "cuts.json").is_file() else None,
        "scan_dir": "media/scan", "counts": counts(rows), "shots": rows,
    }  # fmt: skip
