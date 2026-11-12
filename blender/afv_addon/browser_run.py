"""The shot browser's **Run queue** and **Import take** of a finished shot.

Run queue: afv ``batch`` of the queued shots (``state: queued``), refused while a queued shot has no pick
(those are listed; **Run the N picked ones** sends just those shots). afv serve starts the batch as a
**detached** ``afv batch`` process: it keeps running when Blender closes, a file is opened, the add-on is
disabled or the worker restarts (nothing here ever cancels it except **Stop batch**), and afv reads the
queue again after each shot, so shots queued and picked while it runs are run in the same batch. While
connected, the ``batch`` request streams its progress here. **Open project** asks ``batch_status``: a
running batch is attached to (``batch`` streams its log from the start), else the last batch's results are
shown (when it finished, how each shot ended; each shot's last outcome is in its list row). **Stop batch**
sends ``batch_cancel`` and says how many shots ran and how many did not. Without the worker (one-shot afv
commands) there is no status or cancel command: the batch is then a child of Blender.

Import take opens the shot's take in a scene of its own, marked with ``TAKE_MARK``: written first when it
is missing or older than the shot's results (``take`` request), then built into that scene, whichever
scene the window shows meanwhile; opening it again finds the marked scene (no duplicates).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import bpy
from bpy.props import BoolProperty, StringProperty

from . import browser_core as core
from . import browser_jobs as jobs
from . import builder, ops, takeio
from .browser import SESSION, STAGES, TIMES, _need_project, _Op, _project_arg, _short, props, refresh

STATUS_POLL_S = 2.0
TAKE_MARK = "afv_browser_take"  # scene: JSON {"root_win", "shot"} of the take it holds
# the Run queue's state; "batch": the running batch (core.batch_view); "last": the last finished one;
# "status_cmd": False when the afv in WSL has no batch_status / batch_cancel
RUNQ: dict = {"missing": [], "outcomes": [], "current": None, "summary": "", "batch": None, "last": None,
              "status_cmd": True}


# --- run queue --------------------------------------------------------------------------------------------
def _outcomes(items) -> list[dict]:
    return [{k: o.get(k) for k in ("shot", "outcome", "message")} for o in items if isinstance(o, dict)]


def _unknown(msg: str) -> bool:
    return "unknown cmd" in str(msg)


def running() -> bool:
    """A batch of this project runs in WSL (streamed here, or known from batch_status)."""
    return jobs.busy("batch") or bool((RUNQ.get("batch") or {}).get("running"))


def detached() -> bool:
    """A batch runs in WSL that this Blender does not stream (attaching failed: it is polled)."""
    return bool((RUNQ.get("batch") or {}).get("running")) and not jobs.busy("batch")


def _finish_time(view: dict) -> str | None:
    """When the batch ended: afv's status has no time for it, so the batch log's last write."""
    rel = view.get("log")
    log = core.project_file(SESSION.root_win, rel) if rel and SESSION.root_win else None
    try:
        return time.strftime("%Y-%m-%dT%H:%M", time.localtime(os.stat(log).st_mtime)) if log else None
    except OSError:
        return None


def _remember(view: dict) -> None:
    """A finished (or stopped) batch: its results stay shown, per shot in the list too."""
    if not isinstance(view.get("finished"), str):
        view["finished"] = _finish_time(view)
    RUNQ["last"] = view
    RUNQ["outcomes"] = view["outcomes"]
    SESSION.outcomes = {o["shot"]: o["outcome"] for o in view["outcomes"]}
    RUNQ["summary"] = last_line(view)


def last_line(view: dict) -> str:
    how = "stopped" if view.get("cancelled") else "interrupted" if view.get("interrupted") else "finished"
    at = core.when(view.get("finished"))
    counts = core.outcome_counts(view["outcomes"])
    return f"last batch {view.get('id') or ''}, {how}{' ' + at if at else ''}: {counts}"


def batch_status(context=None, poll=True):
    """Ask afv for the project's batch (``batch_status``): a running one is attached to (or, when that
    fails, asked about again every ``STATUS_POLL_S`` s); a finished one's results are shown. Skipped
    without the worker (no one-shot form) or when afv has no such command."""
    if not SESSION.open or not RUNQ["status_cmd"] or not ops.use_serve() or jobs.busy("bstatus"):
        return None
    project = SESSION.project

    def done(job, res):
        if SESSION.project != project:
            return
        view = core.batch_view(res)
        if view["running"]:
            RUNQ["batch"] = view
            if jobs.busy("batch"):
                return
            RUNQ["outcomes"] = view["outcomes"] or RUNQ["outcomes"]
            RUNQ["current"] = {"shot": view["shot"], "stage": view["stage"], "frac": view["frac"],
                               "index": view["index"], "total": None}
            RUNQ["summary"] = (f"batch {view['id'] or ''} running in WSL: shot {view['shot'] or '-'}"
                               + (f", {view['stage']}" if view["stage"] else "")
                               + f"; {len(view['outcomes'])} shots done")
            if RUNQ.get("no_attach") != view["id"]:
                _stream_batch(context, {"project": project}, "Run queue (attached)", None)
            elif poll and not bpy.app.timers.is_registered(_poll_status):
                bpy.app.timers.register(_poll_status, first_interval=STATUS_POLL_S)
        else:
            was = bool((RUNQ.get("batch") or {}).get("running"))
            RUNQ["batch"], RUNQ["current"] = None, None
            if view["id"] and not jobs.busy("batch"):
                _remember(view)
                if was:
                    refresh(context)
        jobs.redraw()

    def failed(job, msg):
        if _unknown(msg):
            RUNQ["status_cmd"] = False  # an afv from before the detached batch
        else:
            jobs.set_status(f"batch status: {_short(msg)}", "warning")

    return jobs.submit("bstatus", "batch_status", {"project": project}, "batch status", done, failed,
                       root_win=SESSION.root_win)


def _poll_status():
    if detached() and SESSION.open:
        batch_status()
    return None


def _never_ran(outcomes) -> list[str]:
    ran = {str(o.get("shot")) for o in outcomes}
    return [sid for sid in core.queue_check(SESSION.all_rows())[0] + [m for m, _ in core.queue_check(
        SESSION.all_rows())[1]] if sid not in ran]


def _stopped_line(view: dict) -> str:
    left = _never_ran(view["outcomes"])
    n = len(view["outcomes"])
    ran = (f"{n} shot{'s' if n != 1 else ''} ran ({core.outcome_counts(view['outcomes'])})" if n
           else "no shot had finished")
    plural = "s" if len(left) != 1 else ""
    return (f"batch {view.get('id') or ''} stopped: {ran}; {len(left)} queued shot{plural} did not run "
            "(they stay queued)")


def stop_batch(context=None):
    """Stop the project's batch: ``batch_cancel`` (it runs detached), else cancel the request (an older
    afv, or one-shot afv commands)."""
    j = jobs.job("batch")
    if (j is not None and j.oneshot) or not RUNQ["status_cmd"] or not ops.use_serve():
        jobs.cancel("batch")
        return None
    if j is not None:
        j.message = "stopping (the running shot stops at its next step)..."
    jobs.set_status("Run queue: stopping the batch...")

    def done(job, res):
        view = core.batch_view(res)
        RUNQ["batch"], RUNQ["current"] = None, None
        if not res.get("cancelled") and not view["running"]:
            jobs.set_status("Run queue: no batch was running")
            return
        view["cancelled"] = True
        _remember(view)
        RUNQ["summary"] = _stopped_line(view)
        jobs.set_status(f"Run queue: {RUNQ['summary']}")
        refresh(context)

    def failed(job, msg):
        if _unknown(msg):
            RUNQ["status_cmd"] = False
            jobs.cancel("batch")
        else:
            jobs.set_status(f"stop batch: {_short(msg)}", "error")

    return jobs.submit("edit", "batch_cancel", {"project": SESSION.project}, "afv batch cancel", done, failed,
                       root_win=SESSION.root_win)


def run_queue(context, only_ready=False):
    """``batch`` of the queued shots in the background (afv reads the queue again after each shot). Refused
    (the shots are listed) while a queued shot has no pick, unless ``only_ready``: then exactly the picked
    ones run."""
    if running():
        raise ValueError("a batch is already running in WSL: shots you queue and pick now run in it")
    ready, missing = core.queue_check(SESSION.all_rows())
    RUNQ["missing"] = missing
    if not ready and not missing:
        raise ValueError("no queued shots: queue some first")
    if missing and not only_ready:
        jobs.set_status(f"{len(missing)} queued shot{'s' if len(missing) != 1 else ''} need their fighters "
                        "picked (listed under Run queue)", "warning")
        return None
    if not ready:
        raise ValueError("no queued shot has its fighters picked yet")
    to = props(context).run_to
    RUNQ["missing"] = []
    if missing:  # only the picked ones, exactly these
        args = {"project": _project_arg(), "to": to, "shots": ready}
        label = f"Run queue ({len(ready)} picked shots)"
    else:  # every queued shot, also those queued (and picked) while it runs
        args = {"project": _project_arg(), "to": to, "state": "queued"}
        label = "Run queue"
    return _stream_batch(context, args, label, len(ready))


def _stream_batch(context, args, label, total):
    """Start the batch (or attach to the running one: afv then ignores the arguments) and follow its
    stream: the shot and stage running, per-shot outcomes."""
    t0 = time.monotonic()
    RUNQ.update({"outcomes": [], "current": None, "summary": "", "started": None, "total": total})
    to = args.get("to") or "export"

    def event(job, ev):
        kind = ev.get("kind")
        if kind == "batch_started":
            RUNQ["batch"] = {**(RUNQ.get("batch") or {}), "id": ev.get("batch_id"), "log": ev.get("log"),
                             "running": True, "attached": ev.get("attached")}
        elif kind == "batch_start":
            if ev.get("to"):
                RUNQ["to"] = ev["to"]
        elif kind == "batch_shot":
            idx = int(ev.get("index", 0))
            queued = core.counts(SESSION.all_rows())["queued"]  # the batch re-reads the queue
            total = max(idx + 1, int(ev.get("total") or 0), queued, int(RUNQ.get("total") or 0))
            RUNQ["current"] = {"shot": ev.get("shot"), "index": idx, "total": total, "stage": ""}
            if RUNQ.get("started") is None:
                RUNQ["started"] = TIMES["batch_start"] = time.monotonic() - t0
        elif ev.get("event") == "progress" and RUNQ.get("current"):
            cur, f = RUNQ["current"], ev.get("frac")
            if ev.get("stage", "") != cur.get("stage"):
                cur["stage"], cur["frac"] = ev.get("stage", ""), 0.0
            if isinstance(f, int | float) and f >= 0:  # a heartbeat (no fraction) keeps the last one
                cur["frac"] = float(f)
        elif kind == "stage_start" and RUNQ.get("current"):
            RUNQ["current"]["stage"], RUNQ["current"]["frac"] = ev.get("stage", ""), 0.0
        elif kind == "shot_done":
            o = {k: ev.get(k) for k in ("shot", "outcome", "message")}
            RUNQ["outcomes"].append(o)
            SESSION.outcomes[str(o["shot"])] = o["outcome"]
            refresh(context)
        cur = RUNQ.get("current")
        if cur:
            st = cur.get("stage") or ""
            last = RUNQ.get("to") or to
            n = (STAGES.index(last) + 1) if last in STAGES else len(STAGES)
            si = STAGES.index(st) if st in STAGES else 0
            frac = cur.get("frac") if isinstance(cur.get("frac"), int | float) else 0.0
            job.frac = min(1.0, (cur["index"] + min(1.0, (si + frac) / n)) / max(1, cur["total"]))
            job.message = f"shot {cur['index'] + 1}/{cur['total']} ({cur['shot']}): {st} " + (
                f"{100 * frac:.0f}%" if frac else "")

    def done(job, res):
        RUNQ["batch"], RUNQ["current"] = None, None
        view = core.batch_view({**res, "finished": time.strftime("%Y-%m-%dT%H:%M")})
        _remember(view)
        jobs.set_status(f"Run queue: {RUNQ['summary']} in {res.get('elapsed_s', 0):.0f} s",
                        "warning" if res.get("exit_code") else "info")
        refresh(context)

    def failed(job, msg):
        RUNQ["current"] = None
        term = jobs.terminal_event(job) or {}
        rep = term.get("report") if isinstance(term.get("report"), dict) else {}
        if term.get("detached"):  # the worker restarts or goes away: the batch goes on; follow it again
            batch_status(context)
            return
        RUNQ["no_attach"] = (RUNQ.get("batch") or {}).get("id") or term.get("batch_id")
        RUNQ["batch"] = None
        if _unknown(msg):  # an afv whose batch cannot attach: ask batch_status now and then instead
            batch_status(context)
            return
        if rep.get("batch_id"):  # stopped (Stop batch) or ended without finishing (killed)
            view = core.batch_view(rep)
            view["cancelled"] = bool(term.get("cancelled"))
            view["interrupted"] = not view["cancelled"]
            _remember(view)
            if view["cancelled"]:
                RUNQ["summary"] = _stopped_line(view)
                jobs.set_status(f"Run queue: {RUNQ['summary']}")
            else:
                RUNQ["summary"] = f"{last_line(view)} ({_short(msg)})"
                jobs.set_status(f"Run queue: {_short(msg)}", "error")
            refresh(context)
            return
        RUNQ["summary"] = _short(msg)
        jobs.set_status(f"Run queue: {_short(msg)}", "error")
        refresh(context)
        batch_status(context)

    return jobs.submit("batch", "batch", args, label, done, failed, event, root_win=SESSION.root_win)


class AFV_OT_browser_run(_Op, bpy.types.Operator):
    """Run every queued shot (afv batch, in WSL, shot after shot; closing Blender does not stop it). Checks
    first that every queued shot has its fighters picked"""

    bl_idname = "afv.browser_run"
    bl_label = "Run queue"

    only_ready: BoolProperty(name="Only the picked ones", default=False, options={"SKIP_SAVE"},
                             description="Run exactly the queued shots that have their fighters picked now; "
                                         "the others stay queued")

    @classmethod
    def poll(cls, context):
        return _need_project(cls, context) and not running()

    def execute(self, context):
        try:
            run_queue(context, self.only_ready)
        except (ValueError, RuntimeError) as e:
            return self.fail(e)
        return {"FINISHED"}


class AFV_OT_browser_stop_batch(_Op, bpy.types.Operator):
    """Stop the project's batch in WSL (afv batch cancel): the shot running now stops, the others stay
    queued"""

    bl_idname = "afv.browser_stop_batch"
    bl_label = "Stop batch"

    @classmethod
    def poll(cls, context):
        return _need_project(cls, context) and running()

    def execute(self, context):
        try:
            stop_batch(context)
        except (ValueError, RuntimeError) as e:
            return self.fail(e)
        return {"FINISHED"}


# --- import take ------------------------------------------------------------------------------------------
def _mark(shot: str) -> dict:
    return {"root_win": SESSION.root_win, "shot": str(shot)}


def take_scene(shot: str):
    """The scene that holds (or was made for) this shot's take, by its mark."""
    want = _mark(shot)
    for sc in bpy.data.scenes:
        try:
            if json.loads(sc.get(TAKE_MARK) or "null") == want:
                return sc
        except ValueError:
            continue
    return None


TAKE_READY: dict = {}  # a take written while the owner looked elsewhere: {"shot", "scene"} (Show take)


def _window_scene():
    wm = bpy.context.window_manager
    return wm.windows[0].scene if wm is not None and wm.windows else None


def _show(scene) -> None:
    wm = bpy.context.window_manager
    if wm is not None and wm.windows:
        wm.windows[0].scene = scene
    with ops._override():
        ops._look_through_camera(bpy.context, scene)
    TAKE_READY.clear()


def import_into(shot: str, path, switch: bool = True) -> bpy.types.Scene:
    """Build the take at ``path`` into the shot's own scene (made now when missing); show it when
    ``switch``, else offer **Show take**."""
    scene = take_scene(shot)
    if scene is None:
        scene = bpy.data.scenes.new(f"AFV {shot}")
        scene[TAKE_MARK] = json.dumps(_mark(shot))
    info = builder.import_take(path, scene)
    ops._after_import(scene)
    with ops._override():
        ops._undo_push(f"import take {shot}")
    what = (f"take {info['take_id']} in scene {scene.name}: {len(info['fighters'])} fighters, frames "
            f"{info['frame_start']}-{info['frame_end']}")
    if switch:
        _show(scene)
        jobs.set_status(f"{what} (Show shot browser comes back)")
    else:
        TAKE_READY.update({"shot": shot, "scene": scene.name})
        jobs.set_status(f"{what} is ready: Show take")
    return scene


def open_take(context, shot: str):
    """The shot's take: its marked scene if it holds it; else the package (refreshed by afv when missing or
    older than the shot's results) built into that scene. Returns (scene or None while afv works, made)."""
    sc = take_scene(shot)
    if sc is not None and builder.scene_take(sc):
        _show(sc)
        return sc, False
    shot_dir = Path(SESSION.root_win) / "shots" / shot
    try:
        path = takeio.find_take_json(shot_dir)
        stale = takeio.stale_sources(takeio.load_take(path))
    except takeio.TakeError:
        path, stale = None, ["no take yet"]
    if path is not None and not stale:
        return import_into(shot, path), True
    origin = _window_scene()
    origin = origin.name if origin is not None else None

    def done(job, res):
        here = _window_scene()  # the owner moved to another scene meanwhile: don't pull them away
        try:
            import_into(shot, takeio.find_take_json(shot_dir),
                        switch=here is None or here.name == origin)
        except (takeio.TakeError, builder.BuildError, OSError) as e:
            jobs.set_status(f"take {shot}: {_short(e)}", "error")

    def failed(job, msg):
        jobs.set_status(f"take {shot}: {_short(msg)}", "error")

    jobs.submit("take", "take", {"project": _project_arg(), "shot": shot}, f"afv take {shot}", done, failed,
                root_win=SESSION.root_win)
    jobs.set_status(f"writing the take of shot {shot} in WSL; it opens in its own scene when done")
    return None, False


class AFV_OT_browser_show_take(_Op, bpy.types.Operator):
    """Show the take that was written while you were in another scene"""

    bl_idname = "afv.browser_show_take"
    bl_label = "Show take"

    @classmethod
    def poll(cls, context):
        return bool(TAKE_READY) and TAKE_READY.get("scene") in bpy.data.scenes

    def execute(self, context):
        _show(bpy.data.scenes[TAKE_READY["scene"]])
        return {"FINISHED"}


class AFV_OT_browser_take(_Op, bpy.types.Operator):
    """Open the shot's take in a scene of its own ("AFV <shot>"; written in WSL first when needed); Show
    shot browser comes back"""

    bl_idname = "afv.browser_take"
    bl_label = "Import take"

    shot: StringProperty(description="Shot id (default: the selected shot)")

    @classmethod
    def poll(cls, context):
        return _need_project(cls, context) and not jobs.busy("take")

    def execute(self, context):
        shot = self.shot or props(context).shot
        if shot not in SESSION.rows:
            return self.fail("select a shot first")
        try:
            open_take(context, shot)
        except (RuntimeError, ValueError, takeio.TakeError, builder.BuildError) as e:
            return self.fail(e)
        return {"FINISHED"}


CLASSES = (AFV_OT_browser_run, AFV_OT_browser_stop_batch, AFV_OT_browser_show_take, AFV_OT_browser_take)
