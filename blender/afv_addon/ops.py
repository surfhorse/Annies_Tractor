"""Operators of the AFV panel: Import take, Re-import, Next/Previous issue, Export, Cancel; and the
job machinery the cleanup tools (``cleanup.py``) share.

Anything that runs ``afv`` goes through one background job at a time: a request to the session's
``afv serve`` worker (``serve.Request``; the worker is started on first use and restarted when it
died) or, as the fallback, a one-shot ``afv ... --json`` command (``wsl.Job``). The fallback is used
when the worker is switched off in the preferences or could not start (e.g. an afv without
``serve``): ``submit`` hands the same request to ``serve.cli_args``, and an ``edit`` is then written
by the add-on itself (``edits.write_local``). A persistent timer polls the job and applies the result
on the main thread (then pushes an undo step), so Blender never blocks. Opening another file while a
job runs keeps the timer; the finished job's result is then not applied to the new file. Cancel and
unregister always clear the job (the worker gets a ``cancel`` request).

``run_chain`` runs several requests one after the other (e.g. edit -> run -> take) and then a local
step (re-import), timing each; a run paused for tracking hints offers "Run anyway" (``--yes``).
"""

from __future__ import annotations

import os
import shutil
import time
import traceback
from pathlib import Path

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, IntProperty, StringProperty

from . import builder, edits, serve, takeio, wsl

POLL_S = 0.1


class _State:
    job = None  # wsl.Job | serve.Request | LocalJob
    on_done = None
    on_fail = None
    undo: bool = True  # push an undo step when the job's result is applied
    status: str = ""
    error: bool = False
    level: str = "info"  # info | warning (e.g. stale results) | error
    loads: int = 0  # files loaded since register (load_post)
    job_loads: int = 0  # ... when the job started
    review: dict | None = None  # a paused Export / run waiting for "Export anyway" / "Run anyway"
    serve: serve.ServeClient | None = None  # the session's afv serve worker
    serve_default: bool = True  # use the worker when the add-on has no preferences (tests)
    serve_off: str = ""  # why the worker is not used this session ("" = it is)
    failed: dict | None = None  # the last cleanup fix that failed: {"label", "step", "message"}
    timings: dict = {}  # noqa: RUF012  kind -> {"label", "total", "steps": [(name, s)]} of the last chain


STATE = _State()


def prefs(context=None):
    context = context or bpy.context
    addon = context.preferences.addons.get(__package__)
    return addon.preferences if addon is not None else None


def _redraw():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type in ("VIEW_3D", "DOPESHEET_EDITOR", "NLA_EDITOR"):
                area.tag_redraw()


def set_status(text, error=False, warning=False):
    STATE.status = text
    STATE.error = error
    STATE.level = "error" if error else "warning" if warning else "info"
    _redraw()


def busy():
    """A job is running or its result is still to be applied (and the timer that applies it lives)."""
    job = STATE.job
    if job is None:
        return False
    return not job.finished or bpy.app.timers.is_registered(_poll)


def _override():
    """A window context for work done from a timer (bpy.ops need one)."""
    wm = bpy.context.window_manager
    win = wm.windows[0] if wm is not None and wm.windows else None
    if win is None:
        return bpy.context.temp_override()
    area = next((a for a in win.screen.areas if a.type == "VIEW_3D"), None)
    return bpy.context.temp_override(window=win, screen=win.screen, area=area)


def cancel_job():
    """Stop and forget the running job (Cancel, unregister); a worker request gets ``cancel``."""
    job = STATE.job
    STATE.job, STATE.on_done, STATE.on_fail = None, None, None
    if job is not None:
        job.cancel()
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)


def _poll():
    job = STATE.job
    if job is None:
        return None
    try:
        for ev in job.drain():
            line = wsl.status_line(ev)
            if line:
                set_status(f"{job.label}: {line}")
    except Exception as e:  # never let the timer die with a job pending
        set_status(f"{job.label}: {type(e).__name__}: {e}", error=True)
    if not job.done:
        return POLL_S
    _apply(job)
    return POLL_S if STATE.job is not None else None  # a callback may have started the next job


def _apply(job):
    on_done, on_fail, undo = STATE.on_done, STATE.on_fail, STATE.undo
    STATE.job, STATE.on_done, STATE.on_fail = None, None, None
    try:
        if STATE.loads != STATE.job_loads:
            set_status(f"{job.label} finished after another file was opened: result not applied", error=True)
            return
        if getattr(job, "unavailable", False) and not job.cancelled:  # no worker: same request, one-shot
            STATE.serve_off = f"afv serve not available ({job.error}); using one-shot afv commands"
            print(f"AFV: {STATE.serve_off}")
            with _override():
                _start(_oneshot(job.cmd, job.args, job.label), on_done, on_fail, undo)
            set_status(f"{job.label}: afv serve not available, running it as a one-shot command",
                       warning=True)
            return
        if job.ok:
            if on_done is not None:
                with _override():
                    on_done(job)
                    if undo:
                        _undo_push(job.label)
        elif on_fail is not None:
            with _override():
                on_fail(job, "needs_review" if job.review() is not None else job.failure())
                if undo:
                    _undo_push(job.label)  # on_fail may still change the scene (import with a warning)
        else:
            set_status(f"{job.label} failed: {job.failure()}", error=True)
    except Exception as e:
        traceback.print_exc()
        set_status(f"{job.label}: {type(e).__name__}: {e}", error=True)
    finally:
        _redraw()


def _undo_push(message):
    try:
        bpy.ops.ed.undo_push(message=f"AFV: {message}")
    except RuntimeError:  # no undo stack (background) or wrong context
        pass


def _wsl_settings(context=None):
    p = prefs(context)
    return (p.wsl_distro if p else wsl.DEFAULT_DISTRO), (p.afv_command if p else wsl.DEFAULT_AFV)


def _start(job, on_done, on_fail, undo=True):
    STATE.job = job
    STATE.on_done, STATE.on_fail, STATE.undo = on_done, on_fail, undo
    STATE.job_loads = STATE.loads
    if not bpy.app.timers.is_registered(_poll):
        bpy.app.timers.register(_poll, first_interval=POLL_S, persistent=True)


def run_afv(args, label, on_done, context=None, on_fail=None):
    """Start the one-shot ``afv <args> --json`` in WSL. On the main thread afterwards:
    ``on_done(job)`` when it succeeds, else ``on_fail(job, message)`` (default: the status line)."""
    if busy():
        raise RuntimeError(f"afv is busy ({STATE.job.label})")
    distro, afv = _wsl_settings(context)
    STATE.review = None
    _start(wsl.Job(wsl.afv_argv([*args, "--json"], distro=distro, afv=afv), label).start(), on_done, on_fail)
    set_status(f"{label}: started")


class LocalJob:
    """A step done right here (the one-shot fallback of ``edit``), with the job interface."""

    def __init__(self, label, fn, cmd="", args=None):
        self.label, self.cmd, self.args = label, cmd, args or {}
        self.events, self.text = [], []
        self.started = time.monotonic()
        self.cancelled = False
        self.error = None
        self.value = None
        try:
            self.value = fn()
        except (edits.EditError, OSError, ValueError) as e:
            self.error = str(e)
        self.ended = time.monotonic()
        self.finished = self.done = True

    def drain(self):
        return []

    def join(self, timeout=None):
        return None

    def cancel(self):
        self.cancelled = True

    @property
    def ok(self):
        return self.error is None and not self.cancelled

    def result(self):
        return {"event": "result", "outcome": "ok", "path": str(self.value) if self.value else None}

    def review(self):
        return None

    def failure(self):
        return self.error or "cancelled"


def use_serve(context=None) -> bool:
    """Requests go to the afv serve worker (else: one-shot afv commands)."""
    p = prefs(context)
    on = p.use_serve if p is not None else STATE.serve_default
    return bool(on) and not STATE.serve_off


def serve_client():
    if STATE.serve is None:
        STATE.serve = serve.ServeClient(lambda: serve.serve_argv(*_wsl_settings()))
    return STATE.serve


def shutdown_serve(wait=0.0):
    client, STATE.serve = STATE.serve, None
    if client is not None:
        client.shutdown(wait)


def worker_status() -> str:
    if not use_serve():
        return STATE.serve_off or "one-shot afv commands (afv serve is off in the preferences)"
    client = STATE.serve
    return f"afv serve: {client.status() if client is not None else 'starts on first use'}"


def _edit_shot_dir(args):
    shot_dir = args.get("shot_dir")
    if not shot_dir:
        raise edits.EditError("the shot folder of this take is not known; cannot write the edit")
    return shot_dir


ONESHOT_ARGS = {"presets": ["retarget", "--list-presets"]}  # requests without a shot (serve.cli_args: shots)


def _oneshot(cmd, args, label):
    if cmd == "edit":
        return LocalJob(label, lambda: edits.write_local(_edit_shot_dir(args), args["name"], args["data"]),
                        cmd, args)
    distro, afv = _wsl_settings()
    argv = list(ONESHOT_ARGS[cmd]) if cmd in ONESHOT_ARGS else serve.cli_args(cmd, args)
    job = wsl.Job(wsl.afv_argv([*argv, "--json"], distro=distro, afv=afv), label)
    job.cmd, job.args = cmd, args
    return job.start()


def submit(cmd, args, label, on_done, context=None, on_fail=None, undo=True):
    """Run one request (``serve`` protocol: run, take, export, edit, presets) on the worker, or one-shot when
    the worker is off or unavailable. ``args`` may carry ``shot_dir`` (a Windows path, for the local
    ``edit`` fallback; not sent to the worker). Callbacks as ``run_afv``."""
    if busy():
        raise RuntimeError(f"afv is busy ({STATE.job.label})")
    STATE.review = None
    job = None
    if use_serve(context):
        try:
            job = serve_client().submit(cmd, {k: v for k, v in args.items() if k != "shot_dir"}, label)
            job.args = args  # what the one-shot fallback needs
        except serve.ServeError as e:
            STATE.serve_off = f"afv serve not available ({e}); using one-shot afv commands"
    if job is None:
        job = _oneshot(cmd, args, label)
    _start(job, on_done, on_fail, undo)
    set_status(f"{label}: started")
    return job


def run_chain(label, steps, finish, kind, context=None, rerun=None):
    """Run ``steps`` ([(name, cmd, args[, on_done[, on_fail]])]) one after the other, then ``finish()`` (main
    thread, one
    undo step for the whole chain). Times each step into ``STATE.timings[kind]`` and the status line.
    A run paused on tracking hints keeps ``rerun(yes=True)`` for "Run anyway"."""
    t0 = time.monotonic()
    times = []

    def step(i):
        name, cmd, args, *hooks = steps[i]  # (on success, on failure / pause): called with the job
        hook, fail_hook = [*hooks, None, None][:2]
        ts = time.monotonic()
        last = i == len(steps) - 1

        def done(job):
            times.append((name, time.monotonic() - ts))
            if hook is not None:
                hook(job)
            if not last:
                step(i + 1)
                return
            tf = time.monotonic()
            msg = finish(job)
            times.append(("re-import", time.monotonic() - tf))
            total = time.monotonic() - t0
            STATE.timings[kind] = {"label": label, "total": total, "steps": list(times)}
            detail = ", ".join(f"{n} {t:.1f} s" for n, t in times)
            set_status(f"{msg or label} in {total:.1f} s ({detail})")

        def failed(job, msg):
            if fail_hook is not None:
                fail_hook(job, msg)
            review = job.review()
            if review is not None and rerun is not None:
                STATE.review = {**review, "kind": "run", "label": label, "rerun": rerun,
                                "shot_id": args.get("shot", "")}
                set_status(f"{label}: paused, tracking QC hints to review (see the panel)", error=True)
                return
            STATE.failed = {"label": label, "step": name, "message": msg}
            set_status(f"{label}: {name} failed: {msg}", error=True)

        STATE.failed = None
        submit(cmd, args, f"{label}: {name}", done, context, on_fail=failed, undo=last)

    step(0)


@persistent
def on_load_post(*_args):
    STATE.loads += 1
    STATE.review = None


@persistent
def on_save_pre(filepath="", *_args):
    builder.relativize_records(filepath or bpy.data.filepath)  # what the saved file stores


@persistent
def on_save_post(filepath="", *_args):
    """After Save Copy the open file is still the old one: give it back its own relative paths."""
    here = bpy.data.filepath
    def norm(p):
        return os.path.normcase(os.path.abspath(p))

    if here and filepath and norm(filepath) != norm(here):
        builder.relativize_records(here)


# --- where a take lives ---------------------------------------------------------------------------
def target(rec):
    """take.json, shot id, the project's WSL path and the shot folder for a scene record, all
    derived from where the take is now (so a moved project or .blend still works)."""
    tj = builder.resolve_take_json(rec)
    loc = takeio.locate(tj)
    project_wsl = ""
    if loc["project_dir"] is not None:
        try:
            project_wsl = wsl.to_wsl_path(str(loc["project_dir"]))
        except ValueError:
            project_wsl = ""
    project_wsl = project_wsl or loc["project_wsl"]
    if not project_wsl:
        raise builder.BuildError(f"cannot tell which afv project {tj} belongs to")
    return {"take_json": tj, "shot_id": loc["shot_id"] or rec["shot_id"], "project": project_wsl,
            "shot_dir": loc["shot_dir"]}


def take_request(t, preset="", check_plan=True):
    """The ``take`` request for a target (``target``). ``check_plan=False`` (right after a run) skips
    asking which stages ``afv run`` would redo (about 2 s)."""
    return {"project": t["project"], "shot": t["shot_id"], **({"preset": preset} if preset else {}),
            **({} if check_plan else {"check_plan": False})}


def _look_through_camera(context, scene):
    if scene.camera is None or context.screen is None:
        return
    for area in context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.region_3d.view_perspective = "CAMERA"


def _after_import(scene):
    """The cleanup panels' defaults for the take just imported: the shot's smoothing settings, an
    edit range inside the take."""
    from . import cleanup  # (imports ops)

    rec = builder.scene_take(scene)
    if rec is None:
        return
    cleanup.load_smoothing(scene, rec)
    props, a, b = scene.afv, rec["frame_start"], rec["frame_end"]
    if not (a <= props.range_start <= props.range_end <= b) or props.range_end == props.range_start == 0:
        props.range_start, props.range_end = a, min(b, a + 59)


def _do_import(path, context, warnings=()):
    info = builder.import_take(path, context.scene)
    _after_import(context.scene)
    _look_through_camera(context, context.scene)
    n = info["spans"]
    msg = (f"imported {info['take_id']} [{info['instance']}]: {len(info['fighters'])} fighters, frames "
           f"{info['frame_start']}-{info['frame_end']}, {n} QC span{'s' if n != 1 else ''}")
    notes = [*warnings] + ([info["stale_summary"]] if info["stale_summary"] else [])
    for line in info["stale"]:
        print(f"AFV: {info['take_id']}: {line}")  # details in the console; the status has the summary
    if notes:
        msg += " | " + "; ".join(notes)
    set_status(msg, warning=bool(notes))


class AFV_OT_import_take(bpy.types.Operator):
    """Import an afv take: pick the shot folder (projects/<name>/shots/<id>) or its take folder.
    Each import is a separate copy; a missing or out-of-date take is (re)written in WSL first"""

    bl_idname = "afv.import_take"
    bl_label = "Import take"
    bl_options = {"REGISTER", "UNDO"}  # noqa: RUF012 (Blender's API)

    directory: StringProperty(subtype="DIR_PATH")
    filter_folder: BoolProperty(default=True, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return not busy()

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        folder = Path(self.directory)
        try:
            path = takeio.find_take_json(folder)
        except takeio.TakeError:
            path = None
        try:
            if path is not None:
                take = takeio.load_take(path)
                stale = takeio.stale_sources(take)
                if not stale:
                    _do_import(path, context)
                    self.report({"INFO"}, STATE.status)
                    return {"FINISHED"}
                t = target(take.record())
                preset = (take.meta.get("rig") or {}).get("preset", "")
                submit("take", take_request(t, preset), f"afv take {t['shot_id']} (refresh)",
                       lambda job: _do_import(path, bpy.context), context,
                       on_fail=lambda job, msg: _do_import(
                           path, bpy.context, [*stale, f"refresh failed: {msg}"]))
                self.report({"WARNING"}, "the take is older than the shot's results; refreshing it in WSL "
                                         "(imports when done): " + "; ".join(stale))
                return {"FINISHED"}
            shot = takeio.shot_dir_of(folder)
            if shot is None:
                self.report({"ERROR"}, f"{folder} is not a shot folder (no shot.json) and has no take")
                return {"CANCELLED"}
            project = wsl.to_wsl_path(str(shot.parent.parent))  # no package yet: write it, then import
            submit("take", {"project": project, "shot": shot.name}, f"afv take {shot.name}",
                   lambda job: _do_import(shot, bpy.context), context)
        except (takeio.TakeError, builder.BuildError, ValueError, RuntimeError) as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, "writing the take in WSL (afv take); it imports when done")
        return {"FINISHED"}


class AFV_OT_reimport(bpy.types.Operator):
    """Re-import the selected take (or the scene's last one): replaces only the solved motion (base
    NLA strip), camera and markers; the tweak track and your keys stay"""

    bl_idname = "afv.reimport"
    bl_label = "Re-import"
    bl_options = {"REGISTER", "UNDO"}  # noqa: RUF012 (Blender's API)

    @classmethod
    def poll(cls, context):
        return not busy() and builder.record_for(context.scene, context.active_object) is not None

    def execute(self, context):
        rec = builder.record_for(context.scene, context.active_object)
        scene_name = context.scene.name

        def apply(job=None):
            scene = bpy.data.scenes.get(scene_name)
            if scene is None:
                raise builder.BuildError(f"scene {scene_name!r} is gone; nothing re-imported")
            info = builder.reimport_take(scene, rec=rec)
            set_status(f"re-imported {info['take_id']} [{info['instance']}] ({info['spans']} QC spans)")

        try:
            p = prefs(context)
            if p is not None and p.refresh_on_reimport:
                t = target(rec)
                submit("take", take_request(t, rec.get("preset", "")), f"afv take {t['shot_id']}", apply,
                       context)
                self.report({"INFO"}, "refreshing the take in WSL; re-imports when done")
                return {"FINISHED"}
            apply()
        except (takeio.TakeError, builder.BuildError, ValueError, RuntimeError) as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, STATE.status)
        return {"FINISHED"}


class AFV_OT_issue(bpy.types.Operator):
    """Jump to the next (or previous) QC issue on the confidence strip"""

    bl_idname = "afv.issue"
    bl_label = "Issue"

    direction: IntProperty(default=1)

    @classmethod
    def poll(cls, context):
        return bool(builder.scene_spans(context.scene))

    def execute(self, context):
        scene = context.scene
        span = takeio.next_span(builder.scene_spans(scene), scene.frame_current, self.direction)
        if span is None:
            return {"CANCELLED"}
        scene.frame_set(int(span["start"]))
        self.report({"INFO"}, f"{span['label']} (frames {span['start']}-{span['end']})")
        return {"FINISHED"}


def copy_exports(job, shot_dir, copy_dir):
    """Export files named in afv's result (relative to the shot's exports/), copied to ``copy_dir``."""
    res = job.result() or {}
    files = [f for f in res.get("exports", []) if f]
    copied = []
    if copy_dir and shot_dir is not None:
        exports = Path(shot_dir) / "exports"
        dest = Path(bpy.path.abspath(copy_dir))
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            src = exports / f
            if src.suffix.lower() in (".fbx", ".glb", ".gltf", ".bvh") and src.is_file():
                shutil.copy2(src, dest / src.name)
                copied.append(src.name)
    return files, copied


class AFV_OT_export(bpy.types.Operator):
    """Export the solved take with the chosen engine preset (runs 'afv export' in WSL in the
    background; files go to the shot's exports folder, and are copied to 'Copy to' if set)"""

    bl_idname = "afv.export"
    bl_label = "Export"

    yes: BoolProperty(name="Export anyway", default=False, options={"HIDDEN", "SKIP_SAVE"},
                      description="Go on past tracking QC hints (afv export --yes)")
    instance: StringProperty(options={"HIDDEN", "SKIP_SAVE"}, description="Take copy to export (reviewed)")
    reviewed_preset: StringProperty(options={"HIDDEN", "SKIP_SAVE"},
                                    description="Preset (default: the panel's)")

    @classmethod
    def poll(cls, context):
        return not busy() and builder.record_for(context.scene, context.active_object) is not None

    def execute(self, context):
        if self.instance:  # "Export anyway": exactly what was reviewed
            rec = builder.record_by_instance(context.scene, self.instance)
            if rec is None:
                self.report({"ERROR"}, "the reviewed take copy is no longer in this scene")
                return {"CANCELLED"}
        else:
            rec = builder.record_for(context.scene, context.active_object)
        props = context.scene.afv
        preset, copy_dir = self.reviewed_preset or props.export_preset, props.export_copy_dir
        try:
            t = target(rec)
        except (builder.BuildError, takeio.TakeError) as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        def done(job):
            files, copied = copy_exports(job, t["shot_dir"], copy_dir)
            msg = f"exported {len(files)} files with {preset}"
            if copied:
                msg += f"; copied {len(copied)} to {copy_dir}"
            set_status(msg)

        def failed(job, msg):
            review = job.review()
            if review is not None:
                STATE.review = {**review, "shot_id": t["shot_id"], "instance": rec.get("instance", ""),
                                "preset": preset}
                set_status(f"{job.label}: paused, tracking QC hints to review (see the panel)", error=True)
            else:
                set_status(f"{job.label} failed: {msg}", error=True)

        args = {"project": t["project"], "shot": t["shot_id"], "preset": preset, "yes": bool(self.yes)}
        try:
            submit("export", args, f"afv export {t['shot_id']}", done, context, on_fail=failed)
        except (ValueError, RuntimeError) as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, f"exporting {t['shot_id']} with {preset} in the background")
        return {"FINISHED"}


class AFV_OT_dismiss_review(bpy.types.Operator):
    """Hide the tracking review (fix the tracking with afv pick, or use Export anyway)"""

    bl_idname = "afv.dismiss_review"
    bl_label = "Dismiss"

    def execute(self, context):
        STATE.review = None
        _redraw()
        return {"FINISHED"}


class AFV_OT_cancel(bpy.types.Operator):
    """Stop the running afv command"""

    bl_idname = "afv.cancel"
    bl_label = "Cancel"

    @classmethod
    def poll(cls, context):
        return STATE.job is not None

    def execute(self, context):
        label = STATE.job.label if STATE.job is not None else "afv"
        cancel_job()
        set_status(f"{label}: cancelled")
        return {"FINISHED"}


class AFV_OT_go_on(bpy.types.Operator):
    """Run the paused re-solve again past the tracking QC hints (afv run --yes)"""

    bl_idname = "afv.go_on"
    bl_label = "Run anyway"

    @classmethod
    def poll(cls, context):
        return not busy() and STATE.review is not None and STATE.review.get("rerun") is not None

    def execute(self, context):
        rerun = STATE.review["rerun"]
        STATE.review = None
        try:
            rerun(yes=True)
        except (builder.BuildError, takeio.TakeError, edits.EditError, ValueError, RuntimeError) as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class AFV_OT_restart_worker(bpy.types.Operator):
    """Stop the afv serve worker; the next request starts a new one (and retries afv serve after it
    was found unavailable)"""

    bl_idname = "afv.restart_worker"
    bl_label = "Restart worker"

    @classmethod
    def poll(cls, context):
        return not busy()

    def execute(self, context):
        shutdown_serve()
        STATE.serve_off = ""
        set_status("afv serve: stopped; the next request starts it again")
        return {"FINISHED"}


CLASSES = (AFV_OT_import_take, AFV_OT_reimport, AFV_OT_issue, AFV_OT_export, AFV_OT_dismiss_review,
           AFV_OT_cancel, AFV_OT_go_on, AFV_OT_restart_worker)
