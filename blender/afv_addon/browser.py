"""The shot browser: AFV > **Shots** (3D viewport sidebar).

**New project from video** / **Open project** make (or read) an afv project through the ``afv serve``
worker and show it in its own scene, "AFV Shots": the source video as the camera's background (a movie
clip; the 3D viewport looks through the camera), one timeline marker per detected cut (``media/cuts.json``,
at the first frame of the segment after it; **Cut markers** hides them) and a strip of the shots under the
timeline's frame numbers. Scene frame f is source frame f.

The list (sub-panel **Shot list**): thumbnail, id, start, length, cut confidence, state, people found
(``*`` = fighters likely, a hint: sort by it, it never hides a shot), pick status and the last batch's
outcome; filter by state; a click selects a shot and jumps to it (the timeline's preview range becomes the
shot, so Space plays it). Sub-panel **Shot**: Set in / Set out from the playhead, speed, notes, Queue /
Skip / Candidate (then on to the next shot), Pick fighters, Import take, Add shot here. Keys over the 3D
view or timeline of the AFV Shots scene: ``browser_keys.py``. Pick fighters: ``browser_pick.py``; Run
queue and Import take: ``browser_run.py``; panels: ``browser_ui.py``.

Every change goes to afv (serve ``shot_set``, ``shot_add``, ``pick``; one-shot commands as the fallback)
and the row afv sends back replaces the list's row at once. Requests run in the background
(``browser_jobs``).

The project's truth is afv's (``SESSION``, the rows afv sent); the scene only shows it. Opening a project
is one undo step, and after any undo / redo (``on_undo``) the scene and the panel's fields are put back in
line with ``SESSION`` (an undo past the opening closes the browser), so Ctrl+Z never shows something the
project does not hold.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty

from . import browser_core as core
from . import browser_draw as draw
from . import browser_jobs as jobs
from . import ops, wsl

SESSION = draw.SESSION
PICK = draw.PICK
SCENE_NAME = "AFV Shots"
MARKERS_PROP = "afv_browser_markers"
STAGES = ("ingest", "detect_track", "camera", "pose3d", "pair", "contacts", "smooth", "hands_face", "export")
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".ts", ".wmv")
SPEED_SETTLE_S = 0.6  # a Speed change is sent once the slider has rested this long
_FILLING = {"on": False}  # the panel's fields are being filled from a row: don't send them to afv
TIMES: dict = {}  # last durations (s): open, new, set, state, add, pick_prompt, pick_save, scan, batch_start
_REFRESH = {"again": False}
_PENDING_SPEED: dict = {}  # shot -> speed, waiting for the slider to settle


# --- helpers ------------------------------------------------------------------------------------------
def props(context=None):
    scene = browser_scene() or (context or bpy.context).scene
    return scene.afv_shots


def browser_scene():
    return next((s for s in bpy.data.scenes if s.get(draw.SCENE_PROP) is not None), None)


def _project_arg() -> str:
    return SESSION.project


def _wsl(path: str) -> str:
    return wsl.to_wsl_path(path)


def _windows(path: str | None) -> str | None:
    if not path:
        return None
    if path.startswith("/"):
        return wsl.to_windows_path(path, ops._wsl_settings()[0])
    return path


def _row(shot: str | None) -> dict | None:
    return SESSION.rows.get(str(shot)) if shot else None


def selected_row(context=None) -> dict | None:
    return _row(props(context).shot) if SESSION.open else None


def _short(text, n=120) -> str:
    """A message for the panel: the project's paths dropped, CLI advice as the browser's buttons, cut."""
    text = core.humanize(str(text), (SESSION.root_win, SESSION.project))
    return text if len(text) <= n else text[:n - 3] + "..."


def _undo_push(message: str) -> None:
    with ops._override():
        ops._undo_push(message)


# --- the scene ------------------------------------------------------------------------------------------
def video_path(payload: dict) -> str | None:
    """The video to show: afv's proxy when it has one, else the source (a path Blender can open)."""
    root = payload.get("root_win") or ""
    if payload.get("proxy") and root:
        p = core.project_file(root, payload["proxy"])
        if p is not None and p.is_file():
            return str(p)
    src = payload.get("source") or {}
    return src.get("path_win") or _windows(src.get("path"))


def _clip_for(path: str):
    want = os.path.normcase(os.path.abspath(path))
    for c in bpy.data.movieclips:
        if os.path.normcase(os.path.abspath(bpy.path.abspath(c.filepath))) == want:
            return c
    return bpy.data.movieclips.load(path)


def _camera(scene):
    cam = bpy.data.objects.get(draw.CAMERA_NAME)
    if cam is None or cam.type != "CAMERA":
        cam = bpy.data.objects.new(draw.CAMERA_NAME, bpy.data.cameras.new(draw.CAMERA_NAME))
    if cam.name not in scene.collection.objects:
        scene.collection.objects.link(cam)
    scene.camera = cam
    # only a frame for the video: Blender's keys (I keys, S scales...) must never change it
    cam.lock_location = cam.lock_rotation = cam.lock_scale = (True, True, True)
    cam.lock_rotation_w = True
    cam.select_set(False)
    cam.hide_select = True
    return cam


def _set_background(cam, clip=None, image=None, clear=False):
    """The camera's background: the source clip, or (while picking) the prompt frame's image on top;
    ``clear``: nothing (the project's video is missing)."""
    data = cam.data
    data.show_background_images = True
    for bg in list(data.background_images):
        if clear or bg.source == "IMAGE" or (clip is not None and bg.clip != clip):
            data.background_images.remove(bg)
    if clip is not None and not any(bg.clip == clip for bg in data.background_images):
        bg = data.background_images.new()
        bg.source, bg.clip, bg.use_camera_clip = "MOVIE_CLIP", clip, False
        bg.clip_user.use_render_undistorted = False
        bg.frame_method, bg.alpha, bg.display_depth = "STRETCH", 1.0, "BACK"
    for bg in data.background_images:
        bg.show_background_image = image is None
    if image is not None:
        bg = data.background_images.new()
        bg.source, bg.image = "IMAGE", image
        bg.frame_method, bg.alpha, bg.display_depth = "STRETCH", 1.0, "BACK"
        bg.show_background_image = True


def _set_markers(scene, cuts):
    """One marker per cut (this add-on's only: markers you made stay)."""
    try:
        old = {tuple(m) for m in json.loads(scene.get(MARKERS_PROP, "[]"))}
    except (ValueError, TypeError):
        old = set()
    for m in list(scene.timeline_markers):
        if (m.name, m.frame) in old:
            scene.timeline_markers.remove(m)
    made = []
    for f, name in cuts:
        scene.timeline_markers.new(name, frame=int(f))
        made.append([name, int(f)])
    scene[MARKERS_PROP] = json.dumps(made)


def _markers_changed(self, context):
    scene = browser_scene()
    if scene is not None and SESSION.open:
        _set_markers(scene, SESSION.cuts if self.show_markers else [])


def _read_cuts(payload) -> dict | None:
    p = core.project_file(payload.get("root_win") or "", payload.get("cuts"))
    if p is None:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _areas(kind):
    wm = bpy.context.window_manager
    for win in (wm.windows if wm is not None else []):
        for area in win.screen.areas:
            if area.type == kind:
                yield win, area


def look_through_camera(scene) -> None:
    """The 3D viewport looks through the browser camera (the take tools' ``ops._look_through_camera``)."""
    with ops._override():
        ops._look_through_camera(bpy.context, scene)


def timeline_view_all(preview=False):
    """Frame the whole source in the Timeline / Dope Sheet (``preview``: the preview range, i.e. the
    selected shot)."""
    for win, area in _areas("DOPESHEET_EDITOR"):
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        try:
            with bpy.context.temp_override(window=win, area=area, region=region):
                if preview:
                    bpy.ops.anim.scene_range_frame()
                else:
                    bpy.ops.action.view_all()
        except (RuntimeError, TypeError, AttributeError):
            pass


def show_project(context, payload: dict, undo: bool = True) -> dict:
    """Build (or update) the browser scene for a shot list (one undo step); returns what it did."""
    t0 = time.monotonic()
    if SESSION.root_win and SESSION.root_win != payload.get("root_win"):
        _free_previews()  # another project: its thumbnails go
        SESSION.selected = ""
    selected = SESSION.selected
    SESSION.load(payload)
    SESSION.selected = selected if selected in SESSION.rows else ""
    scene = browser_scene() or bpy.data.scenes.new(SCENE_NAME)
    scene[draw.SCENE_PROP] = json.dumps({"project": payload.get("project"),
                                         "root_win": payload.get("root_win")})
    win = context.window or (bpy.context.window_manager.windows[0] if bpy.context.window_manager
                             and bpy.context.window_manager.windows else None)
    if win is not None and win.scene != scene:
        win.scene = scene
    fps = SESSION.fps
    scene.render.fps = max(1, round(fps))
    scene.render.fps_base = scene.render.fps / fps
    scene.frame_start, scene.frame_end = 0, max(0, SESSION.num_frames - 1)
    src = SESSION.source
    if src.get("width") and src.get("height"):
        scene.render.resolution_x, scene.render.resolution_y = int(src["width"]), int(src["height"])
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    cam = _camera(scene)
    path = video_path(payload)
    out = {"video": path, "clip": None, "markers": 0}
    if path and os.path.isfile(path):
        clip = _clip_for(path)
        clip.frame_start = 0  # scene frame f = source frame f
        clip.frame_offset = 0
        _set_background(cam, clip=clip)
        scene.active_clip = clip
        out["clip"] = clip.name
        SESSION.video_warning = ""
    else:  # no video: nothing of the previous project stays on screen; the warning stays in the panel
        _set_background(cam, clear=True)
        scene.active_clip = None
        SESSION.video_warning = (f"video not found: {path or '(none)'}; the list works, the timeline shows "
                                 "no video (move the video back, or afv init --rehash)")
    SESSION.cuts = core.cut_frames(_read_cuts(payload))
    p = scene.afv_shots
    _set_markers(scene, SESSION.cuts if p.show_markers else [])
    out["markers"] = len(SESSION.cuts)
    p.project = payload.get("root_win") or payload.get("project") or ""
    p.shot = SESSION.selected
    fill_fields(p, _row(SESSION.selected))
    look_through_camera(scene)
    scene.use_preview_range = False
    timeline_view_all()
    if undo:
        _undo_push("open shot browser")
    out["seconds"] = time.monotonic() - t0
    return out


def close_session() -> None:
    """Forget the open project here (nothing in afv changes)."""
    from .browser_run import RUNQ

    PICK.reset()
    SESSION.clear()
    _PENDING_SPEED.clear()
    _free_previews()
    RUNQ.update({"missing": [], "outcomes": [], "current": None, "summary": "", "batch": None,
                 "last": None, "status_cmd": True})


# --- thumbnails ----------------------------------------------------------------------------------------
_PREVIEWS: dict = {"coll": None, "keys": {}}


def thumb_icon(row: dict | None) -> int:
    """The preview icon of a row's scan thumbnail (0 when it has none)."""
    if not row or not SESSION.root_win:
        return 0
    rel = (row.get("scan") or {}).get("thumbnail")
    path = core.project_file(SESSION.root_win, rel)
    if path is None:
        return 0
    coll = _PREVIEWS["coll"]
    if coll is None:
        import bpy.utils.previews

        coll = _PREVIEWS["coll"] = bpy.utils.previews.new()
    ck = (str(path), SESSION.version)
    key = _PREVIEWS["keys"].get(ck)
    if key is None:
        try:
            st = os.stat(path)
        except OSError:
            return 0
        key = f"{path}|{st.st_mtime_ns}|{st.st_size}"
        if len(_PREVIEWS["keys"]) > 4096:
            _PREVIEWS["keys"].clear()
        _PREVIEWS["keys"][ck] = key
    pv = coll.get(key) or coll.load(key, str(path), "IMAGE")
    return pv.icon_id


def _free_previews():
    coll = _PREVIEWS["coll"]
    _PREVIEWS["coll"], _PREVIEWS["keys"] = None, {}
    if coll is not None:
        import bpy.utils.previews

        bpy.utils.previews.remove(coll)


def preview_count() -> int:
    coll = _PREVIEWS["coll"]
    return len(coll) if coll is not None else 0


# --- selection ------------------------------------------------------------------------------------------
def fill_fields(p, row):
    _FILLING["on"] = True
    try:
        sid = str((row or {}).get("id") or "")
        p.speed = float(_PENDING_SPEED.get(sid) or (row or {}).get("speed_factor") or 1.0)
        p.notes = str((row or {}).get("notes") or "")
    finally:
        _FILLING["on"] = False


def select(context, shot: str, jump: bool = True) -> None:
    """Make ``shot`` the selected shot; jump to it (preview range = the shot when Loop shot is on)."""
    scene = browser_scene()
    row = _row(shot)
    if scene is None or row is None:
        return
    p = scene.afv_shots
    SESSION.selected = p.shot = str(shot)
    fill_fields(p, row)
    vis = [str(r["id"]) for r in SESSION.visible(p.state_filter, p.order)]
    if str(shot) in vis:
        p.page = vis.index(str(shot)) // max(1, p.rows)
    if jump and core.has_range(row):
        a, b = core.shot_frames(row, SESSION.fps)
        scene.frame_set(a)
        if p.loop_shot:
            scene.use_preview_range = True
            scene.frame_preview_start, scene.frame_preview_end = a, b
            timeline_view_all(preview=True)
    jobs.redraw()


def order_ids(context=None, likely_only=False) -> list[str]:
    """The shot ids in the list's current filter and order (what Next / Previous walk through)."""
    p = props(context)
    rows = SESSION.visible(p.state_filter, p.order)
    return [str(r["id"]) for r in rows if not likely_only or core.likely(r)]


def _after_row(context, res, kind, t0, select_it=False):
    row = res.get("row") if isinstance(res, dict) else None
    SESSION.update_row(row)
    TIMES[kind] = time.monotonic() - t0
    if row is None:  # one-shot fallback: the command gives no row; list again
        refresh(context)
    elif select_it:
        select(context, str(row["id"]))
    elif str(row["id"]) == props(context).shot:
        fill_fields(props(context), row)
        p = props(context)
        scene = browser_scene()
        if scene is not None and p.loop_shot and core.has_range(row):
            a, b = core.shot_frames(row, SESSION.fps)
            scene.frame_preview_start, scene.frame_preview_end = a, b
    jobs.redraw()


def refresh(context=None, on_done=None):
    """List the shots again (``shots``); a refresh asked for while one runs runs after it."""
    if not SESSION.open:
        return
    if jobs.busy("list"):
        _REFRESH["again"] = True
        return
    root = SESSION.root_win

    def done(job, res):
        if SESSION.root_win != root:
            return  # another project was opened meanwhile
        SESSION.load(res)
        if on_done is not None:
            on_done()
        if _REFRESH["again"]:
            _REFRESH["again"] = False
            refresh()
        jobs.redraw()

    jobs.submit("list", "shots", {"project": SESSION.project}, "afv shots", done, root_win=SESSION.root_win)


# --- shot edits -----------------------------------------------------------------------------------------
def set_shot(context, shot, kind, label, **fields):
    """``shot_set`` for ``shot`` (start/end/speed/notes/state); the returned row replaces the list's."""
    t0 = time.monotonic()

    def done(job, res):
        _after_row(context, res, kind, t0)
        notes = [_short(n, 200) for n in res.get("notes_for_user", [])] if isinstance(res, dict) else []
        changed = ", ".join(res.get("changed") or []) if isinstance(res, dict) else ""
        msg = f"shot {shot}: {changed or 'no change'} ({TIMES[kind]:.2f} s)"
        jobs.set_status(msg + ("; " + "; ".join(notes) if notes else ""), "warning" if notes else "info")

    def failed(job, msg):
        jobs.set_status(f"shot {shot}: {_short(msg)}", "error")
        row = _row(shot)
        if row is not None and str(shot) == props(context).shot:
            fill_fields(props(context), row)  # put the fields back

    args = {"project": _project_arg(), "shot": str(shot),
            **{k: v for k, v in fields.items() if v is not None}}
    return jobs.submit("edit", "shot_set", args, label, done, failed, root_win=SESSION.root_win)


def send_speed():
    """Send the Speed values that have settled (one ``shot_set`` per shot)."""
    for shot, value in list(_PENDING_SPEED.items()):
        row = _row(shot)
        if row is not None and abs(float(row.get("speed_factor") or 1.0) - value) > 1e-6:
            set_shot(bpy.context, shot, "set", f"afv shots set {shot} --speed", speed=float(value))
        _PENDING_SPEED.pop(shot, None)
    return None


def _speed_changed(self, context):
    if _FILLING["on"] or not SESSION.open or not self.shot:
        return
    _PENDING_SPEED[self.shot] = float(self.speed)  # dragging the slider: send once it rests
    if bpy.app.timers.is_registered(send_speed):
        bpy.app.timers.unregister(send_speed)
    bpy.app.timers.register(send_speed, first_interval=SPEED_SETTLE_S)


def _notes_changed(self, context):
    if _FILLING["on"] or not SESSION.open or not self.shot:
        return
    row = _row(self.shot)
    if row is not None and (row.get("notes") or "") != self.notes:
        set_shot(context, self.shot, "set", f"afv shots set {self.shot} --notes", notes=self.notes)


def _filter_changed(self, context):
    self.page = 0


# --- properties -----------------------------------------------------------------------------------------
class AFVShotsProps(bpy.types.PropertyGroup):
    project: StringProperty(name="Project", description="The project folder shown in the browser")
    shot: StringProperty(name="Shot", description="The selected shot")
    state_filter: EnumProperty(
        name="Show", default="all", update=_filter_changed,
        items=[("all", "All", "Every shot"),
               ("candidate", "Candidate", "Found by cut detection, not chosen yet"),
               ("queued", "Queued", "Chosen: Run queue runs them"), ("skipped", "Skipped", "Left out")],
    )
    order: EnumProperty(
        name="Order", default="time", update=_filter_changed,
        items=[("time", "Time", "In source order"),
               ("likely", "Fighters likely first", "Shots where two fighter-sized people were found first "
                "(a hint from the scan: check the thumbnails)")],
    )
    page: IntProperty(name="Page", default=0, min=0)
    rows: IntProperty(name="Rows", default=10, min=4, max=40, description="Shots per page of the list")
    speed: FloatProperty(name="Speed", default=1.0, min=0.01, max=10.0, update=_speed_changed,
                         description="Footage speed vs real time (0.25 = 4x slow-motion replay); exported at "
                                     "real time. Sent to afv once the slider rests")
    notes: StringProperty(name="Notes", update=_notes_changed, description="Free text kept with the shot")
    loop_shot: BoolProperty(name="Loop shot", default=True,
                            description="Selecting a shot sets the timeline's preview range to it, so Space "
                                        "plays just the shot")
    advance: BoolProperty(name="Then next shot", default=True,
                          description="After Queue or Skip, select the next shot in the list's order")
    pick_next: BoolProperty(name="Then pick the next", default=True,
                            description="After saving a pick, start picking the next queued shot that needs "
                                        "one")
    show_markers: BoolProperty(name="Cut markers", default=True, update=_markers_changed,
                               description="A timeline marker on every detected cut")
    run_to: EnumProperty(name="Run until", default="export",
                         items=[(s, s, f"Run each queued shot up to {s}") for s in STAGES])


# --- operators --------------------------------------------------------------------------------------------
class _Op:
    bl_options = {"REGISTER"}  # noqa: RUF012 (Blender's API)

    def fail(self, msg):
        msg = _short(msg, 300)
        self.report({"ERROR"}, msg)
        jobs.set_status(msg, "error")
        return {"CANCELLED"}


def _need_project(cls, context):
    return SESSION.open and browser_scene() is not None


def open_payload(context, res, kind="open", t0=None):
    info = show_project(context, res)
    if t0 is not None:
        TIMES[kind] = time.monotonic() - t0
    c = SESSION.counts()
    bad = SESSION.errors()
    jobs.set_status(f"{len(SESSION.rows)} shots ({c['candidate']} candidate, {c['queued']} queued, "
                    f"{c['skipped']} skipped" + (f", {bad} unreadable" if bad else "") + f"), "
                    f"{info['markers']} cuts" + (f" | ready in {TIMES[kind]:.1f} s" if kind in TIMES else ""),
                    "warning" if bad else "info")
    return info


def open_project(context, folder: str, on_done=None):
    """``shots`` for the project in ``folder`` (a Windows path), then the browser scene, then the batch."""
    t0 = time.monotonic()
    folder = str(folder).rstrip("\\/")
    if not os.path.isfile(os.path.join(folder, "project.json")):
        raise ValueError(f"{folder} is not an afv project (no project.json)")

    def done(job, res):
        from .browser_run import RUNQ, batch_status

        if SESSION.root_win != res.get("root_win"):
            RUNQ.update({"missing": [], "outcomes": [], "current": None, "summary": "", "batch": None,
                         "last": None})
        open_payload(bpy.context, res, "open", t0)
        batch_status(bpy.context)  # a running batch: attach to it; else the last one's results
        if on_done is not None:
            on_done()

    PICK.reset()
    return jobs.submit("list", "shots", {"project": _wsl(folder)}, f"open {Path(folder).name}", done,
                       root_win=folder)


class AFV_OT_browser_open(_Op, bpy.types.Operator):
    """Open an afv project (its folder, with project.json) in the shot browser"""

    bl_idname = "afv.browser_open"
    bl_label = "Open project"

    directory: StringProperty(subtype="DIR_PATH")
    filter_folder: BoolProperty(default=True, options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return not jobs.busy("list") and not jobs.busy("init")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            open_project(context, self.directory)
        except (ValueError, RuntimeError) as e:
            return self.fail(e)
        jobs.set_status(f"opening {self.directory}...")
        return {"FINISHED"}


def _projects_dir(context) -> str:
    p = ops.prefs(context)
    return bpy.path.abspath(p.projects_dir) if p is not None and getattr(p, "projects_dir", "") else ""


class AFV_OT_browser_new(_Op, bpy.types.Operator):
    """Make an afv project from a video: probe, hash, detect the cuts (one candidate shot per segment),
    then scan the shots for people (GPU). All in the background in WSL"""

    bl_idname = "afv.browser_new"
    bl_label = "New project from video"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_movie: BoolProperty(default=True, options={"HIDDEN"})
    filter_folder: BoolProperty(default=True, options={"HIDDEN"})
    project_dir: StringProperty(
        name="Project folder", subtype="DIR_PATH",
        description="The new project's folder: a full path, or just a new name (made in the projects "
                    "folder). Empty: <projects folder>/<video name>. The projects folder is set in the "
                    "add-on's preferences, else the video's folder is used")
    title: StringProperty(name="Title", description="The source's title (kept for provenance)")
    scan_after: BoolProperty(name="Scan the shots after", default=True,
                             description="Find the people in every shot right away (one GPU job, about 1-2 "
                                         "min for 120 shots)")

    @classmethod
    def poll(cls, context):
        return not jobs.busy("init") and not jobs.busy("list")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "project_dir")
        col.prop(self, "title")
        col.prop(self, "scan_after")

    def execute(self, context):
        video = bpy.path.abspath(self.filepath)
        if not os.path.isfile(video):
            return self.fail(f"pick the video file (got {video!r})")
        if not video.lower().endswith(VIDEO_EXT):
            self.report({"WARNING"}, f"{Path(video).suffix} is not a usual video type; trying anyway")
        stem = Path(video).stem
        typed = self.project_dir.strip()
        if typed.startswith("//") and bpy.data.filepath:  # Blender-relative to a saved .blend
            typed = bpy.path.abspath(typed)
        folder = core.project_folder(typed, _projects_dir(context) or os.path.dirname(video), stem)
        if os.path.isfile(os.path.join(folder, "project.json")):
            return self.fail(f"{folder} is already an afv project: use Open project")
        t0 = time.monotonic()
        scan_after = self.scan_after

        def done(job, res):
            open_payload(bpy.context, res, "new", t0)
            d = res.get("detect") or {}
            jobs.set_status(f"new project {Path(folder).name}: {d.get('added', '?')} candidate shots from "
                            f"{d.get('cut_count', '?')} cuts in {TIMES['new']:.0f} s")
            if scan_after:
                scan(bpy.context)

        try:
            args = {"video": _wsl(video), "project": _wsl(folder), "detect": True, "proxy": "auto"}
            if self.title:
                args["title"] = self.title
            jobs.submit("init", "init", args, f"new project {Path(folder).name}", done, root_win=folder)
        except (ValueError, RuntimeError) as e:
            return self.fail(e)
        jobs.set_status(f"making {folder} from {Path(video).name} (cut detection: about 1 min per 20 min "
                        "of video)...")
        return {"FINISHED"}


class AFV_OT_browser_refresh(_Op, bpy.types.Operator):
    """List the shots again (e.g. after afv commands run outside Blender)"""

    bl_idname = "afv.browser_refresh"
    bl_label = "Refresh"

    @classmethod
    def poll(cls, context):
        return _need_project(cls, context)

    def execute(self, context):
        from .browser_run import batch_status

        refresh(context)
        batch_status(context)
        return {"FINISHED"}


class AFV_OT_browser_show(_Op, bpy.types.Operator):
    """Show the shot browser's scene in this window"""

    bl_idname = "afv.browser_show"
    bl_label = "Show shot browser"

    @classmethod
    def poll(cls, context):
        return browser_scene() is not None and context.window is not None

    def execute(self, context):
        context.window.scene = browser_scene()
        look_through_camera(browser_scene())
        return {"FINISHED"}


class AFV_OT_browser_select(_Op, bpy.types.Operator):
    """Select this shot and jump to it"""

    bl_idname = "afv.browser_select"
    bl_label = "Select shot"

    shot: StringProperty()

    @classmethod
    def poll(cls, context):
        return _need_project(cls, context)

    def execute(self, context):
        if self.shot not in SESSION.rows:
            return self.fail(f"no shot {self.shot}")
        select(context, self.shot)
        return {"FINISHED"}


class AFV_OT_browser_nav(_Op, bpy.types.Operator):
    """Select the next (or previous) shot in the list's order"""

    bl_idname = "afv.browser_nav"
    bl_label = "Next shot"

    step: IntProperty(default=1)
    likely: BoolProperty(name="Fighters likely", default=False, description="Only shots where fighters are "
                         "likely")

    @classmethod
    def poll(cls, context):
        return _need_project(cls, context)

    def execute(self, context):
        nxt = core.next_in_order(order_ids(context, self.likely), props(context).shot, self.step)
        if nxt is None:
            jobs.set_status("no more shots " + ("with fighters likely " if self.likely else "")
                            + ("after" if self.step > 0 else "before") + " this one in the list")
            return {"CANCELLED"}
        select(context, nxt)
        return {"FINISHED"}


class AFV_OT_browser_page(_Op, bpy.types.Operator):
    """Previous / next page of the shot list"""

    bl_idname = "afv.browser_page"
    bl_label = "Page"

    delta: IntProperty(default=1)

    def execute(self, context):
        p = props(context)
        n = len(SESSION.visible(p.state_filter, p.order))
        pages = max(1, -(-n // max(1, p.rows)))
        p.page = min(max(0, p.page + self.delta), pages - 1)
        return {"FINISHED"}


class AFV_OT_browser_set_edge(_Op, bpy.types.Operator):
    """Set the selected shot's in (start) or out (end) point to the playhead: in = the frame under the
    playhead is the shot's first, out = its last"""

    bl_idname = "afv.browser_set_edge"
    bl_label = "Set in / out"

    edge: EnumProperty(items=[("in", "In", ""), ("out", "Out", "")])

    @classmethod
    def poll(cls, context):
        r = selected_row(context) if _need_project(cls, context) else None
        return r is not None and core.has_range(r)

    def execute(self, context):
        row = selected_row(context)
        f = browser_scene().frame_current
        fps = SESSION.fps
        try:
            if self.edge == "in":
                set_shot(context, row["id"], "set", f"afv shots set {row['id']} --start",
                         start=round(core.start_time(f, fps), 6))
            else:
                set_shot(context, row["id"], "set", f"afv shots set {row['id']} --end",
                         end=round(core.end_time(f, fps), 6))
        except (RuntimeError, ValueError) as e:
            return self.fail(e)
        return {"FINISHED"}


def set_state(context, shot: str, state: str, advance: bool | None = None) -> None:
    """Queue / skip / candidate ``shot``; with ``advance`` (default: the panel's Then next shot) the next
    shot in the list's order is selected right away."""
    p = props(context)
    go_on = p.advance if advance is None else advance
    nxt = core.next_in_order(order_ids(context), shot) if go_on else None
    set_shot(context, shot, "state", f"afv shots {state} {shot}", state=state)
    if nxt is not None and shot == p.shot:
        select(context, nxt)


class AFV_OT_browser_state(_Op, bpy.types.Operator):
    """Queue the shot (Run queue runs it), skip it, or put it back to candidate (then, with Then next shot
    on, the next shot in the list is selected)"""

    bl_idname = "afv.browser_state"
    bl_label = "Shot state"

    state: EnumProperty(items=[(s, s.capitalize(), "") for s in core.STATES])
    shot: StringProperty(description="Shot id (default: the selected shot)")
    advance: EnumProperty(items=[("PANEL", "As the panel", ""), ("YES", "Yes", ""), ("NO", "No", "")],
                          default="PANEL", options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _need_project(cls, context)

    def execute(self, context):
        shot = self.shot or props(context).shot
        if shot not in SESSION.rows:
            return self.fail("select a shot first")
        try:
            set_state(context, shot, self.state, {"PANEL": None, "YES": True, "NO": False}[self.advance])
        except (RuntimeError, ValueError) as e:
            return self.fail(e)
        return {"FINISHED"}


class AFV_OT_browser_add(_Op, bpy.types.Operator):
    """Add a shot (queued) from the playhead to the next cut (at most 10 s; else 5 s); trim it with Set out"""

    bl_idname = "afv.browser_add"
    bl_label = "Add shot here"

    @classmethod
    def poll(cls, context):
        return _need_project(cls, context)

    def execute(self, context):
        scene = browser_scene()
        start, end = core.add_range(scene.frame_current, SESSION.fps, [f for f, _ in SESSION.cuts],
                                    SESSION.num_frames)
        t0 = time.monotonic()

        def done(job, res):
            _after_row(context, res, "add", t0, select_it=True)
            jobs.set_status(f"added shot {res.get('shot')}: {core.fmt_time(start)} - {core.fmt_time(end)} "
                            f"(queued; {TIMES['add']:.2f} s)")

        args = {"project": _project_arg(), "start": round(start, 6), "end": round(end, 6), "state": "queued"}
        try:
            jobs.submit("edit", "shot_add", args, "afv shots add", done, root_win=SESSION.root_win)
        except (RuntimeError, ValueError) as e:
            return self.fail(e)
        return {"FINISHED"}


def scan(context, force=False, shots=None, then=None):
    """``shots_scan`` in the background; each scanned shot's row arrives as it is written. ``then(sid)``
    is called for each shot as its scan arrives (Pick fighters waits on it)."""
    t0 = time.monotonic()
    n = len(shots) if shots else None

    def event(job, ev):
        if ev.get("event") == "log" and ev.get("kind") == "scanned":
            SESSION.update_row(ev.get("row"))
            if then is not None:
                then(str(ev.get("shot")))

    def done(job, res):
        TIMES["scan"] = time.monotonic() - t0
        k, up = len(res.get("scanned") or []), len(res.get("up_to_date") or [])
        failed = res.get("failed") or {}
        likely = sum(1 for r in SESSION.all_rows() if core.likely(r))
        jobs.set_status(f"scanned {k} shots ({up} up to date) in {TIMES['scan']:.0f} s; fighters likely in "
                        f"{likely}" + (f"; {len(failed)} failed: {', '.join(failed)}" if failed else ""),
                        "warning" if failed else "info")
        refresh(context)

    def failed(job, msg):
        jobs.set_status(f"scan: {_short(msg)} (scans written so far are kept)", "error")
        refresh(context)

    args = {"project": _project_arg(), **({"force": True} if force else {}),
            **({"shots": list(shots)} if shots else {})}
    label = f"scan {n} shots (about {core.scan_estimate(n):.0f} s)" if n else "scan"
    return jobs.submit("scan", "shots_scan", args, label, done, failed, event, root_win=SESSION.root_win)


class AFV_OT_browser_scan(_Op, bpy.types.Operator):
    """Find the people on one frame of every candidate and queued shot (one GPU job, about 1-2 min for 120
    shots): thumbnails, the numbered people Pick fighters uses, and the fighters-likely hint. Shots with an
    up-to-date scan are kept"""

    bl_idname = "afv.browser_scan"
    bl_label = "Scan shots"

    force: BoolProperty(name="Scan again", default=False, options={"SKIP_SAVE"},
                        description="Also scan shots whose scan is up to date")
    shots: StringProperty(options={"SKIP_SAVE"}, description="Only these shots (comma separated)")

    @classmethod
    def poll(cls, context):
        return _need_project(cls, context) and not jobs.busy("scan")

    def execute(self, context):
        ids = [s for s in self.shots.split(",") if s] or None
        try:
            scan(context, self.force, ids)
        except (RuntimeError, ValueError) as e:
            return self.fail(e)
        return {"FINISHED"}


class AFV_OT_browser_cancel(_Op, bpy.types.Operator):
    """Stop this background afv request"""

    bl_idname = "afv.browser_cancel"
    bl_label = "Cancel"

    lane: StringProperty()

    def execute(self, context):
        if self.lane == "pick" and PICK.active:
            from .browser_pick import end_pick

            end_pick(context)
        if self.lane == "batch":  # the batch runs detached in WSL: stop it there
            from .browser_run import stop_batch

            stop_batch(context)
            return {"FINISHED"}
        jobs.cancel(self.lane)
        return {"FINISHED"}


# --- file load, undo, registration ----------------------------------------------------------------------
@persistent
def on_load_post(*_args):
    """Another file: forget the open project here. Nothing in afv is cancelled: the batch goes on in WSL
    (Open project attaches again), a scan or a new project finishes and says so."""
    jobs.forget()
    close_session()


@persistent
def on_undo(*_args):
    """After an undo / redo: put the scene and the panel's fields back in line with the project (afv's
    rows); an undo past the opening of the browser closes it."""
    if not SESSION.open:
        return
    scene = browser_scene()
    if scene is None:
        jobs.forget()  # as for another file: nothing in afv is cancelled
        close_session()
        jobs.set_status("undo closed the shot browser: Open project to see it again")
        return
    try:
        held = json.loads(scene.get(draw.SCENE_PROP) or "{}")
    except ValueError:
        held = {}
    if held.get("root_win") != SESSION.root_win:  # undone to another project's scene: show this one
        show_project(bpy.context, SESSION.payload, undo=False)
        return
    p = scene.afv_shots
    if p.shot != SESSION.selected:
        p.shot = SESSION.selected
    fill_fields(p, _row(SESSION.selected))
    _set_markers(scene, SESSION.cuts if p.show_markers else [])
    cam = draw.browser_camera(scene)
    if cam is not None:
        img = bpy.data.images.get(os.path.basename(PICK.image)) if PICK.active and PICK.image else None
        _set_background(cam, clip=scene.active_clip, image=img)
    jobs.redraw()


CLASSES = (AFVShotsProps, AFV_OT_browser_open, AFV_OT_browser_new, AFV_OT_browser_refresh,
           AFV_OT_browser_show, AFV_OT_browser_select, AFV_OT_browser_nav, AFV_OT_browser_page,
           AFV_OT_browser_set_edge, AFV_OT_browser_state, AFV_OT_browser_add, AFV_OT_browser_scan,
           AFV_OT_browser_cancel)
_HANDLERS = ((bpy.app.handlers.load_post, on_load_post), (bpy.app.handlers.undo_post, on_undo),
             (bpy.app.handlers.redo_post, on_undo))


def _all_classes():
    from . import browser_keys, browser_pick, browser_run, browser_ui

    return (*CLASSES, *browser_pick.CLASSES, *browser_run.CLASSES, *browser_keys.CLASSES,
            *browser_ui.CLASSES)


def register():
    from . import browser_keys

    for cls in _all_classes():
        bpy.utils.register_class(cls)
    bpy.types.Scene.afv_shots = bpy.props.PointerProperty(type=AFVShotsProps)
    for handlers, fn in _HANDLERS:
        if fn not in handlers:
            handlers.append(fn)
    draw.register()
    browser_keys.register_keymaps()


def unregister():
    """Disabling the add-on cancels nothing in afv (the batch goes on in WSL)."""
    from . import browser_keys
    from .browser_run import _poll_status

    browser_keys.unregister_keymaps()
    jobs.forget(keep=())
    for fn in (jobs.poll, _poll_status, send_speed):
        if bpy.app.timers.is_registered(fn):
            bpy.app.timers.unregister(fn)
    draw.unregister()
    for handlers, fn in _HANDLERS:
        if fn in handlers:
            handlers.remove(fn)
    close_session()
    del bpy.types.Scene.afv_shots
    for cls in reversed(_all_classes()):
        bpy.utils.unregister_class(cls)
