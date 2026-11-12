"""The shot browser's **Pick fighters** (``browser.py``): the shot's prompt frame with its numbered people
drawn over the camera view (``browser_draw``); click fighter A, then B, then extras, Enter saves (afv
``pick`` with fighters, extras and the prompt frame).

The prompt comes from afv ``pick`` without fighters: from the shot's scan at once (the exact scanned
frame, ``media/scan/<id>.frame.jpg``, is shown). A shot without an up-to-date scan (new, or trimmed) is
not detected on its own (one GPU start, about 35 s, per shot): every queued shot that needs a pick and a
scan is scanned in **one** job first, and the pick starts as soon as this shot's scan arrives (the wait is
shown). **Pick on this frame** detects the people on the playhead's frame instead (GPU, about 35 s).

The pick stays on screen until afv has saved it: a refused save keeps the clicks and shows why. With
**Then pick the next**, the next queued shot that needs a pick comes up in the same modal session, so
clicks and number keys keep working. Scripts (and the panel's number buttons) use the same code:
``afv.browser_pick`` (exec: start), ``afv.browser_pick_box``, ``afv.browser_pick_save``.
"""

from __future__ import annotations

import os
import time

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty

from . import browser_core as core
from . import browser_draw as draw
from . import browser_jobs as jobs
from . import takeio
from .browser import (
    PICK,
    SESSION,
    TIMES,
    _after_row,
    _camera,
    _need_project,
    _Op,
    _project_arg,
    _row,
    _set_background,
    _short,
    _windows,
    browser_scene,
    props,
    scan,
    select,
)

DETECT_S = 35.0  # one detection on a frame that has no scan: the GPU worker's start (measured 30-40 s)


# --- pick fighters ----------------------------------------------------------------------------------------
def _load_image(path: str):
    want = os.path.normcase(os.path.abspath(path))
    for img in bpy.data.images:
        if img.filepath and os.path.normcase(os.path.abspath(bpy.path.abspath(img.filepath))) == want:
            img.reload()
            return img
    return bpy.data.images.load(path)


def _prompt_image(res: dict) -> str | None:
    if res.get("source") == "scan":
        p = core.scan_file(SESSION.root_win, (SESSION.payload or {}).get("scan_dir", "media/scan"),
                           res["shot"], ".frame.jpg")
        if p.is_file():
            return str(p)  # the exact frame, without afv's numbers drawn in (the browser draws its own)
    return res.get("image_win") or _windows(res.get("image"))


def _wait(text: str, seconds: float | None = None) -> None:
    PICK.waiting = True
    PICK.wait_text = text
    PICK.wait_until = time.monotonic() + seconds if seconds else None
    jobs.set_status(f"shot {PICK.shot}: {text}" + (f" (about {seconds:.0f} s)" if seconds else ""))


def start_pick(context, shot: str, frame: int | None = None) -> None:
    """Show shot ``shot``'s numbered people and wait for clicks: from its scan at once; a shot without an
    up-to-date scan is scanned first, together with every other queued shot that needs a pick and a scan
    (one GPU job); ``frame`` (shot-timeline): detect the people on that frame instead (GPU)."""
    if shot not in SESSION.rows:
        raise ValueError(f"no shot {shot}")
    row = SESSION.rows[shot]
    if row.get("error"):
        raise ValueError(f"shot {shot} cannot be read: {row['error']}")
    t0 = time.monotonic()
    if PICK.active:
        end_pick(context)
    if jobs.busy("pick"):
        jobs.cancel("pick")
        jobs.JOBS[:] = [j for j in jobs.JOBS if j.lane != "pick"]
    PICK.reset()
    PICK.active, PICK.shot = True, str(shot)
    select(context, shot)
    scanned = (row.get("scan") or {}).get("status") == "ok"
    if frame is None and not scanned:
        todo = core.needs_scan(SESSION.all_rows())
        todo = [shot, *[s for s in todo if s != shot]] if shot not in todo else todo
        if not jobs.busy("scan"):
            scan(context, shots=todo, then=lambda sid: _scanned(context, sid, shot, t0))
            what = "this shot" if len(todo) == 1 else f"the {len(todo)} queued shots that need a scan"
            _wait(f"scanning {what} first (one GPU job)", core.scan_estimate(len(todo)))
            return
        _wait("a scan is running; finding this shot's people after it", None)
    _request_prompt(context, shot, frame, t0)


def _scanned(context, sid: str, shot: str, t0: float) -> None:
    if sid == shot and PICK.active and PICK.shot == shot and PICK.waiting and not jobs.busy("pick"):
        _request_prompt(context, shot, None, t0)


def _request_prompt(context, shot: str, frame: int | None, t0: float) -> None:
    row = SESSION.rows[shot]
    if frame is not None and frame != (row.get("scan") or {}).get("frame"):
        _wait(f"finding the people on frame {frame} (GPU)", DETECT_S)
    elif (row.get("scan") or {}).get("status") != "ok":
        _wait("finding the people (one GPU job for this shot)", DETECT_S)
    else:
        PICK.waiting = True
        PICK.wait_text = "loading its people"

    def done(job, res):
        if not PICK.active or PICK.shot != str(res.get("shot")):
            return  # cancelled or another pick started meanwhile
        path = _prompt_image(res)
        PICK.prompt, PICK.frame = res.get("prompt") or {}, res.get("frame")
        PICK.source = res.get("source", "")
        PICK.current = res.get("current")
        scene = browser_scene()
        if path and os.path.isfile(path) and scene is not None:
            img = _load_image(path)
            PICK.image, PICK.size = path, (int(img.size[0]), int(img.size[1]))
            _set_background(_camera(scene), image=img)
        else:
            src = SESSION.source
            PICK.size = (int(src.get("width") or 0), int(src.get("height") or 0))
        cur = PICK.current or {}
        if cur.get("fighters") and cur.get("prompt_frame") == PICK.frame:  # the shot's pick, on this frame
            PICK.selected = [int(v) for v in [*cur["fighters"], *(cur.get("extra") or [])]]
        if scene is not None and PICK.frame is not None and core.has_range(SESSION.rows[PICK.shot]):
            scene.frame_set(core.prompt_source_frame(SESSION.rows[PICK.shot], PICK.frame, SESSION.fps))
        PICK.waiting, PICK.wait_text, PICK.wait_until = False, "", None
        TIMES["pick_prompt"] = time.monotonic() - t0
        n = len(PICK.boxes())
        took = TIMES["pick_prompt"]
        if n:
            jobs.set_status(f"shot {shot}: {n} people on frame {PICK.frame} ({PICK.source}, {took:.2f} s): "
                            "click fighter A, then B")
        else:
            jobs.set_status(f"shot {shot}: nobody found on frame {PICK.frame}: Pick on this frame elsewhere "
                            "in the shot, or trim it", "warning")
        jobs.redraw()

    def failed(job, msg):
        if PICK.active and PICK.shot == str(shot):
            end_pick(context)
        jobs.set_status(f"pick {shot}: {_short(msg)}", "error")

    args = {"project": _project_arg(), "shot": str(shot)}
    if frame is not None:
        args["frame"] = int(frame)
    jobs.submit("pick", "pick", args, f"people on shot {shot}", done, failed, root_win=SESSION.root_win)


def end_pick(context=None) -> None:
    """Leave picking: the video comes back behind the camera."""
    PICK.reset()
    scene = browser_scene()
    cam = draw.browser_camera(scene) if scene is not None else None
    if cam is not None:
        _set_background(cam, clip=scene.active_clip)
    jobs.redraw()


def save_pick(context) -> None:
    """Send the chosen numbers to afv (``pick`` with fighters and extras on the prompt frame). The pick stays
    on screen until afv has saved it; then the next queued shot that needs a pick comes up (Then pick the
    next), in the same modal session."""
    if not PICK.active or PICK.waiting or PICK.saving:
        raise ValueError("no pick to save yet")
    if len(PICK.selected) < 2:
        raise ValueError("click fighter A and fighter B first")
    t0 = time.monotonic()
    shot, sel, frame = PICK.shot, list(PICK.selected), PICK.frame
    pick_next = props(context).pick_next
    PICK.saving = True

    def done(job, res):
        _after_row(context, res, "pick_save", t0)
        jobs.set_status(f"shot {shot}: fighters A = #{sel[0]}, B = #{sel[1]}"
                        + (f", extra {sel[2:]}" if sel[2:] else "") + f" saved ({TIMES['pick_save']:.2f} s)")
        if not (PICK.active and PICK.shot == shot):
            return  # the owner moved on meanwhile
        nxt = core.next_to_pick(SESSION.all_rows(), shot) if pick_next else None
        if nxt is not None and nxt != shot:
            start_pick(context, nxt)  # PICK stays active: the modal goes on with the next shot
        else:
            end_pick(context)

    def failed(job, msg):
        if PICK.active and PICK.shot == shot:
            PICK.saving = False  # the clicks stay: fix them and press Enter again
        jobs.set_status(f"pick {shot} not saved: {_short(msg, 200)}", "error")
        jobs.redraw()

    args = {"project": _project_arg(), "shot": shot, "fighters": sel[:2], "extra": sel[2:]}
    if frame is not None:
        args["frame"] = int(frame)
    jobs.submit("edit", "pick", args, f"afv pick {shot} --fighters", done, failed, root_win=SESSION.root_win)


def _view3d_under(context, x, y):
    from .cleanup import _view3d_under as under

    return under(context, x, y)


DIGITS = {"ZERO": 0, "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8,
          "NINE": 9, "NUMPAD_0": 0, "NUMPAD_1": 1, "NUMPAD_2": 2, "NUMPAD_3": 3, "NUMPAD_4": 4, "NUMPAD_5": 5,
          "NUMPAD_6": 6, "NUMPAD_7": 7, "NUMPAD_8": 8, "NUMPAD_9": 9}


class AFV_OT_browser_pick(_Op, bpy.types.Operator):
    """Pick the fighters of the shot: its numbered people are shown over the video; click fighter A, then B,
    then any extras (referee); Enter saves. From a scan this is instant; shots without a scan are scanned
    first, all in one GPU job"""

    bl_idname = "afv.browser_pick"
    bl_label = "Pick fighters"

    shot: StringProperty(description="Shot id (default: the selected shot)")
    on_playhead: BoolProperty(name="On this frame", default=False, options={"SKIP_SAVE"},
                              description="Find the people on the playhead's frame instead of the scanned "
                                          "one (GPU, about 35 s)")

    @classmethod
    def poll(cls, context):
        return _need_project(cls, context)  # any row (e.g. Run queue's list), not only the selected one

    def _begin(self, context):
        shot = self.shot or props(context).shot
        if shot not in SESSION.rows:
            return self.fail("select a shot first")
        frame = None
        if self.on_playhead:
            row = _row(shot)
            frame = core.shot_frame_at(row, browser_scene().frame_current, SESSION.fps)
        try:
            start_pick(context, shot, frame)
        except (ValueError, RuntimeError) as e:
            PICK.reset()
            return self.fail(e)
        return None

    def execute(self, context):  # scripted use: start the pick; clicks are afv.browser_pick_box
        return self._begin(context) or {"FINISHED"}

    def invoke(self, context, event):
        bad = self._begin(context)
        if bad is not None:
            return bad
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if not PICK.active:  # saved (and nothing next) or cancelled from the panel: this event is not ours
            return {"FINISHED", "PASS_THROUGH"}
        if event.value != "PRESS":
            return {"PASS_THROUGH"}
        if event.type == "ESC":
            if PICK.waiting:
                jobs.cancel("pick")
            end_pick(context)
            jobs.set_status("pick: cancelled")
            return {"CANCELLED"}
        if PICK.waiting or PICK.saving:
            return {"PASS_THROUGH"}
        if event.type in ("RET", "NUMPAD_ENTER"):
            try:
                save_pick(context)
            except (ValueError, RuntimeError) as e:
                jobs.set_status(str(e), "warning")
            return {"RUNNING_MODAL"}
        if event.type == "BACK_SPACE":
            if PICK.selected:
                PICK.selected.pop()
            jobs.redraw()
            return {"RUNNING_MODAL"}
        if event.type in DIGITS and not (event.ctrl or event.alt or event.shift):
            PICK.toggle(DIGITS[event.type])
            jobs.redraw()
            return {"RUNNING_MODAL"}
        if event.type not in ("LEFTMOUSE", "RIGHTMOUSE"):
            return {"PASS_THROUGH"}
        region, rv3d = _view3d_under(context, event.mouse_x, event.mouse_y)
        if region is None:
            return {"PASS_THROUGH"}
        boxes, corners = draw.region_boxes(browser_scene(), region, rv3d)
        x, y = event.mouse_x - region.x, event.mouse_y - region.y
        if corners is None or not takeio.inside_frame(corners, x, y):
            return {"PASS_THROUGH"}  # outside the camera frame: the viewport works as usual
        if event.type == "RIGHTMOUSE":
            end_pick(context)
            jobs.set_status("pick: cancelled")
            return {"CANCELLED"}
        k = takeio.box_at(boxes, x, y)
        if k is not None:
            PICK.toggle(k)
            jobs.redraw()
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        end_pick(context)


class AFV_OT_browser_pick_box(_Op, bpy.types.Operator):
    """Choose (or drop) detection number ``index``: the first two chosen are fighters A and B, then extras"""

    bl_idname = "afv.browser_pick_box"
    bl_label = "Choose person"

    index: IntProperty(min=0)

    @classmethod
    def poll(cls, context):
        return PICK.active and not PICK.waiting and not PICK.saving

    def execute(self, context):
        if not any(k == self.index for k, _ in PICK.boxes()):
            return self.fail(f"no person #{self.index} on this frame")
        PICK.toggle(self.index)
        jobs.redraw()
        return {"FINISHED"}


class AFV_OT_browser_pick_save(_Op, bpy.types.Operator):
    """Save the pick (fighter A, fighter B, extras) to the shot"""

    bl_idname = "afv.browser_pick_save"
    bl_label = "Save pick"

    @classmethod
    def poll(cls, context):
        return PICK.active and not PICK.waiting and not PICK.saving and len(PICK.selected) >= 2

    def execute(self, context):
        try:
            save_pick(context)
        except (ValueError, RuntimeError) as e:
            return self.fail(e)
        return {"FINISHED"}


class AFV_OT_browser_pick_undo(_Op, bpy.types.Operator):
    """Drop the last person chosen"""

    bl_idname = "afv.browser_pick_undo"
    bl_label = "Undo last"

    @classmethod
    def poll(cls, context):
        return PICK.active and bool(PICK.selected) and not PICK.saving

    def execute(self, context):
        PICK.selected.pop()
        jobs.redraw()
        return {"FINISHED"}


class AFV_OT_browser_pick_cancel(_Op, bpy.types.Operator):
    """Stop picking (nothing is saved)"""

    bl_idname = "afv.browser_pick_cancel"
    bl_label = "Cancel pick"

    @classmethod
    def poll(cls, context):
        return PICK.active

    def execute(self, context):
        if PICK.waiting:
            jobs.cancel("pick")
        end_pick(context)
        jobs.set_status("pick: cancelled")
        return {"FINISHED"}


class AFV_OT_browser_scan_for_picks(_Op, bpy.types.Operator):
    """Scan, in one GPU job, every queued shot that needs a pick and has no up-to-date scan (new or trimmed
    shots), so picking them is instant"""

    bl_idname = "afv.browser_scan_for_picks"
    bl_label = "Scan them first"

    @classmethod
    def poll(cls, context):
        return (_need_project(cls, context) and not jobs.busy("scan")
                and bool(core.needs_scan(SESSION.all_rows())))

    def execute(self, context):
        todo = core.needs_scan(SESSION.all_rows())
        try:
            scan(context, shots=todo)
        except (RuntimeError, ValueError) as e:
            return self.fail(e)
        return {"FINISHED"}


CLASSES = (AFV_OT_browser_pick, AFV_OT_browser_pick_box, AFV_OT_browser_pick_save, AFV_OT_browser_pick_undo,
           AFV_OT_browser_pick_cancel, AFV_OT_browser_scan_for_picks)
