"""The shot browser's panels: AFV > **Shots** (project, video, scan, background requests, messages) and
its sub-panels **Shot list**, **Shot** and **Run queue** (``browser.py``)."""

from __future__ import annotations

from pathlib import Path

import bpy

from . import browser_core as core
from . import browser_draw as draw
from . import browser_jobs as jobs
from . import ops
from .browser import PICK, SESSION, _row, _short, browser_scene, props, thumb_icon
from .browser_keys import HELP
from .browser_run import RUNQ, TAKE_READY, running


# --- panels ---------------------------------------------------------------------------------------------
def _job_rows(layout, lanes):
    for lane in lanes:
        j = jobs.job(lane)
        if j is None:
            continue
        row = layout.row(align=True)
        if j.frac is not None:
            row.progress(factor=max(0.0, min(1.0, j.frac)), type="BAR", text=_short(j.line(), 90))
        else:
            row.label(text=_short(f"{j.line()} ({j.elapsed:.0f} s)", 90), icon="SORTTIME")
        row.operator("afv.browser_cancel", text="", icon="X").lane = lane


def _status(layout):
    text, level = jobs.STATUS["text"], jobs.STATUS["level"]
    if text:
        col = layout.column()
        col.alert = level == "error"
        icon = {"error": "CANCEL", "warning": "ERROR"}.get(level, "INFO")
        for i, line in enumerate(_wrap(text, 48)[:5]):
            col.label(text=line, icon=icon if i == 0 else "BLANK1")


def _wrap(text, width):
    words, out, line = str(text).split(), [], ""
    for w in words:
        if line and len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


class AFV_PT_shots(bpy.types.Panel):
    bl_label = "Shots"
    bl_idname = "AFV_PT_shots"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AFV"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.operator("afv.browser_new", icon="FILE_MOVIE")
        row.operator("afv.browser_open", icon="FILE_FOLDER")
        scene = browser_scene()
        if not SESSION.open:
            if scene is not None and scene.afv_shots.project:
                box = layout.box()
                box.label(text=f"Last: {scene.afv_shots.project}", icon="INFO")
                box.operator("afv.browser_open", text="Open it again", icon="FILE_REFRESH").directory = (
                    scene.afv_shots.project)
            _job_rows(layout, ("init", "list"))
            _status(layout)
            return
        if context.scene != scene and scene is not None:
            layout.operator("afv.browser_show", icon="SEQUENCE")
        if TAKE_READY and context.scene.name != TAKE_READY.get("scene"):
            layout.operator("afv.browser_show_take", text=f"Show take {TAKE_READY['shot']}",
                            icon="ARMATURE_DATA")
        src, c = SESSION.source, SESSION.counts()
        box = layout.box()
        box.label(text=_short(src.get("title") or Path(SESSION.root_win).name, 60), icon="FILE_MOVIE")
        dur = src.get("duration")
        box.label(text=f"{src.get('width')}x{src.get('height')}, {SESSION.fps:.3f} fps, "
                       f"{core.fmt_time(dur)}" + (" (proxy)" if (SESSION.payload or {}).get("proxy") else ""))
        if SESSION.video_warning:
            col = box.column()
            col.alert = True
            for i, line in enumerate(_wrap(_short(SESSION.video_warning, 200), 48)):
                col.label(text=line, icon="ERROR" if i == 0 else "BLANK1")
        row = box.row(align=True)
        row.label(text=f"{len(SESSION.rows)} shots, {len(SESSION.cuts)} cuts")
        row.prop(props(context), "show_markers", toggle=True)
        box.label(text=f"{c['candidate']} candidate, {c['queued']} queued, {c['skipped']} skipped"
                       + (f", {SESSION.errors()} unreadable" if SESSION.errors() else ""))
        row = box.row(align=True)
        row.operator("afv.browser_scan", icon="VIEWZOOM")
        row.operator("afv.browser_scan", text="Scan all again").force = True
        row.operator("afv.browser_refresh", text="", icon="FILE_REFRESH")
        _job_rows(layout, ("init", "list", "scan", "pick", "edit", "take"))
        _status(layout)
        layout.label(text=ops.worker_status(), icon="CONSOLE")
        if context.scene == scene:
            box = layout.box()
            box.label(text="Keys (mouse over the video or the timeline):", icon="EVENT_RETURN")
            for line in HELP:
                box.label(text=line)


class _ShotsSub:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AFV"
    bl_parent_id = "AFV_PT_shots"

    @classmethod
    def poll(cls, context):
        return SESSION.open and browser_scene() is not None


class AFV_PT_shots_list(_ShotsSub, bpy.types.Panel):
    bl_label = "Shot list"
    bl_idname = "AFV_PT_shots_list"

    def draw(self, context):
        layout = self.layout
        p = props(context)
        layout.row().prop(p, "state_filter", expand=True)
        row = layout.row(align=True)
        row.prop(p, "order", text="")
        row.prop(p, "rows")
        rows = SESSION.visible(p.state_filter, p.order)
        n = len(rows)
        per = max(1, p.rows)
        pages = max(1, -(-n // per))
        page = min(p.page, pages - 1)
        if not rows:
            layout.label(text="No shots here" + ("" if p.state_filter == "all" else f" in {p.state_filter}"))
        col = layout.column(align=True)
        for r in rows[page * per:(page + 1) * per]:
            draw_row(col, r, str(r["id"]) == p.shot)
        row = layout.row(align=True)
        row.operator("afv.browser_nav", text="Previous", icon="TRIA_UP").step = -1
        row.operator("afv.browser_nav", text="Next", icon="TRIA_DOWN").step = 1
        op = row.operator("afv.browser_nav", text="Next likely", icon="SOLO_ON")
        op.step, op.likely = 1, True
        row = layout.row(align=True)
        row.operator("afv.browser_page", text="", icon="TRIA_LEFT").delta = -1
        row.label(text=f"Page {page + 1}/{pages} ({n} shots)  * = fighters likely (a hint)")
        row.operator("afv.browser_page", text="", icon="TRIA_RIGHT").delta = 1


def draw_row(layout, r, selected):
    split = layout.split(factor=0.2, align=True)
    icon = thumb_icon(r)
    if icon:
        split.template_icon(icon_value=icon, scale=2.0)
    else:
        split.label(text="", icon="IMAGE_DATA")
    col = split.column(align=True)
    line = core.row_line(r)
    op = col.operator("afv.browser_select", text=line, depress=selected,
                      icon="SOLO_ON" if core.likely(r) else "BLANK1")
    op.shot = str(r["id"])
    sub = col.row(align=True)
    sub.alert = bool(r.get("pick_stale") or r.get("error"))
    sub.label(text=core.row_detail(r, SESSION.outcomes.get(str(r["id"]))),
              icon="ERROR" if r.get("error") else "NONE")


class AFV_PT_shots_shot(_ShotsSub, bpy.types.Panel):
    bl_label = "Shot"
    bl_idname = "AFV_PT_shots_shot"

    def draw(self, context):
        layout = self.layout
        p = props(context)
        r = _row(p.shot)
        if PICK.active:
            pick_box(layout)
        if r is None:
            layout.label(text="Click a shot in the list", icon="INFO")
        elif r.get("error"):
            col = layout.column()
            col.alert = True
            col.label(text=f"Shot {r['id']}: afv cannot read it", icon="ERROR")
            for line in _wrap(str(r["error"]), 70)[:4]:
                col.label(text=line)
            col.label(text="Fix or delete shots/<id>/shot.json, then Refresh")
        else:
            icon = thumb_icon(r)
            if icon and not PICK.active:
                layout.template_icon(icon_value=icon, scale=7.0)
            span = f"{core.fmt_time(r.get('start'))} - {core.fmt_time(r.get('end'))}"
            layout.label(text=f"Shot {r['id']}: {span} ({core.fmt_len(r.get('duration'))}), cut "
                              f"{core.conf_text(r)}", icon="SEQUENCE")
            pick = core.pick_status(r) or "fighters not picked"
            line = layout.row()
            line.alert = r.get("pick_stale", False)
            last = SESSION.outcomes.get(str(r["id"]))
            line.label(text=f"{r.get('state')}; {core.people_text(r)}; {pick}"
                            + ("; exported" if r.get("exported") else "")
                            + (f"; last run: {core.OUTCOME_TEXT.get(last, last)}" if last else ""))
            row = layout.row(align=True)
            row.operator("afv.browser_set_edge", text="Set in", icon="TRIA_LEFT_BAR").edge = "in"
            row.operator("afv.browser_set_edge", text="Set out", icon="TRIA_RIGHT_BAR").edge = "out"
            row.operator("afv.browser_select", text="", icon="PREVIEW_RANGE").shot = str(r["id"])
            row = layout.row(align=True)
            row.prop(p, "speed")
            row.prop(p, "loop_shot", toggle=True)
            layout.prop(p, "notes")
            row = layout.row(align=True)
            row.prop(p, "advance", toggle=True)
            row = layout.row(align=True)
            for st, ic in (("queued", "CHECKMARK"), ("skipped", "X"), ("candidate", "QUESTION")):
                sub = row.row(align=True)
                sub.enabled = r.get("state") != st
                op = sub.operator("afv.browser_state", text={"queued": "Queue", "skipped": "Skip"}.get(
                    st, "Candidate"), icon=ic)
                op.state, op.shot = st, str(r["id"])
            row = layout.row(align=True)
            row.scale_y = 1.3
            row.operator("afv.browser_pick", icon="RESTRICT_SELECT_OFF").shot = str(r["id"])
            row.operator("afv.browser_queue_pick", icon="CHECKMARK").shot = str(r["id"])
            row = layout.row(align=True)
            op = row.operator("afv.browser_pick", text="Pick on this frame", icon="TIME")
            op.shot, op.on_playhead = str(r["id"]), True
            row.prop(p, "pick_next", text="Then next", toggle=True)
            if (r.get("scan") or {}).get("status") != "ok":
                n = len(core.needs_scan(SESSION.all_rows()) or [r["id"]])
                plural = "s" if n != 1 else ""
                layout.label(text=f"Not scanned yet: Pick scans it first ({n} shot{plural} in one job, "
                                  f"about {core.scan_estimate(n):.0f} s)", icon="INFO")
            if r.get("exported") or "pose3d" in (r.get("stored") or []):
                layout.operator("afv.browser_take", icon="IMPORT").shot = str(r["id"])
        layout.separator()
        layout.operator("afv.browser_add", icon="ADD")


def pick_box(layout):
    box = layout.box()
    if PICK.waiting:
        for i, line in enumerate(_wrap(f"Shot {PICK.shot}: {draw.wait_line()}", 48)):
            box.label(text=line, icon="SORTTIME" if i == 0 else "BLANK1")
    elif PICK.saving:
        box.label(text=f"Shot {PICK.shot}: saving {PICK.text()}...", icon="SORTTIME")
    else:
        n = len(PICK.boxes())
        box.label(text=f"Pick shot {PICK.shot}: {n} people on frame {PICK.frame} ({PICK.source})",
                  icon="RESTRICT_SELECT_OFF")
        box.label(text=PICK.text())
        row = box.row(align=True)
        for k, _ in PICK.boxes()[:12]:
            role = PICK.role(k)
            row.operator("afv.browser_pick_box", text=f"{k}{' ' + role if role else ''}",
                         depress=bool(role)).index = k
    row = box.row(align=True)
    row.operator("afv.browser_pick_save", icon="CHECKMARK")
    row.operator("afv.browser_pick_undo", icon="LOOP_BACK")
    row.operator("afv.browser_pick_cancel", text="", icon="X")


class AFV_PT_shots_run(_ShotsSub, bpy.types.Panel):
    bl_label = "Run queue"
    bl_idname = "AFV_PT_shots_run"

    def draw(self, context):
        layout = self.layout
        p = props(context)
        ready, missing = core.queue_check(SESSION.all_rows())
        layout.label(text=f"Queued: {len(ready) + len(missing)}, fighters picked: {len(ready)}")
        layout.prop(p, "run_to")
        row = layout.row(align=True)
        row.scale_y = 1.3
        row.operator("afv.browser_run", icon="PLAY")
        if running():
            row.operator("afv.browser_stop_batch", text="Stop batch", icon="CANCEL")
        _job_rows(layout, ("batch",))
        b = RUNQ.get("batch") or {}
        if running():
            box = layout.box()
            if b.get("running") and not jobs.busy("batch"):  # known from batch_status only (polled)
                cur = RUNQ.get("current") or {}
                box.label(text=f"Batch {b.get('id') or ''} running in WSL: shot {cur.get('shot') or '-'}"
                               + (f", {cur['stage']}" if cur.get("stage") else ""), icon="SORTTIME")
            box.label(text="Shots you queue and pick now are run by this batch", icon="INFO")
            box.label(text="(afv reads the queue again after each shot).", icon="BLANK1")
            if b.get("log"):
                box.label(text=_short(f"log: {b['log']}", 90))
        if ops.use_serve() and RUNQ.get("status_cmd", True):
            layout.label(text="The batch runs in WSL: closing Blender does not stop", icon="INFO")
            layout.label(text="it. Open project shows it (or its results) again.", icon="BLANK1")
        elif jobs.busy("batch"):
            layout.label(text="One-shot afv commands: closing Blender stops the batch", icon="ERROR")
        shown = RUNQ["missing"] if RUNQ["missing"] else []
        if shown:
            box = layout.box()
            box.alert = True
            box.label(text=f"{len(shown)} queued shots need their fighters picked:", icon="ERROR")
            for sid, why in shown[:12]:
                r = box.row(align=True)
                r.operator("afv.browser_select", text=f"{sid}: {why}").shot = sid
                r.operator("afv.browser_pick", text="", icon="RESTRICT_SELECT_OFF").shot = sid
            if len(shown) > 12:
                box.label(text=f"... and {len(shown) - 12} more")
            todo = core.needs_scan(SESSION.all_rows())
            if todo:
                est = core.scan_estimate(len(todo))
                box.operator("afv.browser_scan_for_picks", icon="VIEWZOOM",
                             text=f"Scan the {len(todo)} unscanned first (about {est:.0f} s)")
            if ready:
                box.operator("afv.browser_run", text=f"Run the {len(ready)} picked ones").only_ready = True
        if RUNQ["summary"]:
            for i, line in enumerate(_wrap(_short(RUNQ["summary"], 200), 48)):
                layout.label(text=line, icon="INFO" if i == 0 else "BLANK1")
        for o in RUNQ["outcomes"][-20:]:
            r = layout.row(align=True)
            r.alert = o.get("outcome") not in ("ok",)
            what = core.OUTCOME_TEXT.get(o.get("outcome"), o.get("outcome"))
            text = _short(f"{o.get('shot')}: {what} {o.get('message') or ''}", 80)
            r.operator("afv.browser_select", text=text).shot = str(o.get("shot"))


CLASSES = (AFV_PT_shots, AFV_PT_shots_list, AFV_PT_shots_shot, AFV_PT_shots_run)
