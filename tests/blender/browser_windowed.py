"""Windowed Blender checks of the shot browser with simulated input (round 6 track B), run by
``test_browser_windowed.py``:

    blender --factory-startup --enable-event-simulate -noaudio --window-geometry 0 0 1600 900 \\
        --python browser_windowed.py -- job.json

job.json: {"result_path", "addon_parent", "video", "project", "worker" (argv of the fake afv serve)}.

New project (fake detection, fake track worker), queue 3 shots, then Pick fighters with real mouse
clicks and number keys through 3 shots with Then pick the next (the modal goes on from shot to shot); every
shortcut pressed with ``event_simulate`` over the 3D view and over the timeline (I, O, Q, S, Shift+Q,
Page Down / Up, L, Enter; and N, which stays Blender's), with the browser camera untouched; and Ctrl+Z
(ed.undo): the scene and the fields follow the project, an undo past the opening closes the browser.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback

import bpy

JOB = json.load(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8"))
sys.path.insert(0, JOB["addon_parent"])
RES: dict = {"log": []}


def save():
    with open(JOB["result_path"], "w", encoding="utf-8") as fh:
        json.dump(RES, fh, indent=1, default=str)


def view3d():
    win = bpy.context.window_manager.windows[0]
    area = next(a for a in win.screen.areas if a.type == "VIEW_3D")
    return win, area, next(r for r in area.regions if r.type == "WINDOW")


def ctx():
    win, area, region = view3d()
    return bpy.context.temp_override(window=win, area=area, region=region, screen=win.screen)


_JIT = [0]


def key(win, k, x, y, shift=False):
    _JIT[0] = (_JIT[0] + 1) % 7  # a real mouse never sits on one pixel: move in, a little apart each time
    for dx in (-40, -20, 0):
        win.event_simulate(type="MOUSEMOVE", value="NOTHING", x=x + dx + _JIT[0], y=y)
        yield 0.02
    if shift:
        win.event_simulate(type="LEFT_SHIFT", value="PRESS", x=x, y=y)
    text = {"ONE": "1", "ZERO": "0"}.get(k, k.lower() if len(k) == 1 else "")
    win.event_simulate(type=k, value="PRESS", x=x, y=y, shift=shift, unicode=text)
    yield 0.03
    win.event_simulate(type=k, value="RELEASE", x=x, y=y, shift=shift)
    if shift:
        win.event_simulate(type="LEFT_SHIFT", value="RELEASE", x=x, y=y)
    yield 0.25


def middle(kind):
    """(window, x, y): the middle of the area's main region (window coordinates)."""
    win = bpy.context.window_manager.windows[0]
    area = next(a for a in win.screen.areas if a.type == kind)
    region = next(r for r in area.regions if r.type == "WINDOW")
    side = sum(r.width for r in area.regions if r.type == "UI" and r.width > 1)  # the sidebar over it
    return win, int(region.x + (region.width - side) * 0.45), int(region.y + region.height * 0.5)


def click(win, x, y):
    win.event_simulate(type="MOUSEMOVE", value="NOTHING", x=x, y=y)
    yield 0.05
    win.event_simulate(type="LEFTMOUSE", value="PRESS", x=x, y=y)
    yield 0.03
    win.event_simulate(type="LEFTMOUSE", value="RELEASE", x=x, y=y)
    yield 0.1


def wait(cond, timeout=60.0):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise RuntimeError(f"timed out waiting ({cond.__name__ if hasattr(cond, '__name__') else cond})")
        yield 0.02


def run():
    import afv_addon
    from afv_addon import browser, browser_draw, ops, serve
    from afv_addon import browser_jobs as jobs

    serve.serve_argv = lambda distro, afv: list(JOB["worker"])
    afv_addon.register()
    ops.STATE.serve_default = True
    win, area, region = view3d()
    for sp in area.spaces:
        if sp.type == "VIEW_3D":
            sp.show_region_ui = True  # the AFV panels are in the sidebar: keep it open, as the owner does
    yield 0.5
    # factory startup without a file shows the splash screen, which takes every key until Esc
    yield from key(win, "ESC", region.x + 200, region.y + 200)
    with ctx():
        bpy.ops.afv.browser_new(filepath=JOB["video"], project_dir=JOB["project"], scan_after=True)
    yield from wait(lambda: browser.SESSION.open and not jobs.busy("init") and not jobs.busy("scan")
                    and not jobs.busy("list"), 240)
    S, PICK = browser.SESSION, browser_draw.PICK
    p = browser.props()
    p.pick_next, p.advance = True, True
    with ctx():
        for sid in ("001", "002", "003"):
            bpy.ops.afv.browser_state(state="queued", shot=sid, advance="NO")
    yield from wait(lambda: not jobs.busy("edit"))
    RES["queued"] = {k: r["state"] for k, r in S.rows.items()}

    # --- Pick fighters through 3 shots with Then next: clicks on shots 1 and 3, number keys on shot 2
    with ctx():
        RES["invoke"] = sorted(bpy.ops.afv.browser_pick("INVOKE_DEFAULT", shot="001"))
    picks = []
    for k in range(3):
        yield from wait(lambda: PICK.active and not PICK.waiting and not PICK.saving and PICK.prompt, 60)
        yield 0.3  # drawn
        shot = PICK.shot
        win, area, region = view3d()
        with bpy.context.temp_override(window=win, area=area, region=region):
            rv3d = area.spaces.active.region_3d
            boxes, corners = browser_draw.region_boxes(browser.browser_scene(), region, rv3d)
        entry = {"shot": shot, "boxes": len(boxes), "corners": corners is not None}
        if k == 1:  # number keys
            cx, cy = region.x + region.width // 2, region.y + region.height // 2
            yield from key(win, "ONE", cx, cy)
            yield from key(win, "ZERO", cx, cy)
        else:
            for kk in (1, 0):
                x0, y0, x1, y1 = dict(boxes)[kk]
                yield from click(win, int(region.x + (x0 + x1) / 2), int(region.y + (y0 + y1) / 2))
        entry["selected"] = list(PICK.selected)
        yield from key(win, "RET", region.x + 20, region.y + 20)
        yield from wait(lambda shot=shot: S.rows[shot].get("picked") and not jobs.busy("edit"), 30)
        yield 0.3
        entry["picked"] = S.rows[shot].get("picked")
        entry["after"] = {"active": PICK.active, "shot": PICK.shot}
        picks.append(entry)
    RES["picks"] = picks
    yield 0.3
    RES["after_last_pick"] = {"active": PICK.active}
    save()

    # --- every shortcut, pressed for real over the 3D view and over the timeline
    keys = {}
    sc = browser.browser_scene()
    cam = sc.camera
    def pose(o):
        return {"location": list(o.location), "scale": list(o.scale), "rotation": list(o.rotation_euler)}

    cam_before = pose(cam)

    def sel(shot, state=None):
        with ctx():
            if state is not None and S.rows[shot]["state"] != state:
                bpy.ops.afv.browser_state(state=state, shot=shot, advance="NO")
            bpy.ops.afv.browser_select(shot=shot)

    for kind in ("VIEW_3D", "DOPESHEET_EDITOR"):
        for a3 in bpy.context.window_manager.windows[0].screen.areas:  # the sidebar open: the AFV panels
            if a3.type == "VIEW_3D":
                a3.spaces.active.show_region_ui = True
        yield 0.3
        win, x, y = middle(kind)
        r = {}
        sel("001")
        yield from key(win, "PAGE_DOWN", x, y)
        r["page_down"] = p.shot
        yield from key(win, "PAGE_UP", x, y)
        r["page_up"] = p.shot
        yield from key(win, "L", x, y)
        r["L"] = p.shot
        sel("002")
        a0, b0 = browser.core.shot_frames(S.rows["002"], S.fps)
        sc.frame_set(a0 + 3)
        yield from key(win, "I", x, y)
        yield from wait(lambda: not jobs.busy("edit"))
        r["I"] = {"frame": sc.frame_current, "start": S.rows["002"]["start"]}
        sc.frame_set(b0 - 3)
        yield from key(win, "O", x, y)
        yield from wait(lambda: not jobs.busy("edit"))
        r["O"] = {"frame": sc.frame_current, "end": S.rows["002"]["end"]}
        sel("002", "candidate")
        yield from wait(lambda: not jobs.busy("edit"))
        yield from key(win, "S", x, y)
        yield from wait(lambda: not jobs.busy("edit"))
        r["S"] = {"state": S.rows["002"]["state"], "selected": p.shot}
        sel("002")
        yield from key(win, "Q", x, y)
        yield from wait(lambda: not jobs.busy("edit"))
        r["Q"] = {"state": S.rows["002"]["state"], "selected": p.shot}
        sel("001", "candidate")
        yield from wait(lambda: not jobs.busy("edit"))
        yield from key(win, "Q", x, y, shift=True)
        yield from wait(lambda: not jobs.busy("edit") and PICK.active and not PICK.waiting, 30)
        r["shift_Q"] = {"state": S.rows["001"]["state"], "picking": PICK.shot}
        yield from key(win, "ESC", x, y)
        r["esc"] = PICK.active
        sel("003")
        yield from key(win, "RET", x, y)
        yield from wait(lambda: PICK.active and not PICK.waiting, 30)
        r["enter"] = PICK.shot
        yield from key(win, "ESC", x, y)
        sel("003")
        yield from key(win, "N", x, y)  # Blender's (the sidebar), not ours
        r["N"] = p.shot
        keys[kind] = r
    keys["camera"] = {"action": bool(cam.animation_data and cam.animation_data.action),
                      "before": cam_before, "after": pose(cam),
                      "hide_select": cam.hide_select, "selected": cam.select_get()}
    RES["keys"] = keys
    save()

    # --- Ctrl+Z: the scene and the fields follow the project
    undo = {}
    with ctx():
        bpy.ops.afv.browser_select(shot="003")
    p.speed = 0.5
    browser.send_speed()
    yield from wait(lambda: not jobs.busy("edit") and S.rows["003"]["speed_factor"] == 0.5)
    with ctx():
        bpy.ops.ed.undo_push(message="speed field")
    yield 0.2
    with ctx():
        undo["ret"] = str(bpy.ops.ed.undo())
    yield 0.5
    sc = browser.browser_scene()
    p = browser.props()
    undo["after_one"] = {"scene": sc.name if sc else None, "open": S.open, "shot": p.shot if sc else None,
                         "speed_field": p.speed if sc else None, "speed_saved": S.rows.get("003", {}).get(
                             "speed_factor"), "set_edge_poll": bpy.types.AFV_OT_browser_set_edge.poll(
                             bpy.context)}
    with ctx():
        try:
            undo["state_op"] = sorted(bpy.ops.afv.browser_state(state="queued", shot="003", advance="NO"))
        except RuntimeError as e:
            undo["state_op"] = str(e)
    yield from wait(lambda: not jobs.busy("edit"))
    for _ in range(12):  # undo past the opening
        with ctx():
            try:
                bpy.ops.ed.undo()
            except RuntimeError:
                break
        yield 0.2
        if browser.browser_scene() is None:
            break
    sc = browser.browser_scene()
    undo["deep"] = {"scene": sc.name if sc else None, "open": S.open, "status": jobs.STATUS["text"],
                    "set_edge_poll": bpy.types.AFV_OT_browser_set_edge.poll(bpy.context)}
    RES["undo"] = undo
    save()


def tick():
    try:
        d = next(GEN)
        return max(0.005, float(d))
    except StopIteration:
        RES["ok"] = True
    except Exception:
        RES["error"] = traceback.format_exc()
    save()
    bpy.app.timers.register(quit_now, first_interval=0.3, persistent=True)
    return None


def quit_now():
    try:
        import afv_addon

        afv_addon.unregister()
    except Exception:
        pass
    save()
    os._exit(0)


def watchdog():
    RES["error"] = RES.get("error") or "watchdog: timed out"
    save()
    os._exit(3)


GEN = run()
bpy.app.timers.register(tick, first_interval=1.5, persistent=True)
_wd = threading.Timer(JOB.get("timeout", 400), watchdog)
_wd.daemon = True
_wd.start()
