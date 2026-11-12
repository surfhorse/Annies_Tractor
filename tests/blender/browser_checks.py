"""Blender-side checks of the shot browser (round 6, track B), run by ``tests/blender/test_browser*.py``:

    blender -b --factory-startup -noaudio --python browser_checks.py -- job.json

job.json: {"mode", "result_path", "addon_parent", "tests_dir", ...}. Background mode: timers do not run, so
the checks drive the browser's job poll themselves (``browser_jobs.pump``) and call the same operators the
panel's buttons call (``bpy.ops.afv.browser_*``); a pick's clicks go through ``afv.browser_pick_box`` (what
a click on a box does after its hit test, which ``test_browser_core.py`` checks).

Modes:
- flow: {"video", "project", "worker": [argv of a fake afv serve], "segments"}: New project from video,
  the list, markers, the clip, Set in / Set out, speed, notes, Queue / Skip, Add shot here, Scan (fake
  track worker), Pick fighters from the scan, the Run queue check and a batch, Import take refused for a
  shot without a take, the panels' draw code, then the same against the one-shot fallback.
- e2e: {"project", "queue": n, "scan": "force" | "resume", "to"}: the scripted end-to-end run on a scratch
  copy of a real project with the real ``afv serve``: open, (re)scan, queue n fighters-likely candidates,
  pick fighters in each, start the queue (then cancel it); every step timed.
- runq: {"project", "shots": [ids], "to"}: queue and pick those shots, Run queue to completion.
- detached: {"project", "cfg", "log"}: a batch already running in WSL (``browser_stub_serve.py`` answers
  ``batch_status`` / ``batch_cancel``): Open project shows it, Stop batch ends it; a row afv cannot read.
- open: {"project"}: open a real project read-only (timing).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import bpy


# --- a recording layout (the panels' draw code against it) --------------------------------------------
class Layout:
    def __init__(self, log):
        self.log = log
        self.alert = False
        self.enabled = True
        self.scale_y = 1.0

    def _sub(self, *a, **kw):
        return Layout(self.log)

    row = column = box = split = _sub

    def label(self, text="", **kw):
        self.log.append(("label", text, self.alert))

    def separator(self, **kw):
        pass

    def prop(self, data, name, **kw):
        getattr(data, name)
        self.log.append(("prop", name))

    def progress(self, factor=0.0, type="BAR", text=""):
        assert 0.0 <= factor <= 1.0
        self.log.append(("progress", text, round(factor, 3)))

    def template_icon(self, icon_value=0, scale=1.0):
        self.log.append(("icon", icon_value))

    def operator(self, idname, text="", **kw):
        mod, op = idname.split(".")
        assert hasattr(getattr(bpy.ops, mod), op), idname
        self.log.append(("operator", idname, text))
        return type("OpProps", (), {})()


def draw_panels(context):
    from afv_addon import browser_ui

    log = []
    for cls in browser_ui.CLASSES:
        if hasattr(cls, "poll") and not cls.poll(context):
            continue
        panel = type("P", (), {"layout": Layout(log)})()
        cls.draw(panel, context)
    return log


def _call(fn, **kw):
    try:
        return sorted(fn(**kw)), None
    except RuntimeError as e:
        return None, str(e)


def _pump(jobs, until=None, timeout=120.0):
    ok = jobs.pump(until, timeout)
    if not ok:
        raise RuntimeError(f"timed out; jobs: {[j.line() for j in jobs.JOBS]}; status {jobs.STATUS}")


# --- open: a real project, read only --------------------------------------------------------------
def open_mode(job):
    import afv_addon
    from afv_addon import browser, browser_jobs, ops

    afv_addon.register()
    out = {}
    try:
        t0 = time.monotonic()
        res, err = _call(bpy.ops.afv.browser_open, directory=job["project"])
        out["open_call"] = [res, err, time.monotonic() - t0]
        _pump(browser_jobs, timeout=120)
        out["open_s"] = time.monotonic() - t0
        out["times"] = dict(browser.TIMES)
        out["status"] = browser_jobs.STATUS
        out["rows"] = len(browser.SESSION.rows)
        scene = browser.browser_scene()
        out["markers"] = len(scene.timeline_markers)
        out["frames"] = [scene.frame_start, scene.frame_end, scene.render.fps, scene.render.fps_base]
        t1 = time.monotonic()
        icons = [browser.thumb_icon(r) for r in browser.SESSION.all_rows()]
        out["thumbs"] = {"n": sum(1 for i in icons if i), "s": time.monotonic() - t1}
        out["draw"] = draw_panels(bpy.context)[:80]
        # a second open (worker already running)
        t0 = time.monotonic()
        bpy.ops.afv.browser_open(directory=job["project"])
        _pump(browser_jobs, timeout=60)
        out["reopen_s"] = time.monotonic() - t0
        out["worker"] = ops.worker_status()
    finally:
        afv_addon.unregister()
    return out


def _rows(browser):
    return {k: {"state": r.get("state"), "start": r.get("start"), "end": r.get("end"),
                "speed": r.get("speed_factor"), "notes": r.get("notes"), "picked": r.get("picked"),
                "scan": (r.get("scan") or {}).get("status"), "people": (r.get("scan") or {}).get("people"),
                "thumb": (r.get("scan") or {}).get("thumbnail")} for k, r in browser.SESSION.rows.items()}


def _timed(jobs, fn, until=None, timeout=60.0):
    """Call ``fn`` (an operator call), pump the jobs it started; (seconds, result)."""
    t0 = time.monotonic()
    res = fn()
    _pump(jobs, until, timeout)
    return time.monotonic() - t0, res


def _pick_flow(browser, jobs, shot, picks, timeout=120.0):
    """Pick fighters on ``shot`` with the panel's operators: start, wait for the people, choose, save."""
    from afv_addon import browser_draw

    pk = browser_draw.PICK
    t0 = time.monotonic()
    res = sorted(bpy.ops.afv.browser_pick(shot=shot))
    _pump(jobs, lambda: not (pk.active and pk.waiting), timeout)
    shown = time.monotonic() - t0
    info = {"result": res, "shown_s": shown, "boxes": len(pk.boxes()), "source": pk.source, "frame": pk.frame,
            "image": os.path.basename(pk.image), "size": list(pk.size), "active": pk.active,
            "preselected": list(pk.selected), "draw": draw_panels(bpy.context)}
    cam = bpy.data.objects.get(browser_draw.CAMERA_NAME)
    info["backgrounds"] = [bg.source for bg in cam.data.background_images if bg.show_background_image]
    pk.selected.clear()
    for k in picks:
        bpy.ops.afv.browser_pick_box(index=k)
    info["roles"] = {str(k): pk.role(k) for k in picks}
    t1 = time.monotonic()
    bpy.ops.afv.browser_pick_save()
    _pump(jobs, lambda: not any((j.cmd == "pick" and j.args.get("fighters")) or j.lane == "list"
                                for j in jobs.JOBS), timeout)
    info["save_s"] = time.monotonic() - t1
    info["picked"] = browser.SESSION.rows[shot].get("picked")
    return info


def flow_mode(job):
    import afv_addon
    from afv_addon import browser, browser_draw, browser_jobs, browser_run, ops, serve

    serve.serve_argv = lambda distro, afv: list(job["worker"])  # the fake afv serve
    afv_addon.register()
    ops.STATE.serve_default = True
    out = {}
    jobs = browser_jobs
    sent = []  # every request the browser submits: (cmd, args)
    real_submit = jobs.submit

    def counting_submit(lane, cmd, args, *a, **kw):
        sent.append((cmd, dict(args)))
        return real_submit(lane, cmd, args, *a, **kw)

    jobs.submit = counting_submit
    try:
        out["draw_empty"] = draw_panels(bpy.context)
        res, err = _call(bpy.ops.afv.browser_new, filepath=job["video"], project_dir=job["project"],
                         scan_after=False)
        out["new_call"] = [res, err]
        _pump(jobs, timeout=180)
        if browser.browser_scene() is None:
            raise RuntimeError(f"New project made no browser scene: {jobs.STATUS}")
        s = browser.SESSION
        scene = browser.browser_scene()
        cam = bpy.data.objects.get(browser_draw.CAMERA_NAME)
        out["new"] = {"status": jobs.STATUS["text"], "rows": _rows(browser), "markers": [
            (m.name, m.frame) for m in scene.timeline_markers], "cuts": s.cuts, "frames": [
            scene.frame_start, scene.frame_end], "fps": scene.render.fps / scene.render.fps_base,
            "res": [scene.render.resolution_x, scene.render.resolution_y], "clip": [
            bg.clip.name if bg.clip else None for bg in cam.data.background_images],
            "camera": scene.camera.name if scene.camera else None, "source_fps": s.fps}
        res, err = _call(bpy.ops.afv.browser_new, filepath=job["video"], project_dir=job["project"])
        out["new_again"] = [res, err]

        # select and jump
        bpy.ops.afv.browser_select(shot="002")
        p = scene.afv_shots
        out["select"] = {"frame": scene.frame_current, "preview": [scene.use_preview_range,
                         scene.frame_preview_start, scene.frame_preview_end], "speed": p.speed}

        times = {}
        # set in / out from the playhead
        scene.frame_set(scene.frame_current + 5)
        times["set_in"], _ = _timed(jobs, lambda: bpy.ops.afv.browser_set_edge(edge="in"))
        r = s.rows["002"]
        out["set_in"] = {"frame": scene.frame_current, "start": r["start"], "end": r["end"],
                         "preview": [scene.frame_preview_start, scene.frame_preview_end]}
        scene.frame_set(scene.frame_current + 10)
        times["set_out"], _ = _timed(jobs, lambda: bpy.ops.afv.browser_set_edge(edge="out"))
        r = s.rows["002"]
        out["set_out"] = {"frame": scene.frame_current, "start": r["start"], "end": r["end"],
                          "preview": [scene.frame_preview_start, scene.frame_preview_end]}
        scene.frame_set(int(scene.frame_preview_start) - 1)
        res, err = _call(bpy.ops.afv.browser_set_edge, edge="out")  # out before in: afv refuses
        _pump(jobs)
        out["set_bad"] = {"status": dict(jobs.STATUS), "end": s.rows["002"]["end"]}
        # speed and notes (the fields send shot_set)
        t0 = time.monotonic()
        n0 = len(sent)
        for v in (0.3, 0.4, 0.45, 0.5):  # dragging the slider
            p.speed = v
        before = len(sent) - n0
        browser.send_speed()  # what the timer does once the slider rests
        _pump(jobs)
        out["speed_debounce"] = {"while_dragging": before, "after": [c for c, _ in sent[n0:]]}
        p.notes = "right hand"
        _pump(jobs)
        times["speed_notes"] = time.monotonic() - t0
        out["fields"] = {"speed": s.rows["002"].get("speed_factor"), "notes": s.rows["002"].get("notes")}
        # queue / skip
        times["queue"], _ = _timed(jobs, lambda: bpy.ops.afv.browser_state(state="queued", shot="001"))
        times["skip"], _ = _timed(jobs, lambda: bpy.ops.afv.browser_state(state="skipped", shot="003"))
        out["states"] = {k: r["state"] for k, r in s.rows.items()}
        out["counts"] = s.counts()
        # filters and order
        p.state_filter = "queued"
        out["filter_queued"] = [r["id"] for r in s.visible(p.state_filter, p.order)]
        p.state_filter = "all"
        # add shot here (playhead inside shot 001)
        scene.frame_set(10)
        times["add"], _ = _timed(jobs, lambda: bpy.ops.afv.browser_add())
        out["add"] = {"selected": p.shot, "row": s.rows.get(p.shot), "status": jobs.STATUS["text"]}
        new_id = p.shot
        # run queue before any pick: refused, the queued shots listed
        res, err = _call(bpy.ops.afv.browser_run)
        out["run_unpicked"] = {"result": res, "error": err, "missing": list(browser_run.RUNQ["missing"]),
                               "batch": jobs.busy("batch"), "draw": draw_panels(bpy.context)}
        # scan (fake track worker), rows arrive as each shot is scanned
        seen = []
        times["scan"], _ = _timed(jobs, lambda: bpy.ops.afv.browser_scan(), timeout=300)
        out["scan"] = {"status": jobs.STATUS["text"], "rows": _rows(browser)}
        out["thumb_files"] = {k: os.path.isfile(os.path.join(job["project"], r["scan"]["thumbnail"]))
                              for k, r in s.rows.items() if (r.get("scan") or {}).get("thumbnail")}
        del seen
        # pick fighters from the scan, then the next queued shot follows (Then pick the next)
        p.pick_next = True
        out["pick1"] = _pick_flow(browser, jobs, "001", [1, 0])
        out["pick_next_started"] = {"active": browser_draw.PICK.active, "shot": browser_draw.PICK.shot}
        _pump(jobs, lambda: not browser_draw.PICK.waiting)
        if browser_draw.PICK.active:
            for k in (0, 1):
                bpy.ops.afv.browser_pick_box(index=k)
            bpy.ops.afv.browser_pick_save()
            _pump(jobs)
        out["pick2"] = {"shot": new_id, "picked": s.rows[new_id].get("picked")}
        # pick cancel and a wrong number
        bpy.ops.afv.browser_pick(shot="002")
        _pump(jobs, lambda: not browser_draw.PICK.waiting)
        res, err = _call(bpy.ops.afv.browser_pick_box, index=7)
        out["pick_bad_box"] = [res, err]
        bpy.ops.afv.browser_pick_cancel()
        out["pick_cancel"] = {"active": browser_draw.PICK.active, "backgrounds": [
            bg.source for bg in cam.data.background_images if bg.show_background_image],
            "picked": s.rows["002"].get("picked")}
        # run queue: every queued shot picked now
        p.run_to = "smooth"
        t0 = time.monotonic()
        res, err = _call(bpy.ops.afv.browser_run)
        out["run_call"] = [res, err]
        out["run_args"] = dict(jobs.job("batch").args) if jobs.busy("batch") else None
        _pump(jobs, lambda: browser_run.RUNQ.get("started") is not None or not jobs.busy("batch"), 120)
        times["batch_start"] = time.monotonic() - t0
        out["run_draw"] = draw_panels(bpy.context)
        _pump(jobs, timeout=300)
        out["run"] = {"outcomes": browser_run.RUNQ["outcomes"], "summary": browser_run.RUNQ["summary"],
                      "status": jobs.STATUS["text"], "total_s": time.monotonic() - t0}
        # a trimmed queued shot: Pick scans it (with every other queued shot that needs it) in one job first
        bpy.ops.afv.browser_select(shot="001")
        scene.frame_set(scene.frame_current + 2)
        bpy.ops.afv.browser_set_edge(edge="in")
        _pump(jobs)
        out["trimmed"] = {"scan": s.rows["001"]["scan"]["status"],
                          "pick_stale": s.rows["001"].get("pick_stale"),
                          "needs_scan": browser.core.needs_scan(s.all_rows())}
        n0 = len(sent)
        p.pick_next = False
        bpy.ops.afv.browser_pick(shot="001")
        out["stale_pick_wait"] = {"waiting": browser_draw.PICK.waiting, "text": browser_draw.PICK.wait_text,
                                  "help": browser_draw.help_line()}
        _pump(jobs, lambda: not browser_draw.PICK.waiting, 120)
        out["stale_pick"] = {"source": browser_draw.PICK.source, "boxes": len(browser_draw.PICK.boxes()),
                             "requests": [(c, a.get("shots"), a.get("frame")) for c, a in sent[n0:]]}
        # a refused save keeps the clicks
        browser_draw.PICK.selected[:] = [0, 6]  # person 6 does not exist: afv refuses
        bpy.ops.afv.browser_pick_save()
        _pump(jobs, lambda: not browser_draw.PICK.saving, 60)
        out["refused_save"] = {"active": browser_draw.PICK.active,
                               "selected": list(browser_draw.PICK.selected),
                               "saving": browser_draw.PICK.saving, "status": dict(jobs.STATUS),
                               "picked": s.rows["001"].get("picked") and not s.rows["001"].get("pick_stale")}
        browser_draw.PICK.selected[:] = [1, 0]
        bpy.ops.afv.browser_pick_save()
        _pump(jobs)
        out["fixed_save"] = {"active": browser_draw.PICK.active, "picked": s.rows["001"].get("picked"),
                             "stale": s.rows["001"].get("pick_stale")}
        # the keys (as the keymap calls them: mouse over the browser scene's views)
        keys = {}
        p.order, p.state_filter, p.advance = "time", "all", True
        bpy.ops.afv.browser_select(shot="002")
        with bpy.context.temp_override(scene=scene):
            keys["poll"] = bpy.types.AFV_OT_browser_key.poll(bpy.context)
            bpy.ops.afv.browser_key(action="next")
            keys["next"] = p.shot
            bpy.ops.afv.browser_key(action="prev")
            keys["prev"] = p.shot
            bpy.ops.afv.browser_key(action="skip")
            keys["after_skip_selected"] = p.shot
            _pump(jobs)
            keys["skipped"] = s.rows["002"]["state"]
            bpy.ops.afv.browser_key(action="next_likely")
            keys["next_likely"] = p.shot
            bpy.ops.afv.browser_select(shot="002")
            bpy.ops.afv.browser_key(action="queue_pick")
            _pump(jobs, lambda: not browser_draw.PICK.waiting, 60)
            keys["queue_pick"] = {"state": s.rows["002"]["state"], "picking": browser_draw.PICK.shot}
            bpy.ops.afv.browser_pick_cancel()
        out["keys"] = keys
        # Import take of a shot without a take: afv is asked to write it (here it cannot: no results)
        res, err = _call(bpy.ops.afv.browser_take, shot="003")
        _pump(jobs, timeout=60)
        out["take_missing"] = [res, err, jobs.STATUS["text"], [sc.name for sc in bpy.data.scenes]]
        out["times"] = times
        out["browser_times"] = dict(browser.TIMES)
        out["draw_end"] = draw_panels(bpy.context)

        # the one-shot fallback (real afv CLI in WSL: listing and edits need no stages)
        if job.get("oneshot"):
            from afv_addon import wsl

            real = wsl.afv_argv
            wsl.afv_argv = lambda args, distro="Ubuntu", afv=None: real(args, distro=distro,
                                                                        afv=job["oneshot"])
            ops.shutdown_serve()
            ops.STATE.serve_default = False
            fb = {}
            t0 = time.monotonic()
            res, err = _call(bpy.ops.afv.browser_open, directory=job["project"])  # no batch_status one-shot
            _pump(jobs, timeout=120)
            fb["open"] = [res, err, jobs.STATUS["text"]]
            fb["refresh_s"] = time.monotonic() - t0
            fb["rows"] = len(s.rows)
            bpy.ops.afv.browser_select(shot="003")
            scene.frame_set(scene.frame_current + 3)
            fb["set_in_s"], _ = _timed(jobs, lambda: bpy.ops.afv.browser_set_edge(edge="in"), timeout=120)
            fb["set_in_start"] = s.rows["003"]["start"]
            fb["queue_s"], _ = _timed(jobs, lambda: bpy.ops.afv.browser_state(state="queued", shot="003"),
                                      timeout=120)
            fb["state"] = s.rows["003"]["state"]
            scene.frame_set(140)
            fb["add_s"], _ = _timed(jobs, lambda: bpy.ops.afv.browser_add(), timeout=120)
            _pump(jobs, timeout=120)
            fb["rows_after_add"] = len(s.rows)
            fb["pick"] = _pick_flow(browser, jobs, "002", [0, 1])
            fb["pick"].pop("draw", None)
            fb["status"] = jobs.STATUS["text"]
            fb["worker"] = ops.worker_status()
            out["fallback"] = fb
        # another file: the project's thumbnails are freed, nothing is left running here
        draw_panels(bpy.context)
        before = browser.preview_count()
        bpy.ops.wm.read_homefile(use_empty=True)
        out["previews"] = {"before": before, "after": browser.preview_count(), "jobs": len(jobs.JOBS),
                           "session": browser.SESSION.open}
    finally:
        jobs.submit = real_submit
        afv_addon.unregister()
    return out


def _two_largest(boxes):
    """A scripted stand-in for the owner's choice: the two largest people (fighter A the larger)."""
    area = sorted(((abs((b[2] - b[0]) * (b[3] - b[1])), k) for k, b in boxes), reverse=True)
    return [k for _, k in area[:2]]


def e2e_mode(job):
    """Open, (re)scan, queue n fighters-likely candidates, pick fighters in each, start the queue."""
    import afv_addon
    from afv_addon import browser, browser_draw, browser_jobs, browser_run, ops

    jobs = browser_jobs
    afv_addon.register()
    ops.STATE.serve_default = True
    out, steps = {}, []
    pk = browser_draw.PICK
    sent = []
    real_submit = jobs.submit

    def counting_submit(lane, cmd, args, *a, **kw):
        sent.append((cmd, args.get("shots"), args.get("frame"), args.get("fighters") is not None))
        return real_submit(lane, cmd, args, *a, **kw)

    jobs.submit = counting_submit
    try:
        # 1. open (cold worker: the first request starts afv serve)
        t0 = time.monotonic()
        bpy.ops.afv.browser_open(directory=job["project"])
        _pump(jobs, timeout=120)
        steps.append(("open", time.monotonic() - t0))
        s = browser.SESSION
        scene = browser.browser_scene()
        out["open"] = {"rows": len(s.rows), "markers": len(scene.timeline_markers),
                       "clip": bool(scene.camera and scene.camera.data.background_images),
                       "status": jobs.STATUS["text"]}
        # 2. (re)scan
        t0 = time.monotonic()
        bpy.ops.afv.browser_scan(force=job.get("scan") == "force")
        prog = []

        def scanning():
            j = jobs.job("scan")
            if j is not None and j.frac is not None and (not prog or prog[-1][1] != round(j.frac, 2)):
                prog.append((round(time.monotonic() - t0, 1), round(j.frac, 2)))
            return j is None

        _pump(jobs, scanning, timeout=900)
        _pump(jobs, timeout=60)  # the refresh after it
        steps.append(("scan", time.monotonic() - t0))
        likely = [r["id"] for r in s.visible("candidate", "likely") if browser.core.likely(r)]
        out["scan"] = {"status": jobs.STATUS["text"], "likely": len(likely), "progress": prog[:: max(1, len(
            prog) // 20)], "scanned_ok": sum(1 for r in s.all_rows() if (r.get("scan") or {}).get("status")
                                             == "ok")}
        # 3. queue n fighters-likely candidates (in time order), each as the owner would: click, Queue
        chosen = sorted(likely[: int(job.get("queue", 20))], key=lambda i: float(s.rows[i]["start"]))
        qt = []
        for sid in chosen:
            t1 = time.monotonic()
            bpy.ops.afv.browser_select(shot=sid)
            bpy.ops.afv.browser_state(state="queued")
            _pump(jobs, lambda sid=sid: s.rows[sid]["state"] == "queued" and not jobs.busy("edit"), 30)
            qt.append(time.monotonic() - t1)
        steps.append(("queue", sum(qt)))
        if job.get("trim"):  # trim every chosen shot by a few frames: their scans and picks go stale
            tt = []
            for sid in chosen:
                t1 = time.monotonic()
                bpy.ops.afv.browser_select(shot=sid)
                scene.frame_set(scene.frame_current + 3)
                bpy.ops.afv.browser_set_edge(edge="in")
                _pump(jobs, lambda sid=sid: not jobs.busy("edit"), 30)
                tt.append(time.monotonic() - t1)
            steps.append(("trim", sum(tt)))
            out["trim"] = {"each_max_s": max(tt), "stale": sum(1 for sid in chosen
                                                          if s.rows[sid]["scan"]["status"] != "ok")}
            n_sent = len(sent)
        out["queue"] = {"shots": chosen, "each_s": [round(x, 3) for x in qt], "max_s": max(qt or [0])}
        # 4. pick fighters in each: Pick fighters on the first, then "Then pick the next" brings the next one
        scene.afv_shots.pick_next = True
        picks, t_pick = [], time.monotonic()
        bpy.ops.afv.browser_pick(shot=chosen[0])
        while True:
            t1 = time.monotonic()
            _pump(jobs, lambda: not pk.waiting or not pk.active, 300)
            if not pk.active:
                break
            shown = time.monotonic() - t1
            shot, boxes = pk.shot, pk.boxes()
            sel = _two_largest(boxes)
            entry = {"shot": shot, "people": len(boxes), "source": pk.source, "shown_s": round(shown, 3)}
            if len(sel) < 2:
                entry["skipped"] = "fewer than two people"
                bpy.ops.afv.browser_pick_cancel()
                picks.append(entry)
                nxt = browser.core.next_to_pick(s.all_rows(), shot)
                if nxt is None or any(p["shot"] == nxt for p in picks):
                    break
                bpy.ops.afv.browser_pick(shot=nxt)
                continue
            pk.selected.clear()
            for k in sel:
                bpy.ops.afv.browser_pick_box(index=k)
            t2 = time.monotonic()
            bpy.ops.afv.browser_pick_save()
            _pump(jobs, lambda shot=shot: s.rows[shot].get("picked") and not any(
                j.cmd == "pick" and j.args.get("fighters") for j in jobs.JOBS), 60)
            entry["save_s"] = round(time.monotonic() - t2, 3)
            entry["fighters"] = sel
            picks.append(entry)
            if len(picks) > len(chosen) + 2:
                break
        steps.append(("pick", time.monotonic() - t_pick))
        if job.get("trim"):
            req = sent[n_sent:]
            out["pick_requests"] = {"scan_jobs": [r[1] for r in req if r[0] == "shots_scan"],
                                    "prompts": sum(1 for r in req if r[0] == "pick" and not r[3]),
                                    "detections": sum(1 for r in req if r[0] == "pick" and not r[3] and r[2]
                                                      is not None),
                                    "scan": dict(browser.TIMES).get("scan")}
        out["pick"] = picks
        ready, missing = browser.core.queue_check(s.all_rows())
        out["queue_check"] = {"ready": len(ready), "missing": missing}
        # 5. start the queue (then cancel it: this is a scratch copy, not a night's run)
        t0 = time.monotonic()
        res, err = _call(bpy.ops.afv.browser_run, only_ready=bool(missing))
        _pump(jobs, lambda: browser_run.RUNQ.get("started") is not None or not jobs.busy("batch"), 300)
        steps.append(("start queue", time.monotonic() - t0))
        out["run"] = {"call": [res, err], "started": browser_run.RUNQ.get("started"),
                      "current": dict(browser_run.RUNQ.get("current") or {})}
        t1 = time.monotonic()
        _pump(jobs, lambda: (browser_run.RUNQ.get("current") or {}).get("stage"), 120)
        out["run"]["first_stage"] = dict(browser_run.RUNQ.get("current") or {})
        out["run"]["first_stage_s"] = time.monotonic() - t1
        bpy.ops.afv.browser_cancel(lane="batch")
        _pump(jobs, lambda: not jobs.busy("batch"), 300)
        out["run"]["after_cancel"] = {"summary": browser_run.RUNQ["summary"], "status": jobs.STATUS["text"]}
        out["steps"] = [(n, round(t, 2)) for n, t in steps]
        out["total_s"] = round(sum(t for _, t in steps), 1)
        out["times"] = dict(browser.TIMES)
    finally:
        jobs.submit = real_submit
        out.setdefault("steps", [(n, round(t, 2)) for n, t in steps])
        afv_addon.unregister()
    return out


def runq_mode(job):
    """Queue and pick ``shots`` (everything else queued is skipped), Run queue to ``to``, to the end."""
    import afv_addon
    from afv_addon import browser, browser_jobs, browser_run, ops

    jobs = browser_jobs
    afv_addon.register()
    ops.STATE.serve_default = True
    out = {}
    try:
        bpy.ops.afv.browser_open(directory=job["project"])
        _pump(jobs, timeout=120)
        s = browser.SESSION
        for sid, r in list(s.rows.items()):
            if r["state"] == "queued" and sid not in job["shots"]:
                bpy.ops.afv.browser_state(state="skipped", shot=sid)
        for sid in job["shots"]:
            bpy.ops.afv.browser_state(state="queued", shot=sid)
        _pump(jobs, timeout=60)
        for sid in job["shots"]:
            if not s.rows[sid].get("picked") or s.rows[sid].get("pick_stale"):
                bpy.ops.afv.browser_pick(shot=sid)
                _pump(jobs, lambda: not browser.PICK.waiting, 300)
                for k in _two_largest(browser.PICK.boxes()):
                    bpy.ops.afv.browser_pick_box(index=k)
                bpy.ops.afv.browser_pick_save()
                _pump(jobs, timeout=60)
        browser.props().run_to = job.get("to", "detect_track")
        seen = []
        t0 = time.monotonic()
        bpy.ops.afv.browser_run()

        def watch():
            j = jobs.job("batch")
            if j is not None:
                line = (round(time.monotonic() - t0, 1), round(j.frac or 0.0, 3), j.message)
                if not seen or seen[-1][2] != line[2]:
                    seen.append(line)
            return j is None

        _pump(jobs, watch, timeout=1800)
        _pump(jobs, timeout=60)
        out["seconds"] = time.monotonic() - t0
        out["progress"] = seen[:: max(1, len(seen) // 40)]
        out["outcomes"] = browser_run.RUNQ["outcomes"]
        out["summary"] = browser_run.RUNQ["summary"]
        out["rows"] = {sid: {"stored": s.rows[sid].get("stored"), "state": s.rows[sid]["state"]}
                       for sid in job["shots"]}
        out["draw"] = draw_panels(bpy.context)
    finally:
        afv_addon.unregister()
    return out


def detached_mode(job):
    """The project's batch as afv serve shows it (stub serve ``browser_stub_serve.py``, the real protocol):
    ``kind`` "stream": a running batch; Open project attaches to it; File > New and disabling the add-on
    send no cancel; Stop batch says what ran. "poll": an afv whose batch cannot attach: batch_status every
    few seconds. "finished": the last batch's results on Open project, per shot in the list; Import take
    into the shot's own scene, once. Also: a row afv cannot read, the missing video, the pick arrow's poll."""
    import afv_addon
    from afv_addon import browser, browser_jobs, browser_pick, browser_run, ops, serve

    stub = os.path.join(job["tests_dir"], "browser_stub_serve.py")
    serve.serve_argv = lambda distro, afv: [sys.executable, stub, job["cfg"]]
    jobs = browser_jobs
    rq = browser_run.RUNQ
    afv_addon.register()
    ops.STATE.serve_default = True
    out = {}
    kind = job["kind"]

    def state():
        j = jobs.job("batch")
        return {"batch": dict(rq["batch"] or {}), "current": dict(rq["current"] or {}),
                "summary": rq["summary"], "detached": browser_run.detached(),
                "running": browser_run.running(),
                "streaming": j is not None, "line": j.line() if j is not None else "",
                "frac": j.frac if j is not None else None, "outcomes": list(rq["outcomes"]),
                "row_outcomes": dict(browser.SESSION.outcomes),
                "run_poll": browser_run.AFV_OT_browser_run.poll(bpy.context),
                "stop_poll": browser_run.AFV_OT_browser_stop_batch.poll(bpy.context)}

    def requests():
        with open(job["log"], encoding="utf-8") as fh:
            return [json.loads(x)["cmd"] for x in fh if x.strip()]

    def open_it():
        bpy.ops.afv.browser_open(directory=job["project"])
        if kind == "stream":
            _pump(jobs, lambda: (rq.get("current") or {}).get("stage") == "pose3d"
                  and (rq.get("current") or {}).get("shot") == "002", 60)
        else:
            _pump(jobs, timeout=60)

    try:
        open_it()
        out["open"] = state()
        out["draw_running"] = draw_panels(bpy.context)
        s = browser.SESSION
        scene = browser.browser_scene()
        out["video"] = {"warning": s.video_warning,
                        "clip": scene.active_clip.name if scene.active_clip else None,
                        "backgrounds": len(scene.camera.data.background_images) if scene.camera else None}
        browser.props().shot = ""
        out["pick_poll_unselected"] = browser_pick.AFV_OT_browser_pick.poll(bpy.context)
        if kind == "stream":  # File > New with the batch streaming: nothing reaches afv but the new file
            bpy.ops.wm.read_homefile(use_empty=True)
            _pump(jobs, timeout=5)
            out["after_new_file"] = {"requests": requests(), "jobs": [j.lane for j in jobs.JOBS],
                                     "session": s.open}
            open_it()  # attaches again
            out["reattached"] = state()
        if kind == "poll":
            browser_run._poll_status()  # what the timer does every STATUS_POLL_S
            _pump(jobs, timeout=30)
            out["polled"] = state()
        if kind in ("stream", "poll"):
            bpy.ops.afv.browser_stop_batch()
            _pump(jobs, timeout=30)
            out["stopped"] = {**state(), "status": browser_jobs.STATUS["text"]}
        if kind == "finished":
            out["finished_draw"] = draw_panels(bpy.context)
            out["take"] = []
            other = bpy.data.scenes.new("Elsewhere")
            where = {"scene": scene}  # the window's scene (background: no window), as the owner moves
            real_ws = browser_run._window_scene
            browser_run._window_scene = lambda: where["scene"]
            for i in range(2):  # the second time: the same scene again, no duplicate
                res, err = _call(bpy.ops.afv.browser_take, shot="001")
                if i == 0:
                    where["scene"] = other  # the owner went elsewhere while afv writes the take
                _pump(jobs, timeout=60)
                if i == 0:
                    out["take_ready"] = {"ready": dict(browser_run.TAKE_READY), "status": jobs.STATUS["text"],
                                         "show_poll": browser_run.AFV_OT_browser_show_take.poll(bpy.context)}
                    bpy.ops.afv.browser_show_take()
                    out["take_ready"]["after_show"] = dict(browser_run.TAKE_READY)
                ts = browser_run.take_scene("001")
                out["take"].append({"result": res, "error": err, "status": jobs.STATUS["text"],
                                    "scene": ts.name if ts else None,
                                    "armatures": sorted(o.name for o in ts.objects if o.type == "ARMATURE")
                                    if ts else [], "scenes": sorted(sc.name for sc in bpy.data.scenes)})
            browser_run._window_scene = real_ws
            out["browser_scene_objects"] = sorted(o.name for o in scene.objects)
        out["bad_row"] = {"line": browser.core.row_line(s.rows["003"]),
                          "missing": browser.core.queue_check(s.all_rows())[1], "counts": s.counts()}
        bpy.ops.afv.browser_select(shot="003")
        out["bad_row"]["selected"] = browser.props().shot
        out["draw_end"] = draw_panels(bpy.context)
    finally:
        afv_addon.unregister()
    time.sleep(0.5)
    out["requests"] = requests()
    return out


def main():
    job_path = sys.argv[sys.argv.index("--") + 1]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    sys.path.insert(0, job["addon_parent"])
    sys.path.insert(0, job["tests_dir"])
    result = {"ok": False}
    try:
        modes = {"open": open_mode, "flow": flow_mode, "e2e": e2e_mode, "runq": runq_mode,
                 "detached": detached_mode}
        result["out"] = modes[job["mode"]](job)
        result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"])
    with open(job["result_path"], "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if result["ok"] else 1)  # the worker's reader threads must not keep Blender alive


if __name__ == "__main__":
    main()
