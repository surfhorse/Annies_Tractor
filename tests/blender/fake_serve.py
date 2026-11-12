"""A fake ``afv serve`` for the add-on tests: speaks the JSON-lines protocol (docs/BUILD_PLAN.md
"Shared formats") without afv. Standard library only (runs in WSL's or Blender's Python).

    python fake_serve.py config.json

config.json (all optional):
  "log": path            every request is appended here as one JSON line
  "takes": [dir, ...]    each ``take`` request copies the next of these take packages into the shot's
                         take folder (files first, take.json last) and replies with its path
  "run_s": seconds       how long a ``run`` takes (it checks for ``cancel`` meanwhile)
  "review_once": true    the first ``run`` without ``yes`` pauses on tracking hints (needs_review)
  "die_after": n         exit (code 9) right after receiving request number n (counting pings)
  "fail_run": message    every ``run`` ends with an ``error`` event
  "distro": name         WSL distribution for /home/... paths on Windows (default Ubuntu)

Project paths in requests are WSL paths (/mnt/<drive>/...); on Windows they are mapped back to
drive paths. ``edit`` writes ``<project>/shots/<shot>/edits/<name>.json`` atomically.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import sys
import threading
import time

CFG = json.load(open(sys.argv[1], encoding="utf-8")) if len(sys.argv) > 1 else {}
OUT = threading.Lock()
CANCELLED: set[int] = set()
STATE = {"n": 0, "takes": 0, "reviewed": False}


def host(path: str) -> str:
    """A WSL path as this process sees it (Windows: a drive path or a wsl.localhost UNC path)."""
    if os.name != "nt" or not path.startswith("/"):
        return path
    m = re.match(r"^/mnt/([a-z])/(.*)$", path)
    if m:
        return f"{m.group(1).upper()}:\\" + m.group(2).replace("/", "\\")
    return "\\\\wsl.localhost\\" + CFG.get("distro", "Ubuntu") + path.replace("/", "\\")


def replace(src, dst):
    """os.replace, retried: Windows refuses briefly while another process still has the file open."""
    for _ in range(50):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(src, dst)


def say(rid, event, **kw):
    with OUT:
        sys.stdout.write(json.dumps({"id": rid, "event": event, **kw}) + "\n")
        sys.stdout.flush()


def shot_dir(args):
    return os.path.join(host(args["project"]), "shots", str(args["shot"]))


def _edits_module():
    """The add-on's ``edits`` (numpy only): the same structure checks as afv.core.edits."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "blender", "afv_addon", "edits.py")
    spec = importlib.util.spec_from_file_location("_fake_serve_edits", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EDITS = _edits_module()


def do_edit(rid, args):
    """As the real worker: validate, keep top-level keys the data leaves out, ``data: null`` deletes."""
    d = os.path.join(shot_dir(args), "edits")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{args['name']}.json")
    data = args.get("data")
    if data is None:
        if os.path.isfile(p):
            os.remove(p)
        say(rid, "result", outcome="ok", path=p, removed=True)
        return
    old = {}
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            old = json.load(fh)
    merged = {**old, **data} if isinstance(old, dict) and isinstance(data, dict) else data
    try:
        EDITS.validate(args["name"], merged)
    except EDITS.EditError as e:
        say(rid, "error", message=str(e))
        return
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=1)
    replace(tmp, p)
    say(rid, "result", outcome="ok", path=p, removed=False)


def _deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def save_sets(args):
    """``sets`` saved to shot.json settings.params before anything runs (as afv run --set)."""
    sets = args.get("sets") or []
    if not sets:
        return
    over = {}
    for item in sets:
        key, _, raw = item.partition("=")
        try:
            val = json.loads(raw)
        except ValueError:
            val = raw
        node = over
        *head, last = key.split(".")
        for k in head:
            node = node.setdefault(k, {})
        node[last] = val
    p = os.path.join(shot_dir(args), "shot.json")
    with open(p, encoding="utf-8") as fh:
        info = json.load(fh)
    st = info.setdefault("settings", {})
    params = _deep_merge(st.get("params") or {}, over)
    st["params"] = {k: v for k, v in params.items() if v}
    with open(p + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    replace(p + ".tmp", p)


def do_run(rid, args):
    save_sets(args)
    if CFG.get("fail_run"):
        say(rid, "error", message=CFG["fail_run"])
        return
    if CFG.get("review_once") and not STATE["reviewed"] and not args.get("yes"):
        STATE["reviewed"] = True
        say(rid, "tracking_review", stage="detect_track", paused=True, hints=[{"kind": "swap"}],
            message="shot: 1 tracking hint\nlikely id swap at 40", review_video=None)
        say(rid, "result", shot=args["shot"], outcome="needs_review", stages=[], message="paused",
            review={"hints": [{"kind": "swap"}]})
        return
    stages = ["contacts", "smooth", "hands_face"]
    t_end = time.monotonic() + float(CFG.get("run_s", 0.0))
    say(rid, "progress", stage=args.get("from"), frac=0.0, message="starting")
    while time.monotonic() < t_end:
        if rid in CANCELLED:
            say(rid, "error", message="cancelled")
            return
        time.sleep(0.02)
    for s in stages:
        say(rid, "progress", stage=s, frac=1.0, message="done")
    say(rid, "result", shot=args["shot"], outcome="ok", message="", exports=[],
        stages=[{"name": s, "state": "ran", "range": args.get("frames")} for s in stages])


def do_take(rid, args):
    takes = CFG.get("takes") or []
    dest = os.path.join(shot_dir(args), "take")
    if takes:
        src = takes[min(STATE["takes"], len(takes) - 1)]
        STATE["takes"] += 1
        os.makedirs(dest, exist_ok=True)
        names = sorted(n for n in os.listdir(src) if n != "take.json" and not n.endswith(".blend"))
        for n in names:
            shutil.copy2(os.path.join(src, n), os.path.join(dest, n))
        shutil.copy2(os.path.join(src, "take.json"), os.path.join(dest, ".take.json.tmp"))
        replace(os.path.join(dest, ".take.json.tmp"), os.path.join(dest, "take.json"))
    say(rid, "result", outcome="ok", path=os.path.join(dest, "take.json"))


def worker(jobs):
    while True:
        req = jobs.get()
        if req is None:
            return
        rid, cmd, args = req.get("id"), req.get("cmd"), req.get("args") or {}
        try:
            if cmd == "ping":
                say(rid, "result", outcome="ok", pong=True)
            elif cmd == "edit":
                do_edit(rid, args)
            elif cmd == "run":
                do_run(rid, args)
            elif cmd == "take":
                do_take(rid, args)
            elif cmd == "export":
                say(rid, "result", outcome="ok", exports=[f"{args['shot']}_A.fbx"])
            elif cmd == "status":
                say(rid, "result", outcome="ok", stages=[])
            else:
                say(rid, "error", message=f"unknown command {cmd!r}")
        except Exception as e:  # the fake reports everything
            say(rid, "error", message=f"{type(e).__name__}: {e}")


def main():
    jobs: queue.Queue = queue.Queue()
    t = threading.Thread(target=worker, args=(jobs,), daemon=True)
    t.start()
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        STATE["n"] += 1
        if CFG.get("log"):
            with open(CFG["log"], "a", encoding="utf-8") as fh:
                fh.write(json.dumps(req) + "\n")
        if CFG.get("die_after") and STATE["n"] >= int(CFG["die_after"]):
            os._exit(9)
        cmd = req.get("cmd")
        if cmd == "cancel":
            CANCELLED.add(int((req.get("args") or {}).get("target_id", -1)))
            say(req.get("id"), "result", outcome="ok")
        elif cmd == "shutdown":
            say(req.get("id"), "result", outcome="ok")
            break
        else:
            jobs.put(req)
    jobs.put(None)
    t.join(5)


if __name__ == "__main__":
    main()
