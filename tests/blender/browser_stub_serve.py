"""A stub ``afv serve`` for the shot browser's tests of the detached batch, following the real protocol of
``src/afv/serve.py`` (round 6): standard library only, run by Blender's own Python.

    python browser_stub_serve.py config.json

config.json: {"log": path (each request appended as a JSON line), "shots": the ``shots`` result,
"statuses": [``batch_status`` results, one per call; the last repeats], "after_cancel": the
``batch_status`` once ``batch_cancel`` was received, "stream": optional [events] of a running batch,
"take": optional {"src": a take folder, "root_win": the project folder}}.

- ``batch`` with "stream": attaches (``batch_started`` ``attached: true`` first, then the events) and stays
  open, like a batch that runs; ``batch_cancel`` ends it as serve does (error ``cancelled`` with the
  ``report``) and answers ``after_cancel`` plus ``cancelled: true``; a ``cancel`` of the stream does the same
  (serve treats it as ``batch_cancel``). Without "stream", ``batch`` is an unknown command.
- ``take`` copies "src" into ``shots/<shot>/take`` (take.json last) and answers its path.
- Any other command gets an ``unknown cmd`` error, like an afv that does not have it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

CFG = json.load(open(sys.argv[1], encoding="utf-8"))
STATE = {"status": 0, "cancelled": False, "batch": None}


def say(rid, event, **kw):
    sys.stdout.write(json.dumps({"id": rid, "event": event, **kw}) + "\n")
    sys.stdout.flush()


def end_batch():
    STATE["cancelled"] = True
    if STATE["batch"] is not None:
        rep = {**CFG["after_cancel"], "cancelled": True}
        say(STATE["batch"], "error", message=f"cancelled after {len(rep.get('outcomes') or [])} shots; they "
            "are kept", cancelled=True, report=rep)
        STATE["batch"] = None


for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    rid, cmd, args = req.get("id"), req.get("cmd"), req.get("args") or {}
    if CFG.get("log"):
        with open(CFG["log"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"cmd": cmd, "args": args}) + "\n")
    if cmd == "ping":
        say(rid, "result", pong=True, protocol=1)
    elif cmd == "shutdown":
        say(rid, "result", shutdown=True)
        if STATE["batch"] is not None:  # the client went away: the batch goes on
            say(STATE["batch"], "error", message="the worker is shutting down; the batch goes on",
                cancelled=True, detached=True)
        break
    elif cmd == "cancel":
        if args.get("target_id") == STATE["batch"]:
            end_batch()
        say(rid, "result", target_id=args.get("target_id"), state="running")
    elif cmd == "shots":
        say(rid, "result", **CFG["shots"])
    elif cmd == "batch_status":
        if STATE["cancelled"]:
            res = CFG["after_cancel"]
        else:
            seq = CFG["statuses"]
            res = seq[min(STATE["status"], len(seq) - 1)]
            STATE["status"] += 1
        say(rid, "result", **res)
    elif cmd == "batch" and CFG.get("stream") and not STATE["cancelled"]:
        STATE["batch"] = rid
        st = CFG["statuses"][0]
        say(rid, "log", kind="batch_started", batch_id=st.get("batch_id"), attached=True, log=st.get("log"))
        for ev in CFG["stream"]:
            say(rid, ev.get("event", "log"), **{k: v for k, v in ev.items() if k != "event"})
    elif cmd == "batch_cancel":
        end_batch()
        say(rid, "result", **CFG["after_cancel"], cancelled=True)
    elif cmd == "take" and CFG.get("take"):
        dst = os.path.join(CFG["take"]["root_win"], "shots", str(args.get("shot")), "take")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(CFG["take"]["src"], dst)
        say(rid, "result", path=os.path.join(dst, "take.json"))
    else:
        say(rid, "error", message=f"unknown cmd {cmd!r}")
