"""Fake CPU stages for the ``afv serve`` tests (imported by worker_main.py from a copy in the test's
folder, so a test can change the code on disk). Chain: ingest; pose3d -> contacts -> smooth (the
last two range-capable, like the real ones)."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
from pydantic import BaseModel

from afv.core.stage import register
from afv.core.types import ActorMotion, BodyMotion, CameraTrack

T = 60


def timeline() -> dict:
    return {"frames": np.arange(T), "timestamps": np.arange(T) / 30.0}


def motion(root: np.ndarray) -> BodyMotion:
    rot = np.zeros((T, 2, 4), np.float32)
    rot[..., 3] = 1
    return BodyMotion(
        **timeline(), skeleton="soma_provisional", fps=30.0,
        actors={"A": ActorMotion(rot=rot, root=np.asarray(root, np.float32))},
    )  # fmt: skip


def blend(ctx, new: np.ndarray) -> np.ndarray:
    if ctx.frame_range is None:
        return new
    a, b = ctx.frame_range
    out = ctx.previous.actors["A"].root.copy()
    out[a:b] = new[a:b]
    (ctx.work_dir / "qc.json").write_text(json.dumps({"range": {"frames": [a, b]}}))  # as the real ones
    return out


@register("default")
class Ingest:
    name, version, env, inputs, licences, weight_hash = "ingest", "1", "cpu", (), (), ""

    class Params(BaseModel):
        pass

    def run(self, ctx, inputs, params):
        print("a stray print on stdout")  # must not reach the protocol stream
        return CameraTrack(**timeline(), width=64, height=64)


@register("default")
class Pose:
    name, version, env, inputs, licences, weight_hash = "pose3d", "1", "cpu", (), (), ""

    class Params(BaseModel):
        pass

    def run(self, ctx, inputs, params):
        return motion(np.tile(np.arange(T, dtype=np.float32)[:, None], (1, 3)))


@register("default")
class Contacts:
    name, version, env, inputs, licences, weight_hash = "contacts", "1", "cpu", ("pose3d",), (), ""
    edits = ("contacts",)
    supports_range = True

    class Params(BaseModel):
        hold_s: float = 0.0  # a slow stage: progress calls meanwhile (cancel checks)
        child_s: float = 0.0  # wait for a child process this long (a stand-in for a model worker)
        pre_s: float = 0.0  # sleep before starting the child (cancel before it exists)
        offset: float = 0.0

    def run(self, ctx, inputs, params):
        end = time.monotonic() + params.hold_s
        while time.monotonic() < end:
            ctx.progress(0.5, "holding")
            time.sleep(0.05)
        time.sleep(params.pre_s)
        if params.child_s:
            p = subprocess.Popen(
                [sys.executable, "-c", f"import time; time.sleep({params.child_s})"], start_new_session=True
            )
            (ctx.work_dir / "child.pid").write_text(str(p.pid))
            print(f"CHILD {p.pid}", file=sys.stderr, flush=True)
            p.wait()
        new = inputs["pose3d"].actors["A"].root + params.offset
        for sp in (ctx.edits.get("contacts") or {}).get("spans", []):  # a painted span adds 1 there
            new[sp["start"] : sp["end"]] += 1
        ctx.progress(1.0, "contacts done")
        return motion(blend(ctx, new))


@register("default")
class Smooth:
    name, version, env, inputs, licences, weight_hash = "smooth", "1", "cpu", ("contacts",), (), ""
    supports_range = True

    class Params(BaseModel):
        strength: float = 0.5

    def run(self, ctx, inputs, params):
        ctx.progress(0.5, "smoothing")
        new = inputs["contacts"].actors["A"].root * params.strength
        return motion(blend(ctx, new))
