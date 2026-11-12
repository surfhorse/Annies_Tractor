"""``afv serve`` with the fake stages of $AFV_TEST_STAGES instead of the real stage modules
(started by the tests as a subprocess, like the add-on starts the real one).

Optional stand-ins (round 6, shot browser tests):

- ``AFV_TEST_SEGMENTS``: JSON list of {"start", "duration"|"end", "confidence"?}: what cut detection finds.
- ``AFV_TEST_FAKE_TRACK=1``: the candidate scan's track worker is a fake (no GPU): it writes the
  ``detect`` mode's files with two made-up people per frame, ``AFV_TEST_TRACK_SLEEP`` seconds per frame,
  and stops like a cancelled worker when the request is cancelled.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from afv.cli import install_signal_handlers
from afv.core import runner

runner.STAGE_MODULES = ()
spec = importlib.util.spec_from_file_location("afv_test_stages", os.environ["AFV_TEST_STAGES"])
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules["afv_test_stages"] = mod
spec.loader.exec_module(mod)

BOXES = [[30.0, 10.0, 70.0, 85.0], [90.0, 12.0, 125.0, 88.0]]


def fake_track_worker(req: dict, work_dir: Path, gpu: bool) -> dict:
    from afv.models.envs import WorkerCancelled, cancel_requested

    out = Path(req["out_dir"])
    prompts = []
    for f in req["frames"]:
        time.sleep(float(os.environ.get("AFV_TEST_TRACK_SLEEP", "0")))
        if cancel_requested():
            raise WorkerCancelled("worker track_worker.py was killed by signal 15")
        info = {"frame": f, "prompt_frame": f, "image": f"prompt_{f:05d}.jpg", "boxes": BOXES,
                "scores": [0.9, 0.8], "threshold": req["threshold"]}
        (out / f"prompt_{f:05d}.json").write_text(json.dumps(info))
        prompts.append(info)
    return {"ok": True, "num_frames": req["num_frames"], "prompts": prompts}


if os.environ.get("AFV_TEST_SEGMENTS"):
    import afv.cli

    segments = json.loads(os.environ["AFV_TEST_SEGMENTS"])
    afv.cli._shot_detector = lambda: lambda video, min_confidence=None: segments  # type: ignore[assignment]

if os.environ.get("AFV_TEST_FAKE_TRACK"):
    import afv.media.shots

    afv.media.shots.scan_shots.__kwdefaults__["worker"] = fake_track_worker

if __name__ == "__main__":
    if len(sys.argv) > 1:  # `afv <args>` with these fakes: the detached batch a test worker starts
        from afv.cli import app

        sys.exit(app())
    from afv.serve import serve

    install_signal_handlers()
    sys.exit(serve())
