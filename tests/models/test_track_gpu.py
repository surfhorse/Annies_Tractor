"""End-to-end tracking on a synthetic video in the real track env (GPU, opt-in).

Run with ``AFV_GPU_TESTS=1 ~/afv/afv-env/bin/python -m pytest tests/models -m gpu -k track``.
Takes about a minute (model loading); holds the shared GPU lock through ``run_worker``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("AFV_GPU_TESTS") != "1", reason="set AFV_GPU_TESTS=1 to run GPU tests"),
    pytest.mark.skipif(not Path("~/afv/track/.venv/bin/python").expanduser().exists(), reason="no track env"),
]

W, H, SRC_FPS, SECONDS = 640, 360, 25, 2.4
ENTER = 1.2  # seconds: the third person walks in


def person_box(k: int, t: float) -> np.ndarray | None:
    """Ground-truth box of synthetic person k at time t (seconds)."""
    if k == 0:
        x = 80 + 60 * t
    elif k == 1:
        x = 420 - 50 * t
    else:
        if t < ENTER:
            return None
        x = W - 20 - 120 * (t - ENTER) * 2  # enters from the right edge
    return np.array([x, 60, x + 70, 330], np.float64)


def make_video(path: Path) -> None:
    import av

    rng = np.random.default_rng(0)
    textures = [rng.integers(0, 60, (H, W, 3)) for _ in range(3)]
    colours = [(220, 40, 40), (40, 80, 230), (40, 200, 60)]
    yy = np.linspace(30, 70, H)[:, None, None]
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=SRC_FPS)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        for i in range(int(SECONDS * SRC_FPS)):
            t = i / SRC_FPS
            img = np.broadcast_to(yy, (H, W, 3)).astype(np.float64).copy()
            for k in range(3):
                b = person_box(k, t)
                if b is None:
                    continue
                x0, y0, x1, y1 = (round(float(v)) for v in b)
                x0c, x1c = max(x0, 0), min(x1, W)
                if x1c <= x0c:
                    continue
                img[y0:y1, x0c:x1c] = np.array(colours[k]) * 0.8 + textures[k][y0:y1, x0c:x1c]
                cy, cx = y0 + 25, (x0 + x1) // 2  # a "head"
                img[cy - 18 : cy + 18, max(cx - 15, 0) : min(cx + 15, W)] = 230
            frame = av.VideoFrame.from_ndarray(np.clip(img, 0, 255).astype(np.uint8), format="rgb24")
            for pkt in s.encode(frame):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def test_synthetic_tracking_with_reprompt(tmp_path):
    from afv.media.normalise import shot_frame_count
    from afv.models.track import TrackParams, load_fighters_edit, run_tracking

    video = tmp_path / "synthetic.mp4"
    make_video(video)
    enter_f = int(np.ceil((ENTER + 0.3) * 30))  # fully inside by then
    box3 = person_box(2, enter_f / 30)
    assert box3 is not None
    edit = load_fighters_edit({
        "objects": [{"id": 1, "role": "A"}, {"id": 2, "role": "B"}, {"id": 3, "label": "late"}],
        "prompts": [
            {"frame": 0, "obj": 1, "box": person_box(0, 0).tolist()},
            {"frame": 0, "obj": 2, "box": person_box(1, 0).tolist()},
            {"frame": enter_f, "obj": 3, "box": box3.tolist()},
        ],
    })  # fmt: skip
    run = run_tracking(video, edit, tmp_path / "out", params=TrackParams())
    tr = run.tracks
    assert tr.num_frames == shot_frame_count(video) == 72
    assert np.allclose(tr.timestamps, np.arange(72) / 30)
    assert tr.fighters == {"A": 1, "B": 2} and tr.track_ids.tolist() == [1, 2, 3]
    ious = {k: [] for k in range(3)}
    for f in range(tr.num_frames):
        for k in range(3):
            truth = person_box(k, f / 30)
            got = tr.boxes[f, k]
            if truth is None or truth[0] > W - 40:  # absent or barely visible
                continue
            truth = truth.copy()
            truth[2] = min(truth[2], W)
            if not np.isfinite(got[0]):
                ious[k].append(0.0)
                continue
            ix = max(0, min(truth[2], got[2]) - max(truth[0], got[0]))
            iy = max(0, min(truth[3], got[3]) - max(truth[1], got[1]))
            inter = ix * iy
            union = (
                (truth[2] - truth[0]) * (truth[3] - truth[1]) + (got[2] - got[0]) * (got[3] - got[1]) - inter
            )
            ious[k].append(inter / union)
    for k in range(3):
        assert np.mean(np.array(ious[k]) > 0.7) > 0.9, (k, np.round(ious[k], 2))
    # the late person is not hallucinated before entering
    assert not np.isfinite(tr.boxes[: int(ENTER * 30) - 3, 2, 0]).any()
    assert (tmp_path / "out" / "review.mp4").stat().st_size > 0
    assert run.qc["stats"]["vram_peak_allocated_mb"] > 0
