"""Pure helpers of the track worker (loaded by path; the worker never imports afv)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from afv.models.track import WORKER


@pytest.fixture(scope="module")
def tw():
    spec = importlib.util.spec_from_file_location("track_worker", WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_worker_does_not_import_afv():
    src = Path(WORKER).read_text()
    assert "import afv" not in src and "from afv" not in src


def test_worker_cli_reports_errors(tw, tmp_path):
    req = tmp_path / "r.json"
    req.write_text('{"mode": "detect", "frames_dir": "/nonexistent", "num_frames": 3}')
    assert tw.main(["w", str(req), str(tmp_path / "out.json")]) == 1
    import json

    resp = json.loads((tmp_path / "out.json").read_text())
    assert resp["ok"] is False and "expected 3" in resp["error"]


def test_sort_by_area(tw):
    b, s = tw.sort_by_area(np.array([[0, 0, 1, 1], [0, 0, 10, 10], [0, 0, 5, 5]]), np.array([0.1, 0.2, 0.3]))
    assert s.tolist() == pytest.approx([0.2, 0.3, 0.1]) and b[0].tolist() == [0, 0, 10, 10]


def test_plan_passes_with_reprompts(tw):
    prompts = [
        {"frame": 0, "obj": 1},
        {"frame": 0, "obj": 2},
        {"frame": 141, "obj": 3},
        {"frame": 200, "obj": 1},
    ]
    plan = tw.plan_passes(prompts, 272)
    assert plan["first"] == {1: 0, 2: 0, 3: 141}
    assert (plan["forward_start"], plan["reverse_start"]) == (0, 141)
    assert plan["prompt_frames"] == [0, 141, 200]
    # every (object, frame) is taken from exactly one pass that actually visits the frame
    for o, first in plan["first"].items():
        for f in range(272):
            fwd = f >= plan["forward_start"] and tw.take_from_pass(f, first, False)
            rev = f <= plan["reverse_start"] and tw.take_from_pass(f, first, True)
            assert fwd != rev, (o, f)
    with pytest.raises(ValueError, match="outside"):
        tw.plan_passes([{"frame": 272, "obj": 1}], 272)


def test_prune_keys(tw):
    keys = list(range(0, 300))
    drop = set(tw.prune_keys(keys, 150, [0, 200], 20))
    kept = set(keys) - drop
    assert kept == set(range(0, 21)) | set(range(130, 171)) | set(range(180, 221))
    assert tw.prune_keys(keys, 150, [0], 0) == []


def test_resolve_prompts(tw):
    dets = {5: (np.array([[0, 0, 10, 20], [1, 1, 5, 5]], np.float32), np.array([0.9, 0.8]))}
    out = tw.resolve_prompts(
        [
            {"frame": 5, "obj": 1, "detection": 1},
            {"frame": 5, "obj": 2, "detection": 0, "box": [7, 7, 9, 9]},
            {"frame": 5, "obj": 3, "detection": 0, "points": [[1, 2, 1]]},
        ],
        lambda f: dets[f],
    )
    assert out[0]["box"] == [1, 1, 5, 5] and out[1]["box"] == [7, 7, 9, 9]
    assert out[2]["box"] == [0, 0, 10, 20] and out[2]["points"] == [[1, 2, 1]]
    with pytest.raises(ValueError, match="detection 4 does not exist"):
        tw.resolve_prompts([{"frame": 5, "obj": 1, "detection": 4}], lambda f: dets[f])
    # review item 6: a click alone is a fresh conditioning frame with a partial mask -> rejected
    with pytest.raises(ValueError, match="needs a box or a detection"):
        tw.resolve_prompts([{"frame": 6, "obj": 3, "points": [[1, 2, 1]]}], lambda f: dets[5])


def test_keep_components_falls_back_to_largest(tw):
    """Review item 2: a mask fragmented into specks (each < min_part) used to crash the run."""
    specks = np.full(60, 9)  # 60 specks of 9 px: each 1.7% < 2%
    specks[17] = 10
    assert tw.keep_components(specks, 0.02).tolist() == [17]
    assert tw.keep_components(np.array([1000, 5, 400]), 0.02).tolist() == [0, 2]
    assert tw.keep_components(np.zeros(0), 0.02).tolist() == []


def rgb_to_cv_hsv(rgb: np.ndarray) -> np.ndarray:
    """OpenCV's uint8 HSV convention (H 0..179, S/V 0..255) in numpy."""
    x = rgb.astype(np.float64) / 255
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-12), 0)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(mx == r, (g - b) / d, np.where(mx == g, 2 + (b - r) / d, 4 + (r - g) / d))
    h = np.where(d > 0, (h * 60) % 360, 0)
    return np.stack([np.round(h / 2) % 180, np.round(s * 255), np.round(mx * 255)], -1).astype(np.uint8)


def test_hist_separates_black_and_white_trunks(tw):
    """Review item 3: hue x saturation alone could not tell black from white clothing."""
    from afv.models.track import HIST_DIM, bhattacharyya

    assert tw.HIST_DIM == HIST_DIM
    rng = np.random.default_rng(0)

    def person(trunks_rgb):
        img = np.full((400, 200, 3), (200.0, 120.0, 90.0))  # skin
        img[200:300] = trunks_rgb
        hsv = rgb_to_cv_hsv(np.clip(img + rng.integers(-6, 7, img.shape), 0, 255))
        return np.concatenate([tw.hsv_hist(hsv[:200].reshape(-1, 3)), tw.hsv_hist(hsv[200:].reshape(-1, 3))])

    black, white, red = person((15, 15, 15)), person((235, 235, 235)), person((200, 30, 30))
    assert bhattacharyya(black, black) < 0.05
    assert bhattacharyya(black, white) > 0.4
    assert bhattacharyya(black, red) > 0.4
    assert tw.hsv_hist(np.zeros((0, 3), np.uint8)).sum() == 0


def test_pairwise_mask_iou(tw):
    a = np.zeros((4, 8), bool)
    a[:, :4] = True
    b = np.zeros((4, 8), bool)
    b[:, 2:6] = True
    small = {1: {0: np.packbits(a), 1: np.packbits(b)}, 2: {0: np.packbits(a)}}
    iou = tw.pairwise_mask_iou(small, 3, 2)
    assert iou.shape == (3, 2, 2) and iou[1, 0, 1] == iou[1, 1, 0] == pytest.approx(8 / 24)
    assert iou[0].sum() == 0 and iou[2].sum() == 0
