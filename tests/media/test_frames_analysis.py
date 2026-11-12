"""Letterbox, static-overlay mask, PNG and proxy tests (synthetic video only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from media_synth import letterbox_frames, logo_frames, textured, write_video

from afv.media.frames import sample_frames, sample_multi
from afv.media.letterbox import detect_letterbox
from afv.media.overlay import static_overlay_mask
from afv.media.png import read_png_gray, write_png_gray
from afv.media.probe import probe
from afv.media.proxy import make_proxy


@pytest.mark.parametrize(("bar", "side"), [(24, 0), (0, 40), (20, 32)])
def test_letterbox(tmp_path: Path, bar: int, side: int) -> None:
    path = write_video(tmp_path / "lb.mp4", letterbox_frames(40, bar=bar, side=side))
    _, frames = sample_frames(path, 12)
    lb = detect_letterbox(frames)
    assert abs(lb.top - bar) <= 1 and abs(lb.bottom - bar) <= 1
    assert abs(lb.left - side) <= 1 and abs(lb.right - side) <= 1
    # same answer from full-resolution gray samples (the stage's path)
    gray = sample_multi(path, 40, {"g": (12, None, True)})["g"]
    assert detect_letterbox(gray) == lb


def test_dark_scene_is_not_letterbox(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(20):
        img = textured(rng, (120, 110, 100), 320, 180, amp=40)
        img[:40] = textured(rng, (20, 20, 22), 320, 40, amp=4)  # dark arena roof, not a bar
        img[:40, rng.integers(0, 300):][:3, :3] = 250  # a light in the roof
        frames.append(img)
    _, sampled = sample_frames(write_video(tmp_path / "dark.mp4", frames), 10)
    assert not detect_letterbox(sampled).any


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def test_static_logo_mask(tmp_path: Path) -> None:
    frames, truth = logo_frames(60)
    path = write_video(tmp_path / "logo.mp4", frames, crf=18)
    _, sampled = sample_frames(path, 40, max_width=320)
    ov = static_overlay_mask(sampled, out_size=(320, 180))
    assert ov.reliable and ov.scene_motion > 0.5
    assert _iou(ov.mask, truth) > 0.75
    assert len(ov.boxes) == 1
    x0, y0, x1, y1 = ov.boxes[0]
    assert abs(x0 - 16) <= 4 and abs(y0 - 12) <= 4 and abs(x1 - 112) <= 4 and abs(y1 - 52) <= 4


def test_no_overlay_in_moving_scene(tmp_path: Path) -> None:
    clean, truth = logo_frames(60, box=(0, 0, 0, 0))
    assert not truth.any()
    path = write_video(tmp_path / "clean.mp4", clean, crf=18)
    _, sampled = sample_frames(path, 40)
    ov = static_overlay_mask(sampled)
    assert ov.coverage < 0.002 and ov.boxes == []


def test_static_scene_is_flagged_unreliable(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    still = textured(rng, (100, 120, 140), 320, 180, amp=60)
    still[60:120, 120:200] = 230  # a bright static object in the middle of the set
    static = [still.copy() for _ in range(30)]
    for i, img in enumerate(static):  # a small moving thing in the middle
        img[80 + i % 10:90 + i % 10, 150:160] = 0
    _, sampled = sample_frames(write_video(tmp_path / "still.mp4", static, crf=18), 30)
    ov = static_overlay_mask(sampled)
    assert not ov.reliable and ov.scene_motion < 0.25
    # the static background (one big component) is not reported as an overlay
    assert not ov.mask[60:120, 140:180].any() and ov.coverage < 0.05


def test_png_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    m = rng.random((37, 53)) > 0.7
    write_png_gray(tmp_path / "m.png", m)
    back = read_png_gray(tmp_path / "m.png")
    assert back.dtype == np.uint8 and back.shape == (37, 53)
    np.testing.assert_array_equal(back, m.astype(np.uint8) * 255)


def test_proxy_downscales_and_keeps_frames(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    frames = [textured(rng, (90, 90, 90), 1280, 960, amp=20) for _ in range(12)]
    src = write_video(tmp_path / "big.mp4", frames, crf=30)
    out = make_proxy(src, tmp_path / "proxy.mp4")
    info = probe(out, hash=False)
    assert (info.width, info.height) == (960, 720) and info.num_frames == 12
    small = make_proxy(src, tmp_path / "p2.mp4", height=2000)  # never upscales
    assert probe(small, hash=False).height == 960
    shot = make_proxy(src, tmp_path / "p3.mp4", start=0.1, duration=0.2, shot_timeline=True)
    assert probe(shot, hash=False).num_frames == 6


@pytest.mark.parametrize("level", [6, 10, 14])
def test_one_sided_dark_crowd_is_not_letterbox(tmp_path: Path, level: int) -> None:
    """A dark crowd across the top (luma ~6-14, sparse dim phone screens) above a lit ring."""
    rng = np.random.default_rng(level)
    frames = []
    for _ in range(30):
        img = textured(rng, (150, 140, 130), 320, 180, amp=40)
        img[:54] = textured(rng, (level, level, level + 2), 320, 54, amp=4)
        img[rng.integers(0, 54, 30), rng.integers(0, 320, 30)] = 60
        frames.append(img)
    path = write_video(tmp_path / "crowd.mp4", frames, crf=18)
    gray = sample_multi(path, 30, {"g": (16, None, True)})["g"]
    assert not detect_letterbox(gray).any


def test_semi_transparent_watermark_is_a_known_gap(tmp_path: Path) -> None:
    """Documented limitation (overlay.py): a 35% white watermark is not masked; opaque logos are."""
    frames, truth = logo_frames(60)
    wm = []
    for f in frames:
        g = f.astype(np.float32)
        g[130:160, 200:300] = 0.65 * g[130:160, 200:300] + 0.35 * 255
        wm.append(g.astype(np.uint8))
    _, sampled = sample_frames(write_video(tmp_path / "wm.mp4", wm, crf=18), 40)
    ov = static_overlay_mask(sampled, out_size=(320, 180))
    assert ov.mask[truth].mean() > 0.9 and ov.mask[130:160, 200:300].mean() < 0.1
