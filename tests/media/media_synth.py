"""Synthetic test videos for afv.media tests (never real footage)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

W, H = 192, 108

GEMX_DIR = Path.home() / "afv" / "gem-x"
GEMX_PY = GEMX_DIR / ".venv" / "bin" / "python"
GEMX_VIDEO = GEMX_DIR / "gem" / "inference" / "video.py"


def write_video(
    path: Path,
    frames: Iterable[np.ndarray],
    *,
    fps: float = 30.0,
    times: Sequence[float] | None = None,
    crf: int = 8,
    audio: bool = False,
) -> Path:
    """Encode RGB uint8 frames to H.264 mp4. ``times`` (seconds) gives a variable frame rate;
    ``audio`` adds a silent AAC track (like real clips: two streams, so ffmpeg muxes with cfr)."""
    frames = list(frames)
    if not frames:
        raise ValueError("no frames")
    with av.open(str(path), "w") as c:
        rate = Fraction(fps).limit_denominator(1001)
        s = c.add_stream("libx264", rate=rate, options={"crf": str(crf), "preset": "ultrafast"})
        s.pix_fmt = "yuv420p"
        s.height, s.width = frames[0].shape[:2]
        if times is not None:
            s.codec_context.time_base = Fraction(1, 1000)
            s.time_base = Fraction(1, 1000)
        a = None
        if audio:
            a = c.add_stream("aac", rate=48000)
            a.layout = "mono"
        for i, img in enumerate(frames):
            f = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
            if times is not None:
                f.pts = round(times[i] * 1000)
                f.time_base = Fraction(1, 1000)
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
        if a is not None:
            span = (times[-1] if times is not None else len(frames) / fps) + 0.1
            n = 0
            while n < span * 48000:
                af = av.AudioFrame.from_ndarray(np.zeros((1, 1024), np.float32), format="fltp",
                                                layout="mono")
                af.sample_rate, af.pts = 48000, n
                for p in a.encode(af):
                    c.mux(p)
                n += 1024
            for p in a.encode():
                c.mux(p)
    return path


def textured(rng: np.random.Generator, colour: Sequence[int], w: int = W, h: int = H,
             amp: int = 25) -> np.ndarray:
    base = np.array(colour, np.int16)[None, None, :]
    noise = rng.integers(-amp, amp + 1, size=(h, w, 1), dtype=np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def moving_blob(img: np.ndarray, i: int, colour: Sequence[int] = (240, 240, 240)) -> np.ndarray:
    """Draw a small block moving across the frame (keeps shots from being static)."""
    out = img.copy()
    h, w = out.shape[:2]
    x = (i * 3) % (w - 16)
    y = h // 3 + int(10 * np.sin(i / 5))
    out[y:y + 14, x:x + 14] = colour
    return out


COLOURS = [(200, 40, 40), (40, 160, 60), (40, 60, 200), (190, 170, 40)]


def colour_cuts_frames(seg_lengths: Sequence[int], *, seed: int = 0, w: int = W, h: int = H,
                       flash_at: Sequence[int] = ()) -> list[np.ndarray]:
    """Shots of distinct colours; ``flash_at`` frames are blown out to near-white (photo flash)."""
    rng = np.random.default_rng(seed)
    frames: list[np.ndarray] = []
    for k, n in enumerate(seg_lengths):
        for _ in range(n):
            img = moving_blob(textured(rng, COLOURS[k % len(COLOURS)], w, h), len(frames))
            frames.append(img)
    for i in flash_at:
        frames[i] = np.clip(frames[i].astype(np.int16) + 170, 0, 255).astype(np.uint8)
    return frames


def pan_frames(n: int, *, speed: int = 40, seed: int = 1, w: int = W, h: int = H) -> list[np.ndarray]:
    """A fast horizontal pan over a wide, varied-colour panorama (no cut)."""
    rng = np.random.default_rng(seed)
    pano_w = w + speed * n + 8
    cols = np.linspace(0, 1, pano_w)[None, :, None]
    base = np.concatenate([
        120 + 80 * np.sin(cols * 9.0), 110 + 70 * np.sin(cols * 5.0 + 1.0), 100 + 60 * np.cos(cols * 7.0),
    ], axis=2)
    pano = np.clip(base + rng.integers(-30, 31, size=(h, pano_w, 1)), 0, 255).astype(np.uint8)
    pano = np.repeat(pano, 1, axis=0)
    return [pano[:, i * speed:i * speed + w] for i in range(n)]


def logo_frames(n: int, *, seed: int = 2, w: int = 320, h: int = 180,
                box: tuple[int, int, int, int] = (16, 12, 96, 40)) -> tuple[list[np.ndarray], np.ndarray]:
    """Panning noisy scene with a static 'scoreboard' (x, y, w, h) on top. Returns frames + truth mask."""
    rng = np.random.default_rng(seed)
    pano = rng.integers(0, 256, size=(h, w + 6 * n, 3), dtype=np.uint8)
    # smooth a little so x264 keeps it as texture, not mush
    pano = ((pano.astype(np.uint16) + np.roll(pano, 1, 1) + np.roll(pano, 1, 0)) // 3).astype(np.uint8)
    x, y, bw, bh = box
    logo = np.full((bh, bw, 3), 30, np.uint8)
    logo[4:-4, 4:-4] = (20, 30, 120)
    for k in range(6):  # "text": bright bars
        logo[10:bh - 10, 10 + k * 13:16 + k * 13] = 235
    frames = []
    for i in range(n):
        img = pano[:, i * 6:i * 6 + w].copy()
        img[y:y + bh, x:x + bw] = logo
        frames.append(img)
    truth = np.zeros((h, w), bool)
    truth[y:y + bh, x:x + bw] = True
    return frames, truth


def letterbox_frames(n: int, *, bar: int = 24, side: int = 0, w: int = 320, h: int = 180,
                     seed: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        content = moving_blob(textured(rng, (120, 100, 90), w - 2 * side, h - 2 * bar, amp=40), i)
        img[bar:h - bar, side:w - side] = content
        frames.append(img)
    return frames


def dissolve_frames(n_a: int, n_mix: int, n_b: int, *, to_black: bool = False, seed: int = 4,
                    w: int = W, h: int = H) -> list[np.ndarray]:
    """Shot A, a linear dissolve of ``n_mix`` frames (or a fade to black), then shot B."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_a + n_mix + n_b):
        a = moving_blob(textured(rng, COLOURS[0], w, h), i).astype(np.float32)
        b = (np.zeros_like(a) if to_black else
             moving_blob(textured(rng, COLOURS[2], w, h), i, (20, 20, 20)).astype(np.float32))
        alpha = float(np.clip((i - n_a + 1) / (n_mix + 1), 0.0, 1.0))
        frames.append(((1 - alpha) * a + alpha * b).round().astype(np.uint8))
    return frames


# ---------------------------------------------------------------------------
# Richer scenes: static multi-colour textures with camera drift, sensor noise and a moving object
# ---------------------------------------------------------------------------


def scene_texture(rng: np.random.Generator, tint: Sequence[int], w: int, h: int,
                  tint_weight: float = 0.55) -> np.ndarray:
    """A static scene: smooth colour blobs plus fine detail, pulled towards ``tint`` (float32)."""
    from scipy import ndimage

    coarse = rng.integers(0, 256, size=(h // 8 + 2, w // 8 + 2, 3)).astype(np.float32)
    big = np.kron(coarse, np.ones((8, 8, 1), np.float32))[:h, :w]
    big = ndimage.uniform_filter(big, size=(9, 9, 1))
    detail = rng.integers(-24, 25, size=(h, w, 1)).astype(np.float32)
    return (1 - tint_weight) * big + tint_weight * np.array(tint, np.float32) + detail


def scene_shot(n: int, k: int, *, seed: int = 0, drift: int = 1, noise: int = 5, w: int = W,
               h: int = H, start_index: int = 0) -> list[np.ndarray]:
    """``n`` frames of scene ``k`` (tint COLOURS[k % 4]): slow pan of ``drift`` px/frame, per-frame
    sensor noise and a moving block."""
    rng = np.random.default_rng(1000 * seed + k)
    tex = scene_texture(rng, COLOURS[k % len(COLOURS)], w + drift * n + 1, h)
    out = []
    for i in range(n):
        f = tex[:, i * drift:i * drift + w] + rng.integers(-noise, noise + 1, size=(h, w, 1))
        out.append(moving_blob(np.clip(f, 0, 255).astype(np.uint8), start_index + i))
    return out


def scene_cuts_frames(seg_lengths: Sequence[int], *, seed: int = 0, w: int = W, h: int = H
                      ) -> list[np.ndarray]:
    """Hard cuts between textured scenes (shot k is tinted COLOURS[k % 4])."""
    frames: list[np.ndarray] = []
    for k, n in enumerate(seg_lengths):
        frames += scene_shot(n, k, seed=seed, w=w, h=h, start_index=len(frames))
    return frames


def flash(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """A photo flash: brighter and washed out, same picture."""
    f = img.astype(np.float32)
    return np.clip(f * (1 + 0.6 * strength) + 90 * strength, 0, 255).astype(np.uint8)


def scene_dissolve_frames(n_a: int, n_mix: int, n_b: int, *, to_black: bool = False, seed: int = 4
                          ) -> list[np.ndarray]:
    """Scene A, a linear dissolve of ``n_mix`` frames (or a fade to black), then scene B."""
    n = n_a + n_mix + n_b
    a = scene_shot(n, 0, seed=seed)
    b = scene_shot(n, 2, seed=seed)
    frames = []
    for i in range(n):
        alpha = float(np.clip((i - n_a + 1) / (n_mix + 1), 0.0, 1.0))
        bb = np.zeros_like(a[i], np.float32) if to_black else b[i].astype(np.float32)
        frames.append(((1 - alpha) * a[i] + alpha * bb).round().astype(np.uint8))
    return frames


def scene_wipe_frames(n_a: int, n_wipe: int, n_b: int, *, seed: int = 5) -> list[np.ndarray]:
    """Scene A, a left-to-right wipe revealing scene B over ``n_wipe`` frames, then scene B."""
    n = n_a + n_wipe + n_b
    a = scene_shot(n, 0, seed=seed)
    b = scene_shot(n, 2, seed=seed)
    w = a[0].shape[1]
    frames = a[:n_a]
    for i in range(n_wipe):
        x = int(w * (i + 1) / (n_wipe + 1))
        f = a[n_a + i].copy()
        f[:, :x] = b[n_a + i][:, :x]
        frames.append(f)
    return frames + b[n_a + n_wipe:]


def arena_frames(n_a: int, n_b: int, *, seed: int = 3) -> list[np.ndarray]:
    """Same arena from two cameras: a structured ring scene; shot B is a mirrored, zoomed crop.
    Colour histograms of the two shots are very similar (a 'similar-shot' cut at n_a)."""
    r = np.random.default_rng(seed)
    sw, sh = 600, 260
    sc = np.zeros((sh, sw, 3), np.int16)
    sc[: sh // 3] = (25, 22, 30)
    sc[: sh // 3] += r.integers(0, 40, size=(sh // 3, sw, 1)).astype(np.int16)  # crowd
    sc[sh // 3:] = (170, 170, 175)  # canvas
    sc[sh // 3: sh // 3 + 6] = (30, 40, 200)  # ropes
    sc[sh // 2: sh // 2 + 4] = (200, 30, 30)
    sc[sh // 3:] += r.integers(-10, 11, size=(sh - sh // 3, sw, 1)).astype(np.int16)
    scene = np.clip(sc, 0, 255).astype(np.uint8)

    def cam(x: int, i: int, flip: bool = False, zoom: float = 1.0) -> np.ndarray:
        cw, ch = int(W * 1.6 / zoom), int(H * 1.6 / zoom)
        crop = scene[40:40 + ch, x:x + cw]
        if flip:
            crop = crop[:, ::-1]
        yi = (np.arange(H) * ch / H).astype(int)
        xi = (np.arange(W) * cw / W).astype(int)
        f = crop[yi][:, xi].copy()
        f[50:95, 60 + i % 20:75 + i % 20] = (40, 30, 30)  # two fighters
        f[50:95, 110 - i % 15:125 - i % 15] = (200, 150, 120)
        return f

    return [cam(50 + i, i) for i in range(n_a)] + [cam(250 - i, i, True, 1.2) for i in range(n_b)]


REMUX_ARGS = {
    "mkv_aac": ["-c:v", "copy", "-c:a", "aac"],
    "mkv_opus": ["-c:v", "copy", "-c:a", "libopus"],
    "webm_opus": ["-c:v", "libvpx-vp9", "-crf", "20", "-b:v", "0", "-deadline", "realtime",
                  "-c:a", "libopus"],
}


def remux_with_audio(src: Path, out: Path, kind: str, ffmpeg: str) -> Path:
    """Copy (or for WebM re-encode) ``src`` into another container with a sine audio track, using
    the pinned ffmpeg; AAC/Opus priming makes the container start time negative."""
    import subprocess

    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(src), "-f", "lavfi", "-i", "sine=duration=30",
                    "-map", "0:v", "-map", "1:a", "-shortest", *REMUX_ARGS[kind], str(out)], check=True)
    return out
