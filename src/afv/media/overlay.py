"""Static-overlay mask: burned-in graphics (scoreboards, logos, tickers) that stay pixel-constant
while the scene moves.

Method, on evenly sampled frames of one shot (downscaled for speed):

1. per pixel, the fraction of frames within ``tau`` grey levels (any channel) of the temporal
   median -> ``static`` where that fraction >= ``static_fraction``;
2. strong edges of the median image that are static -> graphic seeds (text, borders); flat static
   areas alone (a dark backdrop that looks constant during a pan) are not seeds;
3. close the seeds to join glyphs, fill holes, keep only static pixels, grow a bounded distance
   into connected static pixels (flat scoreboard panels), drop small components and
   components with few edges, fill holes again (a ticking clock digit inside a scoreboard), then
   dilate by a small margin (compression halo);
4. bars from letterbox detection are excluded (reported separately).

If the scene hardly moves (static camera, little action) every background pixel is static and
the mask would cover the set, so ``reliable`` is False and only small components touching the
frame border band are kept. An overlay touching a static textured background merges with it and
is then lost; separating those needs frames from other shots (overlays stay, backgrounds change).

Not detected: **semi-transparent** graphics (e.g. a 35% white watermark over a moving scene).
Their pixels change with the scene behind them, so they are never "static"; that would need a
mean-offset / edge-persistence test. Opaque parts (solid text, borders) of a partly transparent
scoreboard are found.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from scipy import ndimage

from afv.media.letterbox import Letterbox, luma


class OverlayParams(BaseModel):
    samples: int = 48  # frames sampled across the shot
    max_width: int = 480  # analysis width (mask is upsampled to the full size)
    tau: float = 12.0  # grey levels a static pixel may deviate (x264 noise)
    static_fraction: float = 0.9  # fraction of samples a pixel must stay within tau
    edge: float = 40.0  # Sobel magnitude (on 0..255 luma) of a graphic edge
    close_px: float = 0.02  # closing size as a fraction of analysis width
    grow_px: float = 0.03  # growth into connected static pixels, fraction of analysis width
    min_area: float = 0.0008  # smallest component, fraction of frame area
    min_edge_density: float = 0.04  # fraction of a component's pixels that are static edges
    dilate_px: int = 2  # margin at analysis resolution
    min_scene_motion: float = 0.25  # moving-pixel fraction below which the mask is unreliable
    border_band: float = 0.2  # unreliable scenes keep components touching this outer band ...
    max_component: float = 0.05  # ... and no larger than this fraction of the frame


@dataclass
class OverlayMask:
    mask: np.ndarray  # [H,W] bool at the requested output size
    scene_motion: float  # fraction of (non-bar) pixels that move
    coverage: float  # fraction of the frame masked
    reliable: bool
    boxes: list[tuple[int, int, int, int]]  # (x0, y0, x1, y1) per component, output pixels


def static_overlay_mask(
    frames: np.ndarray,
    *,
    out_size: tuple[int, int] | None = None,
    letterbox: Letterbox | None = None,
    params: OverlayParams | None = None,
) -> OverlayMask:
    """Mask static overlays in sampled RGB frames ``[N,h,w,3]``; ``out_size=(W, H)`` upsamples."""
    p = params or OverlayParams()
    if frames.ndim != 4 or frames.shape[0] < 3:
        raise ValueError("need at least 3 RGB frames [N,H,W,3]")
    _, h, w, _ = frames.shape
    lb = letterbox.scaled(w, h).mask() if letterbox is not None and letterbox.any else np.zeros((h, w), bool)

    f = frames.astype(np.int16)
    med = np.median(f, axis=0)  # [h,w,3]
    dev = np.abs(f - med[None]).max(axis=3)  # [n,h,w]
    const = (dev <= p.tau).mean(axis=0)  # [h,w]
    static = const >= p.static_fraction
    inside = ~lb
    scene_motion = float((const[inside] < 0.5).mean()) if inside.any() else 0.0

    g = luma(med.astype(np.float32))
    mag = np.hypot(ndimage.sobel(g, axis=0), ndimage.sobel(g, axis=1)) / 4.0
    seeds = static & (mag >= p.edge) & inside

    k = max(3, round(p.close_px * w) | 1)
    grown = ndimage.binary_closing(seeds, structure=np.ones((k, k), bool), border_value=0)
    grown = ndimage.binary_fill_holes(grown | seeds)
    region = grown & static & inside
    # flat panels around the text (scoreboard background) are static but edge-free: grow into
    # connected static pixels, a bounded distance so a static backdrop is not swallowed
    grow = round(p.grow_px * w)
    if grow > 0 and region.any():
        region = ndimage.binary_dilation(region, iterations=grow, mask=static & inside)
    region = ndimage.binary_opening(region, structure=np.ones((2, 2), bool))

    labels, num = ndimage.label(region)
    keep = np.zeros(num + 1, bool)
    min_area = p.min_area * h * w
    reliable = scene_motion >= p.min_scene_motion
    band_y, band_x = int(p.border_band * h), int(p.border_band * w)
    if num:
        areas = ndimage.sum(np.ones_like(region), labels, index=np.arange(1, num + 1))
        edge_px = ndimage.sum(seeds, labels, index=np.arange(1, num + 1))
        slices = ndimage.find_objects(labels)
        for i in range(num):
            if areas[i] < min_area or edge_px[i] < p.min_edge_density * areas[i]:
                continue
            if not reliable:
                if areas[i] > p.max_component * h * w:
                    continue  # the static background itself
                sy, sx = slices[i]
                touches = (sy.start < band_y or sy.stop > h - band_y or sx.start < band_x
                           or sx.stop > w - band_x)
                if not touches:
                    continue
            keep[i + 1] = True
    # holes enclosed by a kept graphic (a ticking clock digit) belong to the overlay too
    mask = ndimage.binary_fill_holes(keep[labels])
    if p.dilate_px > 0 and mask.any():
        mask = ndimage.binary_dilation(mask, iterations=p.dilate_px) & inside

    if out_size is not None and (out_size[0], out_size[1]) != (w, h):
        ow, oh = out_size
        yi = np.minimum((np.arange(oh) * h / oh).astype(int), h - 1)
        xi = np.minimum((np.arange(ow) * w / ow).astype(int), w - 1)
        mask = mask[yi][:, xi]
    lab2, num2 = ndimage.label(mask)
    objs = ndimage.find_objects(lab2) if num2 else []
    boxes = [(s[1].start, s[0].start, s[1].stop, s[0].stop) for s in objs]
    return OverlayMask(mask, scene_motion, float(mask.mean()), reliable, boxes)
