"""Letterbox / pillarbox detection: black bars that stay black for the whole sample."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Letterbox:
    """Bar sizes in pixels of the analysed frames; ``content`` is (x0, y0, x1, y1), exclusive."""

    top: int
    bottom: int
    left: int
    right: int
    width: int
    height: int

    @property
    def content(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.width - self.right, self.height - self.bottom

    @property
    def any(self) -> bool:
        return bool(self.top or self.bottom or self.left or self.right)

    def mask(self) -> np.ndarray:
        """[H,W] bool, True on bar pixels."""
        m = np.ones((self.height, self.width), bool)
        x0, y0, x1, y1 = self.content
        m[y0:y1, x0:x1] = False
        return m

    def scaled(self, width: int, height: int) -> Letterbox:
        sx, sy = width / self.width, height / self.height
        return Letterbox(int(np.floor(self.top * sy)), int(np.floor(self.bottom * sy)),
                         int(np.floor(self.left * sx)), int(np.floor(self.right * sx)), width, height)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def luma(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma of uint8 RGB ``[..., 3]`` as float32."""
    return rgb[..., 0] * np.float32(0.299) + rgb[..., 1] * np.float32(0.587) + rgb[..., 2] * np.float32(0.114)


def detect_letterbox(
    frames: np.ndarray,
    *,
    black: float = 24.0,
    dark_fraction: float = 0.97,
    bar_median: float = 12.0,
    min_bar: float = 0.01,
    max_bar: float = 0.4,
    symmetry: float = 0.02,
) -> Letterbox:
    """Find bars in sampled RGB frames ``[N,H,W,3]`` (or luma ``[N,H,W]``).

    A row (column) belongs to a bar when, over all frames, at least ``dark_fraction`` of its
    pixels are darker than ``black`` (so a small channel logo inside a bar is tolerated) and its
    median is at most ``bar_median`` (bars are encoded black, ~0; a dark arena ceiling is ~20). Bars
    grow inward from each edge; a bar thinner than ``min_bar`` of the frame is ignored, and
    detection stops at ``max_bar`` so a black scene is not taken for a bar.

    Bars come in pairs: top/bottom (and left/right) must both be present and within
    ``symmetry`` of the frame size of each other, and the content must start with a step (the row
    or column just inside a bar is not itself mostly dark). A dark crowd at the top of a frame is
    one-sided and is rejected; a one-sided bar (rare) is missed.
    """
    y = luma(frames) if frames.ndim == 4 else frames  # uint8 luma is used as is (memory)
    _, h, w = y.shape
    dark = y <= black
    rows = y.transpose(1, 0, 2).reshape(h, -1)
    cols = y.transpose(2, 0, 1).reshape(w, -1)
    row_frac = dark.mean(axis=(0, 2))
    col_frac = dark.mean(axis=(0, 1))
    row_dark = (row_frac >= dark_fraction) & (np.median(rows, axis=1) <= bar_median)
    col_dark = (col_frac >= dark_fraction) & (np.median(cols, axis=1) <= bar_median)

    def run(flags: np.ndarray, size: int) -> int:
        limit = int(size * max_bar)
        k = 0
        while k < limit and flags[k]:
            k += 1
        if k >= limit:  # the whole frame (or most of it) is dark: not a letterbox
            return 0
        return k if k >= max(1, round(size * min_bar)) else 0

    def pair(flags: np.ndarray, frac: np.ndarray, size: int) -> tuple[int, int]:
        a, b = run(flags, size), run(flags[::-1], size)
        if not (a and b) or abs(a - b) > max(2, symmetry * size):
            return 0, 0
        # a step into content: the line just inside each bar is not mostly dark
        if frac[min(a + 1, size - 1)] >= 0.9 or frac[::-1][min(b + 1, size - 1)] >= 0.9:
            return 0, 0
        return a, b

    top, bottom = pair(row_dark, row_frac, h)
    left, right = pair(col_dark, col_frac, w)
    return Letterbox(top, bottom, left, right, w, h)
