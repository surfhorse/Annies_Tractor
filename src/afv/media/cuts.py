"""Shot-cut detection on HSV histograms with an adaptive threshold.

Per source frame (downscaled to ``analysis_width``; letterbox bars are *not* cropped, they can
change at a cut) we build a soft-binned hue x saturation histogram of colourful pixels plus a value
histogram of grey pixels, globally and on a 3x3 grid; the distance between two frames is the mean
Hellinger distance of the global and grid histograms (0 = identical, 1 = disjoint). Soft bins keep
the distance continuous when colours drift across a bin edge (dissolves, lighting).

**Hard cuts.** The score of the boundary before frame i is the *minimum* distance over all pairs
(a, b) with a in [i-K, i-1] and b in [i, i+K-1], K = ``flash`` seconds of frames, so a 1-3 frame
flash or glitch is bridged by a pair of normal frames. A cut must

1. beat ``max(abs_threshold, median + k_sigma * 1.4826 * MAD)`` of the scores within ``window``
   seconds (fast pans and handheld shake raise median and MAD);
2. be the largest score within K frames (a flash right after a cut raises one neighbour);
3. be ``isolation`` times the second-highest score between K frames and ``isolation_window``
   seconds away (a cut is an isolated spike; a dissolve or whip pan raises a run of scores);
4. not *return*: if the frame before the boundary matches a frame up to ``flash_return`` seconds
   after it (or the frame after matches one up to that far before), it was a longer flash or a
   glitch, not a cut;
5. not be a luminance-only change (same structure and colour, brightness jumped: flash,
   auto-exposure);
6. lie outside every accepted gradual transition and keep ``min_shot`` seconds from every
   stronger cut, with confidence >= ``min_confidence``.

**Dips to black**: a short run of black frames (mean luma <= ``dip_black``) entered and left by
monotone luminance ramps, with different content before and after, is one gradual transition from
the first fading frame to the end of the fade-in.

**Gradual transitions** (dissolves, fades, wipes). Hard cuts are found first. The distance across
``gradual_span`` seconds each side, d(i-L, i+L), must peak above ``gradual_threshold`` and its own
adaptive threshold. Over the window, frames must move from shot A to shot B *progressively*:

- dissolve/fade: frame j ~ alpha_j A + (1 - alpha_j) B (small misfit), alpha falls monotonically,
  with at least ``gradual_min_frames`` frames strictly in between; or
- wipe: per block of a 6x4 grid, frame j's colour histogram is close to either A's or B's, and
  the fraction close to B rises monotonically through at least ``gradual_min_frames``
  intermediate frames;

and the window must not be camera motion: if phase correlation between consecutive frames finds a
shift that explains most of the change (``pan_ratio``), it is a pan, not a transition. A gradual
window that contains a hard cut outside its own measured extent is rejected; hard-cut spikes inside
an accepted transition's extent (the last dark frames of a fade) are dropped.

Confidence in [0, 1] grows with the margin over the threshold, ``1 - (thr - base) / (s - base)``;
candidates below ``min_confidence`` are never emitted.

Memory/speed: features are computed in batches and stored as float16 (~0.5 KB/frame) with a 16x9
grey thumbnail; full analysis frames are kept only in a ring buffer of a few dozen frames around
the gradual candidates, so a 2-hour source needs a few hundred MB at most.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Literal

import av
import numpy as np
from pydantic import BaseModel

from afv.media.ffmpeg import seek_origin_us
from afv.media.probe import PathLike


class CutParams(BaseModel):
    analysis_width: int = 128
    batch: int = 64  # frames per feature batch
    flash: float = 0.1  # seconds bridged on each side of a boundary (photo flashes, glitches)
    flash_return: float = 0.25  # seconds within which content returning to before = not a cut
    luma_corr: float = 0.93  # structure correlation above which a brightness jump is not a cut
    luma_jump: float = 12.0  # grey levels of mean brightness change for that test ...
    luma_chroma: float = 0.04  # ... with the mean chromaticity (r,g,b / sum) changing at most this
    window: float = 1.0  # seconds each side for the adaptive threshold
    k_sigma: float = 6.0
    abs_threshold: float = 0.08
    min_shot: float = 0.5  # seconds between cuts
    isolation: float = 2.0  # a cut's score / second-highest score nearby (see module docstring)
    isolation_window: float = 0.25  # seconds each side for the isolation test
    min_confidence: float = 0.05  # candidates below this are never emitted
    gradual: bool = True
    gradual_span: float = 0.33  # seconds each side of a gradual transition's centre
    gradual_threshold: float = 0.3
    gradual_max_residual: float = 0.35  # blend misfit relative to |A - B|
    gradual_min_frames: int = 3  # intermediate frames a gradual transition must show
    pan_ratio: float = 0.6  # best-shift residual / zero-shift residual below this = camera motion
    wipe_explained: float = 0.7  # fraction of grid blocks close to A or B during a wipe
    dip_black: float = 16.0  # mean luma of a "black" frame in a dip to black
    dip_fade: float = 1.0  # longest fade-out / fade-in, seconds
    dip_max_dark: float = 3.0  # longest black run of a dip, seconds


@dataclass(frozen=True)
class Cut:
    """A shot boundary. Times are seek times (seconds from the container start, as ``-ss``).

    ``frame`` is the first source frame of the new shot and ``time`` the midpoint between it and
    the previous frame. For a gradual transition ``start_frame``/``start_time`` mark its first
    frame (frames ``start_frame .. frame-1`` are the transition); for a hard cut they equal
    ``frame``/``time``.
    """

    frame: int
    time: float
    confidence: float
    score: float
    kind: Literal["hard", "gradual"] = "hard"
    start_frame: int = -1
    start_time: float = float("nan")

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            object.__setattr__(self, "start_frame", self.frame)
        if math.isnan(self.start_time):
            object.__setattr__(self, "start_time", self.time)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class CutAnalysis:
    cuts: list[Cut]
    times: np.ndarray  # [N] seek time of every source frame
    scores: np.ndarray  # [N] hard-cut score of the boundary before frame i (0 for i = 0)
    thresholds: np.ndarray  # [N] adaptive threshold (NaN where the score never got near it)
    gradual_scores: np.ndarray  # [N] d(i-L, i+L)
    fps: float
    params: CutParams = field(default_factory=CutParams)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

_GLOBAL_BINS = (16, 4, 8)  # hue, saturation (chromatic pixels), value (grey pixels)
_BLOCK_BINS = (8, 2, 4)
_GRID = 3
_NG = _GLOBAL_BINS[0] * _GLOBAL_BINS[1] + _GLOBAL_BINS[2]
_NB = _BLOCK_BINS[0] * _BLOCK_BINS[1] + _BLOCK_BINS[2]
FEATURE_DIM = _NG + _GRID * _GRID * _NB


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uint8 RGB [..., 3] -> h in [0, 1), s, v in [0, 1] (float32)."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    v = np.maximum(np.maximum(r, g), b)
    c = v - np.minimum(np.minimum(r, g), b)
    s = np.where(v > 0, c / np.maximum(v, 1e-6), 0.0)
    safe = np.maximum(c, 1e-6)
    h = np.where(v == r, ((g - b) / safe) % 6.0, np.where(v == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
    h = np.where(c > 0, h / 6.0, 0.0) % 1.0
    return h.astype(np.float32), s.astype(np.float32), v


def _soft(x: np.ndarray, n: int, circular: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear interpolation of positions x in [0, 1] onto n bin centres: (bin0, bin1, weight1)."""
    pos = x * n - 0.5
    if not circular:
        pos = np.clip(pos, 0.0, n - 1.0)
    b0 = np.floor(pos)
    w1 = (pos - b0).astype(np.float32)
    b0 = b0.astype(np.int64)
    if circular:
        return b0 % n, (b0 + 1) % n, w1
    return b0, np.minimum(b0 + 1, n - 1), w1


def _hist_rows(h: np.ndarray, s: np.ndarray, v: np.ndarray, group: np.ndarray, ngroups: int,
               bins: tuple[int, int, int], normalise: bool = True) -> np.ndarray:
    """Soft-binned histograms of pixels grouped by ``group`` (int ids) -> [ngroups, nbins]."""
    nh, ns, nv = bins
    nb = nh * ns + nv
    chroma = (np.clip((s - 0.1) / 0.1, 0, 1) * np.clip((v - 0.08) / 0.08, 0, 1)).astype(np.float32)
    h0, h1, wh = _soft(h, nh, True)
    s0, s1, ws = _soft(np.clip((s - 0.15) / 0.85, 0, 1), ns, False)
    v0, v1, wv = _soft(v, nv, False)
    off = group * nb
    idx = np.concatenate([off + h0 * ns + s0, off + h0 * ns + s1, off + h1 * ns + s0, off + h1 * ns + s1,
                          off + nh * ns + v0, off + nh * ns + v1])
    wts = np.concatenate([chroma * (1 - wh) * (1 - ws), chroma * (1 - wh) * ws, chroma * wh * (1 - ws),
                          chroma * wh * ws, (1 - chroma) * (1 - wv), (1 - chroma) * wv])
    hist = np.bincount(idx, weights=wts, minlength=ngroups * nb).reshape(ngroups, nb)
    if normalise:
        hist = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1e-6)
    return hist.astype(np.float32)


def features_batch(rgb: np.ndarray) -> np.ndarray:
    """Square-rooted histograms [B, FEATURE_DIM] (global + 3x3 blocks) of small RGB frames [B,h,w,3].

    One soft-binned pass per block at the global (fine) bins; the global histogram is the sum over
    blocks and the coarse block histograms sum pairs of fine bins (hue 16->8, sat 4->2, value 8->4).
    """
    b, hh, ww, _ = rgb.shape
    h, s, v = rgb_to_hsv(rgb)
    ys = np.minimum(np.arange(hh) * _GRID // hh, _GRID - 1)
    xs = np.minimum(np.arange(ww) * _GRID // ww, _GRID - 1)
    block = (ys[:, None] * _GRID + xs[None, :])[None]
    gid = (np.arange(b)[:, None, None] * _GRID * _GRID + block).ravel()
    counts = _hist_rows(h.ravel(), s.ravel(), v.ravel(), gid, b * _GRID * _GRID, _GLOBAL_BINS,
                        normalise=False)
    counts = counts.reshape(b, _GRID * _GRID, _NG)
    nh, ns, nv = _GLOBAL_BINS
    g = counts.sum(1)
    hs = counts[..., : nh * ns].reshape(b, _GRID * _GRID, nh // 2, 2, ns // 2, 2).sum((3, 5))
    vv = counts[..., nh * ns:].reshape(b, _GRID * _GRID, nv // 2, 2).sum(3)
    bl = np.concatenate([hs.reshape(b, _GRID * _GRID, -1), vv], axis=2)
    g = g / np.maximum(g.sum(1, keepdims=True), 1e-6)
    bl = bl / np.maximum(bl.sum(2, keepdims=True), 1e-6)
    return np.sqrt(np.concatenate([g, bl.reshape(b, -1)], axis=1)).astype(np.float32)


def frame_features(rgb: np.ndarray) -> np.ndarray:
    """Features of one small RGB frame [h,w,3]."""
    return features_batch(rgb[None])[0]


def feature_distance(fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """Hellinger distance between feature rows (broadcasting over leading axes)."""
    prod = fa.astype(np.float32) * fb.astype(np.float32)
    g = np.sqrt(np.clip(1.0 - prod[..., :_NG].sum(-1), 0.0, 1.0))
    blocks = prod[..., _NG:].reshape(*prod.shape[:-1], _GRID * _GRID, _NB).sum(-1)
    bd = np.sqrt(np.clip(1.0 - blocks, 0.0, 1.0)).mean(-1)
    return 0.5 * g + 0.5 * bd


def _lag_distance(feat: np.ndarray, lag: int, chunk: int = 8192) -> np.ndarray:
    """d(i, i+lag) for i in 0..N-1 (inf past the end), computed in chunks."""
    n = feat.shape[0]
    out = np.full(n, np.inf, np.float32)
    for a in range(0, max(0, n - lag), chunk):
        b = min(n - lag, a + chunk)
        out[a:b] = feature_distance(feat[a:b], feat[a + lag:b + lag])
    return out


def _block_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Block-mean resize of one image, [H,W] or [H,W,C], to (h, w), float32."""
    hh, ww = img.shape[:2]
    ys = np.linspace(0, hh, h + 1).astype(int)
    xs = np.linspace(0, ww, w + 1).astype(int)
    out = np.add.reduceat(np.add.reduceat(img.astype(np.float32), ys[:-1], axis=0), xs[:-1], axis=1)
    cnt = np.outer(np.diff(ys), np.diff(xs)).astype(np.float32)
    return out / (cnt[..., None] if img.ndim == 3 else cnt)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _gray_thumbs(rgb: np.ndarray, w: int = 16, h: int = 9) -> np.ndarray:
    """[B,H,W,3] -> [B, h*w] uint8 block-mean grey thumbnails."""
    b, hh, ww, _ = rgb.shape
    ys = np.linspace(0, hh, h + 1).astype(int)
    xs = np.linspace(0, ww, w + 1).astype(int)
    g = _gray(rgb.astype(np.float32))
    out = np.add.reduceat(np.add.reduceat(g, ys[:-1], axis=1), xs[:-1], axis=2)
    out /= np.outer(np.diff(ys), np.diff(xs))[None]
    return np.round(out).astype(np.uint8).reshape(b, -1)


# ---------------------------------------------------------------------------
# Gradual-transition checks (on analysis-resolution frames from the ring buffer)
# ---------------------------------------------------------------------------


def _blend(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> tuple[float, float]:
    """Best alpha with m ~ alpha*a + (1-alpha)*b and the misfit relative to |a - b|."""
    ab = a - b
    den = float((ab * ab).sum())
    if den < 1e-6:
        return 0.5, 1.0
    alpha = float(((m - b) * ab).sum()) / den
    resid = float(np.linalg.norm(m - (alpha * a + (1 - alpha) * b))) / math.sqrt(den)
    return alpha, resid


def _shift_ratio(g1: np.ndarray, g2: np.ndarray) -> float:
    """Residual after the best (phase-correlation) shift / residual without shift; 1 = no motion."""
    h, w = g1.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    f1 = np.fft.rfft2((g1 - g1.mean()) * win)
    f2 = np.fft.rfft2((g2 - g2.mean()) * win)
    cross = f1 * np.conj(f2)
    r = np.fft.irfft2(cross / np.maximum(np.abs(cross), 1e-6), s=(h, w))
    dy, dx = np.unravel_index(int(np.argmax(r)), r.shape)
    dy = int(dy - h if dy > h // 2 else dy)
    dx = int(dx - w if dx > w // 2 else dx)
    if dx == 0 and dy == 0:
        return 1.0
    # g1[y, x] ~ g2[y - dy, x - dx] on the overlap
    a = g1[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)]
    b = g2[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    if a.size < 0.25 * g1.size:
        return 1.0
    zero = float(np.mean((g1 - g2) ** 2)) + 1e-6
    return float(np.mean((a - b) ** 2)) / zero


def _grid_hists(rgb: np.ndarray, gx: int = 6, gy: int = 4) -> np.ndarray:
    """Square-rooted block colour histograms [n, gx*gy, bins] of frames [n,h,w,3]."""
    n, hh, ww, _ = rgb.shape
    h, s, v = rgb_to_hsv(rgb)
    ys = np.minimum(np.arange(hh) * gy // hh, gy - 1)
    xs = np.minimum(np.arange(ww) * gx // ww, gx - 1)
    gid = (np.arange(n)[:, None, None] * gx * gy + (ys[:, None] * gx + xs[None, :])[None]).ravel()
    hist = _hist_rows(h.ravel(), s.ravel(), v.ravel(), gid, n * gx * gy, _BLOCK_BINS)
    return np.sqrt(hist.reshape(n, gx * gy, -1))


@dataclass
class _Gradual:
    centre: int
    lo: int
    hi: int
    score: float
    start: int
    end: int
    fit: float  # 0..1 quality of the blend/wipe fit
    kind: str  # "blend" | "wipe"


def _check_gradual(frames: list[np.ndarray], lo: int, centre: int, p: CutParams) -> _Gradual | None:
    """Progressive A->B change over ``frames`` (analysis RGB, index 0 = ``lo``), not a pan."""
    n = len(frames)
    thumbs = [_block_resize(f, 48, 27) for f in frames]
    a, b = thumbs[0], thumbs[-1]
    mid = centre - lo
    min_frames = p.gradual_min_frames
    found: tuple[int, int, float, str] | None = None

    alphas, resids = zip(*(_blend(a, b, t) for t in thumbs), strict=True)
    al = np.array(alphas)
    inter = int(np.count_nonzero((al > 0.15) & (al < 0.85)))
    if (resids[mid] <= p.gradual_max_residual and 0.1 < al[mid] < 0.9 and inter >= min_frames
            and np.all(np.diff(al) <= 0.12)):
        start = int(np.argmax(al < 0.9))
        end = int(np.argmax(al <= 0.1)) if (al <= 0.1).any() else n - 1
        found = (start, max(end, start + 1), 1.0 - resids[mid] / p.gradual_max_residual, "blend")
    else:
        # wipe: each block of a 6x4 grid switches from A's colour histogram to B's, the switched
        # fraction rising monotonically (histograms tolerate camera drift inside a block)
        hists = _grid_hists(np.stack(frames))  # [n, blocks, bins], sqrt-normalised
        dist = lambda x, y: np.sqrt(np.clip(1.0 - (x * y).sum(-1), 0.0, 1.0))  # noqa: E731
        da, db, dab = dist(hists, hists[:1]), dist(hists, hists[-1:]), dist(hists[0], hists[-1])
        disc = dab > 0.3
        if disc.mean() >= 0.5:
            near = np.minimum(da, db) < 0.5 * dab[None]
            frac_b = ((db < da) & disc[None]).sum(1) / disc.sum()
            explained = (near & disc[None]).sum(1) / disc.sum()
            mids = (frac_b > 0.1) & (frac_b < 0.9)
            if (int(mids.sum()) >= min_frames and np.all(np.diff(frac_b) >= -0.1)
                    and float(explained[mids].mean()) >= p.wipe_explained):
                start = int(np.argmax(frac_b > 0.1))
                end = int(np.argmax(frac_b >= 0.9)) if (frac_b >= 0.9).any() else n - 1
                found = (start, max(end, start + 1), float(explained[mids].mean()), "wipe")
    if found is None:
        return None
    grays = [_block_resize(_gray(f.astype(np.float32)), 64, 36) for f in frames]
    ratios = [_shift_ratio(grays[j], grays[j + 1]) for j in range(n - 1)]
    if float(np.median(ratios)) < p.pan_ratio:
        return None  # content translates: camera motion, not a transition
    start, end, fit, kind = found
    return _Gradual(centre, lo, lo + n - 1, 0.0, lo + start, lo + end, float(np.clip(fit, 0, 1)), kind)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _robust(scores: np.ndarray, i: int, half: int, guard: int, floor: float, k_sigma: float
            ) -> tuple[float, float]:
    """(threshold, base) for index i from scores within ``half`` excluding ``guard`` around i."""
    n = scores.size
    lo, hi = max(0, i - half), min(n, i + half + 1)
    local = np.concatenate([scores[lo:max(lo, i - guard)], scores[min(hi, i + guard + 1):hi]])
    local = local[np.isfinite(local)]
    if local.size < 3:
        return floor, 0.0
    med = float(np.median(local))
    mad = float(np.median(np.abs(local - med))) * 1.4826
    return max(floor, med + k_sigma * mad), med


def _confidence(s: float, thr: float, base: float) -> float:
    margin = (s - base) / max(thr - base, 1e-6)
    return float(np.clip(1.0 - 1.0 / margin, 0.0, 1.0)) if margin > 0 else 0.0


class _Streamer:
    """Single decoding pass: features and small grey thumbnails for every frame; gradual
    candidates evaluated on the fly from a ring buffer of analysis frames."""

    def __init__(self, fps: float, p: CutParams) -> None:
        self.p = p
        self.span = max(2, round(p.gradual_span * fps))
        self.feats: list[np.ndarray] = []
        self.grays: list[np.ndarray] = []
        self.means: list[np.ndarray] = []
        self.gscore: list[float] = []  # gscore[k] belongs to frame k + span
        self.ring: deque[tuple[int, np.ndarray]] = deque(maxlen=3 * self.span + 2)
        self.candidates: list[_Gradual] = []
        self.n = 0
        self.next_centre = self.span

    def add_batch(self, rgb: np.ndarray) -> None:
        self.feats.append(features_batch(rgb).astype(np.float16))
        self.grays.append(_gray_thumbs(rgb))
        self.means.append(rgb.reshape(len(rgb), -1, 3).mean(axis=1).astype(np.float32))
        for img in rgb:
            self.ring.append((self.n, img))
            self.n += 1
            j = self.n - 1 - self.span  # d(j - span, j + span) is now computable
            if j >= self.span:
                d = feature_distance(self._feat(j - self.span), self._feat(j + self.span))
                self.gscore.append(float(d))
            self._evaluate(final=False)

    def _feat(self, i: int) -> np.ndarray:
        b = self.p.batch
        return self.feats[i // b][i % b]

    def _g(self, j: int) -> float:
        k = j - self.span
        return self.gscore[k] if 0 <= k < len(self.gscore) else float("nan")

    def _evaluate(self, final: bool) -> None:
        if not self.p.gradual:
            return
        span = self.span
        last_known = span + len(self.gscore) - 1
        while self.next_centre <= last_known and (final or self.next_centre + span <= last_known):
            c = self.next_centre
            self.next_centre += 1
            g = self._g(c)
            if not g >= self.p.gradual_threshold:
                continue
            nb = [self._g(j) for j in range(c - span, c + span + 1)]
            if any(x > g for x in nb if np.isfinite(x)) or any(x == g for x in nb[:span]):
                continue  # not the (first) local maximum
            frames = dict(self.ring)
            lo, hi = c - span, c + span
            if lo not in frames or hi not in frames:
                continue
            res = _check_gradual([frames[i] for i in range(lo, hi + 1)], lo, c, self.p)
            if res is not None:
                res.score = g
                self.candidates.append(res)

    def finish(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._evaluate(final=True)
        feats = np.concatenate(self.feats) if self.feats else np.zeros((0, FEATURE_DIM), np.float16)
        grays = np.concatenate(self.grays) if self.grays else np.zeros((0, 144), np.uint8)
        means = np.concatenate(self.means) if self.means else np.zeros((0, 3), np.float32)
        g = np.full(self.n, np.nan, np.float32)
        if self.gscore:
            g[self.span:self.span + len(self.gscore)] = self.gscore
        return feats, grays, means, g


def detect_cuts(path: PathLike, params: CutParams | None = None, *, ffmpeg: str | None = None
                ) -> CutAnalysis:
    """Detect hard cuts and gradual transitions in a video (see module docstring)."""
    p = params or CutParams()
    origin = seek_origin_us(path, ffmpeg) / 1e6
    times: list[float] = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        cc = s.codec_context
        w0, h0 = int(cc.width), int(cc.height)
        aw = min(p.analysis_width, w0)
        ah = max(8, round(h0 * aw / w0))
        fps = float(s.average_rate or s.base_rate or 30)
        st = _Streamer(fps, p)
        buf: list[np.ndarray] = []
        for frame in c.decode(s):
            buf.append(frame.reformat(width=aw, height=ah, format="rgb24").to_ndarray())
            times.append(float(frame.time) - origin if frame.time is not None else len(times) / fps)
            if len(buf) == p.batch:
                st.add_batch(np.stack(buf))
                buf = []
        if buf:
            st.add_batch(np.stack(buf))
    if st.n == 0:
        raise ValueError(f"no frames decoded from {path}")
    feats, grays, means, gscore = st.finish()
    t = np.asarray(times, np.float64)
    return _decide(feats, grays, means, gscore, st.candidates, t, fps, st.span, p)


def _chroma(rgb_mean: np.ndarray) -> np.ndarray:
    return rgb_mean / max(float(rgb_mean.sum()), 1e-3)


def _dips(luma: np.ndarray, feats: np.ndarray, fps: float, p: CutParams) -> list[tuple[int, int, float]]:
    """Dips to black between different content: (first fading frame, first frame after the
    fade-in, distance before/after). Fades are monotone luminance ramps up to ``dip_fade`` s."""
    n = luma.size
    dark = luma <= p.dip_black
    max_fade = max(1, round(p.dip_fade * fps))
    max_dark = max(1, round(p.dip_max_dark * fps))
    out = []
    i = 0
    while i < n:
        if not dark[i]:
            i += 1
            continue
        j = i
        while j < n and dark[j]:
            j += 1
        if i > 0 and j < n and j - i <= max_dark:
            # fade-out starts after the last frame at >= 90% of the level before it; the fade-in
            # ends at the first frame back at >= 90% of the level after it
            pre = luma[max(0, i - max_fade):i]
            post = luma[j:min(n, j + max_fade)]
            a = max(0, i - pre.size) + int(np.nonzero(pre >= 0.9 * pre.max())[0][-1]) + 1
            b = j + int(np.argmax(post >= 0.9 * post.max()))
            before, after = max(0, a - 1), min(n - 1, b + 1)
            d = float(feature_distance(feats[before], feats[after]))
            if d >= 2 * p.abs_threshold:
                out.append((a, b, d))
        i = j
    return out


def _decide(feats: np.ndarray, grays: np.ndarray, means: np.ndarray, gscore: np.ndarray,
            gcands: list[_Gradual], times: np.ndarray, fps: float, span: int, p: CutParams
            ) -> CutAnalysis:
    n = feats.shape[0]
    k = max(1, math.ceil(p.flash * fps))
    m_ret = max(k, round(p.flash_return * fps))
    half = max(5, round(p.window * fps))
    min_gap = max(1, round(p.min_shot * fps))
    iso = max(2 * k, round(p.isolation_window * fps))
    lags = {lag: _lag_distance(feats, lag) for lag in range(1, max(2 * k, m_ret + 2))}

    scores = np.full(n, np.inf, np.float32)
    for da in range(1, k + 1):
        for db in range(k):
            i = np.arange(da, n - db)
            if i.size:
                scores[i] = np.minimum(scores[i], lags[da + db][i - da])
    scores[0] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    thresholds = np.full(n, np.nan, np.float32)

    # --- hard candidates ---
    hard: list[tuple[float, int]] = []
    for i in (int(x) for x in np.nonzero(scores > p.abs_threshold)[0]):
        sc = float(scores[i])
        if i == 0 or sc < scores[max(1, i - k):min(n, i + k + 1)].max():
            continue
        thr, base = _robust(scores, i, half, k, p.abs_threshold, p.k_sigma)
        thresholds[i] = thr
        if sc <= thr:
            continue
        lo, hi = max(1, i - iso), min(n, i + iso + 1)
        others = np.concatenate([scores[lo:max(lo, i - k)], scores[min(hi, i + k + 1):hi]])
        ref = float(np.sort(others)[-2]) if others.size >= 2 else 0.0
        if sc < p.isolation * ref:
            continue
        after = [float(lags[m + 1][i - 1]) for m in range(1, m_ret + 1) if i + m < n]
        before = [float(lags[m + 1][i - 1 - m]) for m in range(1, m_ret + 1) if i - 1 - m >= 0]
        if min(after + before, default=np.inf) < 0.5 * sc:
            continue  # content returns: a flash or glitch, not a cut
        g0, g1 = grays[i - 1].astype(np.float32), grays[i].astype(np.float32)
        same_colour = float(np.abs(_chroma(means[i]) - _chroma(means[i - 1])).sum()) <= p.luma_chroma
        if same_colour and abs(float(g1.mean() - g0.mean())) > p.luma_jump:
            a, b = g0 - g0.mean(), g1 - g1.mean()
            corr = float((a * b).sum() / max(math.sqrt(float((a * a).sum() * (b * b).sum())), 1e-6))
            if corr >= p.luma_corr:
                continue  # same picture, brighter/darker: flash or exposure change
        conf = _confidence(sc, thr, base)
        if conf >= p.min_confidence:
            hard.append((conf, i))

    # --- gradual: dips to black first, then dissolves/wipes with an adaptive threshold and no
    # hard cut in the window outside their own extent ---
    gradual: list[Cut] = []
    dropped_hard: set[int] = set()
    if p.gradual:
        luma = means @ np.array([0.299, 0.587, 0.114], np.float32)
        for a, b, d in _dips(luma, feats, fps, p):
            conf = float(np.clip((d - p.abs_threshold) / (0.3 - p.abs_threshold), 0.0, 1.0))
            if conf < p.min_confidence:
                continue
            dropped_hard.update(i for _, i in hard if a - k <= i <= b + k)
            gradual.append(Cut(b, _mid(times, b), round(conf, 4), round(d, 4), "gradual", a, _mid(times, a)))
    for gc in sorted(gcands, key=lambda g: -g.score):
        thr, base = _robust(gscore, gc.centre, 2 * half, span, p.gradual_threshold, p.k_sigma)
        if gc.score < thr:
            continue
        inside = [i for _, i in hard if gc.lo - k <= i <= gc.hi + k]
        if any(not (gc.start - k <= i <= gc.end + k) for i in inside):
            continue  # a real cut in the window: the "transition" is that cut plus motion
        if any(c.start_frame - span <= gc.centre <= c.frame + span
               or abs(gc.centre - (c.start_frame + c.frame) // 2) < min_gap + span for c in gradual):
            continue
        conf = _confidence(gc.score, thr, base) * gc.fit ** 0.5
        if conf < p.min_confidence:
            continue
        dropped_hard.update(inside)
        gradual.append(Cut(gc.end, _mid(times, gc.end), round(conf, 4), round(gc.score, 4), "gradual",
                           gc.start, _mid(times, gc.start)))

    hard.sort(key=lambda c: (-c[0], -scores[c[1]]))
    chosen: list[Cut] = []
    for conf, i in hard:
        if i in dropped_hard or any(c.start_frame - k <= i <= c.frame + k for c in gradual):
            continue
        if all(abs(i - c.frame) >= min_gap for c in chosen):
            chosen.append(Cut(i, _mid(times, i), round(conf, 4), round(float(scores[i]), 4)))
    cuts = sorted(chosen + gradual, key=lambda c: c.frame)
    return CutAnalysis(cuts, times, scores, thresholds, gscore, fps, p)


def _mid(times: np.ndarray, i: int) -> float:
    return float(0.5 * (times[i - 1] + times[i])) if i > 0 else float(times[0])
