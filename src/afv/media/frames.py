"""Decoding frames with PyAV (RGB output is pixel-identical to GEM-X's OpenCV ``VideoFrames``)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import av
import numpy as np

from afv.media.ffmpeg import seek_origin_us
from afv.media.probe import PathLike


def iter_frames(
    path: PathLike,
    *,
    start: int = 0,
    stop: int | None = None,
    size: tuple[int, int] | None = None,
    gray: bool = False,
) -> Iterator[np.ndarray]:
    """Decode frames ``start..stop-1`` in presentation order as uint8 arrays.

    RGB ``[H,W,3]`` (as GEM-X's ``VideoFrames``) or luma ``[H,W]``; ``size=(w, h)`` rescales.
    """
    for _, _, img in iter_frames_timed(path, start=start, stop=stop, size=size, gray=gray, origin=0.0):
        yield img


def iter_frames_timed(
    path: PathLike,
    *,
    start: int = 0,
    stop: int | None = None,
    size: tuple[int, int] | None = None,
    gray: bool = False,
    select: Sequence[int] | None = None,
    origin: float | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Like ``iter_frames`` but yields ``(index, seek_time, image)``; ``select`` keeps only those
    indices (sorted), decoding sequentially (exact, unlike seeking). Seek times are relative to
    ``origin`` (default: the pinned ffmpeg's seek origin of the file)."""
    fmt = "gray" if gray else "rgb24"
    wanted = None if select is None else set(int(i) for i in select)
    last = None if wanted is None else max(wanted, default=-1)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        if origin is None:
            origin = seek_origin_us(path) / 1e6
        for i, frame in enumerate(c.decode(s)):
            if (stop is not None and i >= stop) or (last is not None and i > last):
                break
            if i < start or (wanted is not None and i not in wanted):
                continue
            if size is not None:
                frame = frame.reformat(width=size[0], height=size[1])
            t = float(frame.time) - origin if frame.time is not None else float("nan")
            yield i, t, frame.to_ndarray(format=fmt)


def video_size(path: PathLike) -> tuple[int, int]:
    with av.open(str(path)) as c:
        cc = c.streams.video[0].codec_context
        return int(cc.width), int(cc.height)


def fit_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """Downscaled (w, h), both even, keeping aspect; unchanged if already narrow enough."""
    if width <= max_width:
        return width - width % 2, height - height % 2
    h = round(height * max_width / width)
    return max_width - max_width % 2, max(2, h - h % 2)


def sample_frames(
    path: PathLike,
    num: int,
    *,
    total: int | None = None,
    max_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Up to ``num`` evenly spaced RGB frames of a clip -> (indices [N], frames [N,H,W,3])."""
    if total is None:
        with av.open(str(path)) as c:
            total = int(c.streams.video[0].frames or 0)
    w, h = video_size(path)
    size = fit_size(w, h, max_width) if max_width else None
    if total and total > 0:
        idx = np.unique(np.linspace(0, total - 1, min(num, total)).round().astype(int))
        got = list(iter_frames_timed(path, size=size, select=idx.tolist()))
    else:
        got = list(iter_frames_timed(path, size=size))
        keep = np.unique(np.linspace(0, len(got) - 1, min(num, len(got))).round().astype(int))
        got = [got[k] for k in keep]
    if not got:
        raise ValueError(f"no frames decoded from {path}")
    return np.array([g[0] for g in got]), np.stack([g[2] for g in got])


def sample_multi(
    path: PathLike,
    total: int,
    specs: dict[str, tuple[int, int | None, bool]],
) -> dict[str, np.ndarray]:
    """Several evenly spaced samples in one decoding pass.

    ``specs[name] = (count, max_width or None, gray)``; returns ``name -> [N,H,W(,3)] uint8``.
    """
    w, h = video_size(path)
    plans = {}
    for name, (count, max_width, gray) in specs.items():
        idx = np.unique(np.linspace(0, max(total - 1, 0), max(1, min(count, total))).round().astype(int))
        size = fit_size(w, h, max_width) if max_width else None
        plans[name] = (set(idx.tolist()), size, gray)
    out: dict[str, list[np.ndarray]] = {name: [] for name in specs}
    last = max((max(p[0]) for p in plans.values()), default=-1)
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(s)):
            if i > last:
                break
            for name, (idx, size, gray) in plans.items():
                if i in idx:
                    f = frame.reformat(width=size[0], height=size[1]) if size else frame
                    out[name].append(f.to_ndarray(format="gray" if gray else "rgb24"))
    for name, imgs in out.items():
        if not imgs:
            raise ValueError(f"no frames decoded from {path} for {name}")
    return {name: np.stack(imgs) for name, imgs in out.items()}
