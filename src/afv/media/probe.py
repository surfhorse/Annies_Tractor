"""Clip probing with PyAV, plus the OpenCV-equivalent numbers GEM-X depends on.

What is measured, and by what:

- **Seek origin** (``ClipInfo.start_time``): the container start as the *pinned ffmpeg binary*
  reports it (``afv.media.ffmpeg.seek_origin_us``), because that is what ``-ss`` is relative to.
  PyAV's bundled libavformat can disagree (MKV/WebM with AAC/Opus: 0 vs -0.023 s); a note says so.
  Every time in ``afv.media`` is a seek time: ``pts * time_base - start_time``. The start is the
  minimum over *all* streams (audio often starts first, or earlier with AAC priming), so the first
  video frame is usually slightly after 0.
- **Packets, codec, size, rates, durations**: PyAV's demuxer (video packet pts equal the pinned
  binary's; the container ``duration`` may differ slightly between libavformat builds).
  ``num_frames`` counts video packets (``packets=False`` skips that full demux and uses the header).
- **OpenCV equivalents** (what GEM-X uses for the default interval), computed with OpenCV's
  FFmpeg-backend rule from PyAV's header values: ``CAP_PROP_FRAME_COUNT`` = the stream header's
  ``nb_frames``, or when that is 0 (MKV/WebM) ``floor(duration * fps + 0.5)``; ``CAP_PROP_FPS`` =
  ``avg_frame_rate`` (falling back to ``r_frame_rate``; checked against OpenCV 4.14 in the gem-x env
  on a 60/30 fps VFR file). OpenCV links its own libavformat, so for files without ``nb_frames``
  the duration (hence the count) could differ in rare cases; the tests compare with real OpenCV.

For constant-frame-rate MP4s the OpenCV numbers equal the real packet count and rate. They differ
for files without ``nb_frames`` (rounded estimate), when the header disagrees with the packets
(edit lists, truncated files), and for VFR files. ``ClipInfo.notes`` lists every difference found.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

from afv.media.ffmpeg import MediaError, seek_origin_us

PathLike = str | os.PathLike[str]
PROBE_VERSION = 2  # bump when ClipInfo's meaning changes (stored copies are then re-probed)


@dataclass(frozen=True)
class ClipInfo:
    path: str
    sha256: str  # "" when hashing was skipped
    size_bytes: int
    container: str
    codec: str
    pix_fmt: str
    width: int  # coded size of the video stream
    height: int
    rotation: int  # display rotation in degrees (counter-clockwise, ffmpeg convention); 0 = none
    display_width: int  # size after rotation (what ffmpeg/OpenCV output by default)
    display_height: int
    start_time: float  # seek-time origin: container start per the pinned ffmpeg, seconds
    first_frame_time: float  # first video frame, in seek time
    duration: float  # container duration (PyAV), seconds
    num_frames: int  # video packets present (header count when probed with packets=False)
    nb_frames: int  # stream header value (0 = unknown)
    avg_fps: float
    r_fps: float  # r_frame_rate (ffmpeg's "base" rate)
    measured_fps: float  # (num_frames - 1) / (last frame time - first frame time)
    vfr: bool
    has_audio: bool
    num_streams: int  # all streams in the file (ffmpeg's fps_mode choice depends on it)
    cv_frame_count: int  # OpenCV CAP_PROP_FRAME_COUNT equivalent
    cv_fps: float  # OpenCV CAP_PROP_FPS equivalent
    notes: list[str] = field(default_factory=list)
    packets_counted: bool = True  # False: num_frames/vfr/measured_fps come from the header only
    probe_version: int = 0  # PROBE_VERSION when probed; 0 = an older stored copy

    @property
    def cv_duration(self) -> float:
        """GEM-X's source duration (OpenCV frame count / fps): the default shot length from 0."""
        return self.cv_frame_count / self.cv_fps

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cv_duration"] = self.cv_duration
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClipInfo:
        """Rebuild from ``to_dict`` output; raises TypeError if fields are missing."""
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


def sha256_file(path: PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _rate(r: Fraction | None) -> float:
    return float(r) if r else 0.0


def _first_frame(stream: av.video.stream.VideoStream, container: av.container.InputContainer
                 ) -> tuple[int, float | None]:
    """Display rotation (degrees) and stream time of the first decoded frame."""
    try:
        frame = next(container.decode(stream))
    except (StopIteration, av.FFmpegError):
        return 0, None
    rot = round(getattr(frame, "rotation", 0) or 0)
    return rot % 360, (float(frame.time) if frame.time is not None else None)


@dataclass(frozen=True)
class PacketTable:
    """Integer timing of a video stream, read from packets (no decoding)."""

    pts: np.ndarray  # [N] int64, sorted (presentation order), stream time base
    time_base: Fraction
    last_duration: int  # duration of the last frame in time-base units (0 if unknown)
    start_time_us: int  # container start time, microseconds

    @property
    def times(self) -> np.ndarray:
        """Seek times (seconds from the container start) of every frame."""
        return self.pts.astype(np.float64) * float(self.time_base) - self.start_time_us / 1e6


def packet_table(path: PathLike, *, origin_us: int | None = None, ffmpeg: str | None = None
                 ) -> PacketTable:
    """Video packet timing; the seek origin comes from the pinned ffmpeg unless given."""
    if origin_us is None:
        origin_us = seek_origin_us(path, ffmpeg)
    with av.open(str(path)) as c:
        if not c.streams.video:
            raise MediaError(f"no video stream in {path}")
        s = c.streams.video[0]
        tb = Fraction(s.time_base)
        pts, durs = [], []
        for p in c.demux(s):
            if p.pts is None or p.size == 0:
                continue
            pts.append(p.pts)
            durs.append(p.duration or 0)
    order = np.argsort(np.asarray(pts, dtype=np.int64), kind="stable")
    arr = np.asarray(pts, dtype=np.int64)[order]
    last = int(np.asarray(durs, dtype=np.int64)[order][-1]) if durs else 0
    return PacketTable(arr, tb, last, int(origin_us))


def packet_times(path: PathLike) -> tuple[np.ndarray, float]:
    """Seek times of every video frame (sorted) and the container start time (seconds).

    Read from packets (no decoding), so it is fast; for B-frame streams the sorted pts give
    presentation order.
    """
    t = packet_table(path)
    return t.times, t.start_time_us / 1e6


def opencv_frame_count_fps(nb_frames: int, r_fps: float, avg_fps: float, duration: float,
                           stream_duration: float) -> tuple[int, float]:
    """OpenCV FFmpeg-backend ``CAP_PROP_FRAME_COUNT`` and ``CAP_PROP_FPS`` from stream metadata."""
    fps = avg_fps if avg_fps > 1e-6 else r_fps
    if nb_frames > 0:
        return nb_frames, fps
    sec = duration if duration > 1e-6 else stream_duration
    return math.floor(sec * fps + 0.5), fps


def probe(path: PathLike, *, hash: bool = True, packets: bool = True, ffmpeg: str | None = None
          ) -> ClipInfo:
    """Probe a video file. ``hash=False`` skips the sha256 (slow on multi-GB files);
    ``packets=False`` skips counting packets (a full demux) and uses header values."""
    p = Path(path)
    if not p.is_file():
        raise MediaError(f"no such video: {p}")
    origin_us = seek_origin_us(p, ffmpeg)
    start = origin_us / 1e6
    try:
        c = av.open(str(p))
    except av.FFmpegError as e:
        raise MediaError(f"cannot open video {p}: {e}") from e
    with c:
        if not c.streams.video:
            raise MediaError(f"no video stream in {p}")
        s = c.streams.video[0]
        cc = s.codec_context
        pyav_start = (c.start_time or 0) / av.time_base
        duration = (c.duration or 0) / av.time_base
        tb = float(s.time_base)
        stream_duration = (s.duration or 0) * tb
        avg_fps, r_fps = _rate(s.average_rate), _rate(s.base_rate)
        nb_frames = int(s.frames or 0)
        info = dict(
            container=c.format.name,
            codec=cc.name,
            pix_fmt=cc.format.name if cc.format else "",
            width=int(cc.width),
            height=int(cc.height),
            has_audio=bool(c.streams.audio),
            num_streams=len(c.streams),
        )
        rotation, first_time = _first_frame(s, c)
    cv_count, cv_fps = opencv_frame_count_fps(nb_frames, r_fps, avg_fps, duration, stream_duration)
    if cv_fps <= 0:
        raise MediaError(f"{p}: no usable frame rate")

    notes: list[str] = []
    vfr, measured = False, avg_fps or r_fps
    if packets:
        times = packet_table(p, origin_us=origin_us).times
        if times.size == 0:
            raise MediaError(f"video stream of {p} has no frames")
        num_frames = int(times.size)
        first = float(times[0])
        span = float(times[-1] - times[0]) if num_frames > 1 else 0.0
        measured = (num_frames - 1) / span if span > 0 else measured
        d = np.diff(times)
        if d.size >= 2:
            med = float(np.median(d))
            tol = max(1.5 * tb, 0.02 * med)
            odd = int(np.count_nonzero(np.abs(d - med) > tol))
            vfr = odd > 0
            if vfr:
                notes.append(f"variable frame rate: {odd} of {d.size} frame intervals irregular "
                             f"(longest {float(d.max() / med):.1f}x the typical one); GEM-X's fps=30 "
                             "filter resamples by timestamp")
        if cv_count != num_frames:
            src = "header nb_frames" if nb_frames > 0 else "round(duration*fps) (no nb_frames)"
            notes.append(f"OpenCV frame count {cv_count} ({src}) != {num_frames} packets")
        if abs(cv_fps - measured) > 0.01 * measured:
            notes.append(f"OpenCV fps {cv_fps:.4f} (avg_frame_rate) != measured {measured:.4f}")
    else:
        num_frames = nb_frames or cv_count
        first = (first_time - start) if first_time is not None else 0.0
        notes.append("header-only probe: packets not counted, VFR not measured")
    if abs(pyav_start - start) > 1e-6:
        notes.append(f"seek origin {start:.6f}s (pinned ffmpeg) != PyAV container start {pyav_start:.6f}s")
    w, h = info["width"], info["height"]
    dw, dh = (h, w) if rotation in (90, 270) else (w, h)
    if rotation:
        notes.append(f"rotation {rotation} deg: ffmpeg and OpenCV auto-rotate to {dw}x{dh}")

    return ClipInfo(
        path=str(p.resolve()),
        sha256=sha256_file(p) if hash else "",
        size_bytes=p.stat().st_size,
        rotation=rotation,
        display_width=dw,
        display_height=dh,
        start_time=start,
        first_frame_time=first,
        duration=duration,
        num_frames=num_frames,
        nb_frames=nb_frames,
        avg_fps=avg_fps,
        r_fps=r_fps,
        measured_fps=float(measured),
        vfr=vfr,
        cv_frame_count=cv_count,
        cv_fps=cv_fps,
        notes=notes,
        packets_counted=packets,
        probe_version=PROBE_VERSION,
        **info,
    )
