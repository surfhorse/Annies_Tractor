"""Shot normalisation identical to GEM-X (``gem/inference/video.py: normalize_video``).

GEM-X converts a source interval to 30 fps H.264 with::

    ffmpeg -hide_banner -loglevel error -nostdin -y -ss <start:.9f> -i <src> -t <dur:.9f>
           -map 0:v:0 -vf "fps=30,pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -crf 18
           -pix_fmt yuv420p -an -movflags +faststart <out.mp4>

where the default ``dur`` is OpenCV frame count / fps minus ``start``, and then reads the result
with OpenCV, so its frame count T is the output's ``nb_frames``. Every frame index of a shot in
afv is an index into exactly that file (0..T-1, timestamps ``start + i/30``).

``normalise_shot`` runs the same command with the same binary (see ``afv.media.ffmpeg``), so the
output file is byte-for-byte what GEM-X decodes. ``shot_frame_count`` predicts T exactly by running
the same input options and filter graph into ffmpeg's null muxer (decode + filter, no encode).
``estimate_frame_count`` is a cheap timestamp-only prediction for shot lists.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from afv.media.ffmpeg import MediaError, run_ffmpeg
from afv.media.frames import video_size
from afv.media.probe import ClipInfo, PacketTable, PathLike, opencv_frame_count_fps, packet_table, probe

GEMX_FPS = 30
GEMX_VF = "fps=30,pad=ceil(iw/2)*2:ceil(ih/2)*2"
GEMX_MAX_DURATION = 60.0
FRAMES_MARKER = ".afv-frames"


@dataclass(frozen=True)
class NormalisedShot:
    source: str
    start: float  # requested start, seek time (seconds)
    duration: float  # GEM-X's selected_duration
    num_frames: int  # T, as OpenCV/GEM-X would read it
    width: int  # even-padded
    height: int
    path: str  # the mp4 (or the frames folder for frames output)
    fps: float = float(GEMX_FPS)

    @property
    def timestamps(self) -> np.ndarray:
        return shot_timestamps(self.start, self.num_frames)


def shot_timestamps(start: float, num_frames: int) -> np.ndarray:
    """Contract timestamps of a shot's frames: ``start + i/30`` (float64 seconds)."""
    return start + np.arange(num_frames, dtype=np.float64) / GEMX_FPS


def select_interval(info: ClipInfo, start: float = 0.0, duration: float | None = None) -> float:
    """Validate an interval exactly as GEM-X does and return its ``selected_duration``."""
    start = float(start)
    if not math.isfinite(start) or start < 0:
        raise MediaError("Video start must be finite and non-negative")
    if duration is not None:
        duration = float(duration)
        if not math.isfinite(duration) or not 0 < duration <= GEMX_MAX_DURATION:
            raise MediaError("Video duration must be finite and within (0, 60] seconds")
    src_dur = info.cv_duration
    if start >= src_dur:
        raise MediaError(f"Video start {start:.3f}s must be before the end of the source ({src_dur:.3f}s)")
    if duration is None and src_dur > GEMX_MAX_DURATION:
        raise MediaError("Source video exceeds 60 seconds; specify the duration explicitly")
    selected = src_dur - start if duration is None else duration
    if start + selected > src_dur + 1e-9:
        raise MediaError(
            f"Requested interval {start:.3f}+{selected:.3f}s extends beyond the end of the source "
            f"({src_dur:.3f}s)"
        )
    return selected


def _input_args(src: PathLike, start: float, selected: float) -> list[str]:
    return ["-ss", f"{start:.9f}", "-i", str(Path(src).resolve()), "-t", f"{selected:.9f}",
            "-map", "0:v:0", "-vf", GEMX_VF]


def gemx_ffmpeg_args(src: PathLike, out: PathLike, start: float, selected: float) -> list[str]:
    """GEM-X's ffmpeg argument list (without the binary), verbatim."""
    return [
        "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        *_input_args(src, start, selected),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an",
        "-movflags", "+faststart", str(Path(out).resolve()),
    ]


def _resolve(src: PathLike, start: float, duration: float | None, info: ClipInfo | None
             ) -> tuple[ClipInfo, float]:
    info = info or probe(src, hash=False, packets=False)
    return info, select_interval(info, start, duration)


def normalise_shot(
    src: PathLike,
    start: float = 0.0,
    duration: float | None = None,
    out: PathLike = "shot.mp4",
    *,
    ffmpeg: str | None = None,
    info: ClipInfo | None = None,
    frames_ext: str = "jpg",
) -> NormalisedShot:
    """Normalise ``[start, start+duration)`` of ``src`` exactly as GEM-X does.

    ``out`` ending in ``.mp4`` writes GEM-X's file. An ``out`` without a suffix is a folder that
    receives the same file's decoded frames as ``00000.<frames_ext>`` .. (``jpg`` at -q:v 2, or
    ``png`` lossless), e.g. for SAM 2; frame i on disk is GEM-X's frame i. An existing folder is
    replaced only if it is empty or was written by this function (it holds a marker file); any
    other suffix is refused.
    """
    out = Path(out)
    if out.suffix and out.suffix.lower() != ".mp4":
        raise MediaError(f"normalised output must be a .mp4 file or a folder, not {out.name}")
    if Path(src).resolve() == out.resolve():
        raise MediaError("Normalized output must have a different path from the source video")
    if out.suffix == "" and out.exists():
        if not out.is_dir():
            raise MediaError(f"{out} exists and is not a folder")
        if any(out.iterdir()) and not (out / FRAMES_MARKER).is_file():
            raise MediaError(f"refusing to replace {out}: not empty and not an afv frames folder")
    info, selected = _resolve(src, start, duration, info)
    if out.suffix.lower() == ".mp4":
        out.parent.mkdir(parents=True, exist_ok=True)
        run_ffmpeg(gemx_ffmpeg_args(src, out, start, selected), ffmpeg)
        n = _nonzero(normalised_frame_count(out), start, selected)
        w, h = video_size(out)
        return NormalisedShot(str(Path(src).resolve()), float(start), selected, n, w, h, str(out.resolve()))

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / FRAMES_MARKER).write_text("frames written by afv.media.normalise_shot\n")
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".norm-") as tmp:
        mp4 = Path(tmp) / "shot.mp4"
        run_ffmpeg(gemx_ffmpeg_args(src, mp4, start, selected), ffmpeg)
        n = _nonzero(normalised_frame_count(mp4), start, selected)
        w, h = video_size(mp4)
        quality = ["-q:v", "2"] if frames_ext == "jpg" else []
        run_ffmpeg(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(mp4),
                    "-map", "0:v:0", "-fps_mode", "passthrough", *quality, "-start_number", "0",
                    str(out / f"%05d.{frames_ext}")], ffmpeg)
    written = len(list(out.glob(f"*.{frames_ext}")))
    if written != n:
        raise MediaError(f"wrote {written} frames but the normalised clip has {n}")
    return NormalisedShot(str(Path(src).resolve()), float(start), selected, n, w, h, str(out.resolve()))


def _nonzero(n: int, start: float, selected: float) -> int:
    if n <= 0:
        raise MediaError(f"the interval {start:.6f}s+{selected:.6f}s contains no 30 Hz frame "
                         "(GEM-X would fail on it too)")
    return n


def normalised_frame_count(path: PathLike) -> int:
    """Frame count of a normalised clip as GEM-X reads it (OpenCV ``CAP_PROP_FRAME_COUNT``)."""
    with av.open(str(path)) as c:
        if not c.streams.video:
            return 0  # ffmpeg wrote no frame at all (interval shorter than one 30 Hz tick)
        s = c.streams.video[0]
        nb = int(s.frames or 0)
        dur = (c.duration or 0) / av.time_base
        sdur = (s.duration or 0) * float(s.time_base)
        avg = float(s.average_rate) if s.average_rate else 0.0
        base = float(s.base_rate) if s.base_rate else 0.0
    return opencv_frame_count_fps(nb, base, avg, dur, sdur)[0]


def shot_frame_count(
    src: PathLike,
    start: float = 0.0,
    duration: float | None = None,
    *,
    ffmpeg: str | None = None,
    info: ClipInfo | None = None,
) -> int:
    """GEM-X's T for this interval, exactly.

    Runs the same ffmpeg with GEM-X's input options and filter into the same (mp4) muxer, with a
    32x32 scale appended and ultrafast x264 so encoding is nearly free. The muxer matters: ffmpeg
    picks ``-fps_mode cfr`` for mp4 (``vscfr`` when the input has a single stream), and cfr
    duplicates the first frame back to t=0 when the first source frame after ``start`` rounds to
    a later 1/30 tick; a null muxer would not. Costs one decode of the interval.
    """
    info, selected = _resolve(src, start, duration, info)
    with tempfile.TemporaryDirectory(prefix="afv-count-") as tmp:
        out = Path(tmp) / "count.mp4"
        args = _input_args(src, start, selected)
        args[args.index("-vf") + 1] = GEMX_VF + ",scale=32:32"
        run_ffmpeg(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args,
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-an", str(out)],
                   ffmpeg)
        return normalised_frame_count(out)


def estimate_frame_count(
    src: PathLike | None,
    start: float = 0.0,
    duration: float | None = None,
    *,
    info: ClipInfo | None = None,
    table: PacketTable | None = None,
) -> int:
    """Predict T from packet timestamps alone (no decoding). ``shot_frame_count`` is the
    authoritative value; this is for shot lists and planning.

    Model of ffmpeg 7's pipeline, in its integer arithmetic:

    1. input ``-ss``: timestamps shift by ``-(ss + container start)`` (microseconds, rescaled to
       the stream time base, round-to-nearest) and frames before 0 are dropped (accurate seek);
    2. ``fps=30`` (round=near): output ticks start at the first kept frame's rounded tick ``n0``
       and stop before the rounded end (pts + duration) of the last frame;
    3. output ``-t`` is enforced twice: by a ``trim`` after the filters (``n - n0 <
       round(30 * duration)``; a duration that rounds to 0 ticks disables the trim) and by the
       encoder's recording-time check (``n/30 < duration``);
    4. mp4 muxing uses ``-fps_mode cfr``, which fills ticks ``0..n0-1`` with copies of the first
       frame, unless the input file has exactly one stream (then ``vscfr``: no fill).

    Option strings are parsed like ``av_parse_time`` (6 decimal places, truncated).
    """
    if info is None:
        if src is None:
            raise ValueError("need src or info")
        info = probe(src, hash=False, packets=False)
    selected = select_interval(info, start, duration)
    if table is None:
        if src is None:
            raise ValueError("need src or table")
        table = packet_table(src)
    tb = table.time_base
    ss_us = _parse_time_us(f"{float(start):.9f}")
    dur_us = _parse_time_us(f"{selected:.9f}")
    offset = _rescale_near(ss_us + table.start_time_us, Fraction(1, 1_000_000), tb)
    rel = table.pts - offset
    kept = rel[rel >= 0]
    if kept.size == 0:
        return 0
    out_tb = Fraction(1, GEMX_FPS)
    n0 = _rescale_near(int(kept[0]), tb, out_tb)
    last_dur = table.last_duration
    if last_dur <= 0:
        rate = Fraction(info.avg_fps or info.r_fps).limit_denominator(100000)
        last_dur = _rescale_near(1, 1 / rate, tb)
    n_end = _rescale_near(int(kept[-1]) + last_dur, tb, out_tb)
    trim_ticks = _rescale_near(dur_us, Fraction(1, 1_000_000), out_tb)
    rec_end = -(-GEMX_FPS * dur_us // 1_000_000)  # ceil: first tick with n/30 >= duration
    limit = min(n0 + trim_ticks, rec_end) if trim_ticks > 0 else rec_end
    first = n0 if info.num_streams == 1 else 0
    return int(max(0, min(n_end, limit) - first))


def _parse_time_us(text: str) -> int:
    """``av_parse_time`` for a non-negative ``S.fffffffff`` string: microseconds, truncated."""
    whole, _, frac = text.partition(".")
    return int(whole) * 1_000_000 + int((frac + "000000")[:6])


def _rescale_near(a: int, b: Fraction, c: Fraction) -> int:
    """``av_rescale_q(a, b, c)``: a*b/c rounded to nearest, halves away from zero."""
    num = a * b.numerator * c.denominator
    den = b.denominator * c.numerator
    q, r = divmod(abs(num), den)
    if 2 * r >= den:
        q += 1
    return q if num >= 0 else -q
