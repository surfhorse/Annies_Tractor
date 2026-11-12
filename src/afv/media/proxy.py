"""720p H.264 proxies (Blender reference video, quick review)."""

from __future__ import annotations

from pathlib import Path

from afv.media.ffmpeg import run_ffmpeg
from afv.media.normalise import GEMX_VF, select_interval
from afv.media.probe import ClipInfo, PathLike, probe


def make_proxy(
    src: PathLike,
    out: PathLike,
    *,
    height: int = 720,
    start: float | None = None,
    duration: float | None = None,
    shot_timeline: bool = False,
    crf: int = 20,
    gop: int = 15,
    ffmpeg: str | None = None,
    info: ClipInfo | None = None,
) -> Path:
    """Write a proxy no taller than ``height`` (never upscaled, even width, short GOP for scrubbing).

    - ``shot_timeline=False``: the source (or ``start``/``duration`` of it) at its own timing,
      with audio if present.
    - ``shot_timeline=True``: GEM-X's interval and ``fps=30`` filter, so proxy frame i is shot
      frame i. If ``src`` is already a normalised shot, pass ``shot_timeline=False``: a 30 fps CFR
      clip keeps its frames.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scale = f"scale=-2:'min({height},ih)':flags=bicubic"
    args = ["-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if shot_timeline:
        info = info or probe(src, hash=False)
        sel = select_interval(info, start or 0.0, duration)
        args += ["-ss", f"{start or 0.0:.9f}", "-i", str(Path(src).resolve()), "-t", f"{sel:.9f}",
                 "-map", "0:v:0", "-vf", f"{GEMX_VF},{scale}", "-an"]
    else:
        if start is not None:
            args += ["-ss", f"{start:.9f}"]
        args += ["-i", str(Path(src).resolve())]
        if duration is not None:
            args += ["-t", f"{duration:.9f}"]
        args += ["-map", "0:v:0", "-map", "0:a:0?", "-vf", scale, "-c:a", "aac", "-b:a", "128k"]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-g", str(gop),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    run_ffmpeg(args, ffmpeg)
    return out
