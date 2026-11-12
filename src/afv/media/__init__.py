"""Module B, media: probing, GEM-X-identical shot normalisation, proxies, cut detection, shot
lists, letterbox detection and static-overlay masks.

All times are seek times: seconds from the container start, as ffmpeg's ``-ss`` measures them.
The ``ingest`` stage lives in ``afv.media.stage`` (imported by the runner).
"""

from afv.media.cuts import Cut, CutAnalysis, CutParams, detect_cuts
from afv.media.ffmpeg import MediaError, check_ffmpeg_version, ffmpeg_dir, ffmpeg_path, seek_origin_us
from afv.media.frames import iter_frames, sample_frames
from afv.media.letterbox import Letterbox, detect_letterbox
from afv.media.normalise import (
    GEMX_FPS,
    NormalisedShot,
    estimate_frame_count,
    normalise_shot,
    select_interval,
    shot_frame_count,
    shot_timestamps,
)
from afv.media.overlay import OverlayMask, OverlayParams, static_overlay_mask
from afv.media.png import read_png_gray, write_png_gray
from afv.media.probe import ClipInfo, packet_times, probe
from afv.media.proxy import make_proxy
from afv.media.result import ShotMedia
from afv.media.shots import Shot, detect_shots, shots_from_cuts

__all__ = [
    "GEMX_FPS",
    "ClipInfo",
    "Cut",
    "CutAnalysis",
    "CutParams",
    "Letterbox",
    "MediaError",
    "NormalisedShot",
    "OverlayMask",
    "OverlayParams",
    "Shot",
    "ShotMedia",
    "check_ffmpeg_version",
    "detect_cuts",
    "detect_letterbox",
    "detect_shots",
    "estimate_frame_count",
    "ffmpeg_dir",
    "ffmpeg_path",
    "iter_frames",
    "make_proxy",
    "normalise_shot",
    "packet_times",
    "probe",
    "read_png_gray",
    "sample_frames",
    "seek_origin_us",
    "select_interval",
    "shot_frame_count",
    "shot_timestamps",
    "shots_from_cuts",
    "static_overlay_mask",
    "write_png_gray",
]
