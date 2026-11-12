"""``ShotMedia``: the ingest stage's result.

``afv.core.types`` has no result kind for ingest, so this module registers ``"shot_media"`` with
``register_result_kind`` at import time (``load_result`` works once ``afv.media`` is imported).
The proposed contract change moves this class into ``afv.core.types``; see the module B report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from afv.core.types import Result, register_result_kind


@register_result_kind
@dataclass
class ShotMedia(Result):
    """The normalised shot (stage ingest): GEM-X's 30 Hz frames 0..T-1 and per-shot media facts.

    ``frames`` = 0..T-1 and ``timestamps`` = start + i/30 (the shot timeline). File names are
    relative to the stage's result folder. ``letterbox`` is [top, bottom, left, right] in pixels of
    the normalised frames; ``overlay_mask`` [H,W] uint8 (1 = static overlay pixel) is also saved
    as ``overlay_png``.
    """

    kind: ClassVar[str] = "shot_media"
    overlay_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.uint8))
    source: str = ""
    source_sha256: str = ""
    start: float = 0.0  # seek time passed to ffmpeg -ss
    duration: float = 0.0  # GEM-X's selected_duration
    fps: float = 30.0
    width: int = 0  # normalised (even-padded) size
    height: int = 0
    video: str = ""  # normalised shot, byte-identical to GEM-X's
    proxy: str = ""  # <=720p proxy on the same frames ("" if not made)
    overlay_png: str = ""
    letterbox: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    overlay_reliable: bool = True
    scene_motion: float = 0.0
    overlay_boxes: list[list[int]] = field(default_factory=list)  # [x0, y0, x1, y1] per component
    probe: dict[str, Any] = field(default_factory=dict)  # ClipInfo.to_dict() of the source
    ffmpeg_version: str = ""

