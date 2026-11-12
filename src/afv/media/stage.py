"""The ``ingest`` stage: normalise one shot exactly as GEM-X does and describe it.

Writes into the stage's work folder:

- ``shot.mp4``: GEM-X's normalised shot (same ffmpeg command, byte-identical);
- ``proxy.mp4``: <=720p, short-GOP H.264 on the same 30 Hz frames (Blender reference video);
- ``overlay_mask.png``: static-overlay mask (255 = scoreboard/logo pixel) at the shot's size;

and returns a ``ShotMedia`` result on the shot timeline (frames 0..T-1, timestamps start + i/30).

Inputs: the source video, shot start and duration come from the params when set, else from
``<project>/project.json`` (``source`` / ``source_path`` / ``video``, ``sha256``) and
``<shot_dir>/shot.json`` (``start``, ``duration``; a missing duration means GEM-X's default, to
the end of the source).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from afv.core.stage import Ctx, StageError, register
from afv.core.types import LicenceRecord, Result
from afv.media.ffmpeg import EXPECTED_VERSION, MediaError, check_ffmpeg_version, ffmpeg_path
from afv.media.frames import sample_multi
from afv.media.letterbox import detect_letterbox
from afv.media.normalise import normalise_shot, shot_timestamps
from afv.media.overlay import OverlayParams, static_overlay_mask
from afv.media.png import write_png_gray
from afv.media.probe import PROBE_VERSION, ClipInfo, probe
from afv.media.proxy import make_proxy
from afv.media.result import ShotMedia

FFMPEG_LICENCE = LicenceRecord(
    component="ffmpeg (johnvansickle static build, with libx264)",
    code_licence="GPL-3.0-or-later",
    weight_licence="n/a",
    training_data_terms="n/a",
    commercial_output_ok=True,
    source_url="https://johnvansickle.com/ffmpeg/",
    notes="Used as an external program; the GPL does not cover the videos it writes.",
)


class IngestParams(BaseModel):
    source: str | None = None  # default: project.json
    start: float | None = None  # default: shot.json
    duration: float | None = None  # default: shot.json, else to the end of the source
    # the binary comes from AFV_FFMPEG_DIR (default ~/.local/bin); its version must match:
    expected_ffmpeg_version: str = EXPECTED_VERSION  # "" disables the check
    proxy: bool = True
    proxy_height: int = 720
    letterbox_samples: int = 16
    overlay: OverlayParams = Field(default_factory=OverlayParams)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise StageError(f"cannot parse {path}: {e}") from e


def _stored_probe(project: dict[str, Any], src: str) -> ClipInfo | None:
    """The ClipInfo module A stored in project.json at ``afv init``, if current and for this file."""
    stored = project.get("source", {}).get("probe") if isinstance(project.get("source"), dict) else None
    if not isinstance(stored, dict):
        return None
    try:
        info = ClipInfo.from_dict(stored)
    except (TypeError, ValueError):
        return None
    try:
        size = Path(src).stat().st_size
    except OSError:
        return None
    if info.probe_version != PROBE_VERSION or info.size_bytes != size:
        return None
    return info


def _source_from_project(project: dict[str, Any]) -> tuple[str | None, str]:
    src = project.get("source")
    sha = str(project.get("sha256", ""))
    if isinstance(src, dict):
        sha = str(src.get("sha256", sha))
        src = src.get("path")
    src = src or project.get("source_path") or project.get("video")
    return (str(src) if src else None), sha


@register("ffmpeg")
class IngestStage:
    name: ClassVar[str] = "ingest"
    version: ClassVar[str] = "2"
    env: ClassVar[str] = "cpu"
    inputs: ClassVar[tuple[str, ...]] = ()
    licences: ClassVar[tuple[LicenceRecord, ...]] = (FFMPEG_LICENCE,)
    Params: ClassVar[type[BaseModel]] = IngestParams
    weight_hash: ClassVar[str] = ""

    def run(self, ctx: Ctx, inputs: dict[str, Result], params: BaseModel) -> Result:
        p = params if isinstance(params, IngestParams) else IngestParams.model_validate(params)
        try:
            return self._run(ctx, p)
        except MediaError as e:
            raise StageError(str(e)) from e

    def _run(self, ctx: Ctx, p: IngestParams) -> ShotMedia:
        project = _read_json(ctx.shot_dir.parent.parent / "project.json")
        shot = _read_json(ctx.shot_dir / "shot.json")
        src, sha = _source_from_project(project)
        src = p.source or src
        if not src:
            raise StageError("ingest: no source video (params.source or project.json 'source')")
        start = p.start if p.start is not None else float(shot.get("start", 0.0))
        duration = p.duration if p.duration is not None else shot.get("duration")
        duration = None if duration is None else float(duration)
        ff = ffmpeg_path()
        version = check_ffmpeg_version(p.expected_ffmpeg_version, ff)
        work = Path(ctx.work_dir)
        work.mkdir(parents=True, exist_ok=True)

        ctx.progress(0.0, "probing source")
        # no full re-probe per shot: reuse afv init's probe, else read the header only
        info = _stored_probe(project, src) or probe(src, hash=not sha, packets=False, ffmpeg=ff)
        sha = sha or info.sha256
        ctx.progress(0.05, "normalising shot to 30 fps (GEM-X)")
        ns = normalise_shot(src, start, duration, work / "shot.mp4", ffmpeg=ff, info=info)
        if ns.num_frames <= 0:
            raise StageError(f"ingest: the shot {start:.3f}s+{ns.duration:.3f}s has no frames")

        ctx.progress(0.5, "letterbox and overlay mask")
        samples = sample_multi(work / "shot.mp4", ns.num_frames, {
            "lb": (p.letterbox_samples, None, True),
            "ov": (p.overlay.samples, p.overlay.max_width, False),
        })
        lb = detect_letterbox(samples["lb"])
        ov = static_overlay_mask(samples["ov"], out_size=(ns.width, ns.height), letterbox=lb,
                                 params=p.overlay)
        write_png_gray(work / "overlay_mask.png", ov.mask)

        proxy = ""
        if p.proxy:
            ctx.progress(0.75, "720p proxy")
            make_proxy(work / "shot.mp4", work / "proxy.mp4", height=p.proxy_height, ffmpeg=ff)
            proxy = "proxy.mp4"
        ctx.progress(1.0, f"shot: {ns.num_frames} frames at 30 fps")

        return ShotMedia(
            frames=list(range(ns.num_frames)),
            timestamps=shot_timestamps(ns.start, ns.num_frames),
            overlay_mask=ov.mask.astype("uint8"),
            source=str(Path(src).resolve()),
            source_sha256=sha,
            start=ns.start,
            duration=ns.duration,
            width=ns.width,
            height=ns.height,
            video="shot.mp4",
            proxy=proxy,
            overlay_png="overlay_mask.png",
            letterbox=[lb.top, lb.bottom, lb.left, lb.right],
            overlay_reliable=ov.reliable,
            scene_motion=round(ov.scene_motion, 4),
            overlay_boxes=[list(b) for b in ov.boxes],
            probe=info.to_dict(),
            ffmpeg_version=version,
        )
