"""Fighter tracking (stage ``detect_track``): RF-DETR people + SAM 2.1 masklets, re-prompting and QC.

Model work runs in the track env through ``workers/track_worker.py`` (``afv.models.envs.run_worker``);
this module (numpy + pydantic) holds the picks schema, frame preparation, QC maths, the ``Tracks``
conversion and the stage.

Workflow
--------
1. ``afv run <shot>``: with no ``edits/fighters.json`` the stage writes ``prompt.jpg``/``prompt.json``
   (people numbered 0..n-1, largest box first) and returns ``Tracks`` without fighters ("needs pick").
   ``detect_prompt(video, out_dir, frames=[...])`` does the same for any frames.
2. ``afv pick <shot> --fighters A,B --extra R`` writes the base picks into ``fighters.json``, with the
   shot interval and source hash they were made for. Changing the picks drops earlier re-prompts
   (their ids would point at other people) unless ``--keep-prompts`` is given.
3. ``afv run`` tracks every picked person through the shot (SAM 2.1, both directions) and writes,
   in the stage folder: ``qc.json`` (``summary`` + run info), ``qc_frames.json`` (per-frame lists),
   ``review.mp4``, ``masks/<id>/<frame>.png`` and ``suggest_<frame>.jpg/json`` for each QC
   suggestion (a person nobody tracks, a drifted mask). Paths in these files are relative to the
   stage folder (``afv.core.project.stage_dir(shot_dir, "detect_track")``).
4. Re-prompt: add the suggested ``{"frame": F, "obj": ID, "detection": K, "box": [...]}`` to
   ``"prompts"`` in ``fighters.json`` (``add_prompt`` does it in Python); a new ``obj`` id adds a
   person, an existing id corrects one. Run again; frames and detections are cached in
   ``<shot>/.cache/track``.

``edits/fighters.json`` (version 1)
-----------------------------------
::

    {
      "fighters": [1, 0],        # base picks on the prompt image (written by `afv pick`):
      "extra": [2],              #   detection numbers of fighter A, fighter B, then other people
      "prompt_frame": 0,         #   frame of the prompt image (default 0)
      "boxes": {"1": [x0, y0, x1, y1], "0": [...], "2": [...]},  # the picked boxes (optional)
      "shot": {"start": 4.0, "duration": 1.5, "source_sha256": "..."},  # what the picks are for
      "objects": [{"id": 3, "label": "referee"}],                # optional labels / roles
      "prompts": [                                               # optional re-prompts
        {"frame": 141, "obj": 4, "detection": 2, "box": [271.6, 176.0, 529.5, 709.4]},
        {"frame": 210, "obj": 1, "box": [500, 120, 760, 680], "points": [[630, 400, 1], [700, 300, 0]]}
      ]
    }

- Object ids: base picks get A = 1, B = 2, extra[i] = 3 + i. These are the ``Tracks.track_ids``.
- ``frame`` is on the shot's 30 Hz timeline (0-based, GEM-X's frame index).
- A prompt needs ``detection`` (number on that frame's image, area-sorted RF-DETR people at the
  stage's ``det_threshold``) and/or ``box`` (shot pixels; wins over ``detection``). ``points``
  (SAM 2 clicks ``[x, y, 1|0]``, 1 = on the person) only refine a box: every prompt is a fresh
  conditioning frame for SAM 2 and a click alone often gives a partial mask.
- A prompt for an existing (frame, obj) replaces it; an unknown ``obj`` id creates a person. Two
  objects prompted with the same detection or overlapping boxes (IoU >= 0.7) on one frame are
  rejected: they would be the same person.
- ``objects`` may label people or give roles; without base picks it must give roles A and B.
- ``shot`` (optional): when present and it differs from the shot's interval or source, the stage
  stops with "re-pick".

``edits/ids.json``: identity swaps
----------------------------------
``{"swaps": [{"start": F, "end": E | null, "objects": [o1, o2]}]}`` (``afv pick <shot> --swap F:O1:O2``,
or the add-on): from frame ``F`` (to ``E`` exclusive, or the shot end) objects ``o1`` and ``o2`` trade
identities, e.g. after SAM 2 jumped from one fighter to the other. Swaps apply in file order, each to
the identities the earlier ones left (what the user sees). They are a post-process of the stored
tracking: when only ``ids.json`` changed since the current result (same picks, params, shot, version,
code and worker), the stage takes ``raw.npz`` and the masks from that result instead of tracking again (no
GPU), swaps the per-object data, recomputes the QC (and so the tracking hints) and re-renders the
review video. ``raw.npz`` always holds the data as tracked; ``masks/<id>/`` and the ``Tracks`` result
follow the swapped identities; ``qc.json`` records the swaps applied (``ids``) and the result the
tracking was reused from (``reused_tracking``). ``afv run --from detect_track`` tracks again.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, ClassVar

import numpy as np
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    WithJsonSchema,
    WrapValidator,
    field_validator,
    model_validator,
)

from afv.core.edits import (
    EditError,
    Frame,
    Int,
    Num,
    Pos,
    Role,
    Str,
    Strict,
    Swap,
    bad,
    describe_errors,
    is_dict,
    is_list,
    null_ok,
    nullable,
    parse_ids,
)
from afv.core.stage import Ctx, StageError, register
from afv.core.types import LicenceRecord, Result, Tracks

FPS = 30.0
WORKER = Path(__file__).with_name("workers") / "track_worker.py"
FIGHTERS_EDIT = "fighters"  # edits/fighters.json
IDS_EDIT = "ids"  # edits/ids.json (identity swaps)
CACHE_SUBDIR = Path(".cache") / "track"
FRAMES_CACHE_VERSION = 2
SAM2_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
RFDETR_M_SHA256 = "749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8"
IDENTITY_REASONS = ("appearance", "swap", "merge", "shared_detection")
REASONS = ("empty", "drift", "jump", "area_jump", *IDENTITY_REASONS)
HIST_DIM = 256  # worker's two-half 8x4x4 hue/saturation/value histogram
SAME_PERSON_IOU = 0.7  # two prompt boxes on one frame overlapping this much are one person
SUMMARY_LIST_CAP = 20  # runs/suggestions listed in qc.json's summary (all of them in qc_frames.json)

LICENCES: tuple[LicenceRecord, ...] = (
    LicenceRecord(
        component="RF-DETR Medium (rfdetr 1.12)",
        code_licence="Apache-2.0",
        weight_licence="Apache-2.0",
        training_data_terms="COCO / Objects365 pretraining; terms not reviewed",
        commercial_output_ok=True,
        source_url="https://github.com/roboflow/rf-detr",
        weight_sha256=RFDETR_M_SHA256,
        licence_text_path="licences/2026-11-09/RF-DETR_LICENSE.txt",
        retrieved="2026-11-09",
        notes="Only N/S/M/L sizes are Apache-2.0; do not install rfdetr[plus].",
    ),
    LicenceRecord(
        component="SAM 2.1 Hiera-L",
        code_licence="Apache-2.0",
        weight_licence="Apache-2.0",
        training_data_terms="SA-V / SA-1B; terms not reviewed",
        commercial_output_ok=True,
        source_url="https://github.com/facebookresearch/sam2",
        weight_sha256=SAM2_SHA256,
        licence_text_path="licences/2026-11-09/SAM2_LICENSE.txt",
        retrieved="2026-11-09",
    ),
)


class TrackError(StageError):
    """Tracking failed or its input is invalid; the message is fit to show the user."""


# ---------------------------------------------------------------------------
# fighters.json
# ---------------------------------------------------------------------------


# The values are strict, like the other edit files (``afv.core.edits``): no bool, no "3", no 3.0 for an
# integer, no NaN/infinity; the messages are ``path: what is wrong, got value``.
Nums = Annotated[list[Num], BeforeValidator(is_list)]
_NUMS = {"type": "array", "items": {"type": "number"}}
Opt = WrapValidator(null_ok)


def _check_box(v: list[float] | None) -> list[float] | None:
    if v is not None and (len(v) != 4 or not (v[2] > v[0] and v[3] > v[1])):
        raise ValueError(f"box must be [x0, y0, x1, y1] with x1 > x0 and y1 > y0, got {v}")
    return v


class TrackObject(Strict):
    model_config = ConfigDict(extra="forbid")
    id: Pos
    role: Annotated[Role, Opt, nullable({"type": "string", "enum": ["A", "B"]})] = None
    label: Str = ""

    @property
    def name(self) -> str:
        return self.label or (f"fighter {self.role}" if self.role else f"person {self.id}")


class TrackPrompt(Strict):
    model_config = ConfigDict(extra="forbid")
    frame: Frame
    obj: Pos
    detection: Annotated[Frame, Opt, nullable({"type": "integer", "minimum": 0})] = None
    box: Annotated[Nums, Opt, nullable(_NUMS)] = None
    points: Annotated[
        list[Nums], BeforeValidator(is_list), Opt, nullable({"type": "array", "items": _NUMS})
    ] = None

    @field_validator("box")
    @classmethod
    def _box(cls, v: list[float] | None) -> list[float] | None:
        return _check_box(v)

    @field_validator("points")
    @classmethod
    def _points(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        for p in v or []:
            if len(p) != 3 or p[2] not in (0, 1):
                raise ValueError(f"points are [x, y, 1|0], got {p}")
        return v

    @model_validator(mode="after")
    def _has_target(self) -> TrackPrompt:
        if self.box is None and self.detection is None:
            raise ValueError(
                f"prompt for object {self.obj} at frame {self.frame} needs a box or a detection "
                "(points only refine a box)"
            )
        return self


class ShotRef(Strict):
    """The shot the picks were made for (written by ``afv pick``)."""

    model_config = ConfigDict(extra="forbid")
    start: Num
    duration: Num
    source_sha256: Str = ""


def _version(v: Any) -> Any:
    if isinstance(v, bool) or not isinstance(v, int) or v != 1:
        raise bad("must be 1", v)
    return v


class FightersFile(Strict):
    """``edits/fighters.json`` as stored (see the module docstring)."""

    model_config = ConfigDict(extra="forbid")
    version: Annotated[Any, BeforeValidator(_version), WithJsonSchema({"const": 1, "type": "integer"})] = 1
    fighters: Annotated[
        list[Int], BeforeValidator(is_list), Opt, nullable({"type": "array", "items": {"type": "integer"}})
    ] = None
    extra: Annotated[list[Int], BeforeValidator(is_list)] = Field(default_factory=list)
    prompt_frame: Annotated[Frame, Opt, nullable({"type": "integer", "minimum": 0})] = None
    boxes: Annotated[dict[str, Nums], BeforeValidator(is_dict)] = Field(default_factory=dict)
    shot: ShotRef | None = None
    objects: Annotated[list[TrackObject], BeforeValidator(is_list)] = Field(default_factory=list)
    prompts: Annotated[list[TrackPrompt], BeforeValidator(is_list)] = Field(default_factory=list)

    @field_validator("boxes")
    @classmethod
    def _boxes(cls, v: dict[str, list[float]]) -> dict[str, list[float]]:
        for b in v.values():
            _check_box(b)
        return v

    @model_validator(mode="after")
    def _check(self) -> FightersFile:
        if self.fighters is not None:
            picks = [*self.fighters, *self.extra]
            if len(self.fighters) != 2:
                raise ValueError(f"fighters needs two detection numbers (A, B), got {self.fighters}")
            if len(set(picks)) != len(picks) or min(picks) < 0:
                raise ValueError(f"fighters/extra must be distinct detection numbers, got {picks}")
        elif self.extra:
            raise ValueError("extra given without fighters")
        keys = [(p.frame, p.obj) for p in self.prompts]
        if len(set(keys)) != len(keys):
            raise ValueError("two prompts for the same object and frame; merge them")
        return self

    def to_edit(self) -> FightersEdit:
        objs: dict[int, TrackObject] = {}
        prompts: dict[tuple[int, int], TrackPrompt] = {}
        pf = self.prompt_frame or 0
        if self.fighters is not None:
            picks = [(self.fighters[0], "A"), (self.fighters[1], "B")] + [(d, None) for d in self.extra]
            for i, (d, role) in enumerate(picks):
                oid = i + 1
                objs[oid] = TrackObject(id=oid, role=role)  # type: ignore[arg-type]
                prompts[(pf, oid)] = TrackPrompt(frame=pf, obj=oid, detection=d, box=self.boxes.get(str(d)))
        for o in self.objects:
            old = objs.get(o.id)
            objs[o.id] = (
                o
                if old is None
                else TrackObject(id=o.id, role=o.role or old.role, label=o.label or old.label)
            )
        for p in self.prompts:
            objs.setdefault(p.obj, TrackObject(id=p.obj))
            prompts[(p.frame, p.obj)] = p
        if self.fighters is None and prompts:
            pf = min(f for f, _ in prompts)
        return FightersEdit(
            objects=sorted(objs.values(), key=lambda o: o.id),
            prompts=sorted(prompts.values(), key=lambda p: (p.frame, p.obj)),
            prompt_frame=pf,
        )


class FightersEdit(BaseModel):
    """Normalised picks: every tracked person and every SAM 2 prompt."""

    model_config = ConfigDict(extra="forbid")
    objects: list[TrackObject]
    prompts: list[TrackPrompt]
    prompt_frame: int = 0  # frame of the base prompt image

    @model_validator(mode="after")
    def _check(self) -> FightersEdit:
        ids = [o.id for o in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate object ids {ids}")
        roles = [o.role for o in self.objects if o.role]
        if sorted(roles) != ["A", "B"]:
            raise ValueError(f"need exactly one person with role A and one with role B, got {roles}")
        seen = set()
        for p in self.prompts:
            if p.obj not in ids:
                raise ValueError(f"prompt at frame {p.frame} refers to unknown object {p.obj}")
            if (p.frame, p.obj) in seen:
                raise ValueError(f"two prompts for object {p.obj} on frame {p.frame}; merge them")
            seen.add((p.frame, p.obj))
        unprompted = set(ids) - {p.obj for p in self.prompts}
        if unprompted:
            raise ValueError(f"objects without a prompt: {sorted(unprompted)}")
        _check_same_person(self.prompts)
        return self

    @property
    def fighters(self) -> dict[str, int]:
        return {o.role: o.id for o in self.objects if o.role}

    @property
    def obj_ids(self) -> list[int]:
        return [o.id for o in self.objects]

    def prompt_frames(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {o.id: [] for o in self.objects}
        for p in self.prompts:
            out[p.obj].append(p.frame)
        return {k: sorted(v) for k, v in out.items()}


def _check_same_person(prompts: Sequence[TrackPrompt]) -> None:
    """Two objects prompted on one frame with the same detection or boxes that mostly overlap would
    track the same person (e.g. a re-prompt left over from an earlier pick)."""
    by_frame: dict[int, list[TrackPrompt]] = {}
    for p in prompts:
        by_frame.setdefault(p.frame, []).append(p)
    for f, ps in by_frame.items():
        for i, a in enumerate(ps):
            for b in ps[i + 1 :]:
                same = a.box is None and b.box is None and a.detection == b.detection
                if a.box is not None and b.box is not None:
                    same = float(box_iou(np.array(a.box), np.array(b.box))[0, 0]) >= SAME_PERSON_IOU
                if same:
                    raise ValueError(
                        f"prompts for objects {a.obj} and {b.obj} on frame {f} point at the same person "
                        "(a re-prompt left from an earlier pick? remove it or re-pick)"
                    )


def _validation_message(e: ValidationError) -> str:
    return describe_errors(e)


def _load_file(src: Path | str | Mapping[str, Any]) -> FightersFile:
    try:
        data = src if isinstance(src, Mapping) else json.loads(Path(src).read_text())
        return FightersFile.model_validate(data)
    except ValidationError as e:
        raise TrackError(f"edits/fighters.json is invalid: {_validation_message(e)}") from None
    except (OSError, ValueError, TypeError, KeyError) as e:  # JSONDecodeError is a ValueError
        raise TrackError(f"edits/fighters.json cannot be read: {type(e).__name__}: {e}") from None


def load_fighters_edit(src: Path | str | Mapping[str, Any]) -> FightersEdit:
    """Parse and check ``fighters.json`` (a path or its parsed data); raises TrackError when invalid."""
    f = _load_file(src)
    try:
        return f.to_edit()
    except ValidationError as e:
        raise TrackError(f"edits/fighters.json is invalid: {_validation_message(e)}") from None


def check_shot(
    src: Path | str | Mapping[str, Any], start: float, duration: float, source_sha256: str
) -> None:
    """Raise TrackError when the picks were made for another interval or source video."""
    ref = _load_file(src).shot
    if ref is None:
        return
    diffs = []
    if abs(ref.start - start) > 1e-6 or abs(ref.duration - duration) > 1e-6:
        diffs.append(
            f"interval {ref.start:.3f}+{ref.duration:.3f} s, the shot is {start:.3f}+{duration:.3f} s"
        )
    if ref.source_sha256 and source_sha256 and ref.source_sha256 != source_sha256:
        diffs.append("a different source video")
    if diffs:
        raise TrackError(
            f"edits/fighters.json was picked for {' and '.join(diffs)}; the detection numbers no longer "
            "apply. Re-pick: afv pick <shot> (shows the prompt image), then afv pick <shot> --fighters A,B"
        )


def fighters_json_schema() -> dict[str, Any]:
    return FightersFile.model_json_schema()


@dataclass
class PromptInfo:
    """RF-DETR people on one frame, as numbered on ``image``."""

    frame: int
    image: Path
    boxes: np.ndarray  # [n,4] largest first
    scores: np.ndarray  # [n]

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> PromptInfo:
        boxes = np.asarray(d["boxes"], np.float32).reshape(-1, 4)
        frame = int(d.get("frame", d.get("prompt_frame", 0)))
        return cls(frame, Path(d.get("image", "")), boxes, np.asarray(d["scores"], np.float32))

    def box(self, k: int) -> list[float]:
        if not 0 <= k < len(self.boxes):
            raise TrackError(f"frame {self.frame} has detections 0..{len(self.boxes) - 1}, not {k}")
        return [round(float(v), 1) for v in self.boxes[k]]


def pick_data(
    prompt: PromptInfo,
    a: int,
    b: int,
    extra: Sequence[int] = (),
    labels: Mapping[int, str] | None = None,
    shot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The base picks as ``afv pick`` writes them (+ ``objects`` labels keyed by detection number,
    + ``shot`` = {start, duration, source_sha256} when given)."""
    picks = [a, b, *extra]
    data: dict[str, Any] = {
        "fighters": [a, b], "extra": list(extra), "prompt_frame": prompt.frame,
        "boxes": {str(k): prompt.box(k) for k in picks},
    }  # fmt: skip
    if shot is not None:
        data["shot"] = dict(shot)
    if labels:
        stray = sorted(set(labels) - set(picks))
        if stray:
            raise TrackError(f"labels for detections that are not picked: {stray}")
        data["objects"] = [{"id": picks.index(k) + 1, "label": lab} for k, lab in labels.items()]
    load_fighters_edit(data)
    return data


def next_object_id(data: Mapping[str, Any]) -> int:
    return max(load_fighters_edit(data).obj_ids) + 1


def add_prompt(
    data: Mapping[str, Any],
    frame: int,
    obj: int,
    *,
    detection: int | None = None,
    box: Sequence[float] | None = None,
    points: Sequence[Sequence[float]] | None = None,
    prompt: PromptInfo | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Copy of ``fighters.json`` data with a (re-)prompt added to ``prompts``; a new ``obj`` id adds a
    person (e.g. a referee who enters later). ``prompt`` (that frame's PromptInfo) stores the box with
    the detection number, so the pick survives detector changes."""
    load_fighters_edit(data)  # the input must be valid before it is edited
    if box is None and detection is not None and prompt is not None:
        if prompt.frame != frame:
            raise TrackError(f"prompt info is for frame {prompt.frame}, not {frame}")
        box = prompt.box(detection)
    new = json.loads(json.dumps(dict(data)))
    entry: dict[str, Any] = {"frame": int(frame), "obj": int(obj)}
    if detection is not None:
        entry["detection"] = int(detection)
    if box is not None:
        entry["box"] = [float(v) for v in box]
    if points is not None:
        entry["points"] = [[float(v) for v in p] for p in points]
    new["prompts"] = [p for p in new.get("prompts", []) if (p["frame"], p["obj"]) != (frame, obj)] + [entry]
    if label:
        new["objects"] = [o for o in new.get("objects", []) if o["id"] != obj] + [
            {"id": int(obj), "label": label}
        ]
    load_fighters_edit(new)
    return new


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


class TrackParams(BaseModel):
    """detect_track parameters (all part of the cache key)."""

    model_config = ConfigDict(extra="forbid")
    video: str | None = None  # overrides of the shot's source/interval (default: shot_context)
    start: float | None = None
    duration: float | None = None
    prompt_frame: int = 0  # frame of the prompt image when nothing is picked yet
    det_threshold: float = 0.4  # RF-DETR person confidence (prompt numbering and QC)
    match_iou: float = 0.5  # mask box vs detection IoU counted as "matched"
    max_unmatched_run: int = 8  # longer unmatched runs are flagged as drift (frames)
    jump_thresh: float = 0.35  # centroid step per frame, in box heights
    area_jump_ratio: float = 2.0  # mask area ratio between consecutive frames
    appearance_rel_margin: float = 0.35  # share of the reference separation that flags a mask
    appearance_margin: float = 0.03  # floor of that margin (Bhattacharyya distance)
    appearance_min_separation: float = 0.15  # references closer than this: check not discriminative
    appearance_min_run: int = 8
    merge_iou: float = 0.5  # two people's masks overlapping this much (IoU) ...
    merge_min_run: int = 3  # ... for this many frames: one mask has drifted onto the other person
    shared_det_min_run: int = 8  # frames two people match the same detection before it is flagged
    untracked_min_score: float = 0.6
    untracked_min_rel_height: float = 0.5  # vs median fighter box height
    untracked_min_run: int = 10
    prune_window: int = 20  # SAM 2 memory kept around the current/prompt frames; 0 = keep all
    offload_state: bool = False
    min_mask_part: float = 0.02  # mask components smaller than this share are ignored for the box
    review_video: bool = True
    review_max_width: int = 1280

    @field_validator("prune_window")
    @classmethod
    def _window(cls, v: int) -> int:
        if v != 0 and v < 16:
            raise ValueError("prune_window must be 0 (keep all) or >= 16 (SAM 2.1 reads 15 frames back)")
        return v


# ---------------------------------------------------------------------------
# QC maths (pure numpy)
# ---------------------------------------------------------------------------


@dataclass
class RawTrack:
    """Worker output: per-object mask features on every frame plus all RF-DETR detections."""

    obj_ids: np.ndarray  # [P]
    boxes: np.ndarray  # [F,P,4] mask boxes, NaN where empty
    area: np.ndarray  # [F,P] mask pixels
    centroid: np.ndarray  # [F,P,2]
    hist: np.ndarray  # [F,P,H] two-half HSV histogram
    det_frame: np.ndarray  # [N] frame-sorted
    det_boxes: np.ndarray  # [N,4] per frame largest first
    det_scores: np.ndarray  # [N]
    width: int = 0
    height: int = 0
    mask_iou: np.ndarray | None = None  # [F,P,P] pairwise mask IoU (low resolution)

    @property
    def num_frames(self) -> int:
        return int(self.boxes.shape[0])

    def dets(self, f: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = np.searchsorted(self.det_frame, [f, f + 1])
        return self.det_boxes[s:e], self.det_scores[s:e]

    @classmethod
    def load(cls, path: Path) -> RawTrack:
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
        return cls(**{k: (int(v) if k in ("width", "height") else v) for k, v in d.items()})

    def save(self, path: Path) -> None:
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        d = {k: (np.int64(v) if k in ("width", "height") else v) for k, v in d.items() if v is not None}
        np.savez(path, **d)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of boxes a [N,4] against b [M,4] -> [N,M] (NaN boxes give 0)."""
    inter, area_a, area_b = _intersections(a, b)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        out = inter / np.maximum(union, 1e-6)
    return np.nan_to_num(out, nan=0.0)


def _intersections(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.nan_to_num(np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None))
    return inter, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) of each run of True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return [(int(s), int(e) - 1) for s, e in zip(d[::2], d[1::2], strict=True)]


def longest_run(mask: np.ndarray) -> int:
    return max((e - s + 1 for s, e in runs(mask)), default=0)


def keep_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros_like(np.asarray(mask, bool))
    for s, e in runs(mask):
        if e - s + 1 >= min_len:
            out[s : e + 1] = True
    return out


def bhattacharyya(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Bhattacharyya distance between histograms along the last axis (each normalised to sum 1)."""
    p = p / np.maximum(p.sum(-1, keepdims=True), 1e-12)
    q = q / np.maximum(q.sum(-1, keepdims=True), 1e-12)
    bc = np.sqrt(p * q).sum(-1)
    return np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))


def _median3(x: np.ndarray) -> np.ndarray:
    """Running median of 3 along axis 0 (NaN-aware, edges kept)."""
    if len(x) < 3:
        return x
    st = np.stack([x[:-2], x[1:-1], x[2:]])
    out = x.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows stay NaN
        out[1:-1] = np.nanmedian(st, axis=0)
    return out


@dataclass
class QCResult:
    summary: dict[str, Any]  # small (qc.json "summary", copied into the stage meta)
    details: dict[str, Any]  # per-frame lists (qc_frames.json)
    match_iou: np.ndarray  # [F,P] best IoU of the mask box with a detection
    match_score: np.ndarray  # [F,P] that detection's confidence when matched, else 0
    flags: dict[str, np.ndarray]  # reason -> [F,P] bool
    untracked: list[np.ndarray] = field(default_factory=list)  # per frame [k,4] boxes


def match_detections(raw: RawTrack, thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best IoU, its detection's score when >= thr (else 0) and its index (-1: none) per frame/object."""
    F, P = raw.boxes.shape[:2]
    best = np.zeros((F, P))
    score = np.zeros((F, P), np.float32)
    idx = np.full((F, P), -1, np.int64)
    for f in range(F):
        db, ds = raw.dets(f)
        if len(db) == 0:
            continue
        iou = box_iou(raw.boxes[f], db)  # [P,n]
        k = iou.argmax(1)
        best[f] = iou[np.arange(P), k]
        score[f] = np.where(best[f] >= thr, ds[k], 0.0)
        idx[f] = np.where(best[f] > 0, k, -1)
    return best, score, idx


def appearance_scores(
    raw: RawTrack, prompt_frames: Mapping[int, Sequence[int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame/object: distance to its own reference minus the smallest distance to another
    object's reference (> 0: the mask looks more like someone else), the own distance, and per object
    the separation of its reference from the nearest other reference.

    References are the mean histograms at each object's prompts (trusted): frames f-2..f+2 around its
    first prompt, f..f+2 for later (correction) prompts, whose earlier frames may be the drift."""
    F, P, _ = raw.hist.shape
    present = raw.area > 0
    med_area = np.array(
        [np.median(raw.area[present[:, k], k]) if present[:, k].any() else 0 for k in range(P)]
    )
    valid = present & (raw.area > 0.25 * med_area[None]) & (raw.hist.sum(-1) > 0)
    refs = np.zeros((P, raw.hist.shape[2]))
    for k, oid in enumerate(raw.obj_ids.tolist()):
        near = np.zeros(F, bool)
        for i, f in enumerate(sorted(prompt_frames.get(int(oid), []))):
            near[max(0, f - 2) if i == 0 else f : f + 3] = True
        sel = near & valid[:, k]
        if not sel.any():
            sel = np.zeros(F, bool)
            sel[np.flatnonzero(valid[:, k])[:15]] = True
        if sel.any():
            h = raw.hist[sel, k]
            refs[k] = (h / np.maximum(h.sum(-1, keepdims=True), 1e-12)).mean(0)
    has_ref = refs.sum(-1) > 0
    d = bhattacharyya(raw.hist[:, :, None, :], refs[None, None])  # [F,P,P]
    own = np.where(valid, d[:, np.arange(P), np.arange(P)], np.nan)
    if P < 2:
        return np.full((F, P), np.nan), own, np.full(P, np.nan)
    rd = bhattacharyya(refs[:, None, :], refs[None, :, :])
    rd[np.arange(P), np.arange(P)] = np.inf
    rd[:, ~has_ref] = np.inf
    sep = np.where(has_ref, rd.min(1), np.nan)
    other = d.copy()
    other[:, np.arange(P), np.arange(P)] = np.inf
    other[:, :, ~has_ref] = np.inf
    margin = own - other.min(-1)
    margin[:, ~has_ref] = np.nan
    margin[~valid] = np.nan
    return _median3(margin), own, sep


def _pair_runs(cond: np.ndarray, min_len: int) -> np.ndarray:
    """[F,P,P] pair condition -> [F,P] flags where any pair holds for >= min_len frames."""
    F, P = cond.shape[:2]
    out = np.zeros((F, P), bool)
    for i in range(P):
        for j in range(i + 1, P):
            r = keep_runs(cond[:, i, j], min_len)
            out[:, i] |= r
            out[:, j] |= r
    return out


def compute_qc(raw: RawTrack, edit: FightersEdit, p: TrackParams) -> QCResult:
    """Per-object QC and an identity check; see the report fields in ``summary``."""
    F, P = raw.boxes.shape[:2]
    ids = [int(i) for i in raw.obj_ids]
    objs = {o.id: o for o in edit.objects}
    fighter = np.array([objs[i].role is not None for i in ids])
    present = np.isfinite(raw.boxes[..., 0])
    best_iou, score, best_idx = match_detections(raw, p.match_iou)
    unmatched = present & (best_iou < p.match_iou)
    h = np.where(present, raw.boxes[..., 3] - raw.boxes[..., 1], np.nan)

    flags = {r: np.zeros((F, P), bool) for r in REASONS}
    flags["empty"] = ~present & fighter[None]
    for k in range(P):
        flags["drift"][:, k] = keep_runs(unmatched[:, k], p.max_unmatched_run + 1)

    # motion: centroid jumps and area jumps between consecutive frames
    both = present[1:] & present[:-1]
    step = np.linalg.norm(raw.centroid[1:] - raw.centroid[:-1], axis=-1)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        hm = np.nanmean(np.stack([h[1:], h[:-1]]), axis=0) if F > 1 else np.zeros((0, P))
        rel = step / np.maximum(hm, 1.0)
        ratio = np.maximum(raw.area[1:], 1) / np.maximum(raw.area[:-1], 1)
    flags["jump"][1:] = both & (rel > p.jump_thresh)
    flags["area_jump"][1:] = both & ((ratio > p.area_jump_ratio) | (ratio < 1 / p.area_jump_ratio))

    # identity 1: two people exchange positions between consecutive frames
    for i in range(P):
        for j in range(i + 1, P):
            ci0, ci1 = raw.centroid[:-1, i], raw.centroid[1:, i]
            cj0, cj1 = raw.centroid[:-1, j], raw.centroid[1:, j]
            keep = np.linalg.norm(ci1 - ci0, axis=-1) + np.linalg.norm(cj1 - cj0, axis=-1)
            swap = np.linalg.norm(ci1 - cj0, axis=-1) + np.linalg.norm(cj1 - ci0, axis=-1)
            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                scale = np.nanmin(np.stack([hm[:, i], hm[:, j]]), axis=0) if F > 1 else np.zeros(0)
                sw = both[:, i] & both[:, j] & (swap < 0.5 * keep) & (keep > 0.5 * p.jump_thresh * scale)
            flags["swap"][1:, i] |= sw
            flags["swap"][1:, j] |= sw

    # identity 2: two masks converge onto one person (gradual drift), or share one detection
    if raw.mask_iou is not None:
        flags["merge"] = _pair_runs(raw.mask_iou > p.merge_iou, p.merge_min_run)
    else:  # older raw files: boxes, stricter
        pair = np.stack([box_iou(raw.boxes[f], raw.boxes[f]) for f in range(F)]) if F else np.zeros((0, P, P))
        pair[:, np.arange(P), np.arange(P)] = 0
        flags["merge"] = _pair_runs(pair > 0.8, p.merge_min_run)
    matched = (best_iou >= p.match_iou) & (best_idx >= 0)
    shared = matched[:, :, None] & matched[:, None, :] & (best_idx[:, :, None] == best_idx[:, None, :])
    shared[:, np.arange(P), np.arange(P)] = False
    flags["shared_detection"] = _pair_runs(shared, p.shared_det_min_run)

    # identity 3: appearance closer to another person's reference than to its own, by a margin
    # calibrated per shot from how far apart the references are
    margin, own, sep = appearance_scores(raw, edit.prompt_frames())
    thr = np.maximum(p.appearance_margin, p.appearance_rel_margin * np.nan_to_num(sep, nan=0.0))
    usable = np.nan_to_num(sep, nan=0.0) >= p.appearance_min_separation
    app = (np.nan_to_num(margin, nan=-1.0) > thr[None]) & usable[None]
    for k in range(P):
        flags["appearance"][:, k] = keep_runs(app[:, k], p.appearance_min_run)

    new_id = max(max(ids), max(edit.obj_ids)) + 1
    untracked, suggestions = _untracked_people(raw, present, fighter, h, p, new_id)
    suggestions += _drift_suggestions(raw, ids, flags["drift"], best_iou, p)

    flagged = np.any(np.stack(list(flags.values())), axis=0)
    objects, details_obj = [], []
    for k, oid in enumerate(ids):
        o = objs[oid]
        fr = [
            [s, e, sorted({r for r in REASONS if flags[r][s : e + 1, k].any()})]
            for s, e in runs(flagged[:, k])
        ]
        objects.append({
            "id": oid, "role": o.role, "label": o.name,
            "present_frames": int(present[:, k].sum()),
            "empty_frames": int((~present[:, k]).sum()),
            "unmatched_frames": int(unmatched[:, k].sum()),
            "longest_unmatched_run": longest_run(unmatched[:, k]),
            "reason_frames": {r: int(flags[r][:, k].sum()) for r in REASONS},
            "appearance_check": "on" if usable[k] else "not discriminative",
            "appearance_separation": None if np.isnan(sep[k]) else round(float(sep[k]), 3),
            "appearance_threshold": round(float(thr[k]), 3),
            "appearance_margin_p95": _nanpct(margin[:, k], 95),
            "appearance_own_median": _nanpct(own[:, k], 50),
            "flagged_frames": int(flagged[:, k].sum()),
            "flagged_runs": fr[:SUMMARY_LIST_CAP],
            "flagged_runs_total": len(fr),
            "identity_ok": not any(flags[r][:, k].any() for r in IDENTITY_REASONS),
            "status": "ok" if not flagged[:, k].any() else "review",
        })  # fmt: skip
        details_obj.append({
            "id": oid, "unmatched_list": np.flatnonzero(unmatched[:, k]).tolist(), "flagged_runs": fr,
            **{f"{r}_frames": np.flatnonzero(flags[r][:, k]).tolist() for r in REASONS},
        })  # fmt: skip
    all_flagged = np.flatnonzero(flagged.any(1)).tolist()
    summary = {
        "frames": F, "objects": objects, "fighters": edit.fighters,
        "flagged_frame_count": len(all_flagged),
        "untracked_frames": int(sum(len(u) > 0 for u in untracked)),
        "suggestions": suggestions[:SUMMARY_LIST_CAP], "suggestion_count": len(suggestions),
        "ok": not all_flagged and not suggestions,
    }  # fmt: skip
    details = {"flagged_frames": all_flagged, "objects": details_obj, "suggestions": suggestions}
    return QCResult(summary, details, best_iou.astype(np.float32), score, flags, untracked)


def _nanpct(x: np.ndarray, q: float) -> float | None:
    x = x[np.isfinite(x)]
    return round(float(np.percentile(x, q)), 3) if len(x) else None


def _box_list(b: np.ndarray) -> list[float]:
    return [round(float(v), 1) for v in b]


def _hint_entry(frame: int, obj: int, det: int, box: list[float]) -> str:
    return json.dumps({"frame": frame, "obj": obj, "detection": det, "box": box})


def _untracked_people(
    raw: RawTrack, present: np.ndarray, fighter: np.ndarray, h: np.ndarray, p: TrackParams, new_id: int
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Confident, fighter-sized detections that no tracked mask covers (e.g. a referee who enters),
    linked into tracklets so that each untracked person gets one suggestion and its own new id."""
    F = raw.num_frames
    fh = h[:, fighter] if fighter.any() else h
    ref_h = float(np.nanmedian(fh)) if np.isfinite(fh).any() else 0.3 * max(raw.height, 1)
    tracklets: list[dict[str, Any]] = []
    gap = 5  # frames a tracklet may miss (occlusion, detector misses)
    for f in range(F):
        db, ds = raw.dets(f)
        cand = (ds >= p.untracked_min_score) & ((db[:, 3] - db[:, 1]) >= p.untracked_min_rel_height * ref_h)
        tb = raw.boxes[f][present[f]]
        if cand.any() and len(tb):
            inter, area_d, _ = _intersections(db, tb)
            contain = (inter / np.maximum(area_d, 1e-6)[:, None]).max(1)
            cand &= (box_iou(db, tb).max(1) < 0.3) & (contain < 0.7)
        ks = list(np.flatnonzero(cand))
        active = [t for t in tracklets if f - t["last"] <= gap]
        pairs = sorted(
            ((float(box_iou(t["box"], db[k])[0, 0]), ti, k) for ti, t in enumerate(active) for k in ks),
            reverse=True,
        )
        used_t, used_k = set(), set()
        for iou, ti, k in pairs:
            if iou < 0.3 or ti in used_t or k in used_k:
                continue
            used_t.add(ti)
            used_k.add(k)
            t = active[ti]
            t["hits"].append((f, int(k)))
            t["last"], t["box"] = f, db[k]
        for k in ks:
            if k not in used_k:
                tracklets.append({"hits": [(f, int(k))], "last": f, "box": db[k]})
    shown: list[np.ndarray] = [np.zeros((0, 4), np.float32)] * F
    suggestions = []
    good = [t for t in tracklets if len(t["hits"]) >= p.untracked_min_run]
    for i, t in enumerate(sorted(good, key=lambda t: t["hits"][0][0])):
        best: tuple[float, int, int] | None = None
        f0 = t["hits"][0][0]
        for f, k in t["hits"]:
            db, ds = raw.dets(f)
            shown[f] = np.vstack([shown[f], db[k][None]])
            if f <= f0 + 45:
                v = float(ds[k]) * float((db[k, 2] - db[k, 0]) * (db[k, 3] - db[k, 1]))
                if best is None or v > best[0]:
                    best = (v, f, k)
        assert best is not None
        _, f, k = best
        db, ds = raw.dets(f)
        oid, box = new_id + i, _box_list(db[k])
        suggestions.append({
            "kind": "untracked_person", "frames": [f0, t["hits"][-1][0]], "frame": f, "obj": oid,
            "detection": k, "box": box, "score": round(float(ds[k]), 3), "image": f"suggest_{f:05d}.jpg",
            "hint": f"a person nobody tracks is visible on frames {f0}-{t['hits'][-1][0]}. To track them add "
                    f'{_hint_entry(f, oid, k, box)} to "prompts" in edits/fighters.json',
        })  # fmt: skip
    return shown, suggestions


def _drift_suggestions(
    raw: RawTrack, ids: list[int], drift: np.ndarray, best_iou: np.ndarray, p: TrackParams
) -> list[dict[str, Any]]:
    """For each drift run: the detection nearest the object's last matched box, as a re-prompt."""
    out = []
    for k, oid in enumerate(ids):
        for s, e in runs(drift[:, k]):
            before = np.flatnonzero(best_iou[:s, k] >= p.match_iou)
            det, box = None, None
            db, _ = raw.dets(s)
            if len(before) and len(db):
                iou = box_iou(raw.boxes[before[-1], k][None], db)[0]
                if iou.max() > 0.2:
                    det, box = int(iou.argmax()), _box_list(db[iou.argmax()])
            hint = f"person {oid} matches no detection on frames {s}-{e}; check review.mp4"
            if det is not None and box is not None:
                hint += (
                    f". If the mask drifted, add {_hint_entry(s, oid, det, box)} "
                    'to "prompts" in edits/fighters.json'
                )
            out.append({
                "kind": "drift", "obj": oid, "frames": [s, e], "frame": s, "detection": det, "box": box,
                "image": f"suggest_{s:05d}.jpg", "hint": hint,
            })  # fmt: skip
    return out


# ---------------------------------------------------------------------------
# Identity swaps (edits/ids.json)
# ---------------------------------------------------------------------------


def swap_permutation(swaps: Sequence[Swap], obj_ids: Sequence[int], num_frames: int) -> np.ndarray:
    """[F,P] ints: on frame f, output object column k shows the tracked column ``perm[f, k]``. Swaps
    apply in order, each exchanging what the two objects show after the earlier swaps."""
    ids = [int(i) for i in obj_ids]
    perm = np.tile(np.arange(len(ids)), (num_frames, 1))
    for sw in swaps:
        where = f"edits/ids.json: swap from frame {sw.start}"
        if missing := [o for o in sw.objects if o not in ids]:
            raise TrackError(f"{where}: object(s) {missing} are not tracked (tracked: {ids})")
        if sw.start >= num_frames:
            raise TrackError(f"{where}: the shot has frames 0..{num_frames - 1}")
        end = num_frames if sw.end is None else min(int(sw.end), num_frames)
        i, j = ids.index(sw.objects[0]), ids.index(sw.objects[1])
        perm[sw.start : end, [i, j]] = perm[sw.start : end, [j, i]]
    return perm


def is_identity(perm: np.ndarray | None) -> bool:
    return perm is None or bool((perm == np.arange(perm.shape[1])[None]).all())


def apply_swaps(raw: RawTrack, perm: np.ndarray) -> RawTrack:
    """The per-object arrays of ``raw`` with identities swapped by ``perm`` (detections unchanged)."""
    if is_identity(perm):
        return raw

    def take(a: np.ndarray) -> np.ndarray:
        return np.take_along_axis(a, perm.reshape(perm.shape + (1,) * (a.ndim - 2)), axis=1)

    mask_iou = raw.mask_iou
    if mask_iou is not None:
        mask_iou = np.take_along_axis(np.take_along_axis(mask_iou, perm[:, :, None], 1), perm[:, None, :], 2)
    return dataclasses.replace(
        raw, boxes=take(raw.boxes), area=take(raw.area), centroid=take(raw.centroid), hist=take(raw.hist),
        mask_iou=mask_iou,
    )  # fmt: skip


def _place(src: Path, dst: Path, move: bool) -> None:
    if move:
        os.replace(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def arrange_masks(
    src: Path, src_perm: np.ndarray | None, dst: Path, perm: np.ndarray, obj_ids: Sequence[int], *, move: bool
) -> int:
    """Mask PNGs ``<src>/<obj id>/<frame:05d>.png`` (arranged by ``src_perm``; None = as tracked) into
    ``dst`` arranged by ``perm``: hard-linked (copied where links fail), or moved. Returns the count."""
    ids = [int(i) for i in obj_ids]
    num_frames = perm.shape[0]
    inv = np.argsort(perm, axis=1)  # tracked column -> output column, per frame
    for oid in ids:
        (dst / str(oid)).mkdir(parents=True, exist_ok=True)
    n = 0
    for ks, oid in enumerate(ids):
        d = Path(src) / str(oid)
        if not d.is_dir():
            continue
        with os.scandir(d) as it:
            entries = [(e.name, Path(e.path)) for e in it]
        for name, path in entries:
            stem, ext = os.path.splitext(name)
            if ext != ".png" or not stem.isdigit() or int(stem) >= num_frames:
                continue
            f = int(stem)
            tracked = int(src_perm[f, ks]) if src_perm is not None else ks
            _place(path, dst / str(ids[int(inv[f, tracked])]) / name, move)
            n += 1
    return n


def stored_tracking(
    ctx: Ctx, p: TrackParams, fighters: Any, version: str, weight_hash: str, *, ids: Any = None,
    code_hash: str | None = None,
) -> Path | None:  # fmt: skip
    """The current detect_track result folder when its tracking can be reused by this run: made by the
    same version, weights, code (``code_hash``, the stage's code record) and worker from the same picks,
    params and shot, and ``ids.json`` (``ids``: its parsed data, or None) is what changed; None
    otherwise or when the run is forced (``afv run --from detect_track``)."""
    if ctx.force or not isinstance(fighters, Mapping) or code_hash is None:
        return None
    from afv.core.cache import hash_obj, read_json
    from afv.core.codehash import script_hash
    from afv.core.config import params_to_json
    from afv.core.project import META_FILE, shot_context, stage_dir

    d = stage_dir(ctx.shot_dir, "detect_track")
    try:
        meta = read_json(d / META_FILE)
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or not all((d / n).exists() for n in ("raw.npz", "qc.json", "masks")):
        return None
    kp = meta.get("key_parts") or {}
    sc = shot_context(ctx.shot_dir)
    shot = {"source_sha256": sc.source_sha256, "start": sc.start, "duration": sc.duration}
    want = [version, weight_hash, params_to_json(p), hash_obj(fighters), shot, ctx.seed, ctx.deterministic]
    have = [
        kp.get("version"), kp.get("weight_hash"), kp.get("params"),
        (kp.get("edits") or {}).get(FIGHTERS_EDIT),
        {k: (kp.get("shot") or {}).get(k) for k in shot}, kp.get("seed"), kp.get("deterministic"),
    ]  # fmt: skip
    if hash_obj(want) != hash_obj(have):
        return None
    if (kp.get("edits") or {}).get(IDS_EDIT) == (None if ids is None else hash_obj(ids)):
        return None  # ids.json did not change: something else did (e.g. --code-changes rerun)
    code = meta.get("code") or {}
    label, h = script_hash(WORKER)
    if code.get("hash") != code_hash or (code.get("files") or {}).get(label) != h:
        return None  # other code (or none recorded): track again
    return d


# ---------------------------------------------------------------------------
# Tracks result, review overlay
# ---------------------------------------------------------------------------


def shot_timeline(num_frames: int, start: float) -> tuple[np.ndarray, np.ndarray]:
    """``frames`` 0..T-1 and ``timestamps`` start + i/30 (BUILD_PLAN: 30 Hz shot timeline)."""
    frames = np.arange(num_frames, dtype=np.int64)
    return frames, float(start) + frames / FPS


def build_tracks(raw: RawTrack, edit: FightersEdit, start: float, qc: QCResult) -> Tracks:
    frames, ts = shot_timeline(raw.num_frames, start)
    if [int(i) for i in raw.obj_ids] != edit.obj_ids:
        raise TrackError(f"worker objects {raw.obj_ids.tolist()} differ from the picks {edit.obj_ids}")
    return Tracks(
        frames=frames,
        timestamps=ts,
        track_ids=raw.obj_ids.astype(np.int64),
        boxes=raw.boxes.astype(np.float32),
        scores=qc.match_score.astype(np.float32),
        fighters=edit.fighters,
    )


def unpicked_tracks(num_frames: int, start: float) -> Tracks:
    """Result while nobody is picked yet (the runner reports 'needs pick')."""
    frames, ts = shot_timeline(num_frames, start)
    return Tracks(
        frames=frames,
        timestamps=ts,
        track_ids=np.zeros(0, np.int64),
        boxes=np.zeros((num_frames, 0, 4), np.float32),
        scores=np.zeros((num_frames, 0), np.float32),
    )


def review_overlay(raw: RawTrack, edit: FightersEdit, qc: QCResult, match_iou: float = 0.5) -> dict[str, Any]:
    objs = {o.id: o for o in edit.objects}
    ids = [int(i) for i in raw.obj_ids]
    unmatched = np.isfinite(raw.boxes[..., 0]) & (qc.match_iou < match_iou)
    tags = []
    for f in range(raw.num_frames):
        row = []
        for k, oid in enumerate(ids):
            o = objs[oid]
            t = f"{oid} {o.role or o.label or 'person'}" + (" ?" if unmatched[f, k] else "")
            rs = [r for r in REASONS if qc.flags[r][f, k] and r != "empty"]
            row.append(t + (" !" + ",".join(rs) if rs else ""))
        tags.append(row)
    return {
        "obj_ids": ids,
        "tags": tags,
        "notes": ["UNTRACKED PERSON" if len(u) else "" for u in qc.untracked],
        "untracked": [np.round(u, 1).tolist() for u in qc.untracked],
        "prompt_frames": {str(k): v for k, v in edit.prompt_frames().items()},
    }


# ---------------------------------------------------------------------------
# Frames and worker bridge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShotFrames:
    frames_dir: Path  # 00000.jpg .. on the 30 Hz shot timeline
    num_frames: int  # GEM-X's T
    start: float
    duration: float
    width: int = 0  # 0 = unknown
    height: int = 0


def _normaliser_id() -> str:
    """Identity of afv.media's GEM-X normalisation command (part of the frames cache key)."""
    from afv.media.normalise import gemx_ffmpeg_args

    cmd = json.dumps([str(a) for a in gemx_ffmpeg_args("SRC", "OUT", 1.5, 2.5)])
    return f"v{FRAMES_CACHE_VERSION}:" + hashlib.sha256(cmd.encode()).hexdigest()[:16]


def prepare_frames(video: Path | str, start: float, duration: float | None, cache_dir: Path) -> ShotFrames:
    """The shot's frames via ``afv.media.normalise_shot`` (GEM-X's exact command), cached in
    ``cache_dir`` while the source file, interval, ffmpeg and normaliser are unchanged. RF-DETR
    detections are cached beside them by the worker and dropped with them. ``cache_dir`` must be
    empty, missing, or a cache this function wrote (it is replaced when stale)."""
    from afv.media.ffmpeg import ffmpeg_path
    from afv.media.normalise import normalise_shot

    src = Path(video).resolve()
    st = src.stat()
    key = {
        "source": str(src),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "start": float(start),
        "duration": duration,
        "ffmpeg": ffmpeg_path(),
        "normaliser": _normaliser_id(),
    }
    cache_dir = Path(cache_dir)
    manifest = cache_dir / "frames.json"
    frames_dir = cache_dir / "frames"
    if manifest.is_file():
        try:
            m = json.loads(manifest.read_text())
        except ValueError:
            m = {}
        if m.get("key") == key and len(list(frames_dir.glob("*.jpg"))) == m.get("num_frames"):
            return ShotFrames(
                frames_dir, int(m["num_frames"]), float(m["start"]), float(m["duration"]),
                int(m.get("width", 0)), int(m.get("height", 0)),
            )  # fmt: skip
    if cache_dir.exists():
        if any(cache_dir.iterdir()) and not manifest.is_file():
            raise TrackError(f"{cache_dir} is not empty and is not a tracking frame cache; not deleting it")
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    ns = normalise_shot(src, start, duration, frames_dir)
    m = {
        "key": key,
        "num_frames": ns.num_frames,
        "start": ns.start,
        "duration": ns.duration,
        "width": ns.width,
        "height": ns.height,
    }
    manifest.write_text(json.dumps(m, indent=1))
    return ShotFrames(frames_dir, ns.num_frames, ns.start, ns.duration, ns.width, ns.height)


WorkerFn = Callable[[dict[str, Any], Path, bool], dict[str, Any]]


def envs_worker(progress: Callable[[float, str], None] | None = None) -> WorkerFn:
    """Bridge to the track env through ``afv.models.envs.run_worker`` (holds the GPU lock for GPU jobs)."""

    def call(request: dict[str, Any], work_dir: Path, gpu: bool) -> dict[str, Any]:
        from afv.models.envs import run_worker

        return run_worker(
            "track", WORKER, request, work_dir, gpu=gpu, name=f"track_{request['mode']}", progress=progress
        )

    return call


def _call(request: dict[str, Any], work_dir: Path, gpu: bool, worker: WorkerFn | None) -> dict[str, Any]:
    resp = (worker or envs_worker())(request, Path(work_dir), gpu)
    if not resp.get("ok", False):
        raise TrackError(f"tracking ({request['mode']}) failed: {resp.get('error', 'unknown error')}")
    return resp


def _frames_request(fr: ShotFrames) -> dict[str, Any]:
    return {"frames_dir": str(Path(fr.frames_dir).resolve()), "num_frames": fr.num_frames}


def check_prompts_in_image(edit: FightersEdit, fr: ShotFrames) -> None:
    """Prompt frames inside the shot, boxes overlapping the image, points inside it."""
    for p in edit.prompts:
        where = f"prompt for object {p.obj} at frame {p.frame}"
        if p.frame >= fr.num_frames:
            raise TrackError(f"{where}: the shot has frames 0..{fr.num_frames - 1}")
        if not (fr.width and fr.height):
            continue
        if p.box is not None:
            x0, y0, x1, y1 = p.box
            if x1 <= 0 or y1 <= 0 or x0 >= fr.width or y0 >= fr.height:
                raise TrackError(f"{where}: box {p.box} is outside the {fr.width}x{fr.height} image")
        for x, y, _ in p.points or []:
            if not (0 <= x < fr.width and 0 <= y < fr.height):
                raise TrackError(f"{where}: point ({x}, {y}) is outside the {fr.width}x{fr.height} image")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_prompt(
    video: Path | str,
    out_dir: Path,
    frames: Sequence[int] = (0,),
    *,
    start: float = 0.0,
    duration: float | None = None,
    cache_dir: Path | None = None,
    threshold: float = 0.4,
    worker: WorkerFn | None = None,
    shot_frames: ShotFrames | None = None,
) -> list[PromptInfo]:
    """Numbered RF-DETR people on any frames: ``out_dir/prompt_<frame:05d>.jpg`` + ``.json``."""
    out_dir = Path(out_dir).resolve()
    fr = shot_frames or prepare_frames(video, start, duration, Path(cache_dir or out_dir / "cache"))
    req = {
        "mode": "detect",
        **_frames_request(fr),
        "out_dir": str(out_dir),
        "frames": [int(f) for f in frames],
        "threshold": threshold,
    }
    resp = _call(req, out_dir, True, worker)
    return [PromptInfo.from_json(p) for p in resp["prompts"]]


@dataclass
class TrackRun:
    tracks: Tracks
    qc: dict[str, Any]  # qc.json: {"summary": ..., "prompts", "plan", "stats", ...}
    masks_dir: Path  # <masks_dir>/<obj id>/<frame:05d>.png, missing = empty
    review: Path | None
    stats: dict[str, Any]


def masks_dir_for(shot_dir: Path) -> Path:
    """Masks of the shot's current detect_track result: ``<dir>/<obj id>/<frame:05d>.png``."""
    from afv.core.project import stage_dir

    return stage_dir(shot_dir, "detect_track") / "masks"


def run_tracking(
    video: Path | str,
    edit: FightersEdit,
    out_dir: Path,
    *,
    start: float = 0.0,
    duration: float | None = None,
    cache_dir: Path | None = None,
    params: TrackParams | None = None,
    worker: WorkerFn | None = None,
    shot_frames: ShotFrames | None = None,
    swaps: Sequence[Swap] = (),
    reuse: Path | None = None,
) -> TrackRun:
    """Track every picked person through the shot. Writes into ``out_dir``: ``qc.json``,
    ``qc_frames.json``, ``review.mp4``, ``masks/``, ``raw.npz``, ``prompt.jpg/json`` (base prompt
    frame) and ``suggest_<frame>.jpg/json`` for each QC suggestion; paths are relative to out_dir.
    ``swaps`` (edits/ids.json) swap identities after tracking; ``reuse`` (a stored detect_track
    result folder, see ``stored_tracking``) takes its tracking instead of running SAM 2."""
    p = params or TrackParams()
    out_dir = Path(out_dir).resolve()
    fr = shot_frames or prepare_frames(video, start, duration, Path(cache_dir or out_dir / "cache"))
    check_prompts_in_image(edit, fr)
    if reuse is not None:
        return _reuse_tracking(Path(reuse), edit, out_dir, fr, p, worker, swaps)
    req = {
        "mode": "track", **_frames_request(fr), "out_dir": str(out_dir), "obj_ids": edit.obj_ids,
        "prompts": [q.model_dump(exclude_none=True) for q in edit.prompts], "det_threshold": p.det_threshold,
        "prune_window": p.prune_window, "offload_state": p.offload_state, "min_mask_part": p.min_mask_part,
    }  # fmt: skip
    resp = _call(req, out_dir, True, worker)
    raw_path, masks_dir = out_dir / resp["raw"], out_dir / resp["masks_dir"]
    raw = RawTrack.load(raw_path)
    if raw.num_frames != fr.num_frames or int(resp["num_frames"]) != fr.num_frames:
        raise TrackError(f"tracking produced {raw.num_frames} frames but the shot has {fr.num_frames}")
    perm = swap_permutation(swaps, raw.obj_ids, raw.num_frames)
    if not is_identity(perm):  # masks/<id>/ follow the swapped identities
        tracked = out_dir / "tmp" / "masks_tracked"
        tracked.parent.mkdir(parents=True, exist_ok=True)
        os.replace(masks_dir, tracked)
        arrange_masks(tracked, None, masks_dir, perm, raw.obj_ids, move=True)
    return _finish_tracking(raw, perm, swaps, resp, edit, out_dir, fr, p, worker, None)


def _reuse_tracking(
    prev: Path, edit: FightersEdit, out_dir: Path, fr: ShotFrames, p: TrackParams, worker: WorkerFn | None,
    swaps: Sequence[Swap],
) -> TrackRun:  # fmt: skip
    """``run_tracking`` from the tracking of a stored result (``prev``): no SAM 2, no GPU."""
    raw = RawTrack.load(prev / "raw.npz")
    if raw.num_frames != fr.num_frames or [int(i) for i in raw.obj_ids] != edit.obj_ids:
        raise TrackError(
            f"stored tracking in {prev} does not match the shot or the picks; run --from detect_track"
        )
    try:
        old = json.loads((prev / "qc.json").read_text(encoding="utf-8"))
        old_perm = swap_permutation(parse_ids(old.get("ids")).swaps, raw.obj_ids, raw.num_frames)
    except (OSError, ValueError, EditError, TrackError) as e:
        raise TrackError(
            f"cannot reuse the stored tracking in {prev}: {e}; run --from detect_track"
        ) from None
    _place(prev / "raw.npz", out_dir / "raw.npz", move=False)
    perm = swap_permutation(swaps, raw.obj_ids, raw.num_frames)
    arrange_masks(prev / "masks", old_perm, out_dir / "masks", perm, raw.obj_ids, move=False)
    resp = {"prompts": old.get("prompts"), "plan": old.get("plan"), "stats": dict(old.get("stats") or {})}
    return _finish_tracking(raw, perm, swaps, resp, edit, out_dir, fr, p, worker, prev.name)


def _finish_tracking(
    raw_tracked: RawTrack, perm: np.ndarray, swaps: Sequence[Swap], resp: dict[str, Any], edit: FightersEdit,
    out_dir: Path, fr: ShotFrames, p: TrackParams, worker: WorkerFn | None, reused: str | None,
) -> TrackRun:  # fmt: skip
    """Swaps, QC, the Tracks result, review renders and qc.json, from tracked data in ``out_dir``."""
    raw_path, masks_dir = out_dir / "raw.npz", out_dir / "masks"
    raw = apply_swaps(raw_tracked, perm)
    if raw is not raw_tracked:  # the renderer reads the swapped data
        raw_path = out_dir / "tmp" / "raw_ids.npz"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(raw_path)
    qc = compute_qc(raw, edit, p)
    tracks = build_tracks(raw, edit, fr.start, qc)
    images = [{"frame": edit.prompt_frame, "stem": str(out_dir / "prompt")}]
    images += [
        {"frame": f, "stem": str(out_dir / f"suggest_{f:05d}")}
        for f in sorted({s["frame"] for s in qc.details["suggestions"]})
    ]
    render: dict[str, Any] = {
        "mode": "render",
        **_frames_request(fr),
        "raw": str(raw_path),
        "images": images,
        "threshold": p.det_threshold,
    }
    review = None
    if p.review_video:
        ov = out_dir / "review_overlay.json"
        ov.write_text(json.dumps(review_overlay(raw, edit, qc, p.match_iou)))
        review = out_dir / "review.mp4"
        render.update(
            masks_dir=str(masks_dir),
            overlay=str(ov),
            out=str(review),
            max_width=p.review_max_width,
            ffmpeg=_ffmpeg(),
        )
    _call(render, out_dir, False, worker)
    stats = dict(resp.get("stats", {}))
    mem_trace = stats.pop("mem_trace", [])
    qc_json = {
        "summary": qc.summary,
        "prompts": resp["prompts"],
        "plan": resp.get("plan"),
        "stats": stats,
        "start": fr.start,
        "duration": fr.duration,
        "ids": {"swaps": [sw.model_dump(mode="json") for sw in swaps]},  # edits/ids.json as applied
        "reused_tracking": reused,  # the stored result whose tracking was reused (None: tracked now)
        "files": {"details": "qc_frames.json", "masks": "masks", "review": "review.mp4" if review else None},
    }
    (out_dir / "qc_frames.json").write_text(json.dumps({**qc.details, "mem_trace": mem_trace}))
    (out_dir / "qc.json").write_text(json.dumps(qc_json, indent=1))
    return TrackRun(tracks, qc_json, masks_dir, review, stats)


def _ffmpeg() -> str:
    from afv.media.ffmpeg import ffmpeg_path

    return ffmpeg_path()


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def shot_source(ctx: Ctx, params: TrackParams) -> tuple[str, float, float | None]:
    """Source video and interval: params override, else ``afv.core.project.shot_context``."""
    if params.video is not None:
        return params.video, params.start or 0.0, params.duration
    from afv.core.project import shot_context

    sc = shot_context(ctx.shot_dir)
    start = sc.start if params.start is None else params.start
    duration = sc.duration if params.duration is None else params.duration
    return sc.source_path, float(start), duration


def check_frame_count(
    video: Path | str, start: float, duration: float | None, n: int, *, overridden: bool = False
) -> None:
    """The prepared frames must be GEM-X's timeline: ``n`` == ``afv.media.normalise.shot_frame_count``
    (GEM-X's own count for this interval, computed fresh)."""
    from afv.media.normalise import shot_frame_count

    want = shot_frame_count(video, start, duration)
    if want != n:
        what = "the detect_track params' interval" if overridden else "the shot"
        dur = "to the end" if duration is None else f"+{duration:.3f} s"
        raise TrackError(
            f"detect_track: prepared {n} frames for {what} ({start:.3f} s {dur}) but GEM-X's timeline has "
            f"{want}; delete <shot>/.cache/track and run again"
        )


def fighters_data_from_ctx(ctx: Ctx) -> Mapping[str, Any] | Path | None:
    for key in (FIGHTERS_EDIT, f"{FIGHTERS_EDIT}.json"):
        if ctx.edits.get(key) is not None:
            return ctx.edits[key]
    path = ctx.shot_dir / "edits" / f"{FIGHTERS_EDIT}.json"
    return path if path.is_file() else None


@register("rfdetr_sam2")
class DetectTrackStage:
    """RF-DETR prompt + SAM 2.1 masklets -> ``Tracks`` on the 30 Hz shot timeline."""

    name: ClassVar[str] = "detect_track"
    version: ClassVar[str] = "2"
    env: ClassVar[str] = "track"
    # No upstream input: ingest always runs first anyway, and keying on its full output would re-run
    # tracking for proxy/mask changes. The frame count is checked against GEM-X's count instead.
    inputs: ClassVar[tuple[str, ...]] = ()
    edits: ClassVar[tuple[str, ...]] = (FIGHTERS_EDIT, IDS_EDIT)
    licences: ClassVar[tuple[LicenceRecord, ...]] = LICENCES
    Params: ClassVar[type[BaseModel]] = TrackParams
    weight_hash: ClassVar[str] = hashlib.sha256(f"{SAM2_SHA256}:{RFDETR_M_SHA256}".encode()).hexdigest()
    worker: WorkerFn | None = None  # tests inject a fake; default: envs_worker(ctx.progress)

    def run(self, ctx: Ctx, inputs: dict[str, Result], params: BaseModel) -> Tracks:
        p = params if isinstance(params, TrackParams) else TrackParams.model_validate(params.model_dump())
        video, start, duration = shot_source(ctx, p)
        data = fighters_data_from_ctx(ctx)
        if data is not None and p.video is None:
            from afv.core.project import shot_context

            sc = shot_context(ctx.shot_dir)
            check_shot(
                data, start, float(duration if duration is not None else sc.duration), sc.source_sha256
            )
        edit = load_fighters_edit(data) if data is not None else None
        try:
            swaps = parse_ids(ctx.edits.get(IDS_EDIT)).swaps
        except EditError as e:
            raise TrackError(str(e)) from None
        worker = self.worker or envs_worker(ctx.progress)
        ctx.progress(-1.0, "preparing 30 Hz frames")
        fr = prepare_frames(video, start, duration, ctx.shot_dir / CACHE_SUBDIR)
        check_frame_count(video, start, duration, fr.num_frames, overridden=p.video is not None
                          or p.start is not None or p.duration is not None)  # fmt: skip
        if edit is None:
            pr = detect_prompt(
                video,
                ctx.work_dir,
                [p.prompt_frame],
                threshold=p.det_threshold,
                worker=worker,
                shot_frames=fr,
            )[0]
            for ext in (".jpg", ".json"):
                os.replace(ctx.work_dir / f"prompt_{pr.frame:05d}{ext}", ctx.work_dir / f"prompt{ext}")
            info = json.loads((ctx.work_dir / "prompt.json").read_text())
            (ctx.work_dir / "prompt.json").write_text(json.dumps({**info, "image": "prompt.jpg"}, indent=1))
            ctx.progress(1.0, f"{len(pr.boxes)} people on frame {pr.frame}: pick the fighters (prompt.jpg)")
            return unpicked_tracks(fr.num_frames, fr.start)
        reuse = None
        if p.video is None:
            from afv.core.codehash import snapshot_code

            code = snapshot_code(DetectTrackStage)
            reuse = stored_tracking(
                ctx, p, data, self.version, self.weight_hash, ids=ctx.edits.get(IDS_EDIT),
                code_hash=(code or {}).get("hash"),
            )  # fmt: skip
        if reuse is not None:
            ctx.progress(-1.0, f"identity swaps only: reusing the tracking of {reuse.name} (no GPU)")
        run = run_tracking(
            video, edit, ctx.work_dir, params=p, worker=worker, shot_frames=fr, swaps=swaps, reuse=reuse
        )
        s = run.qc["summary"]
        ctx.progress(
            1.0,
            f"{'re-checked' if reuse is not None else 'tracked'} {len(edit.objects)} people"
            + (f" with {len(swaps)} identity swap{'s' if len(swaps) != 1 else ''}" if swaps else "")
            + f"; {s['flagged_frame_count']} flagged frames, {s['suggestion_count']} suggestions (qc.json)",
        )
        return run.tracks
