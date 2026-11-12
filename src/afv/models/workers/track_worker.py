"""Tracking worker: RF-DETR person boxes + SAM 2.1 video masklets (stage detect_track).

Runs in the track env (``~/afv/track/.venv``) through ``afv.models.envs.run_worker`` as
``python track_worker.py <request.json> <response.json>``. It never imports ``afv``; it talks to the
afv side through files only. Heavy libraries (cv2, torch, sam2, rfdetr) are imported inside
functions so the pure helpers can be unit-tested in the afv env.

The frames come from the afv side: ``frames_dir`` holds the shot's normalised 30 Hz frames
``00000.jpg ..`` written by ``afv.media.normalise_shot`` (GEM-X's exact ffmpeg command), so frame i
here is GEM-X's frame i. ``num_frames`` is checked against the folder.

Modes (request ``mode``)
------------------------
``detect``  RF-DETR on the listed ``frames``; a numbered prompt image + JSON per frame.
``track``   RF-DETR on every frame (cached next to the frames), resolve the prompts, SAM 2.1
            forwards and backwards with bounded memory, per-object mask PNGs and ``raw.npz``
            (mask boxes, areas, centroids, colour histograms, all detections). QC maths is afv's.
``render``  review video (colour masks, ids, QC tags) and numbered images for chosen frames, from
            frames, masks, ``raw.npz`` and an overlay JSON written by the afv side. CPU only.

Progress lines ``AFV_PROGRESS <0..1> <message>`` go to stdout; on failure the response is
``{"ok": false, "error": "..."}`` and the exit code is 1.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = str(Path.home() / "afv/track/weights/sam2.1_hiera_large.pt")
FPS = 30
HIST_H, HIST_S, HIST_V = 8, 4, 4  # joint hue x saturation x value bins per body half
HIST_DIM = 2 * HIST_H * HIST_S * HIST_V
DETECTOR_ID = "rfdetr-medium"  # part of the detection cache key, with the rfdetr version
MASK_IOU_SCALE = 8  # masks are compared pairwise at 1/8 resolution
COLOURS_BGR = [(245, 135, 66), (66, 90, 245), (120, 200, 80), (40, 200, 230), (220, 80, 200), (220, 220, 40)]


def log(msg: str) -> None:
    print(f"[track_worker] {msg}", file=sys.stderr, flush=True)


def progress(frac: float, msg: str) -> None:
    print(f"AFV_PROGRESS {frac:.3f} {msg}", flush=True)


# ---------------------------------------------------------------------------
# Pure helpers (no cv2/torch; unit-tested from the afv env)
# ---------------------------------------------------------------------------


def sort_by_area(boxes: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order detections largest first (the numbering shown on prompt images)."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    scores = np.asarray(scores, np.float32).reshape(-1)
    order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), kind="stable")
    return boxes[order], scores[order]


def resolve_prompts(prompts: list[dict], dets_for_frame) -> list[dict]:
    """Give every prompt a box: an explicit ``box`` wins, else ``detection`` indexes the frame's
    area-sorted RF-DETR list. ``points`` ([[x, y, label], ...], label 1 = on the person, 0 = not)
    refine a box; they are not accepted alone, because every prompt starts a fresh conditioning
    frame and a single click often gives a partial mask."""
    out = []
    for p in prompts:
        q = {"frame": int(p["frame"]), "obj": int(p["obj"])}
        box = p.get("box")
        if box is None and p.get("detection") is not None:
            boxes, _ = dets_for_frame(q["frame"])
            k = int(p["detection"])
            if not 0 <= k < len(boxes):
                raise ValueError(
                    f"prompt for object {q['obj']} at frame {q['frame']}: detection {k} does not exist "
                    f"({len(boxes)} people detected on that frame)"
                )
            box = boxes[k].tolist()
        if box is None:
            raise ValueError(f"prompt for object {q['obj']} at frame {q['frame']} needs a box or a detection")
        q["box"] = [float(v) for v in box]
        q["points"] = p.get("points") or None
        out.append(q)
    return out


def plan_passes(prompts: list[dict], num_frames: int) -> dict:
    """Propagation plan for prompts on several frames.

    SAM 2 tracks every object independently and conditions on all of that object's prompt frames.
    One forward pass from the earliest prompt frame covers every frame >= each object's first
    prompt; one reverse pass from the latest *first* prompt frame covers the frames before each
    object's first prompt. Object o's mask on frame f is taken from the forward pass when
    f >= first[o], else from the reverse pass.
    """
    if not prompts:
        raise ValueError("no prompts")
    first: dict[int, int] = {}
    for p in prompts:
        f = int(p["frame"])
        if not 0 <= f < num_frames:
            raise ValueError(f"prompt frame {f} outside 0..{num_frames - 1}")
        first[int(p["obj"])] = min(first.get(int(p["obj"]), f), f)
    return {
        "first": first,
        "forward_start": min(first.values()),
        "reverse_start": max(first.values()),
        "prompt_frames": sorted({int(p["frame"]) for p in prompts}),
    }


def take_from_pass(frame: int, first_prompt: int, reverse: bool) -> bool:
    return (frame < first_prompt) if reverse else (frame >= first_prompt)


def keep_components(areas: np.ndarray, min_part: float) -> np.ndarray:
    """Indices of mask components that count for the box: those >= ``min_part`` of the mask area,
    or the largest one when the mask is fragmented into specks only."""
    areas = np.asarray(areas)
    if len(areas) == 0:
        return np.zeros(0, np.int64)
    keep = np.flatnonzero(areas >= max(min_part * areas.sum(), 1))
    return keep if len(keep) else np.array([int(np.argmax(areas))])


def hsv_hist(hsv_pixels: np.ndarray) -> np.ndarray:
    """L1-normalised joint H x S x V histogram of [N,3] OpenCV-HSV pixels (H 0..179, S/V 0..255).
    Value bins separate black from white clothing, which hue x saturation alone cannot."""
    px = np.asarray(hsv_pixels).reshape(-1, 3).astype(np.int64)
    if len(px) == 0:
        return np.zeros(HIST_H * HIST_S * HIST_V, np.float32)
    h = np.minimum(px[:, 0] * HIST_H // 180, HIST_H - 1)
    s = px[:, 1] * HIST_S // 256
    v = px[:, 2] * HIST_V // 256
    counts = np.bincount((h * HIST_S + s) * HIST_V + v, minlength=HIST_H * HIST_S * HIST_V)
    return (counts / counts.sum()).astype(np.float32)


def pairwise_mask_iou(small: dict[int, dict[int, np.ndarray]], n: int, num_obj: int) -> np.ndarray:
    """[F,P,P] IoU of packed low-resolution masks (``small[frame][slot]``; missing = empty)."""
    out = np.zeros((n, num_obj, num_obj), np.float32)
    for f, per in small.items():
        slots = sorted(per)
        for a in slots:
            for b in slots:
                if a < b:
                    ma, mb = np.unpackbits(per[a]), np.unpackbits(per[b])
                    union = int(np.count_nonzero(ma | mb))
                    if union:
                        out[f, a, b] = out[f, b, a] = np.count_nonzero(ma & mb) / union
    return out


def prune_keys(keys, current: int, keep_frames: list[int], window: int) -> list[int]:
    """Non-conditioning frames SAM 2 can no longer use: farther than ``window`` from the current
    frame and from every prompt frame. SAM 2.1 reads at most 6 mask memories and 15 object pointers
    back along the tracking direction, so window >= 16 leaves its outputs unchanged."""
    if window <= 0:
        return []
    kf = np.asarray(keep_frames, np.int64)
    drop = []
    for t in keys:
        if abs(t - current) <= window:
            continue
        if len(kf) and np.abs(kf - t).min() <= window:
            continue
        drop.append(t)
    return drop


# ---------------------------------------------------------------------------
# Frames and RF-DETR
# ---------------------------------------------------------------------------


def frame_path(frames_dir: Path, i: int) -> Path:
    return frames_dir / f"{i:05d}.jpg"


def check_frames(req: dict) -> tuple[Path, int]:
    frames_dir = Path(req["frames_dir"])
    n = int(req["num_frames"])
    found = len(list(frames_dir.glob("*.jpg")))
    if found != n or not frame_path(frames_dir, n - 1).exists():
        raise RuntimeError(f"{frames_dir} has {found} frames, expected {n} (00000.jpg..)")
    return frames_dir, n


def read_rgb(frames_dir: Path, i: int) -> np.ndarray:
    import cv2

    return cv2.cvtColor(cv2.imread(str(frame_path(frames_dir, i))), cv2.COLOR_BGR2RGB)


class Detector:
    def __init__(self) -> None:
        from rfdetr import RFDETRMedium
        from rfdetr.utilities.class_names import COCO_CLASSES

        self.person = next(k for k, v in COCO_CLASSES.items() if v == "person")
        self.model = RFDETRMedium()

    def __call__(self, rgbs: list[np.ndarray], threshold: float) -> list[tuple[np.ndarray, np.ndarray]]:
        from PIL import Image

        dets = self.model.predict([Image.fromarray(x) for x in rgbs], threshold=threshold)
        if not isinstance(dets, list):
            dets = [dets]
        out = []
        for d in dets:
            keep = d.class_id == self.person
            out.append(sort_by_area(d.xyxy[keep], d.confidence[keep]))
        return out


def detect_all(frames_dir: Path, n: int, threshold: float, batch: int = 8) -> dict:
    """RF-DETR people on every frame, cached next to the frames per model and threshold."""
    import importlib.metadata

    model_id = f"{DETECTOR_ID}@{importlib.metadata.version('rfdetr')}"
    path = frames_dir.parent / f"detections_{DETECTOR_ID}_{threshold:.3f}.npz"
    if path.exists():
        with np.load(path) as z:
            if int(z["num_frames"]) == n and str(z["model"]) == model_id:
                log(f"detections cached: {path.name}")
                return {k: z[k] for k in ("det_frame", "det_boxes", "det_scores")} | {"seconds": 0.0}
    import torch

    t0 = time.time()
    det = Detector()
    fr, bx, sc = [], [], []
    for s in range(0, n, batch):
        idx = list(range(s, min(n, s + batch)))
        for i, (b, c) in zip(idx, det([read_rgb(frames_dir, i) for i in idx], threshold), strict=True):
            fr.append(np.full(len(b), i, np.int64))
            bx.append(b)
            sc.append(c)
        progress(0.25 * idx[-1] / n, f"RF-DETR {idx[-1] + 1}/{n}")
    res = {
        "det_frame": np.concatenate(fr),
        "det_boxes": np.concatenate(bx).reshape(-1, 4),
        "det_scores": np.concatenate(sc),
    }
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, num_frames=n, model=np.array(model_id), **res)
    os.replace(tmp, path)
    del det
    torch.cuda.empty_cache()
    res["seconds"] = round(time.time() - t0, 2)
    log(f"RF-DETR on {n} frames in {res['seconds']} s")
    return res


def draw_prompt(bgr: np.ndarray, boxes: np.ndarray, scores: np.ndarray, frame: int) -> np.ndarray:
    import cv2

    img = bgr.copy()
    h = img.shape[0]
    fs = max(0.6, h / 1000)
    for i, (b, s) in enumerate(zip(boxes, scores, strict=True)):
        c = COLOURS_BGR[i % len(COLOURS_BGR)]
        org = (int(b[0]) + 4, int(b[1]) + int(34 * fs))
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 3)
        cv2.putText(img, f"{i} ({s:.2f})", org, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * fs, (0, 0, 0), 6)
        cv2.putText(img, f"{i} ({s:.2f})", org, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * fs, c, 3)
    cv2.putText(img, f"frame {frame}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)
    return img


def write_prompt(
    frames_dir: Path, f: int, boxes: np.ndarray, scores: np.ndarray, stem: Path, threshold: float
) -> dict:
    """``<stem>.jpg`` (numbered people) and ``<stem>.json`` ({frame, prompt_frame, boxes, scores})."""
    import cv2

    img = cv2.imread(str(frame_path(frames_dir, f)))
    cv2.imwrite(str(stem.with_suffix(".jpg")), draw_prompt(img, boxes, scores, f))
    info = {
        "frame": f,
        "prompt_frame": f,
        "image": stem.with_suffix(".jpg").name,  # relative to the folder of this JSON
        "boxes": np.round(boxes, 1).tolist(),
        "scores": np.round(scores, 3).tolist(),
        "threshold": threshold,
    }
    stem.with_suffix(".json").write_text(json.dumps(info, indent=1))
    return info


def mode_detect(req: dict) -> dict:
    frames_dir, n = check_frames(req)
    out = Path(req["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    frames = [int(f) for f in req.get("frames", [0])]
    for f in frames:
        if not 0 <= f < n:
            raise ValueError(f"frame {f} outside 0..{n - 1}")
    det = Detector()
    thr = float(req.get("threshold", 0.4))
    prompts = []
    for f in frames:
        boxes, scores = det([read_rgb(frames_dir, f)], thr)[0]
        prompts.append(write_prompt(frames_dir, f, boxes, scores, out / f"prompt_{f:05d}", thr))
    return {"num_frames": n, "prompts": prompts}


# ---------------------------------------------------------------------------
# SAM 2.1
# ---------------------------------------------------------------------------


class LazyFrames:
    """Frames loaded on demand. SAM 2's own loader holds every frame as float32 at 1024^2
    (12.6 MB each: 3.4 GB for 9 s, 22 GB for a 60 s shot). Same resize/normalisation as
    ``sam2.utils.misc``."""

    def __init__(self, frames_dir: Path, n: int, image_size: int, keep: int = 4) -> None:
        import torch

        self.paths = [frame_path(frames_dir, i) for i in range(n)]
        self.image_size = image_size
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
        self.cache: OrderedDict[int, Any] = OrderedDict()
        self.keep = keep

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from sam2.utils.misc import _load_img_as_tensor

        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        img, _, _ = _load_img_as_tensor(str(self.paths[i]), self.image_size)
        img = (img.float() - self.mean) / self.std
        self.cache[i] = img
        while len(self.cache) > self.keep:
            self.cache.popitem(last=False)
        return img


def mask_features(mask: np.ndarray, hsv_small: np.ndarray, min_part: float) -> dict:
    """Box (ignoring specks < min_part of the mask area), area, centroid and a two-half HSV histogram."""
    import cv2

    area = int(mask.sum())
    feats = {
        "box": np.full(4, np.nan, np.float32),
        "area": 0.0,
        "centroid": np.full(2, np.nan, np.float32),
        "hist": np.zeros(HIST_DIM, np.float32),
    }
    if area == 0:
        return feats
    _, _, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = keep_components(stats[1:, cv2.CC_STAT_AREA], min_part) + 1
    x0 = stats[keep, cv2.CC_STAT_LEFT]
    y0 = stats[keep, cv2.CC_STAT_TOP]
    x1 = x0 + stats[keep, cv2.CC_STAT_WIDTH]
    y1 = y0 + stats[keep, cv2.CC_STAT_HEIGHT]
    box = np.array([x0.min(), y0.min(), x1.max(), y1.max()], np.float32)
    w = stats[keep, cv2.CC_STAT_AREA].astype(np.float64)
    cent = (cents[keep] * w[:, None]).sum(0) / w.sum()
    feats.update(box=box, area=float(area), centroid=cent.astype(np.float32))
    # histogram on the half-resolution frame, top and bottom half of the box separately
    sh, sw = hsv_small.shape[:2]
    small = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
    mid = int(np.rint((box[1] + box[3]) / 2 * sh / mask.shape[0]))
    sel = small > 0
    hist = [hsv_hist(hsv_small[:mid][sel[:mid]]), hsv_hist(hsv_small[mid:][sel[mid:]])]
    feats["hist"] = np.concatenate(hist).astype(np.float32)
    return feats


def mode_track(req: dict) -> dict:
    import cv2
    import sam2.sam2_video_predictor as svp
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    t_all = time.time()
    frames_dir, n = check_frames(req)
    out = Path(req["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    h, w = cv2.imread(str(frame_path(frames_dir, 0))).shape[:2]
    dets = detect_all(frames_dir, n, float(req.get("det_threshold", 0.4)))
    det_frame = dets["det_frame"]
    starts = np.searchsorted(det_frame, np.arange(n + 1))

    def dets_for_frame(f: int):
        s, e = starts[f], starts[f + 1]
        return dets["det_boxes"][s:e], dets["det_scores"][s:e]

    prompts = resolve_prompts(req["prompts"], dets_for_frame)
    obj_ids = [int(o) for o in req["obj_ids"]]
    unknown = {p["obj"] for p in prompts} - set(obj_ids)
    if unknown:
        raise ValueError(f"prompts reference unknown objects {sorted(unknown)}")
    missing = set(obj_ids) - {p["obj"] for p in prompts}
    if missing:
        raise ValueError(f"objects without a prompt: {sorted(missing)}")
    plan = plan_passes(prompts, n)
    slot = {o: k for k, o in enumerate(obj_ids)}
    window = int(req.get("prune_window", 20))
    min_part = float(req.get("min_mask_part", 0.02))

    masks_dir = out / "masks"
    for o in obj_ids:
        (masks_dir / str(o)).mkdir(parents=True, exist_ok=True)
    P = len(obj_ids)
    boxes = np.full((n, P, 4), np.nan, np.float32)
    area = np.zeros((n, P), np.float32)
    cent = np.full((n, P, 2), np.nan, np.float32)
    hist = np.zeros((n, P, HIST_DIM), np.float32)
    done = np.zeros((n, P), bool)
    small_masks: dict[int, dict[int, np.ndarray]] = {}  # frame -> slot -> packed 1/8-res mask
    sw, sh = max(1, w // MASK_IOU_SCALE), max(1, h // MASK_IOU_SCALE)
    mem_trace = []
    total_steps = (n - plan["forward_start"]) + (
        plan["reverse_start"] + 1 if plan["reverse_start"] > 0 else 0
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    progress(0.25, "SAM 2.1 loading")
    predictor = build_sam2_video_predictor(SAM2_CFG, req.get("sam2_ckpt", SAM2_CKPT), device="cuda")
    model_mb = torch.cuda.memory_allocated() / 2**20
    svp.load_video_frames = lambda **kw: (LazyFrames(frames_dir, n, kw["image_size"]), h, w)
    steps = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(
            video_path=str(frames_dir), offload_state_to_cpu=bool(req.get("offload_state", False))
        )
        for p in sorted(prompts, key=lambda q: (q["frame"], q["obj"])):
            kw: dict[str, Any] = {}
            if p["box"] is not None:
                kw["box"] = np.asarray(p["box"], np.float32)
            if p["points"]:
                pts = np.asarray(p["points"], np.float32).reshape(-1, 3)
                kw["points"], kw["labels"] = pts[:, :2], pts[:, 2].astype(np.int32)
            predictor.add_new_points_or_box(state, frame_idx=p["frame"], obj_id=p["obj"], **kw)
        for reverse, start in ((False, plan["forward_start"]), (True, plan["reverse_start"])):
            if reverse and start == 0:
                continue
            for fi, ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=start, reverse=reverse
            ):
                steps += 1
                want = [
                    (k, int(o)) for k, o in enumerate(ids)
                    if take_from_pass(fi, plan["first"][int(o)], reverse) and not done[fi, slot[int(o)]]
                ]  # fmt: skip
                if want:
                    bgr = cv2.imread(str(frame_path(frames_dir, fi)))
                    small = cv2.resize(bgr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                    for k, o in want:
                        msk = (logits[k, 0] > 0).cpu().numpy()
                        s = slot[o]
                        f = mask_features(msk, hsv, min_part)
                        boxes[fi, s], area[fi, s], cent[fi, s], hist[fi, s] = (
                            f["box"], f["area"], f["centroid"], f["hist"])  # fmt: skip
                        done[fi, s] = True
                        if f["area"] > 0:
                            m8 = msk.astype(np.uint8) * 255
                            cv2.imwrite(str(masks_dir / str(o) / f"{fi:05d}.png"), m8)
                            low = cv2.resize(m8, (sw, sh), interpolation=cv2.INTER_AREA) > 127
                            small_masks.setdefault(fi, {})[s] = np.packbits(low)
                for od in state["output_dict_per_obj"].values():
                    nc = od["non_cond_frame_outputs"]
                    for t in prune_keys(list(nc), fi, plan["prompt_frames"], window):
                        del nc[t]
                if steps % 30 == 0:
                    mem_trace.append([
                        steps, round(torch.cuda.memory_allocated() / 2**20, 1),
                        round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                    ])  # fmt: skip
                    progress(0.25 + 0.75 * steps / total_steps, f"SAM 2.1 {steps}/{total_steps}")
        predictor.reset_state(state)
    sam_s = time.time() - t0
    if not done.all():
        raise RuntimeError(f"{int((~done).sum())} object-frames were not produced by propagation")
    np.savez_compressed(
        out / "raw.npz", obj_ids=np.asarray(obj_ids, np.int64), boxes=boxes, area=area, centroid=cent,
        hist=hist, det_frame=det_frame, det_boxes=dets["det_boxes"], det_scores=dets["det_scores"],
        width=np.int64(w), height=np.int64(h), mask_iou=pairwise_mask_iou(small_masks, n, P),
    )  # fmt: skip
    stats = {
        "sam2_seconds": round(sam_s, 2),
        "sam2_frames_processed": steps,
        "sam2_fps": round(steps / max(sam_s, 1e-6), 2),
        "detect_seconds": dets["seconds"],
        "total_seconds": round(time.time() - t_all, 2),
        "vram_model_mb": round(model_mb, 1),
        "vram_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "vram_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "rss_peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "prune_window": window,
        "offload_state": bool(req.get("offload_state", False)),
        "mem_trace": mem_trace,  # [propagation steps, allocated MB, peak RSS MB]
    }
    log(json.dumps({k: v for k, v in stats.items() if k != "mem_trace"}))
    return {
        "num_frames": n, "width": w, "height": h, "raw": "raw.npz", "masks_dir": "masks",
        "prompts": prompts, "stats": stats,
        "plan": {"forward_start": plan["forward_start"], "reverse_start": plan["reverse_start"],
                 "first": {str(k): v for k, v in plan["first"].items()}},
    }  # fmt: skip


# ---------------------------------------------------------------------------
# Review video and numbered images
# ---------------------------------------------------------------------------


def mode_render(req: dict) -> dict:
    """Overlay JSON (written by afv): {obj_ids, tags: [F][P] str, notes: [F] str,
    untracked: [F] [[x0,y0,x1,y1], ...], prompt_frames: {obj: [frames]}}. ``images``:
    [{"frame": f, "stem": path without suffix}] numbered-people images from the cached detections."""
    import cv2

    t0 = time.time()
    frames_dir, n = check_frames(req)
    raw = np.load(req["raw"])
    det_frame = raw["det_frame"]
    images = []
    for it in req.get("images", []):
        f = int(it["frame"])
        s, e = np.searchsorted(det_frame, [f, f + 1])
        images.append(
            write_prompt(
                frames_dir,
                f,
                raw["det_boxes"][s:e],
                raw["det_scores"][s:e],
                Path(it["stem"]),
                float(req.get("threshold", 0.4)),
            )
        )
    if not req.get("out"):
        return {"images": images}
    masks_dir = Path(req["masks_dir"])
    ov = json.loads(Path(req["overlay"]).read_text())
    boxes = raw["boxes"]
    obj_ids = [int(o) for o in ov["obj_ids"]]
    h, w = cv2.imread(str(frame_path(frames_dir, 0))).shape[:2]
    scale = min(1.0, float(req.get("max_width", 1280)) / w)
    ow, oh = round(w * scale / 2) * 2, round(h * scale / 2) * 2
    cmd = [
        req.get("ffmpeg", "ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", req["out"],
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    fs = max(0.6, h / 1000)
    font = cv2.FONT_HERSHEY_SIMPLEX
    prompt_frames = {int(k): set(v) for k, v in ov.get("prompt_frames", {}).items()}
    for fi in range(n):
        img = cv2.imread(str(frame_path(frames_dir, fi)))
        for k, o in enumerate(obj_ids):
            mp = masks_dir / str(o) / f"{fi:05d}.png"
            if mp.exists():
                msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 0
                c = np.array(COLOURS_BGR[k % len(COLOURS_BGR)], np.float32)
                img[msk] = (0.45 * img[msk] + 0.55 * c).astype(np.uint8)
        for b in ov["untracked"][fi]:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 2)
            cv2.putText(img, "untracked", (int(b[0]) + 4, int(b[3]) - 8), font, 0.8 * fs, (255, 255, 255), 2)
        for k, o in enumerate(obj_ids):
            b = boxes[fi, k]
            if not np.isfinite(b[0]):
                continue
            c = COLOURS_BGR[k % len(COLOURS_BGR)]
            tag = ov["tags"][fi][k] + (" [PROMPT]" if fi in prompt_frames.get(o, ()) else "")
            flagged = "!" in tag
            cv2.rectangle(
                img,
                (int(b[0]), int(b[1])),
                (int(b[2]), int(b[3])),
                (0, 0, 255) if flagged else c,
                4 if flagged else 2,
            )
            org = (int(b[0]) + 4, int(b[1]) + int(30 * fs))
            cv2.putText(img, tag, org, font, 0.9 * fs, (0, 0, 0), 6)
            cv2.putText(img, tag, org, font, 0.9 * fs, c, 2)
        note = f"frame {fi}  {ov['notes'][fi]}"
        cv2.putText(img, note, (10, h - 14), font, fs, (0, 0, 0), 6)
        cv2.putText(img, note, (10, h - 14), font, fs, (255, 255, 255), 2)
        if scale != 1.0:
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
        proc.stdin.write(np.ascontiguousarray(img).tobytes())
        if fi % 60 == 0:
            progress(fi / n, f"review video {fi}/{n}")
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError("ffmpeg failed writing the review video")
    return {"video": req["out"], "images": images, "frames": n, "seconds": round(time.time() - t0, 2)}


MODES = {"detect": mode_detect, "track": mode_track, "render": mode_render}


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print("usage: track_worker.py <request.json> [<response.json>]", file=sys.stderr)
        return 2
    req_path = Path(argv[1])
    resp_path = Path(argv[2]) if len(argv) == 3 else req_path.with_name("response.json")
    req = json.loads(req_path.read_text())
    try:
        resp = {"ok": True, **MODES[req["mode"]](req)}
        code = 0
    except Exception as e:  # every failure goes back to the afv side as a readable message
        import traceback

        traceback.print_exc()
        resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        code = 1
    resp_path.write_text(json.dumps(resp))
    return code


if __name__ == "__main__":
    os.environ.setdefault("TQDM_DISABLE", "1")
    sys.exit(main(sys.argv))
