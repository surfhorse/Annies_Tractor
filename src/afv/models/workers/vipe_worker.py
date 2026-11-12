"""ViPE camera worker (runs in the vipe env with its conda CUDA runtime; never imports ``afv``).

``python vipe_worker.py <request.json> [<response.json>]`` (the afv.models.envs protocol; without a
response path it goes to ``response_path`` or ``response.json`` next to the request). Commands:

``vipe``  ``{video, out_npz, pipeline="pose_only_long", gravity=true, gravity_samples=12}``
    Runs ``vipe infer`` once on the shot's normalised 30 Hz video (the same file GEM-X decodes),
    reads its dense pose/intrinsics artifacts exactly as GEM-X's ViPE adapter does
    (``gem.inference.vipe``: every frame present, PINHOLE only), then optionally estimates gravity
    with ViPE's bundled GeoCalib on a few frames. Writes ``c2w`` [T,4,4] (OpenCV camera-to-world,
    ViPE world), ``intrinsics`` [T,4] (fx, fy, cx, cy) and, with gravity, ``up_cam`` [N,3],
    ``up_frames`` [N], ``up_weight`` [N] (1 / variance), ``up_focal`` [N] (GeoCalib's focal, px).

``gravity``  ``{video, out_npz, focal_px?, gravity_samples=12}``
    GeoCalib only (for the static and background camera modes). Give ``focal_px`` only when known;
    otherwise GeoCalib estimates it (reported as ``geocalib_focal_px``).

``flow``  ``{video, out_npz, boxes_npz?, max_points=300, static_std=4.0, box_margin=0.25}``
    Background feature tracks between consecutive frames, for the afv side's rotation (+ zoom)
    fit (``afv.models.camera.flow_rotations``). People are masked with their boxes (``boxes`` [T,N,4]
    in ``boxes_npz``, grown by ``box_margin`` of their size), burned-in overlays and letterbox bars
    with ``overlay_mask``: of the pixels whose grey level varies less than ``static_std`` over ~40
    sampled frames, only border bars and compact blobs (a large static region is the background of a
    locked-off camera and must stay, however much of the frame moves). Shi-Tomasi
    corners, pyramidal LK with a forward-backward check (0.5 px), up to ``max_points`` spread evenly
    over the image (``spread_subsample``). No motion model here: the afv side's RANSAC on the
    rotation + zoom model picks the background (the motion covering most of the frame).
    Writes ``p0``/``p1`` [T-1,M,2] (pixels in frame t and t+1, NaN padded), ``count`` [T-1],
    ``static_fraction`` [].

Gravity convention: ``up_cam`` is the world UP direction in OpenCV camera coordinates (x right,
y down, z forward); GeoCalib's ``Gravity.from_rp(0, 0) = (0, -1, 0)`` = up for a level camera.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def log(msg: str, frac: float = -1.0) -> None:
    """A progress line in afv.models.envs' format (``AFV_PROGRESS <0..1 or -1> <message>``)."""
    print(f"AFV_PROGRESS {frac:.2f} [vipe_worker] {msg}", flush=True)


def _vipe_executable(req: dict) -> str:
    exe = req.get("executable")
    if exe:
        return str(exe)
    local = Path(sys.executable).with_name("vipe")
    return str(local) if local.exists() else "vipe"


def _dense(path: Path, frame_count: int, item_shape: tuple[int, ...]) -> np.ndarray:
    """ViPE indexed arrays, requiring every frame of the normalised timeline (GEM-X's rule)."""
    with np.load(path) as art:
        values = np.asarray(art["data"], dtype=np.float32)
        inds = np.asarray(art["inds"])
    if values.shape != (frame_count, *item_shape) or inds.shape != (frame_count,):
        raise ValueError(f"ViPE {path.name}: shape {values.shape} for {frame_count} frames")
    order = np.argsort(inds)
    if not np.array_equal(inds[order], np.arange(frame_count)):
        raise ValueError(f"ViPE {path.name}: indices are not exactly 0..{frame_count - 1}")
    if not np.isfinite(values).all():
        raise ValueError(f"ViPE {path.name}: non-finite values")
    return values[order]


def frame_count(video: str) -> tuple[int, int, int]:
    import cv2

    cap = cv2.VideoCapture(video)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return n, w, h


def run_vipe(video: str, pipeline: str, executable: str) -> tuple[np.ndarray, np.ndarray]:
    n, _, _ = frame_count(video)
    stem = Path(video).stem
    with TemporaryDirectory(prefix="afv-vipe-") as out:
        log(f"vipe infer {Path(video).name} -p {pipeline} ({n} frames)")
        subprocess.run([executable, "infer", video, "-p", pipeline, "-o", out], check=True)
        out = Path(out)
        poses = _dense(out / "pose" / f"{stem}.npz", n, (4, 4))
        intr = _dense(out / "intrinsics" / f"{stem}.npz", n, (4,))
        cam_txt = out / "intrinsics" / f"{stem}_camera.txt"
        if cam_txt.exists():
            for line in cam_txt.read_text().splitlines():
                kind = line.split(":", 1)[-1].strip()
                if kind != "PINHOLE":
                    raise ValueError(f"ViPE camera model {kind}; only PINHOLE is supported")
    if (intr[:, :2] <= 0).any():
        raise ValueError("ViPE focal lengths must be positive")
    return poses, intr


def estimate_up(
    video: str, frames: np.ndarray, focal_px: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GeoCalib on chosen frames: up vectors (OpenCV camera frame), inverse-variance weights and
    GeoCalib's focal (px). Pass ``focal_px`` only when it is KNOWN (ViPE's solve, or the user's):
    a wrong fixed focal tilts the recovered gravity (~2 deg on telephoto shots)."""
    import cv2
    import torch
    from vipe.priors.geocalib import GeoCalib

    model = GeoCalib(weights="pinhole").cuda().eval()
    cap = cv2.VideoCapture(video)
    ups, weights, focals = [], [], []
    try:
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, bgr = cap.read()
            if not ok:
                raise OSError(f"cannot decode frame {f} of {video}")
            rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
            priors = {} if focal_px is None else {"focal": torch.tensor([float(focal_px)]).cuda()}
            with torch.no_grad():
                res = model.calibrate(rgb.cuda(), priors=priors)
            up = res["gravity"].vec3d.reshape(-1, 3)[0].double().cpu().numpy()
            # GeoCalib's *_uncertainty values are standard deviations (rad): sqrt of the LM
            # covariance diagonal. Weight = 1 / variance.
            std = [
                float(res[k].reshape(-1)[0]) for k in ("roll_uncertainty", "pitch_uncertainty") if k in res
            ]
            ups.append(up / np.linalg.norm(up))
            weights.append(1.0 / (float(np.mean(np.square(std))) + 1e-8) if std else 1.0)
            focals.append(float(res["camera"].f.reshape(-1)[-1]))
    finally:
        cap.release()
    return np.asarray(ups, np.float32), np.asarray(weights, np.float32), np.asarray(focals, np.float32)


def _sample_frames(n: int, k: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, max(1, min(k, n))).round().astype(int))


def cmd_vipe(req: dict) -> dict:
    poses, intr = run_vipe(req["video"], req.get("pipeline", "pose_only_long"), _vipe_executable(req))
    arrays = {"c2w": poses.astype(np.float32), "intrinsics": intr.astype(np.float32)}
    response = {"out_npz": req["out_npz"], "frames": len(poses), "focal_px": float(np.median(intr[:, :2]))}
    if req.get("gravity", True):
        try:
            frames = _sample_frames(len(poses), int(req.get("gravity_samples", 12)))
            ups, w, foc = estimate_up(req["video"], frames, float(np.median(intr[:, :2])))
            arrays |= {"up_cam": ups, "up_frames": frames.astype(np.int64), "up_weight": w, "up_focal": foc}
            response["gravity"] = "geocalib"
        except Exception as e:  # gravity is optional; the afv side falls back to a level camera
            log(f"GeoCalib gravity failed: {e!r}")
            response["gravity"] = f"failed: {e!r}"
    np.savez_compressed(req["out_npz"], **arrays)
    return response


def cmd_gravity(req: dict) -> dict:
    n, _, _ = frame_count(req["video"])
    frames = _sample_frames(n, int(req.get("gravity_samples", 12)))
    focal = req.get("focal_px")  # None: GeoCalib estimates the focal too
    ups, w, foc = estimate_up(req["video"], frames, focal)
    np.savez_compressed(
        req["out_npz"], up_cam=ups, up_frames=frames.astype(np.int64), up_weight=w, up_focal=foc
    )
    log(
        f"GeoCalib focal {'fixed at ' + str(focal) if focal else 'estimated'}: median {np.median(foc):.0f} px"
    )
    return {
        "out_npz": req["out_npz"],
        "frames": int(n),
        "gravity": "geocalib",
        "focal_prior_px": focal,
        "geocalib_focal_px": float(np.median(foc)),
    }


def _static_pixels(cap, n: int, static_std: float, samples: int = 40) -> np.ndarray:
    """Pixels whose grey level hardly changes over the shot (bool [H,W], not dilated)."""
    import cv2

    stack = []
    for f in np.unique(np.linspace(0, n - 1, min(samples, n)).round().astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, bgr = cap.read()
        if ok:
            stack.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32))
    return np.stack(stack).std(0) < static_std


def overlay_mask(
    static: np.ndarray,
    *,
    bar_fraction: float = 0.95,
    max_bar_width: float = 0.2,
    max_blob_fraction: float = 0.05,
    grow: int = 7,
) -> np.ndarray:
    """Static pixels to exclude from the background (bool [H,W]): letterbox/pillarbox bars (rows or
    columns static over ``bar_fraction`` of their length, contiguous from an image edge, at most
    ``max_bar_width`` of the image per side: 4:3 in 16:9 is 12.5%) and compact
    static blobs of at most ``max_blob_fraction`` of the frame each (burned-in scoreboards, logos),
    grown by ``grow`` px. Larger static regions are kept: on a locked-off camera they ARE the
    background, and masking them would leave only the moving people/crowd to fit."""
    from scipy import ndimage

    static = np.asarray(static, dtype=bool)
    h, w = static.shape
    out = np.zeros_like(static)
    cols, rows = static.mean(axis=0) >= bar_fraction, static.mean(axis=1) >= bar_fraction
    for flags, axis in ((cols, 1), (rows, 0)):
        n = len(flags)
        lead = int(np.argmin(flags)) if not flags.all() else n
        trail = int(np.argmin(flags[::-1])) if not flags.all() else n
        if lead + trail >= n:  # everything static: a locked-off, featureless or frozen shot
            continue
        lead = lead if lead <= max_bar_width * n else 0  # wider: a still part of the scene, not a bar
        trail = trail if trail <= max_bar_width * n else 0
        if axis == 1:
            out[:, :lead] = True
            out[:, w - trail :] = True
        else:
            out[:lead] = True
            out[h - trail :] = True
    labels, k = ndimage.label(static & ~out)
    if k:
        sizes = np.bincount(labels.ravel(), minlength=k + 1)
        small = sizes <= max_blob_fraction * h * w
        small[0] = False
        out |= small[labels]
    if grow > 0 and out.any():
        out = ndimage.binary_dilation(out, iterations=grow)
    return out


def spread_subsample(p: np.ndarray, k: int, shape: tuple[int, ...], rng, grid: tuple[int, int] = (9, 16)):
    """Indices of ``k`` of the points ``p`` [N,2] spread evenly over a grid of the image (round robin
    over cells, random within a cell), so densely textured regions do not outvote the rest."""
    h, w = shape[:2]
    cell = (
        np.clip(p[:, 1] * grid[0] // h, 0, grid[0] - 1) * grid[1]
        + np.clip(p[:, 0] * grid[1] // w, 0, grid[1] - 1)
    ).astype(int)
    order = rng.permutation(len(p))
    rank = np.empty(len(p))
    seen: dict[int, int] = {}
    for i in order:
        rank[i] = seen.get(cell[i], 0) + rng.uniform(0, 0.5)
        seen[cell[i]] = seen.get(cell[i], 0) + 1
    return np.sort(np.argsort(rank, kind="stable")[:k])


def background_tracks(
    video: str,
    boxes: np.ndarray | None,
    *,
    max_points: int = 300,
    static_std: float = 4.0,
    box_margin: float = 0.25,
    min_points: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Background feature correspondences between consecutive frames (see the ``flow`` command)."""
    import cv2

    n, _, _ = frame_count(video)
    cap = cv2.VideoCapture(video)
    try:
        static = overlay_mask(_static_pixels(cap, n, static_std))
        static_fraction = float(static.mean())  # fraction of the frame masked as overlay/bars
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        p0s = np.full((max(n - 1, 0), max_points, 2), np.nan, np.float32)
        p1s = np.full_like(p0s, np.nan)
        count = np.zeros(max(n - 1, 0), np.int32)
        rng = np.random.default_rng(0)
        ok, bgr = cap.read()
        prev = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if ok else None
        for t in range(n - 1):
            ok, bgr = cap.read()
            if not ok or prev is None:
                break
            cur = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            mask = (~static).astype(np.uint8) * 255
            if boxes is not None and t < len(boxes):
                for b in boxes[t]:
                    if not np.isfinite(b).all():
                        continue
                    bw, bh = b[2] - b[0], b[3] - b[1]
                    x0, x1 = int(max(0, b[0] - box_margin * bw)), int(max(0, b[2] + box_margin * bw))
                    y0, y1 = (
                        int(max(0, b[1] - 0.6 * box_margin * bh)),
                        int(max(0, b[3] + 0.6 * box_margin * bh)),
                    )
                    mask[y0:y1, x0:x1] = 0
            frame0, prev = prev, cur
            q0 = cv2.goodFeaturesToTrack(frame0, 4 * max_points, 0.01, 8, mask=mask)
            if q0 is None or len(q0) < min_points:
                continue
            q1, st, _ = cv2.calcOpticalFlowPyrLK(frame0, cur, q0, None, winSize=(21, 21), maxLevel=4)
            qb, st2, _ = cv2.calcOpticalFlowPyrLK(cur, frame0, q1, None, winSize=(21, 21), maxLevel=4)
            good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(qb - q0, axis=2)[:, 0] < 0.5)
            a, b2 = q0[good, 0], q1[good, 0]
            if len(a) < min_points:
                continue
            if len(a) > max_points:
                keep = spread_subsample(a, max_points, frame0.shape, rng)
                a, b2 = a[keep], b2[keep]
            p0s[t, : len(a)], p1s[t, : len(a)], count[t] = a, b2, len(a)
    finally:
        cap.release()
    return p0s, p1s, count, static_fraction


def cmd_flow(req: dict) -> dict:
    boxes = None
    if req.get("boxes_npz"):
        with np.load(req["boxes_npz"]) as z:
            boxes = np.asarray(z["boxes"], dtype=np.float64)
    p0, p1, count, static_fraction = background_tracks(
        req["video"],
        boxes,
        max_points=int(req.get("max_points", 300)),
        static_std=float(req.get("static_std", 4.0)),
        box_margin=float(req.get("box_margin", 0.25)),
    )
    np.savez_compressed(
        req["out_npz"], p0=p0, p1=p1, count=count, static_fraction=np.float32(static_fraction)
    )
    log(
        f"flow: {len(count)} frame pairs, "
        f"median {np.median(count) if len(count) else 0:.0f} background points"
    )
    return {
        "out_npz": req["out_npz"],
        "pairs": len(count),
        "median_points": float(np.median(count)) if len(count) else 0.0,
        "pairs_without_points": int((count == 0).sum()),
        "static_fraction": round(static_fraction, 4),
    }


COMMANDS = {"vipe": cmd_vipe, "gravity": cmd_gravity, "flow": cmd_flow}


def main(argv: list[str]) -> int:
    request_path = Path(argv[1])
    req = json.loads(request_path.read_text())
    cmd = req["command"]
    if cmd not in COMMANDS:
        raise SystemExit(f"unknown command {cmd!r}; known: {sorted(COMMANDS)}")
    response = COMMANDS[cmd](req)
    default = req.get("response_path", request_path.with_name("response.json"))
    out = Path(argv[2]) if len(argv) > 2 else Path(default)
    out.write_text(json.dumps(response, indent=2) + "\n")
    log(f"{cmd} done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
