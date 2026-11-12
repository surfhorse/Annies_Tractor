"""GEM-X worker (runs in the gemx env; never imports ``afv``).

``python gemx_worker.py <request.json> [<response.json>]`` (the afv.models.envs protocol; without a
response path it goes to ``response_path`` or ``response.json`` next to the request). Commands:

``normalize``  ``{video, start, duration|null, out_mp4}``
    GEM-X's own 30 Hz shot normalisation (``gem.inference.video.normalize_video``), so other
    workers (ViPE) see exactly GEM-X's timeline. Response: frame count, size, fps.

``background_camera``  ``{video: normalised mp4, boxes_npz, slot, intrinsics|null, out_npz}``
    GEM-X's background SIFT/homography rotation estimate with the chosen person's box masked and
    every other tracked person masked too. Writes ``R_w2c`` [T,3,3] (OpenCV, frame 0 = identity)
    and ``K`` [3,3].

``pose3d``  ``{video, start, duration|null, boxes_npz, fighters: [{role, slot}], camera: {mode:
"provided"|"static", rotations_npy?, intrinsics?, intrinsics_npy?}, device, postprocess, out_dir,
gemx_root?, config?, soma_assets?, lod?, deterministic?}``
    GEM-X motion-video for each fighter with SUPPLIED boxes (SAM 2 masklets from detect_track) and
    one shared camera. ``intrinsics_npy`` [T,3,3] gives K per frame (a zoom): GEM-X's pose features,
    CLIFF box features and translation decode all take a per-frame ``K_fullimg``; only its
    ``motion_intrinsics`` config helper wants one 3x3 K (``intrinsics``, used when no per-frame K).
    This is a library-level composition of GEM-X's public building blocks
    (``build_body_conditions`` with ``detect_track`` replaced by the supplied boxes); GEM-X has no
    API for supplied boxes in motion-video, but no GEM-X code is patched. Each model is loaded once
    and run for every fighter. Per fighter it writes ``<role>/motion.pt`` (GEM-X's own format,
    usable by ``gem-render``) and ``<role>/soma_fk.npz`` (see ``soma_worker``).

``boxes_npz`` holds ``boxes`` [T, N, 4] (x0, y0, x1, y1 source pixels, NaN where absent), N people.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")

import numpy as np

GEMX_ROOT = Path.home() / "afv/gem-x"
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def log(msg: str, frac: float = -1.0) -> None:
    """A progress line in afv.models.envs' format (``AFV_PROGRESS <0..1 or -1> <message>``)."""
    print(f"AFV_PROGRESS {frac:.2f} [gemx_worker] {msg}", flush=True)


def ensure_ffmpeg_first_on_path() -> str | None:
    """Put afv's pinned ffmpeg (``$AFV_FFMPEG_DIR`` or ``~/.local/bin``) FIRST on PATH.

    GEM-X shells out to ``ffmpeg`` by name and its 30 Hz frame count depends on the binary, so
    every normalisation (camera stage and pose3d) must use the same one. Returns the ffmpeg used.
    """
    import shutil

    local = Path(os.environ.get("AFV_FFMPEG_DIR", str(Path.home() / ".local/bin")))
    if (local / "ffmpeg").exists():
        rest = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p and Path(p) != local]
        os.environ["PATH"] = os.pathsep.join([str(local), *rest])
    return shutil.which("ffmpeg")


# --- supplied boxes ------------------------------------------------------------------------------


MIN_BOX_PX = 2.0  # boxes thinner than this after clipping to the image count as absent


def fill_gaps(sel: np.ndarray) -> np.ndarray:
    """Hold endpoints and linearly interpolate interior gaps (GEM-X's own policy).

    A row is a gap when ANY of its four values is not finite.
    """
    ok = np.isfinite(sel).all(axis=1)
    if not ok.any():
        raise ValueError("supplied track is empty on every frame")
    t = np.arange(len(sel))
    return np.stack([np.interp(t, t[ok], sel[ok, c]) for c in range(4)], axis=1).astype(np.float32)


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip [..., 4] boxes to the image; boxes left (nearly) empty become NaN."""
    b = np.array(boxes, dtype=np.float32, copy=True)
    b[..., [0, 2]] = np.clip(b[..., [0, 2]], 0, width - 1)
    b[..., [1, 3]] = np.clip(b[..., [1, 3]], 0, height - 1)
    with np.errstate(invalid="ignore"):
        empty = (b[..., 2] - b[..., 0] < MIN_BOX_PX) | (b[..., 3] - b[..., 1] < MIN_BOX_PX)
    b[empty | ~np.isfinite(b).all(-1)] = np.nan
    return b


def prepare_boxes(boxes: np.ndarray, slot: int, width: int, height: int):
    """Pure-numpy part of ``supplied_track``: (filled primary [T,4], observed [T], others, everyone).

    Boxes are clipped to the image first; partly-NaN or empty-after-clipping boxes are absent,
    then gaps are filled (held/interpolated) like GEM-X's own tracker output.
    """
    clipped = clip_boxes(boxes, width, height)
    observed = np.isfinite(clipped[:, slot]).all(axis=1)
    sel = fill_gaps(clipped[:, slot])

    def finite(b):
        return b[np.isfinite(b).all(axis=1)].astype(np.float32).reshape(-1, 4)

    n = boxes.shape[0]
    others = [finite(np.delete(clipped[t], slot, axis=0)) for t in range(n)]
    everyone = [finite(clipped[t]) for t in range(n)]
    return sel, observed, others, everyone


def supplied_track(boxes: np.ndarray, slot: int, width: int, height: int, *, bbox_enlarge: float = 1.2):
    """A GEM-X ``TrackResult`` from supplied boxes, matching ``detect_track``'s crop rules."""
    from gem.common.detectors import tracks as gem_tracks

    n = boxes.shape[0]
    sel, observed, others, everyone = prepare_boxes(boxes, slot, width, height)
    idx = np.flatnonzero(observed)
    return gem_tracks.TrackResult(
        np.arange(n),
        sel,
        gem_tracks.square_crops(gem_tracks.smooth_boxes(sel), bbox_enlarge),
        np.full(n, slot, dtype=np.int32),
        observed,
        (int(idx[0]), int(idx[-1])),
        everyone,
        others,
        everyone,
        [np.ones(len(b), np.float32) for b in everyone],
        ["supplied" if o else "supplied_interpolated" for o in observed],
    )


# --- commands ------------------------------------------------------------------------------------


def cmd_normalize(req: dict) -> dict:
    from gem.inference.video import normalize_video

    meta = normalize_video(
        req["video"], req["out_mp4"], start=req.get("start", 0.0), duration=req.get("duration")
    )
    norm = meta["normalized"]
    return {
        "out_mp4": norm["path"],
        "frame_count": norm["frame_count"],
        "width": norm["width"],
        "height": norm["height"],
        "fps": norm["fps"],
        "source": meta["source"],
        "ffmpeg": ensure_ffmpeg_first_on_path(),
    }


def cmd_background_camera(req: dict) -> dict:
    from gem.inference.body_video import motion_intrinsics
    from gem.inference.camera_motion import estimate_camera_rotations
    from gem.inference.video import VideoFrames

    boxes = np.load(req["boxes_npz"])["boxes"]
    slot = int(req["slot"])
    with VideoFrames(req["video"]) as frames:
        if len(frames) != boxes.shape[0]:
            raise ValueError(f"boxes have {boxes.shape[0]} frames, normalised video has {len(frames)}")
        track = supplied_track(boxes, slot, frames.width, frames.height)
        K = motion_intrinsics(frames.width, frames.height, req.get("intrinsics")).numpy()
        rot, diag = estimate_camera_rotations(
            frames, K, track.boxes_xyxy, return_diagnostics=True, other_person_boxes=track.other_person_boxes
        )
    np.savez_compressed(req["out_npz"], R_w2c=rot.astype(np.float32), K=K.astype(np.float32))
    return {"out_npz": req["out_npz"], "diagnostics": diag}


def _load_config(req: dict) -> dict:
    from gem.inference.body_video import load_body_video_config

    root = Path(req.get("gemx_root", GEMX_ROOT))
    config = load_body_video_config(req.get("config") or root / "weights/body_video_native77.yaml")
    cam = req.get("camera", {"mode": "static"})
    config["camera"] = {"mode": cam["mode"]}
    if cam.get("intrinsics") is not None:
        config["camera"]["intrinsics"] = cam["intrinsics"]
    if cam.get("rotations_npy") is not None:
        config["camera"]["rotations"] = cam["rotations_npy"]
    return config


def cmd_pose3d(req: dict) -> dict:
    import soma_worker
    import torch
    from gem.common.pose2d import load_native77_detector
    from gem.inference.body_video import body_release, extract_body_features, motion_intrinsics
    from gem.inference.camera_motion import camera_motion
    from gem.inference.motion import load_predictor, run_motion
    from gem.inference.serialization import save_predictions, tensor_tree
    from gem.inference.video import VideoFrames, normalize_video
    from gem.motion.api import MotionConditionSpec
    from gem.motion.frontend.joint_sets import get_observation_schema
    from gem.pose.api import load_pose_checkpoint

    if req.get("deterministic"):
        torch.manual_seed(int(req.get("seed", 0)))
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = req.get("device", "cuda")
    config = _load_config(req)
    mode = config["camera"]["mode"]
    if mode not in ("provided", "static", "background"):
        raise ValueError(
            "pose3d camera mode must be provided, static or background (ViPE runs in the camera stage)"
        )
    fighters = req["fighters"]
    boxes = np.load(req["boxes_npz"])["boxes"]
    out_dir = Path(req["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    expected = MotionConditionSpec(**config["motion"]["condition_spec"])
    schema = get_observation_schema(expected.observation_joint_set)
    release = body_release(config)
    actual = MotionConditionSpec(
        f"{release.pose_release_id}:pose_tokens", config["observations"]["release_id"], schema.name
    )
    if actual != expected:
        raise ValueError(f"GEM-X feature/detector release mismatch: {actual} != {expected}")

    tmp = out_dir / "_normalized.mp4"
    t0 = time.time()
    log("normalising the shot to 30 fps")
    metadata = normalize_video(req["video"], tmp, start=req.get("start", 0.0), duration=req.get("duration"))
    sequences: dict[str, dict] = {}
    tracks: dict[str, object] = {}
    cameras: dict[str, dict] = {}
    try:
        with VideoFrames(tmp) as frames:
            n = len(frames)
            if boxes.shape[0] != n:
                raise ValueError(f"supplied boxes have {boxes.shape[0]} frames, normalised shot has {n}")
            intrinsics = motion_intrinsics(frames.width, frames.height, config["camera"].get("intrinsics"))
            k_npy = req.get("camera", {}).get("intrinsics_npy")
            if k_npy is not None:
                k_frames = np.load(k_npy).astype(np.float32)
                if k_frames.shape != (n, 3, 3) or not np.isfinite(k_frames).all():
                    raise ValueError(f"intrinsics_npy must be finite [{n},3,3], got {k_frames.shape}")
                intrinsics = torch.from_numpy(k_frames)
            k_seq = intrinsics if intrinsics.ndim == 3 else intrinsics.expand(n, 3, 3)
            for f in fighters:
                tracks[f["role"]] = supplied_track(boxes, int(f["slot"]), frames.width, frames.height)
                log(
                    f"fighter {f['role']}: slot {f['slot']}, boxes observed on "
                    f"{tracks[f['role']].observed_primary.mean():.1%} of {n} frames"
                )

            log("extracting native77 2D observations")
            obs = config["observations"]
            detector = load_native77_detector(
                obs["checkpoint"],
                release_id=obs["release_id"],
                device=device,
                batch_size=obs.get("batch_size", 4),
            )
            keypoints = {r: detector.predict(frames, t.boxes_xys) for r, t in tracks.items()}
            detector_release = detector.release_id
            del detector

            rotations = config["camera"].get("rotations")
            rotations = None if rotations is None else np.load(rotations)
            for role, t in tracks.items():
                cameras[role] = camera_motion(
                    frames,
                    k_seq[0].numpy(),
                    t.boxes_xyxy,
                    mode=mode,
                    rotations=rotations,
                    keypoints=keypoints[role].numpy(),
                    **({"other_person_boxes": t.other_person_boxes} if mode == "background" else {}),
                )

            log("extracting frozen body Pose features")
            pose = load_pose_checkpoint(config["pose"]["checkpoint"], release, device=device)
            for role, t in tracks.items():
                features, feature_version = extract_body_features(
                    frames,
                    t.boxes_xys,
                    intrinsics,
                    pose,
                    batch_size=config["pose"].get("batch_size", 4),
                    device=device,
                )
                spec = MotionConditionSpec(feature_version, detector_release, schema.name)
                sequences[role] = {
                    "condition_spec": asdict(spec),
                    "length": torch.tensor([n]),
                    "f_imgseq": features.unsqueeze(0),
                    "kp2d_native77": keypoints[role].unsqueeze(0),
                    "bbx_xys": torch.from_numpy(t.boxes_xys).unsqueeze(0),
                    "K_fullimg": k_seq.unsqueeze(0).clone(),
                    "cam_angvel": cameras[role]["cam_angvel"].unsqueeze(0),
                    "frame_times": (torch.arange(n, dtype=torch.float64) / 30).unsqueeze(0),
                    "sampling_fps": 30.0,
                }
            del pose
    finally:
        tmp.unlink(missing_ok=True)
    timings["conditions_s"] = time.time() - t0

    log("predicting native temporal Motion")
    t0 = time.time()
    predictor = load_predictor(
        config["motion"]["checkpoint"],
        config["motion"],
        stats_path=config["motion"].get("stats"),
        device=device,
    )
    metadata["normalized"].pop("path", None)
    results = {}
    for role in tracks:
        res = run_motion(predictor, sequences[role], device=device)
        results[role] = tensor_tree(
            {
                **res,
                "format": "gem_body_video_predictions_v1",
                "video": metadata,
                "release": config,
                "conditions": sequences[role],
                "tracking": asdict(tracks[role]),
                "camera": cameras[role],
            }
        )
    del predictor
    timings["motion_s"] = time.time() - t0

    soma_assets = req.get("soma_assets", str(Path(req.get("gemx_root", GEMX_ROOT)) / "assets/soma"))
    if req.get("postprocess"):
        from gem.inference.postprocess import postprocess_motion_result

        log("applying GEM-X contact translation and CCD IK")
        t0 = time.time()
        for role in results:
            results[role] = postprocess_motion_result(results[role], soma_assets=soma_assets, device=device)
        timings["postprocess_s"] = time.time() - t0

    log("SOMA forward kinematics")
    t0 = time.time()
    body = soma_worker.build_body(soma_assets, req.get("fk_device", "cpu"), req.get("lod", "mid"))
    response: dict = {
        "fighters": {},
        "frames": n,
        "video": metadata,
        "timings": timings,
        "ffmpeg": ensure_ffmpeg_first_on_path(),
    }
    for role, res in results.items():
        d = out_dir / role
        d.mkdir(parents=True, exist_ok=True)
        save_predictions(res, d / "motion.pt")
        variant = "postprocessed" if req.get("postprocess") else "raw"
        soma_worker.convert_result(body, res, d / "soma_fk.npz", variant=variant)
        response["fighters"][role] = {
            "motion_pt": str(d / "motion.pt"),
            "soma_fk": str(d / "soma_fk.npz"),
            "observed_fraction": float(tracks[role].observed_primary.mean()),
            "camera_method": cameras[role]["diagnostics"].get("method"),
        }
    timings["fk_s"] = time.time() - t0
    if torch.cuda.is_available():
        response["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    return response


COMMANDS = {"normalize": cmd_normalize, "background_camera": cmd_background_camera, "pose3d": cmd_pose3d}


def main(argv: list[str]) -> int:
    request_path = Path(argv[1])
    req = json.loads(request_path.read_text())
    ffmpeg = ensure_ffmpeg_first_on_path()
    log(f"ffmpeg: {ffmpeg}")
    cmd = req["command"]
    if cmd not in COMMANDS:
        raise SystemExit(f"unknown command {cmd!r}; known: {sorted(COMMANDS)}")
    response = COMMANDS[cmd](req)
    default = req.get("response_path", request_path.with_name("response.json"))
    out = Path(argv[2]) if len(argv) > 2 else Path(default)
    out.write_text(json.dumps(response, indent=2, default=str) + "\n")
    log(f"{cmd} done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
