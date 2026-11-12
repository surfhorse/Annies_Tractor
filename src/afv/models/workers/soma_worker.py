"""SOMA forward kinematics worker (runs in the gemx env; never imports ``afv``).

Turns GEM-X SOMA parameters into what the afv side needs to build an ``ActorMotion`` on the
``soma77`` skeleton, without the afv side ever importing torch or soma:

* per-frame world transforms of the 77 public SOMA joints (SOMA's own FK, procedural twist on),
  for the camera-frame (``incam``) and GEM-X world (``global``) parameter sets;
* the fighter's T-pose (zero pose) joint positions and joint orientations for the shot-averaged
  identity and bone scales (= locked bone lengths) and the lowest T-pose mesh vertex (sole height);
* the GEM-X side data needed later (contact logits, ViTPose77 2D, intrinsics, camera rotations).

Commands (``python soma_worker.py <request.json> [<response.json>]``, the afv.models.envs protocol;
without a response path it goes to ``response_path`` or ``response.json`` next to the request):

``{"command": "skeleton", "out_json": path}``
    Neutral SOMA skeleton (identity 0, unit scales) in afv's skeleton JSON format.
``{"command": "convert", "motion_pt": path, "out_npz": path, "variant": "raw"|"postprocessed"}``
    Convert a GEM-X motion.pt (P0 outputs or ``gemx_worker`` results) to the FK NPZ below.

Common keys: ``soma_assets`` (default ``~/afv/gem-x/assets/soma``), ``device`` ("cpu"),
``lod`` ("mid": SOMA's default and GEM-X's renderer; GEM-X's contact/IK postprocess uses "xlo";
joints differ by <= 2.5 mm, fingers only).

FK NPZ (all float32 unless noted; T frames, J = 77 public joints, parents in SOMA order):
``incam_rot``/``global_rot`` [T,J,3,3] and ``incam_pos``/``global_pos`` [T,J,3]: joint world
rotation/position (incam = OpenCV camera metres; global = GEM-X Y-up world metres).
``tpose_pos`` [J,3], ``tpose_rot`` [J,3,3]: zero-pose joints with hips at the origin (SOMA world:
Y up, +Z forward, character's left +X). ``sole_y`` []: lowest T-pose mesh vertex height.
``contact_logits`` [T,6] (order ``CONTACT_ORDER``), ``kp2d`` [T,J,3] (x, y, conf; normalised-video
pixels), ``K`` [T,3,3], ``R_w2c`` [T,3,3], ``bbx_xys`` [T,3], ``boxes_xyxy`` [T,4], ``observed``
[T] bool, ``frame_times`` [T] float64, ``identity`` [128] / ``scale`` [57] (shot means), raw
params ``incam_global_orient``/``global_global_orient``/``incam_transl``/``global_transl`` [T,3]
and ``body_pose`` [T,228]; ``meta`` (JSON string): joint names, parents, versions, variant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SOMA_ASSETS = Path.home() / "afv/gem-x/assets/soma"
# GEM-X's static_conf_logits order (gem.motion.postprocess.api) and the public joint each refers
# to (gem.motion.postprocess.contact.SOMA_CONTACT_JOINT_IDS = 69, 70, 74, 75, 14, 42).
CONTACT_ORDER = ("left_ankle", "left_foot", "right_ankle", "right_foot", "left_wrist", "right_wrist")
CONTACT_JOINTS = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase", "LeftHand", "RightHand")


def build_body(soma_assets: str | Path = SOMA_ASSETS, device: str = "cpu", lod: str = "mid"):
    """GEM-X's fixed-GS1 SOMA adapter with the deployed Motion conventions (reference 0.2.0)."""
    from gem.common.soma import SomaFixedGs1BodyAdapter, SomaRuntimeConfig

    return SomaFixedGs1BodyAdapter(
        SomaRuntimeConfig(
            data_root=str(soma_assets),
            device=device,
            lod=lod,
            mode="warp",
            apply_correctives=False,
            enable_procedural_transforms=True,
            procedural_reference="runtime",
        )
    )


def _np(x):
    return x.detach().cpu().numpy()


def joint_info(body) -> tuple[list[str], list[int]]:
    """Public joint names and parents without the virtual Root (Hips = index 0, parent -1)."""
    rt = body.runtime
    names = [str(n) for n in rt.public_joint_names][1:]
    parents = [int(p) - 1 for p in rt.output_joint_parent_ids.cpu().tolist()][1:]
    return names, parents


def tpose(body, identity, scale):
    """Zero-pose joints, joint orientations and lowest mesh vertex for one identity (hips at 0)."""
    import torch

    zero = {
        "global_orient": torch.zeros(1, 1, 3),
        "body_pose": torch.zeros(1, 1, 228),
        "identity_coeffs": torch.as_tensor(identity, dtype=torch.float32).reshape(1, 1, -1),
        "scale_params": torch.as_tensor(scale, dtype=torch.float32).reshape(1, 1, -1),
        "transl": torch.zeros(1, 1, 3),
    }
    with torch.inference_mode():
        out = body(**zero, return_joints_only=False, apply_correctives=False)
    tf = _np(out["transforms"])[0, 0]  # [78,4,4] incl. virtual Root
    verts = _np(out["vertices"])[0, 0]
    return tf[1:, :3, 3], tf[1:, :3, :3], float(verts[:, 1].min())


def fk(body, params: dict):
    """SOMA FK of one [1,T] parameter set; identity/scale are averaged over T by the adapter."""
    import torch

    with torch.inference_mode():
        out = body(**{k: v.float() for k, v in params.items()}, return_joints_only=True)
    tf = _np(out["transforms"])[0]  # [T,78,4,4]
    joints = _np(out["joints"])[0]
    if np.abs(tf[:, 1:, :3, 3] - joints).max() > 1e-5:
        raise RuntimeError("SOMA transforms and joints disagree")
    return tf[:, 1:, :3, :3], tf[:, 1:, :3, 3]


def _sequence_means(params: dict) -> tuple[np.ndarray, np.ndarray]:
    identity = _np(params["identity_coeffs"])[0]
    scale = _np(params["scale_params"])[0]
    if identity.ndim == 2:
        identity, scale = identity.mean(0), scale.mean(0)
    scale = scale.copy()
    scale[0] = 1.0  # exact GS1 global scale (the adapter requires exactly one)
    return identity.astype(np.float32), scale.astype(np.float32)


def select_variant(result: dict, variant: str) -> dict:
    """GEM-X's raw vs contact/IK-postprocessed parameters (gem.inference.postprocess)."""
    if variant == "postprocessed":
        if not result.get("postprocessing"):
            raise ValueError("motion.pt has no postprocessed prediction")
        return result
    if variant != "raw":
        raise ValueError("variant must be raw or postprocessed")
    if result.get("postprocessing") and "raw_predictions" in result:
        return {**result, **result["raw_predictions"]}
    return result


def convert_result(body, result: dict, out_npz: str | Path, *, variant: str = "raw") -> dict:
    """Write the FK NPZ for one GEM-X body-video result (a loaded motion.pt dict)."""
    sel = select_variant(result, variant)
    p_cam, p_glob = sel["pred_soma_params_incam"], sel["pred_soma_params_global"]
    names, parents = joint_info(body)
    identity, scale = _sequence_means(p_glob)
    # Both parameter sets share identity/scale; FK uses the shot mean (= locked bone lengths).
    incam_rot, incam_pos = fk(body, p_cam)
    global_rot, global_pos = fk(body, p_glob)
    tp_pos, tp_rot, sole_y = tpose(body, identity, scale)
    cond = result["conditions"]
    tracking = result.get("tracking", {})
    n = incam_pos.shape[0]
    observed = np.asarray(tracking.get("observed_primary", np.ones(n, bool)), dtype=bool)
    arrays = {
        "incam_rot": incam_rot.astype(np.float32),
        "incam_pos": incam_pos.astype(np.float32),
        "global_rot": global_rot.astype(np.float32),
        "global_pos": global_pos.astype(np.float32),
        "tpose_pos": tp_pos.astype(np.float32),
        "tpose_rot": tp_rot.astype(np.float32),
        "sole_y": np.float32(sole_y),
        "contact_logits": _np(sel["static_conf_logits"])[0].astype(np.float32),
        "kp2d": _np(cond["kp2d_native77"])[0].astype(np.float32),
        "K": _np(cond["K_fullimg"])[0].astype(np.float32),
        "R_w2c": _np(result["camera"]["R_w2c"]).astype(np.float32),
        "bbx_xys": _np(cond["bbx_xys"])[0].astype(np.float32),
        "boxes_xyxy": np.asarray(tracking.get("boxes_xyxy", np.full((n, 4), np.nan)), np.float32),
        "observed": observed,
        "frame_times": _np(cond["frame_times"])[0].astype(np.float64),
        "identity": identity,
        "scale": scale,
        "incam_global_orient": _np(p_cam["global_orient"])[0].astype(np.float32),
        "global_global_orient": _np(p_glob["global_orient"])[0].astype(np.float32),
        "incam_transl": _np(p_cam["transl"])[0].astype(np.float32),
        "global_transl": _np(p_glob["transl"])[0].astype(np.float32),
        "body_pose": _np(p_glob["body_pose"])[0].astype(np.float32),
    }
    video = result.get("video", {})
    meta = {
        "format": "afv_soma_fk_v1",
        "joint_names": names,
        "parents": parents,
        "contact_order": list(CONTACT_ORDER),
        "contact_joints": list(CONTACT_JOINTS),
        "variant": variant,
        "lod": body.runtime_config.lod,
        "soma_version": _soma_version(),
        "incam": "OpenCV camera metres (x right, y down, z forward)",
        "global": "GEM-X world rollout metres (Y up), origin at the first frame's hips",
        "camera_method": result.get("camera", {}).get("diagnostics", {}).get("method"),
        "normalized_video": {k: v for k, v in video.get("normalized", {}).items() if k != "path"},
        "source_video": video.get("source", {}),
    }
    arrays["meta"] = np.array(json.dumps(meta, default=str))
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    return {"out_npz": str(out_npz), "frames": int(n), "variant": variant}


def _soma_version() -> str:
    try:
        import soma

        return str(soma.__version__)
    except Exception:  # pragma: no cover - soma always present in the gemx env
        return "unknown"


def neutral_skeleton(body) -> dict:
    """afv skeleton JSON for the neutral SOMA identity (identity 0, unit bone scales).

    Rest pose = SOMA's zero pose (T-pose, Y up, faces +Z, left = +X, reference 0.2.0), with
    identity rest rotations: offsets are world-aligned T-pose bone vectors, and a local rotation is
    the joint's rotation relative to the T-pose. SOMA's own joint frames (bone-aligned axes) are
    kept as ``soma_joint_orient`` for tools that need them (world = local_rel * orient).
    """
    from scipy.spatial.transform import Rotation

    names, parents = joint_info(body)
    scale = np.ones(57, np.float32)
    pos, rot, sole_y = tpose(body, np.zeros(128, np.float32), scale)
    offsets = np.zeros_like(pos, dtype=np.float64)
    for j, p in enumerate(parents):
        offsets[j] = pos[j] - (pos[p] if p >= 0 else np.array([0.0, sole_y, 0.0]))
    return {
        "name": "soma77",
        "joint_names": names,
        "parents": parents,
        "rest_offsets": np.round(offsets, 6).tolist(),
        "rest_rotations": None,
        "soma_joint_orient": np.round(Rotation.from_matrix(rot.astype(np.float64)).as_quat(), 7).tolist(),
        "sole_height": round(-sole_y, 6),
        "soma_version": _soma_version(),
        "reference_pose": "SOMA 0.2.0 reference (as configured by GEM-X for SOMA 0.3.3)",
        "lod": body.runtime_config.lod,
    }


def main(argv: list[str]) -> int:
    request_path = Path(argv[1])
    req = json.loads(request_path.read_text())
    body = build_body(req.get("soma_assets", SOMA_ASSETS), req.get("device", "cpu"), req.get("lod", "mid"))
    cmd = req["command"]
    if cmd == "skeleton":
        sk = neutral_skeleton(body)
        Path(req["out_json"]).write_text(json.dumps(sk, indent=1) + "\n")
        response = {"out_json": req["out_json"], "joints": len(sk["joint_names"])}
    elif cmd == "convert":
        import torch

        result = torch.load(req["motion_pt"], map_location="cpu", weights_only=False)
        response = convert_result(body, result, req["out_npz"], variant=req.get("variant", "raw"))
    else:
        raise SystemExit(f"unknown command {cmd!r}")
    default = req.get("response_path", request_path.with_name("response.json"))
    out = Path(argv[2]) if len(argv) > 2 else Path(default)
    out.write_text(json.dumps(response, indent=2) + "\n")
    print(f"[soma_worker] {cmd} done: {response}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
