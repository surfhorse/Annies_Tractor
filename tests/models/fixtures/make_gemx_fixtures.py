"""Regenerate the GEM-X/SOMA test fixtures from P0 outputs (motion data only, no footage).

Step 1 (gemx env), per run: ``soma_worker.py`` with
``{"command": "convert", "motion_pt": ~/afv/runs/p0_shared/<clip>/rank<k>/motion.pt, "out_npz": ...}``
(the P0 FK NPZs live in ``~/afv/runs/afv_c_fk/<clip>_rank<k>.npz``).
Step 2 (afv env): ``python tests/models/fixtures/make_gemx_fixtures.py [fk_dir]``:

* ``gemx_*.npz``: every ``STEP``-th frame (at most ``MAX_FRAMES``) of the runs below, FK NPZ format.
* ``pose3d_<clip>.npz`` + ``camera_<clip>.npz``: a full, consecutive-frame pose3d result
  (``BodyMotion``, both fighters, floor re-fitted, ``world_shift`` and ``root_prior`` set) and its
  ``CameraTrack`` (GEM-X camera + gravity, ground from the P0 SAM 2 boxes), via ``save_result``.
  For other modules' tests (pair, contacts, QC). Reproject with ``cam_pos + world_shift``.
  Round 2: no longer written here (``POSE3D_CLIPS`` is empty); see
  ``tests/motion/fixtures/make_real_fixtures.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RUNS = {
    "gemx_ufc_cross_rank0.npz": "UFC_cross_rank0.npz",  # wide lens, focal ~1200 px, ~3.5 m
    "gemx_who_am_i_rank1.npz": "Who_Am_I_rank1.npz",  # telephoto film, focal ~7600 px, ~20 m
}
# Round 2: pose3d_who_am_i/camera_who_am_i now come from the pipeline's own stage results
# (tests/motion/fixtures/make_real_fixtures.py); P0's GEM-X cameras had wrong telephoto rotations.
POSE3D_CLIPS: tuple[str, ...] = ()
STEP = 7
MAX_FRAMES = 40
PER_FRAME = {
    "incam_rot",
    "incam_pos",
    "global_rot",
    "global_pos",
    "contact_logits",
    "kp2d",
    "K",
    "R_w2c",
    "bbx_xys",
    "boxes_xyxy",
    "observed",
    "frame_times",
    "incam_global_orient",
    "global_global_orient",
    "incam_transl",
    "global_transl",
    "body_pose",
}


def main(fk_dir: Path) -> None:
    out_dir = Path(__file__).parent
    for out_name, src in RUNS.items():
        with np.load(fk_dir / src) as z:
            n = z["incam_pos"].shape[0]
            keep = np.arange(0, n, STEP)[:MAX_FRAMES]
            arrays = {k: (z[k][keep] if k in PER_FRAME else z[k]) for k in z.files}
        arrays["source_frames"] = keep
        np.savez_compressed(out_dir / out_name, **arrays)
        print(out_name, len(keep), "frames", (out_dir / out_name).stat().st_size // 1024, "KiB")
    for clip in POSE3D_CLIPS:
        make_pose3d_fixture(fk_dir, clip, out_dir)


def make_pose3d_fixture(fk_dir: Path, clip: str, out_dir: Path) -> None:
    from afv.core.types import save_result
    from afv.models.camera import place_on_ground
    from afv.models.gemx import Pose3dParams, SomaFK, body_motion_from_fk, camera_from_gemx

    z = np.load(Path.home() / "afv/runs/track" / clip / "tracks.npz")
    boxes = z["boxes"].astype(np.float64)[:, list(z["fighter_slots"])]
    fks = {r: SomaFK.load(fk_dir / f"{clip}_rank{k}.npz") for k, r in enumerate("AB")}
    cam, _ = place_on_ground(camera_from_gemx(fks["A"]), boxes)
    params = Pose3dParams()  # the pose3d stage's settings (root mode, floor scale)
    motion, _ = body_motion_from_fk(
        fks, cam, root_mode=params.root_mode, floor_scale_max=params.floor_scale_max
    )
    for kind, res in (("pose3d", motion), ("camera", cam)):
        path = out_dir / f"{kind}_{clip.lower()}.npz"
        save_result(res, path)
        print(path.name, res.num_frames, "frames", path.stat().st_size // 1024, "KiB")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "afv/runs/afv_c_fk")
