"""Regenerate the real-data motion fixtures (GEM-X motion only, no footage) at full frame rate.

Round 2: built from the pipeline's own ``pose3d`` and ``camera`` stage results of the integration
projects (``afv run`` on the Media Samples; camera/pose3d stage version 2: background-flow rotations,
``root_mode="auto"``), cut to a consecutive frame range. Run in the afv env:
``python tests/motion/fixtures/make_real_fixtures.py [projects_dir]`` (default
``<repo>/projects``, one project folder per clip as listed in ``CLIPS``).

Writes ``tests/motion/fixtures/{pose3d,camera}_{ufc_cross,close_boxing,ref_walk}.npz`` and module
C's full-shot ``tests/models/fixtures/{pose3d,camera}_who_am_i.npz``.

(Round 1 built them from the P0 SOMA FK NPZs in ``~/afv/runs/afv_c_fk`` with P0's ViPE cameras,
whose rotations were wrong on the telephoto shots.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from afv.core.project import Project
from afv.core.types import ActorMotion, BodyMotion, CameraTrack, save_result

REPO = Path(__file__).resolve().parents[3]
MOTION = Path(__file__).parent
MODELS = REPO / "tests" / "models" / "fixtures"
CLIPS = {
    # name: (project folder, shot, first frame, end frame, output folder)
    "ufc_cross": ("ufc_cross", "001", 140, 260, MOTION),  # wide lens ~3.5 m: footwork and punches
    "close_boxing": ("close_boxing", "001", 0, 120, MOTION),  # telephoto ~17 m boxing (92 frames)
    "ref_walk": ("ref_walk", "001", 0, 120, MOTION),  # telephoto ~23 m walking (shot 1, before the cut)
    "who_am_i": ("who_am_i", "001", 0, 10_000, MODELS),  # telephoto film fight, the whole shot
}


def _cut(a: np.ndarray | None, sl: slice) -> np.ndarray | None:
    return None if a is None else np.asarray(a)[sl]


def cut_motion(bm: BodyMotion, sl: slice) -> BodyMotion:
    return BodyMotion(
        frames=bm.frames[sl],
        timestamps=bm.timestamps[sl],
        skeleton=bm.skeleton,
        fps=bm.fps,
        contact_channels=bm.contact_channels,
        world_shift=bm.world_shift,
        actors={
            r: ActorMotion(
                a.rot[sl], a.root[sl], _cut(a.conf, sl), _cut(a.contact, sl), a.rest_offsets,
                _cut(a.root_prior, sl),
            )
            for r, a in bm.actors.items()
        },
    )  # fmt: skip


def cut_camera(cam: CameraTrack, sl: slice) -> CameraTrack:
    return CameraTrack(
        frames=cam.frames[sl], timestamps=cam.timestamps[sl], cam_rot=cam.cam_rot[sl],
        cam_pos=cam.cam_pos[sl], focal_px=cam.focal_px[sl], width=cam.width, height=cam.height,
        cx=cam.cx, cy=cam.cy, ground_normal=cam.ground_normal, ground_offset=cam.ground_offset,
        static=cam.static,
    )  # fmt: skip


def main(projects: Path) -> None:
    for name, (proj, shot_id, f0, f1, out_dir) in CLIPS.items():
        shot = Project(projects / proj).shot(shot_id)
        bm, cam = shot.load_result("pose3d"), shot.load_result("camera")
        assert isinstance(bm, BodyMotion) and isinstance(cam, CameraTrack)
        meta = shot.meta("pose3d") or {}
        sl = slice(f0, min(f1, bm.num_frames))
        save_result(cut_motion(bm, sl), out_dir / f"pose3d_{name}.npz")
        save_result(cut_camera(cam, sl), out_dir / f"camera_{name}.npz")
        print(
            name, sl.stop - sl.start, "frames from", shot.dir, f"(pose3d v{meta.get('version')},",
            f"root_mode {meta.get('params', {}).get('root_mode')})",
            (out_dir / f"pose3d_{name}.npz").stat().st_size // 1024, "KiB",
        )  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "projects")
