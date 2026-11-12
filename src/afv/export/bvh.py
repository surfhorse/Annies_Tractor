"""BVH writer (export rig) and reader (tests, inspection).

Conventions written: Y-up, character faces +Z (afv's internal frame, which BVH shares), lengths x
``scale`` (default 100: centimetres). Rotation channels ``Zrotation Xrotation Yrotation`` in
degrees, meaning local R = Rz * Rx * Ry. Bones with animated translation (top bone, and Hips under
a root bone) get 6 channels; their position channels are the joint's local translation (they
*replace* OFFSET, the convention of Blender's importer and most tools). Each leaf joint gets an
``End Site`` at its rest tail.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from afv.export.retarget import RigMotion
from afv.motion.rotations import qmul, qrotate

ROT_ORDER = "ZXY"
_AXIS = {"X": 0, "Y": 1, "Z": 2}


def _fmt(v: np.ndarray) -> str:
    return " ".join(f"{x:.6f}" for x in v)


def write_bvh(m: RigMotion, path: Path, *, scale: float = 100.0) -> Path:
    rig = m.rig
    path = Path(path)
    trans = set(m.translating)
    rest_tail = rig.tails - rig.rest_positions()
    lines = ["HIERARCHY"]
    order: list[int] = []

    def emit(b: int, depth: int) -> None:
        ind = "  " * depth
        kw = "ROOT" if rig.parents[b] < 0 else "JOINT"
        lines.append(f"{ind}{kw} {rig.names[b]}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind}  OFFSET {_fmt(rig.offsets[b] * scale)}")
        rot_ch = " ".join(f"{a}rotation" for a in ROT_ORDER)
        if b in trans:
            lines.append(f"{ind}  CHANNELS 6 Xposition Yposition Zposition {rot_ch}")
        else:
            lines.append(f"{ind}  CHANNELS 3 {rot_ch}")
        order.append(b)
        kids = rig.children(b)
        for k in kids:
            emit(k, depth + 1)
        if not kids:
            lines.append(f"{ind}  End Site")
            lines.append(f"{ind}  {{")
            lines.append(f"{ind}    OFFSET {_fmt(rest_tail[b] * scale)}")
            lines.append(f"{ind}  }}")
        lines.append(f"{ind}}}")

    for b, p in enumerate(rig.parents):
        if p < 0:
            emit(b, 0)

    n_frames = m.num_frames
    cols = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # gimbal-lock warnings: the angles are still a valid triple
        for b in order:
            if b in trans:
                cols.append(m.loc[:, b] * scale)
            eul = Rotation.from_quat(m.rot[:, b]).as_euler(ROT_ORDER, degrees=False)
            cols.append(np.degrees(np.unwrap(eul, axis=0)))
    data = np.concatenate(cols, axis=1) if cols else np.zeros((n_frames, 0))
    lines += ["MOTION", f"Frames: {n_frames}", f"Frame Time: {1.0 / m.fps:.10f}"]
    lines += [_fmt(row) for row in data]
    path.write_text("\n".join(lines) + "\n")
    return path


@dataclass
class BvhData:
    names: list[str]
    parents: list[int]
    offsets: np.ndarray  # [J,3] file units
    channels: list[list[str]]
    end_sites: dict[int, np.ndarray]
    motion: np.ndarray  # [F,C]
    frame_time: float

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    @property
    def num_frames(self) -> int:
        return int(self.motion.shape[0])


def read_bvh(path: Path) -> BvhData:
    tokens = Path(path).read_text().split()
    i = 0
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []
    end_sites: dict[int, np.ndarray] = {}
    stack: list[int] = []

    def take() -> str:
        nonlocal i
        t = tokens[i]
        i += 1
        return t

    if take() != "HIERARCHY":
        raise ValueError("not a BVH file")
    while True:
        t = take()
        if t in ("ROOT", "JOINT"):
            names.append(take())
            parents.append(stack[-1] if stack else -1)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            stack.append(len(names) - 1)
            if take() != "{":
                raise ValueError("expected {")
        elif t == "OFFSET":
            offsets[stack[-1]] = [float(take()) for _ in range(3)]
        elif t == "CHANNELS":
            channels[stack[-1]] = [take() for _ in range(int(take()))]
        elif t == "End":
            take()  # "Site"
            take()  # "{"
            take()  # "OFFSET"
            end_sites[stack[-1]] = np.array([float(take()) for _ in range(3)])
            take()  # "}"
        elif t == "}":
            stack.pop()
        elif t == "MOTION":
            break
        else:
            raise ValueError(f"unexpected BVH token {t!r}")
    take()  # Frames:
    n_frames = int(take())
    take()  # Frame
    take()  # Time:
    frame_time = float(take())
    n_ch = sum(len(c) for c in channels)
    vals = np.array([float(x) for x in tokens[i : i + n_frames * n_ch]])
    if vals.size != n_frames * n_ch:
        raise ValueError(f"BVH motion has {vals.size} values, expected {n_frames * n_ch}")
    return BvhData(
        names, parents, np.array(offsets), channels, end_sites, vals.reshape(n_frames, n_ch), frame_time
    )


def bvh_fk(d: BvhData, *, scale: float = 100.0) -> np.ndarray:
    """World joint positions [F,J,3] (file units / ``scale``) from a parsed BVH."""
    n_frames = d.num_frames
    pos = np.empty((n_frames, len(d.names), 3))
    wrot = np.empty((n_frames, len(d.names), 4))
    col = 0
    for j, (p, chans) in enumerate(zip(d.parents, d.channels, strict=True)):
        trans = np.broadcast_to(d.offsets[j], (n_frames, 3)).copy()
        rot_axes = ""
        rot_cols = []
        for c in chans:
            v = d.motion[:, col]
            col += 1
            if c.endswith("position"):
                trans[:, _AXIS[c[0]]] = v
            else:
                rot_axes += c[0]
                rot_cols.append(v)
        if rot_axes:
            local = Rotation.from_euler(rot_axes, np.stack(rot_cols, axis=1), degrees=True).as_quat()
        else:
            local = np.tile([0.0, 0.0, 0.0, 1.0], (n_frames, 1))
        if p < 0:
            pos[:, j] = trans
            wrot[:, j] = local
        else:
            pos[:, j] = pos[:, p] + qrotate(wrot[:, p], trans)
            wrot[:, j] = qmul(wrot[:, p], local)
    return pos / scale
