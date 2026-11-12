from __future__ import annotations

import numpy as np
import pytest
from export_helpers import fighter_offsets, synthetic_motion

from afv.core.skeletons import SOMA_PROVISIONAL
from afv.export.bvh import bvh_fk, read_bvh, write_bvh
from afv.export.retarget import retarget, rig_fk
from afv.export.rig import build_rig
from afv.export.root_motion import apply_root_motion


@pytest.mark.parametrize(("root_bone", "mode", "fps"), [(False, "world", 30.0), (True, "extracted", 29.97)])
def test_bvh_round_trip(tmp_path, root_bone, mode, fps):
    rig = build_rig(SOMA_PROVISIONAL, rest_offsets=fighter_offsets(SOMA_PROVISIONAL, 2), root_bone=root_bone,
                    prefix="mixamorig:")
    rot, root = synthetic_motion(SOMA_PROVISIONAL, 37, seed=9, amp=1.2)  # large angles: gimbal cases
    m = apply_root_motion(retarget(rig, rot, root, fps), mode)
    path = write_bvh(m, tmp_path / "take.bvh")
    d = read_bvh(path)
    assert d.names == list(rig.names)
    assert d.parents == list(rig.parents)
    assert d.num_frames == 37
    assert abs(d.fps - fps) < 1e-4
    assert len(d.end_sites) == sum(1 for b in range(rig.num_bones) if not rig.children(b))
    pos = bvh_fk(d)
    want, _ = rig_fk(m)
    assert np.abs(pos - want).max() < 1e-5  # 6 decimals in cm / degrees
    six = [n for n, c in zip(d.names, d.channels, strict=True) if len(c) == 6]
    assert six == [rig.names[b] for b in m.translating]


def test_bvh_scale_metres(tmp_path):
    rig = build_rig(SOMA_PROVISIONAL)
    rot, root = synthetic_motion(SOMA_PROVISIONAL, 5, seed=1)
    m = retarget(rig, rot, root, 30.0)
    d = read_bvh(write_bvh(m, tmp_path / "m.bvh", scale=1.0))
    assert np.abs(bvh_fk(d, scale=1.0) - rig_fk(m)[0]).max() < 1e-5
