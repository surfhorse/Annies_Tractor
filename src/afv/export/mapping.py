"""Source-skeleton -> export-rig mapping tables (data, one JSON file per source skeleton).

A table maps each canonical rig bone (unprefixed Mixamo name, see ``afv.export.rig.HUMANOID``) to
the source joint whose world position is that bone's head. File format
(``src/afv/export/data/<source_skeleton>.json``)::

    {"source_skeleton": "soma_provisional",
     "bones": {"Hips": "pelvis", "Spine": "spine1", ...}}

Rules (checked by ``build_rig``): source joints are unique; a rig bone's mapped joint must be a
descendant of its rig parent's mapped joint; required bones must be present. Rig bones absent from
the table are dropped (their children re-parent to the nearest present ancestor). Source joints
absent from the table are *folded*: their rotation still reaches the rig because rotations are
transferred in world space (see ``afv.export.retarget``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_DATA = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Mapping:
    source_skeleton: str
    bones: dict[str, str]  # canonical rig bone -> source joint name
    notes: str = ""
    path: str = field(default="", compare=False)

    @classmethod
    def from_dict(cls, d: dict, path: str = "") -> Mapping:
        return cls(d["source_skeleton"], dict(d["bones"]), d.get("notes", ""), path)

    def to_dict(self) -> dict:
        return {"source_skeleton": self.source_skeleton, "notes": self.notes, "bones": dict(self.bones)}


_MAPPINGS: dict[str, Mapping] = {}


def load_mapping_json(path: Path) -> Mapping:
    return Mapping.from_dict(json.loads(Path(path).read_text()), str(path))


def register_mapping(m: Mapping) -> None:
    _MAPPINGS[m.source_skeleton] = m


def get_mapping(source_skeleton: str) -> Mapping:
    """Mapping table for a source skeleton (registered, or ``data/<name>.json``)."""
    if source_skeleton not in _MAPPINGS:
        f = _DATA / f"{source_skeleton}.json"
        if not f.is_file():
            known = sorted({*_MAPPINGS, *(p.stem for p in _DATA.glob("*.json"))})
            raise KeyError(f"no export mapping for skeleton {source_skeleton!r}; known: {known}")
        register_mapping(load_mapping_json(f))
    return _MAPPINGS[source_skeleton]
