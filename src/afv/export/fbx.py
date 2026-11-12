"""Minimal binary FBX reader for checks: global settings, Model nodes (type, Lcl TRS) and parents."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SCALAR = {"Y": "<h", "C": "<?", "I": "<i", "L": "<q", "F": "<f", "D": "<d"}
_ARRAY = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "?", "c": "?"}


@dataclass
class FbxNode:
    name: str
    props: list
    children: list[FbxNode] = field(default_factory=list)

    def child(self, name: str) -> FbxNode | None:
        return next((c for c in self.children if c.name == name), None)

    def properties70(self) -> dict[str, list]:
        p70 = self.child("Properties70")
        return {} if p70 is None else {p.props[0]: p.props[4:] for p in p70.children if p.name == "P"}


@dataclass
class FbxModel:
    uid: int
    name: str
    kind: str  # "Null", "LimbNode", "Mesh", ...
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)  # degrees
    scaling: tuple[float, float, float] = (1.0, 1.0, 1.0)
    parent: str | None = None  # parent Model name, None = scene root


@dataclass
class FbxSummary:
    version: int
    global_settings: dict[str, list]
    models: dict[str, FbxModel]


def read_fbx(path: Path) -> list[FbxNode]:
    data = Path(path).read_bytes()
    if not data.startswith(b"Kaydara FBX Binary"):
        raise ValueError(f"{path}: not a binary FBX")
    version = struct.unpack_from("<I", data, 23)[0]
    wide = version >= 7500
    head = struct.Struct("<QQQ" if wide else "<III")

    def prop(off: int):
        t = chr(data[off])
        off += 1
        if t in _SCALAR:
            fmt = _SCALAR[t]
            return struct.unpack_from(fmt, data, off)[0], off + struct.calcsize(fmt)
        if t in _ARRAY:
            _n, enc, clen = struct.unpack_from("<III", data, off)
            off += 12
            raw = data[off : off + clen]
            if enc:
                raw = zlib.decompress(raw)
            return np.frombuffer(raw, _ARRAY[t]), off + clen
        n = struct.unpack_from("<I", data, off)[0]
        off += 4
        raw = data[off : off + n]
        return (raw.decode("utf-8", "replace") if t == "S" else raw), off + n

    def node(off: int) -> tuple[FbxNode | None, int]:
        end, nprops, _plen = head.unpack_from(data, off)
        off += head.size
        if end == 0:
            return None, off
        nl = data[off]
        name = data[off + 1 : off + 1 + nl].decode()
        off += 1 + nl
        props = []
        for _ in range(nprops):
            v, off = prop(off)
            props.append(v)
        n = FbxNode(name, props)
        while off < end:
            k, off = node(off)
            if k is None:
                break
            n.children.append(k)
        return n, end

    off = 27
    top: list[FbxNode] = []
    while off < len(data):
        n, off = node(off)
        if n is None:
            break
        top.append(n)
    return top


def fbx_summary(path: Path) -> FbxSummary:
    data = Path(path).read_bytes()
    version = struct.unpack_from("<I", data, 23)[0]
    top = {n.name: n for n in read_fbx(path)}
    gs = top["GlobalSettings"].properties70() if "GlobalSettings" in top else {}
    by_uid: dict[int, FbxModel] = {}
    for n in top["Objects"].children:
        if n.name != "Model":
            continue
        uid, label, kind = n.props[0], n.props[1].split("\x00")[0], n.props[2]
        p = n.properties70()

        def vec(key: str, default: tuple[float, float, float], p=p) -> tuple[float, float, float]:
            v = p.get(key)
            return default if v is None else tuple(float(x) for x in v[:3])  # type: ignore[return-value]

        by_uid[uid] = FbxModel(uid, label, kind, vec("Lcl Translation", (0.0, 0.0, 0.0)),
                               vec("Lcl Rotation", (0.0, 0.0, 0.0)), vec("Lcl Scaling", (1.0, 1.0, 1.0)))
    for c in top["Connections"].children:
        if c.props[0] == "OO" and c.props[1] in by_uid and c.props[2] in by_uid:
            by_uid[c.props[1]].parent = by_uid[c.props[2]].name
    return FbxSummary(version, gs, {m.name: m for m in by_uid.values()})
