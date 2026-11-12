"""Minimal 8-bit grayscale PNG read/write (no imaging dependency)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def write_png_gray(path: str | Path, img: np.ndarray) -> None:
    """Write a 2-D uint8 (or bool -> 0/255) array as a grayscale PNG."""
    a = np.asarray(img)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    if a.ndim != 2 or a.dtype != np.uint8:
        raise ValueError("expected a 2-D uint8 or bool array")
    h, w = a.shape
    raw = b"".join(b"\x00" + a[r].tobytes() for r in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    Path(path).write_bytes(_SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 9))
                           + _chunk(b"IEND", b""))


def read_png_gray(path: str | Path) -> np.ndarray:
    """Read a PNG written by ``write_png_gray`` (8-bit grayscale, non-interlaced, any filter)."""
    data = Path(path).read_bytes()
    if not data.startswith(_SIG):
        raise ValueError(f"{path}: not a PNG")
    pos, idat, w = len(_SIG), b"", 0
    h = 0
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if kind == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if (depth, ctype, interlace) != (8, 0, 0):
                raise ValueError(f"{path}: only 8-bit grayscale non-interlaced PNGs are supported")
        elif kind == b"IDAT":
            idat += body
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w + 1)
    out = np.zeros((h, w), np.uint8)
    prev = np.zeros(w, np.int32)
    for r in range(h):
        f, line = rows[r, 0], rows[r, 1:].astype(np.int32)
        if f == 0:
            cur = line
        elif f == 2:
            cur = (line + prev) & 0xFF
        elif f in (1, 3, 4):
            cur = np.zeros(w, np.int32)
            for x in range(w):
                left = cur[x - 1] if x else 0
                if f == 1:
                    pred = left
                elif f == 3:
                    pred = (left + prev[x]) // 2
                else:
                    up_left = prev[x - 1] if x else 0
                    p = left + prev[x] - up_left
                    pa, pb, pc = abs(p - left), abs(p - prev[x]), abs(p - up_left)
                    pred = left if pa <= pb and pa <= pc else (prev[x] if pb <= pc else up_left)
                cur[x] = (line[x] + pred) & 0xFF
        else:
            raise ValueError(f"{path}: bad PNG filter {f}")
        out[r] = cur
        prev = cur
    return out
