"""``afv retarget``: a shot's fighters onto the owner's own character, headless (WSL side).

Builds the shot's take into a work folder (``afv.take.make_take``; the shot's own ``take/`` and its
``take.blend`` are not touched), then runs ``blender/afv_addon/retarget_headless.py`` in Blender: the
take is built with the add-on's builder, the character is imported (FBX / GLB / a .blend's armature),
mapped (a ``*.afvmap.json`` file, else the engine's ``guess_map``), each fighter is retargeted onto it
(``blender/afv_addon/retarget.py``) and exported as FBX with an engine preset's exporter arguments
(``afv.export.presets``; only those: no root bone, root-motion mode, armature rename or Unreal
centimetre scene, unlike ``afv export``). Files: ``<shot>_<role>_<character>.fbx`` (``<character>``:
the file stem made file-name safe, by ``retarget_io.safe_name`` in Blender; the FBX clip has the same
name) in ``shots/<id>/exports/retarget/`` (or ``out_dir``), moved into place when complete. Both
fighters are retargeted onto the same imported character, so their files share the take's world
origin (like the pipeline's exports). The character keeps its own size and units.

The character file's type is checked early here (``CHARACTER_EXTS``, equal to
``retarget_io.CHARACTER_EXTS`` in Blender, which checks again: a test compares the two lists).

Refusals the user can fix (a required bone not mapped, a bad map, an engine refusal such as a
non-uniform scale) raise ``RetargetError`` with the message (``missing``: the required bones not
mapped); Blender failures raise ``afv.export.blender.BlenderError``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import textwrap
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from afv.core.project import Shot
from afv.export.presets import PRESETS, get_preset

HEADLESS_SCRIPT = "afv_addon/retarget_headless.py"
EXPORT_SUBDIR = "retarget"
CHARACTER_EXTS = (".fbx", ".glb", ".gltf", ".blend")  # == blender/afv_addon/retarget_io.py (tested)
ROLES = ("A", "B")


class RetargetError(RuntimeError):
    """The retarget was refused (the message says what to fix); ``missing``: required bones not mapped."""

    def __init__(self, message: str, missing: Sequence[str] = (), result: dict[str, Any] | None = None):
        super().__init__(message)
        self.missing = list(missing)
        self.result = result or {}


def presets_payload() -> list[dict[str, Any]]:
    """Every engine preset with its exporter arguments (``afv retarget --list-presets``, serve ``presets``):
    the add-on's Export character takes the FBX arguments from here."""
    return [p.to_dict() for p in PRESETS.values()]


def format_presets() -> str:
    """``afv retarget --list-presets`` as text: each preset and the FBX exporter arguments it uses."""
    lines = []
    for p in presets_payload():
        lines.append(f"{p['name']}: {p['label']} (afv export formats: {', '.join(p['formats'])})")
        args = ", ".join(f"{k}={v}" for k, v in sorted(p["fbx"].items()))
        lines += textwrap.wrap(f"fbx: {args}", width=106, initial_indent="  ", subsequent_indent="    ")
    return "\n".join(lines)


def parse_roles(role: str) -> list[str]:
    r = role.strip()
    if r.lower() == "both":
        return list(ROLES)
    if r.upper() in ROLES:
        return [r.upper()]
    raise RetargetError(f"--role: expected A, B or both, got {role!r}")


def retarget_shot(
    shot: Shot,
    character: Path,
    *,
    map_path: Path | None = None,
    roles: Sequence[str] = ROLES,
    preset: str | None = None,
    out_dir: Path | None = None,
    lock_feet: bool = True,
    save_map: Path | None = None,
    check: bool = False,
    blender: Path | None = None,
) -> dict[str, Any]:
    """Retarget ``roles`` of ``shot`` onto ``character`` and export one FBX per role (see module doc).

    ``preset``: the engine preset whose FBX arguments are used (default: the shot's export preset).
    ``check``: re-import each file in Blender and compare it with the retarget (bone heads, mm).
    Returns ``{"files": {role: path}, "roles": {role: {"report", "export", "check"?}}, "target",
    "map_source", "map", "saved_map", "preset", "seconds", "blender_s", "take_s", ...}``. The caller
    holds the shot lock."""
    from afv import take as tk
    from afv.export import blender as bl

    t0 = time.perf_counter()
    character = Path(character).expanduser().resolve()
    if character.suffix.lower() not in CHARACTER_EXTS:
        raise RetargetError(f"--character: expected a character file ({', '.join(CHARACTER_EXTS)}), "
                            f"got {character.name}")
    if not character.is_file():
        raise RetargetError(f"--character: no such file: {character}")
    if map_path is not None:
        map_path = Path(map_path).expanduser().resolve()
        if not map_path.is_file():
            raise RetargetError(f"--map: no such file: {map_path}")
    roles = list(dict.fromkeys(roles))
    if not roles or any(r not in ROLES for r in roles):
        raise RetargetError(f"roles: expected A and/or B, got {roles}")
    rig = tk.shot_rig_settings(shot)
    preset_name = preset or rig.preset
    try:
        p = get_preset(preset_name)
    except KeyError as e:
        raise RetargetError(str(e.args[0])) from None
    exe = blender or bl.find_blender()
    if exe is None:
        raise bl.BlenderError("Blender not found; set AFV_BLENDER or install Blender 5.2")
    out_dir = (Path(out_dir).expanduser().resolve() if out_dir is not None
               else shot.exports_dir / EXPORT_SUBDIR)
    shot.work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="retarget_", dir=shot.work_root))
    ok = False
    try:
        t = time.perf_counter()
        tj = tk.make_take(shot, work / "take")  # with the plan check: stale results are warned about
        take_s = time.perf_counter() - t
        summary = tk.take_summary(tj)
        missing_roles = [r for r in roles if r not in summary["fighters"]]
        if missing_roles:
            raise RetargetError(f"shot {shot.id} has no fighter {', '.join(missing_roles)} "
                                f"(fighters: {', '.join(summary['fighters'])})")
        files_dir = work / "files"
        files_dir.mkdir()
        job: dict[str, Any] = {
            "take": bl.host_path(tj, exe),
            "character": bl.host_path(character, exe),
            "map": bl.host_path(map_path, exe) if map_path is not None else None,
            "roles": roles,
            "lock_feet": bool(lock_feet),
            "fbx_args": dict(p.fbx),
            "shot_id": shot.id,
            "out_dir": bl.host_path(files_dir, exe),
            "save_map": None,
            "check": bool(check),
        }
        if save_map is not None:
            save_map = Path(save_map).expanduser().resolve()
            save_map.parent.mkdir(parents=True, exist_ok=True)
            job["save_map"] = bl.host_path(save_map, exe)
        t = time.perf_counter()
        res = bl.run_blender(HEADLESS_SCRIPT, job, work / "blender", blender=exe)
        blender_s = time.perf_counter() - t
        out = _result(res, save_map)
        if not res.get("ok"):
            why = str(res.get("refused") or res.get("error") or res)
            raise RetargetError(why, res.get("missing") or [], out)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for r in roles:
            f = files_dir / Path(str(res.get("files", {}).get(r, "")).replace("\\", "/")).name
            dest = out_dir / f.name
            if not f.name or not f.is_file():
                raise bl.BlenderError(f"Blender did not write the FBX of fighter {r}")
            try:
                os.replace(f, dest)
            except OSError:
                shutil.move(str(f), str(dest))
            files[r] = str(dest)
            out["roles"][r]["export"]["path"] = str(dest)
        ok = True
        out.update(shot=shot.id, character=str(character), preset=p.name, files=files,
                   out_dir=str(out_dir), lock_feet=bool(lock_feet), take_s=round(take_s, 3),
                   blender_s=round(blender_s, 3), seconds=round(time.perf_counter() - t0, 3),
                   stale_summary=summary.get("stale_summary", ""))
        return out
    finally:
        if ok or not os.environ.get("AFV_DEBUG"):
            shutil.rmtree(work, ignore_errors=True)


def _result(res: dict[str, Any], save_map: Path | None) -> dict[str, Any]:
    keep = ("target", "bones", "meshes", "objects", "character_name", "map_source", "map", "import", "roles",
            "fps")
    out = {k: res[k] for k in keep if k in res}
    out.setdefault("roles", {})
    out["saved_map"] = str(save_map) if save_map is not None and res.get("saved_map") else None
    return out


def format_report(role: str, rep: dict[str, Any]) -> list[str]:
    """Readable lines for one role's retarget report."""
    slide = rep.get("foot_slide_cm_s") or {}

    def cm(v: Any) -> str:
        return "-" if v is None else f"{v:.2f}"

    lines = [
        f"{role}: action {rep['action']}, frames {rep['frames'][0]}-{rep['frames'][1]}, "
        f"{rep['mapped']} bones mapped, {rep['seconds']:.1f} s",
        f"   facing {rep['facing']} (yaw {rep['facing_yaw_deg']:.1f} deg), "
        f"hips scale {rep['hips_scale']:.4f}",
        f"   foot slide L {cm(slide.get('left'))} / R {cm(slide.get('right'))} cm/s, penetration "
        f"{rep['penetration_cm']:.2f} cm, max direction error {rep['max_dir_err_deg']:.2f} deg, twist "
        f"{rep['max_twist_err_deg']:.2f} deg" + (" (IK legs left out)" if rep.get("ik_bones") else ""),
    ]
    if rep.get("unmapped"):
        u = rep["unmapped"]
        lines.append(f"   unmapped ({len(u)}): {', '.join(u[:12])}{' ...' if len(u) > 12 else ''}")
    lines += [f"   note: {n}" for n in rep.get("notes", [])]
    return lines
