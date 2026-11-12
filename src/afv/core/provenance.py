"""Provenance and licence records (spec NFR-12, ``afv provenance``, ``afv models --check``).

Every export must be traceable to its source (path, sha256, title, URL, in/out times) and to the
models that produced it (stage implementation, version, weight hash, licences). Stage metas already
store this per stage; these helpers gather it per shot and project. A result made by a range re-run
(``afv run --from S --frames a:b``) also carries its ``range`` record: the frames, the result it
replaced, the full run the chain started from and every range since.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

from afv.core.cache import hash_file
from afv.core.project import Project, Shot
from afv.core.stage import STAGE_ORDER
from afv.core.types import LicenceRecord


def licence_to_dict(rec: LicenceRecord) -> dict[str, Any]:
    return dataclasses.asdict(rec)


def licence_problems(
    component: str, records: Iterable[LicenceRecord], *, needs_licence: bool
) -> tuple[list[str], list[str]]:
    """(problems, warnings) for one stage implementation's licence records.

    Problems: no records for a model stage, ``commercial_output_ok`` unknown (None), or outputs known
    not to be cleared for commercial use (False): NFR-12 requires commercial-OK licences for export.
    Warnings: missing weight hashes.
    """
    problems: list[str] = []
    warnings: list[str] = []
    recs = list(records)
    if needs_licence and not recs:
        problems.append(f"{component}: no licence records")
    for r in recs:
        tag = f"{component}: {r.component}"
        if r.commercial_output_ok is None:
            problems.append(
                f"{tag}: commercial use of outputs unknown (code {r.code_licence or '?'}, "
                f"weights {r.weight_licence or '?'})"
            )
        elif r.commercial_output_ok is False:
            problems.append(f"{tag}: outputs not cleared for commercial use (NFR-12)")
        if needs_licence and not r.weight_sha256:
            warnings.append(f"{tag}: no weight sha256 recorded")
    return problems, warnings


def take_provenance(shot: Shot) -> dict[str, Any]:
    """Compact source record for ``Take.provenance`` and export metadata."""
    src, info = shot.project.info.source, shot.info
    return {
        "source_path": src.path,
        "source_sha256": src.sha256,
        "title": src.title,
        "url": src.url,
        "shot": shot.id,
        "in_s": info.start,
        "out_s": info.end,
        "speed_factor": info.speed_factor,
    }


def shot_provenance(shot: Shot, *, hash_exports: bool = True) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for name in STAGE_ORDER:
        meta = shot.meta(name)
        if not meta:
            continue
        stages[name] = {
            k: meta.get(k)
            for k in ("impl", "version", "env", "weight_hash", "key", "output_hash", "finished", "params")
        }
        stages[name]["licences"] = meta.get("licences", [])
        if meta.get("relevel"):  # camera re-levelled from pose3d's gravity (runner's re-level pass)
            stages[name]["relevel"] = meta["relevel"]
        if isinstance(meta.get("range"), dict):  # a range re-run (afv run --from S --frames a:b)
            stages[name]["range"] = meta["range"]
            stages[name]["base_key"] = meta.get("base_key")
    exports = []
    if shot.exports_dir.is_dir():
        for p in sorted(shot.exports_dir.rglob("*")):
            if p.is_file():
                e: dict[str, Any] = {"path": str(p.relative_to(shot.dir)), "size": p.stat().st_size}
                if hash_exports:
                    e["sha256"] = hash_file(p)
                exports.append(e)
    edits = sorted(p.stem for p in shot.edits_dir.glob("*.json")) if shot.edits_dir.is_dir() else []
    return {
        **take_provenance(shot),
        "notes": shot.info.notes,
        "stages": stages,
        "edits": edits,
        "exports": exports,
    }


def project_provenance(
    project: Project, shot_ids: list[str] | None = None, *, hash_exports: bool = True
) -> dict[str, Any]:
    src = project.info.source
    ids = shot_ids if shot_ids is not None else project.shot_ids()
    return {
        "project": str(project.root),
        "source": {"path": src.path, "sha256": src.sha256, "title": src.title, "url": src.url},
        "shots": [shot_provenance(project.shot(i), hash_exports=hash_exports) for i in ids],
    }


_COMMERCIAL = {True: "", False: ", not commercial", None: ", unknown"}


def format_provenance(prov: dict[str, Any]) -> str:
    src = prov["source"]
    lines = [f"source  {src['path']}", f"sha256  {src['sha256']}"]
    if src.get("title"):
        lines.append(f"title   {src['title']}")
    if src.get("url"):
        lines.append(f"url     {src['url']}")
    for s in prov["shots"]:
        lines.append("")
        lines.append(
            f"shot {s['shot']}: {s['in_s']:.3f}-{s['out_s']:.3f} s, speed x{s['speed_factor']:g}"
            + (f"  ({s['notes']})" if s.get("notes") else "")
        )
        for name, st in s["stages"].items():
            wh = (st.get("weight_hash") or "")[:12]
            lic = ", ".join(
                f"{r['component']} [{r['code_licence']}/{r['weight_licence']}"
                f"{_COMMERCIAL[r['commercial_output_ok']]}]"
                for r in st.get("licences", [])
            )
            lines.append(
                f"  {name:<12} {st.get('impl')} v{st.get('version')}"
                + (f" weights {wh}" if wh else "")
                + (f"  {lic}" if lic else "")
            )
            if isinstance(st.get("range"), dict):
                rg = st["range"]
                hist = rg.get("history") or []
                frames = ", ".join(f"{h['frames'][0]}:{h['frames'][1]}" for h in hist if h.get("frames"))
                lines.append(
                    f"  {'':<12} range re-runs since the full run of "
                    f"{(rg.get('base') or {}).get('finished')}: "
                    f"frames {frames or rg.get('frames')}"
                )
            if st.get("relevel"):
                rl = st["relevel"]
                lines.append(
                    f"  {'':<12} re-levelled from pose3d's gravity {rl.get('override')} "
                    f"(flags {rl.get('trigger', {}).get('flags', [])})"
                )
        if s["edits"]:
            lines.append(f"  edits: {', '.join(s['edits'])}")
        for e in s["exports"]:
            lines.append(f"  export {e['path']}" + (f"  sha256 {e['sha256'][:16]}" if "sha256" in e else ""))
    return "\n".join(lines)
