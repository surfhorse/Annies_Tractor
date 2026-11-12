"""Tracking review: which detect_track QC findings should stop ``afv run`` before the model stages.

detect_track's QC summary (``qc.json`` "summary", copied into its meta; written by
``afv.models.track.compute_qc``) flags runs of frames per tracked object with reasons
(``empty``, ``drift``, ``jump``, ``area_jump``, ``appearance``, ``swap``, ``merge``,
``shared_detection``) and lists suggestions (``untracked_person``, ``drift``). A *tracking hint* is
a finding that can make a fighter's motion wrong:

- a fighter's run with an identity reason: likely ID swap or two masks on one person;
- a fighter's run with ``drift`` (mask matches no detection), ``empty`` (no mask: fighter lost) or
  ``jump`` (box jumps between frames, often onto another person);
- a person nobody tracks (``untracked_person``): a lost fighter, or someone (e.g. a referee) who
  should be tracked so the fighters' masks do not leak onto them.

``area_jump``-only runs and anything about extra people (referee, corner) are not hints: they do
not change the fighters' boxes. Each hint carries the fix: ``afv pick <shot> --frame F`` to see the
numbered people on that frame, then ``afv pick <shot> --prompt F:OBJ:DET``; for an identity hint also
``afv pick <shot> --swap F:O1:O2`` (the fighters traded identities from frame F: ``edits/ids.json``).

Acknowledging: ``--yes`` (afv run, export, batch) records the hints of the current detect_track result in
``<shot>/review.json``, keyed on its output hash and the hint set: a new tracking result (e.g. after
a re-prompt) or a new set of hints (QC code or thresholds changed) is reviewed afresh.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from afv.core.cache import atomic_write_json, hash_obj, read_json

REVIEW_FILE = "review.json"
IDENTITY = ("swap", "merge", "shared_detection", "appearance")
HINT_REASONS = (*IDENTITY, "drift", "empty", "jump")
MAX_LISTED = 12


def _who(obj: dict[str, Any]) -> str:
    role = obj.get("role")
    return f"fighter {role}" if role else str(obj.get("label") or f"person {obj.get('id')}")


def _kind(reasons: set[str]) -> tuple[str, str]:
    if reasons & set(IDENTITY):
        named = ", ".join(r.replace("_", " ") for r in IDENTITY if r in reasons)
        return "identity", f"likely ID swap or mix-up ({named})"
    if "empty" in reasons:
        return "lost", "no mask (fighter lost)"
    if "drift" in reasons:
        return "drift", "mask matches no detection (drifted or lost)"
    return "jump", "box jumps between frames (mask may have moved to another person)"


def tracking_hints(qc: Any, shot_id: str) -> list[dict[str, Any]]:
    """Hints from detect_track's QC summary (see the module docstring); [] when there are none or the
    summary is not in the expected form."""
    if not isinstance(qc, dict):
        return []
    hints: list[dict[str, Any]] = []
    objects = [o for o in qc.get("objects") or [] if isinstance(o, dict)]
    drift_sugg = {
        (s.get("obj"), int((s.get("frames") or [s.get("frame", -1)])[0])): s
        for s in qc.get("suggestions") or []
        if isinstance(s, dict) and s.get("kind") == "drift"
    }
    for o in objects:
        if not o.get("role"):
            continue
        for run in o.get("flagged_runs") or []:
            try:
                s, e, reasons = int(run[0]), int(run[1]), set(run[2])
            except (TypeError, ValueError, IndexError):
                continue
            if not reasons & set(HINT_REASONS):
                continue
            kind, what = _kind(reasons)
            oid = o.get("id")
            sugg = drift_sugg.get((oid, s))
            det = sugg.get("detection") if sugg else None
            fix = (
                f"afv pick {shot_id} --prompt {s}:{oid}:{det}"
                if det is not None
                else f"afv pick {shot_id} --frame {s}, then afv pick {shot_id} --prompt {s}:{oid}:DET"
            )
            other = next((x.get("id") for x in objects if x.get("role") and x.get("id") != oid), None)
            if kind == "identity" and other is not None:
                fix += (
                    f"; if the fighters traded identities from frame {s}: "
                    f"afv pick {shot_id} --swap {s}:{oid}:{other}"
                )
            hints.append({
                "kind": kind, "object": oid, "role": o.get("role"), "frames": [s, e],
                "reasons": sorted(reasons), "message": f"{_who(o)}: {what} on frames {s}-{e}", "fix": fix,
            })  # fmt: skip
    for sg in qc.get("suggestions") or []:
        if not isinstance(sg, dict) or sg.get("kind") != "untracked_person":
            continue
        f, oid, det = sg.get("frame"), sg.get("obj"), sg.get("detection")
        fr = sg.get("frames") or [f, f]
        hints.append({
            "kind": "untracked_person", "object": oid, "role": None, "frames": list(fr), "reasons": [],
            "message": f"a person nobody tracks is visible on frames {fr[0]}-{fr[-1]} (a lost fighter, or "
                       "someone to track as an extra, e.g. the referee)",
            "fix": f"afv pick {shot_id} --prompt {f}:{oid}:{det} --label NAME to track them "
                   f"(if it is a lost fighter: --prompt {f}:1:{det} for A, {f}:2:{det} for B)",
        })  # fmt: skip
    return hints


def format_review(review: dict[str, Any], shot_id: str, *, paused: bool) -> str:
    """Text for the user: the hints, how to fix each, and how to go on."""
    hints = review.get("hints") or []
    n = len(hints)
    head = f"tracking QC for shot {shot_id}: {n} hint{'s' if n != 1 else ''}"
    if paused:
        head += " - stopped after detect_track"
    elif review.get("acknowledged"):
        head += " (going on: --yes)"
    lines = [head]
    for h in hints[:MAX_LISTED]:
        lines += [f"  - {h['message']}", f"    fix: {h['fix']}"]
    if n > MAX_LISTED:
        lines.append(f"  ... and {n - MAX_LISTED} more (qc.json)")
    if review.get("review_video"):
        lines.append(f"  watch: {review['review_video']}")
    if paused:
        lines.append(
            f"after a fix: afv run {shot_id} (tracks again); if the tracking is fine: the same command with "
            f"--yes, e.g. afv run {shot_id} --yes"
        )
    return "\n".join(lines)


def review_record(shot_dir: Path) -> dict[str, Any]:
    try:
        data = read_json(Path(shot_dir) / REVIEW_FILE)
    except (FileNotFoundError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


HINT_IDENTITY = ("kind", "object", "role", "frames", "reasons")  # what an acknowledgement covers


def hints_hash(hints: list[dict[str, Any]]) -> str:
    """Hash of what the hints say (not their wording or fix text, which may be reworded)."""
    return hash_obj([{k: h.get(k) for k in HINT_IDENTITY} for h in hints])


def acknowledged(shot_dir: Path, stage: str, output_hash: str, hints: list[dict[str, Any]]) -> bool:
    """These hints were acknowledged for this result: same tracking output *and* same hints (a QC-only
    change, e.g. new QC code or thresholds, can raise new hints on unchanged tracks)."""
    rec = review_record(shot_dir).get(stage)
    if not isinstance(rec, dict):
        return False
    return rec.get("output_hash") == output_hash and rec.get("hints_hash") == hints_hash(hints)


def acknowledge(shot_dir: Path, stage: str, output_hash: str, hints: list[dict[str, Any]], when: str) -> None:
    data = review_record(shot_dir)
    data[stage] = {"output_hash": output_hash, "hints_hash": hints_hash(hints), "hints": len(hints),
                   "acknowledged": when, "messages": [h["message"] for h in hints[:MAX_LISTED]]}  # fmt: skip
    atomic_write_json(Path(shot_dir) / REVIEW_FILE, data)
