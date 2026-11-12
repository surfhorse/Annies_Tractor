# afv: video-to-3D animation tool

A tool that turns fight footage (boxing, MMA, kung fu films) into 3D animation of the two fighters, for
cleanup in Blender and export to game engines. It was built by one part-time developer for their own
games, so the design and infrastructure are deliberately small. This repository is the developer
edition of the public release: the tool, its tests and the development notes, for people who want to
change it.

**Agents: read `AGENTS.md` before changing anything** (runtime, pipeline, tests, rules, how rounds work).

## Spec
The design document is `docs/SPEC.md` (a Markdown export; large: search it for the section you need).
Main sections: Goals and scope, Requirements, Architecture (a 10-stage pipeline plus a Blender fix
loop), Technology (the model used at each stage), Data model and CLI, Cleanup workflow, Development
plan (phases), Testing, Risks, Design review. Phase 0 changed parts of it: `docs/P0_FINDINGS.md`
overrides the spec, and the code overrides both.

## Current state (2026-11-11)
- Phase 0 is complete: see `docs/P0_FINDINGS.md` (verdict: GEM-X passes) and `THIRD_PARTY_LICENSES.md`.
- The architecture, conventions, module ownership and shared formats are in `docs/BUILD_PLAN.md`. Read
  it before changing code.
- Rounds 1–6 are done: `afv run` works end to end. Results per round: `docs/ROUND2_NOTES.md` …
  `docs/ROUND6_NOTES.md`. The open list is "Open for round 7" in `docs/ROUND6_NOTES.md`.
- First on that list: **a take imported inside the shot browser's own .blend doesn't play** (it plays
  in a new file). Next: engine-ready export (root motion, engine presets) for retargeted characters.
- Review and cleanup happen in **Blender 5.2**: the AFV extension is in `blender/afv_addon/` (install
  notes and tools in its README). It talks to one `afv serve` worker; the serve protocol is in
  `src/afv/serve.py`, the edit file formats in `docs/BUILD_PLAN.md` and `src/afv/core/edits.py`.
- Shot browser (AFV > Shots): candidate shots from cut detection, one GPU scan of all candidates,
  queue / skip / trim, pick fighters by clicking, a detached overnight `afv batch`.
- Retargeting onto your own characters: AFV > Retarget and `afv retarget` (engine:
  `blender/afv_addon/retarget.py`; bone map files `*.afvmap.json`).
- Stage code is hashed into each result (`afv.core.codehash`): CPU stages re-run on a code change, model
  stages show "stale (code)" (`--code-changes rerun` or `--from`). Still bump `version` when output
  *semantics* change.
- `afv run` / `afv export` pause (exit 3) on tracking QC hints until fixed with `afv pick` or confirmed
  with `--yes`.
- `afv` runs inside WSL2: `~/afv/afv-env/bin/python`. Model environments: `~/afv/gem-x`, `~/afv/track`,
  `~/afv/vipe` (installers in `install/`; guided setup in `HOLD_MY_HAND.md`).
- From Windows, run WSL work via a script file: `MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/d/.../script.sh`.
- **No footage, characters or footage-derived data are in this repository.** The author's test clips
  and characters are named throughout the docs and tests; tests that need them skip (AGENTS.md
  section 4 says how to use your own).
- Reviews of delegated work are capped at two passes per track (AGENTS.md section 6).
- Licences: afv's code is Apache-2.0, everything under `blender/` is GPL-3.0-or-later; the models keep
  their own licences (`THIRD_PARTY_LICENSES.md`).
