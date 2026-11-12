# Agent guide: working on `afv`

Read this before you change anything in this repo. It says what the project is, where things
live, how to run and test code, and the rules every change must follow. If this file and the code
disagree, the code is right. If you are the PM (see section 6), fix this file in the same
change; otherwise report the mismatch in your final report.

Other documents, and when to read them:

| Document | Read it when |
|---|---|
| `CLAUDE.md` | Always (loaded automatically). Short current-state summary. |
| `AGENTS.md` (this file) | Always, before your first change. |
| `docs/BUILD_PLAN.md` | Before changing a module: binding conventions, module ownership, shared formats. |
| `docs/ROUND6_NOTES.md` | Before starting new work: what was built last and the open list ("Open for round 7"). |
| `docs/ROUND2_NOTES.md`, `docs/ROUND3_NOTES.md` | When you need the history of a design decision. |
| `WORKFLOW.md` | To learn how the user runs the tool, video to Blender. |
| `blender/afv_addon/README.md` | Before changing the Blender add-on. |
| `docs/P0_FINDINGS.md` | When choosing or changing a model. P0 results override the spec. |
| `docs/SPEC.md` | For requirements (FR / NFR numbers, targets). It is the design document, exported as Markdown, and it is large: search it for the section you need, don't read it whole. Where it and the code disagree, the code is right. |
| `README.md`, `HOLD_MY_HAND.md`, `THIRD_PARTY_LICENSES.md` | What users are told: requirements, install, licences. Keep them true when behaviour changes. |

## 1. What the project is

`afv` is a personal tool for one part-time developer (the owner). It turns fight footage (boxing,
UFC, kung fu films) into 3D animation of the two fighters for the owner's own games. Output goes
to Blender 5.2 for review and cleanup, then to game engines as FBX/GLB/BVH.

Keep the design and infrastructure small: no services, no databases, no new frameworks.

Terms used below:

- **owner**: the user. (`docs/BUILD_PLAN.md` also says "owner" for the agent that owns a module's
  paths in a round; this file says "owned paths" for that.)
- **PM**: the main agent the user talks to. It plans rounds, delegates, integrates and commits.
- **round**: one batch of planned work (rounds 1-4 are done).
- **track**: one agent's assignment within a round (unrelated to the `track` model environment).
- **implementation agent**: an agent the PM gave a track to.
- **shared contract**: the PM-owned files listed in section 6.

## 2. Runtime: Windows host, everything runs in WSL2

- The examples use a repo cloned to `D:\afv` on Windows, which is `/mnt/d/afv` in WSL (distribution
  `Ubuntu`). Use the real path of the checkout you are in. Install: `HOLD_MY_HAND.md` (the scripts in
  `install/`; `install_afv_env.sh` installs afv with its `[dev]` extras: pytest and ruff).
- `afv` and all tests run **inside WSL**, in the env `~/afv/afv-env` (an editable install of the
  repo, so code edits take effect without reinstalling). The CLI is `~/afv/afv-env/bin/afv`; the
  Python is `~/afv/afv-env/bin/python`. `afv` is **not** on the WSL `PATH`; use the full path.
- Model environments (each used only by its stage's worker subprocess):

  | Env | Path (WSL) | Stage |
  |---|---|---|
  | track | `~/afv/track/.venv` | detect_track (RF-DETR + SAM 2.1) |
  | vipe | `~/afv/vipe` (started through `~/afv/vipe_launcher.sh`: conda env `cu128`, then `~/afv/vipe/.venv`) | camera (ViPE) |
  | gemx | `~/afv/gem-x/.venv` | pose3d (GEM-X), SOMA FK |
  | Blender | `/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe` (WSL interop) | export, `afv blender`, Blender tests |

- **Running WSL commands from a Windows shell:** write the commands to a `.sh` file outside the
  repo (your scratchpad), starting with `cd /mnt/d/afv`. Convert its Windows
  path to WSL form (drive `X:\` becomes `/mnt/x/`, backslashes become `/`), then run it:
  - from the Bash tool (Git Bash): `MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/c/Users/.../script.sh`
    (`MSYS_NO_PATHCONV=1` stops Git Bash rewriting the `/mnt/...` path);
  - from PowerShell: `wsl -d Ubuntu -- bash /mnt/c/Users/.../script.sh` (no variable prefix: it is
    a syntax error there).
  - Never put `$VARIABLES` in an inline `wsl ... bash -c '...'`: wsl.exe expands them first.
  - Never start a background job *through* wsl.exe (it outlives the call). To run something long,
    run the `wsl` call itself in the background with your tool's background option.
  - Delete scratch scripts or keep them out of the repo.
- **GPU:** developed on one GPU with 16 GB VRAM (RTX 5080) and about 15 GB RAM in WSL. The GPU lock is `~/afv/.gpu.lock`.
  - `afv` commands and pytest (`gpu`-marked tests, golden tests) take the lock themselves.
    **Never wrap them in `flock`**: their lock waits on a separate file handle with no time limit,
    so an outer `flock` deadlocks them.
  - Use `flock ~/afv/.gpu.lock <command>` only for GPU commands you write yourself (scripts that
    load a model directly).
  - Run these one at a time, never in parallel with each other: golden shots, `afv run` on sample
    shots, `gpu`-marked tests, `afv blender`. Running all golden shots at once was killed for low
    memory.
- Transient CUDA "unknown error" in ViPE has happened once and passed on retry. Retry once before
  debugging.

## 3. The pipeline

Fixed stage order (`STAGE_ORDER` in `src/afv/core/stage.py`):

`ingest → detect_track → camera → pose3d → pair → contacts → smooth → hands_face → export`

| Stage | Kind | Code |
|---|---|---|
| ingest | CPU | stage class `src/afv/media/stage.py`; normalising to the 30 Hz shot timeline in `src/afv/media/normalise.py` |
| detect_track | GPU, env track | `src/afv/models/track.py`, worker `src/afv/models/workers/track_worker.py` |
| camera | GPU, env vipe | `src/afv/models/camera.py`, worker `vipe_worker.py` |
| pose3d | GPU, env gemx | `src/afv/models/gemx.py`, workers `gemx_worker.py`, `soma_worker.py` |
| pair | CPU | `src/afv/motion/pair.py` (both fighters in one world) |
| contacts | CPU, range-capable | `src/afv/motion/contacts.py`, `ik.py`, `ranges.py` |
| smooth | CPU, range-capable | `src/afv/motion/smooth.py`, `pins.py` |
| hands_face | CPU, range-capable | `src/afv/motion/hands_face.py` |
| export | CPU + Blender | stage class `src/afv/export/stage.py`; the rest of `src/afv/export/` |

The stage classes of pair, contacts, smooth and hands_face (including `supports_range`) are in
`src/afv/motion/stages.py`; the files in the table hold their algorithms.

Other code: CLI `src/afv/cli.py`; project, cache, runner, edits, code hashing in `src/afv/core/`;
QC in `src/afv/qc/`; the Blender take package `src/afv/take.py`; the add-on's worker protocol
`src/afv/serve.py`; the add-on itself `blender/afv_addon/`. Retargeting onto the owner's characters
(round 5): the engine `blender/afv_addon/retarget.py` (bpy only, no afv imports; API in its docstring
and in `docs/BUILD_PLAN.md` round 5), the panel `retarget_ui.py`, import/export `retarget_io.py`,
headless `retarget_headless.py`, and `afv retarget` (`src/afv/retarget.py`). The shot browser (round 6):
`blender/afv_addon/browser*.py` (panel AFV > Shots); its backend is shot states in
`src/afv/core/project.py`, the candidate scan in `src/afv/media/shots.py`, and the `shots`, `pick`,
`batch` commands in `cli.py` / `serve.py` (the batch runs detached; `media/batch/`).

Key facts:

- **Shot timeline is 30 Hz.** Every stage result is indexed on frames `0..T-1` with
  `timestamps = start + i/30`. Anything that decodes frames must match the normaliser exactly.
- **Model workers never import `afv`.** They import only their model's libraries and numpy, and
  talk to `afv` through files: a JSON request in, NPZ/JSON out, in the stage's work folder.
- **CPU stages never import torch.**
- **Results are versioned folders:** `shots/<id>/stages/<stage>/v<NNNN>/` with `meta.json`,
  `qc.json` and the result NPZ. Save and load with `save_result` / `load_result` (NPZ, no pickle).
- **Cache key** = inputs + params + implementation + `version` + edit files the stage reads +
  weights hash + code hash. A stage is re-run only when its key changes.
- **Code hash** (`src/afv/core/codehash.py`): each result stores a hash of its stage's code. On a
  code change, CPU stages re-run automatically; model stages show "stale (code)" and keep their
  result unless run with `--code-changes rerun` (or `accept` to record the new code as theirs when
  the change cannot alter output). Still **bump the stage's `version`** when the meaning of its
  output changes.
- **Edits** (user corrections) live in `shots/<id>/edits/<name>.json`:
  - `ids`, `contacts`, `pins`, `hands`: formats in `docs/BUILD_PLAN.md`, "Round 4 assignments >
    Shared formats" (plus round 5's pin keys); validation in `src/afv/core/edits.py`, mirrored by
    `blender/afv_addon/edits.py` (both strict and kept identical: shared vectors in
    `tests/core/edit_vectors.py`). Frames are shot-timeline frames, `end` is exclusive, `role` is
    `"A"` or `"B"`. Writers validate, write atomically and keep unknown keys; `afv run` refuses a bad
    edit file before any stage runs.
  - `fighters`: written by `afv pick` and replaced as a whole; format and validation in
    `src/afv/models/track.py` (`load_fighters_edit`).
- **Range re-runs:** `afv run <shot> --from <stage> --frames a:b`. Only stages with
  `supports_range = True` use the range; see the `Ctx` docstring in `src/afv/core/stage.py`.
- **Exit codes** of `afv run` / `afv export` / `afv batch`: 0 ok, 1 error, 3 waiting. Exit 3 has
  three causes; the run's outcome (printed, and `outcome` in `--json` output) says which:
  - `needs_pick`: no fighters picked yet. Fix: `afv pick`.
  - `needs_review`: detect_track's QC raised tracking hints nobody reviewed. Fix the tracking with
    `afv pick --frame/--prompt/--swap`, or `--yes` to record them as reviewed and go on.
  - `code_changed`: a stage's code was edited while `afv` was running. Fix: run again.

Conventions (binding, from `docs/BUILD_PLAN.md`): Y-up, right-handed, metres; quaternions `xyzw`;
local joint rotations on top of the skeleton's rest rotations; **address joints by name, never by
index**. Skeleton is SOMA (root + 76 joints).

## 4. Data on disk

- **This repository contains no footage, no characters and no data derived from footage.** The
  original author developed against four short fight clips (`Close boxing short.mp4`, `UFC cross.mp4`,
  `Who Am I.mp4`, `ref_walk.mp4`), a 16-minute fight (`Full Fight.mp4`) and three test characters, all
  in a git-ignored `Media_Samples/` folder. They are copyrighted or personal and are not distributed.
  The docs and tests still name them. What that means for you:
  - Tests that need them **skip** when they are missing (real-footage fixtures, the retarget character
    tests, tests on the sample projects). Synthetic tests always run.
  - To get equivalent coverage, put your own short two-fighter clips in `Media_Samples/` (or point
    `AFV_MEDIA_SAMPLES` at a folder), make projects from them, and regenerate the fixtures with
    `tests/motion/fixtures/make_real_fixtures.py` and `tests/models/fixtures/make_gemx_fixtures.py`
    (read their docstrings; they expect the original clip and project names, so adapt the names).
  - The golden test's expected values (`tests/golden/golden_shots.json`) belong to those clips: with
    your own clips, regenerate them (`AFV_GOLDEN_UPDATE=1`) after adapting `SHOTS` in
    `tests/golden/test_golden.py`.
  - **Never commit clips, frames, renders or data derived from footage you don't own.**
- `projects/` (git-ignored): working projects. The author's were `close_boxing`, `ufc_cross`,
  `who_am_i` (shot `001` each), `ref_walk` (shots `001`, `002`), all made from the sample clips, and
  `full_fight`. `projects/_golden/` holds the golden-test projects; don't edit it by hand.
- Project layout: `projects/<name>/project.json`; per shot `shots/<id>/` with `shot.json`
  (interval, settings), `edits/`, `stages/`, `exports/`, `review.json` (tracking hints marked
  reviewed), `take/` (the Blender package and `take.blend`), `.addon/history.json` (the add-on's
  fix history).
- `dist/` (git-ignored): the built Blender extension zip.

Do not delete or overwrite anything under `projects/` except by running `afv` commands.

To re-run your sample shots after a change that affects results (PM only, one shot at a time), for
each project folder `P` in `/mnt/d/afv/projects/` and each shot id `S` in it:

```bash
~/afv/afv-env/bin/afv run S -p P --code-changes rerun   # model stages re-run if their code changed
~/afv/afv-env/bin/afv blender S -p P --check
```

- Use `--code-changes accept` instead of `rerun` only when the model-stage code change cannot
  alter output (a pure refactor); `accept` keeps the old results and marks them current.
- If `afv run` exits 3 with `needs_review`, report the hints. Add `--yes` only if the user has
  already confirmed those hints are false (the author's `close_boxing` shot raised a false hint and
  was always run with `--yes`).

## 5. Tests and checks

Run from WSL (through a script file, see section 2):

```bash
cd /mnt/d/afv
~/afv/afv-env/bin/python -m pytest -q -p no:cacheprovider          # full suite, ~9-10 min
~/afv/afv-env/bin/python -m pytest -q -p no:cacheprovider tests/motion   # one module
~/afv/afv-env/bin/ruff check src tests blender                      # lint, must be clean
```

- Baseline after round 6, on the author's machine with all the test assets: **2074 passed, 15
  skipped** (~15 min), ruff clean. On a checkout without the assets: **1872 passed, 156 skipped** (~8 min; every skip names what it needs). Run the
  suite once before you change anything and use your own numbers as the baseline. A change that
  lowers the pass count, or adds skips, needs a reason in your report.
- Markers: `gpu` (opt-in with `AFV_GPU_TESTS=1`, holds the GPU lock), `blender` (needs headless
  Blender), `slow`. Mark new tests accordingly. Windowed Blender tests (real mouse/key events with
  `--enable-event-simulate`, e.g. `tests/blender/test_browser_windowed.py`) run only with
  `AFV_WINDOWED_TESTS=1`; they open Blender windows on the desktop and close them themselves.
- Where tests go: `src/afv/<m>/` → `tests/<m>/`; `src/afv/serve.py` → `tests/serve/`;
  `src/afv/take.py` and `blender/afv_addon/` → `tests/blender/`; a `cli.py` command → the test
  folder of the module the command drives.

**Golden regression** (`tests/golden/`, GPU, minutes per shot): runs three sample shots end to end
and compares summary values with `tests/golden/golden_shots.json`.

```bash
cd /mnt/d/afv
AFV_GPU_TESTS=1 ~/afv/afv-env/bin/python -m pytest tests/golden -q -p no:cacheprovider -k ufc_cross
```

- Run it if you changed any non-test file under `src/afv/media/`, `src/afv/models/`,
  `src/afv/motion/`, `src/afv/export/`, `src/afv/qc/`, or `src/afv/core/{runner,cache,types}.py`.
  Implementation agents may run it (the tests take the GPU lock, so they wait for other agents).
  Run shots one at a time: `-k close_boxing`, `-k ufc_cross`, `-k who_am_i`.
- If a key fails, including because your change was meant to alter results: don't update the
  values or widen a tolerance. Report each failing key with its expected and actual value; the PM
  decides.
- Environment variables: `AFV_GOLDEN_UPDATE=1` rewrites the expected values (PM only);
  `AFV_GOLDEN_DUMP=<dir>` also writes this run's values to `<dir>/<shot>.json`;
  `AFV_GOLDEN_RESUME=1` continues an interrupted run; `AFV_GOLDEN_DIR` moves the golden projects
  (default `projects/_golden`).
- **Only the PM regenerates golden values**, and only after the change is reviewed. Review the
  diff of `golden_shots.json` before committing it.
- **Tolerance policy:** widen a tolerance in `tests/golden/golden.py` only from a measured repeat
  (run the same code several times and use the observed spread), and write the observed values in
  a comment next to it. Never widen a tolerance to make a real regression pass.
- The golden test needs the sample clips (`Media_Samples/`, or `AFV_MEDIA_SAMPLES`); with
  `AFV_GPU_TESTS=1` a missing clip fails, it never skips.
- Run-to-run noise (round 5, measured): `smooth.*.root_path_m` varies up to 0.043 m (residual
  plant-edge flips; tolerance 0.06 m), `pair.distance_median_m` up to 0.031 m (GEM-X depth drift;
  tolerance 0.045 m), `contacts.*.contact_frames` up to ~20 frames on ufc_cross A. Details in
  `tests/golden/golden.py` and `docs/ROUND5_NOTES.md`.

## 6. How work is organised

Which rules apply to you:

- If your prompt assigns you a track from `docs/BUILD_PLAN.md`, you are an implementation agent:
  follow "Rules for implementation agents".
- Otherwise you are working directly for the user, and you are the PM: you may edit any file,
  but change the shared contract only when the task needs it, and say so in your reply.
- "Rules for everyone" apply to both.

Work is done in **rounds**:

1. The PM writes the round's assignments into `docs/BUILD_PLAN.md` ("Round N assignments"): one
   track per agent, each with the paths it owns, its goal and its acceptance criteria.
2. Implementation agents work in parallel, **only in their owned paths**.
3. Each track gets an independent strict review by a separate agent that has not seen the
   implementer's reasoning; findings are fixed; then a second review pass. **Two review passes per
   track, no more** (owner decision after round 5, whose 3-5 passes per track cost ~4M tokens): what
   the second pass still finds goes on the round's open list, unless the PM judges it a blocker (then
   the PM gets it fixed, checks the diff itself and reports it).
4. The PM integrates, runs the full suite and golden tests, re-runs the sample shots, writes
   `docs/ROUND<N>_NOTES.md` (results and the open list), updates `CLAUDE.md`, and commits when
   the user asks (the user has asked for a commit at the end of each round so far).

Rules for implementation agents:

- Work only in your owned paths. **Do not commit**; the PM commits.
- **Do not edit the shared contract** (PM-owned): `src/afv/core/types.py`, `src/afv/core/stage.py`,
  `src/afv/core/skeletons.py`, `src/afv/motion/rotations.py`, `src/afv/motion/fk.py`,
  `pyproject.toml`, `CLAUDE.md`, `AGENTS.md`, `docs/BUILD_PLAN.md`, and the "Shared formats" of the
  current round. If you need a change, put the exact diff you want in your report.
- No new dependencies without listing them in your report. Available in the afv env: numpy,
  scipy, pydantic, typer, av, pytest (model envs have their own).
- Python 3.11+ style with type hints; small modules; line length 110; ruff clean.
- Don't regenerate golden files.
- Final report must contain: files created and changed; public API; how you ran the tests and the
  result (counts); deviations from the assignment; open questions; shared-contract changes you
  need.

Rules for everyone:

- Commit only when the user asks, PM included. End commit messages with the attribution line given by the
  session.
- When behaviour visible to the user changes, update `WORKFLOW.md`,
  `blender/afv_addon/README.md` or `--help` text in the same change.
- Report results honestly: failing tests, widened tolerances and skipped steps are stated, with
  numbers.

## 7. Blender add-on

- Source: `blender/afv_addon/` (a Blender 5.2 extension, id `afv`, manifest
  `blender_manifest.toml`). The add-on calls `afv` in WSL through one long-lived `afv serve`
  worker (protocol in the docstring of `src/afv/serve.py`).
- Build the zip (Windows PowerShell):
  `& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --command extension build --source-dir D:\afv\blender\afv_addon --output-dir D:\afv\dist`
  Use absolute paths: with a relative `--source-dir` the build fails unless the shell is in the
  repo root.
- The user reviews every result in Blender. When you change anything the user sees there, verify
  it in Blender (headless `afv blender <shot> --check`, or Blender tests), not only in unit tests.
- After add-on code changes the user must rebuild and reinstall the zip; say so in your report.

## 8. Open work

The current open list is "Open for round 7" in `docs/ROUND6_NOTES.md`. Start a new round only
when the user asks for it.
