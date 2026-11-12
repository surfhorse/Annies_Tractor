# Build plan: round 1 (footage → engine-ready animation from the CLI)

Read with `CLAUDE.md`, `docs/SPEC.md` (the spec) and `docs/P0_FINDINGS.md`. (The `spike/` scripts this
plan mentions were Phase 0 experiments; they are not in this repository. The install scripts are in
`install/`, the licence register is `THIRD_PARTY_LICENSES.md`.) **P0 results override
the spec** where they differ; the differences are listed under "What P0 changed".

## Goal of this round

A working command-line path for one shot: video → shot → pick fighters → track → camera → GEM-X per
fighter → one shared world → contacts/foot-skate fix → strike-aware smoothing → retime → BVH + FBX/GLB
per fighter on a humanoid rig. That is the spec's P1 + P2 minus the Blender add-on (P3), fingers and
range re-runs.

## What P0 changed (binding)

1. **Tracking** = RF-DETR boxes on a prompt frame + user's pick + SAM 2.1 masklets
   (`spike/track_fighters.py`). GEM-X's own YOLOX/ByteTrack is bypassed by supplying boxes
   (`spike/gemx_fighter.py`, `AFV_BOXES_NPZ`).
2. **Camera** is solved once per shot with ViPE and passed to GEM-X for both fighters in `provided`
   mode (`spike/make_provided_camera.py`). `background` and `static` remain options.
3. **No separate kp2d stage**: GEM-X runs its matched ViTPose inside pose3d. New **pair** stage
   after pose3d. Order: `ingest, detect_track, camera, pose3d, pair, contacts, smooth, hands_face, export`.
4. **Shot timeline is 30 Hz**: GEM-X normalises each shot interval with
   `ffmpeg -ss <start> -i src -t <dur> -map 0:v:0 -vf "fps=30,pad=ceil(iw/2)*2:ceil(ih/2)*2"`, where the
   default duration is OpenCV frame count / fps. Every stage result of a shot is indexed on that
   timeline (`frames` = 0..T-1, `timestamps` = start + i/30). Anything that decodes frames must
   reproduce this exactly; frame counts must match GEM-X's.
5. **Skeleton** is SOMA: GEM-X outputs root + 76 local rotations (axis-angle), identity and scale
   coefficients; joint positions come from `py-soma-x` 0.3.3 FK. The provisional skeleton in
   `src/afv/core/skeletons.py` is a stand-in until the real one is extracted (module C).
6. **Contacts**: GEM-X gives 6 contact logits: left/right ankle, foot, wrist. Its own IK leaves
   12–65 cm/s foot skate; our contact stage must reach ≤ 3 cm/s (NFR-3).
7. **Two fighters' depth** is unreliable with telephoto broadcast lenses (jumps up to 1.9 m/frame);
   the pair stage fixes it.

## Runtime layout

`afv` runs **inside WSL2 Ubuntu**, never through `wsl.exe` from Windows at run time.

| Environment | Path (WSL) | Used for |
| --- | --- | --- |
| afv | `~/afv/afv-env` (editable install of `/mnt/d/afv`) | CLI, orchestration, CPU stages, tests |
| gemx | `~/afv/gem-x/.venv` (+ `~/afv/gem-x/weights`, `assets/soma`) | pose3d, SOMA FK |
| track | `~/afv/track/.venv` (+ `~/afv/track/weights`, SAM 2 repo at `~/afv/track/sam2`) | detect_track |
| vipe | `~/afv/vipe_launcher.sh` (conda + `~/afv/vipe/.venv`) | camera |
| Blender | `/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe` via WSL interop | export |

Heavy stages run as **subprocesses in their model environment**, using standalone worker scripts in
`src/afv/models/workers/` that import only their model's libraries plus numpy (never `afv`). They
talk to `afv` through files only: a JSON request in, NPZ/JSON results out, in the stage's temp work
folder. The `afv` side converts worker output into the typed results in `afv.core.types`.

Project folders live on the Windows drive (e.g. `/mnt/d/...`) so Blender can open them later.

## Shared contract (owned by the PM; request changes, don't make them)

`src/afv/core/types.py`, `src/afv/core/stage.py`, `src/afv/core/skeletons.py`,
`src/afv/motion/rotations.py`, `src/afv/motion/fk.py`, `pyproject.toml`, `CLAUDE.md`, this file.
If you need a change (a field, a dependency), say so in your final report with the exact diff you want.

Conventions: Y-up right-handed metres; quaternions xyzw; local joint rotations applied on top of the
skeleton's rest rotations (see `fk.py`); address joints by **name**, never index. Results are saved
with `save_result`/`load_result` (NPZ, no pickle). `ActorMotion.rest_offsets` holds each fighter's own
bone offsets (from SOMA identity/scale).

## Modules and owners

Each agent owns only its listed paths plus its own `tests/<module>/` folder.

### A. Pipeline core and CLI: `src/afv/core/{project,cache,runner,config,provenance}.py`, `src/afv/cli.py`, `src/afv/models/envs.py`
- Project folder: `<project>/project.json` (source video path, sha256, probe info), `shots/<id>/`
  with `shot.json` (start, duration, speed factor, notes), `stages/<stage>/v<NNNN>/result.npz + meta.json`
  (versioned so an in-use folder is never renamed on the Windows drive; find the current one with `stage_dir()`),
  `edits/` (fighters.json etc.), `exports/`. Safe writes (temp dir + rename).
- Cache key = hash(stage name, stage version, weight hash, params, upstream keys); skip when unchanged;
  `--from <stage>` forces from there on. Edits that a stage reads are part of its key.
- Runner: resolves the stage order, loads inputs, calls stages, writes meta (runtime, params, QC
  summary, licences), keeps earlier results on failure, readable errors.
- `envs.py`: env registry (paths above, overridable in `~/.config/afv/envs.toml`), `run_worker(env,
  script, request: dict, work_dir) -> dict` subprocess bridge with streaming log, timeout, and a GPU
  lock (`flock ~/afv/.gpu.lock`) so only one GPU job runs at a time.
- CLI (Typer): `afv init <video> --project DIR`, `afv shots detect|add|list`, `afv pick <shot>`
  (prints/saves the numbered prompt image path), `afv run <shot> [--from S] [--camera vipe|background|static]`,
  `afv batch`, `afv export <shot> --preset P`, `afv models --check`, `afv provenance`. Progress as human
  text or `--json` lines.
- Register stage implementations by importing `afv.stages` modules lazily (a fixed list in the runner).
- Tests: fake stages proving caching, invalidation, failure handling, safe writes, CLI smoke.

### B. Media: `src/afv/media/`
- `probe(path)` → Clip info (codec, size, rotation, fps, frame count, VFR flag, duration, sha256);
  uses PyAV; OpenCV-equivalent frame count/fps where GEM-X's normalisation depends on it (document
  any difference; GEM-X uses `cv2.CAP_PROP_FRAME_COUNT / CAP_PROP_FPS`).
- `normalise_shot(src, start, duration, out)` reproducing GEM-X's ffmpeg command exactly (frames
  or mp4), and `shot_frame_count` that predicts GEM-X's T.
- 720p proxy for Blender and for cut detection; **cut detection** (HSV histogram + adaptive
  threshold + minimum shot length; returns cut times and confidence); shot list from cuts.
- Letterbox detection and a static-overlay mask (pixels that stay constant while the camera moves,
  e.g. scoreboards) → per-shot mask PNG.
- `ingest` stage (CPU). ffmpeg binary: `~/.local/bin/ffmpeg` in WSL, configurable.
- Tests generate synthetic videos (colour cuts, letterbox bars, a static logo over moving noise, VFR).

### C. GEM-X + SOMA + camera adapters: `src/afv/models/{gemx,camera}.py`, `src/afv/models/workers/{gemx_worker,soma_worker,vipe_worker}.py`, `src/afv/core/data/skeletons/soma77.json`
- **Extract the real SOMA skeleton** from py-soma-x 0.3.3 + `assets/soma` (joint names, parents,
  rest offsets for the neutral identity, rest rotations if any) into `soma77.json`; document which
  joints are body vs hands vs face and which joints the 6 contact channels refer to.
- **pose3d worker**: runs GEM-X motion-video for one fighter with supplied boxes and a provided
  camera (port `spike/gemx_fighter.py` logic as a library call, not a CLI patch where possible),
  then converts SOMA params → per-joint local quaternions on the soma77 skeleton, root translation
  in world, per-fighter rest offsets (identity/scale averaged over the shot = locked bone lengths),
  contact probabilities (sigmoid of logits), per-joint confidence if available.
  **Verify**: our `afv.motion.fk` on the converted data reproduces soma's own joint positions
  ≤ 1 mm on the P0 outputs in `~/afv/runs/p0_shared/*/rank*/motion.pt`.
- Determine GEM-X's world axes and camera conventions (R_w2c, OpenCV vs OpenGL) and convert to afv's
  Y-up world and `CameraTrack` convention; test the conversion by reprojecting joints.
- **camera worker**: ViPE once per shot → `CameraTrack` (+ the `provided` config for GEM-X);
  static and background modes.
- `pose3d` and `camera` stages. Use the GPU lock. Keep GPU tests few and short.

### D. Tracking: `src/afv/models/track.py`, `src/afv/models/workers/track_worker.py`
- Port `spike/track_fighters.py`: prompt-frame detection with numbered image, pick (fighters +
  extra people such as the referee), SAM 2.1 propagation both ways, QC (empty frames, unmatched
  runs), review video. Add **re-prompting**: a list of (frame, detection index, object id) prompts so
  a referee entering later or a drifted fighter can be corrected; and an automatic check that flags
  frames where a fighter mask's box matches no detection for > N frames.
- Output a `Tracks` result (boxes for every tracked person, `fighters` = the two ids) on the 30 Hz
  shot timeline; frame counts must equal GEM-X's.
- `detect_track` stage; picks stored as an edit (`edits/fighters.json`).

### E. Motion processing (CPU, numpy/scipy only): `src/afv/motion/{contacts,smooth,retime,pair}.py`, `src/afv/qc/`
- **contacts**: start from GEM-X contact probabilities; hysteresis + minimum duration; foot plant
  detection backed by height/velocity; floor plane fit; skate removal by pinning planted foot
  targets and solving two-bone leg IK (thigh/calf/foot by name) with root adjustment, blended at
  plant boundaries; no ground penetration. Painted contacts (`edits/contacts.json`) override.
  Target: foot speed in contact ≤ 3 cm/s, penetration ≤ 1.5 cm, on the P0 data.
- **smooth**: sliding-window optimiser on rotations (in a local tangent space) and root, with
  velocity/acceleration terms, strength global and per region; detect strike peaks (wrist/ankle
  speed) and ease smoothing there (keep ≥ 80% peak speed, NFR-4); contacts held. One-Euro fallback.
- **retime**: speed factor and resampling from the 30 Hz shot timeline to the export fps (slerp,
  root interpolation, contacts by threshold).
- **pair**: put both fighters on one ground plane, smooth each fighter's depth along the camera axis
  (keep the projection onto the image consistent), and push apart interpenetrating bodies.
- **qc**: per-frame metrics (foot speed in contact, penetration, acceleration, strike peak speed,
  reprojection error given a camera) and the spec's metrics against hand labels.
- Stages `pair`, `contacts`, `smooth`. Work on any `Skeleton` by joint names; tests use synthetic
  motion. Never import torch.

### F. Export: `src/afv/export/`, `blender/`
- Canonical humanoid export rig (Mixamo names, configurable prefix, T-pose rest, optional root bone)
  and a name-based mapping from the source skeleton (soma77 when available; build against the
  provisional skeleton and keep the mapping table as data).
- Retarget by rest-relative rotation transfer; extra spine joints spread; bone lengths from the
  fighter's own rest offsets so positions are preserved.
- Root-motion modes and engine presets exactly as the spec's table (Unity Humanoid/Generic,
  Unreal, Godot, Blender) with the full exporter argument sets the spec lists.
- Own **BVH writer**; **FBX/GLB** via headless Blender 5.2 (script in `blender/`, run from WSL via
  the Windows exe with `wslpath -w` paths); attacker and defender as separate files with one origin.
- **Round-trip test**: re-import in Blender, sample joint positions every frame, ≤ 1 mm, exact frame
  count and fps, no axis flips, no extra root node.
- `export` stage.

## Round 2 (after review): integration

Wire A–F into `afv run` end to end on the four Media Samples clips, the soma77 export mapping,
fist preset (hands_face), golden-file regression on 3 shots, and the review loop again.

## Rules for every agent

- Work only in your owned paths. Do not commit. Do not edit the shared contract; ask in your report.
- Python ≥ 3.11 style, type hints, small modules; `ruff check` clean (config in `pyproject.toml`).
- Tests with pytest in `tests/<module>/`; they must pass in the WSL afv env:
  `~/afv/afv-env/bin/python -m pytest tests/<module>`. Mark GPU tests `@pytest.mark.gpu`, Blender
  tests `@pytest.mark.blender`, slow ones `@pytest.mark.slow`.
- No new dependencies without listing them in your report (available: numpy, scipy, pydantic,
  typer, av, pytest; model envs have their own).
- **WSL from the Windows shell**: put commands in a script file and run
  `MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/d/.../script.sh`. Never put `$VARS` in an inline
  `wsl ... bash -c '...'` (wsl.exe expands them). Never start background jobs through `wsl.exe`
  (they outlive the call). Keep scratch scripts out of the repo or delete them.
- GPU work: hold `flock ~/afv/.gpu.lock` around it; the machine has 16 GB VRAM and ~15 GB WSL RAM,
  and other agents run in parallel.
- Source footage is copyrighted: never copy clips or frames into the repo (tests use synthetic video).
- Final report: files created, public API, how to run tests and their result, deviations from this
  plan, open questions, and any shared-contract changes you need.

## Round 2 assignments

Inputs: `docs/ROUND2_NOTES.md` (every open item from round 1). Same agent rules as above.

### I. Integration (owner: integration agent; may edit any module's wiring, not its algorithms)
- `afv run` end to end on all four `Media Samples` clips into a project under
  `/mnt/d/afv/projects/` (git-ignored): ingest → … → export FBX/GLB, with the
  ref walk shot split at its cut. Fix every wiring break found, with a test per fix.
- Stage dependencies (ingest as an input), ContactParams carried from contacts to smooth/retime,
  retime wired into export (`ExportParams.fps`, shot `speed_factor`, `afv export --fps`).
- Re-prompt CLI: `afv pick <shot> --frame F` and `--prompt F:OBJ:DET [--label]`.
- Golden-file regression on 3 shots (opt-in GPU/slow marker; stored hashes/tolerances, not footage).
- Measure GEM-X metacarpal rotations on the real runs and report.

### T. Telephoto root investigation (owner: telephoto agent; owns `src/afv/models/camera.py`,
`src/afv/models/workers/vipe_worker.py`, and the root/world code in `src/afv/models/gemx.py`)
- Why the camera-based root moves ~3× GEM-X's own rollout on ref walk / Who Am I, and the hips
  reprojection peaks (13–22 px vs 15 px). Re-run ref walk as two shots first. Fix the cause if it is
  in our code; otherwise make the pipeline degrade cleanly (e.g. fall back to `root_prior`) and flag it.

## Round 3 assignments

Target engine for now: **Blender** (the user imports the exports into Blender). PM check before
the round: all 30 real exports (5 shots x 2 fighters x FBX/GLB/BVH) import headless in Blender 5.2
with 65 bones and exact frame counts, and the three formats agree to 0.02 mm per joint (bones
matched by name; bone order differs between formats). Spec sections this round builds:
"Cleanup workflow in Blender", "What the add-on adds to Blender", "Blender add-on and worker", P3.

Same agent rules as above, plus:
- **Any change to a stage's output bumps its `version`** (until C lands source hashing).
- Don't regenerate golden files or fixtures; report the diffs you expect. The PM regenerates at
  integration.
- `src/afv/cli.py` is shared by BL and C: add only your own commands/options, keep diffs small,
  re-read the file right before each edit.

### BL. Blender add-on, slice 1 (owner: `blender/afv_addon/`, `src/afv/take.py` (new), `afv take`
and `afv blender` commands in `cli.py`, `tests/blender/`)
- **Take package** (WSL side, `afv take <shot>`): writes `<shot>/take/` with plain JSON + NPZ that
  Blender's Python (numpy only, no pydantic/afv) can read: per fighter the canonical export rig
  (same bones/rest as the exports) and its per-frame local rotations + root translation, the
  solved camera (position, rotation, focal per frame, sensor/resolution) **in the same world as the
  bodies** (pair/contacts world_shift and floor applied), per-frame QC spans for the confidence
  strip, plant masks, shot fps/frame range/speed_factor, and the proxy video path as a Windows path.
- **Scene builder** (Blender side, one module used by both the add-on and headless):
  both fighters as armatures with a simple proxy mesh, solved motion as the base action on an NLA
  strip plus an empty additive track above it for the user's tweaks; animated camera; proxy video
  as camera background; scene fps and frame range; confidence strip as timeline markers.
- **`afv blender <shot> [--out take.blend]`**: build the .blend headless (Blender on Windows via
  interop, like export), so a take can be opened with File > Open without installing anything.
- **Extension** `blender/afv_addon/` (`blender_manifest.toml`, Blender 5.2): sidebar panel "AFV"
  with Import take (pick a shot folder), Next/Previous issue, Re-import (replaces only the base
  NLA action), Export (preset picker; runs `afv export` through WSL on a background thread, results
  applied with `bpy.app.timers`, UI never blocks). Document install (`blender --command extension
  build` / Install from Disk) in `blender/afv_addon/README.md`. No `afv serve`, contact painting,
  pins or range re-runs yet (slice 2).
- Tests: synthetic take end to end; take joints vs the exported FBX for the same shot <= 1 mm;
  overlay alignment: joints projected through the Blender camera vs the pipeline's own 2D projection
  of the same 3D joints <= 1 px at sampled frames on a real shot (blender + slow markers);
  re-import keeps the additive track. Report the alignment numbers for all 5 shots.

### C. Core (owner: `src/afv/core/*`, `src/afv/models/envs.py`, your commands/options in `cli.py`,
`tests/core/`)
- **Stage source in the cache key**: a code change to a stage (its module and the afv modules it
  imports) must make its results stale without a manual version bump; comment/docstring/format-only
  edits must not. Decide how GPU stages are treated (a refactor must not silently force a 10-minute
  GEM-X re-run: e.g. show "stale (code)" in status and require `--from`, or hash only declared
  modules); explain the choice. `afv status` shows why a result is stale.
- **Tracking QC pause**: `afv run` stops after detect_track when its QC raises tracking hints
  (likely ID swap, lost fighter), prints them and how to fix (`afv pick`), unless `--yes`.
- Any runner support BL needs for `afv take` (ask via report if it touches the contract).

### M. Motion quality (owner: `src/afv/motion/*`, `src/afv/qc/*` except `qc/camera.py`,
`tests/motion/`, `tests/qc/`)
- Hover metric (max 3–4 cm on real shots) vs contacts `max_plant_hover` 2 cm: find which sole
  reference differs and make the metric and the solver agree; fix whichever is wrong.
- Who Am I fighter A keeps 25% of GEM-X plants: find why (GEM-X contacts, our drops, pin misses)
  and raise it if the cause is ours, without reintroducing floating or sliding plants.
- close_boxing B knockdown: verify the fighters' separation and floor contact through the
  knockdown are physically plausible; fix or flag.
- Strike thresholds (smooth's strike detection) checked against the real shots' strikes; report.

### Z. Camera and GEM-X (owner: `src/afv/models/camera.py`, `src/afv/models/gemx.py`,
`src/afv/models/workers/*`, `src/afv/qc/camera.py`, `tests/models/`)
- **Zoom**: close_boxing (~1.5x) and ufc_cross (~1.2x) zoom is flagged, not corrected. Make the
  per-frame focal length reach GEM-X and the CameraTrack, and measure the effect on reprojection
  and root motion.
- **Lateral root check** windowed within a shot rather than per shot.
- **GEM-X raw-output cache** so the gravity re-level pass doesn't repeat the 40–60 s GPU run when
  the network input is unchanged (check what actually changes on re-level first).

## Round 4 assignments: Blender add-on slice 2 (spec P3 cleanup tools)

Inputs: `docs/ROUND3_NOTES.md` ("Open for round 4"); spec sections "Cleanup workflow in Blender",
"What the add-on adds to Blender", "Blender add-on and worker". Spec targets: a smoothing change
previews in Blender in <= 5 s for 10 s of motion; a contacts/smooth re-run of a 2 s range <= 30 s.
Same agent rules as rounds 1-3 (code hashing now covers CPU-stage changes; still bump `version` when
output semantics change). Don't regenerate golden files (PM does).

### Shared formats (PM-owned contract; request changes, don't make them)
All edits live in `shots/<id>/edits/<name>.json`; frames are shot-timeline frame numbers, `end`
exclusive; `role` is "A"/"B". Writers write atomically and keep unknown keys.
- `contacts.json` (exists): `{"spans": [{"role", "start", "end", "state": "plant"|"free",
  "foot": "left"|"right"} | {..., "channels": [...]}]}`. Read by contacts.
- `pins.json` (new): `{"pins": [{"role", "frame", "radius": N frames (default 8), "preset": export
  preset of the take, "bones": {"<canonical export bone>": [x, y, z, w]}, "root": [x, y, z] | null}]}`.
  Rotations are export-rig **local rotations** (afv convention: Y-up, xyzw, same as the take
  package's `rot` arrays); `root` is the Hips world position in afv world metres. Read by smooth.
- `hands.json` (new): `{"spans": [{"role", "start", "end", "side": "left"|"right"|"both"}]}`
  "hands off": blend to the relaxed hand pose. Read by hands_face.
- `ids.json` (new): `{"swaps": [{"start", "end": f | null, "objects": [o1, o2]}]}` swap two tracked
  objects' identities from `start` (to `end`, or the shot end). Applied by detect_track as a
  post-process of its stored tracks (no re-tracking, no GPU).
- Smoothing sliders are stage params: `smooth.strength`, `smooth.region_strength.<region>`
  (torso/head/arms/hands/legs/root), set like `afv run --set` and saved to the shot.
- **Range re-runs**: `afv run <shot> --from <stage> --frames a:b` re-solves only frames [a, b) plus a
  margin, blended into the stage's previous result at the edges, producing a new full-length result
  version. Stages opt in with `supports_range: ClassVar[bool] = True` and receive `ctx.frame_range`
  and `ctx.previous` (their previous Result) ; others run in full. CPU stages contacts, smooth and
  hands_face opt in this round.
- **`afv serve`** (JSON lines on stdin/stdout, one worker per Blender session, any project):
  request `{"id": n, "cmd": ..., "args": {...}}`, cmds: `ping`, `status`, `run` (args as `afv run`:
  project, shot, from, to, frames, sets, yes, code_changes), `take` (as `afv take`), `export`,
  `edit` (`{"project", "shot", "name", "data"}`: validate + write an edit file atomically),
  `cancel` (`{"target_id"}`), `shutdown`. Replies: `{"id": n, "event": "progress"|"log"|
  "tracking_review"|"result"|"error", ...}`; every request ends with exactly one `result` or `error`.
  Events carry the same payloads as the CLI's `--json` events. Requests run one at a time per shot,
  in order; `cancel` stops a running one cleanly (stage work dir dropped, previous result kept).

### S. Serve and range re-runs (owner: `src/afv/core/*`, `src/afv/cli.py`, new `src/afv/serve.py`,
`src/afv/models/track.py` (ids.json only), `tests/core/`, `tests/serve/`)
- `afv serve` per the protocol; keeps afv modules imported; CPU stages in-process; GPU stages through
  the existing worker subprocesses; robust to a client that disappears (exit when stdin closes).
- Range re-runs in the runner (`--frames a:b`, `supports_range`, `ctx.previous`, cache key includes the
  range and the previous result's hash; provenance records it); `afv status` shows range versions.
- `ids.json` swaps in detect_track with QC/hints recomputed after the swap; `afv pick --swap F:O1:O2`.
- Measure: a smooth re-run through serve on ufc_cross (10 s) and a 2 s contacts+smooth range re-run.

### P. Motion edits (owner: `src/afv/motion/*`, `src/afv/qc/*`, `tests/motion/`, `tests/qc/`)
- **Pins** in smooth: convert pins (export-rig local rotations, via `afv.export.retarget`'s rig and
  mapping; read-only use of `afv/export`) to source-skeleton targets; re-optimise +-radius frames so
  the motion passes exactly through the pinned pose with smooth blending; contacts re-pinned after.
- **Hands off** in hands_face (`hands.json`).
- **Range support** (`supports_range`) for contacts, smooth and hands_face: solve [a-margin, b+margin)
  and blend into `ctx.previous` with C1-continuous edges; identical output outside the range.
- Make `region_strength` behave as the spec's per-region sliders (check every region name works and
  is monotone); speed target: smooth on 10 s of motion well under 5 s total incl. stage overhead.

### B2. Add-on slice 2 (owner: `blender/afv_addon/*`, `src/afv/take.py`, `tests/blender/`)
- Talk to one `afv serve` worker (started on first use via WSL, absolute afv path from preferences,
  no console window, restarted if it dies); keep the slice-1 one-shot path as fallback.
- **Contact painting**: select a fighter's foot bone, scrub/select a frame range, press Plant / Free
  -> `contacts.json` -> range re-run of contacts+smooth -> re-import solved action only.
- **Pin pose**: pose bones with Blender tools, press Pin -> `pins.json` (current frame, radius slider)
  -> smooth re-run -> re-import; list/delete pins.
- **Smoothing panel**: global + per-region sliders, Apply -> smooth only; report elapsed time.
- **Re-run range**: frame range + stage picker -> range re-run.
- **Hands off**: mark span -> `hands.json` -> hands_face re-run.
- **Pick fighters / swap IDs**: draw detect_track boxes with ids over the camera view (take package
  gains per-frame boxes), click two boxes to swap from the current frame -> `ids.json` -> re-run.
- **Confidence strip** drawn with the `gpu` module in the timeline/dope sheet (coloured by severity),
  markers kept as fallback; Next/Prev issue unchanged.
- Measure the end-to-end preview times (smoothing Apply, 2 s contact paint) on ufc_cross and report.

## Round 5 assignments: retarget onto the owner's characters, stable plants, strict edits

Inputs: `docs/ROUND4_NOTES.md` ("Open for round 5"). Goal: the owner gets fighter motion onto their own
game characters, from Blender (tweak keys included) and headless; regression results stop flipping run
to run; edit files are validated strictly. Same rules as before (AGENTS.md section 6). Every track gets
a strict independent review, fixes, then a second review.

Test characters (owner assets in git-ignored `Media_Samples/`, may be absent on other machines: tests
that need them skip when the files are missing; tests that must always run build synthetic armatures):
- `Media_Samples/Mixamo Mech/mixamo_mech.fbx`: `mixamorig:`
  names, 65 bones, ~1.1 m tall, arms ~60 deg below horizontal (strong A-pose), 13 skinned mesh parts.
- `Media_Samples/Unity Humanoid Mech/Unity_Humanoid_Mech_rightarm_fixed.fbx`: Unity Humanoid names
  (`Hips`, `Spine`, `Chest`, `UpperChest`, `Neck`, `Head`, `Left_Shoulder`, `Left_UpperArm`,
  `Left_LowerArm`, `Left_Hand`, `Left_UpperLeg`, `Left_LowerLeg`, `Left_Foot`, `Left_Toes`, fingers
  `Left_IndexProximal`..., plus `Neck_Twist_A`, eye bones), 67 bones, ~0.9 m, A-pose, 124 skinned
  parts. Use the `_rightarm_fixed` file; the original export next to it has a broken
  right forearm (kept for reference only).
- `Media_Samples/Girush Young Warrior (CC0)/baseMesh_A1_(human_YW)_by_Girush.blend`: anatomical names
  (`Sacrum` pointing forward, `Sacrum.001` parent of the legs, `Spine1`, `Spine1.001`, `Spine2`,
  `Spine3`, `Spine3.002` neck, `head`, `Clavicle.L`, `Humerus.L` + twist `HumerusRoll.L`,
  `LowerArm.L` + `LowerArm.L.001`, `hand001.L` + fingers `hand002..017.L`, `Femur.L`, `Tibia.L`
  (short knee piece) + `Tibia.L.001`, `Foot.L`, `Foot.L.001`), extras (`right_sword_mount`,
  `genitalia`, `eye.L/R`), centimetre scale (~176 units tall), arms ~25 deg down, ships with its own
  actions. Open it with `--disable-autoexec`.

### Shared formats (PM-owned contract; request changes, don't make them)
- **Bone map file** `*.afvmap.json` (UTF-8 JSON; writers write atomically and keep unknown keys):
  `{"format": "afv_retarget_map", "version": 1, "bones": {"<canonical afv bone>": "<target bone>"},
  "facing": null | "-Y" | "+Y" | "-X" | "+X" | "-Z" | "+Z" (+-Z for Y-up armature data),
  "hips_scale": null | number, "target": "<armature name, informational>"}`. Canonical afv bones are the unprefixed export-rig names (`afv.export.rig`:
  `Hips`, `Spine`, `Spine1`, `Spine2`, `Neck`, `Head`, `LeftShoulder`, `LeftArm`, `LeftForeArm`,
  `LeftHand`, finger bones, `LeftUpLeg`, `LeftLeg`, `LeftFoot`, `LeftToeBase`, ...). A canonical
  bone missing from `bones` is unmapped. `facing` null = detect; otherwise the character's forward in
  the **target armature's own axes** (so a stored map stays valid when the object is rotated);
  `hips_scale` null = ratio of the pelvis anchors' rest heights (midpoint of the two thigh joints)
  above the floor (target / source). The same JSON is kept on the target armature object as the
  custom property `afv_retarget_map` (string), so a saved .blend remembers it.
- **Retarget API** (`blender/afv_addon/retarget.py`; bpy + numpy only, no afv imports, no operators
  or UI in this module):
  - `canonical_bones(source_obj) -> dict[str, str]`: canonical name -> bone name on an afv armature
    (a take armature or an imported afv export, any bone prefix).
  - `guess_map(target_obj) -> dict[str, str]`: canonical -> target bone, from bone names and
    hierarchy (must fully map the three test characters' body bones; fingers best effort).
  - `missing_required(bone_map) -> list[str]`: required canonical bones not mapped.
  - `read_map(path) -> dict`, `write_map(path, data) -> None`, `map_from_object(obj) -> dict | None`,
    `map_to_object(obj, data) -> None` (file format above).
  - `retarget(scene, source_obj, target_obj, bone_map, *, frames=None, lock_feet=True,
    hips_scale=None, facing=None, action_name=None) -> dict`: samples the source's **evaluated** pose
    every frame (so NLA tweak layers and the active action are included), writes a new action with
    one key per frame on the target (all mapped bones, hips location; unmapped bones keep their rest
    pose), assigns it to the target, and returns a report: `{"action", "frames": [first, last],
    "mapped": n, "unmapped": [...], "hips_scale", "facing", "facing_yaw_deg", "max_dir_err_deg",
    "max_twist_err_deg" (with lock_feet both leave out the IK'd legs), "ik_bones": [the target bones
    the foot lock's IK solved: the mapped thigh, shin and foot of each locked leg and the bones between
    them; [] without lock_feet or plant masks], "foot_slide_cm_s": {"left", "right"} (median
    planted-foot speed),
    "penetration_cm" (deepest point of the feet below the floor: toe tip, heel and the foot/toe-weighted
    mesh vertices when the target has skinned meshes), "notes": [...], "seconds"}`. Report lengths are cm
    at the fighter's scale (target length / `hips_scale`). `bone_map` may be a bones dict or a whole map
    dict (its `facing` / `hips_scale` apply when the keywords are None).
    It never deletes or edits existing actions (the new one gets a fake user; on an error no partial
    action is left behind) and does not change the target's rest pose, meshes or object transform. To
    make the new action play it may leave NLA tweak mode and set the action blend to Replace at
    influence 1; both are reported in `notes`. With `lock_feet` the hips may be lowered (at most 25%
    of the leg length, smoothed) where a planted foot is out of reach; reported in `notes`.
  - **Refuse, never guess silently**: `retarget` raises `ValueError` (message says what to fix) for a
    non-uniformly scaled or sheared source/target, a target not standing upright in its rest pose, a
    source not evaluated in the scene's view layer, a map that contradicts the target's hierarchy (a
    mapped bone that is not a descendant of its mapped canonical parent's bone), and a generated Rigify
    rig (retarget onto the metarig or a deform-only game rig instead). Suspected left/right swaps are
    noted.
- Target and source bones are addressed by name only.
- **Edit files (round 4 formats, made explicit; track V)**: validation is strict on both sides (see V).
  `pins.json` pins also carry the optional keys the add-on writes and smooth reads: `"root_rotation":
  [x, y, z, w] | null` (unit quaternion), `"prefix": "<bone prefix>"`, `"root_motion": "<mode>"`;
  `radius` is an int **>= 1** (smooth's reader and the add-on's slider already require it).
  `fighters.json` (written by `afv pick` / serve `edit`) is validated with the same strictness.

### R1. Retarget engine (owner: new `blender/afv_addon/retarget.py`, new `tests/blender/test_retarget*.py`,
new `tests/blender/retarget_*` helper scripts)
- Implement the API above. Method (improve it if you can show why; the acceptance numbers are
  binding): rest-relative world rotation transfer, `W_t(t) = (W_s(t) W_s_rest^-1) A W_t_rest`, where
  `A` is a per-bone **rest alignment** that swings the target bone's rest direction onto the source
  bone's rest direction (shortest arc, so the target keeps its own roll and twist transfers
  relatively). Directions are **joint to mapped child joint**, not Blender head->tail (Girush's
  `Sacrum` points forward); a bone with no mapped child inherits its parent's alignment; the hips get
  only a facing (yaw) alignment. Unmapped bones inside a chain (twist bones, the `Tibia.L` knee piece,
  `Spine1.001`) stay at rest and their children are solved relative to the actual parent pose.
- Hips location: target hips = target rest hips + `hips_scale` x (source hips displacement from its
  rest), in world space, so unit scale (cm rigs) and object transforms are handled.
- `lock_feet`: when the source has plant masks (`afv_plant_<channel>` properties, see
  `builder.py`), keep each planted target foot fixed in world for the span (two-bone IK on the mapped
  thigh/shin, foot orientation kept, blended over a few frames at span edges, knee bending in its
  natural plane), and keep feet above the floor (afv floor = world z 0 on take armatures). Without
  plant masks, skip IK and say so in `notes`.
- Interactive speed: one fighter, 300 frames, ~65 target bones in **<= 10 s**.
- Acceptance (tests; always on synthetic rigs, and on the three test characters x the `ufc_cross`
  and `close_boxing` takes when present):
  1. `lock_feet=False`: every mapped bone's world direction (joint -> mapped child joint) matches the
     source's within **1 deg** on every frame, and twist (rotation about that direction relative to
     the aligned rest) within **2 deg**. Exception (PM decision, round 5 review): `Foot`, `ToeBase`
     and their children keep their own rest pitch to the floor (facing/yaw alignment only), so for
     them the rest-relative rotation `W_t(t) W_t_rest^-1` (after facing) matches the source's
     `W_s(t) W_s_rest^-1` within **1 deg** instead.
  2. `lock_feet=True`: median planted-foot speed **<= 3 cm/s**; penetration (as in the report: toe
     tip, heel, foot mesh) **<= 1.5 cm** on planted frames and never more than 1.5 cm deeper than the
     source's own lowest foot point on other frames; at **each** lock span edge the peak ankle
     acceleration within +-5 frames is at most **max(1.5x the free (unlocked) result's peak in the same
     window, 1800 cm/s^2 at fighter scale)** (the floor covers edges where the free foot is nearly
     still; it is a gentle foot movement, not a visible pop) (on the real takes; synthetic
     plant flags don't stop the synthetic foot, so there the lock blend is bounded by its design limit
     instead); unreachable IK targets are reported in `notes`, never silently dropped.
  3. A tweak key added on the source's "AFV tweaks" track changes the retargeted result accordingly.
  4. `guess_map` maps all required bones of the three test characters and of a synthetic
     Rigify-style rig (`upper_arm.L`, `forearm.L`, `thigh.L`, `shin.L`); it never maps a bone to the
     wrong side or one target bone twice (tested).
  5. Nothing else in the target changes (rest pose, other actions, meshes, transforms).

### R2. Retarget in the add-on and headless (owner: `blender/afv_addon/{ui,ops,__init__}.py`, new
`blender/afv_addon/retarget_ui.py` (or similar), new `blender/afv_addon/retarget_headless.py`, the new
`afv retarget` command in `src/afv/cli.py`, new `src/afv/retarget.py` (WSL-side driver),
`blender/afv_addon/README.md`, `WORKFLOW.md`, new `tests/blender/test_retarget_ui*.py`; starts after R1's
review)
- Add-on sub-panel **Retarget**: pick the target armature (or **Import character**: FBX / GLB /
  append from .blend), **Guess map**, an editable list (canonical bone -> target bone with search),
  **Load / Save map** (`*.afvmap.json`), fighter A/B, **Lock feet** (default on), **Retarget** (one
  undo step; report shown: mapped/unmapped, foot slide, notes), **Export character** (FBX of the
  target armature + its meshes with the retargeted action, using the chosen engine preset's FBX
  arguments from `afv.export.presets`, handed to the add-on, not imported there).
- `afv retarget <shot> --character <fbx|glb|blend> [--map m.afvmap.json] [--role A|B|both]
  [--preset P] [--out DIR] [--no-lock-feet] [--save-map PATH] [--check] [--list-presets] [--json]`
  (`--check`: re-import each FBX and compare with the in-Blender retarget; `--list-presets`: print the
  engine presets as JSON). Serve gains `presets` (`{}` -> result `{"presets": [EnginePreset.to_dict()
  ...]}`, answered at once, no project), so the add-on gets the FBX arguments without importing afv.
  It builds the take headless,
  imports the character, maps (guess or file), retargets, exports `<shot>_<role>_<character>.fbx`
  (default `shots/<id>/exports/retarget/`), prints the report; exits 1 listing the missing bones when
  a required bone is unmapped. Holds the shot lock like `afv blender`.
- Acceptance: the exported FBX re-imported in Blender matches the in-Blender retarget (bone world
  heads <= 1 mm every frame), exact frame count and fps; works for the three test characters and both
  fighters of `ufc_cross`; README and WORKFLOW updated; the add-on zip builds.

### M. Stable plant decisions (owner: `src/afv/motion/*`, `src/afv/qc/*`, `tests/motion/`, `tests/qc/`)
- Golden keys `pair.distance_median_m` and `smooth.*.root_path_m` are bimodal run to run: GPU-level
  noise in pose3d flips discrete foot-plant decisions (round 4: close_boxing A smoothed root path 3.97
  vs 4.07-4.09 m; who_am_i pair distance 2.22 vs 2.245 m). Make those decisions stable (hysteresis,
  margins, minimum span lengths, or deciding on smoothed evidence) without losing quality: the
  contact targets still hold (foot speed in contact <= 3 cm/s, penetration <= 1.5 cm).
- First build a **CPU-only noise harness**: perturb a shot's pose3d result with noise of the measured
  GPU run-to-run size (measure it from repeat runs or existing result versions), run pair -> contacts
  -> smooth N >= 10 times, and report the spread of the golden keys before and after the change.
  Then confirm with real repeats (golden shots `close_boxing` and `who_am_i`, 3 runs each, one at a
  time).
- Also fix smooth's contact re-pin leaving a 3.3 cm/s planted step (ufc_cross B frame 248).
- Report the measured spreads so the PM can tighten the two golden tolerances; don't edit
  `tests/golden/`. Bump stage `version`s whose output meaning changes.

### V. Strict edit validation (owner: `src/afv/core/edits.py`, `blender/afv_addon/edits.py` (validation
and writers only), `tests/core/`, the edit-file tests in `tests/blender/`)
- Both validators (afv core, pydantic; add-on, hand-written) reject lax values: booleans where a
  number is expected, numeric strings (`"1"`), floats with a fraction where an int is expected,
  NaN/inf. Messages name the file, the field path and the bad value. Both sides accept and reject
  exactly the same inputs (shared test vectors).
- Every writer (CLI, serve `edit`, add-on) still produces files that pass. Scan every existing edit
  file under `projects/` (including `projects/_golden/`): report any that would now fail; don't modify
  them.

## Round 6 assignments: shot browser

Inputs: `docs/ROUND5_NOTES.md` ("Open for round 6"; the owner chose the shot browser first, engine
export next). Spec targets (requirements table): "Detect shot cuts in a long video and build a shot list;
mark in/out ranges by hand; queue chosen shots"; "Shot browser in Blender: scrub a long video, see detected
cuts, add shots to the queue"; "From a 20-minute fight or a 90-minute film, you can mark and queue 20
candidate shots in <= 15 min, then process them unattended"; batch mode overnight. Same rules as round 5
(AGENTS.md section 6), with the review loop capped (owner decision, 2026-11-11): **two review passes per
track** (the review, then one review of the fixes). Whatever the second pass still finds goes on the round's
open list, unless the PM judges it a blocker (then fixed and checked by the PM, and reported).

Test source: `Media_Samples/Full Fight.mp4` (owner footage, git-ignored): 16 min 21 s, 1280x720, 29.97 fps,
no audio; pre-fight, the fight, post-fight. `detect_shots` today: 126 segments in 47 s (97 of >= 3 s).
Work on projects made from it in your scratch folder or `projects/full_fight` (created by the PM; don't
delete it). Never commit footage, frames or thumbnails.

Problems today: `afv shots detect` turns every segment into a shot and `afv batch` runs every shot (126
shots overnight, most of them crowd, walkouts and replays); `afv batch` skips shots without picked
fighters, and picking needs a GPU detector start per shot (`afv pick` / `track.detect_prompt`); in/out times
are typed by hand.

### Shared formats (PM-owned contract; request changes, don't make them)
- **Shot state** in `shot.json`: `"state": "candidate" | "queued" | "skipped"`. Missing = `"queued"` (every
  shot made before round 6 keeps today's behaviour). `afv shots detect` adds `candidate` shots;
  `afv shots add` adds `queued` shots (hand-marked means chosen). Only `queued` shots run in `afv batch`
  (`--state` to override). A shot's state never changes by itself.
- **`media/cuts.json`** (written by `afv shots detect`, read by the browser):
  `{"format": "afv_cuts", "version": 1, "source_sha256", "fps", "params": {...}, "cuts": [{"frame", "time",
  "confidence", "kind": "hard" | "gradual"}]}`; times in source seconds, frames source frame indices.
- **Candidate scan** `media/scan/<shot id>.json` + `media/scan/<shot id>.jpg` (thumbnail, <= 480 px wide):
  `{"format": "afv_scan", "version": 1, "shot_hash": <hash of the shot's start, duration and source>,
  "frame": <shot-timeline frame, 30 Hz>, "prompt": <exactly what track.detect_prompt returns for that shot
  and frame: numbered people with boxes and scores>, "people": n, "fighters_likely": bool}`. A scan whose
  `shot_hash` no longer matches the shot (it was trimmed) is stale and ignored. `afv pick --fighters i,j`
  (and serve `pick`) on that frame uses the stored prompt instead of starting the detector, so the numbers
  the browser shows are the numbers pick uses. As built (track Q, accepted): extra keys `shot`,
  `fighter_sized`, `size`, `scanned`; extra files `<id>.frame.jpg` (the exact frame) and `<id>.prompt.jpg`
  (numbered image, made on demand); scan frame = shot-timeline frame `min(15, T-1)`. `fighters_likely` is
  **computed when the list is read** (from the stored people), not trusted from the file, so the rule can
  change without re-scanning; it is a hint (round 6 rule: 32 of 47 marked are fight action, 9 of 41 fight
  shots missed), so the browser sorts and highlights by it and never hides shots because of it.
  `cuts.json` cuts also carry `start_frame`, `start_time`; `params` = `{detector, min_confidence,
  include_gradual}`.
- **CLI**: `afv shots detect` (as today plus `cuts.json` and states), `afv shots scan [IDS] [--state ...]`
  (one GPU worker for all, progress, resumable), `afv shots set ID [--start --end --duration --speed
  --notes --state]`, `afv shots queue IDS`, `afv shots skip IDS`, `afv shots list [--state]` (state, range,
  pick status, scan status), `afv batch [--state queued]`.
- **Serve** (protocol as in `src/afv/serve.py`): `init` (new project from a video: probe, proxy, detect),
  `shots` (list with states, ranges, cut confidences, scan and pick status, thumbnail paths), `shot_set`,
  `shots_detect`, `shots_scan` (progress events, cancellable), `pick` (shot, fighters, extra, frame; uses the
  stored scan prompt when it matches), `batch` (queued shots, progress per shot and stage, cancellable,
  ends with a per-shot outcome list). Every command validates like the CLI and returns readable errors.
  As built (track Q, accepted): also `shot_add`; `pick` without fighters returns the prompt (numbered
  image) to show; the exact arguments, events and results are in the `src/afv/serve.py` docstring, which
  is the reference for track B. Project commands run on per-project lanes so listing and editing never
  wait behind a scan or a batch. No proxy is made for sources of 720 px or less (`init --proxy` forces it).
- **Batch survives the client** (PM decision after Q review pass 1): an overnight queue must not die when
  Blender closes or the add-on restarts its worker. Serve `batch` starts a **detached** `afv batch --json`
  process (one per project; a second start returns the running one) that writes every event to
  `media/batch/<batch id>.jsonl` and its per-shot outcomes there; serve streams that log to the client while
  it is connected. New serve commands `batch_status` {project} (running?, batch id, current shot and stage,
  outcomes so far, the log path; also for a batch started earlier or from the CLI) and `batch_cancel`
  {project}. `afv batch` re-reads the queued set after each shot, so shots queued while it runs are done
  in the same night; it stops when no queued shot is left to run.

### Q. Queue, scan and browser backend (owner: `src/afv/core/project.py` (shot state), the `shots`,
`pick`, `batch` and `init` commands in `src/afv/cli.py`, `src/afv/serve.py` (the new commands),
`src/afv/media/shots.py`, `src/afv/models/track.py` + `src/afv/models/workers/track_worker.py` (batched
prompt detection only), `tests/core/`, `tests/serve/`, `tests/media/`, `tests/models/`)
- Implement the contract. Batched prompt detection: one worker process for many (shot, frame) pairs,
  model loaded once, frames read straight from the source (or the proxy) at exact shot-timeline times
  (must match what `detect_prompt` sees for that shot: same frame, same boxes, same numbering; test it).
- `fighters_likely`: at least two people whose boxes are large enough to be fighters (define and test the
  rule on the Full Fight candidates; report how many it marks and how many of those you checked by eye in
  their thumbnails are really fight shots).
- Acceptance (Full Fight.mp4): `init` + `detect` <= 2 min; `scan` of all candidates <= 4 min on this GPU;
  `pick` from a stored scan <= 1 s with no GPU start; existing projects (copies of the sample projects)
  behave exactly as before (all shots queued, golden tests pass); `afv batch` runs only queued shots;
  serve commands covered by tests; GPU jobs hold the GPU lock like other GPU stages.

### B. Shot browser in Blender (owner: `blender/afv_addon/` new `browser*.py` modules plus their
registration in `__init__.py`/`ui.py`, `blender/afv_addon/README.md`, `WORKFLOW.md`, new
`tests/blender/test_browser*.py`; starts when Q has reported, against the contract)
- AFV > Shots panel: **New project from video** / **Open project**; the source (or its 720p proxy) on
  Blender's timeline (scrubbable at its frame rate; a sequencer strip or a movie clip, your choice) with
  cut markers from `cuts.json`; the shot list (thumbnail, range, length, cut confidence, state, people /
  fighters-likely, pick status; filter by state and fighters-likely; click to jump); **Set in / Set out**
  from the playhead, speed factor, notes, **Queue / Skip**, **Add shot here** (manual range); **Scan
  candidates** (background, progress); **Pick fighters**: the shot's prompt frame with its numbered boxes
  drawn over the video, click fighter A, then B, then extras (from the stored scan: instant; otherwise the
  detector runs in the background); **Run queue** (`batch` in the background with per-shot progress,
  checks first that every queued shot has its fighters picked and lists those that don't); open a finished
  shot's take with the existing Import take.
- Everything through the one `afv serve` worker (one-shot fallback like the other tools); the UI never
  blocks; every action that changes the project is reflected in the list within 1 s.
- Acceptance (Full Fight.mp4): browser ready (video on the timeline, markers, list with thumbnails) <= 15 s
  after Open project once the project exists; set in/out, queue/skip, add shot: <= 1 s each round trip;
  pick from a scanned candidate <= 1 s; a scripted end-to-end run (open, scan, queue 20 fighters-likely
  candidates, pick fighters in each, start the queue) whose machine time (everything the owner would wait
  for) totals <= 3 min, so the 15-min owner target is about attention, not waiting; README and WORKFLOW
  updated; the add-on zip builds.
