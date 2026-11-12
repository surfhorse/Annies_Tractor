# Round 2 integration notes (collected from round 1 reports)

## From module A (core + CLI)
- Nothing declares `ingest` as an input (detect_track has `inputs=()`), so `afv run` skips ingest unless `--to ingest`. Decide: detect_track (and camera) should depend on ingest.
- Export stage writes files at the top of `work_dir` instead of `work_dir/exports/`; the runner has a fallback. Make export use `exports/`.
- `ExportParams` has no `fps`, so `afv export --fps N` fails. Add it or drop the CLI option.
- Export's manifest result kind must be registered with `register_result_kind` (added to types.py) to be loadable.
- D's `prompt.json` `"image"` points into the temp work folder, which is gone after the stage commits; store a path relative to the stage folder.
- C writes `pose3d_qc.json`; only `qc.json` reaches `meta.json`. Standardise on `qc.json`.
- `afv models --check`: decide whether `commercial_output_ok=False` fails (spec NFR-12 says export requires commercial-OK licences, so it should fail).

## From module F (export)
- `ExportManifest` lives in `afv/export/stage.py`; switch its `RESULT_KINDS.setdefault` to `register_result_kind` (types.py) and make the runner import the export module before loading export results.
- Export writes to the top of `ctx.work_dir`; move outputs to `ctx.work_dir/exports/` (runner publishes that).
- Retime not applied in export yet (exports at motion fps, speed 1); wire `afv.motion.retime` and add `fps` to `ExportParams`.
- `Take.rest_offsets` now exists in the contract: use it instead of the separate `rest_offsets` argument.
- Add `src/afv/export/data/soma77.json` mapping once C's soma77 skeleton lands.
- Open: Unity presets default prefix (`""` vs `mixamorig:`); root heading smoothing 0.2 s default; real engine import checks (Unity/Unreal/Godot) not yet done.
- Blender quirks found (keep the workarounds): glTF fps*fps_base bug for non-integer rates; FBX importer integer fps; BVH importer start frame 1.

## From module C (GEM-X, SOMA, camera)
- soma77 skeleton is in `src/afv/core/data/skeletons/soma77.json` (77 joints, root `Hips`, T-pose rest = afv conventions, `soma_joint_orient` kept for export twist axes). Contact channels = joint names: LeftFoot, LeftToeBase, RightFoot, RightToeBase, LeftHand, RightHand.
- E: build the joint-role config for soma77; F: add `export/data/soma77.json` mapping (use `soma_joint_orient` if useful).
- Stage inputs: camera ← detect_track; pose3d ← detect_track + camera. Video path/start/duration read from shot.json/project.json with guessed key names: align with A's `shot_context()`.
- Shot normalisation uses GEM-X's own `normalize_video` inside the gemx worker; B's normaliser must agree (frame counts) or be used only for proxies/cuts.
- World is fixed by the camera stage (GeoCalib up, ground from boxes + 1.65 m height prior = first guess). Open: let pair/contacts refit the floor from the feet and record one world transform (e.g. `world_from_camera` on BodyMotion). Close boxing feet were ~0.4 m above y=0 before an edge-box fix; re-check.
- GEM-X postprocess (contact/IK) off by default; our contacts stage owns that.
- Static/background camera modes use GEM-X's diagonal focal; GeoCalib's focal could be used.
- GPU tests are opt-in: `AFV_GPU_TESTS=1`.

## From module B (media)
- Normalisation is frame-identical to GEM-X's `normalize_video` (framemd5) and T depends on ffmpeg version and stream count: GEM-X must run with the same ffmpeg (`~/.local/bin` first on PATH) and get the ORIGINAL source + start/duration, never B's re-encoded `shot.mp4`.
- `ref walk.mp4` has a hard cut at 8.782 s (shot 0 T=264, shot 1 T=8). P0 solved straight through it; Close boxing has NO cut (continuous re-framing).
- A's `Project.add_shot` should bound shots by `cv_duration` (GEM-X's limit), not container duration; use `ClipInfo.to_dict()`.
- `ShotMedia` stays in `afv/media/result.py`, registered with `register_result_kind` (decision: not moved into the shared types).
- Per-shot overlay masks are unreliable on static-camera shots; consider a source-level mask in round 2.
- ffmpeg GPL-3.0 LicenceRecord on ingest with commercial_output_ok=True: fine (ffmpeg's output is not covered by its licence).

## From module D (tracking)
- `fighters.json` v1 = A's pick output + optional `objects` (labels/roles) and `prompts` (frame, obj, detection|box|points). Object ids: A=1, B=2, extra[i]=3+i.
- Add a re-prompt CLI: `afv pick <shot> --frame F` (numbered image for any frame) and `--prompt F:OBJ:DET [--label]` calling `add_prompt`; today users hand-edit fighters.json from qc.json hints.
- `<shot>/.cache/track` holds frames/detections (safe to delete): accept in the project layout.
- Decide whether the runner pauses when tracking QC has suggestions (untracked person, drift) instead of continuing.
- Masks: `stages/detect_track/masks/<id>/<frame:05d>.png` (missing = empty). C should use them (with all other people) to mask the camera solve.
- Shots must be cut before tracking: ref walk's cut at ~frame 264 moves fighter A's mask to another person (QC flags it).
- Swap/appearance checks are validated only on synthetic swaps; Close boxing B (knockdown through ropes, frames 78-91) is a false identity flag.

## From module E (motion)
- Floor y=0 after pair and contacts; `world_shift` accumulated (pair, contacts) and copied through (smooth, retime).
- soma77 roles registered (`SOMA77_ROLES`): thigh LeftLeg, calf LeftShin, foot LeftFoot, toe LeftToeBase; name-guessing would mislabel soma77.
- Retime is a library function (`afv.motion.retime.retime`), not a stage: export (F) must call it with fps + speed_factor.
- Stages write `qc.json` + `qc_frames.npz` into work_dir (runner keeps work_dir files; check they survive commit).
- Not done: pins (`edits/pins.json`), hand/wrist contact pinning.
- Strike thresholds (2.5 m/s punch, 5 m/s kick, 0.2 m travel) need checking on real footage.
- Pair: common depth offset unobservable; stays at GEM-X mean depth unless `camera_ground_sigma` set.
- Nothing yet run on real GEM-X data: first thing in round 2.

## Module A review fixes (done)
- Stage results are versioned: `stages/<stage>/v<NNNN>/`; always resolve with `afv.core.project.stage_dir()` / `Shot.stage_dir()`.
- Freshness requires `npz_hash(result.npz) == meta.output_hash`; consumers always get the stored float32 data.
- `ingest` always runs. Replaced source detected (size+mtime) -> `afv init <video> --project <dir> --rehash`.
- `afv export --fps` removed until retime is wired.

## Module C review fixes (done)
- pose3d re-fits the floor from solved feet (contact-weighted median sole), shifts both fighters, sets `world_shift`; residual floor error ≤ 0.1 m except Who Am I A (+0.19 m; flagged `fighters_disagree_on_floor` for pair).
- `conf` = 0 on unobserved (interpolated/held box) frames; `unobserved_spans` in qc.json.
- `CameraTrack.focal_px` = the single median focal GEM-X used. Use `afv.models.camera.shifted_camera(cam, world_shift)` for reprojection.
- Both camera and pose3d normalise from the original source with ffmpeg at `$AFV_FFMPEG_DIR` (default ~/.local/bin).
- To fix in D: `tests/models/test_track_stage.py::test_runner_needs_pick_then_tracks` fails now that ingest always runs (A's change).

## Module F review fixes (done)
- Root heading from the hip line (robust to flips/lying); separate 0.5 s yaw smoothing.
- soma77 export mapping: metacarpals, Neck2, Jaw, eyes folded; wrist exact; knuckles shift ≤ 2·L·sin(θ/2) (up to ~26 mm at 0.45 rad). Round 2: measure real GEM-X metacarpal rotations; if large, add optional metacarpal bones to the rig.
- FBX has no rotated Armature null now (identity, Y-up); GLB unchanged. `afv/export/fbx.py` reads FBX node trees for tests.
- Outputs in `work_dir/exports/`; provenance from `shot_context`; `ExportParams.fps` exists but a different fps raises until retime is wired; `shot_fields=("speed_factor",)`.
- Blender launch retries once on WSL interop `UtilAcceptVsock` failures.

## Module E review fixes (done) and open upstream issues
- Real-data results: planted-foot speed ≤ 0.8 cm/s mean, pelvis drop ≤ 1.8 cm mean (cap 2.5 cm), knee change 12-23°/frame, punch peaks keep 85-97%. Real-data tests in tests/motion/test_motion_real.py.
- OPEN (upstream, C/camera): on telephoto pans (ref walk, Who Am I) the camera-based root moves ~1.36 m/s across the ray vs GEM-X's own 0.43 m/s while 2D hips are smooth, so most GEM-X plants are vetoed (recall 0.01-0.09 on ref walk). First re-run ref walk as two shots (the P0 data spans the cut at frame 264, where the root glitches are), then investigate camera rotation.
- OPEN: hips reprojection peaks 13-22 px vs 15 px target.
- OPEN: smooth/retime re-pin with default ContactParams, not those contacts ran with (runner would need to pass them).
- Kick strike threshold kept at 4 m/s (3 m/s adds pops); foot peaks 3-4 m/s keep 51-74%.

## Round 2: telephoto investigation (T) results
- RESOLVES the E "OPEN (upstream, C/camera)" item. Cause: ViPE rotations are wrong on low-parallax
  telephoto shots (ref walk 3.6x too fast, Who Am I the wrong way, Close boxing barely rotating),
  measured against background flow (ViPE 4.6-11.4 px vs flow fit 0.26-0.36 px; UFC control 0.52 px).
  Not a convention, focal or root-placement bug; the cut was not the cause.
- Camera stage v2: `rotation_source="auto"` keeps ViPE while its flow residual <= 2 px, else uses
  flow-fitted rotations (gravity re-levelled). Flow is not always used: it drifted 2.3 deg pitch on UFC.
- pose3d v2: `root_mode="auto"` falls back per fighter to GEM-X's world rollout when the camera root's
  lateral speed > 3x prior and > prior + 0.5 m/s (flag `{role}_root_fallback_gemx_world`).
- Ref walk shot 0 plant recall 0.08/0.07 -> 0.65/0.28; Who Am I 0.08/0.27 -> 0.28/0.31.
- Hips reprojection "13-22 px" is a constant SOMA-Hips vs ViTPose-hip offset (15-23 px, spread 4-6 px):
  measure hips against pose3d's own projection instead.
- Zoom is measured and flagged (`zoom`), not applied: correction worsened the Close boxing floor.
- To do: regenerate motion fixtures built on P0's wrong cameras and re-baseline real-data metrics.
- Open: remaining recall ~0.3 on Who Am I / ref walk B (depth noise, GEM-X contact labels); a
  ViPE-free flow camera mode would save ~10 GB VRAM but needs a focal source (GeoCalib).

## Round 2: integration (I) results
- `afv run` goes ingest → export on all five shots (ref walk split at 8.782 s: 001 = 264 frames,
  002 = 8 frames); FBX/GLB/BVH/json per fighter in `projects/<clip>/shots/<id>/exports/`.
- New: hands_face stage (passthrough / fist / neutral face), detect_track ← ingest, runner
  `inherit_params` (ContactParams into smooth/export), retime in export (v3, `--fps`, speed_factor),
  `afv pick --frame/--prompt`, `models --check` fails on non-commercial, golden tests (3 shots, opt-in).
- Telephoto review fixes (T, round 2): relative ViPE-vs-flow choice + drift check, shot-level root
  fallback, border/blob-only overlay mask, coverage-scored RANSAC.
- Metacarpals: median 7-12 deg, max 21.6 deg (Who Am I) -> folding OK, no extra bones needed.
- Open: golden tolerances not calibrated (one repeat run); runner does not pause on tracking QC
  suggestions; unity_humanoid preset has no GLB by default; Who Am I camera height -0.35 m.
- Motion (E) after fixture regeneration: 5 strict xfails (pelvis drop, knee change x2, smooth-0 not
  a no-op x2) + retime plant-edge bug (interpolated contact >= 0.5 counted planted) -> motion agent.

## Round 2: final state (committed)
- All five shots run end to end; second `afv run` is fully fresh. 644 tests pass, 0 xfail.
- Camera: flow-checked rotations, worker cache (`.cache/camera`, `--from` forces), GEM-X gravity
  re-level pass (close_boxing, who_am_i, ref_walk/001). Camera heights 1.12-4.41 m, all above floor.
- pose3d v3: per-fighter floor scale about the camera centre (cap 5%, min contact weight 10);
  statures 1.60-1.78 m (HeadEnd); ref_walk/002 B capped.
- pair v3: floor-mean/depth-mean rows (kept on: +6-14 pts recall for <1 px), push-apart with
  exact episodes, one order per episode by cost, ramp handling as HEAD; banded solver.
- contacts v4: capped pin raise (3 cm), knee-space blend (per-plant), edge-slide trim, plant counts.
- Contacts on final runs: recall 0.25-0.88 (Who Am I A lowest), planted slide <= 0.35 cm/s.
- Golden: 3 shots, tolerances ~4x one observed repeat per shot (small sample); BVH hash not compared.

## Open for round 3
- Cache keys don't include stage source code: a code change without a version bump leaves stale
  results "fresh" (happened in round 2). Consider hashing each stage module into its key.
- Who Am I A recall 0.25 (GEM-X gives few contacts); close_boxing B knockdown separation unverified.
- Zoom flagged, not corrected (close_boxing 1.5x, ufc 1.2x).
- Lateral root check not windowed within a shot.
- GEM-X re-level pass costs 40-60 s GPU (no raw-output cache).
- Hover metric max 3-4 cm vs contacts max_plant_hover 2 cm (sole reference may differ): check.
- Real engine imports (Unity/Unreal/Godot) not done; runner doesn't pause on tracking QC hints;
  unity_humanoid preset has no GLB by default; strike thresholds; pins (edits/pins.json).
- Licences: training-data review and GeoCalib CC BY credit before any sale.
