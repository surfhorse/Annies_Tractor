# Round 3 notes (2026-11-09 to 2026-11-10)

Target engine: Blender 5.2 (the owner imports and cleans up there). Assignments: `docs/BUILD_PLAN.md`,
"Round 3 assignments". Every track had an independent strict review; findings were fixed and
re-reviewed (core 2 passes, Blender 2, motion 2, camera 3).

## Pre-round check
All 30 round-2 exports (5 shots x 2 fighters x FBX/GLB/BVH) import headless in Blender 5.2: 65 bones,
exact frame counts, formats agree to 0.02 mm per joint (match bones by name; order differs).

## BL: Blender add-on, slice 1
- `afv take <shot>` writes `shots/<id>/take/` (numpy/JSON package: export rig + motion, camera in
  the bodies' world with per-frame focal, QC spans, plant masks, proxy video path; atomic commit).
- `afv blender <shot> [--check]` builds `take/take.blend` headless.
- Extension `blender/afv_addon/` (id `afv`, Blender >= 5.2): AFV panel with Import take, Prev/Next
  issue, Re-import (replaces only the "AFV solved" NLA action; "AFV tweaks" Combine layer kept with
  its strip settings), Export (WSL background job; pauses on tracking hints with "Export anyway").
- Overlay (Blender camera vs pipeline projection) <= 0.003 px, take vs exported FBX <= 0.003 mm on
  all 5 shots; video frame k = scene frame k.

## C: core
- Stage code hash (normalised AST of the stage module and afv modules it imports, worker scripts,
  data JSON) recorded per result; snapshot per process. Policies: CPU "auto", model "report",
  camera "cached" (re-run from worker caches only; `Ctx.cache_only` + `CacheOnlyMiss`).
  `--code-changes report|rerun|accept`; `afv status` explains staleness.
- Tracking QC pause in `afv run`/`batch`/`export` (exit 3, `review.json` ack keyed on output +
  hint hash, `--yes`).

## M: motion
- QC and solver share the sole reference (`foot_sole_heights`); foot hover <= 2.07 cm.
- Toe pivots and toe-only frames pinned; knee IK bends forward (no backward flips beyond input).
- Root offset rate limit (0.4 m/s + 0.5 x input, along the camera ray); plant-step / edge-pop /
  knee-snap trims; held one-frame holes merged; smooth holds clip ends.
- QC `foot_pops` = 3-step lurch measure. Contacts lurches 157 vs 229 before round 3; smooth adds 0.
- GEM-X recall (contacts) e.g. Who Am I A 0.25 -> ~0.5, ufc 0.73/0.88 -> 0.96/0.93.
- Strike thresholds checked against footage: no tuning (misses are GEM-X's).

## Z: camera and GEM-X
- Zoom > 10 % applied as per-frame focal into CameraTrack and GEM-X (K [F,3,3]); verified correct
  (SIFT background scale 1.203 vs flow 1.203 on UFC; d*h/f invariant). Fighters really move away
  during close_boxing/UFC zooms.
- Windowed ViPE/flow decision with flow splicing ("vipe+flow"); root fallback gated on shared excess
  only (wrong-way pans fall back).
- GEM-X output cache (`<shot>/.cache/pose3d/`): re-level pass pose3d 0.5-0.8 s instead of 36-72 s.
- Gravity re-level limited by the feet (robust bootstrap interval + 1.5 deg bias allowance, rule "2"
  in relevel.json): who_am_i 80 % (camera 1.06 m), ref_walk/001 79 % (4.53 m), close_boxing refuted
  (2.16 m).
- Weak 2D evidence (torso/leg joints hidden): pose3d publishes `ActorMotion.weak`; pair follows the
  rollout there under an image-error budget (no edge pops). close_boxing B pair root path 4.7 m.

## Open for round 4
- Blender add-on slice 2: `afv serve` worker, contact painting, pins (`edits/pins.json`), smoothing
  panel, frame-range re-runs, gpu-drawn confidence strip, shot browser, ID-swap fix.
- close_boxing: contacts recall 0.50/0.54 (weak spans, occlusion); remaining root depth catch-ups
  (lurches) traded against recall at the rate limit.
- Feet-tilt bias allowance (1.5 deg) and weak-frame thresholds calibrated on 5 shots only; the
  re-level lands where feet still disagree on who_am_i and ref_walk/001 (flagged).
- close_boxing B tracking hint (frames 82-91) is a known false positive; needs `--yes`.
- One unexplained pytest hang when an agent ran four suites in one process (not reproduced: full
  suite 841 passed in one run).
- Licences: training-data review and GeoCalib CC BY credit before any sale.
