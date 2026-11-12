# P0 spike findings

*(Phase 0 is complete. The `spike/` scripts and render videos this page mentions were the author's
experiments on their own footage and are not in this repository; the licence register it calls
`models/LICENCES.md` is `THIRD_PARTY_LICENSES.md`.)*

Started 2026-11-09. Goal: confirm the chosen models work on real fight footage before building the
pipeline (spec: Development plan, P0 gate). Scripts in this folder; results in WSL `~/afv/runs/`.

## Environment (spec open item 1)

| Item | Value |
| --- | --- |
| GPU | RTX 5080, 16 GB (Blackwell sm_120), driver 617.42 |
| Model env | WSL2 Ubuntu 26.04; no sudo needed (uv, user-space static FFmpeg, Miniforge) |
| GEM-X | commit 4aef145, Python 3.12, torch 2.11.0+cu128; weights hashed in `models/gemx_lock.txt` |
| ViPE | builds on sm_120 with conda nvcc 12.8; torch 2.9.0+cu128 in its own env |
| Rendering | GEM-X `gem-render` works with WSLg's Mesa EGL + Open3D 0.19 |

Native Windows was not attempted: GEM-X documents Linux only. WSL2 it is.

## Facts that differ from the spec

1. **GEM-X release date** is 2026-03-16 (model card), not 2026-10-30.
2. **Single-person demo.** `motion-video` solves only the track with the largest cumulative box
   area and has no box/track option. `gemx_fighter.py` patches the selection to solve any track
   (by rank or id); run once per fighter.
3. **SOMA output** (spec open item 5): 77 joints = root + 76 local rotations (`body_pose` 228 =
   76 × axis-angle), plus 128 identity coeffs and 57 scale params per frame; joint positions need
   SOMA's FK (`py-soma-x` 0.3.3). Contact channels (6): left/right ankle, foot, wrist.
4. **Camera.** The released Motion model uses camera **rotations only**; ViPE translation/depth
   do not anchor world motion. GEM-X's ViPE pipeline (`pose_only_long`) freezes intrinsics after
   the first keyframe retirement, i.e. **one focal length per clip**, not per frame (spec FR-4).
5. **Shared world (G-2)** is not provided: each fighter is integrated in its own world frame.
6. **Weights are not gated** (GEM-X: NVIDIA Open Model License; SOMA-X: Apache-2.0).

## Runs on Media Samples (background camera mode)

| Clip | Fighter | Wall time | s per s of video | Track coverage | Contact foot speed after GEM-X IK |
| --- | --- | --- | --- | --- | --- |
| Close boxing (3.1 s, 29.97 fps) | A / B | 82 s / 82 s | 27 | 100% / 88% | 25 / 18 cm/s |
| UFC cross (10.1 s) | A / B | 253 s / 247 s | 25 | 100% / 62% | 13 / 20 cm/s |
| Who Am I (4.1 s, 23.976 fps film) | A / B | 98 s / 110 s | 24–27 | 100% / 100% | 64 / 33 cm/s |
| Ref walk (9.1 s) | A / B | 200 s / failed | 22 | 90% / – | 29 cm/s / – |

Peak VRAM 6.4–7.2 GB total (about 4–4.4 GB above the desktop baseline). The person detector
(YOLOX) runs on CPU by default.

Ref walk fighter B failed: "Background camera tracking was lost" — the background camera estimate
masks only the solved person, so the other people read as background motion.

## Visual review (overlay renders)

- **Pose fit: good.** Guards, jabs and body shape align on the boxing clips; Who Am I's jumps and
  crouches track through motion blur; the kicker's blurred spin is the roughest.
- **Tracking: weak (spec risk R2).** ByteTrack fragments fighters into several ids (UFC fighter B:
  62% + 30% in two ids). GEM-X then interpolates/holds the box, so at the end of UFC cross both
  solves sit on the same fighter. Ref walk fighter A appears to switch fighters around 6 s.
- **Cuts:** Close boxing contains a camera change at ~1.2 s and is solved straight through it;
  shot-cut detection must come before solving (FR-1).
- **Foot skate:** 13–64 cm/s in contact vs the NFR-3 target of ≤ 3 cm/s; our own contact stage is
  required.

## Throughput vs NFR-1

NFR-1 wants a 5 s two-fighter shot in ≤ 3 min (static camera). Measured: ~25 s per second of
video per fighter → ~4 min for 5 s × 2 fighters, before ViPE. Detection on GPU and solving both
fighters in one process (shared decode/detection) are the obvious savings.

## ViPE camera mode vs GEM-X's background camera

| Clip | Fighter | Wall time background → ViPE | Peak VRAM (total of 16.3 GB) | Focal estimate (px, constant per clip) |
| --- | --- | --- | --- | --- |
| Close boxing | A / B | 82 → 201 s / 82 → 131 s | 14.8 / 14.4 GB | 7296 / 7333 |
| UFC cross | A / B | 253 → 297 s / 247 → 327 s | 13.0 / 13.0 GB | 1202 / 1202 |
| Who Am I | A / B | 98 → 166 s / 110 → 156 s | 12.5 / 12.4 GB | 7665 / 7630 |
| Ref walk | A / B | 200 → 266 s / failed → 274 s | 14.0 / 14.0 GB | 8448 / 8581 |

- **ViPE fixed the one failure** (ref walk B), so it is the more robust camera.
- **World motion is nearly identical** to background mode on UFC cross and Close boxing (world
  renders compared side by side), as expected when only camera rotation feeds the model.
- **Cost:** +20–145% wall time and ~10–12 GB VRAM above the desktop baseline; peak 14.8 of
  16.3 GB, too close to the limit to run beside anything else.
- **Zoom (spec open item 3): not handled.** Focal length is one value per clip (confirmed in the
  outputs). The camera solve is also re-run per fighter and lands on slightly different focals
  (7296 vs 7333 px), so the two fighters do not share one camera either.
- Recommendation: background mode by default, ViPE as the fallback when background tracking is
  lost; solve the camera once per shot and reuse it for both fighters (needs a `provided` camera
  file, which GEM-X supports).

## Tracking: RF-DETR + SAM 2.1 (spec's planned tracker)

Setup: `install_track.sh` (RF-DETR 1.12 Medium, SAM 2.1 Hiera-L, both Apache-2.0; hashes in
`models/track_lock.txt`). `track_fighters.py detect` numbers RF-DETR's people on a prompt frame;
the user picks the fighters (and referee); `track` propagates SAM 2 masklets both ways and QC-checks
each mask box against RF-DETR detections on every frame. Frames match GEM-X's normalised timeline
(same ffmpeg interval and filter; verified frame counts 92/302/122/272).

| Clip | Fighter coverage, ByteTrack (GEM-X demo) | Fighter coverage, SAM 2 | Mask frames with no matching detection (longest run) | Swaps seen in review |
| --- | --- | --- | --- | --- |
| Close boxing | 100% / 88% | 100% / 100% | 0 / 10 (10, at the camera change) | none, held across the camera change |
| UFC cross | 100% / 62% | 100% / 100% | 0 / 7 (2) | none, fighters cross sides at ~185 |
| Who Am I | 100% / 100% | 100% / 100% | 0 / 0 | none |
| Ref walk | 90% / 47% | 100% / 100% | 13 / 3 (6) | none through the referee crossing (136–204) |

SAM 2 tracking takes well under a minute per clip on the 5080. A referee who enters after the
prompt frame is not tracked (ref walk); masking them from the camera needs a re-prompt or a
detection-based "everyone else" mask.

## Final P0 configuration: SAM 2 boxes + one shared camera per shot

`gemx_fighter.py` (AFV_BOXES_NPZ/AFV_SLOT) replaces GEM-X's YOLOX+ByteTrack with the SAM 2 boxes and
masks the other tracked people out of the camera estimate. Background camera mode then failed on 2
of 8 solves (fewer background features once everyone is masked), so the camera is solved once per
shot with ViPE and passed to both fighters in GEM-X's `provided` mode (`make_provided_camera.py`,
`p0_shared_batch.sh`). Renders: `renders/sam2_tracks_shared_camera/`, tracking reviews:
`renders/sam2_tracking_review/`.

| Clip | Wall time per fighter (GEM-X only) | VRAM above baseline | Contact foot speed after GEM-X IK | Swaps in review |
| --- | --- | --- | --- | --- |
| Close boxing | 22 / 19 s | 4.1 GB | 24 / 20 cm/s | none |
| UFC cross | 38 / 38 s | 4.1 / 3.9 GB | 12 / 12 cm/s | none |
| Who Am I | 21 / 27 s | 4.1 GB | 65 / 33 cm/s | none |
| Ref walk | 31 / 30 s | 4.1 / 4.4 GB | 19 / 13 cm/s | none |

All 8 solves succeed. GEM-X is ~5× faster than the demo path because YOLOX on CPU was the
bottleneck. Per shot add SAM 2 (< 1 min) and one ViPE solve (~1–2 min). A 5 s two-fighter shot is
about 2–3 min end to end, inside NFR-1.

**Shared world (G-2), measured as the distance between the two fighters' roots in the shared
camera frame:** UFC cross (wide lens, focal 1202 px, ~3.5 m away) median 1.26 m, range 0.66–1.74 m,
smooth. The telephoto clips (focal 7,300–8,500 px, 17–26 m away) are plausible side to side but jump
up to 1.9 m per frame in depth, and ref walk's minimum of 0.18 m means overlapping bodies. Depth
along the camera axis is ill-posed at broadcast range, so the pipeline needs a joint step: both
fighters on one ground plane, depth smoothed, and inter-person penetration resolved.

Process note: one batch was accidentally run twice at once (a `wsl.exe` call returned early while its
job kept running), which corrupted renders and timings; those runs were discarded and redone, and the
batch scripts now take a lock.

## P0 gate (spec) status

| Gate item | Status |
| --- | --- |
| All primary models run | Pass: GEM-X, ViPE, RF-DETR, SAM 2. RTMW (QC/hands only) not yet run |
| Peak VRAM and runtime recorded per model | Pass for GEM-X and ViPE; SAM 2/RF-DETR timed only roughly |
| Both fighters tracked without swaps on ≥ 8 of 12 shots | Pass on 4 of 4 sample shots with SAM 2 (12-shot set not assembled) |
| ViPE separates zoom from motion on shot 3 | Fail by design: one focal per clip. No sample has a strong zoom yet |
| ≥ 4 of 6 shots "usable after cleanup" | Your call; your first look: "the videos look good so far" |
| 12-shot test set + hand labels | Not done (needs you) |
| Cloud mocap baseline on 3 shots | Not done |

## Verdict

**GEM-X passes as the core model; do not re-plan.** Pose quality on broadcast and film fight
footage is good, and with SAM 2 tracking and a shared camera every solve on the samples works.
Spec changes the build should absorb:

1. Tracking = RF-DETR prompt + SAM 2 masklets, feeding boxes to GEM-X (replaces GEM-X's tracker).
2. Camera solved once per shot (ViPE), shared by both fighters via `provided` mode.
3. New joint two-fighter stage: shared ground plane, depth smoothing, penetration (G-2).
4. Contact/skate stage is essential: 12–65 cm/s after GEM-X's own IK vs the ≤ 3 cm/s target.
5. Cut detection before tracking (Close boxing has a camera change).
6. Zoom: per-frame focal is not available from this stack; treat zoom shots as a known limitation
   or test ViPE's other pipelines later.
7. Referees who enter after the prompt frame need a re-prompt (or detection-based masking).
8. Real SOMA skeleton (77 joints, `py-soma-x`) replaces the provisional one in `src/afv`.

## Licences

Checked 2026-11-09; see `models/LICENCES.md`. Every checkpoint on the final path allows commercial
use of outputs; none is non-commercial. Conditions: GeoCalib weights are CC BY 4.0 (credit
required) and GEM-X is under the NVIDIA Open Model License. ViPE's pipeline uses DROID-SLAM (not
DPVO as the spec assumed). Training-data terms still need review before a game is sold.

## Still open

- A clip with a real broadcast zoom, to measure how badly a single focal length hurts.
- Training-data terms per checkpoint (before any sale).
