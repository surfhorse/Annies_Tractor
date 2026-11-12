# Round 5 notes (2026-11-11): retarget onto the owner's characters, stable plants, strict edits

Assignments, shared formats (bone map file, Retarget API, edit files) and acceptance numbers:
`docs/BUILD_PLAN.md`, "Round 5 assignments". Every track had an independent strict review by an agent
that never saw the implementer's reasoning; findings were fixed and the fixes reviewed again until
nothing above a nit was left (review passes: R1 5, R2 3, M 4, V 3). Contract changes made during
the reviews are in BUILD_PLAN (each marked where a PM decision changed an acceptance number).

Test characters (owner assets, git-ignored `Media_Samples/`): the owner's mech as a Mixamo rig and as
a Unity Humanoid rig (`Unity_Humanoid_Mech_rightarm_fixed.fbx`: the original Tripo rig's right
forearm/hand bones and weights were broken and were mirrored from the left arm), and the CC0 Girush
"Young Warrior" human (OpenGameArt; anatomical bone names, twist and knee pieces, cm scale).

## R1: retarget engine (`blender/afv_addon/retarget.py`)
- Rest-relative world rotation transfer with a per-bone rest alignment (joint -> mapped child joint;
  hips facing only; feet and toes keep their own rest pitch to the floor), unmapped twist/knee pieces
  left at rest, Copy Rotation knees (Girush) solved by a damped Gauss-Newton fit. Facing detected in
  the target armature's own axes. Hips placed from the pelvis anchor (mid-thighs) x `hips_scale`.
- Lock feet from the take's plant masks: two-bone IK, C1 Hermite fades at lock edges (acceleration
  bounded), a mesh-based floor (toe tip, heel and the foot-weighted vertices at Rest Position), hips
  lowered (<= 25 % leg) where a planted foot is out of reach, reported in `notes`.
- `guess_map` maps Mixamo (any prefix), Unity Humanoid, Rigify metarig, anatomical (`Humerus.L`),
  3ds Max Biped, Unreal/DAZ/CC4-style names; never wrong side or duplicates; checks the target's
  hierarchy. Refuses generated Rigify rigs, non-uniform scale, a character not standing upright, a
  source outside the view layer, maps that contradict the hierarchy; notes suspected L/R swaps.
- Never touches the target's rest pose, meshes, transform or other actions; unwinds completely on any
  error (action, tweak mode, blend settings, modifiers).
- Measured, 3 characters x ufc_cross / close_boxing x both fighters: free runs direction error
  <= 0.071 deg, twist <= 0.08 deg (Girush; 0.000 on the mechs); locked: median planted-foot slide
  0.000-0.069 cm/s, planted soles 0.00-0.01 cm from the floor, mesh never > 0.55 cm below the
  source's own lowest foot point, every lock edge within max(1.5x free, 1800 cm/s^2). 0.14-0.9 s per
  fighter; 3.4M-vertex targets 1.45 s.

## R2: Retarget panel and `afv retarget` (`blender/afv_addon/retarget_ui.py`, `retarget_io.py`,
`retarget_headless.py`, `src/afv/retarget.py`)
- AFV > Retarget: Import character (FBX / GLB / .blend append without running scripts: Python drivers
  are muted, ones on data linked from a third file are refused), Guess / Load / Save map
  (`*.afvmap.json`, also kept on the armature), facing and hips-scale overrides, bone list with
  problems shown, Lock feet, Retarget (one undo step; report with notes), Fix rest matrices (for
  armatures with rest-matrix drift Blender's FBX importer would drop), Export character (FBX with the
  engine preset's exporter arguments, fetched through `afv serve`'s new `presets` command).
- `afv retarget <shot> --character F [--map] [--role A|B|both] [--preset] [--out] [--no-lock-feet]
  [--save-map] [--check] [--list-presets] [--json]`: headless, both fighters on one origin, the clip
  (FBX animation stack) named after the action, stale-result warning, exit 1 with the missing bones.
- FBX round trip (re-import vs the in-Blender retarget, every frame): 0.007-0.020 mm at fighter
  scale on the three characters, 302 frames at 30 fps. Both fighters with `--check`: 9-17 s.
- The panel replaces its own previous Retarget action only while it is untouched (still active, same
  name, same datablock on the same character, unchanged keys/handles/interpolation/modifiers);
  otherwise it adds a new one. A reopened file never matches.

## M: stable plant decisions (`src/afv/motion/contacts.py`, `ik.py`)
- Cause of the round-4 bimodality: contacts' root-correction rate limit made the solve depend on its
  iteration path, so a one-frame plant flip moved the root by up to 8 cm. Now a convex one-sided
  penalty with a stop rule on the estimated distance to the converged solve (<= 0.11 mm on 79 crops,
  real inputs and harness draws); `clean_contacts` is as fast as before (-30 % to +13 %).
- smooth's re-pin uses the IK's real reach, so it no longer adds planted steps (ufc_cross B frame 248/249).
- `max_plant_step` 3 -> 2.5 cm/s (worst planted step now <= 2.4 cm/s, e.g. ufc_cross B 2.94 -> 0.03),
  at the cost of up to 2 plant frames per fighter and a larger contact-frame flip on ufc_cross A.
- Real repeats, smoothed root path: close_boxing A spread 0.120 m (bimodal) -> 0.022 m; all keys
  within 0.043 m. `pair.distance_median_m` is GEM-X depth noise, not a decision flip.
- CPU noise harness: `tests/motion/noise_harness.py` (an upper bound; tolerances come from real repeats).
- Versions: contacts 5 -> 6, smooth 6 -> 7.

## V: strict edit files (`src/afv/core/edits.py`, `blender/afv_addon/edits.py`, `models/track.py`)
- Both validators reject bools, numeric strings, fractional floats for ints, NaN/inf and non-JSON
  values; they agree on every input and report the same first error (shared vectors in
  `tests/core/edit_vectors.py`, differential fuzz tests). Pin radius >= 1; pins' `root_rotation`,
  `prefix`, `root_motion` checked; fighters.json strict too.
- Every writer validates (`Shot.write_edit`); `afv run` / `batch` / `export` / serve refuse a bad edit
  file before any stage runs or settings are saved; `afv status` / `--dry-run` show the stage table
  with blocked stages and exit 1; `afv take` / `afv blender` name the bad file. Messages end with
  "fix or delete that entry in edits/<name>.json". No existing project file is affected.

## Integration (PM)
- Golden tests had been skipping silently since the clips' folder became `Media_Samples/`: they now
  find it (or `AFV_MEDIA_SAMPLES`, or the old name) and a missing clip fails.
- Golden values regenerated after review (all three shots); tolerance `smooth.*.root_path_m` tightened
  from (0.05 m + 3 %) to 0.06 m from the measured repeats (largest spread 0.043 m).
- Full suite: **1972 passed, 11 skipped** (was 1024 / 11 before the round), ruff clean; the three
  golden shots pass against the regenerated values.
- The sample projects pointed at the old `Media Samples/` paths: relinked with `afv init <clip>
  --rehash` (all four hashes unchanged, so nothing upstream re-ran). All five sample shots re-run
  (`--code-changes accept`: detect_track's only change is fighters.json validation, which cannot alter
  its output; verified on all 8 fighters files) and exported; `afv blender --check` (into a scratch
  folder, so the owner's `take.blend` files are untouched): joints <= 0.007 mm, overlay <= 0.003 px.
- The add-on changed (Retarget panel, strict edits): rebuild and reinstall the zip (`dist/afv-0.1.0.zip`).
  Existing `take.blend` files: press Re-import to pick up the new contacts/smooth results.

## Open for round 6
- Plant decisions still flip at their edges (close_boxing A contact frames 99-107; ufc_cross A one vs
  two toe plants, ~3 cm root path); onset hysteresis was tried and cost 12-20 plant frames.
- close_boxing A right heel plant frames 84-88 hovers 2.04 cm (inside the 2 cm + 1 mm rule, old code
  too); the retargeted foot floats there.
- `pair.distance_median_m` follows GEM-X's depth drift (pose3d), up to 0.031 m run to run.
- Export character / `afv retarget` apply only the preset's FBX exporter arguments: no root bone,
  root-motion mode, armature rename or Unreal cm scale. Characters keep their own size (Girush 175 m).
- Engine-side check (import the exported FBX into Unity / Unreal) not done; Unity's own Humanoid
  retargeting is an alternative for Unity-only use.
- Retarget report error numbers come from the internal solve (tests measure the written action).
- From round 4: shot browser / fighters picker UI; adjacent opposite pins overshoot; close_boxing
  recall and false tracking hint; calibration on more shots; licences (training data, GeoCalib
  credit) before any sale.
