# Round 6 notes (2026-11-11): shot browser

Assignments, shared formats (shot states, `cuts.json`, candidate scan, CLI and serve commands, the
detached batch) and acceptance numbers: `docs/BUILD_PLAN.md`, "Round 6 assignments". New this round:
**two review passes per track** (owner decision; AGENTS.md section 6). Q and B each had two
independent reviews; after the second pass the PM had the blockers fixed and checked them itself
(Q: a code change mid-batch froze serve, `batch_cancel` could signal a stale pid; B: the I/O/S keys
didn't work and could key or scale the browser camera). Everything else left is in the open list.

Test source: the owner's `Media_Samples/Full Fight.mp4` (16:21, 720p, 29.97 fps). `projects/full_fight`
holds the starting state: 126 candidate shots, `media/cuts.json`, a complete `media/scan/`, nothing
queued or picked.

## Q: queue, scan and browser backend (`src/afv/core/project.py`, `cli.py`, `serve.py`, `media/shots.py`)
- Shot states `candidate` / `queued` / `skipped` (missing = queued, so older projects are unchanged);
  `afv shots detect` adds candidates and writes `media/cuts.json`; `afv shots set / queue / skip`;
  `afv batch` runs queued shots only.
- Candidate scan (`afv shots scan`, serve `shots_scan`): one RF-DETR worker for all shots, frames made
  with the normaliser's own commands (byte-identical to `prepare_frames`, verified on all 126), so the
  stored numbered people are exactly what `afv pick` would detect; picking from a scan needs no GPU.
  `fighters_likely` (computed when the list is read): 47 of 126 marked, 32 of them fight action, 9 of 41
  fight shots missed (mostly clinches): a hint for sorting, never a filter.
- Picks never use a prompt from an old interval or detection threshold; trimming marks scan and picks stale.
- The batch runs detached (`afv batch --json` in its own session, one per project): it survives Blender
  closing, the worker restarting and a code-change restart, logs to `media/batch/<id>.jsonl`, re-reads
  the queue after each shot (shots queued during the night run), stops cleanly on SIGTERM;
  `batch_status` / `batch_cancel` from any client. Lost-update races in shot.json and shot ids fixed
  with short locks; one bad shot.json is an error row, not a failed list.
- Measured on Full Fight: init + detect 64-65 s, scan of 126 candidates 69-85 s, pick from a scan
  0.07-0.4 s, serve `shots` ~0.23 s, `shot_set` 0.05-0.45 s.

## B: shot browser in Blender (`blender/afv_addon/browser*.py`; add-on 0.2.0)
- AFV > Shots: New project from video / Open project; the source as a movie clip behind the browser
  camera (no proxy needed for 720p: jump-to-frame 9-79 ms, 30 fps playback); cut markers (toggle); a
  gpu shot strip under the timeline; the shot list (thumbnail, range, cut confidence, state,
  fighters-likely, pick status, last run) sorted by time or fighters-likely first.
- Set in / out from the playhead (the frames the owner sees, checked against the normaliser), speed,
  notes, Queue / Skip (auto-advance), Add shot here, Queue & pick; Pick fighters on the scan frame or
  "on this frame" (click A, B, extras; digits; Enter; Then next keeps going); stale or unscanned shots are
  scanned in one job before picking.
- Keys in the AFV Shots scene: I / O, Q / S, Shift+Q, Page Down / Up, L (next likely), Enter (N stays
  Blender's sidebar). The browser camera can't be keyed or moved.
- Run queue: checks picks first, starts or reattaches to the detached batch, per-shot progress, Stop,
  last night's results on open; Import take into a marked scene (Show take).
- File loads, add-on disable and undo never stop a batch or scan or desync the list from the project;
  project edits are saved at once (Ctrl+Z doesn't take them back).
- Measured: ready 6-8 s after Open project (cold worker; < 1 s warm); edits 0.06-0.46 s; pick shown
  0.11-0.21 s, saved 0.06-0.2 s; scripted end to end (open, scan 126, queue 20, pick 20, start)
  87-92 s, 111 s with every shot trimmed (target <= 180 s); about 150-190 actions for 20 of 126 shots.

## Integration (PM)
- Full suite: **2074 passed, 15 skipped** (round 5: 1972 / 11; the new skips are opt-in GPU and windowed
  tests: `AFV_GPU_TESTS=1`, `AFV_WINDOWED_TESTS=1`), ruff clean.
- Golden: all three shots pass against round 5's values (no pipeline result changed; not regenerated).
- Sample projects: every stage still fresh (no stage code hash changed), so they were not re-run.
- `projects/full_fight`: 126 candidates, `cuts.json`, complete scan, nothing queued or picked.
- The add-on is now 0.2.0: rebuild and reinstall the zip (`dist/afv-0.2.0.zip`).
- Owner test (first run as a new user): New project from video failed with "cannot map ... to a WSL
  path" when the Project folder field held a plain new name. Fixed by the PM (`browser_core.project_folder`:
  a plain or relative name goes under the projects folder; 9 test cases). The same test found the Import
  take bug below.
- New `HOLD_MY_HAND.md`: a guide a new user points their agent at (computer check, install in order with
  the scripts adapted to the machine, the add-on, a first project).

## Open for round 7
- **Bug, found by the owner: a take imported inside the shot browser's own .blend doesn't play.** The
  shot's scene (`AFV <id>`) shows both skeletons but they don't move when scrubbing or playing. The same
  take opened in a new file plays correctly, and building it into a second scene headless animates too
  (bones move up to 13 cm over 80 frames), so the data and `builder.import_take` are fine. Suspects: the
  timeline driving the browser scene instead of the take scene (Blender 5 sequencer-scene time sync), or
  the browser's handlers (frame change, undo re-sync, Loop shot preview range). Reviews tested Import take
  with stub takes only. Workaround (in `HOLD_MY_HAND.md`): open takes in a new file (AFV > Import take,
  or `afv blender <id>`). Fix it first, with a windowed test that plays a real take in the browser's file.
- **Engine-ready export** (the owner's next priority): root bone / root-motion modes and engine presets
  for retargeted characters, real-world size, batch retarget of many shots onto one character, and an
  automated engine import check. The owner targets several engines: ask which first.
- Cut detection: some hard cuts are 1-3 frames late or false (e.g. a light flare at frame 15800 of Full
  Fight); no progress during detection; `duration` stored with float noise.
- `fighters_likely` misses clinches (detected as one box) and marks walkouts / corner shots.
- Lock files pile up in `~/afv/locks` (`shots-*`, `shotinfo-*`, `batch-*`), and `tests/media`,
  `tests/serve` write there; two simultaneous batch starts can both try; the GPU-wait message can name
  a stale holder; `batch_status` has no finish time (the browser uses the log's mtime).
- Not tested end to end yet: a real overnight batch to export from the browser, File > Revert mid-batch,
  "Pick on this frame" on the GPU, Windows sleep / WSL shutdown during a long batch.
- The owner should try the keyboard shortcuts in a real session (tested with simulated key events).
- From round 5: plant-edge flips (~1.5 cm root path), close_boxing A right heel plant floats 2 cm,
  GEM-X depth drift in `pair.distance_median_m`, retarget report metrics from the internal solve.
- From round 4: adjacent opposite pins overshoot; close_boxing recall and false tracking hint;
  calibration on more shots; licences (training data, GeoCalib credit) before any sale.
