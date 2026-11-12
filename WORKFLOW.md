# Workflow: from a video clip to a finished animation in Blender

This walks through the whole path once: a fight video goes in, and two cleaned-up, animated
fighters in Blender come out, ready to export to a game engine, on afv's rig or on your own
characters. Steps 1–4 (project, shots, picks, the queue) are done in Blender's shot browser, with a WSL
command for each as the alternative; steps 5–6 are WSL commands you rarely need; steps 7–11 are in
Blender 5.2 on Windows (step 11 also has a WSL command).

Paths: your `D:\` drive is `/mnt/d/` inside WSL. The examples use the repo at
`D:fv` and a clip called `fight.mp4`.

## 0. One-time setup

**In WSL**, make `afv` a normal command (add this line to `~/.bashrc`, then open a new terminal):

```bash
alias afv=~/afv/afv-env/bin/afv
```

Check it works: `afv --help` should list `init`, `shots`, `pick`, `run`, and so on.

**In Windows**, build and install the Blender extension:

1. In PowerShell, build the zip (one line):

   ```powershell
   & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --command extension build --source-dir D:fv\blender\afv_addon --output-dir D:fv\dist
   ```

2. In Blender: **Edit > Preferences > Get Extensions**, the drop-down menu at the top right,
   **Install from Disk...**, then pick `D:fv\dist\afv-0.2.0.zip`.
3. In **Edit > Preferences > Add-ons > AFV**, set:
   - **WSL distribution**: `Ubuntu`
   - **afv command (in WSL)**: `~/afv/afv-env/bin/afv`
   - Leave **Use the afv serve worker**, **Re-import refreshes the take** and **Draw the
     confidence strip** on.

Rebuild and reinstall the zip whenever the add-on's code changes.

## 1. Create a project for the video

One project holds one source video and all the shots cut from it. The main path is Blender's shot
browser; every step also has a WSL command (below each step).

**In Blender**: open the 3D viewport's sidebar (**N**), tab **AFV**, panel **Shots**, and press **New
project from video**. Pick the video; in the file browser's side panel check the **Project folder**
(default: `<projects folder>\<video name>`, the projects folder is set in the add-on's preferences) and
leave **Scan the shots after** on. afv hashes and probes the video, finds the cuts (about 50 s for a
16-minute fight) and adds one **candidate** shot per segment, then scans them (step 2). The browser
opens in its own scene, **AFV Shots**: the video behind the camera, a marker on every cut, the shots as
a strip under the timeline's frame numbers. For a project that exists, use **Open project** (pick its
folder).

**In WSL**:

```bash
cd /mnt/d/afv/projects
afv init "/mnt/d/Videos/fight.mp4" --project fight --title "Fight night" --detect
cd fight
```

`afv init` hashes and probes the video; `--detect` finds the cuts and adds one **candidate** shot per
segment (`--title` and `--url` are optional, kept for provenance). Without `--detect`, add shots later
with `afv shots detect` or `afv shots add` (step 2). Every later command finds the project from the
current folder (or pass `-p <folder>`).

## 2. Choose the shots

A shot is one continuous camera take (no cuts) that you want animation from: both fighters fully
visible most of the time, a few seconds long. Only **queued** shots are run; candidates and skipped
shots are not. Shots made before shot states existed count as queued.

**In Blender** (panel Shots):

1. **Scan shots** (done for you after New project) finds the people on one frame of every shot (one GPU
   job, about 75 s for 126 shots): a thumbnail, the numbered people, and a **fighters likely** hint.
2. In **Shot list**, order by **Fighters likely first** (the `*` shots) and look at the thumbnails: the
   hint is often right but not always (on the sample, 32 of its 47 marks are fight action, and it
   misses 9 of 41 fight shots), so check the other shots too. Click a shot to jump to it; Space plays
   just that shot.
3. Trim it: put the playhead on the first good frame and press **Set in** (key **I**), on the last one
   and press **Set out** (**O**). Set **Speed** for slow-motion replays (0.25 = 4x slow motion; the export
   plays at real time) and add **Notes** if you like.
4. **Queue** the shots you want (**Q**), **Skip** the rest (**S**), or leave them candidates; each moves
   on to the next shot in the list. **Page Down** / **Page Up** go to the next / previous shot, **L** to
   the next likely one. The keys work with the mouse over the video or the timeline of the AFV Shots
   scene. These edits are saved to the project at once: Ctrl+Z does not take them back (do them again,
   e.g. Queue a skipped shot).
5. A cut the detection missed, or one exchange inside a long segment: put the playhead there and press
   **Add shot here** (a queued shot up to the next cut, else 5 s), then trim it.

**In WSL**: `afv shots scan` finds the people on one frame of each shot in a single GPU job (about 1-2
min for 120 shots) and saves a thumbnail (`media/scan/<id>.jpg`) and the numbered people. `afv shots
list --likely` lists shots where two fighters are probably visible. Trim with `afv shots set ID --start
3:02.5 --end 3:09`, choose with `afv shots queue 12,14-16` (or `afv shots skip ...`). Trimming a shot
makes its scan and picks stale: scan it and pick it again. Add a shot by hand (queued) with:

```bash
afv shots add --start 1:31.2 --end 1:36.8 --notes "right cross"
afv shots add --start 2:10 --duration 4 --speed 0.25 --notes "4x slow-mo replay"
```

Times are in the source, in seconds or m:ss.s; shot ids are `001`, `002`, and so on.

## 3. Pick the fighters

afv needs to know who the two fighters are in every queued shot.

**In Blender**: select a queued shot and press **Pick fighters** (key **Enter**; **Shift+Q** queues and
picks in one go). The shot's scanned frame shows with a numbered box on every person: click fighter **A**
(red), then fighter **B** (blue), then any extras you want tracked too (the referee helps afv keep
people apart when they overlap). Press **Enter** to save (**Backspace** drops the last click, **Esc**
cancels; the number keys choose by number). With **Then next** on, the next queued shot that still
needs a pick comes up right away and you keep clicking, so 20 shots take a couple of minutes. From a scan
this is instant and needs no GPU. Shots you trimmed (or never scanned) are scanned first, all in one GPU
job (about 20 s for 20 shots; the panel shows the wait), so trim first, then pick. If the scanned frame
is a bad one, put the playhead on a better frame and press **Pick on this frame** (GPU, about 35 s).

**In WSL**:

```bash
afv pick 001                             # the numbered people (from the scan: no GPU)
afv pick 001 --fighters 1,0              # fighter A is person 1, fighter B is person 0
afv pick 001 --fighters 1,0 --extra 2    # also track the referee (person 2)
```

Without a scan, `afv pick 001` (or `afv run 001`, which stops with exit code 3 until the fighters are
picked) finds the people first and prints the path of a **numbered prompt image** (from Windows: replace
`/mnt/d/` with `D:\`).

## 4. Run the queue

**In Blender**: panel **Run queue**, press **Run queue**. It first lists any queued shot whose fighters
are not picked (pick them, or **Run the N picked ones**). Then afv runs every queued shot, one after
the other: tracking, camera solve, 3D pose, pairing, foot contacts, smoothing, hands and export. The
panel shows the shot, stage and progress, and each shot's outcome. The GPU stages take a few minutes for
a 10 s shot, so a queue of 20 shots is an overnight job.

**The queue runs in WSL, not in Blender**: you can close Blender (or open another file) and the batch
goes on. afv reads the queue again after each shot, so shots you queue and pick while it runs are run by
the same batch. Open the project again later and Run queue shows where it is, or, once it has finished,
last night's results (each shot's outcome is also in its list row). **Stop batch** stops it (the shot
running now stops; the others stay queued).

**In WSL**: `afv batch` runs only the queued shots (`--state` or `--all` to choose others) and lists
those waiting for a pick or a tracking review. One shot: `afv run 001`. Stages already up to date are
skipped on later runs; `afv status 001` shows which stages are done or stale.

## 5. Tracking QC hints

A shot can stop after tracking with **QC hints** (outcome "needs a tracking review" in Run queue, exit
code 3 in WSL: a likely ID swap, a lost fighter, and so on). Check the review video and the hints, then
either fix the tracking or go on:

```bash
afv pick 001 --frame 140                     # numbered people on frame 140
afv pick 001 --swap 140:1:2                  # from frame 140, fighters A and B swapped identity
afv pick 001 --prompt 140:4:2 --label referee   # object 4 is detection 2 on frame 140
afv run 001                                  # re-check / re-track
afv run 001 --yes                            # or: the hints are wrong, go on anyway
```

You can also fix an ID swap later in Blender (step 8).

## 6. Build the take for Blender

```bash
afv take 001
```

This writes `shots/001/take/`: both fighters on the export rig, the solved camera, the video and
the QC data. You can skip this step, because the add-on runs it for you on import. To look without
the add-on, `afv blender 001` also writes `shots/001/take/take.blend`, which you can open with
**File > Open**.

## 7. Import into Blender

1. Open Blender, press **N** in the 3D viewport and select the **AFV** tab.
2. Press **Import take** and pick the shot folder:
   `D:fv\projects\fight\shots\001`.
3. You get two armatures (`001_A`, `001_B`) with coloured proxy meshes, and the camera
   `001_camera` with the video behind the fighters. The scene's frame rate, range and resolution
   match the shot.
4. Press **Space** to play. Look through the camera to compare the fighters with the video.

The coloured **confidence strip** under the frame numbers (Timeline, Dope Sheet, NLA, Graph) marks
problem spans: amber means warn, red means bad. Use **Previous / Next issue** to jump between them;
the panel lists the issues at the current frame (low confidence, reprojection error, foot slide,
floor penetration, pops).

## 8. Fix the solve

Work through the issues. Every fix below is saved to the shot, re-solves in WSL in a few seconds,
and replaces only the solved motion in Blender. Your camera view, frame and hand-keyed tweaks stay.

| Problem | Tool (AFV sidebar) | What to do |
|---|---|---|
| Foot slides when it should be planted, or sticks when it should lift | **Contacts** | Select a foot or leg bone (Pose Mode), set the frame range, press **Plant** or **Free** (**Clear** lets the solver decide again). |
| A pose is wrong at a frame | **Pins** | Pose the bones in Pose Mode (don't key them), select them, press **Pin pose**. The solve passes through that pose and blends back over **Radius** frames. |
| Motion is jittery, or too smooth | **Smoothing** | Move the global or per-region sliders (torso, head, arms, hands, legs, root), then **Apply**. |
| Hands look wrong | **Hands** | Select the fighter, pick Both/Left/Right, set the range, press **Hands off** (relaxed hand). |
| The fighters swap identity partway through | **Tracking** | Go to the first wrong frame, press **Swap IDs** and click the two boxes. This re-runs 3D pose on the GPU, so it takes minutes. |
| Something looks off after other fixes | **Re-run range** | Pick a stage and re-solve the range, or press **Whole shot**. |

Set the edit range with Start/End in the panel (the arrow buttons use the current frame), or with
the timeline's preview range (Ctrl+P), which takes priority.

**Editing the files by hand.** The tools write `shots/001/edits/*.json` for you. If you edit one by
hand, write frames, radius and ids as plain whole numbers (`12`, radius at least 1) and rotations and
positions as plain numbers: no quotes, no `true`/`false`, no `NaN`. `afv run`, `afv status` and the
add-on refuse a file with a wrong value and name the field, so you know what to fix. `fighters.json`
is checked the same way: a hand-edited `"fighters": ["1", "0"]` is refused; pick again with
`afv pick` instead.

**To undo a fix, use Revert, not Ctrl+Z.** Ctrl+Z only undoes the re-import in Blender, and the
shot keeps the edit. The fix list in the main AFV panel has **Revert** for each fix and **Revert
last fix**.

## 9. Hand-polish with the tweak track

For small changes the solver can't make (a sharper punch, a cleaner guard), key them by hand on
top of the solve:

1. Open the **NLA editor** and select the strip on the **AFV tweaks** track of the fighter.
2. Press **Tab** (tweak mode).
3. Pose bones in Pose Mode and insert keys (**I**). Keys are relative to the solved pose.
4. Press **Tab** again to leave tweak mode.

Tweak keys survive re-solves and re-imports, because those only replace the **AFV solved** track.
Do the solver fixes (step 8) first, then hand-polish.

## 10. Export the animation

In the AFV panel, pick a **preset** (`unity_humanoid`, `unity_generic`, `unreal`, `godot`,
`blender`), optionally a **Copy to** folder, and press **Export**. Files (FBX, GLB, BVH and a JSON
side file per fighter) land in `shots/001/exports/` and are copied to **Copy to** if set. If it
pauses on tracking hints nobody reviewed, check them and press **Export anyway**.

The same from WSL:

```bash
afv export 001 --preset unreal --fps 60
```

The preset and frame rate are saved to the shot, so later runs export the same way.

Export uses the pipeline's solve, including your contact paint, pins, smoothing, hands and ID
swaps. **It does not include hand-keyed tweak-track keys** (step 9). To ship those, retarget onto
your character (step 11), or export from Blender itself (**File > Export > FBX**, with the
fighter's armature selected).

## 11. Put the motion on your own character

In Blender, open the **Retarget** sub-panel (AFV tab):

1. **Import character** and pick your character: `.fbx`, `.glb`, or a `.blend`. From a `.blend` only
   its armature and the meshes on it are appended (with the data they use, such as materials and the
   character's own actions); the file's scripts are not read, and its Python drivers are muted so
   they cannot run. Or pick an armature already in the scene as **Target**. The scene keeps the
   take's frame rate.
2. The bone map is guessed on import (**Guess map** does it again). Check the list: required bones
   that are not mapped show in red. Click a target bone to change it (search by name). **Save** the
   map as `<character>.afvmap.json` to reuse it; **Load** reads one back. The map is also stored on
   the armature, so a saved .blend keeps it.
3. Pick fighter **A** or **B**, leave **Lock feet** on, press **Retarget** (not in Edit Mode). The
   character gets a new action with the fighter's motion, tweak keys included; the report under the
   button shows foot slide, floor penetration and notes. **Ctrl+Z** undoes it. Retargeting the same
   fighter again replaces that action while it is still the character's active action, unchanged
   and with its name; rename or edit it to keep it, and the next Retarget makes a new one.
4. Pick a **Preset** and press **Export character**: an FBX of the character's armature, meshes and
   the new action, by default in `shots/001/exports/retarget/001_A_<character>.fbx`. The clip in the
   file is called `001_A_<character>` too. The preset gives only the FBX exporter settings: no root
   bone is added, no root-motion mode applies, the armature keeps its name and nothing is rescaled
   for Unreal. The character keeps its own size (a character made in centimetres without a unit scale,
   like the Girush sample, comes out 100 times too big: scale it in the engine). If the panel warns
   about the character's rest matrices, press **Fix rest matrices**, then **Retarget** again (the fix
   changes how existing actions play, slightly), then export.

The same without opening Blender, from WSL (it builds the take, imports the character, retargets
both fighters and writes one FBX each, both in the take's world):

```bash
afv retarget 001 --character "/mnt/d/Characters/mech.fbx"                 # both fighters, guessed map
afv retarget 001 --character mech.fbx --save-map mech.afvmap.json         # also write the map used
afv retarget 001 --character mech.fbx --map mech.afvmap.json --role A --preset unreal
afv retarget 001 --character hero.blend --out /mnt/d/Game/Animations --check
```

Files: `shots/001/exports/retarget/001_<A|B>_<character>.fbx` (or `--out`). `--preset` works as in
the panel: only the FBX exporter settings (`afv retarget --list-presets` prints them). If a required
bone can't be guessed, afv stops (exit code 1) and names it: write the guess with `--save-map`, fill
in the bone in the JSON, and pass it with `--map`. `--no-lock-feet` skips the foot locking;
`--check` re-imports each FBX in Blender and prints how far it is from the retarget (mm at the
fighter's scale, and in the character's own units). Like `afv take`, it warns when the shot's
results are older than its settings or code. Measured on a 10 s shot, both fighters: about 10 s per
character (17 s for the Unity mech with 124 mesh parts).

## Doing many shots

Queue them (step 2), pick them (step 3) and run the queue (step 4) overnight; then review each shot in
Blender. In WSL: `afv batch` runs every queued shot; shots waiting for a pick or a tracking review are
skipped and listed. Pick those shots, run `afv batch` again.

## Quick reference

```bash
# Blender: AFV > Shots > New project / Open project > Scan > trim, Queue > Pick fighters > Run queue
afv init <video> --project <name> --detect   # 1. project and candidate shots
afv shots scan | afv shots list --likely     # 2. people per shot; likely fight shots
afv shots set ID --start S --end E | afv shots queue IDS | afv shots skip IDS | afv shots add ...
afv pick 001 [--fighters A,B]                # 3. pick (from the scan: no GPU)
afv batch                                    # 4. every queued shot (or: afv run 001)
afv run 001 [--yes]                          # 5. after fixing tracking hints
afv take 001                                 # 6. take (optional: the add-on does it)
# Blender: AFV > Import take > fix > tweak > Export   # 7–10
afv export 001 --preset <preset>         # or export from WSL
afv retarget 001 --character <file>      # 11. onto your own character (or the Retarget panel)
afv status 001 | afv shots list | afv provenance 001
```

More detail: the add-on's tools are in `blender/afv_addon/README.md`; the edit file formats are
described in the docstrings of `src/afv/core/edits.py`.
