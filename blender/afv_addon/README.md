# AFV add-on for Blender 5.2

Review and clean up an afv take in Blender: both fighters on the export rig, the solved camera with
the shot's video behind it, and the QC confidence strip (coloured bars in the timeline plus
markers). From the sidebar you can jump between issues, fix the solve (paint contacts, pin poses,
smoothing sliders, re-run a frame range, hands off, swap tracking IDs), re-import and export with an
engine preset, and retarget a fighter onto your own character (tweak keys included) and export it.
Each fix re-solves only what it needs in WSL through one `afv serve` worker and re-imports the result
in a few seconds.

Before that, the **Shots** panel (the shot browser) turns a long video into chosen shots: it shows the
video with its detected cuts, lets you trim, queue or skip each candidate shot, pick the fighters by
clicking them, and runs the queue in the background.

The take's own **Export** runs the pipeline's export, so it does not include your tweak-track keys;
**Retarget** + **Export character** does (onto your character).

## Without installing anything: `afv blender`

In WSL, inside the project folder:

```bash
afv blender 001            # writes shots/001/take/ (the package) and shots/001/take/take.blend
afv blender 001 --check    # also prints joints vs take (mm) and overlay vs the pipeline (px)
```

Open `take.blend` with File > Open. The viewport looks through the take camera, so the video shows
behind the fighters. Press Space to play.

## Install the add-on

1. Build the extension zip (Windows PowerShell; one line, works from any folder):

   ```powershell
   & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --command extension build --source-dir D:fv\blender\afv_addon --output-dir D:fv\dist
   ```

   This writes `dist\afv-0.2.0.zip` in the repo (`dist/` is git-ignored). Rebuild after the
   add-on's code changes, then install the new zip over the old one.
2. In Blender: Edit > Preferences > Get Extensions > the drop-down menu (top right) > Install from
   Disk..., pick the zip. Or from the command line:
   `blender --command extension install-file -r user_default -e D:fv\dist\afv-0.2.0.zip`.
3. In the add-on's preferences (Edit > Preferences > Add-ons > AFV) check:
   - **WSL distribution**: `Ubuntu`
   - **afv command (in WSL)**: `~/afv/afv-env/bin/afv`
   - **Re-import refreshes the take**: on (Re-import runs `afv take` first).
   - **Use the afv serve worker**: on. The add-on starts one `afv serve` in WSL on the first request
     (no console window) and keeps it for the Blender session; it is restarted if it dies (up to 3
     times a minute). Off, or when `afv serve` cannot start (an older afv), every request runs as a
     one-shot `afv` command instead (slower: afv is imported each time). The line under the option
     shows the worker's state; **Restart worker** stops it (the next request starts a new one).
   - **Draw the confidence strip**: on (the coloured bars; the markers stay either way).
   - **Projects folder**: where **New project from video** makes projects (e.g.
     `D:fv\projects`); empty: next to the video.

## Shot browser (panel Shots)

The **Shots** panel is the first one in the AFV tab. It works on one afv project (one source video) at
a time, in a scene of its own, **AFV Shots**: your take scenes are not touched.

- **New project from video**: pick the video; in the file browser's side panel set the **Project
  folder** (empty: `<projects folder>\<video name>`), a **Title**, and **Scan the shots after** (on).
  afv probes and hashes the video and detects the cuts (one **candidate** shot per segment; about 50 s for
  a 16-minute fight), then scans the shots (below). All in WSL in the background; the progress is in
  the panel. A source taller than 720 px also gets afv's 720p proxy (`media/proxy.mp4`), which the
  browser then shows.
- **Open project**: pick the project folder (the one with `project.json`). About 6-8 s the first time
  in a Blender session (the `afv serve` worker starts), under 1 s after that. It also shows the
  project's batch: a running one is followed live, a finished one's results are listed (below).
- The scene: the source video behind the camera of the 3D viewport (a movie clip; scene frame f is
  source frame f, at the source's frame rate), a timeline marker per cut (`cut 86%` / `fade 28%`, on the
  first frame of the segment after the cut; for a fade that is where the fade ends; **Cut markers** hides
  them), and a strip of the shots under the timeline's frame numbers (grey candidate, green queued, dark
  red skipped, a yellow tick where fighters are likely, the selected shot outlined). Scrub and play as
  usual: a jump shows its frame in 9-80 ms on the 720p sample with no proxy, and playback holds 30 fps.
  When the project's video is missing, the panel says so (in red) and the timeline shows no video.
- **Scan shots** finds the people on one frame of every candidate and queued shot (one GPU job: about
  75 s for the 126 shots of the 16-minute sample); each row fills in as its shot is scanned. Shots with an
  up-to-date scan are kept; **Scan all again** redoes them. The X next to the progress stops it (what
  was scanned is kept).
- **Shot list**: per shot a thumbnail, the id, start, length and cut confidence (`*` = **fighters
  likely**), then the state, the people found, the pick status and the last batch's outcome. Filter by
  state; order by **Time** or **Fighters likely first**; **Previous** / **Next** / **Next likely** walk the
  list. Fighters-likely is only a hint from the scan (on the sample, 32 of its 47 marks are fight action
  and it misses 9 of 41 fight shots), so it sorts and highlights, it never hides a shot: look at the
  thumbnails. Click a shot to select it and jump to it; with **Loop shot** on, the timeline's preview
  range becomes the shot (Space plays just the shot, the timeline frames it). A shot whose `shot.json`
  afv cannot read is listed in red with the error, whatever the filter: fix or delete that file, then
  **Refresh** (also after using afv commands outside Blender).
- **Shot** (the selected shot): a large thumbnail, its range, state, people, pick status, last run.
  - **Set in** / **Set out**: the frame under the playhead becomes the shot's first / last frame. Trim
    to the fighting (both fighters visible, no cut inside). A trimmed shot's scan and picks are stale.
  - **Speed** (0.25 = a 4x slow-motion replay, exported at real time) is sent once the slider rests;
    **Notes** when you leave the field.
  - **Queue** (Run queue runs it), **Skip**, **Candidate**. With **Then next shot** on, the next shot in
    the list's order is selected right after.
  - **Pick fighters**: the shot's scanned frame is shown with a numbered box on every person. Click
    fighter **A** (red), then **B** (blue), then any extras to track too (referee, yellow); click a
    chosen box again to drop it; the number keys choose by number too. **Enter** (or **Save pick**)
    saves, **Backspace** drops the last one, **Esc** (or right-click in the frame) cancels. The pick stays
    on screen until afv has saved it: if afv refuses it, the message says why and your clicks stay.
    From a scan the people show in about 0.1 s and the save takes 0.1-0.4 s, no GPU. A shot without an
    up-to-date scan (new or trimmed) is not detected on its own (about 35 s of GPU start per shot): Pick
    scans it together with every other queued shot that needs a pick and a scan, in one GPU job (20
    trimmed shots: 20 s), and shows the wait. **Pick on this frame** finds the people on the playhead's
    frame instead (GPU, about 35 s). With **Then next** on, saving moves on to the next queued shot that
    still needs a pick, in the same session (keep clicking). Picking again later starts from the
    current pick when it was made on the same frame.
  - **Queue & pick**: queue the shot and pick it in one go.
  - **Import take**: shown once the shot has a pose3d result or an export. It opens the take in a scene
    of its own, `AFV <shot>` (written by afv first when it is missing or older than the shot's results;
    built into that scene; opening it again shows the same scene). When afv had to write it first, the
    window switches to the take only if you are still where you pressed Import take; otherwise the panel
    offers **Show take**. **Show shot browser** in the Shots panel goes back.
  - **Add shot here**: a new queued shot from the playhead to the next cut (if that is 0.5-10 s away,
    else 5 s); trim it with Set out. For a part of a long segment, or a cut the detection missed.
- **Keys**, with the mouse over the video or the timeline of the AFV Shots scene (the panel lists them;
  in other scenes, and over other editors, the keys keep Blender's meaning): **I** / **O** set in / out,
  **Q** / **S** queue / skip (then the next shot), **Shift+Q** queue & pick, **Page Down** / **Page Up**
  next / previous shot, **L** next likely shot, **Enter** pick fighters, **Space** play. **N** stays
  Blender's (it shows and hides the sidebar, where the AFV panels are). In that scene Blender's own I
  (insert keyframe) and S (scale) are replaced over the video; the browser's camera is locked and cannot
  be selected, so no key can move or key it.
- **Run queue**: lists the queued shots whose fighters are not picked yet (click to select, the arrow to
  pick; **Scan the N unscanned first** when some need a scan) and runs nothing until they are picked
  (**Run the N picked ones** runs exactly those). Then `afv batch` runs every queued shot, shot after
  shot, up to **Run until** (default export), with the progress (shot n of N, stage, percent) and each
  shot's outcome (ran ok, needs a pick or a tracking review, failed: click it to select the shot).
  - **The batch runs in WSL, not in Blender**: closing Blender, opening another file, disabling the
    add-on or a worker restart does not stop it. afv reads the queue again after each shot, so shots you
    queue and pick while it runs are run by the same batch. **Open project** follows it again (afv streams
    its log); a batch started from WSL (`afv batch`) shows the same way. When it has finished, Open
    project lists its results: `last batch <id>, finished <time>: 5 ran ok, 1 needs a tracking review`,
    each shot clickable, and each shot's outcome in its list row. **Stop batch** (or the X) stops it: the
    shot running now stops, the others stay queued, and the panel says how many shots ran and how many
    did not. (With one-shot afv commands, when the worker is off, the batch is a child of Blender and
    closing Blender stops it.)
- Every change is sent to afv and the row afv sends back replaces the list's row: Set in / out, Queue /
  Skip, Add shot and a pick took 0.06-0.46 s each on the sample. Requests run in the background and the
  scan or the batch keeps running while you trim, queue and pick. Without the worker (preferences), the
  one-shot afv commands do the same, about 1-1.5 s each.
- **Ctrl+Z does not undo project edits.** Set in / out, Queue / Skip / Candidate, Speed, Notes, Add shot
  and picks are saved to the project at once and stay; to change one back, do it again (for example Set
  in at the old frame, or Queue again). Ctrl+Z right after them steps back through the browser's own
  Blender-side changes (the selected shot, the preview range, the panel's settings), and after any undo
  the panel shows the project as it is. Opening the browser is one undo step; undoing past it closes the
  browser (Open project brings it back).

## Use

The panel is in the 3D viewport's sidebar (N), tab **AFV**.

- **Import take**: pick the shot folder (`projects\<name>\shots\<id>`) or its `take` folder. If the
  shot has no take yet, or the take is older than the shot's results (a stage re-ran since), the
  add-on runs `afv take` in WSL in the background first and imports when done. Each fighter is an
  armature (`<shot>_A`, `<shot>_B`) with a coloured proxy mesh; the camera is `<shot>_camera`, and the
  scene gets the shot's fps, frame range and resolution. If `afv run` would redo some of the shot's
  results (settings or code changed), the status line says so after the import.
- **Each Import is its own copy** (collection `AFV <shot> [<id>]`): importing the same take twice
  gives two independent copies, each with its own camera. Re-import and Export act on the copy of
  the selected object, or on the last imported/re-imported copy when nothing of a take is selected.
- **Re-import**: replaces only the solved motion (and camera, video, markers) of that copy. Your
  corrections stay (see below), and the tweak track is stretched if the take got longer. If the bone
  lengths changed, the armature's rest pose is updated in place; if the rig changed (another
  preset), Re-import refuses: use Import take for a new copy. It refreshes the take with the same
  preset the copy was made with.
- **Previous / Next issue**: jump between the QC spans. The panel lists the issues at the current
  frame. Marker names are `<fighter> <kind> <worst value>`, with `!` when bad:
  - `low conf`: median keypoint confidence of the frame (warn < 0.35, bad < 0.2);
  - `reproj`: median reprojection error vs GEM-X's 2D keypoints (warn > 25 px, bad > 45 px at 720p);
  - `slide`: planted foot speed (warn > 5 cm/s, bad > 15 cm/s);
  - `floor pen`: sole below the floor (warn > 1 cm, bad > 3 cm);
  - `pop`: joints with a single-frame pop.
- **Export**: pick a preset and press Export. `afv export <shot> --preset <preset>` runs in WSL in
  the background (the UI stays responsive; the status line shows progress). Files land in the
  shot's `exports` folder and are copied to **Copy to** if set. The preset is saved to the shot, as
  with the CLI. If the tracking QC raised hints nobody reviewed yet, afv pauses: the panel shows the
  hints, and **Export anyway** runs it again with `--yes` (or fix the tracking with `afv pick`).
- **Cancel** (the X next to the status line) stops the running `afv` command in WSL. A command still
  runs if you open another file meanwhile, but its result is then not applied to the new file.
- Every result applied from WSL is one undo step.

## Cleanup tools (sub-panels under AFV)

Every tool writes its edit file in the shot (`shots/<id>/edits/*.json`) and/or
stage settings, re-solves in WSL (`afv run --from <stage> [--frames a:b] --to hands_face`, then
`afv take`) and re-imports. Re-solves of the motion stages (contacts, smooth, hands_face) replace
only the solved motion: camera, video, current frame, Pose Mode and NLA tweak mode stay. The panel
shows how long the last fix took per tool. Only one cleanup request runs at a time (Cancel is the X
next to the status line); the shot browser's requests run beside them.

Edit files are checked strictly, by the add-on and by `afv run` alike. Frames, radius and object ids
must be whole JSON numbers (`12`, not `12.0`, `"12"` or `true`), a pin's radius is at least 1, and
rotations and positions must be finite numbers; the pin keys `root_rotation` (a unit quaternion),
`prefix` and `root_motion` (text) are checked too. A file that breaks a rule is refused with the
file, the field and the value, for example `edits/pins.json: pins[2].bones.LeftArm[1]: must be a
finite number, got 'x'`. Until that entry is fixed, the add-on's tools for that edit and `afv run`
both stop: fix or delete the entry in the file and try again.

**Ctrl+Z does not undo a fix in the shot.** It undoes the re-import in Blender, but the edit file
(or the smoothing settings) in the shot stays changed, and the next re-solve or re-import shows it
again. To take a fix back use the fix list in the main AFV panel (newest first): **Revert** puts
the edit file, or the smoothing settings set at shot level, back as they were before that fix
(settings that were not set at shot level are removed again, not written as defaults) and
re-solves. A fix whose file or settings changed since (a later fix of the same file, a hand edit, a
setting changed elsewhere) is marked "changed since" and cannot be reverted: **Forget** drops it,
and the older fixes stay revertable (**Revert last fix** takes the newest one that still can be).
The last 30 fixes are kept in `shots/<id>/.addon/history.json` (each with the state before it and a
hash of the state after it). A smoothing Apply is recorded even when its run pauses on tracking hints
or fails, since the settings are saved before the stages run. A fix whose re-solve fails (for
example a pin the smoother rejects) shows the error in the same box: revert it, or remove that pin /
span from its list, so a bad edit never blocks later fixes.

The **edit range** (Contacts, Hands, Re-run range): Start/End in the panel (the arrow buttons set
them to the current frame), both frames included; or the timeline's preview range when it is on
(Ctrl+P / Alt+P in the timeline), which then wins. It is clipped to the take's frames.

- **Contacts**: select a foot or leg bone of a fighter (Pose Mode), set the range, press **Plant**
  (foot on the floor), **Free** (in the air) or **Clear** (the solver decides again). Writes
  `edits/contacts.json` (overlapping earlier paint of that foot is cut), re-runs contacts + smooth on
  that range. About 4.6 s for a 2 s range on ufc_cross. The panel lists the painted spans (the
  selected fighter's first): click one to jump to it, X removes it and re-solves its frames.
- **Pins**: in Pose Mode pose bones with Blender's tools (don't key them), select them and press
  **Pin pose**: their local rotations at the current frame (and the hips position when the hips are
  selected) go to `edits/pins.json` (canonical bone names, with the take's preset, root motion and
  bone prefix); smoothing re-runs (whole shot, ~3 s) and passes through the pose, blending over
  +-**Radius** frames. The pin means "the final pose looks like this": every layer above the solve
  (the "AFV tweaks" track, tracks you pushed down above "AFV solved", the active action) is taken
  out of it (pin = shown x layers^-1), so after the re-solve solve x layers is again the pose you
  pinned. So pinning a pose that comes from keys in a layer changes nothing: the solve already is
  that pose's base; pin unkeyed poses. All those layers must blend with Combine (the add-on's
  default); a pin is refused with a message if one blends otherwise (mute it or set Combine).
  With a root bone (unity_generic, unreal), the pin also records the root rotation you saw.
  Pinning the same fighter at the same frame again adds the new bones to that pin (new values win;
  its hips position stays unless you pin the hips again). Pins use the bones' own transforms:
  constraints you add to the rig are not part of a pin. The list shows each pin (click to jump),
  how far the solve is off
  it and any conflict (e.g. a leg pin on a planted frame loses to the contacts: paint the foot Free
  first); X deletes a pin. A separate root bone cannot be pinned: pin the hips. Finger pins are
  currently overridden by the fist / hands-off pose.
- **Smoothing**: global strength and per-region sliders (torso, head, arms, hands, legs, root; 1 =
  default, up to 4 meaningful). **Apply** saves them to the shot (`smooth.strength`,
  `smooth.region_strength.<region>`, like `afv run --set`) and re-runs smoothing: ~3.1 s for the 10 s
  ufc_cross shot. **Reset** puts the sliders back to 1, **From shot** loads the saved values (done on
  Import too).
- **Re-run range**: pick a stage and press **Re-run range** (or **Whole shot**). contacts, smooth and
  hands_face re-solve just the range, blended into the previous result at the edges; earlier stages
  run on the whole shot (pose3d and detect_track take minutes) and then everything is re-imported.
- **Hands**: select a fighter, pick Both/Left/Right, press **Hands off** over the range (blend to a
  relaxed hand: `edits/hands.json`) or **Hands on** to undo it; re-runs hands_face there. The spans
  are listed like the contacts (jump, X to remove).
- **Tracking**: **Show boxes** draws detect_track's boxes over the camera view (id, fighter A red, B
  blue, others grey with their label) of the selected take. **Swap IDs**: go to the first wrong frame,
  press it (the viewport looks through that take's camera), click the two boxes inside the camera
  frame (Esc, or right-click inside the frame, cancels; clicks outside the frame, on sidebars or in
  other editors work as usual). The scene camera is set back when the pick ends. The swap goes to `edits/ids.json` (from
  this frame to the end) and the shot re-solves from detect_track: pose3d runs again on the GPU, so
  this takes minutes. The list shows the swaps (click to jump, X removes one and re-solves).
- A re-solve that stops on tracking QC hints shows them with **Run anyway** (`--yes`).

### The confidence strip

The Timeline, Dope Sheet, NLA and Graph editors show one thin row per fighter just under the frame
numbers: amber = warn, red = bad (the same spans as the markers and Next issue).

### Fixing a pose: the tweak track

Each armature has two NLA tracks: **AFV solved** (the solved motion, blend Replace) and **AFV tweaks**
above it (an empty action, blend Combine: rotations multiply, locations add). To correct a pose:

1. Open the NLA editor, select the strip on **AFV tweaks**, press Tab (tweak mode).
2. Pose the bone in Pose Mode and insert keys (I). Keys are relative to the solved pose.
3. Tab again to leave tweak mode.

Re-import swaps the action on **AFV solved** only, so the tweak keys survive a re-solve. Keys made
without tweak mode go into a new active action that is also combined on top of the solve (the
armature's action blend is Combine), so they are relative too; push it down to a track to keep it
separate from later keys.

### Other things in the scene

- Plant masks: each armature has animated custom properties `afv_plant_<channel>` (1 = planted),
  visible in the Graph Editor / Dope Sheet.
- The floor (afv's y = 0) is Blender's ground grid (z = 0).

## Retarget: the motion on your own character (sub-panel Retarget)

Puts a fighter's motion, as you see it (the solve plus the tweak track and any active action), onto
any humanoid armature, as a new action, and exports it as FBX. The maths is in `retarget.py`.

1. **Target**: pick the character's armature (the take's own fighters are not offered), or **Import
   character**: an `.fbx`, `.glb`/`.gltf`, or a `.blend`. The character lands in its own collection
   `Character <file>`. The scene keeps the take's fps and frame range (the FBX importer would set the
   file's fps; the status line says so). A newly imported character gets a guessed map at once.
   - From a `.blend`, the armature with the most bones is appended with its child objects and the
     meshes it deforms; the file's other objects are not kept. The data those objects use comes along
     (meshes, materials, the character's own actions); data only the dropped objects used stays as
     orphan data, which Blender does not save unless it has a fake user.
   - What can run: nothing from the file. Its text blocks and registered scripts are never read, and
     every driver in what is appended whose expression needs Python is **muted** before anything
     reaches the scene, so it cannot run even with Auto Run Python Scripts on (the status line says
     how many; unmute them in the Drivers editor only if you trust the file). Simple expressions
     (`var * 2`, ...) keep working: Blender evaluates them without Python. The mute is saved with
     your file. Data that the character's file links from yet another file cannot be changed, so a
     file that links data carrying such drivers is refused (the message names the data and its
     file: make it local in the character's file, or remove those drivers).
   - A file whose bones' stored rest matrices carry a slight scale (Girush: 0.12 %) gets them
     re-derived from head, tail and roll, as Edit Mode does; the bones do not move (the file's own
     actions then play very slightly differently, as after Fix rest matrices below).
   - An armature that was hidden or unselectable in its file is shown and made selectable (the
     status line says so).
   - A damaged .blend, or one from a newer Blender, is refused with a message; nothing is left behind.
2. **Guess map** fills the bone list from the bone names and the hierarchy (Mixamo, Unity Humanoid,
   Unreal, Biped, Rigify metarigs, anatomical names; fingers best effort). The list shows every afv
   bone (`*` = required; **Fingers** shows the finger bones too; the filter searches by name) with the
   character's bone next to it: click it to pick another bone (search over the armature's bones), or
   clear it to leave that bone unmapped (it keeps its rest pose). Above the list, in red: required
   bones not mapped, names the armature does not have, a bone used twice.
3. **Load** / **Save** a map as `*.afvmap.json` (a small JSON file: bone names of afv's rig -> your rig's; use it
   with `afv retarget --map`). The map is also kept on the armature (custom property
   `afv_retarget_map`), so a saved .blend remembers it and picking the armature again brings it back.
   **Facing** (Detect, or the character's forward axis in its own armature axes) and **Hips scale**
   (0 = from the pelvis heights) are saved with the map.
4. Choose fighter **A** or **B** (the take of the selected object, else the last imported one),
   **Lock feet** (on: planted feet stay in place, with foot IK on the character's legs; the hips come
   down a little where a leg is too short), press **Retarget** (in Object or Pose Mode: it refuses in
   Edit Mode, since it reads the armature's saved bones). It is one undo step: a new action
   `<shot>_<fighter>_<armature>` on the character, shown at once. The character's other actions are
   never deleted, with one exception so that retargeting again does not pile up actions: the
   previous action is replaced when all of these hold: it is still the character's active action;
   this panel made it for the same character object, take copy and fighter in this Blender session
   (a copy, another character with the same name, or a reopened file does not count); it still has
   its name; and nothing about it changed (keys, handles, interpolation, easing, F-curve modifiers,
   the F-curves, its slots) and nothing else uses it. Rename it, push it to an NLA strip, assign
   another action or edit it, and the next Retarget makes a new action instead. The report lists:
   frames, bones mapped and unmapped, facing, hips scale, foot slide on planted frames, floor
   penetration (cm at the fighter's size), the worst direction / twist error, notes (e.g. the hips
   lowered, constraints on the character) and the time (measured for 302 frames, lock feet on: 0.4 s
   per fighter for the Mixamo mech, 0.8 s for the Unity mech with 124 mesh parts, 0.7-0.9 s for
   Girush). What it
   refuses (non-uniform scale, a character lying down, a map against the hierarchy, a generated
   Rigify rig) is shown as a message saying what to fix.
5. **Export character**: pick a **Preset** and a file (default
   `shots/<id>/exports/retarget/<shot>_<fighter>_<armature>.fbx`). It writes the armature, its
   meshes and its action as FBX with that engine preset's **FBX exporter settings** (the same as
   afv's exports: no leaf bones, every frame keyed, deform bones only, ...), taken from afv once per
   Blender session through the worker (`presets` request; the first export waits for it, and any
   problem shows in the panel). Only the exporter settings: unlike the take's Export, no root bone is
   added, no root-motion mode applies, the armature keeps its name and the scene is not rescaled for
   Unreal (the file declares metres; Unreal converts on import). The clip in the file is named after
   the action (`001_A_<armature>`), so engines show that name instead of "Scene". Nothing else in the
   scene is written; selection, visibility and the frame range stay. Leave Edit Mode first.
   - The FBX keeps the character's own size and units: Girush comes out 175 m tall in the FBX
     because its file has no unit scale (its 175 units are meant as cm); scale it in your engine or
     apply a scale in Blender first.
   - Measured, the re-imported FBX against the character in Blender, every frame, at the fighter's
     scale: 0.007-0.013 mm for the two mechs, 0.013-0.020 mm for Girush (1.4-2.4 mm in Girush's own
     oversized units). A character whose stored rest matrices carry a scale and that was not brought
     in with Import character re-imports up to that fraction off (Girush taken as is: 1.3 mm at
     fighter scale): Retarget and Export character warn about it, and **Fix rest matrices** (shown
     then, one undo step) re-derives them the same way Import character does. The armature stays in
     its mode. Its existing poses and actions then play slightly differently (Girush: up to about
     2 mm at fighter scale), so retarget again after pressing it.
   - Retarget again after changing tweak keys; Export character exports what the character shows.

The same from WSL, without Blender's UI: `afv retarget` (see `WORKFLOW.md`, step 11).

## Files and coordinates

`afv take <shot>` writes `shots/<id>/take/`: per fighter `<shot>_<role>.<token>.npz`, and
`camera.<token>.npz`, `timeline.<token>.npz`, `qc.<token>.npz`, then `take.json` (the commit, naming
those files), then deletes the files the previous `take.json` named. Nothing else in the folder is
touched (`take.blend` lives there too), and a non-empty folder without a take is refused. It holds
the shot lock while reading the results, and prints a warning for results `afv run` would redo. See
`src/afv/take.py` for the exact format.

Moving things: the shot and project folders are found from where the take is (`take.json` stores
them relative to itself), and a .blend stores the take's path both absolute and relative to the
.blend (updated on every save), so moving the project folder, with or without the .blend, keeps
Re-import and Export working. The
rig arrays are the ones the FBX export uses, and the armature objects carry the same +90 degrees
about X, so the take matches the exported FBX (measured: <= 0.003 mm on the five sample shots). The
camera is in the bodies' world (the pair/contacts `world_shift` applied); joints through the Blender
camera match the pipeline's own projection to <= 0.003 px.

The take also holds detect_track's boxes (`tracks.<token>.npz`, in camera image pixels) for the box
overlay; takes written by an older afv have none (refresh the take).

The shot browser's files: `browser.py` (the scene, the list's state, the edit operators, undo),
`browser_ui.py` (panels), `browser_pick.py` (Pick fighters), `browser_run.py` (Run queue, Import take),
`browser_keys.py` (the keys),
`browser_jobs.py` (background requests, several at a time, with the one-shot fallback),
`browser_draw.py` (the pick boxes and the timeline strip), `browser_core.py` (frames and times, the
list's order, the fallback's command lines; no bpy).

The add-on's files: `builder.py` (scene building; shared with `headless.py`, which `afv blender`
runs), `takeio.py` (package reader and the overlay geometry, numpy only), `edits.py` (edit files,
numpy only), `serve.py` (the `afv serve` client and the one-shot command lines, no bpy), `wsl.py`
(one-shot `afv` runs), `ops.py` (jobs, Import / Re-import / Export), `cleanup.py` (the cleanup
tools), `overlay.py` (gpu drawing: strip and boxes), `ui.py`; the retarget: `retarget.py` (the
engine), `retarget_io.py` (character import, FBX export and the round-trip check), `retarget_ui.py`
(the Retarget sub-panel), `retarget_headless.py` (what `afv retarget` runs in Blender).
