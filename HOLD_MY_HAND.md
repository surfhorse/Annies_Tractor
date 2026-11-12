# HOLD_MY_HAND: set up afv and make a first project, with an agent's help

**To the user:** point your coding agent (for example Claude Code) at this file: "Read HOLD_MY_HAND.md
and take me through it." It checks your computer, installs what's missing with your OK, and walks you
through your first project, from a fight video to animated fighters in Blender.

**To the agent:** this file is your script. Follow the phases in order. Rules:

- **Check before you act, and ask before anything that installs, downloads gigabytes, needs admin
  rights, reboots, or changes the user's Blender preferences.** Say what it does, how big and how long.
- Explain each step in plain words; the user may never have used WSL, a terminal or Blender add-ons.
- Run WSL commands from a script file, not inline: on Windows write the commands to a `.sh` file, then
  `wsl -d Ubuntu -- bash /mnt/c/.../file.sh` (from Git Bash prefix `MSYS_NO_PATHCONV=1`). Never put
  `$VARIABLES` in an inline `wsl ... bash -c '...'` (wsl.exe expands them first).
- The install scripts are safe to re-run (they skip what is done). If one fails, read its last lines,
  fix the cause, run it again.
- Report results as a short table: what you checked, what you found, pass or not.
- Never commit or upload the user's videos, frames or renders.
- The full user workflow is `WORKFLOW.md`; the Blender add-on's reference is
  `blender/afv_addon/README.md`; licences are in `README.md` and `THIRD_PARTY_LICENSES.md`. Use them when
  something here is not enough.
- Tell the user once, before installing, that the models have their own licences (GEM-X: NVIDIA Open
  Model License; GeoCalib: credit required) and that they need the right to use their footage.

What afv is: a tool that turns fight footage (boxing, UFC, kung fu films) into 3D animation of the two
fighters. Heavy models run on an NVIDIA GPU inside WSL2 (Linux on Windows); review and cleanup happen in
Blender 5.2 on Windows through the AFV add-on.

---

## Phase 1: check the computer

Run these checks and show the user the table. Hard requirements must pass before Phase 2.

| Check | How (in WSL unless noted) | Needed | Notes |
|---|---|---|---|
| Windows | PowerShell: `[System.Environment]::OSVersion.Version`, `winver` | Windows 11 (or 10 21H2+) | GPU in WSL needs this |
| WSL2 + Ubuntu | PowerShell: `wsl -l -v` | a distro named `Ubuntu`, VERSION 2 | Another name works: set it in the add-on preferences later |
| NVIDIA GPU | `nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv` | NVIDIA, **16 GB VRAM** | The camera stage peaks at ~14.8 GB. AMD / Intel GPUs are not supported |
| Driver supports CUDA 12.8 | `nvidia-smi` header: "CUDA Version" | 12.8 or higher | Update the **Windows** NVIDIA driver; never install a driver inside WSL |
| GPU compute capability | the `compute_cap` column above (e.g. `12.0` RTX 50, `8.9` RTX 40, `8.6` RTX 30) | note it | The installers read it themselves; tested on `12.0` only |
| WSL memory | `free -g` | ~14 GB or more total | Raise it in `%UserProfile%\.wslconfig` (`[wsl2]` `memory=16GB`), then `wsl --shutdown` |
| Disk in WSL | `df -h ~` | **60 GB free** | Model environments take ~34 GB; work files grow per shot |
| Disk for footage | Windows drive with the repo and videos | a few GB per long video | Projects are folders next to the repo |
| Tools | `git --version`, `curl --version`, `python3 --version` | present | Ubuntu has them; else `sudo apt install git curl` (asks the user's Linux password) |
| Blender 5.2 | PowerShell: `Test-Path "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"` | installed | Elsewhere is fine: note the path for `AFV_BLENDER` |
| Internet | `curl -sI https://huggingface.co` | reachable | ~30 GB of downloads in Phase 2 |

If a hard requirement fails, stop and tell the user plainly what to change (for example "this GPU has
8 GB; afv needs 16 GB") and what the options are. Do not install anything to work around a GPU that is
too small.

Already installed? If `~/afv/afv-env/bin/afv` exists, run `~/afv/afv-env/bin/afv models --check` (from
the repo folder in WSL): if it lists every model with no error, skip to Phase 3.

## Phase 2: install (about 1-2 hours, mostly downloads and CUDA builds)

Ask the user before starting; tell them it downloads ~30 GB and uses ~34 GB in WSL. Do the steps in
this order; each depends on the one before.

### 2.1 Things only the user can do (with your guidance)

- **WSL2 + Ubuntu** if missing: PowerShell **as administrator**: `wsl --install -d Ubuntu`, reboot,
  open Ubuntu once and create the Linux user and password. Admin rights and a reboot: ask first.
- **NVIDIA driver** if CUDA is below 12.8: the user installs the latest Game Ready or Studio driver
  from nvidia.com on Windows, then `wsl --shutdown` and check `nvidia-smi` in WSL again.
- **Blender 5.2** if missing: from blender.org, the default install location.

### 2.2 Get the repo

If the user doesn't have it yet, clone this repository onto a Windows drive so Blender can reach it,
e.g. `D:\afv` (ask the user for the repository's address if you don't have it):

```powershell
git clone <repository address> D:\afv
```

Its WSL path is the Windows path with `D:\` -> `/mnt/d/` and `\` -> `/`
(`/mnt/d/afv`). Call it `REPO` below. The examples in the docs use `D:\afv`; use the real path.

### 2.3 What the installers do

Everything installs under `~/afv` in the user's WSL home, without sudo. The scripts in `REPO/install/`
find the repo from their own location and read the GPU's compute capability from `nvidia-smi` (set
`AFV_CUDA_ARCH`, e.g. `AFV_CUDA_ARCH=8.9`, to override). They were written and tested on one computer
(RTX 5080, Ubuntu in WSL2): on other GPUs a CUDA build step may need attention, so read the output.
They install the latest upstream versions; the versions afv was tested with are in
`REPO/install/tested_versions/` (check out those commits in `~/afv/gem-x`, `~/afv/vipe` and
`~/afv/track/sam2` if a newer upstream version breaks something).

### 2.4 Run the installers, one at a time, in this order

| # | Script | What it installs | Size, time (approx.) |
|---|---|---|---|
| 1 | `install_gemx.sh` | `uv` and a static ffmpeg in `~/.local/bin`; GEM-X (3D pose) with CUDA 12.8 PyTorch; its weights from Hugging Face (`nvidia/GEM-X`, `nvidia/SOMA-X`, no login needed) | ~17 GB; the longest download |
| 2 | `install_vipe.sh` | Miniforge in `~/afv/miniforge`, conda env `cu128` (nvcc 12.8), ViPE (camera solve) with its CUDA extensions built for this GPU | ~13 GB; the build takes a while |
| 3 | `install_track.sh` | RF-DETR (people) + SAM 2.1 (masks) with their weights; builds SAM 2's small CUDA extension with the `cu128` env from step 2 | ~1.5 GB |
| 4 | `install_afv_env.sh` | afv itself (`~/afv/afv-env`, an editable install of the repo) | ~100 MB, a minute |

Run each with `bash $REPO/install/<script>.sh` in WSL (through a script file, see the rules), and keep the user
informed: these print progress and end with `DONE` (or `afv ok` for the last). Each smoke-tests itself
(e.g. "torch ... available True"). On a failure: CUDA not available -> driver (2.1); out of disk ->
Phase 1; a CUDA build error -> check the compute capability (2.3).

The scripts also record the installed versions, weight hashes and dated licence copies in
`~/afv/provenance/`.

### 2.5 Verify and make `afv` easy to call

```bash
cd $REPO && ~/afv/afv-env/bin/afv models --check
```

Every model should be listed with its licence (all allow commercial use of the output; GeoCalib needs a
credit, see `THIRD_PARTY_LICENSES.md`). A warning "no weight sha256 recorded" is fine. Then add
`alias afv=~/afv/afv-env/bin/afv` to `~/.bashrc` so `afv --help` works in new terminals.

If Blender is not at `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`, add
`export AFV_BLENDER="/mnt/c/<path>/blender.exe"` to `~/.bashrc`. If the model environments live
elsewhere, see `~/.config/afv/envs.toml` in `src/afv/models/envs.py`'s docstring.

## Phase 3: the Blender add-on

1. Build the extension zip (PowerShell; Blender must not need to be open):

   ```powershell
   & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --command extension build --source-dir D:\afv\blender\afv_addon --output-dir D:\afv\dist
   ```

2. Install it, with Blender **closed** (ask first: it changes the user's Blender setup):

   ```powershell
   & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --command extension install-file -r user_default -e D:\afv\dist\afv-0.2.0.zip
   ```

   (Or in Blender: Edit > Preferences > Get Extensions > the menu at the top right > Install from Disk.)
3. Guide the user to **Edit > Preferences > Add-ons > AFV** and set:
   - **WSL distribution**: `Ubuntu` (or the name from Phase 1)
   - **afv command (in WSL)**: `~/afv/afv-env/bin/afv`
   - **Projects folder**: e.g. `D:\afv\projects` (new projects go here)
   - leave **Use the afv serve worker** on.
4. In the 3D viewport press **N** (sidebar), tab **AFV**: the panels **Shots** (the shot browser) and
   **AFV** (takes and cleanup) appear.

Repeat steps 1-2 whenever the add-on is updated (the zip name follows its version).

## Phase 4: the first project, from a video

Coach the user through this in Blender; you watch for errors and explain. Use a real fight video the
user has the right to use (5-20 minutes is ideal for a first try). Times are measured on an RTX 5080
with a 16-minute 720p fight.

1. **New project from video** (panel Shots). Pick the video. In the dialog, **Project folder** can be
   just a new name (e.g. `my_first_fight`): it's made in the Projects folder. Leave **Scan the shots
   after** on. Wait: ~1 min to find the cuts, ~1-1.5 min to scan the shots (thumbnails fill in).
   Result: a scene **AFV Shots** with the video, cut markers, and ~one candidate shot per camera cut.
2. **Choose shots.** Set the order to **Fighters likely first** (a hint, not a filter: it misses some
   clinches and marks some walkouts). Click a row to jump there, **Space** plays. With the mouse over
   the video or the timeline: **Q** queue, **S** skip (both move on), **I / O** set the shot's in and out
   at the playhead, **Page Down / Page Up** next / previous shot, **L** next likely shot. For a first run,
   queue **2-3 short shots where both fighters are clearly visible, whole body, no clinch**.
3. **Pick the fighters.** On a queued shot press **Enter** (or Pick fighters): numbered boxes appear.
   Click fighter A, then fighter B (then any referee you want tracked), **Enter** to save; **Then next**
   moves to the next queued shot. Trimmed shots are rescanned first in one short job.
4. **Run the queue.** In Run queue, for the first try set **Run until** to `export` and run the 2-3
   shots (a few minutes per shot; the GPU is busy meanwhile). It runs in WSL and keeps going if Blender
   is closed; reopening the project shows its progress or last results. A shot may stop at "needs a
   tracking review" (the tracker is unsure): tell the user what it means (`WORKFLOW.md`, Tracking QC
   hints) and either fix the pick or confirm.
5. **Look at the result.** **Known issue: a take imported inside the shot browser's own
   .blend shows the skeletons but doesn't play.** Until it's fixed, open the take in a fresh file:
   File > New > General, then panel **AFV** > **Import take** and pick the shot's folder
   (`<project>\shots\<id>`). Or in WSL: `afv blender <id> -p <project>`, then open
   `<project>\shots\<id>\take\take.blend`. The two fighters play over the video, seen through the solved
   camera.
6. **What next** (point the user to `WORKFLOW.md`): cleanup tools (contact painting, pins, smoothing,
   hands), export for a game engine (AFV > Export or `afv export`), and putting the motion on their own
   character (AFV > Retarget, or `afv retarget`). For a full night: queue ~20 shots, pick them, Run queue,
   close Blender.

When the user has seen their fighters move, ask what was confusing and note it for the developers.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| "cannot map ... to a WSL path" | An old add-on build: rebuild and reinstall it (Phase 3); the Project folder field takes a plain name now |
| The panel says the worker can't start / afv not found | Preferences: WSL distribution name and `~/afv/afv-env/bin/afv` (Phase 3) |
| Waiting "for the GPU" | Another afv job holds the GPU (one at a time, by design); it continues when that ends |
| CUDA out of memory in the camera stage | Close other GPU programs (games, browsers with video); 16 GB is the floor |
| `torch ... available False` | Windows driver too old for CUDA 12.8, or WSL not restarted after the update (`wsl --shutdown`) |
| A run stops with exit 3 | A shot needs fighters picked (`needs_pick`) or a tracking review (`needs_review`): see `WORKFLOW.md` |
| "edits/... .json: ... (fix or delete that entry ...)" | A hand-edited edit file has a bad value; fix or delete that entry |
| WSL prints "nested virtualization" or "NAT network ... conflicts" warnings | Harmless here |
| Blender not found by `afv blender` / export | Set `AFV_BLENDER` (Phase 2.5) |
