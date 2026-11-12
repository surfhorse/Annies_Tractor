Just setting things up. Feel free to support this landlocked quadruped. 
<a href="https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=9F7FSSBA59T8S">
  <img src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif" alt="Donate with PayPal" />
</a>

# afv: Animation From Video

afv turns fight footage (boxing, MMA, kung fu films) into 3D animation of the two fighters, ready to
clean up in Blender and export to a game engine. You give it a video; it finds the shots, tracks the
fighters you pick, solves the camera and both bodies in one 3D world, fixes foot sliding, and hands you
two animated skeletons in Blender with the video behind them. From there you can polish the motion,
export FBX / GLB / BVH, or put the motion onto your own character.

It is free and open source. It was built by one person for their own games and is shared as it is:
it works on the author's computer, it has rough edges, and nobody is on call for support.

## What you need

| | |
|---|---|
| Computer | Windows 11 (or 10 21H2+) with WSL2 and Ubuntu |
| Graphics card | **NVIDIA with 16 GB of video memory**, and a current driver (CUDA 12.8). Developed and tested on an RTX 5080 only. AMD and Intel cards are not supported |
| Memory and disk | About 16 GB of RAM for WSL, and about **60 GB free** (the AI models take about 34 GB) |
| Blender | 5.2, on Windows |
| Time | 1-2 hours for the first install, mostly downloads |

There is no Mac or Linux-desktop version: the pipeline runs in WSL and talks to Blender on Windows.

## Install and first project

**The easy way: let an AI coding agent do it with you.** Clone this repository, open it with an agent
such as Claude Code, and say: *"Read HOLD_MY_HAND.md and take me through it."* It checks your computer,
installs what is missing (asking first), sets up the Blender add-on and walks you through a first
project from one of your videos.

**By hand**, the same steps are in [HOLD_MY_HAND.md](HOLD_MY_HAND.md). In short:

1. Clone this repository onto a Windows drive, e.g. `D:\afv`.
2. In WSL (Ubuntu), run the four installers in this order. They put everything under `~/afv`, need no
   `sudo`, and can be re-run:

   ```bash
   bash /mnt/d/afv/install/install_gemx.sh     # uv, ffmpeg, GEM-X (3D pose) and its weights
   bash /mnt/d/afv/install/install_vipe.sh     # ViPE (camera solve)
   bash /mnt/d/afv/install/install_track.sh    # RF-DETR + SAM 2.1 (finding and tracking people)
   bash /mnt/d/afv/install/install_afv_env.sh  # afv itself
   ~/afv/afv-env/bin/afv models --check        # lists every model and its licence
   ```

3. Build and install the Blender add-on (PowerShell, Blender closed):

   ```powershell
   & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --command extension build --source-dir D:\afv\blender\afv_addon --output-dir D:\afv\dist
   & "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --command extension install-file -r user_default -e D:\afv\dist\afv-0.2.0.zip
   ```

4. In Blender: Edit > Preferences > Add-ons > AFV, set **afv command (in WSL)** to
   `~/afv/afv-env/bin/afv` and a **Projects folder**.

## Using it

Everything starts in Blender's 3D viewport sidebar (**N**), tab **AFV**:

1. **Shots > New project from video.** afv finds the camera cuts and lists every shot with a thumbnail.
2. **Choose shots**: queue the ones you want (**Q**), skip the rest (**S**), trim with **I** / **O**.
3. **Pick the fighters**: click fighter A, then fighter B, on each queued shot.
4. **Run queue.** It runs in the background (a few minutes per shot) and keeps going if you close
   Blender, so a long list can run overnight.
5. **Review and clean up** each result: paint foot contacts, pin poses, adjust smoothing.
6. **Export** for Unity, Unreal, Godot or Blender, or **Retarget** the motion onto your own character.

The full walkthrough, with the command-line equivalent of every step, is [WORKFLOW.md](WORKFLOW.md).
The add-on's reference is [blender/afv_addon/README.md](blender/afv_addon/README.md).

## Known limits

- **A take imported inside the shot browser's own .blend file doesn't play.** Open results in a new
  file instead (File > New, then AFV > Import take).
- Tested on one computer. Other NVIDIA cards with 16 GB should work, but the install's CUDA build steps
  have not been tried on them.
- The installers fetch the newest version of each model's code, which can change. The versions afv was
  tested with are recorded in `install/tested_versions/`.
- It expects two fighters who are mostly visible. Clinches, heavy occlusion, zooming cameras and very
  fast camera moves give worse results and need more cleanup.
- Exporting onto your own character writes the motion only: no root-motion bone or engine-specific
  scale is added yet, and it has not been test-imported into a game engine.
- This release contains the tool only. The author's tests and development notes are not included.
- User may need to start new project in Blender after picking fighters and exporting the take.
- When importing models onto animations, the user may need "retarget" prior to working. 

## Licences

**afv's own code** is copyright 2026 Surfhorse Productions.

- Everything except `blender/` is under the **Apache License 2.0**: see [LICENSE](LICENSE) and
  [NOTICE](NOTICE). You may use, change and redistribute it, including commercially, if you keep the
  notices.
- Everything under `blender/` (the Blender add-on and the scripts afv runs inside Blender) is under the
  **GNU GPL, version 3 or later**: see [blender/afv_addon/LICENSE](blender/afv_addon/LICENSE). Code that
  runs inside Blender has to be GPL-compatible.

It comes with **no warranty**, as both licences say.

**The AI models are not part of afv and are not covered by its licence.** The installers download them
from their publishers, and each has its own terms. All of them allow commercial use of what you make,
but two have conditions you should know before you publish anything:

- **GEM-X** (the 3D pose model, by NVIDIA) is under the **NVIDIA Open Model License**. Read it on the
  [model card](https://huggingface.co/nvidia/GEM-X).
- **GeoCalib** (used by the camera solve) is **CC BY 4.0: you must credit it** in what you publish.

The full list, with every model's licence and what was checked, is in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). It is a record, not legal advice.

**Your footage is your responsibility.** Fight broadcasts and films are usually copyrighted, and the
people in them have rights too. afv gives you no rights to any video, and motion taken from someone
else's footage may be restricted. Use footage you own or are allowed to use. This repository contains no
footage.
