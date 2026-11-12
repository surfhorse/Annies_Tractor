#!/usr/bin/env bash
# Step 1 of 4: uv, a static ffmpeg, and GEM-X Body Motion (3D pose) with its weights, in WSL2 Ubuntu,
# without sudo. Safe to re-run. Usage (in WSL): bash <repo>/install/install_gemx.sh
set -euo pipefail
ROOT="$HOME/afv"
GEMX="$ROOT/gem-x"
BIN="$HOME/.local/bin"
mkdir -p "$ROOT" "$BIN"
export PATH="$BIN:$PATH"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v uv

log "ffmpeg (static build, user space)"
if ! command -v ffprobe >/dev/null; then
  tmp=$(mktemp -d)
  curl -fsSL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o "$tmp/ff.tar.xz"
  tar -xf "$tmp/ff.tar.xz" -C "$tmp"
  cp "$tmp"/ffmpeg-*-static/ffmpeg "$tmp"/ffmpeg-*-static/ffprobe "$BIN/"
  rm -rf "$tmp"
fi
ffmpeg -version | head -1

log "GEM-X checkout"
[ -d "$GEMX/.git" ] || git clone https://github.com/NVlabs/GEM-X.git "$GEMX"
cd "$GEMX"
git rev-parse HEAD

log "venv + torch (CUDA 12.8 wheels)"
[ -d .venv ] || uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install "torch>=2.10,<3" torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e '.[video]' "huggingface_hub[cli]"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "arch list", torch.cuda.get_arch_list())
assert torch.cuda.is_available(), "CUDA not available in torch"
x = torch.randn(1024, 1024, device="cuda"); print("matmul ok", float((x @ x).sum()) == float((x @ x).sum()))
PY

log "weights: GEM-X native77 video set"
mkdir -p weights assets/soma
hf download nvidia/GEM-X body_s240000.ckpt vitpose.pth motion_native77_e774_s292175.ckpt \
  motion_stats_raw.npz body_video_native77.yaml motion_native77.yaml body_config.json README.md --local-dir weights
log "weights: SOMA-X v0.3.3 assets"
hf download nvidia/SOMA-X SOMA_neutral.npz SOMA_template_rig.usda SOMA_procedural_transforms.json \
  correctives_model.pt --revision v0.3.3 --local-dir assets/soma
log "weights: YOLOX HumanArt person detector"
if [ ! -f weights/yolox_humanart.onnx ]; then
  curl --fail --location https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip \
    --output weights/yolox_humanart.zip
  python - <<'PY'
from pathlib import Path
from shutil import copyfileobj
from zipfile import ZipFile
with ZipFile("weights/yolox_humanart.zip") as a:
    m = next(n for n in a.namelist() if n.endswith("/end2end.onnx"))
    with a.open(m) as s, Path("weights/yolox_humanart.onnx").open("wb") as t:
        copyfileobj(s, t)
PY
fi

log "record: versions, weight hashes and dated licence copies -> ~/afv/provenance/"
OUT="$ROOT/provenance"; LIC="$OUT/licences/$(date +%F)"; mkdir -p "$LIC"
cp LICENSE "$LIC/GEM-X_code_LICENSE.txt"
cp weights/README.md "$LIC/GEM-X_hf_model_card.md"
[ -d LICENSES ] && cp -r LICENSES "$LIC/GEM-X_LICENSES" || true
cp ATTRIBUTIONS.md "$LIC/GEM-X_ATTRIBUTIONS.md" 2>/dev/null || true
curl -fsSL https://huggingface.co/nvidia/SOMA-X/raw/v0.3.3/README.md -o "$LIC/SOMA-X_hf_model_card.md" || true
{
  echo "# generated $(date +%F) by install/install_gemx.sh"
  echo "gem-x commit: $(git rev-parse HEAD)"
  echo "torch: $(python -c 'import torch;print(torch.__version__)')"
  echo "py-soma-x: $(python -c 'import importlib.metadata as m;print(m.version("py-soma-x"))')"
  echo "sha256:"
  (cd weights && sha256sum *.ckpt *.pth *.npz *.onnx)
  (cd assets/soma && sha256sum *)
} > "$OUT/gemx_lock.txt"
cat "$OUT/gemx_lock.txt"

log "smoke: CLI help"
gem-infer motion-video --help > "$OUT/gem-infer_motion-video_help.txt"
head -60 "$OUT/gem-infer_motion-video_help.txt"
log "DONE"
