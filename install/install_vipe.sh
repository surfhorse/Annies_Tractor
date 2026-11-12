#!/usr/bin/env bash
# Step 2 of 4: ViPE (camera solve) in its own environment, without sudo. Safe to re-run.
# Miniforge (user space) supplies nvcc 12.8 + native libs; uv builds .venv with cu128 torch.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ROOT="$HOME/afv"; VIPE="$ROOT/vipe"; CONDA="$ROOT/miniforge"
log() { echo "[$(date +%H:%M:%S)] $*"; }
ARCH="${AFV_CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')}"  # e.g. 12.0 (RTX 50), 8.9 (RTX 40), 8.6 (RTX 30)

if [ ! -x "$CONDA/bin/conda" ]; then
  log "miniforge"
  curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$CONDA"
fi
source "$CONDA/etc/profile.d/conda.sh"
[ -d "$VIPE/.git" ] || git clone https://github.com/nv-tlabs/vipe.git "$VIPE"
cd "$VIPE"; git rev-parse HEAD
if ! conda env list | grep -q "^cu128 "; then log "conda env cu128"; conda env create -f envs/cu128.yml; fi
set +u; conda activate cu128; set -u
export CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS=6
nvcc --version | tail -1
log "uv sync (builds CUDA extensions for compute capability $ARCH)"
uv sync
log "smoke"
.venv/bin/vipe --help | head -20
.venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
OUT="$ROOT/provenance"; LIC="$OUT/licences/$(date +%F)"; mkdir -p "$LIC"
echo "vipe commit: $(git rev-parse HEAD)" > "$OUT/vipe_lock.txt"
cp LICENSE "$LIC/ViPE_LICENSE.txt" 2>/dev/null || true
log "DONE"
