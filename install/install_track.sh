#!/usr/bin/env bash
# Step 3 of 4: tracking environment (RF-DETR person detection + SAM 2.1 video masklets) in WSL, no sudo.
# Safe to re-run. Run install_vipe.sh first: its conda env builds SAM 2's small CUDA extension.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ENV="$HOME/afv/track"; mkdir -p "$ENV/weights"; cd "$ENV"
log() { echo "[$(date +%H:%M:%S)] $*"; }
ARCH="${AFV_CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')}"  # e.g. 12.0 (RTX 50), 8.9 (RTX 40), 8.6 (RTX 30)
[ -d .venv ] || uv venv .venv --python 3.12
source .venv/bin/activate
log "torch cu128"
uv pip install "torch>=2.8" torchvision --index-url https://download.pytorch.org/whl/cu128
log "rfdetr + sam2 (no custom CUDA ext; SAM2 works without it)"
[ -d sam2 ] || git clone --depth 1 https://github.com/facebookresearch/sam2.git sam2
SAM2_BUILD_CUDA=0 uv pip install -e ./sam2
uv pip install rfdetr opencv-python-headless
log "SAM 2 _C extension (mask hole filling), built in place with the ViPE conda env's CUDA 12.8"
# torch is cu128 and the system gcc 15 is too new for nvcc 12.8, so this uses the conda env's nvcc and
# gcc 13. Only sam2/sam2/_C.so is written (no package changes); at run time torch's own
# libcudart.so.12 satisfies it. Without the conda env SAM 2 just warns and skips hole filling.
CU="$HOME/afv/miniforge/envs/cu128"
if [ -x "$CU/bin/nvcc" ] && [ ! -f sam2/sam2/_C.so ]; then
  ( cd sam2 && CUDA_HOME="$CU" PATH="$CU/bin:$PATH" CC="$CU/bin/x86_64-conda-linux-gnu-gcc" \
      CXX="$CU/bin/x86_64-conda-linux-gnu-g++" TORCH_CUDA_ARCH_LIST="$ARCH" SAM2_BUILD_CUDA=1 \
      SAM2_BUILD_ALLOW_ERRORS=0 python setup.py build_ext --inplace )
fi
log "SAM 2.1 Hiera-L checkpoint"
[ -f weights/sam2.1_hiera_large.pt ] || curl -fL https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -o weights/sam2.1_hiera_large.pt
log "RF-DETR Medium weights (first load downloads them)"
python - <<'PY'
import torch, rfdetr, sam2
from rfdetr import RFDETRMedium
m = RFDETRMedium()
print("rfdetr", rfdetr.__version__ if hasattr(rfdetr, "__version__") else "?", "torch", torch.__version__, torch.cuda.is_available())
PY
OUT="$HOME/afv/provenance"; LIC="$OUT/licences/$(date +%F)"; mkdir -p "$LIC"
cp sam2/LICENSE "$LIC/SAM2_LICENSE.txt"
python -c "import rfdetr,os;p=os.path.dirname(rfdetr.__file__);print(p)" > /tmp/rfp
find "$(cat /tmp/rfp)/.." -maxdepth 2 -iname "LICENSE*" -path "*rfdetr*" -exec cp {} "$LIC/RF-DETR_LICENSE.txt" \; 2>/dev/null || true
{
  echo "# generated $(date +%F) by install/install_track.sh"
  echo "sam2 commit: $(git -C sam2 rev-parse HEAD)"
  echo "rfdetr: $(python -c 'import importlib.metadata as m;print(m.version("rfdetr"))')"
  echo "torch: $(python -c 'import torch;print(torch.__version__)')"
  echo "sha256:"
  sha256sum weights/*.pt
  find "$HOME" -maxdepth 4 -name "rf-detr-medium*.pth" -exec sha256sum {} \; 2>/dev/null
  find . -maxdepth 1 -name "*.pth" -exec sha256sum {} \;
} > "$OUT/track_lock.txt"
cat "$OUT/track_lock.txt"
log DONE
