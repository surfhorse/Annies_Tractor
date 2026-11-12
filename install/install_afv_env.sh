#!/usr/bin/env bash
# Step 4 of 4: afv's own environment inside WSL (CPU stages, CLI, orchestration): ~/afv/afv-env, an
# editable install of this repository, so `git pull` updates afv without reinstalling. Safe to re-run.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p "$HOME/afv"
cd "$HOME/afv"
[ -d afv-env ] || uv venv afv-env --python 3.12
source afv-env/bin/activate
uv pip install -e "$REPO"
python -c "import afv, numpy, scipy, av, typer, pydantic; from afv.core.stage import STAGE_ORDER; print('afv ok', STAGE_ORDER)"
