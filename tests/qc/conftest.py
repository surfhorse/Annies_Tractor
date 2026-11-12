"""QC tests reuse the module E synthetic motion (``tests/motion/motion_synth.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

_MOTION = str(Path(__file__).parents[1] / "motion")
if _MOTION not in sys.path:
    sys.path.insert(0, _MOTION)
