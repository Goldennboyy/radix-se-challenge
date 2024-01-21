"""Constants."""

from __future__ import annotations

from pathlib import Path

# Data binaries folder
DATA_F = Path(__file__).parent / "data/binaries"
DATA_F.mkdir(exist_ok=True, parents=True)

# Model binaries folder
MDL_F = Path(__file__).parent / "model/binaries"
MDL_F.mkdir(exist_ok=True, parents=True)
