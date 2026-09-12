#!/usr/bin/env python3
"""Panel collector -- MultiJail (10 langs, low-resource-first order; MJ order/benign/harm auto)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common
ROOT = str(Path(__file__).resolve().parent / "results")
if __name__ == "__main__":
    raise SystemExit(_common.run("MultiJail", ROOT, "_mj", 64))
