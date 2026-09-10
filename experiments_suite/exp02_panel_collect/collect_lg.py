#!/usr/bin/env python3
"""Panel collector -- Lingua-SafetyBench (10 langs, resource order; LG order/benign/harm auto)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common
ROOT = str(Path(__file__).resolve().parent / "results")
if __name__ == "__main__":
    raise SystemExit(_common.run("Lingua-SafetyBench", ROOT, "_lg"))
