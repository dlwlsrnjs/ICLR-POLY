#!/usr/bin/env python3
"""Offline sanity check for the historical 23-arm archive."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_COMMIT = "18fe64aeb4f5bc2ef9ee3b134774de219a9a1565"

AMOUNT = [f"amt_n{n}" for n in range(2, 11)]
DISORDER = [f"dis_{d}" for d in (0.25, 0.5, 0.75, 1.0)]
COMPOSITION = ["combo_ours", "combo_ours_persona", "combo_ours_incept", "combo_incept_only"]
ROLE = ["tri_hi_en", "tri_triple_en"]
SINGLE = ["m_aim", "m_deepinception", "m_pap"]
TRANSLATION = ["m_translated"]
ARMS = AMOUNT + DISORDER + COMPOSITION + ROLE + SINGLE + TRANSLATION

REQUIRED = [
    "scripts/amount_disorder_2phase.py",
    "scripts/mj_2phase_generic.py",
    "scripts/triple_combo_eval.py",
    "scripts/triple_hien_eval.py",
    "scripts/extra_arms.py",
    "scripts/benign_arms_probe.py",
    "scripts/benign_signals_probe.py",
    "scripts/benign_prior_selection.py",
    "scripts/arm_scoring.py",
    "scripts/gp_bai.py",
    "scripts/mj_bandit_full.py",
    "scripts/lingua_bandit_full.py",
    "scripts/heldout_selector.py",
    "scripts/permodel_technique_tables.py",
    "paper/tab_heldin.tex",
    "paper/tab_heldout.tex",
    "docs/audit_20260910_before/polyjigsaw_iclr2026.tex",
]


def main() -> int:
    assert len(ARMS) == 23, f"arm count drifted: {len(ARMS)}"
    assert len(set(ARMS)) == 23, "duplicate arm name"
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("missing required files:\n  " + "\n  ".join(missing))

    bad = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            bad.append(f"{path.relative_to(ROOT)}: {exc}")
    if bad:
        raise SystemExit("syntax failures:\n  " + "\n  ".join(bad))

    print(f"source commit: {SOURCE_COMMIT}")
    print(f"arm count: {len(ARMS)}")
    print(f"python files compiled: {len(list((ROOT / 'scripts').glob('*.py')))}")
    print("23-arm archive check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
