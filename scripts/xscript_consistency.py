#!/usr/bin/env python3
"""Cross-script consistency audit (reviewer W6): several quantities are computed by more than one
generator, and must agree to the precision the paper prints. This asserts the known identities and
also flags any \\pj macro defined in two numbers files with different values. Exit non-zero on failure.
Run after regenerating tables. No GPU, no data access -- reads paper/*.tex only."""
from __future__ import annotations
import re, sys
from pathlib import Path

PAPER = Path("paper")


def macros(*files):
    m = {}
    for f in files:
        p = PAPER / f
        if not p.exists():
            continue
        for name, val in re.findall(r"\\newcommand\{\\(pj[A-Za-z]+)\}\{([^}]*)\}", p.read_text()):
            m.setdefault(name, {})[f] = val
    return m


def num(s):
    mt = re.search(r"-?\d+\.\d+", s or "")
    return float(mt.group()) if mt else None


def main():
    fails = []
    M = macros("selector_numbers.tex", "query_numbers.tex", "heldout_numbers.tex",
               "heldout_ablation_numbers.tex", "robust_numbers.tex", "persona_numbers.tex",
               "domain_numbers.tex", "ppl_numbers.tex", "factorization_numbers.tex")

    # (1) any macro defined in >1 file with a different value
    for name, byfile in M.items():
        vals = set(byfile.values())
        if len(vals) > 1:
            fails.append(f"macro \\{name} defined with conflicting values: {byfile}")

    def val(name):
        d = M.get(name, {})
        return num(next(iter(d.values()))) if d else None

    # (2) known cross-generator identities (value, value, tolerance)
    checks = [
        ("held-out adapt LG: heldout_selector vs heldout_ablation",
         val("pjhoLGadapt"), val("pjablLGkept"), 0.0005),
        ("held-out adapt MJ: heldout_selector vs heldout_ablation",
         val("pjhoMJadapt"), val("pjablMJkept"), 0.0005),
    ]
    for label, a, b, tol in checks:
        if a is None or b is None:
            fails.append(f"{label}: missing macro ({a}, {b})")
        elif abs(a - b) > tol:
            fails.append(f"{label}: {a} != {b} (tol {tol})")

    # (3) the best-fixed point estimate in the merged main table must equal \pjfixedMJ / \pjfixedLG
    main = (PAPER / "tab_sel_main.tex").read_text()
    row = re.search(r"best fixed configuration.*?&\s*([0-9.]+).*?&\s*([0-9.]+)", main)
    if row:
        for got, mac in ((float(row.group(1)), "pjfixedMJ"), (float(row.group(2)), "pjfixedLG")):
            exp = val(mac)
            if exp is None or abs(got - exp) > 0.0005:
                fails.append(f"tab_sel_main best-fixed {got} != \\{mac} {exp}")
    else:
        fails.append("could not parse best-fixed row in tab_sel_main.tex")

    # (4) the tau=0.5,cap8 grid cell in tab_heldout_ablation must equal the 'with indicators' default
    ab = (PAPER / "tab_heldout_ablation.tex").read_text()
    star = re.search(r"cap \$8\$\$\^\\star\$ & [0-9.]+ \([0-9.]+\) & ([0-9.]+)", ab)
    dflt = re.search(r"with indicators \(default\) & [0-9.]+ & ([0-9.]+)", ab)
    if star and dflt and abs(float(star.group(1)) - float(dflt.group(1))) > 0.0005:
        fails.append(f"tab_heldout_ablation LG: grid tau0.5/cap8 {star.group(1)} != default {dflt.group(1)}")

    if fails:
        print("CROSS-SCRIPT CONSISTENCY: FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("CROSS-SCRIPT CONSISTENCY: PASS (all shared quantities agree to printed precision)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
