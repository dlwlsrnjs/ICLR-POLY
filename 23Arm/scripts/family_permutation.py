#!/usr/bin/env python3
"""Is the family-level pattern across targets real, or the same kind of noise we rejected for domains?

We retracted the harm-domain axis after a permutation test. The same standard has to be applied to
the claim we do keep: that which configuration \\emph{family} wins is tied to the model family, with
persona attacks winning on Gemma and multilingual role separation on the larger Qwen targets.

Null: the winning family is unrelated to the model family. We permute the model-family label across
the nine targets, keeping each target's own winning family fixed, and recount how concentrated the
pattern is. The statistic is the number of target cells whose winning family equals the modal winning
family of its model family, summed over the two collections; concentration by chance is what the
permutation distribution measures.

Writes results/family_permutation_20260908.json and macros in paper/robust_numbers.tex.
"""
from __future__ import annotations
import io, json, contextlib, re, sys
from pathlib import Path
from collections import Counter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import mj_bandit_full as MJ
    import lingua_bandit_full as LG

PERM = 20000
rng = np.random.default_rng(0)
MODEL_FAMILY = {"qwen25_3b": "Qwen", "qwen25_7b": "Qwen", "qwen25_14b": "Qwen", "qwen25_32b": "Qwen",
                "llama32_3b_it": "Llama", "llama31_8b_it": "Llama",
                "gemma2_2b_it": "Gemma", "gemma2_9b_it": "Gemma", "gemma2_27b": "Gemma"}


def arm_family(a):
    if a.startswith("amt_"):
        return "amount"
    if a.startswith("dis_"):
        return "disorder"
    if a.startswith("tri_"):
        return "role separation"
    if a.startswith("m_aim") or a == "combo_ours_persona":
        return "persona"
    if a.startswith("m_deepinception") or a in ("combo_ours_incept", "combo_incept_only"):
        return "fiction"
    if a.startswith("m_"):
        return "other single-vector"
    return "composition"


def concentration(labels, winners):
    """How many cells match the modal winning family of their model family."""
    by = {}
    for lab, w in zip(labels, winners):
        by.setdefault(lab, []).append(w)
    return sum(Counter(v).most_common(1)[0][1] for v in by.values())


def main():
    targets, winners, labels = [], [], []
    for name, mod in (("MultiJail", MJ), ("Lingua-SafetyBench", LG)):
        models = mod.STRONG if hasattr(mod, "STRONG") else mod.MODELS
        for t in models:
            w = mod.names[int(np.argmax(mod.G[t]))]
            targets.append((name, t))
            winners.append(arm_family(w))
            labels.append(MODEL_FAMILY[t])
    labels = np.array(labels); winners = np.array(winners)

    obs = concentration(labels, winners)
    # permute the model-family label across targets, keeping the two collections' cells paired the
    # way they are: a target keeps its own two winners, only its family label moves.
    uniq = sorted({t for _, t in targets})
    pos = {t: [i for i, (_, tt) in enumerate(targets) if tt == t] for t in uniq}
    base = [MODEL_FAMILY[t] for t in uniq]
    null = []
    for _ in range(PERM):
        perm = rng.permutation(base)
        lab = np.empty(len(targets), dtype=object)
        for t, p in zip(uniq, perm):
            for i in pos[t]:
                lab[i] = p
        null.append(concentration(lab, winners))
    null = np.array(null)
    p = float(np.mean(null >= obs))

    out = {"cells": len(targets), "observed_concentration": int(obs),
           "null_mean": round(float(null.mean()), 2), "p": round(p, 4),
           "winners": {f"{c}/{t}": w for (c, t), w in zip(targets, winners)}}
    Path("results/family_permutation_20260908.json").write_text(json.dumps(out, indent=2))

    lines = [f"\\newcommand{{\\pjfamobs}}{{{obs}}}",
             f"\\newcommand{{\\pjfamcells}}{{{len(targets)}}}",
             f"\\newcommand{{\\pjfamnull}}{{{null.mean():.1f}}}",
             f"\\newcommand{{\\pjfamp}}{{{p:.3f}}}"]
    txt = Path("paper/robust_numbers.tex").read_text()
    start, end = "% begin family_permutation", "% end family_permutation"
    body = "\n".join([start] + lines + [end])
    if start in txt:
        txt = re.sub(re.escape(start) + r".*?" + re.escape(end), body, txt, flags=re.S)
    else:
        txt = txt.rstrip("\n") + "\n" + body + "\n"
    Path("paper/robust_numbers.tex").write_text(txt)

    print(f"cells {len(targets)}  observed concentration {obs}  null mean {null.mean():.2f}  p={p:.4f}")
    for (c, t), w in zip(targets, winners):
        print(f"  {c:20s} {t:16s} {MODEL_FAMILY[t]:6s} -> {w}")


if __name__ == "__main__":
    raise SystemExit(main())
