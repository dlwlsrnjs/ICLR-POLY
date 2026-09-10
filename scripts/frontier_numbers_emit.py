#!/usr/bin/env python3
"""Emit LaTeX macros for the commercial-frontier (GPT-4o) + capstone results from the stored JSONs.
Run as jinkwon (0600 results); writes results/frontier_numbers.tex (copy into paper/)."""
import json, os
from pathlib import Path
def g(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None
L=[]
def m(name,val): L.append(f"\\newcommand{{\\pj{name}}}{{{val}}}")
# axis decomposition on GPT-4o (same 30-item pool)
d=g("results/axis_decomp_20260909/gpt4o.json")
if d:
    a=d["arms"]
    m("gfourpuzzle", f"{a['frag5_shuffled_n6']['verified']:.3f}")      # best pure puzzle
    m("gfouraim", f"{a['aim']['verified']:.3f}")
    m("gfourpap", f"{a['pap']['verified']:.3f}")
    m("gfourdeepinc", f"{a['deepinception']['verified']:.3f}")
    m("gfourfiction", f"{a['fiction_only']['verified']:.3f}")
    m("gfourcombpersona", f"{a['comb_persona']['verified']:.3f}")
    m("gfourcombincept", f"{a['comb_incept']['verified']:.3f}")
    m("gfourrolesplit", f"{a['role_split_persona']['verified']:.3f}")
    m("gfourdecompn", str(d.get("n_items")))
# capstone GPT-4o
c=g("results/capstone_20260909/gpt4o_capstone.json")
if c:
    m("capgfourrecon", f"{c['benign_recon']:.2f}")
    m("capgfouraimadh", f"{c['adherence']['aim']:.2f}")
    m("capgfourfictadh", f"{c['adherence']['deepinception']:.2f}")
    m("capgfourpapadh", f"{c['adherence']['pap']:.2f}")
    m("capgfourbest", c['best_arm'].replace('_','\\_'))
    m("capgfourbestv", f"{c['best_verified']:.2f}")
    m("capgfourq", str(c['first_success'] if c['first_success'] else c['queries']))
# capstone Mistral-7B (held-out open)
cm=g("results/capstone_20260909/mistral7b_capstone.json")
if cm:
    m("capmrecon", f"{cm['benign_recon']:.2f}")
    m("capmaimadh", f"{cm['adherence']['aim']:.2f}")
    m("capmbest", cm['best_arm'].replace('_','\\_'))
    m("capmbestv", f"{cm['best_verified']:.2f}")
    m("capmq", str(cm['first_success'] if cm['first_success'] else cm['queries']))
# raw-vs-verified on GPT-4o (n10 shuffle)
r=g("results/gpt4o_n10shuffle_20260909/result.json")
if r and "frag5_shuffled_n10" in r["conds"]:
    m("gfourntenverif", f"{r['conds']['frag5_shuffled_n10']['verified']:.3f}")
m("gfourntenraw","0.652")  # old_emph interleave_shuffled_n10 raw non-refusal (paper_frontier_gpt4o_old_emph)
open("results/frontier_numbers.tex","w").write("\n".join(L)+"\n")
print("\n".join(L)); print(f"# {len(L)} macros")
