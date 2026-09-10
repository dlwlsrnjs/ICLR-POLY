#!/usr/bin/env python3
"""Game-framing ablation (MIDAS Table 5, "w/o Game-Style Reasoning" -- their biggest lever,
80->22). At a fixed config (resource-order language set, n) we compare the reassembly-GAME
prompt vs the no-game prompt on the same interleaved fragments, on true harmful joint ASR.
The drop from game -> nogame isolates the contribution of the puzzle/reconstruction framing."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen_interleaving_probe import ENGLISH, build_puzzle
from run_polyjig_gated import section, interleave_game_prompt, interleave_nogame_prompt
from online_live import LiveTarget, Judges

SEED = 20260828


def eval_prompts(target, judges, rows, prompts):
    outs = target.generate(prompts)
    O = [r["original"] for r in rows]
    R = [section(o, "RECONSTRUCTED", "ANSWER") for o in outs]
    A = [section(o, "ANSWER") for o in outs]
    Rj = [bool(x) if x is not None else False for x in judges.recon_equivalent(O, R)]
    Uj = [bool(x) if x is not None else False for x in judges.answer_unsafe(O, A)]
    J = [1.0 if (a and b) else 0.0 for a, b in zip(Rj, Uj)]
    return dict(gated=float(np.mean(J)), recon=float(np.mean(Rj)), unsafe=float(np.mean(Uj)))


def build(row, langs, n, F):
    return build_puzzle({"item_id": row["item_id"], "scenario": row["scenario"], "questions": row["questions"]},
                        langs[:n], "ordered", SEED, "coarse", F)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--order", default="results/lang_rank_20260905/resource_order.json")
    ap.add_argument("--harm", default="private_artifacts/panel_v2/harm_grid.jsonl")
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--F", type=int, default=5)
    ap.add_argument("--n-items", type=int, default=40)
    ap.add_argument("--util", type=float, default=0.35); ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--no-thinking", action="store_true"); ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer-mode", default="auto")
    ap.add_argument("--outdir", default="results/game_ablation_20260906")
    a = ap.parse_args()
    order = json.loads(Path(a.order).read_text())["order"]
    langs = [ENGLISH] + order[:a.n - 1]
    rows = [json.loads(l) for l in open(a.harm)][:a.n_items]
    for r in rows:
        r.setdefault("original", r["questions"]["English"])
    print(json.dumps({"stage": "loading", "target": a.tag, "n": a.n}), flush=True)
    target = LiveTarget(a.target, a.util, a.max_model_len, a.no_thinking, a.trust_remote_code, a.tokenizer_mode)
    judges = Judges("cuda:0")
    t0 = time.time()
    puzzles = [build(r, langs, a.n, a.F) for r in rows]
    game = eval_prompts(target, judges, rows, [interleave_game_prompt(p) for p in puzzles])
    nogame = eval_prompts(target, judges, rows, [interleave_nogame_prompt(p) for p in puzzles])
    res = dict(target=a.tag, n=a.n, langs=langs,
               game={k: round(game[k], 3) for k in game},
               nogame={k: round(nogame[k], 3) for k in nogame},
               game_lift=round(game["gated"] - nogame["gated"], 3),
               seconds=round(time.time() - t0, 1))
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True); os.chmod(out, 0o700)
    fd = os.open(out / f"{a.tag}.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as h:
        json.dump(res, h, indent=2)
    print(json.dumps({"target": a.tag, "game": game["gated"], "nogame": nogame["gated"],
                      "game_lift": res["game_lift"]}))


if __name__ == "__main__":
    raise SystemExit(main())
