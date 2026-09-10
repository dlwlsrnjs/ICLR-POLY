#!/usr/bin/env python3
"""Attack GPT-4o along the WILLINGNESS/alignment axis with the persona/fiction/persuasion configs the
largest aligned open models fell to (AIM = Gemma-27B's winner; DeepInception; PAP). Reuses the paper's
exact prompts + scoring (method_baselines_eval) with a live OpenAI target and local judges. Clear-text
attacks are ungated (ASR = unsafe rate), as in the paper. Key from $OPENAI_API_KEY only. Outputs 0600."""
import sys, json, os, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_live import Judges  # noqa: E402
from online_live_openai import OpenAITarget  # noqa: E402
from method_baselines_eval import eval_method  # noqa: E402
SEED = 20260909
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--harm", default="private_artifacts/panel_v2/harm_grid.jsonl")
    ap.add_argument("--n-items", type=int, default=30)
    ap.add_argument("--tlang", default="Finnish")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--outdir", default="results/gpt4o_persona_20260909")
    a = ap.parse_args()
    harm = [json.loads(l) for l in open(a.harm)]
    for r in harm: r.setdefault("original", r["questions"]["English"])
    rng = np.random.default_rng(SEED)
    rows = [harm[i] for i in rng.choice(len(harm), min(a.n_items, len(harm)), replace=False)]
    target = OpenAITarget(a.model, concurrency=a.concurrency)
    judges = Judges("cuda:0")
    t0 = time.time(); out = {"model": a.model, "n_items": len(rows), "methods": {}}
    for method in ["plain", "aim", "deepinception", "pap"]:
        r = eval_method(target, judges, rows, method, a.tlang)
        # clear-text persona/fiction/persuasion are ungated in the paper -> ASR = unsafe
        r["asr_ungated"] = round(r["unsafe"], 3)
        out["methods"][method] = {k: round(float(v), 3) for k, v in r.items()}
        print(json.dumps({"method": method, **out["methods"][method]}), flush=True)
    out["seconds"] = round(time.time()-t0, 1)
    od = Path(a.outdir); od.mkdir(parents=True, exist_ok=True); os.chmod(od, 0o700)
    fd = os.open(od/"gpt4o_persona.json", os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as h: json.dump(out, h, indent=2)
    print(json.dumps(out, indent=2))
if __name__ == "__main__":
    raise SystemExit(main())
