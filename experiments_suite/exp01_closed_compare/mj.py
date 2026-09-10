#!/usr/bin/env python3
"""EXP01 closed-model comparison -- MultiJail driver (baselines + ours, collected together).

MultiJail settings (DIFFERENT from Lingua): 10 languages, low-resource-first resource order
(Bengali, Swahili, Javanese < Arabic, Korean, Thai < Chinese, Italian, Vietnamese). Translation
baseline language = the lowest-resource present language (Bengali). Order / benign / harm files are
MultiJail's; the engine resolves them from the collection name.

Phases (see the shared engine / scripts/closed_compare.py):
  audit  offline, no API/GPU: build every arm, report duplication  (safe pilot)
  probe  HARMLESS: benign plaintext setting-selection -> shortlist
  attack HARMFUL: fire ours(shortlist) + baselines, verified scoring, raw retained

Usage:
  python mj.py audit
  python mj.py probe  [--fp-benign 12] [--judge-device cuda:1]
  python mj.py attack [--n-items 40] [--judge-device cuda:1]
"""
import sys, argparse, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import engine  # noqa: E402

COLLECTION = "MultiJail"
TLANG = "Bengali"                       # lowest-resource present language (NOT Norwegian: absent in MJ)
ROOT = str(Path(__file__).resolve().parent / "results")
TAG = "gpt-4o_mj"
BASELINES = "plain,translated,cipher_base64,aim,deepinception,pap"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["audit", "probe", "attack"])
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--backend", default="openai")
    ap.add_argument("--judge-device", default="cuda:0")
    ap.add_argument("--fp-benign", type=int, default=12)
    ap.add_argument("--n-items", type=int, default=40)
    a = ap.parse_args()
    if a.phase == "audit":
        return engine.run_audit(COLLECTION, ROOT, n_items=3)
    if a.phase == "probe":
        return engine.run_probe(COLLECTION, ROOT, model=a.model, backend=a.backend, tag=TAG,
                                fp_benign=a.fp_benign, judge_device=a.judge_device)
    # attack: read phase-1 shortlist, fire ours + baselines
    short = ",".join(json.loads((Path(ROOT) / "benign" / f"{TAG}.json").read_text())["shortlist"])
    print(f">> {TAG} shortlist = {short}", flush=True)
    return engine.run_attack(COLLECTION, ROOT, TLANG, engine.CC.COLLECTION_INPUTS[COLLECTION]["harm"],
                             model=a.model, backend=a.backend, tag=TAG, shortlist=short,
                             methods=BASELINES, n_items=a.n_items, judge_device=a.judge_device)


if __name__ == "__main__":
    raise SystemExit(main())
