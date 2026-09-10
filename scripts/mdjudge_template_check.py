#!/usr/bin/env python3
"""Why did judge_crosscheck*.py get MD-Judge unsafe=0.073 while judge_crosscheck_rows.py got 0.594 on the
same 384 (original, Swahili answer) pairs? The only prompt difference: the former formats the template
without .strip() (leading newline before [INST]); the latter uses .strip() like judge_mdjudge.py (the
paper's official runner). This prints raw MD-Judge outputs under both formattings for the first N rows."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_mdjudge import MD_JUDGE_TEMPLATE, parse_mdjudge
from vllm import LLM, SamplingParams

N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
rows = [json.loads(l) for l in open("results/judge_crosscheck_20260907/crosscheck_rows.jsonl")][:N]
O = [r["original"] for r in rows]; A = [r["answer"] for r in rows]
llm = LLM(model="OpenSafetyLab/MD-Judge-v0.1", dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=4096, enforce_eager=True)
sp = SamplingParams(temperature=0.0, max_tokens=32)
res = {}
for name, tpl in (("stripped", MD_JUDGE_TEMPLATE.strip()), ("raw", MD_JUDGE_TEMPLATE)):
    outs = [o.outputs[0].text for o in llm.generate([tpl % (o, a) for o, a in zip(O, A)], sp, use_tqdm=False)]
    labels = [parse_mdjudge(t)[0] for t in outs]
    res[name] = {"unsafe": sum(l == "unsafe" for l in labels) / N, "parse_valid": sum(l is not None for l in labels) / N,
                 "first_lines": [t.strip().split("\n")[0][:40] for t in outs[:12]]}
print(json.dumps(res, indent=2, ensure_ascii=False))
