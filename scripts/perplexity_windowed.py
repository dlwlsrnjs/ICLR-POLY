import json, math, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
rows=[json.loads(l) for l in open("private_artifacts/paper_main/lingua_qwen/guard/restricted_qwen3guard_audit.jsonl")]
by=defaultdict(list)
for r in rows: by[r["condition"]].append(r)
keep={"english_direct":"english","interleave_ordered_n4":"interleave(n=4)","interleave_ordered_n6":"interleave(n=6)","csrt_all":"CSRT"}
tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16, device_map="cuda:0").eval()
W=16
def wmax(text):
    ids=tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids.to("cuda:0")
    if ids.shape[1]<W+1: return None
    with torch.no_grad():
        logits=model(ids).logits
    logp=torch.log_softmax(logits[0,:-1],dim=-1)
    tgt=ids[0,1:]
    nll=-logp[range(tgt.shape[0]),tgt]  # per-token nll
    # sliding window mean nll -> max window ppl
    if nll.shape[0]<W: return math.exp(min(nll.mean().item(),20))
    wins=nll.unfold(0,W,1).mean(dim=1)  # sliding-window mean nll
    return math.exp(min(wins.max().item(),20))
rng=random.Random(0); res={}
for cond,lab in keep.items():
    rs=by.get(cond,[]); rng.shuffle(rs); rs=rs[:100]
    vals=[wmax(r.get("prompt") or "") for r in rs]; vals=[v for v in vals if v]; vals.sort()
    res[lab]={"n":len(vals),"median_wmax_ppl":round(vals[len(vals)//2],1),"p90":round(vals[9*len(vals)//10],1)}
    print(lab,res[lab])
def auc(pos,neg):
    c=t=0
    for x in pos:
        for y in neg: t+=1; c+=1 if x>y else 0.5 if x==y else 0
    return c/t if t else 0
def vf(cond):
    rs=by.get(cond,[]); rng.shuffle(rs); rs=rs[:100]
    return [v for v in (wmax(r.get("prompt") or "") for r in rs) if v]
res["_auc_windowedmax_interleaveN4_vs_english"]=round(auc(vf("interleave_ordered_n4"),vf("english_direct")),3)
print("windowed-max AUC interleave-n4 vs english:",res["_auc_windowedmax_interleaveN4_vs_english"])
json.dump(res, open("results/paper_perplexity_windowed.json","w"), indent=2)
