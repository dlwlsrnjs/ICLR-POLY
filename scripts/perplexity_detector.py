import json, argparse, math, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--audit", default="private_artifacts/paper_main/lingua_qwen/guard/restricted_qwen3guard_audit.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--per-cond", type=int, default=120)
    ap.add_argument("--out", default="results/paper_perplexity_detector.json")
    a=ap.parse_args()
    rows=[json.loads(l) for l in open(a.audit)]
    from collections import defaultdict
    by=defaultdict(list)
    for r in rows: by[r["condition"]].append(r)
    keep={"english_direct":"english_direct","translated_direct_Finnish":"translation (Finnish)",
          "csrt_all":"CSRT (all)","interleave_ordered_n4":"interleave (n=4)",
          "interleave_ordered_n6":"interleave (n=6)","interleave_ordered_n10":"interleave (n=10)",
          "slot_k1":"slot k=1"}
    rng=random.Random(0)
    tok=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    def ppl(text):
        ids=tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids.to("cuda:0")
        if ids.shape[1]<2: return None
        with torch.no_grad():
            out=model(ids, labels=ids)
        return float(math.exp(min(out.loss.item(),20)))
    res={}
    for cond,label in keep.items():
        rs=by.get(cond,[])
        rng.shuffle(rs); rs=rs[:a.per_cond]
        vals=[]
        for r in rs:
            p=r.get("prompt") or ""
            v=ppl(p)
            if v is not None: vals.append(v)
        vals.sort()
        if vals:
            mean=sum(vals)/len(vals); med=vals[len(vals)//2]
            res[label]={"n":len(vals),"mean_ppl":round(mean,1),"median_ppl":round(med,1),
                        "p10":round(vals[len(vals)//10],1),"p90":round(vals[9*len(vals)//10],1)}
            print(label, res[label])
    # detection: threshold on median english ppl *k; report AUC interleave-n4 vs english
    import itertools
    def auc(pos,neg):
        c=0;t=0
        for x in pos:
            for y in neg:
                t+=1; c+= 1 if x>y else 0.5 if x==y else 0
        return c/t if t else 0
    # recompute raw vals for auc
    def vals_for(cond):
        rs=by.get(cond,[]); rng.shuffle(rs); rs=rs[:a.per_cond]
        vv=[ppl(r.get("prompt") or "") for r in rs]; return [v for v in vv if v]
    pos=vals_for("interleave_ordered_n4"); neg=vals_for("english_direct")
    res["_detection_auc_interleaveN4_vs_english"]=round(auc(pos,neg),3)
    print("AUC interleave-n4 vs english:", res["_detection_auc_interleaveN4_vs_english"])
    json.dump(res, open(a.out,"w"), indent=2)
    print("saved", a.out)

if __name__=="__main__": main()
