import json,argparse,os,hashlib,time,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from polyjig_pilot import reconstruction_score
from run_polyjig_gated import section
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--outdir",required=True)
    ap.add_argument("--model",required=True); ap.add_argument("--max-new-tokens",type=int,default=640)
    ap.add_argument("--concurrency",type=int,default=8); ap.add_argument("--max-retries",type=int,default=6)
    a=ap.parse_args(); os.makedirs(a.outdir,exist_ok=True)
    key=os.environ["OPENAI_API_KEY"]; from openai import OpenAI; client=OpenAI(api_key=key)
    jobs=[json.loads(l) for l in open(a.data)]
    def call(p):
        d=2.0
        for att in range(a.max_retries):
            try:
                r=client.chat.completions.create(model=a.model,messages=[{"role":"user","content":p}],temperature=0.0,max_tokens=a.max_new_tokens,store=False)
                return r.choices[0].message.content or ""
            except Exception as e:
                if att==a.max_retries-1: return f"__API_ERROR__: {e}"
                time.sleep(d); d=min(d*2,40)
    outs=[None]*len(jobs); done=0
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        fut={ex.submit(call,j["prompt"]):i for i,j in enumerate(jobs)}
        for f in as_completed(fut):
            outs[fut[f]]=f.result(); done+=1
            if done%100==0 or done==len(jobs): print(json.dumps({"stage":"openai","completed":done,"total":len(jobs)}),flush=True)
    op=f"{a.outdir}/restricted_target_outputs.jsonl"; fd=os.open(op,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,"w",encoding="utf-8") as h:
        for j,out in zip(jobs,outs):
            if out.startswith("__API_ERROR__"): continue
            rec=section(out,"RECONSTRUCTED","ANSWER"); ans=section(out,"ANSWER"); rs=reconstruction_score(rec,j["original"])
            h.write(json.dumps({**{k:j[k] for k in ("item_id","scenario","risk_type","condition","is_direct","original")},
                "prompt":j["prompt"],"raw_output":out,"reconstructed":rec,"answer_section":ans,
                "reconstruction_score":round(rs,4),"reconstruction_pass_080":int(rs>=0.8),
                "response_sha256":hashlib.sha256(out.encode()).hexdigest(),"target_model":a.model,"engine":"openai"},ensure_ascii=False)+"\n")
    os.chmod(op,0o600); print("wrote",op)
if __name__=="__main__": main()
