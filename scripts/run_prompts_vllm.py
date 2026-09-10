"""Generate target responses for a pre-built prompts jsonl (with 'prompt','original',
'condition',...) via vLLM, parsing [RECONSTRUCTED]/[ANSWER]. For SOTA baselines."""
import json,argparse,os,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from polyjig_pilot import reconstruction_score
from run_polyjig_gated import section
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--outdir",required=True)
    ap.add_argument("--model",default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new-tokens",type=int,default=640)
    ap.add_argument("--gpu-memory-utilization",type=float,default=0.9)
    a=ap.parse_args(); os.makedirs(a.outdir,exist_ok=True)
    jobs=[json.loads(l) for l in open(a.data)]
    from vllm import LLM,SamplingParams
    llm=LLM(model=a.model,dtype="bfloat16",gpu_memory_utilization=a.gpu_memory_utilization,max_model_len=8192)
    convs=[[{"role":"user","content":j["prompt"]}] for j in jobs]
    outs=[o.outputs[0].text for o in llm.chat(convs,SamplingParams(temperature=0.0,max_tokens=a.max_new_tokens),use_tqdm=True)]
    op=f"{a.outdir}/restricted_target_outputs.jsonl"
    fd=os.open(op,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,"w",encoding="utf-8") as h:
        for j,out in zip(jobs,outs):
            rec=section(out,"RECONSTRUCTED","ANSWER"); ans=section(out,"ANSWER")
            rs=reconstruction_score(rec,j["original"])
            h.write(json.dumps({**{k:j[k] for k in ("item_id","scenario","risk_type","condition","is_direct","original")},
                "prompt":j["prompt"],"raw_output":out,"reconstructed":rec,"answer_section":ans,
                "reconstruction_score":round(rs,4),"reconstruction_pass_080":int(rs>=0.8),
                "response_sha256":hashlib.sha256(out.encode()).hexdigest(),
                "target_model":a.model,"engine":"vllm"},ensure_ascii=False)+"\n")
    os.chmod(op,0o600); print("wrote",op,len(jobs))
if __name__=="__main__": main()
