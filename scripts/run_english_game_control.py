"""English-only game control: fragment the English request and wrap it in the
reconstruct-then-answer game, with NO multilingual dispersion. Isolates game-framing
compliance from dispersion-driven intent relocation."""
import json, argparse, os, hashlib, importlib.util as U
def load_mod(name,path):
    s=U.spec_from_file_location(name,path); m=U.module_from_spec(s); s.loader.exec_module(m); return m
probe=load_mod("probe","scripts/run_qwen_interleaving_probe.py")
gated=load_mod("gated","scripts/run_polyjig_gated.py")
def section(text,start,end=None):
    s=text.find(f"[{start}]");
    if s<0: return ""
    s+=len(f"[{start}]")
    if end:
        e=text.find(f"[{end}]",s); return (text[s:e] if e>=0 else text[s:]).strip()
    return text[s:].strip()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--outdir",required=True)
    ap.add_argument("--model",default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--gpu-memory-utilization",type=float,default=0.9)
    a=ap.parse_args(); os.makedirs(a.outdir,exist_ok=True)
    rows=[json.loads(l) for l in open(a.data)]
    from vllm import LLM,SamplingParams
    llm=LLM(model=a.model,dtype="bfloat16",gpu_memory_utilization=a.gpu_memory_utilization,max_model_len=8192)
    convs=[]; meta=[]
    for i,r in enumerate(rows):
        eng=r["questions"]["English"] if "questions" in r else r["original"]
        row={"item_id":r["item_id"],"scenario":r.get("scenario",""),"questions":{"English":eng}}
        pz=probe.build_puzzle(row,["English"],condition="shuffled",seed=7,granularity="coarse",fragments_per_language=5)
        puzzle={"condition":"shuffled","languages":["English"],"mixed_paragraph":pz["mixed_paragraph"]}
        prompt=gated.interleave_game_prompt(puzzle)
        convs.append([{"role":"user","content":prompt}]); meta.append((r,prompt,eng))
    outs=[o.outputs[0].text for o in llm.chat(convs,SamplingParams(temperature=0.0,max_tokens=700),use_tqdm=True)]
    fd=os.open(f"{a.outdir}/restricted_target_outputs.jsonl",os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,"w") as h:
        for (r,prompt,eng),out in zip(meta,outs):
            rec=section(out,"RECONSTRUCTED","ANSWER"); ans=section(out,"ANSWER")
            h.write(json.dumps({"item_id":r["item_id"],"scenario":r.get("scenario",""),"risk_type":r.get("risk_type",""),
                "condition":"english_game","is_direct":0,"original":eng,"prompt":prompt,"raw_output":out,
                "reconstructed":rec,"answer_section":ans,"reconstruction_score":0.0,"reconstruction_pass_080":0,
                "response_sha256":hashlib.sha256(out.encode()).hexdigest(),"target_model":a.model,"engine":"vllm",
                "defense_system_prompt":""},ensure_ascii=False)+"\n")
    print("wrote",a.outdir)
if __name__=="__main__": main()
