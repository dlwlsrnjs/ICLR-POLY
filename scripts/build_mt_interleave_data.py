"""Build a Lingua-format jsonl whose per-language 'questions' are NLLB machine
translations of the English request (instead of the official parallel translations),
to test PolyJigsaw's robustness to translation quality."""
import json, argparse, os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

NLLB_CODE={"Chinese":"zho_Hans","French":"fra_Latn","Arabic":"arb_Arab","Russian":"rus_Cyrl",
 "Spanish":"spa_Latn","German":"deu_Latn","Japanese":"jpn_Jpan","Finnish":"fin_Latn","Norwegian":"nob_Latn"}
ORDER=["Chinese","French","Arabic","Russian","Spanish","German","Japanese","Finnish","Norwegian"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--model",default="facebook/nllb-200-distilled-1.3B")
    ap.add_argument("--device",default="cuda:0"); ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args()
    rows=[json.loads(l) for l in open(a.data)]
    if a.limit: rows=rows[:a.limit]
    tok=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForSeq2SeqLM.from_pretrained(a.model,torch_dtype=torch.bfloat16).to(a.device).eval()
    def translate(texts,tgt):
        out=[]
        for i in range(0,len(texts),a.batch_size):
            batch=texts[i:i+a.batch_size]
            enc=tok(batch,return_tensors="pt",padding=True,truncation=True,max_length=256).to(a.device)
            bos=tok.convert_tokens_to_ids(tgt)
            with torch.no_grad():
                gen=model.generate(**enc,forced_bos_token_id=bos,max_length=256)
            out.extend(tok.batch_decode(gen,skip_special_tokens=True))
        return out
    eng=[r["original"] for r in rows]
    mt={"English":eng}
    for lang in ORDER:
        print("translating ->",lang,flush=True)
        mt[lang]=translate(eng,NLLB_CODE[lang])
    fd=os.open(a.output,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,"w") as h:
        for i,r in enumerate(rows):
            q={"English":eng[i]}
            for lang in ORDER: q[lang]=mt[lang][i]
            newr=dict(r); newr["questions"]=q; newr["mt_source"]="nllb-200-distilled-1.3B"
            h.write(json.dumps(newr,ensure_ascii=False)+"\n")
    os.chmod(a.output,0o600)
    print("wrote",a.output,len(rows))
if __name__=="__main__": main()
