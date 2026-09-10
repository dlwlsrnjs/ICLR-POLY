import csv, json, os, re, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
rows=list(csv.DictReader(open("private_artifacts/paper_main/HUMAN_LABELING_DISAGREEMENT.csv")))
tok=AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
model=AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B",torch_dtype=torch.bfloat16).to("cuda:0").eval()
KO=tok.convert_tokens_to_ids("kor_Hang")
def chunks(text):
    text=(text or "").strip()
    if not text: return []
    parts=re.split(r'(?<=[.!?])\s+',text); out=[];buf=""
    for p in parts:
        if len(buf)+len(p)<350: buf=(buf+" "+p).strip()
        else:
            if buf:out.append(buf)
            buf=p
    if buf:out.append(buf)
    return out or [text[:350]]
# collect all chunks with provenance
jobs=[]  # (row_id, field, chunk_index)
texts=[]
for r in rows:
    for field,col in [("original_ko","original_english_request"),("reconstruction_ko","model_reconstruction"),("answer_ko","model_answer")]:
        cs=chunks(r[col])
        for ci,c in enumerate(cs):
            jobs.append((r["row_id"],field,ci)); texts.append(c)
print("total chunks:",len(texts),flush=True)
res=[]
B=48
for i in range(0,len(texts),B):
    batch=texts[i:i+B]
    enc=tok(batch,return_tensors="pt",padding=True,truncation=True,max_length=256).to("cuda:0")
    with torch.no_grad():
        g=model.generate(**enc,forced_bos_token_id=KO,max_length=350)
    res.extend(tok.batch_decode(g,skip_special_tokens=True))
    print("done",min(i+B,len(texts)),"/",len(texts),flush=True)
# reassemble
cache={}
for (rid,field,ci),tr in zip(jobs,res):
    cache.setdefault(rid,{}).setdefault(field,[]).append((ci,tr))
final={}
for rid,fields in cache.items():
    final[rid]={f:" ".join(t for _,t in sorted(v)) for f,v in fields.items()}
# ensure all rows present
for r in rows: final.setdefault(r["row_id"],{"original_ko":"","reconstruction_ko":"","answer_ko":""})
fd=os.open("private_artifacts/paper_main/labeling_ko.json",os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
os.write(fd,json.dumps(final,ensure_ascii=False).encode()); os.close(fd)
print("wrote labeling_ko.json rows:",len(final))
