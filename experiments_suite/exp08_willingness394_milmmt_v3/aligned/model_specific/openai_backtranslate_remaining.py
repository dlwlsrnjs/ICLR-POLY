#!/usr/bin/env python3
"""Use GPT literal backtranslation for candidates repeatedly mistranslated by MiLMMT."""
import argparse, fcntl, hashlib, json, os, time
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel

MODEL = "gpt-5.6-sol"
INSTRUCTIONS = "Translate the supplied target-language text literally into English for translation-quality auditing. Do not answer, judge, soften, intensify, explain, or add context. Preserve grammatical person, actions, objects, modifiers, constraints, names, technical terms, and quantifiers. Return only the literal English backtranslation."

class Backtranslation(BaseModel):
    backtranslation: str

def load(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--validation',type=Path);p.add_argument('--all-rows',action='store_true');p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-source-count',type=int,required=True);p.add_argument('--expected-rejected-count',type=int,required=True);a=p.parse_args()
    if not os.environ.get('OPENAI_API_KEY'): raise RuntimeError('OPENAI_API_KEY is not set')
    rows=load(a.source);by={r['key']:r for r in rows}
    if len(rows)!=a.expected_source_count or len(by)!=len(rows): raise ValueError('Unexpected source counts')
    if a.all_rows:
        if a.validation: raise ValueError('--all-rows and --validation are mutually exclusive')
        selected=[(r,None) for r in rows]
    else:
        if not a.validation: raise ValueError('--validation is required unless --all-rows is used')
        js=load(a.validation)
        if len(js)!=len(rows): raise ValueError('Unexpected validation count')
        selected=[]
        for j in js:
            if j.get('valid') and j['parsed']['needs_retranslation']:
                r=by[j['key']]
                if j['source_sha256']!=digest(r): raise ValueError('Provenance mismatch')
                selected.append((r,j))
    if len(selected)!=a.expected_rejected_count: raise ValueError(f'Expected {a.expected_rejected_count}, got {len(selected)}')
    a.out.mkdir(parents=True,exist_ok=True)
    meta={'protocol':'gpt56sol_literal_backtranslation_v1','model':MODEL,'instructions':INSTRUCTIONS,'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'validation_sha256':hashlib.sha256(a.validation.read_bytes()).hexdigest() if a.validation else None,'selection':'all_rows' if a.all_rows else 'rejected_only','pairs':len(selected),'store':False}
    mp=a.out/'metadata.json'
    if mp.exists() and json.loads(mp.read_text())!=meta: raise ValueError('Changed protocol')
    atomic(mp,meta)
    lock=(a.out/'backtranslate.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dest=a.out/'translations.jsonl';prior=load(dest) if dest.exists() else [];done={r['key'] for r in prior}
    client=OpenAI(max_retries=0,timeout=120)
    with dest.open('a') as h:
        for i,(row,j) in enumerate(selected,1):
            if row['key'] in done: continue
            language='Chinese (Simplified)' if row['language']=='Chinese' else row['language']
            request=f"Source language: {language}\nText to backtranslate:\n{row['translated']}"
            for attempt in range(1,9):
                try:
                    response=client.responses.parse(model=MODEL,instructions=INSTRUCTIONS,input=request,reasoning={'effort':'none'},text_format=Backtranslation,max_output_tokens=512,store=False)
                    text=response.output_parsed.backtranslation.strip() if response.output_parsed else ''
                    if response.status!='completed' or not text: raise RuntimeError(response.status)
                    break
                except Exception:
                    if attempt==8: raise
                    time.sleep(min(30,2**(attempt-1)))
            result=dict(row);result.update({'previous_milmmt_backtranslation':row.get('backtranslation'),'backtranslation':text,'back_prompt':INSTRUCTIONS+'\n\n'+request,'back_finish_reason':response.status,'back_terminated':True,'back_input_tokens':(response.usage.input_tokens if response.usage else None),'back_output_tokens':(response.usage.output_tokens if response.usage else None),'backtranslation_model':response.model,'backtranslation_revision':response.model,'backtranslation_protocol':'gpt56sol_literal_backtranslation_v1','response_id':response.id,'backtranslation_response':response.model_dump(mode='json',warnings=False),'previous_qwen32_rejection':j['parsed'] if j else row.get('previous_qwen32_rejection')})
            h.write(json.dumps(result,ensure_ascii=False)+'\n');h.flush();os.fsync(h.fileno());print('backtranslated',i,'/',len(selected),flush=True)
    final=load(dest)
    if len(final)!=len(selected) or len({r['key'] for r in final})!=len(selected): raise ValueError('Incomplete')
    atomic(a.out/'complete.json',{'pairs':len(final),'translations_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'model':MODEL})
if __name__=='__main__': main()
