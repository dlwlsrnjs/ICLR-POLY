#!/usr/bin/env python3
"""Build the final 2317/2317 accepted multilingual translation release."""
import argparse, hashlib, json, shutil
from collections import Counter
from pathlib import Path

def load(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def write_json(path,value): path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as h:
        for row in rows:h.write(json.dumps(row,ensure_ascii=False)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists(): raise ValueError('Output exists')
    shutil.copytree(a.base,a.out,ignore=shutil.ignore_patterns('.cache'))
    rows=load(a.base/'data/translations_all.jsonl');by={r['key']:r for r in rows};unresolved={r['key'] for r in rows if not r['qa_pass']}
    if len(rows)!=2317 or len(by)!=2317 or len(unresolved)!=54: raise ValueError('Unexpected base release')
    stages=[
      ('remaining54_v2',a.runs/'gpt56sol_remaining54_v2_milmmt_backtranslation',a.runs/'gpt56sol_remaining54_v2_qwen32_validation',37),
      ('remaining17_v3',a.runs/'gpt56sol_remaining17_v3_milmmt_backtranslation',a.runs/'gpt56sol_remaining17_v3_qwen32_validation',9),
      ('remaining8_v3_gpt_back',a.runs/'gpt56sol_remaining8_v3_gpt_backtranslation',a.runs/'gpt56sol_remaining8_v3_qwen32_validation',4),
      ('remaining4_v4',a.runs/'gpt56sol_remaining4_v4_gpt_backtranslation',a.runs/'gpt56sol_remaining4_v4_qwen32_validation',3),
      ('remaining1_v7',a.runs/'gpt56sol_remaining1_v7_gpt_backtranslation',a.runs/'gpt56sol_remaining1_v7_qwen32_validation',1),
    ]
    replacements={};stage_for={};verdict_for={}
    for name,source_dir,qa_dir,expected in stages:
        candidates={r['key']:r for r in load(source_dir/'translations.jsonl')}
        judgments={j['key']:j for j in load(qa_dir/'judgments.jsonl')}
        accepted={k for k,j in judgments.items() if j.get('valid') and not j['parsed']['needs_retranslation']}
        if len(accepted)!=expected: raise ValueError(f'{name}: expected {expected}, got {len(accepted)}')
        for key in accepted:
            if key in replacements: raise ValueError(f'Duplicate accepted key: {key}')
            replacements[key]=candidates[key];stage_for[key]=name;verdict_for[key]=judgments[key]
    if set(replacements)!=unresolved: raise ValueError('Final stages do not cover all 54 unresolved keys')

    merged=[]
    for old in rows:
        key=old['key']
        if key not in replacements:
            merged.append(old);continue
        c=replacements[key];j=verdict_for[key];usage=(c.get('response') or {}).get('usage') or {}
        row=dict(old)
        row.update({
          'translated':c['translated'],'forward_prompt':c.get('request_instructions','')+'\n\n'+c.get('request_input',''),
          'forward_finish_reason':(c.get('response') or {}).get('status','completed'),'forward_terminated':True,
          'forward_input_tokens':usage.get('input_tokens'),'forward_output_tokens':usage.get('output_tokens'),
          'translation_model':c.get('response_model','gpt-5.6-sol'),'translation_revision':c.get('response_model','gpt-5.6-sol'),
          'backtranslation':c['backtranslation'],'back_prompt':c.get('back_prompt'),'back_finish_reason':c.get('back_finish_reason'),
          'back_terminated':c.get('back_terminated',True),'back_input_tokens':c.get('back_input_tokens'),'back_output_tokens':c.get('back_output_tokens'),
          'backtranslation_model':c.get('backtranslation_model'),'backtranslation_revision':c.get('backtranslation_revision'),
          'qa_pass':True,'qa_reason':j['parsed']['reason'],'qa_source_sha256':j['source_sha256'],
          'qa_protocol':'translation_semantics_rereview_v1','repair_stage':stage_for[key],
          'repair_protocol':c.get('repair_protocol'),'response_id':c.get('response_id'),
        })
        row.pop('latest_repair_status',None);row.pop('latest_repair_provenance',None)
        if row['english_original']!=old['english_original'] or row['language']!=old['language']: raise ValueError('Identity changed')
        merged.append(row)
    if len(merged)!=2317 or len({r['key'] for r in merged})!=2317 or not all(r['qa_pass'] for r in merged): raise ValueError('Final data invariant failed')
    if [r['key'] for r in merged]!=[r['key'] for r in rows]: raise ValueError('Ordering changed')
    counts=Counter(r['language'] for r in merged)
    if set(counts.values())!={331}: raise ValueError('Language count changed')
    write_jsonl(a.out/'data/translations_all.jsonl',merged);write_jsonl(a.out/'data/translations_qa_accepted.jsonl',merged);write_jsonl(a.out/'data/translations_needs_review.jsonl',[])
    summary={'source_bank_items':331,'english_unchanged':True,'foreign_languages':7,'foreign_pairs':2317,'qa_pass':2317,'qa_needs_review':0,'languages':{lang:{'total':331,'pass':331,'needs_review':0} for lang in sorted(counts)},'repair_resolution':{'original_release_v2_accepted':2263,**{name:expected for name,_,_,expected in stages}},'qa_is_automatic_not_human_gold':True,'intended_use':'Multilingual overrefusal prior evaluation; all 2317 translation pairs passed the frozen Qwen2.5-32B semantics-QA protocol.'}
    write_json(a.out/'SUMMARY.json',summary)
    prov=a.out/'provenance/repair_20260920/final54'
    dirs=['gpt56sol_remaining54_v2','gpt56sol_remaining54_v2_milmmt_backtranslation','gpt56sol_remaining54_v2_qwen32_validation','gpt56sol_remaining17_v3','gpt56sol_remaining17_v3_milmmt_backtranslation','gpt56sol_remaining17_v3_qwen32_validation','gpt56sol_remaining8_v3_gpt_backtranslation','gpt56sol_remaining8_v3_qwen32_validation','gpt56sol_remaining4_v4','gpt56sol_remaining4_v4_gpt_backtranslation','gpt56sol_remaining4_v4_qwen32_validation','gpt56sol_remaining1_v5','gpt56sol_remaining1_v5_gpt_backtranslation','gpt56sol_remaining1_v5_qwen32_validation','gpt56sol_remaining1_v6','gpt56sol_remaining1_v6_gpt_backtranslation','gpt56sol_remaining1_v6_qwen32_validation','gpt56sol_remaining1_v7','gpt56sol_remaining1_v7_gpt_backtranslation','gpt56sol_remaining1_v7_qwen32_validation']
    for name in dirs: shutil.copytree(a.runs/name,prov/name,ignore=shutil.ignore_patterns('*.lock'))
    write_json(prov/'FINAL_AUDIT.json',{'input_unresolved':54,'accepted_by_stage':{name:expected for name,_,_,expected in stages},'final_unresolved':0,'final_qa_pass':2317,'resolution_by_key':stage_for})
    readme=(a.base/'README.md').read_text()
    readme=readme.replace("- config_name: qa_needs_review\n  data_files:\n  - split: train\n    path: data/translations_needs_review.jsonl\n","")
    readme+='''\n\n## Final repair revision\n\n남아 있던 54쌍을 반복 교정하고 고정 Qwen2.5-32B 의미 검증으로 모두 통과시켰습니다. 최종 상태는 전체 2,317쌍 중 QA 통과 2,317쌍, 재검토 0쌍입니다. 영어 원문331개와 언어별331개 구성은 변경하지 않았습니다. 모든 중간 번역, 역번역, 실패 판정, 프롬프트, 모델 설정과 API 응답은 `provenance/repair_20260920/final54/`에 보존했습니다.\n'''
    (a.out/'README.md').write_text(readme)
    hashes={}
    for path in sorted(a.out.rglob('*')):
        if path.is_file() and path.name!='SHA256SUMS.json' and not any(x.startswith('.') for x in path.relative_to(a.out).parts): hashes[str(path.relative_to(a.out))]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(a.out/'SHA256SUMS.json',hashes)
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('hashed_files',len(hashes))
if __name__=='__main__':main()
