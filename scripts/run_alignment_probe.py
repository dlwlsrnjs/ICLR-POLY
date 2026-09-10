#!/usr/bin/env python3
"""Generate a target model's responses to an alignment probe file (vLLM, greedy).

The probe measures the target's refusal barrier; it builds no PolyJigsaw puzzle.
Over-refusal output is benign (0644); direct-harm output is controlled (0600) and
carries the field aliases that rejudge_qwen3guard_official.py expects, so the same
official safety judge used for the attack results scores it.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--no-thinking", action="store_true",
                    help="disable Qwen3-style thinking mode, matching the panel runs")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    jobs = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    restricted = any(j.get("probe") == "direct_harm" for j in jobs)

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_memory_utilization,
              max_model_len=a.max_model_len, tensor_parallel_size=a.tensor_parallel_size,
              trust_remote_code=True)
    convs = [[{"role": "user", "content": j["prompt"]}] for j in jobs]
    chat_kwargs = {"chat_template_kwargs": {"enable_thinking": False}} if a.no_thinking else {}
    outs = [o.outputs[0].text for o in
            llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens),
                     use_tqdm=True, **chat_kwargs)]

    name = "restricted_probe_outputs.jsonl" if restricted else "probe_outputs.jsonl"
    op = Path(a.outdir) / name
    mode = 0o600 if restricted else 0o644
    fd = os.open(op, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as h:
        for j, o in zip(jobs, outs):
            row = {"probe": j["probe"], "id": j["id"], "type": j["type"],
                   "prompt": j["prompt"], "response": o,
                   "response_sha256": hashlib.sha256(o.encode()).hexdigest(),
                   "target_model": a.model, "engine": "vllm",
                   "thinking_mode": "off" if a.no_thinking else "default"}
            if restricted:
                # aliases consumed by the official Qwen3Guard judging script
                row.update({"item_id": j["id"], "condition": "english_direct",
                            "original": j.get("original", j["prompt"]), "raw_output": o})
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.chmod(op, mode)
    print(json.dumps({"wrote": str(op), "n": len(jobs), "model": a.model, "restricted": restricted}))

if __name__ == "__main__":
    main()
