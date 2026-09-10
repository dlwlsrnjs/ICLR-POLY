#!/usr/bin/env python3
"""Shared engine for the PolyJigsaw experiment suite.

The per-dataset driver files (each experiment's mj.py / lg.py) bake in their dataset's settings and
call these helpers. The engine wraps the proven scripts/closed_compare.py phases so there is one
implementation of arm construction, judging, scoring and storage, while the drivers stay dataset-
specific (different languages, order, tlang, benign/harm files).

Guard: `validate_tlang` refuses a translation language that is not actually present in the dataset's
rows, because scripts/method_baselines_eval.build_prompts silently falls back to English for a missing
tlang -- which would turn the "translated" baseline into plain English without any error.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # PolyJigsaw/
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
# All data/order/benign/harm paths in closed_compare are relative to the repo root, so run from there
# no matter which experiment folder the driver was launched from (driver ROOT dirs are absolute).
os.chdir(REPO)
import closed_compare as CC  # noqa: E402


def validate_tlang(harm_path, tlang):
    r = json.loads(open(harm_path).readline())
    langs = r.get("questions", {})
    if tlang not in langs:
        raise SystemExit(f"tlang '{tlang}' is NOT in this dataset's languages {sorted(langs)}; "
                         f"the translated baseline would silently become English. Pick a present language.")
    return tlang


def _ns(**kw):
    d = dict(backend="openai", model="gpt-4o", concurrency=8, max_tokens=320,
             judge_device="cuda:0", util=0.45, max_model_len=4096, no_thinking=False,
             trust_remote_code=False, tokenizer_mode="auto",
             order="AUTO", benign="AUTO", harm="AUTO",
             fp_benign=12, shortlist_k=8, n_items=40, selected="", shortlist="", all_arms=False,
             methods="plain,translated,cipher_base64,aim,deepinception,pap", tlang="Norwegian")
    d.update(kw)
    return argparse.Namespace(**d)


def run_audit(collection, root, **kw):
    # phase_audit does not resolve per-collection defaults, so fill order/harm explicitly.
    ci = CC.COLLECTION_INPUTS[collection]
    kw.setdefault("order", ci["order"])
    kw.setdefault("harm", ci["harm"])
    return CC.phase_audit(_ns(collection=collection, root=root, **kw))


def run_probe(collection, root, **kw):
    return CC.phase_probe(_ns(collection=collection, root=root, **kw))


def run_attack(collection, root, tlang, harm_for_check, **kw):
    validate_tlang(harm_for_check, tlang)
    return CC.phase_attack(_ns(collection=collection, root=root, tlang=tlang, **kw))


def run_full_matrix(collection, root, tag, **kw):
    """FULL-MATRIX harmful collection: evaluate EVERY arm on a target (panel oracle/heterogeneity).
    No shortlist, no method baselines -- every arm is scored and labelled by its own name."""
    return CC.phase_attack(_ns(collection=collection, root=root, tag=tag, all_arms=True,
                               methods="", **kw))


# The panel from the paper: held-in (9) + held-out (7). vLLM HF ids. Closed targets run via exp01.
PANEL_HELD_IN = [
    "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it", "google/gemma-2-9b-it", "google/gemma-2-27b-it",
]
PANEL_HELD_OUT = [
    "microsoft/Phi-3.5-mini-instruct", "mistralai/Mistral-7B-Instruct-v0.3",
    "tiiuae/Falcon3-7B-Instruct", "THUDM/glm-4-9b-chat", "mistralai/Mistral-Small-24B-Instruct-2501",
    "allenai/OLMo-2-1124-7B-Instruct", "HuggingFaceH4/zephyr-7b-beta",
]
PANEL_TAGS = {  # hf id -> short tag used in filenames
    "Qwen/Qwen2.5-3B-Instruct": "qwen25_3b", "Qwen/Qwen2.5-7B-Instruct": "qwen25_7b",
    "Qwen/Qwen2.5-14B-Instruct": "qwen25_14b", "Qwen/Qwen2.5-32B-Instruct": "qwen25_32b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama32_3b_it", "meta-llama/Llama-3.1-8B-Instruct": "llama31_8b_it",
    "google/gemma-2-2b-it": "gemma2_2b_it", "google/gemma-2-9b-it": "gemma2_9b_it",
    "google/gemma-2-27b-it": "gemma2_27b", "microsoft/Phi-3.5-mini-instruct": "phi35_mini",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral7b", "tiiuae/Falcon3-7B-Instruct": "falcon3_7b",
    "THUDM/glm-4-9b-chat": "glm4_9b", "mistralai/Mistral-Small-24B-Instruct-2501": "mistral24b",
    "allenai/OLMo-2-1124-7B-Instruct": "olmo2_7b", "HuggingFaceH4/zephyr-7b-beta": "zephyr7b",
}
