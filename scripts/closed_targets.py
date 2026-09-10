#!/usr/bin/env python3
"""Commercial closed-model targets for the closed-panel expansion (GPT-4o, Gemini, Claude), each a
drop-in with a `.generate(prompts) -> list[str]` method matching online_live_openai.OpenAITarget, so the
two-axis selector / capstone / axis-decomposition drivers run unchanged. Keys come ONLY from the 0600
files under /home/ubuntu/342/jinkwon/.secrets (loaded into env by the caller; never inline, never logged).
Judges stay local. Authorized red-team evaluation only; restricted outputs 0600.

Backends:
  openai    : OpenAI chat.completions (key OPENAI_API_KEY).
  gemini    : Gemini via its OpenAI-compatible endpoint (key GEMINI_API_KEY); no `store` arg.
  anthropic : Anthropic Messages API (key ANTHROPIC_API_KEY).
"""
from __future__ import annotations
import os, time
from concurrent.futures import ThreadPoolExecutor

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

class _ChatTarget:
    """OpenAI-compatible chat backend (OpenAI or Gemini compat), threaded, temp 0."""
    def __init__(self, model, api_key, base_url=None, max_tokens=320, concurrency=8, store=False):
        from openai import OpenAI
        if not api_key: raise SystemExit("missing API key (load from .secrets file into env)")
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.base_url = base_url
        self.model = model; self.mt = max_tokens; self.cc = concurrency; self.store = store
    def _one(self, p):
        gem = self.base_url is not None
        for a in range(5):
            try:
                kw = dict(model=self.model, messages=[{"role":"user","content":p}],
                          temperature=0.0, max_tokens=self.mt)
                if self.store is not None and self.model.startswith("gpt"): kw["store"] = self.store
                # Gemini 3.x spends tokens on hidden "thinking"; disable it so max_tokens is all answer
                # (otherwise a long jailbreak answer truncates, understating ASR). reasoning_effort is
                # the Gemini OpenAI-compat knob -- pass ONLY this (thinking_config alongside it is a 400).
                if gem:
                    kw["reasoning_effort"] = "none"
                r = self.client.chat.completions.create(**kw)
                return r.choices[0].message.content or ""
            except Exception:
                if a == 4: return ""
                time.sleep(min(2*2**a, 20))
    def generate(self, prompts):
        with ThreadPoolExecutor(max_workers=self.cc) as ex:
            return list(ex.map(self._one, prompts))

class _AnthropicTarget:
    def __init__(self, model, api_key, max_tokens=320, concurrency=4):
        import anthropic
        if not api_key: raise SystemExit("missing ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key); self.model = model
        self.mt = max_tokens; self.cc = concurrency
    def _one(self, p):
        for a in range(5):
            try:
                # NB: the installed anthropic SDK (1.3.0) rejects `temperature` as a kwarg
                # (TypeError), so we omit it and take the model's default; extracting only text blocks.
                r = self.client.messages.create(model=self.model, max_tokens=self.mt,
                        messages=[{"role":"user","content":p}])
                return "".join(getattr(b,"text","") for b in r.content if getattr(b,"type","")=="text")
            except Exception:
                if a == 4: return ""
                time.sleep(min(2*2**a, 20))
    def generate(self, prompts):
        with ThreadPoolExecutor(max_workers=self.cc) as ex:
            return list(ex.map(self._one, prompts))

def make_target(backend, model, concurrency=8, max_tokens=320):
    if backend == "openai":
        return _ChatTarget(model, os.environ.get("OPENAI_API_KEY"), None, max_tokens, concurrency)
    if backend == "gemini":
        return _ChatTarget(model, os.environ.get("GEMINI_API_KEY"), GEMINI_BASE, max_tokens, concurrency, store=None)
    if backend == "anthropic":
        return _AnthropicTarget(model, os.environ.get("ANTHROPIC_API_KEY"), max_tokens, min(concurrency,4))
    raise SystemExit(f"unknown backend {backend}")
