#!/usr/bin/env python3
"""The warm-start prior, computed from harmless probes only.

Everything that needs a prior imports `prior_for` from here, so there is exactly one definition and
it cannot silently fall back to the reconstruction column of the harmful runs (which shares a factor
with the target: verified ASR is recon AND unsafe).

Which harmless measurement to use is not hard-coded. `benign_prior_selection.py` compares candidates
and records, for every target, the candidate chosen on the *other* targets; `prior_for` replays that
leave-one-target-out choice, so a number produced with this prior is honest about probe selection.

Sources, all harmless:
  results/benign_arms_20260907        reconstruction of harmless FLORES puzzles, Lingua construction
  results/benign_arms_mj_20260907     the same for the MultiJail construction
  results/benign_signals_20260907     answer-frame signals on 16 innocuous requests (Norwegian)
  results/benign_signals_mj_20260907  the same with Swahili
  results/benign_borderline_*         the same signals on 16 harmless but safety-adjacent requests
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

SELECTION = Path("results/benign_prior_selection_20260907.json")
SOURCES = {
    "MultiJail": dict(recon="results/benign_arms_mj_20260907",
                      sig="results/benign_signals_mj_20260907",
                      bord="results/benign_borderline_mj_20260907"),
    "Lingua-SafetyBench": dict(recon="results/benign_arms_20260907",
                               sig="results/benign_signals_20260907",
                               bord="results/benign_borderline_20260907"),
}
SIGNALS = ("nonrefusal", "lang_fidelity", "persona_hold", "fiction_hold", "detail")
DEFAULT = "benign-recon"          # used when no selection file exists yet


def _read(path, target, key):
    f = Path(path) / f"{target}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())[key]


def candidates(collection, target, arms):
    """{candidate name: vector over `arms`}, harmless measurements only."""
    src = SOURCES[collection]
    out = {}
    ar = _read(src["recon"], target, "arm_recon")
    if ar is not None:
        # tri_hi_en and tri_triple_en share the high-resource puzzle, so they reconstruct like the
        # probe's tri_triple_en / tri_hi_wl entries; map new arm names onto the probed constructions.
        alias = {"tri_hi_en": "tri_hi_wl", "tri_triple_en": "tri_triple_en"}
        out["benign-recon"] = np.array([1.0 if a == "m_translated" else ar[alias.get(a, a)] for a in arms], float)
    for tag, key in (("", "sig"), ("borderline-", "bord")):
        sg = _read(src[key], target, "arm_signals")
        if sg is None:
            continue
        salias = {"tri_hi_en": "tri_hi_wl", "tri_triple_en": "tri_triple_en", "m_translated": "m_pap"}
        for s in SIGNALS:
            v = np.array([sg[salias.get(a, a)][s] for a in arms], float)
            out[f"benign-{tag}{s}"] = v
            if "benign-recon" in out:
                out[f"benign-recon x {tag}{s}"] = out["benign-recon"] * v
    return out


def chosen_candidate(collection, target):
    """The candidate selected on the other targets, or the default if no selection has been run."""
    if not SELECTION.exists():
        return DEFAULT
    sel = json.loads(SELECTION.read_text())
    if collection not in sel:
        return DEFAULT
    return sel[collection]["__loo__"]["picked_per_target"].get(target, DEFAULT)


def prior_for(collection, target, arms, strict=True):
    """Warm-start prior vector for one target. Raises if the harmless probe is missing and
    `strict`, because silently substituting anything else is how the leak happened."""
    cand = candidates(collection, target, arms)
    name = chosen_candidate(collection, target)
    if name in cand:
        return cand[name], name
    if DEFAULT in cand:
        return cand[DEFAULT], DEFAULT
    if strict:
        raise FileNotFoundError(
            f"no harmless probe for {collection}/{target}; run the probe rather than falling back")
    return np.full(len(arms), 0.5), "flat (probe missing)"
