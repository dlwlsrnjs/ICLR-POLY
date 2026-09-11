# exp05 — willingness-axis prior (over-refusal frame probe)

Canonical recorder: [`willingness_prior.py`](willingness_prior.py).
Full rationale, provenance schema, results, and honesty boundaries: [WILLINGNESS_PRIOR.md](WILLINGNESS_PRIOR.md).

TL;DR: plain benign probes saturate the willingness axis on helpful models, so we probe the frames on
**over-refusal** datasets (XSTest → OR-Bench-Hard → FalseReject) to de-saturate it and read a per-model
willingness fingerprint that transfers to harmful comply. Outputs are
`results/willingness_prior_<dataset>_<tag>.json` with full provenance (dataset+n+sha, frames, scoring,
metrics, frame-effect-vs-plain).

> The earlier `orbench_frames.py` / `falsereject_frames.py` throwaways printed "XSTest" regardless of
> input and saved provenance-less JSONs; they are superseded by `willingness_prior.py`. Their numbers
> were retro-wrapped into the v1 schema (`live:false`).
