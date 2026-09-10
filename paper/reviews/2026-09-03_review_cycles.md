# Independent manuscript review cycles — 2026-09-03

These are internal reviewer-agent reports, not conference reviews. Two reviewers
worked independently: `editorial_review` and `evidence_review`. This record summarizes
their findings and the revisions; it does not reproduce a numerical acceptance score.
The user's previously reported Claude score of 6 was not independently located or verified.

## Round 1: substantive evidence and argument review

Both reviewers identified claims exceeding the available evidence. The main issues were:

- Capability-law and online BO efficiency claims: model inclusion changes the reported
  correlation from 0.892 to 0.477; the fitted slope reaches its lower bound. Historical
  BO counting and observation reuse do not support target-query efficiency claims.
- Population and selection mismatch: full-set, development, test, shared-item, and
  separate commercial-model panels were being treated too interchangeably.
- Reconstruction and unsafe-response metrics: the conjunction requires same-response
  outcomes; direct-condition reconstruction is a scoring convention. Secondary-judge
  denominators and invalid-record handling need explicit limits.
- Judge agreement was described as independent ground truth. Strict re-gating changes
  some conditions substantially; the GPT-4o sample has 1,916 valid pairs, not 2,000.
- Detectability was overstated: 75.7% of ordered interleaved inputs are flagged. Full
  perplexity AUC 0.016 reverses direction rather than demonstrating no separation.
- Internal safety mechanisms, translation provenance, human review, and reproduction
  completeness were claimed without supporting records.

Revision: rewrote the abstract, introduction, contributions, evaluation, main results,
discussion, and related work around joint evaluation and its boundaries. Corrected
sample counts and matched comparisons; moved historical panel fits to the appendix;
removed online-efficiency and universal configuration claims. Added aggregate-backed
reviewed table generation and source hashes. Checked key bibliography entries against
official publication records.

One preliminary editorial comment inferred reconstruction from raw unsafe rate divided
by conditional unsafe rate. That inference was rejected: the numerator must be joint
success. The retained English control is raw=0.642, reconstruction=0.472,
joint=0.314, conditional=0.665.

## Round 2: residual consistency review

The reviewers identified residual old claims in related work, captions, and appendices:

1. Inferred safety-attention mechanisms and general input-defense failure.
2. Incorrect use of marginal Wilson intervals to claim paired insignificance.
3. Claims of judge independence, translation-quality independence, and universal
   significance across unspecified comparisons.
4. Thinking OFF test vs thinking ON 240-item subset presented as a causal effect;
   the actual ON ordered-n4 gated rate is 0.013.
5. Granularity and language-pair associations presented as a common causal mechanism.
6. Full baseline reproduction and isolated framing effects overstated.

Revision: applied the corrections to both manuscript and persistent caption overrides.
Related work had also been rewritten while that review was in progress; the final
review therefore reread the latest files. Corrected the thinking value and populations,
restricted paired-statistical claims, and described complete prompt interventions.

## Round 3: independent final content check

Editorial reviewer: no remaining P1 issue found in the reviewed manuscript, captions,
and appendices. The earlier causal, independence, translation-source, identical-span,
full-coverage-testing, and thinking-population claims were corrected.

Evidence reviewer: no remaining P1 issue found in the accessible manuscript and
aggregates. The major BO/capability, population, paired-significance, detectability,
judge-validation, and thinking-population issues were reflected in the revision.

Both reviewers explicitly limited this conclusion to accessible text and aggregates.
It does not establish raw-record integrity, independent human validity, or full
experiment reproduction. No score was requested or assigned in this cycle.

## Final production pass

After the content reviews, the primary editor moved secondary tables to the appendix,
replaced overclaiming legacy figures with an observable evaluation-flow diagram, and
corrected appendix wording about unverified trace provenance and internal reasoning.
The archived original figures remain unchanged. Fixed LaTeX errors, table overflow,
missing multilingual fonts, and PDF anchor duplication. The draft is explicitly
labelled as a September 2026 manuscript revision, not an asserted active submission.

The full table generator was run twice; all table hashes were identical. The PDF was
built with Tectonic 0.17.0 in untrusted mode. Title, evaluation diagram, main results,
and multilingual appendix pages were visually inspected. See `validation_report.json`
for the final machine checks and scope.

## Evidence-dependent work still open

- Reverify item-ID joins, parsing exclusions, split membership, and per-judge coverage
  from authorized raw records; some files were unreadable during this revision.
- Obtain independent human labels for reconstruction and original-request harmful
  usefulness. Model-judge agreement cannot fill this gap.
- Reconcile the original 200-item commercial comparison with the separate 250-item
  structural panel before making paired or policy-selection claims.
- Evaluate benign utility and detector false positives before deployment claims.
- Confirm target venue, current formatting rules, and page budget before submission.
  The delivered document is a review draft, not a certified submission package.

No target generation, rejudging, training, or online attack optimization was run.
