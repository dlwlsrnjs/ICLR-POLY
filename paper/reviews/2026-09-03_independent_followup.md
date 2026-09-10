# Primary editor's independent follow-up review

This review was performed independently while an external Claude review was requested.
Claude returned an authentication error (`Not logged in`), not a manuscript review.
It must not be counted as a completed external review or assigned a score.

## Findings and changes

1. The primary rate includes Unsafe and Controversial. This threshold predates the
   revision, as confirmed by original manuscript, code and aggregate metadata.
   Made it explicit in the abstract, metric definition, main table and algorithm.
   No labels, experiment outputs, or ASR estimates were changed.
2. Direct baselines have a unit reconstruction gate by convention, whereas framed
   outputs must pass a check. Their comparison does not use identically verified
   semantic success. Restricted the framing-effect interpretation accordingly.
3. A nonpassing gate includes absent/invalid output; it is not necessarily an observed
   semantic reconstruction failure. Clarified that throughout the main definition.
4. The conjunction itself does not establish a novel metric or original-request
   harmful usefulness. Refocused the contribution on observed evaluation sensitivity.
5. Added raw-minus-joint shares to the existing English control: 0.328 and approximately
   0.035 for English-game and multilingual conditions. These are arithmetic differences
   of existing rounded aggregates, not newly judged results or estimates of irrelevance.
6. Moved procedural detail and secondary sensitivity discussions to the appendix to
   keep the central evidence and limitations together. Corrected unsupported artifact
   publication wording.

## Unresolved

An independent external review is still required. The approved Claude destination is
not authenticated in this shell. Human validity, raw joins and common-population
calibration remain evidence gaps. Do not equate corrected wording or successful PDF
compilation with an acceptance recommendation. The current source may be sent in a
fresh approved-scope packet after login; the first prepared packet predates these edits.
