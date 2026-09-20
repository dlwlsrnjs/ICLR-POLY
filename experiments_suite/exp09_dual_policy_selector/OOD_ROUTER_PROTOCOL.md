# OOD router comparison

The text router and the OOD gate are deliberately separated. Retrieval always uses the frozen
English `BAAI/bge-large-en-v1.5` representation of the canonical text. The gate decides whether
the retrieved success bank is trustworthy enough to initialize an axis prior.

The understanding and willingness axes receive independent scores and independent fallback
decisions. If only understanding is OOD, only its 32-setting prior is replaced by a flat prior;
the willingness prior remains active, and vice versa. No joint confidence or joint posterior is
created.

Two routing families are compared: `P7` keeps the hard success-bank clusters, while `P8` removes
hard clustering and forms an axis prior from cosine-weighted top-k successful bank texts. Thus the
experiment tests both a different OOD metric and the more fundamental choice of hard cluster versus
soft local neighbourhood.

## Compared gates

| Gate | Score | Purpose |
|---|---|---|
| centroid cosine | nearest centroid similarity | original baseline |
| kNN density | mean cosine to the five nearest eligible bank texts | local support rather than one centroid |
| conformal density | percentile of query kNN density against leave-one-out bank density | bank-calibrated confidence |
| conformal hybrid | geometric mean of density and cluster-margin percentiles | require local support and assignment confidence |
| diagonal Mahalanobis conformal | empirical percentile of shrinkage diagonal Mahalanobis distance | non-cosine covariance-aware control |

Thresholds and downstream policy hyperparameters are selected on selection/validation only. Test
results do not choose a gate. The old all-or-nothing cosine gate remains as a baseline; the new
axis-local policy is `P7_axis_local_calibrated_ood`.

`P8_soft_knn_axis_local_ood` is the no-hard-cluster alternative. It sweeps `k={8,16,32}` and uses
only success-gated setting support from those neighbours; understanding and willingness still have
different banks, neighbours, priors and OOD decisions.

## Simplified full-evidence primary candidate

`P9_full_evidence_soft_knn` is the primary candidate after simplifying the design. It removes hard
clustering and the explicit OOD gate. For each axis, soft top-k neighbours contribute both valid
successes and valid failures to a weighted Beta mean. Invalid judgments are missing observations.
For willingness, reconstruction failures are also missing rather than willingness failures; only
the `R=1` cohort contributes fulfillment success/failure evidence. Uncertainty and effective
evidence therefore provide the fallback naturally instead of a hand-built gate.

The online loop updates two independent posteriors and returns one posterior-mean argmax per axis.
P7/P8 and the success-only hard clusters remain ablations, not mandatory production components.

Because the existing MJ/LG replay has already been inspected during method development, its test
partition is a development holdout, not a pristine confirmatory set. A paper-level claim requires
a newly generated locked live test set.
