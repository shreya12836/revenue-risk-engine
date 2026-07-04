# ML Pipeline

This document currently covers known limitations surfaced during Day 4
modeling. Problem framing, data flow, and leakage prevention are documented
inline in `src/features/` and `README.md` for now; a full write-up is
deferred to the documentation pass in `docs/roadmap.md`.

## Limitations

**Feature-to-sample ratio.** The feature set (33 columns) is large
relative to the training population (~2,600 customers at the train
snapshot) — roughly a 1:79 ratio. Linear models tolerate this better than
tree ensembles, which is one contributing factor in the baseline
outperforming XGBoost on the current validation split (see README's
Modeling section). Future work: feature selection via RFE or permutation
importance, or dimensionality reduction, once more training snapshots or
a larger dataset are available.
