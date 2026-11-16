# Phase 1 — I/O, metric, validators, splits
- `src/io_utils.py`, `src/metrics.py` (exact macro F0.5, closed form 1.25·tp/(0.25·n_true+n_pred), empty/empty=1), `src/validate_outputs.py` (local checks + official validator), `src/splits.py`.
- `work/splits.parquet` (written once, seed 2026): 21% of train S1 dropped (463,296) -> distractor share of S2/S3 = **41.5%** (target 41–44%; original 26%). Holdout 349,103 S1 (20% of kept); 5 folds of ~279K S1 each; 25% iteration sample (records: matched ones follow their S1, 25% of unmatched).
- tests/test_phase1.py: 5/5 pass (metric 0.7143 example + edge cases, validator error detection, GT facts 7,638,365 pairs / partition / 5.58% singletons, splits).
