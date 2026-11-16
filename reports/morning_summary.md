# Morning summary — updated 2026-09-26 12:25 IST

## >>> Current recommendation for upload #3: `output/variants/v3ace_e5base/matching_results.tsv` <<<
(v3a = root-cause fix only, run from raw data on PC3 + the bigger transformer judge multilingual-e5-base, MIT, 278M).
Official validator incl. `--check-ids`: PASS.

| version | fair holdout | test-density holdout | India | US |
|---|---|---|---|---|
| v1 (uploaded, LB 0.9757) | 0.98848 | 0.98809 | 0.98702 | 0.98946 |
| v4 (synthetic siblings) | 0.98989 | 0.98942 | 0.98895 | 0.99052 |
| v3a (fix only) | 0.99010 | 0.98965 | 0.98914 | 0.99074 |
| v4ce | 0.99082 | 0.99050 | 0.99044 | 0.99107 |
| v4avgce (2 seeds + transformer) | 0.99088 | 0.99055 | 0.99051 | 0.99113 |
| v3 (look-alike weighting, PC2) | 0.99015 | 0.98975 | 0.98919 | 0.99079 |
| v3ace (small transformer) | 0.99098 | 0.99068 | 0.99058 | 0.99125 |
| v3ce | 0.99099 | 0.99075 | 0.99061 | 0.99125 |
| **v3ace_e5base (bigger transformer)** | **0.99109** | **0.99082** | **0.99067** | **0.99137** |
| v3ace2 (both transformers) | 0.99111 | 0.99083 | 0.99071 | 0.99137 |

v3ce and v3ace are equivalent within noise (differ in <0.5% of US/India rows, 2.2% of France rows); v3ace is kept as
the recommendation because its pipeline is simpler (no weighting step). Both ran end-to-end from raw data on the remote PCs.
Blending v3a with v4 did not help (0.99011 vs 0.99010) and was dropped.

(fair = same holdout S1, real records only, same calibration for all; test-density = generated look-alikes
duplicated to the measured test density. Script: `src/fair_eval.py`, `DENSE=1` for the second column.)
On test, all fixed versions accept India "… Center" near-certain pairs ~95.6% vs 13% in v1.
v3a also proved the one-command reproducibility run: `src/run.py --force` from raw data, 198 min, validator PASS.

### France — the one open decision (no French labels)
The fixed versions are more conservative in France than v1 (v3ace: 843K matched France pairs vs v1 ~863K; 21% of France
rows differ). Likely cause: with the shared rarity table French words count as globally rare, so an unmatched French word
looks like strong "different business" evidence. Whether that is right cannot be measured locally.
- Upload #3 = v3ace as is (France model view).
- Ready alternative: `output/variants/v3ace_fr+1.50/` (France-only logit shift +1.5 → back to v1's France volume,
  US/India unchanged). Validator: PASS. Decide for the final upload after seeing #3's score.

e5-base vs e5-small: +0.0001 (fair) / +0.00014 (test-density); using both adds nothing (+0.00002) → one model.
e5-base is also less conservative in France (+8.5K / −3.7K France pairs vs v3ace).

---
## Final clean run (started 12:40 on this PC, ETA ~19:00)
The final recipe (v3a settings + e5-base transformer, all in `src/run.py`, 15 stages) is running from raw data with
no caches in `~/Documents/final_run/amazon_final/` — exactly what the zip will contain. Its output (not the
current v3ace_e5base file) should become the FINAL upload + zip, so code and outputs match. The France setting can be
changed afterwards in `config.yaml` (`decision.country_delta`) with `python src/run.py --from-stage ce` (minutes).
Environment: the conda env `amazon_ce` now holds everything (same pinned versions + torch/transformers);
requirements.txt updated. README rewritten.

(earlier notes below)


## (09:20 note, superseded by the top section) v4ce was the first candidate
(v4 + transformer judge). Official validator: PASS (incl. `--check-ids` on v4; v4ce uses v4's candidate file,
its matches ⊆ candidates is checked by the local validator: PASS).

## Why
| | v1 (uploaded, LB 0.9757) | v4 | **v4ce** |
|---|---|---|---|
| Holdout macro F0.5 | 0.98849 | 0.98969 | **0.99050** |
| — India | 0.98703 | 0.98865 | **0.98997** |
| — US | 0.98946 | 0.99038 | **0.99085** |
| Test: India "… Center" near-certain pairs accepted | 20.4% | 98.9% | ≈99% |
| Test: uncertain records India / US / France | 4.4% / 3.0% / 6.3% | 2.0% / 1.9% / 5.9% | **1.4% / 1.6% / 4.9%** |
| Test: matches per S1 India / US / France | 3.26 / 3.40 / 3.33 | 3.36 / 3.39 / 3.23 | 3.36 / 3.38 / 3.21 |

- **Root cause fixed:** the model had used the exact rarity value of noise words ("center", "services") as word IDs;
  values differed between train and test. v4 uses one shared rarity table → the India symptom is gone
  (20% → 99% acceptance) and test uncertainty fell to train levels for US/India.
- **Transformer judge** (multilingual-e5-small, MIT, 118M params, fine-tuned 29 min on this PC's GPU on folds 0–2 only)
  re-scores the uncertain pairs; a logistic combiner fitted on folds 3–4 adds +0.0011 (on v2) and +0.0008 (on v4)
  on the holdout, and reads raw text, so it cannot be fooled by universe-dependent feature values.
- Expected leaderboard (my estimate): ~0.980–0.985 (v1 was 0.9757). India should gain most.

## Fair comparison (same holdout, real records only, same calibration procedure — src/fair_eval.py)
| version | holdout | India | US |
|---|---|---|---|
| v1 | 0.98848 | 0.98702 | 0.98946 |
| v2 | 0.98860 | 0.98736 | 0.98943 |
| v2 + transformer | 0.98974 | 0.98930 | 0.99003 |
| v4 | 0.98989 | 0.98895 | 0.99052 |
| **v4ce** | **0.99082** | **0.99044** | **0.99107** |

## In progress (started 09:25)
- Bigger transformer judge (multilingual-e5-base, MIT, 278M) training on this PC's GPU, ETA ~10:40, then scoring.
  If it beats e5-small on the holdout it replaces it (or both are combined).

## Risks / open questions
- **France** (15% of test, no labels): v4 is more conservative in France (3.33 → 3.21 matches per S1, 21% of France rows
  changed vs v1). Test uncertainty in France is still highest (4.9%). The upload will tell us whether this helped.
- v4's synthetic siblings are copies of true records (possible shortcut). v3 / v3a (no synthetic siblings) are running
  on the remote PCs (done ~12:00) as the comparison.

## Still running / next
- PC2 (v3) and PC3 (v3a): full runs, ~12:00. I will pull the results over SSH, add the transformer to them, and compare
  all versions on the same holdout (synthetic records excluded so the comparison is fair).
- Uploads left: 2. Suggested use: #3 = v3ace (see top); #4 = the final file, decided after #3 (France setting, e5-base).
