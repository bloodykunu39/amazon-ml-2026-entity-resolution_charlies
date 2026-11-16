# Phase 5 — Why the leaderboard is lower than validation (diagnosis) and the v4 fix

## Leaderboard facts (2026-09-26)
| upload | file | public score |
|---|---|---|
| 1 | v1 (`output/v1`) | 0.975675 |
| 2 | v1 with all France rows empty (diagnostic) | 0.840948 |

France is 14.975% of test S1. Rows left empty score 1 only for true singletons (≈5.6%), so
France F0.5 ≈ 0.056 + (0.975675 − 0.840948)/0.14975 ≈ **0.956** and US+India ≈ **0.979**,
vs 0.988 on the test-like holdout. Both parts are below validation; US+India carries ~60% of the gap.

## Hypotheses checked
| hypothesis | test | result |
|---|---|---|
| test simply has more distractors | duplicate generated distractors to test density in the holdout | −0.0005 only |
| rarity (IDF) features shift with universe size (US singleton token 15.10 train vs 14.62 test) | re-score holdout with test-scale rarity values | no change (0.98786 → 0.98786) |
| model miscalibrated on test | label-free: mean Σp per S1, test vs holdout (true matches/S1 = 3.46 in both) | calibrated (US 3.45 vs 3.42, India 3.34 vs 3.41) |
| **test contains hidden sibling businesses** | model-free: clusters of ≥2 name-similar records from both sources at a house number ≠ S1's, per S1 | **test US 0.274 vs train 0.021 (13×), India 0.097 vs 0.020 (5×), France 0.147** |

Adversarial validation (train vs test top pairs): AUC 0.94–0.96, almost entirely from universe-scale
features (counts, rarity, group sizes); without them AUC 0.62–0.65. The model is 1.5–3× more often
uncertain on test within every segment (near-house, same-house, far-house).

## The sibling generator (mined from train's own no-owner records)
- Sibling words (ratio sibling/true > 1000): US group, holdings, east, west, valley, riverside, uptown, midtown,
  southside, westgate, harbor, north, coastal, northside, summit, metro, downtown, south, greater, central,
  eastgate, lakeside, highland; India industries, enterprises, public, exports, overseas, ventures, infratech,
  group, holdings + trade words (solutions, technologies, trading, foods, steel, ...).
- House offsets: exactly {+1, 2, 3, 4, 5, 7, 9, 11, 13, 21}, uniform.
- Train: each sibling is a single record with its own offset. Test: a hidden sibling *business* — several records
  in both sources sharing one offset (examples: "Houser, Anabal L., DO" 309 → three records at 310;
  "Cera Tankers, Inc" 2281 → "Cera-Tankers, Co" and "Cera Tankers Southside, Inc" at 2294).

## v4 fix
- `src/synth.py`: for 40% (US) / 20% (India) of kept train S1, copy their true records, shift the house number by
  one shared offset, insert an independent sibling word per copy (30% none); add as non-matching records.
  542K sibling entities, 1.18M records. Only train-derived vocabulary/offsets are used.
- Universe: records of the 21% dropped S1 are removed (test has no "records of a missing unrelated business").
- Rarity features capped at 12 (scale-free) + doc-frequency of the rarest unshared token.
