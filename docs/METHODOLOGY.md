# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** charlies
**Team Members:** Karan Singh, Aditya Baghel
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
We find candidates with per-country TF-IDF search and score them with a two-level LightGBM on about 145 name and
address features. A fine-tuned multilingual transformer (multilingual-e5-large, MIT, 560M) re-judges the uncertain
pairs, and a decision layer picks, per Source-1 business, the matches that maximise expected F0.5.
Holdout macro F0.5 (India + US) is 0.99207, and the public leaderboard score is 0.98804.

---

## 2. Methodology

### 2.1 Problem Analysis
- Train: 2.21M S1, 5.03M S2, 5.29M S3. Test: 1.73M S1, 4.89M S2, 5.08M S3.
- Each S2/S3 record belongs to at most one S1, and all true pairs are within one country. 5.6% of S1 have no match;
  the mean is 3.46 matches.
- Train has only US and India. The test also has France (15%), which has no labels.
- Names alone are weak: about half of S1 names collide with another S1, and only about 65% of true pairs have
  identical normalized names.
- Name noise: typos and OCR digits, dropped or reordered words, legal-form swaps (LLC↔Ltd), junk characters,
  websites, alias forms (DBA, a/k/a), and Indic scripts in about 24% of India S2 names.
- Address noise: dropped house-number digits, state codes vs full names, and empty addresses (about 3%).
- Unmatched records are mostly look-alikes: the same name ± one word at a nearby house number. The test has about
  1.9× more of them per S1 than train.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-level GBDT classifier + transformer re-scoring + expected-F0.5 decision
**Core Innovation:** one shared rarity (IDF) table for train and test.
- *The problem:* we first computed word rarity separately for train and test. The trees then used the exact value as
  a word identifier. For example, the noise word "center" had rarity 3.622 in India train and 3.663 in India test. A
  tree split between those two values cost −5.6 in log-odds. Near-certain India pairs whose name ends in "… Limited
  Center" were accepted 98.9% of the time on the holdout but only 20.4% on test.
- *How we found it without test labels:* adversarial validation (train vs test candidate pairs, AUC 0.95, driven
  almost entirely by rarity and count features), and comparing how often near-certain pairs were accepted on the
  holdout vs test.
- *The fix:* one table of token counts over all provided files (unsupervised, no labels), with rarity capped, so a
  word has the same value everywhere. This gave most of a +0.011 leaderboard gain.

We also use a transformer only on uncertain pairs, and choose matches per business by expected F0.5.

---

## 3. Candidate Generation (Blocking)
Blocking runs per country and never compares all pairs (`src/blocking.py`, `src/prune.py`).

- **Blocking keys used** (the union of these channels; k = how many S1 are kept per record):
  - *Name:* character 3-grams of the cleaned core name (legal forms, honorifics and junk removed, OCR digits
    repaired, the real name taken from alias forms). Hashed TF-IDF, with grams that occur in over 1% of names
    dropped. Top 30 S1 per record by cosine.
  - *Address:* canonical address words (street types, directions, known typos unified). Top 25 per record, plus a
    containment variant that does not penalise a long S1 address when the record has only a fragment of it.
  - *Combined:* name and address vectors joined, so the cosine is the mean of both. One wide pass takes the top 90 S1
    per record. From it we keep the top 30 per record, and in reverse the top 30 records per S1 and source.
  - *Records without an address:* a wider name search (top 100).
  - *Exact keys:* normalized name (sorted core words), and house number + street. A key shared by more than 50 S1 is
    skipped.
- **Why it scales:** every channel is a sparse matrix product within one country that returns only the top k per
  record. The candidate set therefore grows linearly with the number of records, not with the number of pairs. The
  product runs in parallel, in blocks sized to fit the CPU cache. Blocking train and test takes about 2 hours on one
  24-thread CPU.
- **Pruning:** a LightGBM on 28 cheap signals (channel cosines and ranks, key hits, equal house number, street, city
  and state). It keeps every S1 with a score of at least 2e-5, at most 60 S1 per record and at most 150 records per
  S1. It is cross-fitted: model A is trained on folds 0–1 and model B on folds 2–3. Each fold is scored by the model
  that did not see it, so the score can also be used as a feature without leakage. This pruned set is exactly what
  the matching model scores, and it is written as `candidate_pairs.tsv`.
- **Candidate pairs generated:** 50.0M on test, about 29 per S1 (median 22, maximum 150), against 3.4 final matches
  per S1. That is 0.0003% of all S1 × record pairs.
- **How true matches were not lost:** after every change we measured recall on a test-like holdout.
  - Pruning removes 86% of the blocked pairs while losing only 0.01 points of recall (first settings, train:
    255.7M → 34.9M pairs, recall 99.17% → 99.16%).
  - The final settings keep 99.41% of true pairs. The first, narrower settings gave about 20 candidates per S1 and
    99.21% recall. Widening to 29 per S1 recovered 2,390 more true pairs (+0.0004 F0.5).
  - The per-record and per-S1 caps are single config values (`prune.top_k`, `prune.s1_cap`).
  - Most remaining misses are records without an address whose generic name matches over 50 businesses. These
    could not be resolved without an address anyway.

---

## 4. Matching Model

**Features used** (about 145, `src/features.py`):
- Name features: RapidFuzz ratios, Jaro-Winkler, token-set, Monge-Elkan, rarity-weighted token overlap, acronym
  match, legal form, transliteration and alias flags, and counts of known "noise" and "sibling" words.
- Address features: token overlap; street, city, state and postal-code equality; house-number relations (equal,
  digit dropped, range, difference).
- Other: blocking scores and ranks; how many businesses share the name or address; the pair's rank and gap to the
  best alternative candidate.

**Model type:**
- Two-level LightGBM with 5-fold GroupKFold by S1. Level 2 adds candidate context: ranks, gaps, and similarity to the
  business's other confident records.
- Transformer: `intfloat/multilingual-e5-large`, fine-tuned as a cross-encoder on training pairs only.
  - Input: "name | address" of both records, up to 128 tokens.
  - Architecture: mean pooling and a linear head.
  - Training: 1 epoch on 2.4M pairs from folds 0–2 (bf16, batch 128, lr 3e-5).
  - It re-scores pairs with a level-2 probability between 0.005 and 0.995, about 7% of pairs.
  - A second e5-large, trained the same way on 4.16M pairs, feeds the India/US combiner.
- Combiner: fitted on folds 3–4, which the transformers never saw. A LightGBM with candidate context is used for
  India and US, and a logistic combiner for France.

**Threshold selection method:** each record keeps only its most likely business, and probabilities are isotonic-
calibrated on out-of-fold predictions. For each business we sort its candidates by probability and pick the top k that
maximises expected F0.5. Assuming independent pairs, a dynamic program gives the exact distribution of true matches
inside and outside the top k. "No match" (k = 0) is also an option, scored by the probability that no candidate is
true. A global logit shift is then tuned on out-of-fold predictions. This beats the best global threshold on every
split.

**France:** France has no labels, so we set two general France settings using public-leaderboard feedback, one
change per upload. The first is a logit shift of −0.65. The second accepts same-address pairs whose record adds a
French noise word from a hand-written list (fils, frères, associés, groupe…). Nothing is tuned record by record.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro):** 0.99207 on the holdout (India + US); 0.98804 on the public leaderboard.

| Step | Holdout | Public LB |
|---|---|---|
| First pipeline: narrower blocking, pruning, two-level LightGBM, expected-F0.5 decision; no transformer | 0.98848 | 0.9757 |
| + shared rarity table | 0.99010 | – |
| + transformer (e5-base) | 0.99109 | 0.9869 |
| + wider blocking, e5-large | 0.99176 | 0.98756 |
| + India/US context combiner | 0.99196 | 0.98778 |
| + second transformer (India/US) | 0.99207 | 0.98789 |
| + French noise-word rule (final) | 0.99207 | 0.98804 |

- **Where the remaining error is (holdout):** 70% of the lost F0.5 comes from businesses with some missed matches,
  19% from businesses we predicted as having no match, and about 10% from wrong merges.
- **Common false positives (wrong merges):**
  - Look-alikes at a neighbouring house number with nearly the same name ("Winni's Homecare, 875 Quinnipiac Ave" vs
    "Winni's Homecare Group, 877 Quinnipiac Ave").
  - Unrelated trade names at the exact same address.
- **Common false negatives (missed matches):**
  - Records without an address and with a generic name. These are about 80% of misses. The model's probability
    there matches the true rate, so they are genuinely ambiguous.
  - Heavily corrupted names.
  - Non-Latin names that are missing from our transliteration dictionary.

---

## 6. Conclusion
Capped multi-channel blocking, a stacked GBDT and a multilingual transformer on uncertain pairs, joined by a
calibrated expected-F0.5 decision, reach 0.992 on a test-like holdout. The main lesson: features that depend on the
dataset, such as rarity and counts, must be computed the same way for train and test. Otherwise tree models turn them
into identifiers that do not transfer.

---

## Appendix

### A. Code Artefacts
See `code/business_entity_resolution/README.md`. The command `python src/run.py --force` regenerates
`output/matching_results.tsv` and `output/candidate_pairs.tsv` from the raw TSVs.
- Stages: dicts → splits → norm → extra → block → prune → feats → model → predict → ce.
- The transformer stages need a GPU with about 24 GB of memory.

### B. Additional Results
- **Tried and not kept:**
  - Synthetic look-alike businesses.
  - mDeBERTa and XLM-R-large (no gain over e5-large).
  - Blending two transformers (+0.00002).
  - Wider blocking than the final setting.
- **Reproducibility:**
  - A rerun on a second machine gave holdout 0.99195 (submitted run: 0.99196).
  - The final stage reproduces the submitted file byte for byte.

### C. Runtime
Measured on the final run. CPU stages ran on one 24-thread PC with 64 GB RAM. Transformer stages ran on one A100 GPU
(about 24 GB of GPU memory used).

| Stage | What it does | Time |
|---|---|---|
| dicts, splits, norm, extra | dictionaries, folds, normalization of all six files | 5 min |
| block | candidate search, train + test | 122 min (test only: 59 min) |
| prune | first-stage ranker | 35 min (test only: 16 min) |
| feats | about 145 pair features | 16 min (test only: 8 min) |
| model | two-level LightGBM, 5 folds × 2 levels | 64 min |
| tune | calibration and decision tuning | 3 min |
| predict | scoring the 50M test pairs with the 10 LightGBM models | 31 min |
| write | decision and writing both output files | 1 min |
| **CPU total** | | **4.6 h** |
| transformer 1, training | e5-large on 2.4M pairs (about 940 pairs/s) | 43 min |
| transformer 2, training | e5-large on 4.16M pairs | 76 min |
| transformer scoring | about 4,500 pairs/s; 3.4M pairs (model 1) and 5.0M pairs (model 2), train + test | 12 + 19 min |

**Test-time path with trained models:** normalizing, blocking, pruning, features and LightGBM prediction for the test
set take about 1.9 h on one CPU. Transformer scoring of the test pairs (2.3M and 3.5M pairs) takes about 21 min on
the GPU.

