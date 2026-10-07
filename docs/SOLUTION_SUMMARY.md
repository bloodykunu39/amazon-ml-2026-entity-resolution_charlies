# Business Entity Resolution: team charlies (Amazon ML Challenge 2026)

**Team:** Karan Singh, Aditya Baghel · **Public leaderboard:** 0.98804 macro F0.5 · **Holdout (India + US):** 0.99207

## 1. Approach
The task is to find, for every Source-1 business, its records in Sources 2 and 3. Each record belongs to at most one
business, and the metric is precision-weighted (F0.5 per S1, macro-averaged). Our solution is a reproducible 11-step
pipeline (`python src/run.py --force`):

1. **Dictionaries from the training data:** Indic → Latin transliterations, state codes and names in native scripts,
   address typos, and "noise" vs "sibling" words (words added to true matches vs to look-alike businesses).
2. **Normalisation:** lower case, accents and scripts folded, legal forms separated (LLC, Pvt Ltd, SARL), OCR fixes,
   acronym, and address split into house number, street, city and state.
3. **Blocking:** per country, TF-IDF top-k over name 3-grams, address words and a combined view in both directions,
   plus a wide name search for records without an address and exact keys. It keeps 99.4% of true pairs.
4. **Pruning** with a cross-fitted LightGBM on cheap scores: about 4 candidates per record.
5. **About 145 pair features:** fuzzy similarities (Levenshtein, Jaro-Winkler, token-set, TF-IDF cosine, Monge-Elkan),
   one shared rarity table for train and test, house-number relations, acronym match, noise/sibling word counts,
   ambiguity and group context.
6. **Two-level LightGBM** (5-fold GroupKFold by S1). Level 2 adds candidate context (ranks, gaps, co-reference).
7. **Transformer judges:** `intfloat/multilingual-e5-large` (MIT, 560M) fine-tuned as cross-encoders on training pairs
   only. It re-scores uncertain pairs.
8. **Combiners:** a logistic blend for France, and a cross-fitted LightGBM context combiner for India/US (the best
   competing company of the record, confident matches per source, empty address).
9. **Decision:** each record goes to at most one company; calibrated probabilities; per company, the subset that
   maximises expected F0.5, where "no match" is also an option.

## 2. Key experiments (holdout = 20% of training companies; public LB in brackets)

| Step | Holdout F0.5 |
|---|---|
| First pipeline | 0.9885 (0.9757) |
| One shared rarity table, as described in finding 1 below | 0.9901 |
| + e5-base transformer judge | 0.9911 (0.9869) |
| + wider blocking, + e5-large | 0.99176 (0.98756) |
| + context combiner for India/US | 0.99196 (0.98778) |
| + second e5-large on 4.16M pairs for India/US | 0.99207 (0.98789) |
| + France noise-word rule | – (**0.98804**) |

**Tried and not kept:** mDeBERTa and XLM-R-large (no gain over e5-large), combining transformers or seeds (+0.00002),
synthetic look-alike businesses, wider blocking beyond the final setting (worse), accent and cross-source features
(noise), and a zero-shot 7B LLM judge (AUC 0.55).

## 3. Main findings
1. **Features that depend on the dataset must be defined the same way for train and test.** Rarity values computed
   per dataset shifted between train and test and acted as word identifiers inside the trees. On test, near-certain
   India pairs with a noise suffix were accepted only 20% of the time (99% on the holdout). One shared table fixed it
   (+0.011 on the leaderboard).
2. **Where the remaining error is:** only 745 false merges on the holdout against about 29K missed pairs. About 80% of
   the misses are records without an address whose name is shared by several companies. The model's probabilities are
   well calibrated there, so these cases are close to the limit of what the data can tell.
3. **France (15% of test) has no training labels.** Every France-only rule was checked on the public leaderboard, one
   change per upload. A shift toward fewer matches helped. Applying the US/India-trained combiner to France hurt. The
   main discovery: French words such as *fils, frères, associés, groupe, (France)* are harmless noise in this data,
   while their English equivalents (sons, brothers, associates, group) mark look-alikes in training. Accepting
   same-address French pairs whose record adds such a word gained +0.00014. The rule is hand-written general knowledge
   in the code (`decision.fr_sibling_accept`).

## 4. Reproducibility and rules
- Only the provided files are used: no external data, APIs or geocoders, and no test labels or pseudo-labels.
  TF-IDF and rarity counts over all records are unsupervised.
- Models: multilingual-e5-large (MIT, 560M parameters). Libraries are permissively licensed.
- Checked on separate machines: a fresh-environment rerun of the GBDT part (holdout 0.99195 vs 0.99196), transformer
  retraining from the config (identical scores), and the final stage reproducing the submitted file byte for byte.
- Hardware: 16-core CPU, 64 GB RAM. The transformer stages need a GPU with about 24 GB (we used an A100).

## 5. Conclusion
Careful normalisation, wide blocking, a stacked GBDT and fine-tuned multilingual cross-encoders, combined through a
calibrated expected-F0.5 decision, reach 0.992 on a test-like holdout for the two labelled countries. The remaining gap
is the unlabelled country (France). There, general rules checked on the leaderboard, rather than per-record tuning,
gave safe gains.
