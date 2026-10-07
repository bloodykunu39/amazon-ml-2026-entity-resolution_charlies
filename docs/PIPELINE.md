# Business Entity Resolution: Amazon ML Challenge 2026 (team charlies)

Matches every Source-1 business to its Source-2/Source-3 records. Each S2/S3 record goes to at most one S1.

**Pipeline:**
1. Multi-view normalization.
2. Wide multi-channel blocking.
3. Cross-fitted first-stage ranker.
4. About 145 pairwise features.
5. Two-level LightGBM (5-fold GroupKFold by S1).
6. Transformer judge (**multilingual-e5-large**, MIT) re-scores the uncertain pairs.
7. Combiner: logistic, or a LightGBM context combiner for the train countries.
8. Partition + calibrated expected-F0.5 decision.

## Reproduce (one command)

```bash
conda create -n amazon -y python=3.11 pip && conda activate amazon
pip install -r requirements.txt          # includes torch (CUDA 12.8 wheels) and transformers
# data: copy the provided student_resource/ folder (dataset/ + utils/) into this folder, i.e.
#   code/business_entity_resolution/student_resource/dataset/{train,test}/*.tsv  and  .../student_resource/utils/
#   (paths are set in config.yaml -> paths.data_train / paths.data_test)
python src/run.py --force                # raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv
python src/validate_outputs.py           # local checks + official validator
```

**Hardware.**
- **GPU:** the transformer stages need an NVIDIA GPU with **about 24 GB or more**, because e5-large is fine-tuned at
  batch 128. We used an A100 80 GB: about 45 min to train on 2.4M pairs (first model) and 75 min on 4.16M pairs (second model), about 25 min to score. The pretrained model
  `intfloat/multilingual-e5-large` (MIT licence, 560M parameters) is downloaded from the Hugging Face hub on first use
  into `models/hf/`.
- **CPU:** everything else runs on CPU. The GBDT part takes about 4.6 h on a 24-thread / 64 GB PC (blocking about 2 h,
  pruning 35 min, features 15 min, LightGBM 64 min, prediction 31 min).

**Other commands.**
- `python src/run.py --from-stage X` reruns stage X and everything after it (cached parquet in `work/`).
- Stages: dicts → splits → norm → extra → synth → block → prune → feats → model → tune → predict → write →
  ce_prep → ce_train → ce.
- `write` produces the GBDT-only submission; `ce` overwrites `output/` with the final one.
- Decision settings live in `config.yaml` (`decision.country_delta`, `ce.l3_countries`). They only need
  `python src/run.py --from-stage ce` (minutes, uses cached transformer scores).
- Tests: `python -m pytest tests/`.
- Reproducibility (checked on a second machine from this folder): a full rerun gives holdout macro F0.5 0.99195 vs
  0.99196 for the submitted run. Multi-threaded LightGBM makes about 0.2% of India/US and 1% of France output rows
  differ from run to run.

## Stages (src/)

| Stage | Module | What |
|---|---|---|
| dicts | mine_dicts.py | Mined from the training pairs: state variants (codes, native script), transliteration tokens, address token synonyms, noise/sibling words. S1 state names and the S1 name vocabulary: unsupervised counts over S1 records. French regions and département → region: hand-written general knowledge (France is not in train). No test labels are used anywhere. |
| splits | splits.py | Fixed-seed test-like universe (21% of train S1 dropped), 20% holdout, 5 GroupKFold folds |
| norm | normalize.py / textnorm.py | Name views (core, legal form, alias/DBA split, domain, phonetic, acronym, transliteration, OCR repair) and address views (house number, street, street type, city, state, unit, postal code) |
| extra | mine_dicts.py | Noise vs sibling extra-word lists (train pairs only) |
| synth | synth.py | Optional synthetic sibling businesses (disabled in the final config) |
| block | blocking.py | Per country: TF-IDF top-k on name char 3-grams (30), address tokens (25), combined view in both directions (30/30), wide name search (90), empty-address name search (100), exact name / house+street keys (cap 50). Records of dropped S1 are excluded from the train universe. |
| prune | prune.py | Cross-fitted LightGBM first-stage ranker → top-60 per record, ≤150 per S1 → final candidate set (`candidate_pairs.tsv`) |
| feats | features.py | RapidFuzz similarities, rarity-weighted overlaps from ONE shared rarity table (a word has the same value in train and test), Monge-Elkan, house-number relations, acronym match, noise/sibling word counts, ambiguity counts, group context |
| model / tune / predict / write | model.py, submit.py, decision.py | Level-1 + level-2 LightGBM, isotonic calibration, expected-F0.5 subset selection, logit shift δ |
| ce_prep / ce_train | crossencoder.py | Cross-encoder (multilingual-e5-large, mean pooling + linear head) fine-tuned on 2.4M pairs from folds 0–2 only (1 epoch, batch 128, lr 3e-5, bf16) |
| ce | ce_stack.py, l3_stack.py | The transformer scores pairs with 0.005 ≤ p2 ≤ 0.995. Logistic combiner [logit p2, transformer logit, product] fitted on folds 3–4. For India/US, a LightGBM combiner adds candidate context (best competing S1 of the record by GBDT and transformer, ranks, confident records per source, empty address); it is cross-fitted on folds 3/4. France keeps the logistic combiner with a France-only logit shift of −0.65, plus the France rule `decision.fr_sibling_accept` (same-address pairs whose record adds a French noise word such as fils / associés / groupe are accepted from probability 0.3). Final decision: partition → isotonic → expected-F0.5 subset. |
| ce (2nd model) | crossencoder.py, ce_stack.py | Config `ce_l3`: a second e5-large trained on 4.16M pairs from folds 0–2 (same settings); it scores 0.001 ≤ p2 ≤ 0.999 and feeds only the India/US context combiner. France keeps the first model with the logistic combiner. |

## Validation design

- The test set has about 1.9× more distractors per S1 than train.
- We drop 21% of train S1 entities from the universe, together with their records (records of missing businesses
  do not exist in the test).
- We keep a 20% holdout and 5 GroupKFold folds on the rest (`work/splits.parquet`, fixed seed).
- Versions are compared on the same holdout with `src/fair_eval.py`, also at test look-alike density with `DENSE=1`.
- The transformer only sees pairs of folds 0–2 in training. Its scores on folds 3–4, the holdout and test are
  out-of-sample.
- Holdout macro F0.5 (India + US) of the final recipe: **0.99207**.
- France has no labels. Its decision settings were chosen with public-leaderboard feedback and apply to all France
  rows as one rule.

## Hardware used

- Intel i9-12900K (16C/24T), 61 GiB RAM, RTX 3050 6 GB for the CPU stages.
- A100 80 GB (Lightning AI) for the transformer.

## Licences

- Model: intfloat/multilingual-e5-large, MIT (560M parameters).
- Libraries: polars, pyarrow (MIT/Apache-2.0), numpy, scipy, scikit-learn, pandas, joblib, psutil (BSD),
  LightGBM (MIT), RapidFuzz (MIT), sparse_dot_topn (Apache-2.0), anyascii (ISC), indic-transliteration (MIT),
  jellyfish (MIT), PyYAML (MIT), PyTorch (BSD-3), transformers / tokenizers / safetensors / huggingface_hub (Apache-2.0).
- No external data; only the provided files are used.
