# Business Entity Resolution: Amazon ML Challenge 2026 (team charlies)

Our solution to the Amazon ML Challenge 2026 entity-resolution task: for every business record in Source 1, find
the records in Sources 2 and 3 that describe the same real-world business. The data covers India, the US and France,
with about 11.7M test records. Scoring is macro F0.5 per Source-1 business, which weights precision over recall.

| | Macro F0.5 |
|---|---|
| Public leaderboard (final submission) | **0.988035** |
| Own holdout (India + US, 20% of training businesses) | **0.99207** |

## How it works

```
raw records ─► normalise ─► blocking (TF-IDF top-k, 6 channels) ─► pruning (LightGBM ranker, ~29 candidates per S1)
            ─► ~145 pair features ─► 2-level LightGBM ─► e5-large cross-encoder on uncertain pairs
            ─► combiner ─► calibrated expected-F0.5 decision per business ─► matching_results.tsv
```

1. **Normalisation:** names split into core / legal form / alias / acronym views, Indic scripts transliterated, OCR
   fixes. Addresses split into house number, street, city, state and postal code.
2. **Blocking:** per-country sparse TF-IDF searches over name 3-grams, address words and a combined view, plus exact
   keys. It keeps 99.4% of true pairs at about 29 candidates per Source-1 record.
3. **Two-level LightGBM** on about 145 features: fuzzy name similarity, rarity-weighted overlap, house-number
   relations, noise vs "sibling" words, and candidate context.
4. **Cross-encoder:** `intfloat/multilingual-e5-large` (MIT), fine-tuned on training pairs only. It re-scores the
   roughly 7% of pairs the trees are unsure about.
5. **Decision:** isotonic calibration, then for each business the subset of candidates that maximises *expected*
   F0.5, computed exactly with a dynamic program. "No match" is also an option.

**The biggest single lesson (+0.011 on the leaderboard):** compute dataset-dependent features (word rarity, counts)
from **one shared table for train and test**. Separate tables shifted the values slightly, and the trees used the
exact values as word identifiers. Near-certain matches were then rejected on test (99% → 20% acceptance for one
pattern). The details are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Repository

| Path | Contents |
|---|---|
| [src/](src/) | The full pipeline (`run.py` runs every stage) |
| [config.yaml](config.yaml) | All settings of the final submission |
| [tests/](tests/) | Metric, validator and text-normalisation tests |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Methodology write-up submitted with the solution (blocking, features, models, error analysis, runtime) |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Stage-by-stage reference and reproduction notes |
| [docs/SOLUTION_SUMMARY.md](docs/SOLUTION_SUMMARY.md) | Two-page summary |
| [docs/SUBMISSIONS.md](docs/SUBMISSIONS.md) | Every leaderboard upload, what changed and its score |
| [experiments/post_challenge/](experiments/post_challenge/) | Experiments after the challenge: a bi-encoder test and a leaderboard-probing demo |

## Reproduce

The competition data is **not included**. It belongs to the organisers. Place the provided `student_resource/`
folder (with `dataset/` and `utils/`) in the repository root, then:

```bash
conda create -n amazon -y python=3.11 pip && conda activate amazon
pip install -r requirements.txt
python src/run.py --force          # raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv
python src/validate_outputs.py     # local checks + the official validator
python -m pytest tests/
```

Hardware: the CPU stages take about 4.6 h on a 24-thread, 64 GB PC. The transformer stages need a GPU with about
24 GB (we used an A100). See [docs/PIPELINE.md](docs/PIPELINE.md) for per-stage commands and timings.

## France: an unlabelled country

Training data covers only India and the US, but 15% of the test set is French. All France-specific choices are two
general settings in `config.yaml`, checked against the public leaderboard one change per upload:
- a logit shift of −0.65
- a rule that accepts same-address pairs whose record adds a French noise word (*fils, frères, associés, groupe*)

Nothing is tuned record by record. [experiments/post_challenge/](experiments/post_challenge/) shows, on our own
labelled holdout, why general rules carry over from a public to a private split while pair-level leaderboard
probing does not.

## Team

charlies: [Karan Singh](https://github.com/bloodykunu39), [Aditya Baghel](https://github.com/Adibaghel232).

Only the provided data was used: no external data, APIs or test labels. Model: multilingual-e5-large (MIT, 560M
parameters).
