# Post-challenge experiments

Both experiments run after the challenge closed. They use the cached outputs of the final pipeline run (`work/`),
and neither changes the submission.

## 1. Does a contrastive bi-encoder help? (`biencoder.py`)

Some competitors built their pipelines on fine-tuned embedding models (bi-encoders). We tested whether such a model
adds information to ours:

- `multilingual-e5-base`, fine-tuned with a contrastive loss on training folds 0–2 only. Training used in-batch
  negatives plus one hard negative per business, batch 128, 1 epoch over 455K positive pairs, and took 88 min on an
  RTX 3050 (6 GB).
- Four new features for the final L3 combiner: the cosine, its rank among the business's candidates, the gap to the
  best candidate, and the record's best cosine to other businesses.

| Model | AUC on the uncertain pairs |
|---|---|
| Bi-encoder (e5-base) | 0.916 |
| Our cross-encoder (e5-large) | 0.977 |
| Our LightGBM (≈145 features) | 0.981 |

| | Holdout F0.5 |
|---|---|
| Final L3 combiner | 0.99207 |
| + bi-encoder features | 0.99209 (+0.00002, within noise) |

The noise level is about ±0.00006: a random feature changed the score by −0.00006.

**Result: no gain.** A bi-encoder embeds each record on its own, so it misses the small differences (house number,
an extra word) that separate a business from its look-alikes. Those decide a precision-weighted metric, and the
cross-encoder and hand-built features already capture them.

```bash
python experiments/post_challenge/biencoder.py train    # ~1.5 h on an RTX 3050
python experiments/post_challenge/biencoder.py embed
python experiments/post_challenge/biencoder.py eval
```

## 2. Why leaderboard probing does not transfer (`probe_demo.py`)

The holdout (349K businesses, with labels) is split 50/50 into a fake public and private leaderboard. An "attacker"
who only sees the public score adds groups of 400 uncertain pairs and works out from the score change how many are
true:

| Upload | Public score change | Inferred true | Actually true |
|---|---|---|---|
| 1 | −0.000349 | 102 | 109 |
| 2 | −0.000342 | 112 | 110 |
| 3 | −0.000313 | 119 | 115 |

Even if every probed label is learned (which would take far more uploads than a challenge allows):

| | Public | Private |
|---|---|---|
| Add the true pairs among 13,547 probed pairs | +0.00203 | **+0.00000** |
| A general rule (threshold 0.5 → 0.45 for everyone) | −0.00060 | −0.00065 |

Labels learned from the public score only cover public records, so they are worth nothing on the private split.
General rules move both splits the same way. That is why our France settings are general rules and not
per-record choices.

```bash
python experiments/post_challenge/probe_demo.py
```
