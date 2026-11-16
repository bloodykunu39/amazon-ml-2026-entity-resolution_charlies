# Phase 3 — Blocking + first-stage ranker (sample results; full-train numbers appended when available)

## Channels (per country, `src/blocking.py`)
- name: char 3-grams of the compact core name (domain if no core), hashed TF-IDF (sublinear tf, idf, max_df 1%), rec->S1 top-10
- addr: canonical address word tokens + glued alphanumeric ids (#b126, #c6), max_df 2%, rec->S1 top-10
- addrc: same tokens, S1 vectors divided by sqrt(norm) (containment-style, less penalty for long S1 addresses), top-10
- comb: [name, addr]/sqrt(2), one wide pass rec->S1 top-40; derived rec->S1 top-10 and S1->rec top-10 per source (reverse channel)
- nwide: records with no usable address: name top-50
- exact keys: normalized name key, (house, street core) — block cap 20
- Speed: target S1 matrix split into 131K-column blocks so each thread's accumulator fits in L2 (4.5x faster on full India).

## Sample (25% of S1, test-like universe) raw-union recall by iteration
| version | pair recall | India | US | cands/rec |
|---|---|---|---|---|
| name+addr+comb+rev+keys | 0.99379 | 0.99125 | 0.99549 | 22.2 |
| + id tokens, addrc, empty-addr wide name | 0.99633 | 0.99457 | 0.99750 | 25.0 |

Channel recall@10: name 0.816, addr 0.919, addrc 0.921, comb 0.990, rev 0.972; unique contributions: nwide 0.18%, comb 0.07%, addrc 0.014%, name 0.011%.
Missed pairs are dominated by empty R addresses with ambiguous names (~60%), non-Latin names with partial India addresses, and random trade names with partial addresses.

## First-stage ranker (`src/prune.py`)
LightGBM on 28 cheap features (channel cosines/ranks, key hits, hashed equality of house/street/city/state/name key/address, group context of cos_comb). Top gains: rec_gap, eq_a_house, r_comb, s1_gap, eq_a_state.
Keep per record all S1 with p0 >= 5e-5 (top-50), cap 100 per S1.

**Pruned sample: pair recall 0.99627 (India 0.99447, US 0.99747), 3.2 candidates/record, 17.9 per S1 (max 100). Gate (>= 99.5%) PASS on sample.**

## Full train (test-like universe, 1.74M S1 × 10.3M records)
- Blocking time (24 threads, with concurrent jobs): India 22.5 min, US 34.6 min. Peak RSS 5.2 GB.
- Raw union: 255.7M pairs (24.8/record, 147/S1 mean). **Pair recall 0.99166** (India 0.98704, US 0.99475) — below the 99.5% gate.
- Missed pairs: US 19.0K (16.2K empty R address — name-only records whose true S1 is not in the name top-50, i.e. highly ambiguous names), India 31.4K (11.7K with address, 11.4K empty address, 6.5K non-Latin names, 1.7K domains).
- Gate decision: proceed to a first submission with this blocking (most misses are empty-address, ambiguous-name records the matcher cannot resolve anyway); larger top-k per channel is the next blocking iteration.
