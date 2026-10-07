# Submission history: team charlies

Every leaderboard upload in order, with the output folder, what changed and the public score.

*Note: the `File` paths below point to folders of our original working directory. The outputs are not part of this repository.*
Checksum = first 12 hex characters of the MD5 of `matching_results.tsv`.

| # | Day | File | What changed | Public F0.5 | Checksum |
|---|---|---|---|---|---|
| 1 | Sat 26 | `output/v1/` | First full pipeline: blocking, pruning, 2-level LightGBM, expected-F0.5 decision | 0.975675 | 5eee5e1e387b |
| 2 | Sat 26 | `output/variants/diag_france_empty/` | Diagnostic: v1 with all France rows empty, to split the score by country | 0.840948 | 195a39b979c8 |
| 3 | Sat 26 | `output/variants/v3ace_e5base/` | Shared rarity table + records of dropped S1 removed + e5-base transformer judge | 0.986907 | 6ceaa31d33d6 |
| 4 | Sat 26 | `output/variants/v3ace_e5base_fr+1.50/` | France logit shift +1.5 (probe) | 0.986065 | 42c9cdb44938 |
| 5 | Sat 26 | `output/variants/v3ace_e5base_fr-1.50/` | France logit shift −1.5 (probe) | 0.986991 | 53316b9b5945 |
| 6 | Sun 27 | `upload/1_e5large/` | Wider blocking + e5-large transformer; France shift matched to probe 5 | 0.987559 | 06457bb318ad |
| 7 | Sun 27 | `upload/4_final_bbWL3_nosib/` | Context combiner on all countries + French sibling-word filter | 0.986914 | aa365c624d4f |
| 8 | Sun 27 | `upload/2_e5large_nosib/` | #6 minus France pairs whose record adds a French sibling word (filter test) | 0.987093 | b88351ed68b7 |
| 9 | Sun 27 | `upload/5_hybrid_safe/` | Context combiner for India/US only; France identical to #6 | 0.987775 | e4c52feef380 |
| 10 | Sun 27 | `upload/7_max_frm065/` | + second transformer (e5-large on 4.16M pairs, wider band) for India/US | 0.987891 | c7fa6c1ae561 |
| 11 | Sun 27 | `upload/8_max_frp010/` | #10 with France shift +0.10 (probe) | 0.987845 | f5cc950ef371 |
| 12 | Sun 27 | `upload/10_fr_sibaccept/` | **#10 + France rule: accept same-address pairs whose record adds a French noise word (p ≥ 0.3)** | **0.988035** | **1d4f6fce9ea4** |
| 13 | Sun 27 | `upload/13_big_gamble/` | #12 + the rule from p ≥ 0.1, 8 more words, and any same-address name change (probe) | 0.987345 | cbf927f06b62 |

A teammate's independent pipeline also scored 0.98342 on the shared quota (not part of this code).

**Final solution = #12 (0.988035).** `charlies_submission.zip` contains exactly this file (checksum 1d4f6fce9ea4). The code in
the zip reproduces it: the final stage run with `src/run.py --from-stage ce` gives a byte-identical file.
