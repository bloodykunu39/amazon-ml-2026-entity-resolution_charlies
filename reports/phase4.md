# Phase 4 — Features, level-1 OOF, level-2, decision (full train, test-like universe)

Pairs: 34.9M pruned candidates (6.0M positives). Training matrix: positives + hard negatives (p0 >= 0.02) + 15% of easy negatives (weight 1/0.15): 10.3M rows x 137 (L1) / 156 (L2) features.

## Per-fold validation logloss
| fold | L1 best_it | L1 logloss | L2 best_it | L2 logloss |
|---|---|---|---|---|
| 0 | 994 | 0.01278 | 419 | 0.01111 |
| 1 | 1117 | 0.01261 | 460 | 0.01116 |
| 2 | 1170 | 0.01314 | 433 | 0.01145 |
| 3 | 1349 | 0.01314 | 335 | 0.01158 |
| 4 | 903 | 0.01291 | 622 | 0.01126 |

Folds 0-1 vs 2-4 logloss are now similar (cross-fitted first-stage ranker removed the p0 leak seen in the first sample run).

## Macro F0.5 (test-like universe)
| split | L1 threshold best | L1 expected-F | L2 threshold best | L2 expected-F (delta=0) |
|---|---|---|---|---|
| oof | 0.98671 | 0.98659 | 0.98799 | 0.98818 |
| holdout | 0.98687 | 0.98686 | 0.98821 | 0.98849 |

Chosen on OOF: expected_f delta=0.0 (OOF 0.98818); **holdout 0.98849**.
Delta sweep (holdout, L2): -1.0: 0.98759, -0.5: 0.98822, -0.25: 0.98840, 0.0: 0.98849, 0.25: 0.98849, 0.5: 0.98839, 1.0: 0.98761

## Holdout breakdown
```
macro F0.5 = 0.98849 over 349103 S1
  by_country:
    India          n=  139846  f=0.98703
    US             n=  209257  f=0.98946
  by_size:
    0              n=   19535  f=0.98853
    1              n=   18916  f=0.95905
    2              n=   59269  f=0.98708
    3+             n=  251383  f=0.99103
  by_country_size:
    India,0        n=    7919  f=0.98649
    India,1        n=    7515  f=0.95420
    India,2        n=   23674  f=0.98552
    India,3+       n=  100738  f=0.98988
    US,0           n=   11616  f=0.98993
    US,1           n=   11401  f=0.96225
    US,2           n=   35595  f=0.98812
    US,3+          n=  150645  f=0.99181
  pair_by_src:
    S2: P=0.99796 R=0.97142 tp=567809 pred=568967 true=584517
    S3: P=0.99779 R=0.96936 tp=604517 pred=605857 true=623626
  errors: {'fp_distractor': 2448, 'fp_wrong_s1': 50, 'fn_total': 35817, 'fn_blocking_miss': 10264, 'fn_model_miss': 25553, 'cand_pair_recall': 0.9915043169558571}
```

## Top-30 level-1 features (gain share)
p0 0.569, g_p0_gap_rec 0.305, nt_unsh_r 0.013, house_logdiff 0.011, g_sc_fz_gap_rec 0.009, fn_fold_ratio 0.007, num_cont_l 0.005, legal_eq 0.004, num_cont_r 0.004, name_len_ratio 0.004, r_has_legal 0.004, fn_norm_tset 0.003, at_unsh_r 0.003, nt_nlow 0.003, r_ntok 0.002, num_jac 0.002, nt_me_r 0.002, nt_unsh_l 0.002, r_vocab_frac 0.002, g_fa_tset_gap_rec 0.002, fa_tset 0.002, fh_lev 0.002, g_fa_tset_gap_s1 0.001, house_affix 0.001, sc_fz 0.001, g_p0_sum_s1 0.001, fn_jw 0.001, r_cnt_key_rec 0.001, at_cont_r 0.001, num_nr 0.001

## Top-15 level-2 features (gain share)
p1 0.811, c_gap_rec 0.128, p0 0.042, c_n05_rec 0.003, c_gap_s1 0.003, co_name_max 0.002, g_p0_gap_rec 0.001, c_sum_rec 0.001, co_n 0.001, rec_gap 0.001, co_house_any 0.001, c_rk_rec 0.000, c_sum_s1 0.000, co_addr_max 0.000, co_both_max 0.000
