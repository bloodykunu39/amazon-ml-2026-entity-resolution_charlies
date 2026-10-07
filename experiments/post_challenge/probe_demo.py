"""Demo: why leaderboard probing does not transfer to a private split.

Our labelled holdout S1 are split 50/50 into a fake 'public' and 'private' leaderboard. The 'attacker' sees only the
public macro F0.5 (6 decimals) and uses score changes to infer how many probed pairs are true. Needs the cached
pipeline outputs in work/ (pred_train_bbWL3ce_e5large4x.parquet, splits, ground truth).

  python experiments/post_challenge/probe_demo.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from splits import load_splits, universe_gt  # noqa: E402

W = ROOT / "work"
KEY = ["s1", "src", "rid"]
rng = np.random.default_rng(7)
ho = load_splits().filter(~pl.col("dropped") & (pl.col("fold") < 0)).select("s1")
ho = ho.with_columns(pl.Series("public", rng.random(ho.height) < 0.5))
gt = universe_gt(False).join(ho, on="s1", how="semi").select(KEY).with_columns(pl.lit(1, pl.Int8).alias("y"))
pr = pl.read_parquet(W / "pred_train_bbWL3ce_e5large4x.parquet", columns=KEY + ["p3"]).join(ho, on="s1", how="semi")
pr = pr.join(gt, on=KEY, how="left").with_columns(pl.col("y").fill_null(0))
ntrue = gt.group_by("s1").len().rename({"len": "t"})


def score(sub: pl.DataFrame, which: bool) -> float:
    """Macro F0.5 over public (which=True) or private S1, singletons included."""
    s = sub.join(gt, on=KEY, how="left").with_columns(pl.col("y").fill_null(0)) \
        .group_by("s1").agg(pl.len().alias("k"), pl.col("y").sum().alias("tp"))
    u = ho.filter(pl.col("public") == which).join(ntrue, on="s1", how="left").join(s, on="s1", how="left").fill_null(0)
    k, tp, t = u["k"].to_numpy(), u["tp"].to_numpy(), u["t"].to_numpy()
    P = np.divide(tp, k, out=np.zeros(len(k)), where=k > 0)
    R = np.divide(tp, t, out=np.zeros(len(k)), where=t > 0)
    f = np.divide(1.25 * P * R, 0.25 * P + R, out=np.zeros(len(k)), where=(0.25 * P + R) > 0)
    f[(k == 0) & (t == 0)] = 1.0
    return round(float(f.mean()), 6)


def gain_loss(k):
    """Per-record change if the added pair is true (a) or false (b), assuming current predictions are all correct."""
    F = lambda P, R: 1.25 * P * R / (0.25 * P + R)
    a = np.where(k == 0, 1.0, 1 - F(1.0, k / np.maximum(k + 1, 1)))
    b = np.where(k == 0, 1.0, 1 - F(k / np.maximum(k + 1, 1), 1.0))
    return a, b


sub = pr.filter(pl.col("p3") >= 0.5).select(KEY)
n_pub = ho["public"].sum()
base_pub, base_priv = score(sub, True), score(sub, False)
print(f"holdout S1: {ho.height:,} (public {n_pub:,}); our 'submission' = p3 >= 0.5 -> {sub.height:,} pairs")
print(f"BASE  public LB {base_pub:.6f}   private {base_priv:.6f}\n")

# attacker: rejected but uncertain pairs (p3 0.15-0.5), one per S1, only public S1 (assume membership already probed)
k_now = sub.group_by("s1").len().rename({"len": "k"})
cand = pr.filter(pl.col("p3").is_between(0.15, 0.5, closed="left")).join(ho.filter("public"), on="s1", how="semi") \
    .sample(fraction=1.0, shuffle=True, seed=1).unique("s1", keep="first", maintain_order=True) \
    .join(k_now, on="s1", how="left").with_columns(pl.col("k").fill_null(0))

G = 400
print(f"PROBES: add a group of {G} uncertain pairs, read the public score change, infer how many are true")
print(f"{'upload':>6} {'LB change':>11} {'inferred true':>14} {'ACTUAL true':>12}")
learned = []
for g in range(6):
    grp = cand.slice(g * G, G)
    d = score(pl.concat([sub, grp.select(KEY)]), True) - base_pub
    a, b = gain_loss(grp["k"].to_numpy())
    est = (d * n_pub + b.sum()) / (a + b).mean()          # solve d*N = sum(a*y) - sum(b*(1-y))
    print(f"{g + 1:>6} {d:>+11.6f} {est:>14.0f} {int(grp['y'].sum()):>12}")
    learned.append((grp, est / G))

grp, _ = max(learned, key=lambda x: x[1])
print("\nSPLIT the best group into quarters (4 more uploads):")
for q in range(4):
    part = grp.slice(q * G // 4, G // 4)
    d = score(pl.concat([sub, part.select(KEY)]), True) - base_pub
    a, b = gain_loss(part["k"].to_numpy())
    est = (d * n_pub + b.sum()) / (a + b).mean()
    print(f"  quarter {q + 1}: LB change {d:+.6f}, inferred true {est:5.0f} / {G // 4}, actual {int(part['y'].sum())}")

# upper bound for the attacker: every probed pair's label learned (needs far more uploads than a challenge allows)
for n in (2400, len(cand)):
    tp = cand.head(n).filter(pl.col("y") == 1).select(KEY)
    s4 = pl.concat([sub, tp])
    print(f"\nENDGAME: attacker learned the true pairs among {n:,} probed public pairs -> adds {tp.height:,}")
    print(f"  public LB {base_pub:.6f} -> {score(s4, True):.6f} ({score(s4, True) - base_pub:+.6f})")
    print(f"  private   {base_priv:.6f} -> {score(s4, False):.6f} ({score(s4, False) - base_priv:+.6f})")

# compare: a GENERAL rule (threshold 0.5 -> 0.45 for all records) measured the same way
s3 = pr.filter(pl.col("p3") >= 0.45).select(KEY)
print("\nGENERAL rule instead (threshold 0.5 -> 0.45 for everyone):")
print(f"  public LB {score(s3, True) - base_pub:+.6f}   private {score(s3, False) - base_priv:+.6f}  <- moves both the same way")
