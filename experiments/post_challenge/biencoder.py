"""Post-challenge experiment: a contrastive bi-encoder (as teamx_1/teamx_2 used) as an extra signal for the L3 combiner.

A multilingual-e5-base is fine-tuned on folds 0-2 only (positives of work/ce_train_pairs.parquet, in-batch negatives
plus one hard negative of the same S1), so its cosine is honest on folds 3-4 and the holdout. The cosine and its
candidate-context ranks are added to the L3 features; holdout F0.5 is compared with the same L3 without them.
Does not touch src/ or config.yaml; writes only models/bienc_<NAME>/ and work/bienc_*.

  python experiments/post_challenge/biencoder.py train          # ~1 h on an RTX 3050 (6 GB)
  python experiments/post_challenge/biencoder.py embed          # cosines of the train band pairs -> work/bienc_cos_train_<NAME>.parquet
  python experiments/post_challenge/biencoder.py eval           # L3 with vs without the bi-encoder features (holdout F0.5)
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("CE_LO", "0.001")                 # band of the final L3 (config ce_l3 lo/hi)
os.environ.setdefault("CE_HI", "0.999")
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

WORK = ROOT / "work"
NAME = os.environ.get("BE_NAME", "e5base")
MODEL_ID = {"e5base": "intfloat/multilingual-e5-base", "e5small": "intfloat/multilingual-e5-small"}[NAME]
OUT = ROOT / "models" / f"bienc_{NAME}"
IN_TAG = os.environ.get("BE_IN_TAG", "bbWce_e5large4x")   # input of the final L3 (holdout 0.99207 after L3)
MAX_LEN = 64
BS = int(os.environ.get("BE_BS", 64))
LR = float(os.environ.get("BE_LR", 3e-5))
TAU = float(os.environ.get("BE_TAU", 0.05))
SEED = 2026
KEY = ["s1", "src", "rid"]


def model_path():
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.safetensors", "sentencepiece.bpe.model", "tokenizer*"])


def embed_batch(enc, ids, mask):
    h = enc(input_ids=ids, attention_mask=mask).last_hidden_state
    m = mask.unsqueeze(-1).to(h.dtype)
    return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)


def tokenize(tok, texts):
    return tok(["query: " + t for t in texts], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")


# ----------------------------------------------------------------------------- train
def train():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = torch.device("cuda")
    d = pl.read_parquet(WORK / "ce_train_pairs.parquet")        # folds 0-2 only (crossencoder.prep)
    pos = d.filter(pl.col("y") == 1).select(["s1", "ta", "tb"])
    neg = d.filter(pl.col("y") == 0).group_by("s1").agg(pl.col("tb").first().alias("hn"))
    pos = pos.join(neg, on="s1", how="left").sample(fraction=1.0, shuffle=True, seed=SEED)
    print(f"positives {pos.height:,}, with a hard negative {pos['hn'].is_not_null().sum():,}", flush=True)
    path = model_path()
    tok = AutoTokenizer.from_pretrained(path)
    enc = AutoModel.from_pretrained(path).to(dev)
    enc.embeddings.word_embeddings.weight.requires_grad_(False)       # frozen vocab (fits 6 GB)
    enc.gradient_checkpointing_enable()
    params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    steps = int(os.environ.get("BE_MAX_STEPS", 0)) or pos.height // BS
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 300) * max(0.0, 1 - s / steps))
    s1, ta, tb, hn = pos["s1"].to_numpy(), pos["ta"].to_list(), pos["tb"].to_list(), pos["hn"].to_list()
    enc.train()
    t0, run = time.time(), 0.0
    for step in range(steps):
        b = slice(step * BS, (step + 1) * BS)
        q_t, d_t = ta[b], tb[b]
        h_t = [x for x in hn[b] if x is not None]
        qs = s1[b]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eq, ed = tokenize(tok, q_t), tokenize(tok, d_t + h_t)
            q = embed_batch(enc, eq["input_ids"].to(dev), eq["attention_mask"].to(dev))
            docs = embed_batch(enc, ed["input_ids"].to(dev), ed["attention_mask"].to(dev))
        sim = (q.float() @ docs.float().T) / TAU                              # [B, B + H]
        n = len(q_t)
        same = torch.from_numpy(qs[:, None] == qs[None, :]).to(dev)          # other positives of the same S1
        same &= ~torch.eye(n, dtype=torch.bool, device=dev)
        mask = torch.zeros_like(sim, dtype=torch.bool)
        mask[:, :n] = same
        sim = sim.masked_fill(mask, -1e4)
        loss = F.cross_entropy(sim, torch.arange(n, device=dev))
        loss = 0.5 * loss + 0.5 * F.cross_entropy(sim[:, :n].T, torch.arange(n, device=dev))   # symmetric
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if (step + 1) % 200 == 0:
            el = time.time() - t0
            print(f"step {step + 1}/{steps} loss {run:.4f} {(step + 1) / el:.2f} it/s eta {(steps - step - 1) * el / (step + 1) / 60:.0f} min "
                  f"vram {torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    enc.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print("saved", OUT, f"{(time.time() - t0) / 60:.1f} min", flush=True)


# ----------------------------------------------------------------------------- embed
@torch.no_grad()
def embed():
    from crossencoder import texts
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(OUT)
    enc = AutoModel.from_pretrained(OUT).to(dev).eval().half()
    lo, hi = float(os.environ["CE_LO"]), float(os.environ["CE_HI"])
    band = pl.read_parquet(WORK / f"pred_train_{IN_TAG}.parquet", columns=KEY + ["p2", "ce_logit", "fold"]) \
        .filter(pl.col("p2").is_between(lo, hi) & pl.col("ce_logit").is_not_null()).select(KEY)
    s1, r = texts("train")
    s1 = s1.join(band.select("s1").unique(), on="s1").with_columns((pl.col("n1") + " | " + pl.col("a1").fill_null("")).alias("t"))
    r = r.join(band.select(["src", "rid"]).unique(), on=["src", "rid"]).with_columns((pl.col("n2") + " | " + pl.col("a2").fill_null("")).alias("t"))

    def run(tx):
        out = np.empty((len(tx), enc.config.hidden_size), dtype=np.float16)
        order = np.argsort([len(t) for t in tx])
        t0 = time.time()
        for i in range(0, len(tx), 512):
            idx = order[i:i + 512]
            e = tokenize(tok, [tx[j] for j in idx])
            out[idx] = embed_batch(enc, e["input_ids"].to(dev), e["attention_mask"].to(dev)).cpu().numpy()
            if (i // 512) % 200 == 0:
                print(f"  {i + len(idx):,}/{len(tx):,} {(i + len(idx)) / (time.time() - t0):.0f} texts/s", flush=True)
        return out

    e1, er = run(s1["t"].to_list()), run(r["t"].to_list())
    i1 = dict(zip(s1["s1"].to_list(), range(s1.height)))
    ir = dict(zip(zip(r["src"].to_list(), r["rid"].to_list()), range(r.height)))
    a = np.array([i1[x] for x in band["s1"].to_list()])
    b = np.array([ir[k] for k in zip(band["src"].to_list(), band["rid"].to_list())])
    cos = np.einsum("ij,ij->i", e1[a].astype(np.float32), er[b].astype(np.float32))
    band.with_columns(pl.Series("emb_cos", cos.astype(np.float32))).write_parquet(WORK / f"bienc_cos_train_{NAME}.parquet")
    print("wrote", WORK / f"bienc_cos_train_{NAME}.parquet", band.height, flush=True)


# ----------------------------------------------------------------------------- eval
EMB_FEATS = ["emb_cos", "emb_rank_s1src", "emb_gap_s1src", "rec_max_other_emb"]


def emb_context(d: pl.DataFrame) -> pl.DataFrame:
    c = pl.read_parquet(WORK / f"bienc_cos_train_{NAME}.parquet")
    d = d.join(c, on=KEY, how="left")
    return d.with_columns(
        pl.col("emb_cos").rank("ordinal", descending=True).over(["s1", "src"]).cast(pl.Float32).alias("emb_rank_s1src"),
        (pl.col("emb_cos") - pl.col("emb_cos").max().over(["s1", "src"])).alias("emb_gap_s1src"),
        pl.col("emb_cos").max().over(["src", "rid"]).alias("_m"),
        pl.col("emb_cos").sort(descending=True, nulls_last=True).slice(1, 1).first().over(["src", "rid"]).alias("_m2"),
    ).with_columns(pl.when(pl.col("emb_cos") == pl.col("_m")).then(pl.col("_m2")).otherwise(pl.col("_m"))
                   .alias("rec_max_other_emb")).drop(["_m", "_m2"])


def holdout(tr: pl.DataFrame, col: str) -> dict:
    from ce_stack import best_delta
    from decision import evaluate, fit_isotonic
    from splits import load_splits, universe_gt
    sp_ = load_splits().filter(~pl.col("dropped"))
    gt = universe_gt(False)
    f34 = tr.filter(pl.col("fold").is_in([3, 4]))
    iso = fit_isotonic(f34[col].to_numpy(), f34["y"].to_numpy())
    u34 = sp_.filter(pl.col("fold").is_in([3, 4])).select("s1")
    dl, sc34 = best_delta(tr.join(u34, on="s1", how="semi"), gt.join(u34, on="s1", how="semi"), u34, col, iso)
    rep = {"folds34": round(sc34, 5)}
    for c in ("India", "US", None):
        uh = sp_.filter((pl.col("fold") < 0) & ((pl.col("country") == c) if c else pl.lit(True))).select("s1")
        rep[f"holdout_{c or 'all'}"] = round(evaluate(tr.join(uh, on="s1", how="semi"), gt.join(uh, on="s1", how="semi"), uh,
                                                      col, iso=iso, deltas=(dl,), thresholds=())["expected_f"][dl], 5)
    return rep


def evaluate_l3():
    import lightgbm as lgb
    import l3_stack as L
    tr = L.context(pl.read_parquet(WORK / f"pred_train_{IN_TAG}.parquet"), WORK, "train")
    tr = emb_context(tr)
    band = tr.select(pl.col("p2").is_between(L.LO, L.HI) & pl.col("ce_logit").is_not_null()).to_series().to_numpy()
    bt = tr.filter(band)
    print(f"band {bt.height:,}, cosine missing {bt['emb_cos'].is_null().sum():,}", flush=True)
    yb = bt["y"].to_numpy()
    for f in (3, 4, -1):
        s = bt.filter(pl.col("fold") == f)
        from sklearn.metrics import roc_auc_score
        print(f"fold {f}: AUC emb_cos {roc_auc_score(s['y'], s['emb_cos']):.4f}  ce {roc_auc_score(s['y'], s['ce_logit']):.4f}  "
              f"p2 {roc_auc_score(s['y'], s['p2']):.4f}", flush=True)
    res = {}
    for name, feats in (("L3 (final features)", L.FEATS), ("L3 + bi-encoder", L.FEATS + EMB_FEATS)):
        models = {}
        for f in (3, 4):
            fit = bt.filter(pl.col("fold") == f)
            models[f] = lgb.train(L.PARAMS, lgb.Dataset(fit.select(feats).to_numpy(), fit["y"].to_numpy()), L.ROUNDS)
        X = bt.select(feats).to_numpy()
        fold = bt["fold"].to_numpy()
        pa, pb = models[3].predict(X), models[4].predict(X)
        p = np.where(fold == 3, pb, np.where(fold == 4, pa, (pa + pb) / 2))
        p3 = tr["p2"].to_numpy().astype(np.float64).copy()
        p3[band] = p
        res[name] = holdout(tr.select(["s1", "src", "rid", "y", "fold", "country"]).with_columns(pl.Series("p3", p3.astype(np.float32))), "p3")
        imp = dict(zip(feats, models[3].feature_importance("gain")))
        tot = sum(imp.values())
        print(name, res[name], {k: round(v / tot, 3) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:8]}, flush=True)
    base, new = res["L3 (final features)"]["holdout_all"], res["L3 + bi-encoder"]["holdout_all"]
    print(f"\nholdout F0.5: {base:.5f} -> {new:.5f}  ({new - base:+.5f})")


if __name__ == "__main__":
    {"train": train, "embed": embed, "eval": evaluate_l3}[sys.argv[1]]()
