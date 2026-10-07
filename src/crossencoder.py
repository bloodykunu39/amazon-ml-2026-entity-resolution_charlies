"""Cross-encoder judge: a multilingual transformer (final config: intfloat/multilingual-e5-large, MIT, 560M) reads
the raw name + address of an S1 record and a candidate S2/S3 record and outputs a match logit. It reads text directly,
so it is immune to universe-dependent feature values (rarity, counts) and knows French/Indic words from pre-training.

Run by src/run.py (stages ce_prep, ce_train, ce); manual use:
  python src/crossencoder.py prep                      # training pairs from folds 0-2 (real records only)
  python src/crossencoder.py train                     # -> models/ce_<CE_NAME>/
  python src/crossencoder.py score PAIRS.parquet OUT.parquet [split]   # PAIRS: s1, src, rid -> adds ce_logit
Scores are used only for pairs of S1 NOT in folds 0-2 (folds 3-4, holdout, test), so they are honest there.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "6")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

WORK = ROOT / "work"
import yaml  # noqa: E402
_CE_CFG = (yaml.safe_load(open(ROOT / "config.yaml")) or {}).get("ce", {}) or {}
CE_NAME = os.environ.get("CE_NAME", _CE_CFG.get("model", "e5small"))   # e5small | e5base (both MIT)
MODEL_ID = {"e5small": "intfloat/multilingual-e5-small", "e5base": "intfloat/multilingual-e5-base",
            "e5base2x": "intfloat/multilingual-e5-base",                    # e5base2x: e5-base on 2x training pairs
            "mdeberta": "microsoft/mdeberta-v3-base",                       # mdeberta: different family (MIT)
            "e5large": "intfloat/multilingual-e5-large",                     # e5large: 560M params (MIT)
            "e5large4x": "intfloat/multilingual-e5-large",                   # e5large4x: e5-large on ~4M training pairs
            "xlmrlarge": "FacebookAI/xlm-roberta-large",                     # xlmrlarge: 560M, plain MLM pre-training (MIT)
            "e5largefr": "intfloat/multilingual-e5-large",                   # e5largefr: e5-large + French-style copies (first dictionary, flawed)
            "e5largefr2": "intfloat/multilingual-e5-large",                  # e5largefr2: same with the one-to-one dictionary
            "e5largehard": "intfloat/multilingual-e5-large",                 # e5largehard: specialist on hard pairs only (GBDT p2 0.02-0.98)
            "e5large4xs2": "intfloat/multilingual-e5-large"}[CE_NAME]        # e5large4xs2: e5large4x with another seed
OUT_DIR = ROOT / "models" / f"ce_{CE_NAME}"
MAX_LEN = 128
KEY = ["s1", "src", "rid"]
TRAIN_FOLDS = [0, 1, 2]
N_BAND = int(os.environ.get("CE_N_BAND", _CE_CFG.get("n_band", 1_000_000)))
N_EASY = int(os.environ.get("CE_N_EASY", _CE_CFG.get("n_easy", 200_000)))
SEED = int(os.environ.get("CE_SEED", 2026))
MAX_STEPS = int(os.environ.get("CE_MAX_STEPS", 0))         # smoke tests only
CKPT_EVERY = int(os.environ.get("CE_CKPT_EVERY", 0))       # >0: save a resumable checkpoint every N steps (rented GPUs)


def texts(split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Raw text of S1 and S2/S3 records (synthetic records included for train if present)."""
    s1 = pl.read_parquet(WORK / f"raw_{split}_s1.parquet", columns=["idx", "name", "addr"]).rename(
        {"idx": "s1", "name": "n1", "addr": "a1"})
    parts = []
    for s in (2, 3):
        df = pl.read_parquet(WORK / f"raw_{split}_s{s}.parquet", columns=["idx", "name", "addr"])
        syn = WORK / f"synth_train_s{s}.parquet"
        if split == "train" and syn.exists():
            df = pl.concat([df, pl.read_parquet(syn, columns=["idx", "name", "addr"])])
        parts.append(df.with_columns(pl.lit(s, pl.Int8).alias("src")))
    r = pl.concat(parts).rename({"idx": "rid", "name": "n2", "addr": "a2"})
    return s1, r


def attach_text(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    s1, r = texts(split)
    x = pairs.join(s1, on="s1", how="left").join(r, on=["src", "rid"], how="left")
    return x.with_columns((pl.col("n1") + " | " + pl.col("a1")).alias("ta"),
                          (pl.col("n2") + " | " + pl.col("a2")).alias("tb")).drop(["n1", "a1", "n2", "a2"])


# ----------------------------------------------------------------------------- data
def prep(sample: bool = False):
    import sys as _s
    _s.path.insert(0, str(ROOT / "src"))
    from io_utils import load_gt
    from splits import load_splits
    sp = load_splits().filter(pl.col("fold").is_in(TRAIN_FOLDS)).select("s1")
    gt = load_gt().with_columns(pl.lit(1, pl.Int8).alias("y"))
    parts = []
    for c in ("India", "US"):
        d = pl.read_parquet(WORK / f"cand_train{'_sample' if sample else ''}_{c}.parquet", columns=KEY + ["p0"])
        d = d.filter(pl.col("rid") < 10_000_000).join(sp, on="s1", how="semi")      # real records, folds 0-2
        parts.append(d.join(gt, on=KEY, how="left").with_columns(pl.col("y").fill_null(0)))
    d = pl.concat(parts)
    band = d.filter(pl.col("p0").is_between(0.01, 0.99))
    easy = d.filter(~pl.col("p0").is_between(0.01, 0.99))
    tr = pl.concat([band.sample(min(N_BAND, band.height), seed=SEED), easy.sample(min(N_EASY, easy.height), seed=SEED)])
    tr = attach_text(tr.select(KEY + ["y"]), "train").sample(fraction=1.0, shuffle=True, seed=SEED)
    tr.write_parquet(os.environ.get("CE_PAIRS", WORK / "ce_train_pairs.parquet"))
    print(f"ce train pairs {tr.height:,}  pos rate {tr['y'].mean():.3f}")


# ----------------------------------------------------------------------------- model
class CE(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.enc = AutoModel.from_pretrained(path)
        self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, ids, mask, tt=None):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def model_path():
    from huggingface_hub import snapshot_download
    pats = ["*.json", "*.safetensors", "sentencepiece.bpe.model", "spm.model", "tokenizer*"]
    return snapshot_download(MODEL_ID, allow_patterns=pats + (["pytorch_model.bin"] if CE_NAME == "mdeberta" else []))


def encode(tok, ta, tb):
    e = tok(ta, tb, truncation=True, max_length=MAX_LEN, padding=False)
    return e["input_ids"]


def batches_by_length(ids, bs, shuffle, rng=None):
    """Length-bucketed batches (dynamic padding)."""
    lens = np.array([len(x) for x in ids])
    order = np.argsort(lens, kind="stable")
    chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
    if shuffle:
        rng.shuffle(chunks)
    return chunks


def collate(ids, idx, pad_id, device):
    seqs = [ids[i] for i in idx]
    L = max(len(s) for s in seqs)
    a = np.full((len(seqs), L), pad_id, dtype=np.int64)
    m = np.zeros((len(seqs), L), dtype=np.int64)
    for j, s in enumerate(seqs):
        a[j, :len(s)] = s
        m[j, :len(s)] = 1
    return torch.from_numpy(a).to(device, non_blocking=True), torch.from_numpy(m).to(device, non_blocking=True)


def train(bs: int = 64, lr: float | None = None, epochs: int = 1):
    lr = lr or float(os.environ.get("CE_LR", 0)) or float(_CE_CFG.get("lr", 0)) or (4e-5 if CE_NAME == "e5small" else 2e-5 if CE_NAME == "e5large" else 3e-5)
    torch.manual_seed(SEED)
    dev = torch.device("cuda")
    path = model_path()
    tok = AutoTokenizer.from_pretrained(path)
    df = pl.read_parquet(os.environ.get("CE_PAIRS", WORK / "ce_train_pairs.parquet"))
    t0 = time.time()
    ids = encode(tok, df["ta"].to_list(), df["tb"].to_list())
    y = torch.tensor(df["y"].to_numpy(), dtype=torch.float32)
    print(f"tokenized {len(ids):,} pairs in {time.time() - t0:.0f}s, mean len {np.mean([len(x) for x in ids]):.1f}", flush=True)
    model = CE(path).to(dev)
    model.enc.embeddings.word_embeddings.weight.requires_grad_(False)          # frozen vocab (saves memory)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    steps = epochs * math.ceil(len(ids) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * max(0.0, 1 - s / steps))
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    step, t0, run, start = 0, time.time(), 0.0, 0
    ckpt = OUT_DIR / "ckpt.pt"
    if CKPT_EVERY and ckpt.exists():
        st = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start, run = st["step"], st["run"]
        print(f"resumed from step {start}", flush=True)
    for ep in range(epochs):
        for b in batches_by_length(ids, bs, True, rng):       # same seed -> same batch order, so resume skips done batches
            if step < start:
                step += 1
                continue
            a, m = collate(ids, b, tok.pad_token_id, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(a, m)
            loss = lossf(logit.float(), y[b].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step == MAX_STEPS:
                break
            if CKPT_EVERY and step % CKPT_EVERY == 0:
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "run": run}, ckpt.with_suffix(".tmp"))
                os.replace(ckpt.with_suffix(".tmp"), ckpt)
            if step % 500 == 0:
                el = time.time() - t0
                done = step - start
                print(f"step {step}/{steps} loss {run:.4f} {done * bs / el:.0f} pairs/s "
                      f"eta {(steps - step) * el / done / 60:.0f} min vram {torch.cuda.max_memory_allocated() / 1e9:.2f}GB",
                      flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "model.pt")
    (OUT_DIR / "info.txt").write_text(f"{MODEL_ID} (MIT) fine-tuned {steps} steps, bs {bs}, lr {lr}, max_len {MAX_LEN}\n")
    print("saved", OUT_DIR, f"{(time.time() - t0) / 60:.1f} min", flush=True)


@torch.no_grad()
def score(pairs_path: str, out_path: str, split: str, bs: int = int(os.environ.get("CE_SCORE_BS", 512))):
    dev = torch.device(os.environ.get("CE_DEVICE", "cuda"))          # CE_DEVICE=cpu when the GPU is busy
    if dev.type == "cpu":
        torch.set_num_threads(int(os.environ.get("CE_THREADS", "16")))
        bs = 64
    path = model_path()
    tok = AutoTokenizer.from_pretrained(path)
    model = CE(path)
    model.load_state_dict(torch.load(OUT_DIR / "model.pt", map_location="cpu"))
    model.to(dev).eval()
    pairs = pl.read_parquet(pairs_path).select(KEY).unique()
    out = np.empty(pairs.height, dtype=np.float32)
    t0 = time.time()
    CH = 400_000
    for s in range(0, pairs.height, CH):
        x = attach_text(pairs.slice(s, CH), split)
        ids = encode(tok, x["ta"].to_list(), x["tb"].to_list())
        res = np.empty(len(ids), dtype=np.float32)
        for b in batches_by_length(ids, bs, False):
            a, m = collate(ids, b, tok.pad_token_id, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                res[b] = model(a, m).float().cpu().numpy()
        out[s:s + len(ids)] = res
        print(f"scored {min(s + CH, pairs.height):,}/{pairs.height:,} {(s + len(ids)) / (time.time() - t0):.0f} pairs/s", flush=True)
    pairs.with_columns(pl.Series("ce_logit", out)).write_parquet(out_path)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(sample="--sample" in sys.argv)
    elif cmd == "train":
        train(bs=int(os.environ.get("CE_BS", _CE_CFG.get("batch", 64))))
    elif cmd == "score":
        score(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "test")
