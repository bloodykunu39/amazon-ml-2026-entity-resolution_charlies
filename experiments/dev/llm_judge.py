"""Zero-shot LLM judge (experimental): Qwen2.5-7B-Instruct (Apache-2.0, 7.6B) in 4-bit reads two business records and
answers Yes/No; score = logit(Yes) - logit(No) for the first answer token. Fits a 6 GB GPU: 4-bit decoder layers on GPU,
embeddings on CPU, only the two answer rows of the output layer are used.
  python src/llm_judge.py PAIRS.parquet OUT.parquet SPLIT      # PAIRS: s1, src, rid -> adds llm
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import polars as pl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(ROOT / "src"))
from crossencoder import texts  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
KEY = ["s1", "src", "rid"]
SYS = "You are an expert in business entity resolution."
ASK = ("Do these two records describe the same business entity at the same location? Records may contain typos, "
       "abbreviations, reordered address parts, missing fields and legal-form variations (Inc, Ltd, SARL, SAS, Pvt). "
       "A different branch, a sister company or a different street address is NOT the same entity.\n"
       "Record A: {a}\nRecord B: {b}\nAnswer Yes or No.")


def load():
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"])
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True, llm_int8_enable_fp32_cpu_offload=True)
    dm = {"model.embed_tokens": "cpu", "model.layers": 0, "model.norm": 0, "model.rotary_emb": 0, "lm_head": "cpu"}
    m = AutoModelForCausalLM.from_pretrained(path, quantization_config=q, device_map=dm, dtype=torch.bfloat16)
    m.eval()
    yes, no = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
    import json
    from safetensors import safe_open
    shard = json.load(open(Path(path) / "model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
    with safe_open(str(Path(path) / shard), framework="pt") as f:          # CPU-offloaded head sits on meta: read rows
        w = f.get_tensor("lm_head.weight")[[yes, no]]
    w = w.to("cuda", torch.float32)
    return tok, m, w


def prompts(tok, ta, tb):
    return [tok.apply_chat_template([{"role": "system", "content": SYS},
                                     {"role": "user", "content": ASK.format(a=a, b=b)}],
                                    tokenize=False, add_generation_prompt=True) for a, b in zip(ta, tb)]


@torch.no_grad()
def judge(tok, m, w, ta, tb, bs=8):
    ps = prompts(tok, ta, tb)
    order = np.argsort([len(p) for p in ps])
    out = np.empty(len(ps), dtype=np.float32)
    for i in range(0, len(ps), bs):
        idx = order[i:i + bs]
        e = tok([ps[j] for j in idx], return_tensors="pt", padding=True)
        emb = m.model.embed_tokens(e["input_ids"]).to("cuda")
        h = m.model(inputs_embeds=emb, attention_mask=e["attention_mask"].to("cuda")).last_hidden_state[:, -1].float()
        lg = h @ w.T
        out[idx] = (lg[:, 0] - lg[:, 1]).cpu().numpy()
    return out


def rec(n, a):
    return f"name: {n} | address: {a}"


if __name__ == "__main__":
    pairs_path, out_path, split = sys.argv[1:4]
    s1, r = texts(split)
    x = pl.read_parquet(pairs_path).select(KEY).join(s1, on="s1", how="left").join(r, on=["src", "rid"], how="left")
    tok, m, w = load()
    t0 = time.time()
    CH = 2000
    res = []
    for s in range(0, x.height, CH):
        c = x.slice(s, CH)
        sc = judge(tok, m, w, [rec(n, a) for n, a in zip(c["n1"], c["a1"])], [rec(n, a) for n, a in zip(c["n2"], c["a2"])])
        res.append(c.select(KEY).with_columns(pl.Series("llm", sc)))
        pl.concat(res).write_parquet(out_path)
        print(f"llm {min(s + CH, x.height):,}/{x.height:,} {(s + c.height) / (time.time() - t0):.1f} pairs/s", flush=True)
