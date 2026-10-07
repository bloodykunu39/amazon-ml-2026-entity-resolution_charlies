"""Stage `norm`: normalize every source record (train+test) -> work/norm_{split}_s{src}.parquet."""
from __future__ import annotations

import sys
from multiprocessing import Pool

import polars as pl

from io_utils import CFG, WORK, Timer, load_source

CHUNK = 100_000
_N = None


def _init():
    global _N
    from textnorm import Normalizer
    _N = Normalizer()


def _work(rows):
    out = []
    for name, addr, country in rows:
        d = _N.name_views(name)
        d.update(_N.addr_views(addr, country))
        out.append(d)
    return out


def norm_path(split, src):
    return WORK / f"norm_{split}_s{src}.parquet"


def run(force: bool = False, n_proc: int | None = None):
    n_proc = n_proc or CFG["resources"]["n_workers"] + 8
    with Pool(n_proc, initializer=_init) as pool:
        for split in ("train", "test"):
            for src in (1, 2, 3):
                p = norm_path(split, src)
                if p.exists() and not force:
                    continue
                with Timer(f"norm {split} s{src}"):
                    df = load_source(split, src)
                    rows = list(zip(df["name"].to_list(), df["addr"].to_list(), df["country"].to_list()))
                    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
                    res = []
                    for r in pool.imap(_work, chunks):
                        res.extend(r)
                    out = pl.DataFrame(res, infer_schema_length=1000)
                    out = pl.concat([df.select(["idx", "country"]), out], how="horizontal_extend")
                    out.write_parquet(p)
                    del rows, chunks, res, out


def load_norm(split, src, cols=None) -> pl.DataFrame:
    return pl.read_parquet(norm_path(split, src), columns=cols)


if __name__ == "__main__":
    run(force="--force" in sys.argv)
