"""Country-specific decision shift on top of a combined (p3) version, e.g. France only.
Rebuilds the version's calibration exactly as ce_stack.py (isotonic on folds 3-4 of p3, delta 0), then adds a logit
shift for one country and reports that country's matched-record share; optionally writes the variant.
  python src/country_shift.py TAG COUNTRY d1,d2,... [--write d]
"""
import shutil
import sys

import numpy as np
import polars as pl

from decision import fit_isotonic, partition, select_expected_f, shift
from io_utils import ROOT, WORK, load_source, write_id_lists_df

tag, country, deltas = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(",")]
write_d = float(sys.argv[sys.argv.index("--write") + 1]) if "--write" in sys.argv else None
tr = pl.read_parquet(WORK / f"pred_train_{tag}.parquet", columns=["fold", "y", "p3"]).filter(pl.col("fold").is_in([3, 4]))
iso = fit_isotonic(tr["p3"].to_numpy(), tr["y"].to_numpy())
te = pl.read_parquet(WORK / f"pred_test_{tag}.parquet", columns=["s1", "src", "rid", "country", "p3"])
d = partition(te, "p3")
pc = iso.predict(d["p3"].to_numpy())
isc = (d["country"] == country).to_numpy()
nrec = te.filter(pl.col("country") == country).select(["src", "rid"]).n_unique()
for dl in deltas + ([write_d] if write_d is not None and write_d not in deltas else []):
    p = shift(pc, np.where(isc, dl, 0.0)).astype(np.float32)
    sel = select_expected_f(d.with_columns(pl.Series("_p", p)), "_p")
    n = sel.join(d.filter(pl.col("country") == country).select(["s1", "src", "rid"]), on=["s1", "src", "rid"], how="semi").height
    print(f"{country} delta {dl:+.2f}: matched pairs {n:,}  matched-record share {n / nrec:.4f}")
    if write_d is not None and dl == write_d:
        out = ROOT / "output" / "variants" / f"{tag}_{country[:2].lower()}{dl:+.2f}"
        out.mkdir(parents=True, exist_ok=True)
        s1 = load_source("test", 1).select(["idx", "entity_id"])
        rec = pl.concat([load_source("test", s).select(pl.lit(s, pl.Int8).alias("src"), pl.col("idx").alias("rid"),
                                                        pl.col("entity_id").alias("rec_entity_id")) for s in (2, 3)])
        m = sel.join(s1.rename({"idx": "s1", "entity_id": "s1_entity_id"}), on="s1").join(rec, on=["src", "rid"]) \
            .select(["s1_entity_id", "rec_entity_id"])
        write_id_lists_df(out / "matching_results.tsv", ("source1_entity_id", "matched_entity_ids"), s1, m)
        shutil.copy(ROOT / "output" / "variants" / tag / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
        print("wrote", out)
