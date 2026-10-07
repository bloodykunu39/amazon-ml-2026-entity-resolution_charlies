"""End-to-end orchestration.

  python src/run.py                      # full pipeline from raw data (uses cached stages)
  python src/run.py --from-stage feats   # rerun feats and everything after
  python src/run.py --force              # ignore every cache (clean run)
  python src/run.py --sample             # train-side stages on the 25% iteration sample (no test output)

Stages: dicts -> splits -> norm -> extra -> synth -> block -> prune -> feats -> model -> tune -> predict -> write
        -> ce_prep -> ce_train -> ce   (transformer judge, config `ce`; needs a CUDA GPU; writes the final output/)
Every stage logs wall time + peak RSS to reports/run_report.json.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import ROOT  # noqa: E402

STAGES = ["dicts", "splits", "norm", "extra", "synth", "block", "prune", "feats", "model", "tune", "predict", "write",
          "ce_prep", "ce_train", "ce"]
REPORT = ROOT / "reports" / "run_report.json"


class RSSMonitor(threading.Thread):
    def __init__(self, interval=30):
        super().__init__(daemon=True)
        self.interval, self.peak, self.stop = interval, 0.0, False

    def run(self):
        p = psutil.Process()
        while not self.stop:
            rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
            self.peak = max(self.peak, rss / 1e9)
            avail = psutil.virtual_memory().available / 1e9
            if avail < 8:
                print(f"[mem] WARNING available RAM {avail:.1f} GB", flush=True)
            time.sleep(self.interval)


def stage(name, fn, report):
    mon = RSSMonitor(10)
    mon.start()
    t = time.time()
    print(f"==== stage {name}", flush=True)
    fn()
    mon.stop = True
    peak_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    report[name] = {"seconds": round(time.time() - t, 1), "peak_rss_gb": round(max(mon.peak, peak_self), 2)}
    print(f"==== {name}: {report[name]}", flush=True)
    REPORT.write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-stage", default="dicts", choices=STAGES)
    ap.add_argument("--to-stage", default="ce", choices=STAGES)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sel = STAGES[STAGES.index(a.from_stage): STAGES.index(a.to_stage) + 1]
    force = a.force or a.from_stage != "dicts"
    for d in ("reports", "logs", "models", "output", "work/tmp", "work/dicts"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}

    import blocking
    import features
    import mine_dicts
    import model
    import normalize
    import prune
    import submit

    def run_dicts():
        if a.force or not (ROOT / "work/dicts/state.json").exists():
            import runpy
            runpy.run_path(str(Path(mine_dicts.__file__)), run_name="__main__")

    def run_splits():
        import splits
        if not splits.SPLITS.exists():          # deterministic (fixed seed); never regenerated once written
            splits.make_splits()

    def run_extra():
        p = ROOT / "work/dicts/name_extra.json"
        if a.force or not p.exists():
            mine_dicts.mine_name_extra()

    def run_synth():
        import synth
        from io_utils import CFG, WORK
        if not CFG.get("synth", {}).get("enabled", False):
            return
        existed = all((WORK / f"synth_train_s{s}.parquet").exists() for s in (2, 3))
        synth.run(force=a.force)
        if a.force or not existed:
            for s in (2, 3):                    # train S2/S3 now include the synthetic records: re-normalize them
                normalize.norm_path("train", s).unlink(missing_ok=True)
            normalize.run(force=False)

    import os
    import subprocess
    from io_utils import CFG as _CFG, WORK as _W
    ce_on = bool((_CFG.get("ce", {}) or {}).get("enabled", False))

    def ce_cmd(*args, extra_env=None):
        subprocess.run([sys.executable, str(ROOT / "src" / "crossencoder.py"), *args], check=True,
                       env={**os.environ, **(extra_env or {})})

    l3m = (_CFG.get("ce_l3", {}) or {}) if ce_on else {}     # optional 2nd transformer, used only by the context combiner
    l3_env = {"CE_NAME": str(l3m.get("model", "")), "CE_PAIRS": str(_W / "ce_train_pairs_l3.parquet"),
              "CE_N_BAND": str(l3m.get("n_band", "")), "CE_N_EASY": str(l3m.get("n_easy", "")),
              "CE_BS": str(l3m.get("batch", "")), "CE_LR": str(l3m.get("lr", "")),
              "CE_LO": str(l3m.get("lo", "")), "CE_HI": str(l3m.get("hi", ""))} if l3m else {}

    def run_ce_prep():
        if ce_on:
            ce_cmd("prep", *(["--sample"] if a.sample else []))
        if l3m:
            ce_cmd("prep", *(["--sample"] if a.sample else []), extra_env=l3_env)

    def run_ce_train():
        if ce_on:
            ce_cmd("train", extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
            import ce_stack
            ce_stack.clear_cache()
        if l3m:
            ce_cmd("train", extra_env={**l3_env, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
            for split in ("train", "test"):
                (_W / f"ce_scores_{split}_{l3m['model']}.parquet").unlink(missing_ok=True)

    def run_ce():
        if ce_on:
            import ce_stack
            ce_stack.main("final", str(model.pred_path(2, "train", a.sample)),
                          None if a.sample else str(model.pred_path(2, "test", False)),
                          base_out=str(ROOT / "output"), out_dir=str(ROOT / "output"), sample=a.sample)
            if (_CFG.get("ce", {}) or {}).get("l3", False) and not a.sample:
                tag, sfx = f"finalce{ce_stack.SFX}", ce_stack.SFX
                if l3m:                          # 2nd transformer: its own band + scores, stacked in a separate process
                    subprocess.run([sys.executable, str(ROOT / "src" / "ce_stack.py"), "finall3",
                                    str(model.pred_path(2, "train", False)), str(model.pred_path(2, "test", False)),
                                    str(ROOT / "output")], check=True, env={**os.environ, **l3_env})
                    tag, sfx = f"finall3ce_{l3m['model']}", f"_{l3m['model']}"
                    os.environ["CE_LO"], os.environ["CE_HI"] = l3_env["CE_LO"], l3_env["CE_HI"]
                import l3_stack                  # LightGBM combiner with candidate context -> overwrites output/ rows
                l3_stack.main(tag, f"finalL3ce{sfx}")    # of the countries in ce.l3_countries
                assert l3_stack.finalize(f"finalL3ce{sfx}", str(ROOT / "output"), str(ROOT / "output"))

    def run_prune():
        prune.clear_chunk_cache()
        m = prune.train_rankers(a.sample) if (force or not prune.MODEL_B.exists()) else prune.load_rankers()
        prune.apply_prune("train", a.sample, m)
        if not a.sample:
            prune.apply_prune("test", False, m)
        prune.clear_chunk_cache()

    actions = {
        "dicts": run_dicts,
        "splits": run_splits,
        "norm": lambda: normalize.run(force=a.force),
        "extra": run_extra,
        "synth": run_synth,
        "block": lambda: [blocking.run("train", a.sample, force=force)] + ([] if a.sample else
                                                                           [blocking.run("test", False, force=force)]),
        "prune": run_prune,
        "feats": lambda: [features.run("train", a.sample, force=force)] + ([] if a.sample else
                                                                           [features.run("test", False, force=force)]),
        "model": lambda: model.run_train(a.sample),
        "tune": lambda: submit.tune(a.sample),
        "predict": lambda: None if a.sample else model.run_predict("test", sample_models=False),
        "write": lambda: None if a.sample else submit.write(),
        "ce_prep": run_ce_prep,
        "ce_train": run_ce_train,
        "ce": run_ce,
    }
    for s in sel:
        stage(s, actions[s], report)


if __name__ == "__main__":
    main()
