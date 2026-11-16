# Mined dictionaries

Built by `src/mine_dicts.py` (pipeline stage `dicts`) from the training data. These are the exact files used for the final submission.

| File | Contents |
|---|---|
| `name_vocab.json` | business-name token vocabulary |
| `translit.json` | transliteration map |
| `addr_syn.json` | address synonyms |
| `name_extra.json` | noise and sibling words |
| `state.json` | state-name normalization |
| `fr_region.json` | French regions |

To use them without re-mining, copy them to `work/dicts/` and start the pipeline after the `dicts` stage:

```bash
mkdir -p work/dicts && cp dicts/*.json work/dicts/
python src/run.py --from-stage splits
```
