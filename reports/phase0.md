# Phase 0 — Environment
- Resources measured 2026-09-25: i9-12900K 16C/24T; RAM 61 GiB (49 GiB available at start, ~12 GiB used by desktop/browser); swap 8 GiB; RTX 3050 6 GB (driver 595.91.07, CUDA 13.2, ~0.75 GB used by desktop); NVMe 643 GB free (not 799). `/tmp` and `/dev/shm` are tmpfs.
- conda env `amazon` was empty: installed Python 3.11 + packages; pinned in `requirements.txt` (all MIT/BSD/Apache/ISC; anyascii ISC instead of GPL unidecode).
- TMPDIR/JOBLIB_TEMP_FOLDER/POLARS_TEMP_DIR -> `work/tmp` (set in `src/io_utils.py`).
- RAM budget lowered to 40 GB peak (less free RAM than assumed); heavy jobs run under `systemd-run --user --scope -p MemoryMax=40G`.
- Gate PASS: all 7 row counts equal the expected values exactly (polars, quoting disabled, all-string); `__MACOSX` never read. Cross-country true pairs: 0.
- Recommendation for the user: add a 32 GB swap file on the NVMe disk as an OOM safety net (not done by me; needs sudo).
