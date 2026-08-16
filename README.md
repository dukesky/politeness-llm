# Should I Be Polite to My LLM Relevance Judge? Tone as a Severity Operating-Point Shift

Tian Zhang and Meng Li. *RecSys '26 (Reproducibility & Practice Notes)*,
Minneapolis, MN, USA. https://doi.org/10.1145/3773078.3841249

This repository is the artifact for the paper: all 15 prompt wrappers, model
configuration, the collection/analysis code, the pre-registered predictions,
and the notebook that produced every number and figure in the camera-ready.

- 📄 Camera-ready: [`paper/recsys26_notes_camera_ready.pdf`](paper/recsys26_notes_camera_ready.pdf)
- 🏷️ Code state at camera-ready: tag `recsys26-notes`

## What's in the paper (one paragraph)

Eight LLM judges score 3,498 TREC DL19/DL20 query–passage pairs (UMBRELA
0–3 rubric) under five classifier-calibrated politeness levels (L1 rude → L5
deferential), three paraphrases per level; the rubric is byte-identical
across all 15 variants. Effects are strongly model-dependent (DeepSeek V4
shows a U-shaped Δκ up to +0.054; most judges shift < 0.02). Where tone moves
agreement, the pattern matches a shift in the judge's *severity operating
point* — leniency relative to human qrels — rather than a change in
discrimination: alignment change *A(ℓ)* and Δκ are negatively associated
(query-disjoint cross-fit ρ = −0.683, exact model-block permutation p = 0.019).
Ranking metrics move far less than κ (max ΔNDCG@10 = 0.011, Kendall τ ≥ 0.743).

## Repository layout

| Path | Contents |
|---|---|
| `config/prompts.yaml` | Shared rubric + 15 tone wrappers (`L{1..5}_{a,b,c}`), with Intel `polite-guard` scores |
| `config/models.yaml` | 8 judges, OpenRouter IDs, pinned providers, decoding params, prices |
| `src/build_pairs.py` | Build the frozen pair set (BM25 top-50 ∩ qrels) from a run file |
| `src/collect.py` | Resumable OpenRouter collector; writes append-only `raw/*.jsonl` |
| `src/parse.py` | `raw/` → `derived/judgments.parquet` |
| `src/metrics.py` | Linear-weighted κ, per-level Δκ, strictness bias Δ, tonal drift D |
| `scripts/preflight_distributions.py` | Blind distribution pre-check (no κ) used before registering predictions |
| `scripts/validate_model.py` | Per-model acceptance (κ tables) |
| `scripts/make_flagship_sample.py` | Frozen 40 % stratified subsample for flagship models |
| `scripts/check_providers.py` | Verify OpenRouter model IDs / provider pins |
| `notebooks/01_collect.ipynb` | End-to-end Colab notebook: collection, parsing, all analyses, camera-ready audits, figures |
| `paper/PREDICTIONS.md` | Pre-registered per-model predictions (commit timestamps = freeze times) and unblinded outcomes |
| `paper/findings/L5a_paraphrase_case.md` | Case note on the anomalous Gemini 3.5 Flash L5_a paraphrase |
| `paper/references.bib` | Bibliography |

Data (`raw/`, `derived/`) is **not** in this repo; see below.

## Reproducing

Collection ran on Colab against OpenRouter; the notebook is the executable
record. Environment:

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...          # only needed for collection
export DATA_DIR=/path/to/data          # inputs/ raw/ derived/
```

Pipeline:

```bash
# 1. BM25 runs (pyserini, msmarco-v1-passage, dl19/dl20 topics) → $DATA_DIR/inputs/bm25_dl{19,20}.txt
python -m src.build_pairs --dataset dl19 --run-file $DATA_DIR/inputs/bm25_dl19.txt --top-k 50 --out $DATA_DIR/inputs/pairs_dl19.jsonl
python -m src.build_pairs --dataset dl20 --run-file $DATA_DIR/inputs/bm25_dl20.txt --top-k 50 --out $DATA_DIR/inputs/pairs_dl20.jsonl
python scripts/make_flagship_sample.py --data-dir $DATA_DIR

# 2. Collect (resumable; --dry-run first)
python -m src.collect --pairs $DATA_DIR/inputs/pairs_dl19.jsonl --data-dir $DATA_DIR --models <id,...>

# 3. Parse + metrics
python -m src.parse --data-dir $DATA_DIR
python scripts/validate_model.py --model <model_id> --data-dir $DATA_DIR
```

Sections 9–11 and "Camera Ready Update" of `notebooks/01_collect.ipynb`
contain the Spearman / cross-fit / permutation / bootstrap analyses and the
figure code. Cell outputs are retained so the reported numbers are visible
without re-running.

**Raw judgments.** The full `raw/` logs (~630k API responses) and
`derived/judgments.parquet` are available on request (open an issue or email
the first author) and will be deposited with the full-paper release.

## Design invariants (also enforced in `CLAUDE.md`)

- The scoring rubric block is byte-identical across all variants; only
  `wrapper_prefix` / `wrapper_suffix` change.
- Prompt texts are frozen once collection starts; changes get a new `prompt_id`.
- Experiment key: `(model_id, prompt_id, qid, docid, run)`.
- Predictions were registered in `paper/PREDICTIONS.md` *before* running any
  κ script for that model (blind protocol described there).

## Citation

```bibtex
@inproceedings{zhang2026polite,
  author    = {Tian Zhang and Meng Li},
  title     = {Should I Be Polite to My LLM Relevance Judge? Tone as a Severity Operating-Point Shift},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems (RecSys '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {New York, NY, USA},
  doi       = {10.1145/3773078.3841249}
}
```

Licensed CC BY 4.0 (paper) / MIT (code).
