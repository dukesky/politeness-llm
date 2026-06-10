# Tone Sensitivity and Cost-Effectiveness of LLM Relevance Judges

Empirical study: do surface-level tone/politeness changes in prompts shift
LLM relevance judgments and rankings — and what does each politeness level
cost? (Working title; venue targets: RecSys'26 R&P Notes -> arXiv -> ECIR'27 short.)

## Architecture

- **GitHub (this repo)**: code + prompt/model configs. Single source of truth.
- **Google Drive** (`MyDrive/llm-ranker-tone-data/`): `inputs/` (BM25 runs),
  `raw/` (append-only API logs), `derived/` (parquet for analysis).
- **Colab**: stateless executor — clones this repo, mounts Drive, runs `src/`.

## Colab session boilerplate

```python
from google.colab import drive, userdata
import os
drive.mount('/content/drive')
token = userdata.get('GH_TOKEN')
!git clone https://{token}@github.com/<YOU>/politeness-llm.git 2>/dev/null || (cd politeness-llm && git pull)
%cd politeness-llm
!pip install -q -r requirements.txt
os.environ['OPENROUTER_API_KEY'] = userdata.get('OPENROUTER_API_KEY')
os.environ['DATA_DIR'] = '/content/drive/MyDrive/llm-ranker-tone-data'
```

## Workflow

1. `python -m src.build_pairs --dataset dl19 --run-file $DATA_DIR/inputs/bm25_dl19.txt --top-k 50 --out $DATA_DIR/inputs/pairs_dl19.jsonl`
2. `python -m src.collect --pairs $DATA_DIR/inputs/pairs_dl19.jsonl --data-dir $DATA_DIR --dry-run`  (sanity check, ~$0.01)
3. Same without `--dry-run` for the full grid. Safe to interrupt/restart — resumes automatically.
4. `python -m src.parse --data-dir $DATA_DIR`
5. Analysis in `notebooks/02_analysis.ipynb` using `src/metrics.py`.

## W1 TODO (before any paid collection)

- [ ] Finalize all 15 prompt variants in `config/prompts.yaml`
- [ ] Validate politeness monotonicity with a classifier; fill `politeness_classifier_score`
- [ ] Verify OpenRouter model IDs, pinned providers, prices in `config/models.yaml`
- [ ] Download BM25 run files into Drive `inputs/`
- [ ] Dry run end-to-end; inspect raw JSONL by hand

## Rules

- `raw/` is append-only; never edit or regenerate. `derived/` is disposable.
- Experiment key: `(model_id, prompt_id, qid, docid, run)`.
- The scoring rubric block is byte-identical across all prompt variants —
  tone wrappers are the ONLY thing that changes.
- Secrets live in Colab Secrets, never in code or git.
