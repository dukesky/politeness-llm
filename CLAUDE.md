# Project: Tone Sensitivity of LLM Relevance Judges
Research codebase. Data collection via OpenRouter API runs on Colab;
this local repo is the single source of truth for code and configs.

## Hard rules — never violate
- config/prompts.yaml: the `rubric` block must stay byte-identical across
  all variants. Only wrapper_prefix/wrapper_suffix may differ.
- Once data collection has started, NEVER edit an existing variant's text.
  Wording changes require a NEW prompt_id (e.g. L1_a_v2).
- Experiment key is (model_id, prompt_id, qid, docid, run). Never change
  this schema without explicit instruction.
- raw/ data on Google Drive is append-only; this repo never contains data
  files (*.jsonl, *.parquet are gitignored — keep it that way).
- Never commit secrets. API keys live in Colab Secrets only.
- After any code change: run `python -m py_compile src/*.py` and validate
  configs with `python -c "import yaml; yaml.safe_load(open('config/prompts.yaml'))"`.
