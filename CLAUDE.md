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

## 每模型验收流程（顺序固定，不得跳步）

每个模型数据采集收齐后，严格按以下顺序执行：

**Step A — 分布预检（盲态）**
```bash
python scripts/preflight_distributions.py --model <model_id> --data-dir $DATA_DIR
```
此步仅看分数分布和 Δ/D 值，**不得运行 src/metrics.py 或任何含 kappa 的脚本**。

**Step B — 登记预测并 commit**
将 preflight 输出的粘贴块填入 `paper/PREDICTIONS.md`，填写盲态声明，commit。
commit 时间戳即为预测冻结时间，**commit 后不得修改预测**。

**Step C — 验收脚本（开箱）**
运行含 kappa 的验收脚本（`src/metrics.py` / `notebooks/02_analysis.ipynb`）。
将 κ 实际值和命中/未中/tie 回填到 PREDICTIONS.md 对应登记记录的"开箱结果"栏，commit。

**盲态标注规则**：
- `blind`：Step A 前未见过该模型的 κ 或一致性指标
- `non-blind`：已因任何原因（调试、dry-run 输出等）看过 κ，须说明原因
- haiku（anthropic/claude-haiku-4.5）和 gpt-5.4-mini（openai/gpt-5.4-mini）：
  κ 未暴露，按正常 blind 流程登记

## Internal docs
Strategy, roadmap, outline and status live in the private repo
`dukesky/politeness-llm-internal`. Do not add planning documents here;
this repo is the public paper artifact.

## 异常改写处理规则（L5_a 起）
若某改写数据干净（parse_ok 正常、finish_reason 正常）但行为畸形（分布/输出
长度显著偏离同档其他改写），不作为脏数据剔除，而是：(a) 主表保留、(b) 稳健性表
剔除后并列报告、(c) 若影响某个已登记预测的判定，在 PREDICTIONS.md 标注判定
敏感性。后续旗舰模型开箱遇类似情况按同口径处理。
