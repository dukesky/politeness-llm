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
- **haiku（anthropic/claude-haiku-4.5）和 gpt-5.4-mini（openai/gpt-5.4-mini）**：
  若在协议建立前（2026-06-12 前）已暴露过 κ，标注 `non-blind（pre-protocol）`

## Novelty 定位（写作时直接取用）

**核心声明**：首个研究"语气作为 IR relevance judge 严苛度工作点调节器"的工作。
机制假设——极端语气把 LLM judge 推向人类标注者的严苛度工作点，U 型 κ 曲线由此
产生——在可引用文献中无人认领，截至 2026-06 经深搜确认（最接近的表述仅见于非学
术博客）。

**三句差异化定位**：

(a) vs Arabzadeh & Clarke 2025（最近邻，SIGIR，`arabzadeh2025prompt`）：
    他们研究 prompt wording/source diversity 的敏感性，而非语气本身，且未提出
    任何机制解释；我们系统操纵礼貌档并提出+检验"严苛度调节"机制。

(b) vs 礼貌-准确率三部曲（`yin2024respect` / `dobariya2025tone` / `cai2025tone`）：
    三篇全是生成任务（MCQ/QA），均非 relevance judging；且均无分类器校准的语气档
    对照改写。我们两者皆有，使 Yin-vs-Dobariya 矛盾首次可以机制解释。

(c) vs 校准文献（CalibraEval、Two Ways to De-Bias 等）：
    他们做 post-hoc 输出端校准；我们揭示 prompt 语气是输入侧的系统性驱动因素，
    属于 validity threat 层面的不同问题。

**引用纪律**：
- `dobariya2025tone` 的"粗鲁有益"结论来自小样本（50 题 × 10 runs，~4 pt delta），
  已被 `cai2025tone` 部分反驳。只作为"矛盾一极"引用，不作为定论。
- `arabzadeh2025prompt`（2504.12408）与 `arabzadeh2025benchmarking`（2504.12558）
  是同组不同论文，分开引用，切勿合并。

**叙事降级预案**：若 U 型 κ 曲线在少于 2 个非 DeepSeek 模型上复现，核心声明降级为：
"语气是 LLM-based IR 评测的 systematic validity threat（model-dependent，
幅度可达 Δκ ≈ 0.05）"；机制解释移入 Discussion 作 exploratory 处理。

**白送加分项**：语气 × reasoning-token 消耗的交互无任何已发表研究。
Gemini 3.1 Pro reasoning-native 子组提供天然实验条件，标注为 exploratory analysis。

**时间窗口**：该方向 2025 末—2026 初密集出 preprint（见 `docs/CONTEXT.md` §7），
窗口持续收窄。7/15 R&P Notes → 8 月上旬 arXiv 时间戳不可往后推迟。
