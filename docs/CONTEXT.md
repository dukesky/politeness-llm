# 项目状态文档：Politeness-LLM（LLM Judge 语气敏感性研究）
更新于 2026-06-11。我是 Tim，Snap 的 Staff MLE（搜索推荐方向），产假期间做独立研究。
本 session 角色：研究顾问（设计/审结果/写作）。代码修改走本地 Claude Code（给我自包含
prompt 我转发）。实验在 Colab 跑，API 经 OpenRouter。

## 1. 研究定位
- 题目：prompt 语气/礼貌程度对 LLM relevance judgment 与 reranking 的影响
  ——定位为 LLM-based IR 评测的 validity threat 研究 + 成本效益分析。
- RQ1 语气是否系统性改变判断质量；RQ2 跨模型是否一致（回应文献矛盾：
  Yin et al. 2024 说粗鲁有害 vs Dobariya & Kumar 2025 说粗鲁更准）；
  RQ3 成本-质量前沿；RQ4 稳定性（翻转率/解析失败/排序τ）。
- 投稿路线：RecSys 2026 R&P Notes（7/15 截稿，extended abstract）→
  8 月上旬 arXiv → ECIR 2027 short（10 月截稿）。
- 核心机制假设（DeepSeek 数据支撑，待跨模型检验）：语气拨动的是 judge 的
  "严苛度工作点"而非判断质量本身；κ 升降取决于漂移方向是否对准人类标注者
  的严格度——该机制可统一解释前人矛盾。可检验预测："分数分布漂移方向 ×
  模型相对人类宽严"应能预测 κ 变化方向。

## 2. 实验设计
- 主任务：pointwise relevance judgment（UMBRELA 风格 0-3 四级，输出
  {"score": N}）。副实验（后做）：派生排序（按分数排序，BM25 破平局，
  condensed-list NDCG@10 / Kendall τ，零额外成本）+ RankGPT 式 listwise
  小规模探针（重点看格式崩坏率×语气）。
- Prompt：5 礼貌档 × 每档 3 改写 = 15 变体。评分 rubric 跨变体逐字一致，
  只动语气包装层（prefix/suffix）。已用 Intel/polite-guard 分类器校准并冻结
  （档均值 0.00/0.68/1.00/2.33/3.00，单调；L1/L5 与 L3 的改写区间不重叠）。
  铁律：采集开始后改措辞必须用新 prompt_id。
- 数据：TREC DL19（1,533 pairs）+ DL20（1,965）= 3,498 个
  "BM25 top-50 ∩ 有人工标注" 的 (query, passage) 对，pairs 文件已冻结。
  W2 计划加 BEIR 子集（SciFact/NFCorpus/TREC-COVID，先看主结果再定规模）。
- 指标：linear-weighted κ vs NIST qrels、分数分布×档、单点翻转率
  （对照 6.4% 的 temperature=0 非确定性底噪）、解析失败率、$/1k judgments、
  成本-质量前沿图。

## 3. 模型阵容（OpenRouter，provider 全部锁定 + 实弹审计 8/8 通过）
便宜档（全网格 15 变体 × n_runs=2）：
  openai/gpt-5.4-mini@OpenAI | anthropic/claude-haiku-4.5@Anthropic |
  google/gemini-3.5-flash@Google | deepseek/deepseek-v4-flash@WandB(fp8) |
  qwen/qwen3.7-plus@Alibaba
旗舰档（20% 确定性子样本 × n_runs=1）：
  openai/gpt-5.5@OpenAI | anthropic/claude-opus-4.8@Anthropic |
  google/gemini-3.1-pro-preview@Google
Reasoning：7 个模型已实测归零（OpenAI 用 effort:"none"，Anthropic 用
enabled:false，DeepSeek/Qwen 用 enabled:false）；Gemini 3.1 Pro 强制思考
（API 不可关），归为 reasoning-native 子组：单独 max_tokens 4000、单独汇报、
附加分析"语气×思考长度"。总预算 < $200。

## 4. 基建
- GitHub private repo `politeness-llm`：config/prompts.yaml（冻结）、
  config/models.yaml、src/{collect,parse,metrics,build_pairs}.py、
  scripts/check_providers.py（实弹审计）、CLAUDE.md（项目铁律）。
- Google Drive `.../politeness-llm/`：inputs/（BM25 runs + pairs）、
  raw/（按模型子目录，append-only，逐条 flush）、derived/（parquet）。
- Colab：无状态执行器，开机 boilerplate = mount Drive + git pull +
  pip install + Secrets 读 OPENROUTER_API_KEY/GH_TOKEN。
- 采集器关键机制：唯一键 (model_id, prompt_id, qid, docid, run) 断点续跑；
  失败不落盘、重发命令补漏直到 "To collect: 0"；HTTP 200 一律记录
  （content 空 → parse_ok=False 是数据）；只对网络错误重试；任何单任务
  异常不得击落整体（return_exceptions=True）；每条记录带 git hash、
  finish_reason、usage（含 reasoning_tokens、实报 cost）。

## 5. 已踩过的雷（勿重复）
provider 名大小写精确匹配；OpenRouter 账户隐私设置会过滤 provider
（DeepSeek 官方不可见→锁 WandB）；reasoning 吃 max_tokens 导致 content=None
（已全面禁用推理 + 旗舰特例）；HTTP 200 畸形响应（无 choices）曾击落全局
（已兜底）；Qwen 隐形思考曾使成本 ×33（已归零）；pyserini 需 Java 21；
Colab pillow 损坏需 force-reinstall + 重启内核。

## 6. 已有结果（DeepSeek V4 Flash，104,940 条，完整）
- parse_ok = 100%（全档）。
- 分数分布呈以 L3 为底的对称 U 型：score=0 比例
  25.1/20.4/17.8/20.7/24.0%（L1→L5）；L3 给 3 分最多（20.8% vs ~17.5%）。
- κ 呈 U 形（L3 最低，极端档 κ 升高）：0.4835/0.4567/0.4296/0.4637/0.4829——极端语气反而更贴近
  人类（机制：NIST 比模型严格，极端语气把模型推严，撞向人类工作点）。
- 三关检验：档间/档内 std 比 1.58（L1、L5 与 L3 改写区间零重叠，中间档
  毛糙，L2_c=0.496 是异常点待定性分析）；dl19/dl20 独立复现 U 型；
  run1==run2 一致率 93.6%。

## 7. 当前进行中
- 采集：gpt-5.4-mini + haiku 跑批中；下一批 flash + qwen；最后三旗舰。
  每批收尾：补漏到 0 → parse → 验收脚本按 (model, level) 分组看 κ。
- 文献深搜在另一 session 进行（重点核查"严苛度校准机制"是否已被发表）。
- W5 分析清单：混合效应/bootstrap 显著性、机制预测链检验、成本前沿图、
  语气间翻转率 vs 6.4% 底噪、L2_c 异常点、Gemini Pro 思考长度×语气。
- 时间线：7/15 R&P Notes 摘要 → 8 月上旬 arXiv → 10 月 ECIR 2027 short。
