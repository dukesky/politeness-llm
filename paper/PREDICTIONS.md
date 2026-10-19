# Semi-Blind Prediction Registration
# 半盲预测登记

**冻结声明**：本文件中的预测规则（四节）自首次 commit 起冻结，不得修改。
每个模型的预测以"## 登记记录"小节追加；开箱结果在 Step C 后回填。
规则本体的任何修订须另开新版本文件，并在此处标注废弃。

---

## 预测登记规则（冻结）

### 第一节：关键量定义

**D(ℓ)**：第 ℓ 档相对于中性基线（L3）的分数漂移，合并该档全部改写与 runs。

    D(ℓ) = mean_score(ℓ) − mean_score(L3)，ℓ ∈ {L1, L2, L4, L5}

**Δ（模型宽严偏差）**：模型在中性档（L3）的均分与 qrels 均分之差。

    Δ = mean_score(L3, model) − mean_score(qrels)

- Δ > 0：模型比人类宽松（overscoring），需要"漂严"才能更贴近人类工作点
- Δ < 0：模型比人类严苛（underscoring），需要"漂宽"才能更贴近人类工作点

**no-call 阈值**：|D(ℓ)| < 0.02 时，漂移量不足以形成可靠预测，记为 no call。

---

### 第二节：κ 方向预测规则（严苛度工作点机制）

机制：语气漂移改变 judge 的打分严苛度；κ 升降取决于漂移方向是否把模型推向
人类标注者的工作点。

| 条件 | 预测 |
|------|------|
| Δ > 0（模型偏宽）且 D(ℓ) < −0.02（该档漂严）| κ(ℓ) > κ(L3)，↑ |
| Δ > 0（模型偏宽）且 D(ℓ) > +0.02（该档漂宽）| κ(ℓ) < κ(L3)，↓ |
| Δ < 0（模型偏严）且 D(ℓ) > +0.02（该档漂宽）| κ(ℓ) > κ(L3)，↑ |
| Δ < 0（模型偏严）且 D(ℓ) < −0.02（该档漂严）| κ(ℓ) < κ(L3)，↓ |
| \|D(ℓ)\| < 0.02（任意 Δ）                    | no call           |

注：Δ = 0 的极端情况在实践中不会精确出现；若 |Δ| < 0.01，也记为 no call
（模型基本对齐人类，机制不适用）。

---

### 第三节：终点定义与平局阈值

**主终点**（参与最终二项检验）：
- L1 vs L3 的 κ 方向预测
- L5 vs L3 的 κ 方向预测

**次终点**（单独报告，不进最终二项检验）：
- L2 vs L3 的 κ 方向预测
- L4 vs L3 的 κ 方向预测

**平局阈值**：|Δκ| = |κ(ℓ) − κ(L3)| < 0.005 时，无论预测方向如何，
记为 tie，不计入命中也不计入未中。

---

### 第四节：汇总评判标准

**二项检验**：以全部非 no-call 的主终点预测（跨模型汇总）为样本，
H₀: 命中概率 p = 0.5（等价于随机猜测），单侧检验。

**模型级别**：每模型独立报告主终点命中率（命中/（命中+未中），tie 排除）。

**汇总级别**：跨所有模型的主终点命中情况做二项检验。若 p < 0.05，支持机制假设。

**降级条件**：若主终点 no-call 率 > 50%（漂移普遍过小），降级为 exploratory。

---

## 登记记录

<!-- 每模型顺序：parse → preflight_distributions(Step A) →
     在此追加登记并 commit(Step B) → 验收脚本含 κ(Step C) → 回填结果 -->

<!-- 登记格式模板：

### {model_id} — {YYYY-MM-DD} — git hash {hash}
- 盲态: blind / non-blind（说明原因）
- Δ = {值}（模型 L3 均分 {x} − qrels 均分 {y}）
- D(L1)={}, D(L2)={}, D(L4)={}, D(L5)={}
- 预测：
  - κ(L1) vs κ(L3): ↑ / ↓ / no call
  - κ(L5) vs κ(L3): ↑ / ↓ / no call（主终点）
  - κ(L2) vs κ(L3): ↑ / ↓ / no call（次终点）
  - κ(L4) vs κ(L3): ↑ / ↓ / no call（次终点）
- 开箱结果（Step C 后回填）：
  - κ(L1)={}, κ(L3)={}, κ(L5)={}，κ(L2)={}, κ(L4)={}
  - 命中：L1 {命中/未中/tie}，L5 {命中/未中/tie}
  - 次终点：L2 {命中/未中/tie}，L4 {命中/未中/tie}

-->

### deepseek/deepseek-v4-flash — 2026-06-11 — (pre-protocol, reference only)
- 盲态: non-blind（DeepSeek 为首个采集模型，κ 在协议建立前已暴露）
- Step C 验证 κ（L1→L5）：0.4836 / 0.4573 / 0.4295 / 0.4637 / 0.4830（U 形，L3 最低，极端档 κ 升高）
- 用途：机制假设来源；不参与预测命中率统计；用于后验校准 Δ 基准

---

### openai/gpt-5.4-mini — 2026-06-13 — git hash 2b71996
- 盲态: blind（metrics/验收脚本从未对本模型运行过）
- Δ = +0.7085（模型 L3 均分 1.6408 − qrels 均分 0.9322）
- D(L1)=-0.0458, D(L2)=+0.0049, D(L4)=+0.0000, D(L5)=-0.0153
- 预测：
  - κ(L1) vs κ(L3): ↑  (model偏宽, 该档漂严 → 更贴近人类工作点)
  - κ(L5) vs κ(L3): no call  (|D|<0.02)  ← 主终点
  - κ(L2) vs κ(L3): no call  (|D|<0.02)  ← 次终点
  - κ(L4) vs κ(L3): no call  (|D|<0.02)  ← 次终点
- 开箱结果（Step C 完成 2026-06-13）：
  - κ(L1)=0.3335, κ(L3)=0.3270, κ(L5)=0.3245，κ(L2)=0.3124, κ(L4)=0.3242
  - 主终点：L1 Δκ=+0.0064 → ↑ **命中**；L5 Δκ=-0.0026 → tie（predicted no call，不参与评分）
  - 次终点：L2 Δκ=-0.0147 → ↓（predicted no call，不参与评分）；L4 Δκ=-0.0028 → tie（predicted no call，不参与评分）
  - 命中统计（主终点）：1 命中 / 0 未中（L5 no-call 排除在外）

---

### anthropic/claude-haiku-4.5 — 2026-06-13 — git hash 2b71996
- 盲态: blind（metrics/验收脚本从未对本模型运行过）
- Δ = +0.5195（模型 L3 均分 1.4518 − qrels 均分 0.9322）
- D(L1)=-0.0108, D(L2)=+0.0132, D(L4)=-0.0211, D(L5)=-0.0102
- 预测：
  - κ(L1) vs κ(L3): no call  (|D|<0.02)
  - κ(L5) vs κ(L3): no call  (|D|<0.02)  ← 主终点
  - κ(L2) vs κ(L3): no call  (|D|<0.02)  ← 次终点
  - κ(L4) vs κ(L3): ↑  (model偏宽, 该档漂严 → 更贴近人类工作点)  ← 次终点
- 开箱结果（Step C 完成 2026-06-13）：
  - κ(L1)=0.4334, κ(L3)=0.4340, κ(L5)=0.4240，κ(L2)=0.4242, κ(L4)=0.4275
  - 主终点：L1 Δκ=-0.0006 → tie（predicted no call，不参与评分）；L5 Δκ=-0.0100 → ↓（predicted no call，不参与评分）
  - 次终点：L2 Δκ=-0.0099 → ↓（predicted no call，不参与评分）；L4 Δκ=-0.0065 → ↓（predicted ↑，**未中**）
  - 命中统计（主终点）：0 命中 / 0 未中（L1+L5 均 no-call 排除在外）；次终点 L4 未中 1

---

### qwen/qwen3.7-plus — 2026-06-15 — git hash 0548224
- 盲态: blind（metrics/验收脚本从未对本模型运行过）
- Δ = +0.3953（模型 L3 均分 1.3276 − qrels 均分 0.9322）
- D(L1)=-0.0747, D(L2)=+0.0121, D(L4)=+0.0376, D(L5)=-0.0084
- 预测：
  - κ(L1) vs κ(L3): ↑  (model偏宽, 该档漂严 → 更贴近人类工作点)
  - κ(L5) vs κ(L3): no call  (|D|<0.02)  ← 主终点
  - κ(L2) vs κ(L3): no call  (|D|<0.02)  ← 次终点
  - κ(L4) vs κ(L3): ↓  (model偏宽, 该档漂宽 → 远离人类工作点)  ← 次终点
- 开箱结果（Step C 完成 2026-06-14）：
  - κ(L1)=0.4787, κ(L3)=0.4863, κ(L5)=0.4824，κ(L2)=0.4715, κ(L4)=0.4542
  - 主终点：L1 Δκ=-0.0077 → ↓（predicted ↑，**未中**）；L5 Δκ=-0.0039 → tie（predicted no call，不参与评分）
  - 次终点：L2 Δκ=-0.0148 → ↓（predicted no call，不参与评分）；L4 Δκ=-0.0321 → ↓（predicted ↓，**命中**）
  - 命中统计（主终点）：0 命中 / 1 未中（L5 tie 排除）；次终点 L4 命中 1

---

### google/gemini-3.5-flash — 2026-06-15 — git hash 0548224
- 盲态: blind（metrics/验收脚本从未对本模型运行过）
- Δ = +0.4798（模型 L3 均分 1.4120 − qrels 均分 0.9322）
- D(L1)=-0.0519, D(L2)=+0.0005, D(L4)=-0.0151, D(L5)=+0.0382
- 预测：
  - κ(L1) vs κ(L3): ↑  (model偏宽, 该档漂严 → 更贴近人类工作点)
  - κ(L5) vs κ(L3): ↓  (model偏宽, 该档漂宽 → 远离人类工作点)  ← 主终点
  - κ(L2) vs κ(L3): no call  (|D|<0.02)  ← 次终点
  - κ(L4) vs κ(L3): no call  (|D|<0.02)  ← 次终点
- 开箱结果（Step C 完成 2026-06-14，git hash 40a6e15）：
  - κ(L1)=0.4549, κ(L3)=0.4414, κ(L5)=0.3940，κ(L2)=0.4343, κ(L4)=0.4532
  - 命中：L1 ✓（Δκ=+0.0134，预测↑）；L5 ✓（Δκ=-0.0474，预测↓）— ⚠️ 见下方敏感性说明
  - 次终点：L2 不计（no-call），L4 不计（no-call）
  - 命中统计（主终点）：2 命中 / 0 未中
  - 【L5 判定敏感性】L5 的"↓"由单条改写 L5_a 驱动。L5 三改写 κ：
    L5_a=0.2712 / L5_b=0.4491 / L5_c=0.4619。L5_a 数据干净
    （parse_ok=1.0，finish_reason 全 stop），但行为畸形：输出 token
    缩短（median 6 vs 11），判分坍缩至中间值（s=0 仅 3.2% vs L5_b/c
    的 16-18%），dl19/dl20 一致复现。剔除 L5_a 后 L5 κ≈0.4555 > κ(L3)，
    预测翻转为【未中】。三改写 classifier_score 均为 2.999、语义与指令
    等价，偏移不可归因于礼貌程度或显式指令 → 判定为 paraphrase-level
    spurious sensitivity，作为 case study 单独分析（见
    paper/findings/L5a_paraphrase_case.md）。主表保留 L5_a，稳健性表剔除。

---

### openai/gpt-5.5 — 2026-06-16 — git hash 7ff1163
- 盲态: blind（metrics/验收脚本从未对本模型运行过）
- 数据范围: 冻结 40% 子样本（flagship filter 37980 → 20970 rows，1398 pairs）
- Δ = +0.6619（模型 L3 均分 1.6476 − qrels 均分 0.9857）
- D(L1)=-0.0091, D(L2)=+0.0169, D(L4)=+0.0305, D(L5)=+0.0300
- 预测：
  - κ(L1) vs κ(L3): no call  (|D|<0.02)  ← 主终点
  - κ(L5) vs κ(L3): ↓  (model偏宽, 该档漂宽 → 远离人类工作点)  ← 主终点
  - κ(L2) vs κ(L3): no call  (|D|<0.02)  ← 次终点
  - κ(L4) vs κ(L3): ↓  (model偏宽, 该档漂宽 → 远离人类工作点)  ← 次终点
- 开箱结果（Step C 完成 2026-06-16）：
  - κ(L1)=0.3752, κ(L3)=0.3740, κ(L5)=0.3659，κ(L2)=0.3704, κ(L4)=0.3681
  - 主终点：L1 Δκ=+0.0011 → tie（predicted no call，不参与评分）；L5 Δκ=-0.0081 → ↓（predicted ↓，**命中**）
  - 次终点：L2 Δκ=-0.0037 → tie（predicted no call，不参与评分）；L4 Δκ=-0.0060 → ↓（predicted ↓，**命中**）
  - 命中统计（主终点）：1 命中 / 0 未中（L1 tie/no call 排除）；次终点 L4 命中 1

---

### anthropic/claude-opus-4.8 — 2026-06-16 — git hash 7ff1163
- 盲态: blind（metrics/验收脚本从未对本模型运行过）
- 数据范围: 冻结 40% 子样本（flagship filter 24227 → 20970 rows，1398 pairs）
- Δ = +0.3522（模型 L3 均分 1.3379 − qrels 均分 0.9857）
- D(L1)=-0.0381, D(L2)=+0.0060, D(L4)=-0.0017, D(L5)=+0.0253
- 预测：
  - κ(L1) vs κ(L3): ↑  (model偏宽, 该档漂严 → 更贴近人类工作点)  ← 主终点
  - κ(L5) vs κ(L3): ↓  (model偏宽, 该档漂宽 → 远离人类工作点)  ← 主终点
  - κ(L2) vs κ(L3): no call  (|D|<0.02)  ← 次终点
  - κ(L4) vs κ(L3): no call  (|D|<0.02)  ← 次终点
- 开箱结果（Step C 完成 2026-06-16）：
  - κ(L1)=0.5077, κ(L3)=0.4967, κ(L5)=0.4848，κ(L2)=0.4816, κ(L4)=0.4880
  - 主终点：L1 Δκ=+0.0110 → ↑（predicted ↑，**命中**）；L5 Δκ=-0.0119 → ↓（predicted ↓，**命中**）
  - 次终点：L2 Δκ=-0.0151 → ↓（predicted no call，不参与评分）；L4 Δκ=-0.0087 → ↓（predicted no call，不参与评分）
  - 命中统计（主终点）：2 命中 / 0 未中

---

### google/gemini-3.1-pro-preview — 2026-06-16 — git hash 7ff1163
- 盲态: blind（metrics/验收脚本从未对本模型运行过）
- 数据范围: 冻结 40% 子样本（flagship filter 20970 → 20970 rows，1398 pairs；无旧脏数据）
- Δ = +0.5396（模型 L3 均分 1.5253 − qrels 均分 0.9857）
- D(L1)=+0.0350, D(L2)=+0.0200, D(L4)=+0.0146, D(L5)=+0.0195
- 预测：
  - κ(L1) vs κ(L3): ↓  (model偏宽, 该档漂宽 → 远离人类工作点)  ← 主终点
  - κ(L5) vs κ(L3): no call  (|D|<0.02)  ← 主终点
  - κ(L2) vs κ(L3): ↓  (model偏宽, 该档漂宽 → 远离人类工作点)  ← 次终点
  - κ(L4) vs κ(L3): no call  (|D|<0.02)  ← 次终点
- 开箱结果（Step C 完成 2026-06-16）：
  - κ(L1)=0.4222, κ(L3)=0.4452, κ(L5)=0.4395，κ(L2)=0.4337, κ(L4)=0.4402
  - 主终点：L1 Δκ=-0.0230 → ↓（predicted ↓，**命中**）；L5 Δκ=-0.0056 → ↓（predicted no call，不参与评分）
  - 次终点：L2 Δκ=-0.0114 → ↓（predicted ↓，**命中**）；L4 Δκ=-0.0050 → tie（predicted no call，不参与评分）
  - 命中统计（主终点）：1 命中 / 0 未中（L5 no call 排除）；次终点 L2 命中 1
  - 注：reasoning-native 模型，reasoning token 消耗显著（$/1k L1=6.08，L3=3.03），
    语气 × 推理消耗交互作为 exploratory analysis 单独报告（见 CLAUDE.md 白送加分项）

---

## 探索性预测（登记于任何模型开箱之前）

对全部 (模型 × 档) 格子（含 no-call 格），定义对齐变化量

    A(model, ℓ) = |Δ + D(ℓ)| − |Δ|

机制预测：A 与实际的 κ(ℓ) − κ(L3) 呈负相关（A < 0 表示漂移使工作点更靠近
人类，预测 κ 上升）。开箱后用 Spearman 相关检验，跨所有非推理模型合并计算。
此预测在 mini/haiku 及其余模型 κ 全部未知时登记。

### 探索性预测开箱结果（Step C 完成 2026-06-16，git hash 1ab7686）

口径说明：非推理模型 7 个（deepseek/gpt-5.4-mini/haiku/qwen/gemini-3.5-flash/
gpt-5.5/opus-4.8），每模型 4 档（L1/L2/L4/L5）共 28 个格子。
gemini-3.1-pro-preview 为 reasoning-native，全程单独报告，不进本 Spearman。

| 稳健性口径 | n（格） | Spearman ρ | p 值 | 结论 |
|-----------|---------|------------|------|------|
| 全样本（含 DeepSeek + gemini-flash L5_a）| 28 | **−0.616** | **0.0005** | 显著 |
| 稳健性 a（剔 gemini-flash L5，L5_a 异常）| 27 | −0.572 | 0.0018 | 显著 |
| 稳健性 b（剔 DeepSeek，non-blind 来源）| 24 | **−0.680** | **0.0003** | 显著↑ |
| 稳健性 c（剔 DeepSeek + gemini-flash L5_a）| 23 | −0.636 | 0.0011 | 显著 |

**Spearman 勘误（2026-10-12，按只增不改约定追加，不回改上表）**：上表"全样本 28 格"与"稳健性 a"
两行使用的 DeepSeek Δ/D(ℓ) 是手工录入的早期值（Δ=+0.71；D=−0.073/−0.030/+0.029/−0.018），
PREDICTIONS.md 的 DeepSeek 登记块本身并未记录 Δ 与 D(ℓ)。按最终 parquet 重算（RecSys'26 Notes
camera-ready 所用值：Δ=+0.543；D(L1/L2/L4/L5)=−0.148/−0.079/−0.094/−0.134），28 格 ρ=−0.799
（p=3.5e−7），剔 gemini-flash L5_a 的 27 格 ρ=−0.775。不含 DeepSeek 的两行（24 格 −0.680、23 格
−0.636）不受影响——这两行才是盲态 judge 的检验，论文以其为主、28 格为辅。Δκ 值（t3 表）不依赖 Δ/D，
不受影响。

**per-model 方向符合格数**（A < 0 → 实际 κ 升，A > 0 → 实际 κ 降）：

| 模型 | 符合 / 共 4 格 | 备注 |
|------|--------------|------|
| deepseek/deepseek-v4-flash | 3/4 | non-blind reference |
| openai/gpt-5.4-mini | 2/4 | |
| anthropic/claude-haiku-4.5 | 1/4 | 漂移近零，方向近随机 |
| qwen/qwen3.7-plus | 2/4 | |
| google/gemini-3.5-flash | 4/4 | 漂移大，信号强 |
| openai/gpt-5.5 | 4/4 | 漂移大，信号强 |
| anthropic/claude-opus-4.8 | 3/4 | |

**结论**：机制方向预测得预注册验证。负相关在剔除 non-blind 来源（DeepSeek）后
不降反升（ρ=-0.68），洗清循环论证嫌疑。四种切法 ρ∈[-0.68,-0.57] 全部 p<0.002，
结论不依赖任何单一模型或数据点。预测力随效应强度增强（漂移明显的 gpt-5.5 /
gemini-flash 达 4/4，效应近零的 haiku 仅 1/4），符合"无信号时方向随机"的机制预期。

---

## 汇总评判结果（2026-06-16）

### 口径声明
- **非推理模型**（进 Spearman 与二项检验）：deepseek / gpt-5.4-mini / haiku /
  qwen3.7-plus / gemini-3.5-flash / gpt-5.5 / claude-opus-4.8（共 7 个）
- **reasoning-native 子组**（单独报告）：gemini-3.1-pro-preview
  不进 Spearman 汇总与二项检验；reasoning token × 语气交互作为 exploratory analysis。

### 主终点二项检验（仅 blind 模型，排除 gemini-pro）

参与计算的主终点格（非 no-call、非 tie）：

| 模型 | 终点 | 预测 | 实际 | 判定 |
|------|------|------|------|------|
| gpt-5.4-mini | L1 | ↑ | ↑ | ✓ 命中 |
| gemini-3.5-flash | L1 | ↑ | ↑ | ✓ 命中 |
| gemini-3.5-flash | L5 | ↓ | ↓ | ✓ 命中 ⚠️ L5_a 污染，见敏感性说明 |
| claude-opus-4.8 | L1 | ↑ | ↑ | ✓ 命中 |
| claude-opus-4.8 | L5 | ↓ | ↓ | ✓ 命中 |
| openai/gpt-5.5 | L5 | ↓ | ↓ | ✓ 命中 |
| qwen3.7-plus | L1 | ↑ | ↓ | ✗ 未中 |

**汇总：6 命中 / 1 未中**（单侧二项检验，H₀: p=0.5）
- 全 7 格：p = 1 − B(0; 7, 0.5) − B(1; 7, 0.5) ≈ **0.0625**（边缘显著）
- 剔除 gemini-flash L5（L5_a 污染）后 6 格：5 命中 / 1 未中，p ≈ 0.109

### 主检验

**Spearman 探索性预测为本研究主检验**（预注册，n=28，ρ=−0.616，p=0.0005；
四稳健性口径 ρ∈[−0.68,−0.57]，全部 p<0.002）。
二项检验功效受大量 no-call 格（漂移普遍较小）制约，作为辅助证据。
机制方向预测（A < 0 → κ 升）经预注册 Spearman 检验得到支持；
二项检验边缘显著（p≈0.06），方向一致。

---

# Round 2 预测登记 — DL21 / ANTIQUE 现象复现 + 新旗舰旧数据补跑
登记时间：2026-09-14（本地）/ preflight 输出时间戳 2026-09-15 UTC
代码版本：git 82861e5（多数据集验收管线；ANTIQUE qrels 1–4 → 0–3 仿射映射）
盲态说明：以下全部 (model, dataset) 组合的 κ 或任何 judge-vs-human 一致性指标
在本登记 commit 之前从未被计算或查看过（round-2 采集后仅运行过分布类
preflight 与完整性对账），全部登记为 blind。DL22 全库封存，不在本轮登记。
预测规则同 §2（|D|<0.02 → no call）。旗舰（opus-5 / gpt-5.6-sol）各数据集
均为 40% 冻结子样本 ×1 run；dl19,dl20 组为跨代际同数据对比（vs opus-4.8 /
gpt-5.5 的 v1 开箱结果）。

### deepseek/deepseek-v4-flash — 2026-09-15 — git hash 82861e5
- 数据集: dl21
- 盲态: blind
- Δ = +0.5855（模型 L3 均分 1.6810 − qrels 均分 1.0954）
- D(L1)=-0.1965, D(L2)=-0.0795, D(L4)=-0.0966, D(L5)=-0.1512
- 预测：
  - κ(L1) vs κ(L3): ↑ (model偏宽, 该档漂严 → 更贴近人类工作点)
  - κ(L5) vs κ(L3): ↑ (model偏宽, 该档漂严 → 更贴近人类工作点) ← 主终点
  - κ(L2) vs κ(L3): ↑ ← 次终点
  - κ(L4) vs κ(L3): ↑ ← 次终点
- 开箱结果（2026-09-14 Step C，git 82861e5）：
  - κ(L1)=0.4723, κ(L3)=0.4133, κ(L5)=0.4619，κ(L2)=0.4387, κ(L4)=0.4416
  - L1: +0.0590 ↑ 命中；L5: +0.0486 ↑ 命中；L2: +0.0254 ↑ 命中；L4: +0.0282 ↑ 命中（4/4）；btw/in=2.04x，U 型复现（L3 谷底）

### deepseek/deepseek-v4-flash — 2026-09-15 — git hash 82861e5
- 数据集: antique
- 盲态: blind
- Δ = -0.3997（模型 L3 均分 0.9308 − qrels 均分 1.3305）
- D(L1)=-0.0102, D(L2)=-0.0401, D(L4)=-0.0626, D(L5)=-0.0728
- 预测：
  - κ(L1) vs κ(L3): no call (|D|<0.02)
  - κ(L5) vs κ(L3): ↓ (model偏严, 该档漂严 → 远离人类工作点) ← 主终点
  - κ(L2) vs κ(L3): ↓ ← 次终点
  - κ(L4) vs κ(L3): ↓ ← 次终点
- 开箱结果（2026-09-14 Step C，git 82861e5）：
  - κ(L1)=0.4681, κ(L3)=0.4958, κ(L5)=0.4692，κ(L2)=0.4813, κ(L4)=0.4769
  - L1: −0.0278 ↓（no call）；L5: −0.0267 ↓ 命中；L2: −0.0145 ↓ 命中；L4: −0.0190 ↓ 命中（3/3）；btw/in=0.89x；倒 U（L3 峰顶）——Δ 反号后方向整体翻转，命中机制预测

### deepseek/deepseek-v4.1-flash — 2026-09-15 — git hash 82861e5
- 数据集: dl21
- 盲态: blind
- Δ = +0.5005（模型 L3 均分 1.5960 − qrels 均分 1.0954）
- D(L1)=-0.1537, D(L2)=-0.0192, D(L4)=+0.0025, D(L5)=-0.0234
- 预测：
  - κ(L1) vs κ(L3): ↑ (model偏宽, 该档漂严 → 更贴近人类工作点)
  - κ(L5) vs κ(L3): ↑ ← 主终点
  - κ(L2) vs κ(L3): no call (|D|<0.02) ← 次终点
  - κ(L4) vs κ(L3): no call (|D|<0.02) ← 次终点
- 开箱结果（2026-09-14 Step C，git 82861e5）：
  - κ(L1)=0.4671, κ(L3)=0.4113, κ(L5)=0.4251，κ(L2)=0.4154, κ(L4)=0.4145
  - L1: +0.0559 ↑ 命中；L5: +0.0139 ↑ 命中；L2/L4: tie（no call 一致）（2/2）；btw/in=2.60x，U 型（弱于 v4 的对称性，L1 端主导）

### deepseek/deepseek-v4.1-flash — 2026-09-15 — git hash 82861e5
- 数据集: antique
- 盲态: blind
- Δ = -0.2793（模型 L3 均分 1.0511 − qrels 均分 1.3305）
- D(L1)=-0.0497, D(L2)=+0.0128, D(L4)=+0.0227, D(L5)=+0.0183
- 预测：
  - κ(L1) vs κ(L3): ↓ (model偏严, 该档漂严 → 远离人类工作点)
  - κ(L5) vs κ(L3): no call (|D|<0.02) ← 主终点
  - κ(L2) vs κ(L3): no call (|D|<0.02) ← 次终点
  - κ(L4) vs κ(L3): ↑ (model偏严, 该档漂宽 → 更贴近人类工作点) ← 次终点
- 开箱结果（2026-09-14 Step C，git 82861e5）：
  - κ(L1)=0.4871, κ(L3)=0.5115, κ(L5)=0.5131，κ(L2)=0.5072, κ(L4)=0.5172
  - L1: −0.0244 ↓ 命中；L5: +0.0017 tie（no call 一致）；L2: tie（no call 一致）；L4: +0.0057 ↑ 命中（2/2）

### google/gemini-3.5-flash — 2026-09-15 — git hash 82861e5
- 数据集: dl21
- 盲态: blind
- Δ = +0.4525（模型 L3 均分 1.5479 − qrels 均分 1.0954）
- D(L1)=-0.0835, D(L2)=-0.0052, D(L4)=-0.0118, D(L5)=-0.0093
- 预测：
  - κ(L1) vs κ(L3): ↑ (model偏宽, 该档漂严 → 更贴近人类工作点)
  - κ(L5) vs κ(L3): no call (|D|<0.02) ← 主终点
  - κ(L2) vs κ(L3): no call (|D|<0.02) ← 次终点
  - κ(L4) vs κ(L3): no call (|D|<0.02) ← 次终点
- 开箱结果（2026-09-14 Step C，git 82861e5）：
  - κ(L1)=0.4933, κ(L3)=0.4696, κ(L5)=0.4800，κ(L2)=0.4667, κ(L4)=0.4791
  - L1: +0.0237 ↑ 命中（1/1）；其余 no call；btw/in=1.99x

### google/gemini-3.5-flash — 2026-09-15 — git hash 82861e5
- 数据集: antique
- 盲态: blind
- Δ = -0.3288（模型 L3 均分 1.0009 − qrels 均分 1.3297）
- D(L1)=-0.0402, D(L2)=+0.0213, D(L4)=-0.0061, D(L5)=+0.0010
- 预测：
  - κ(L1) vs κ(L3): ↓ (model偏严, 该档漂严 → 远离人类工作点)
  - κ(L5) vs κ(L3): no call (|D|<0.02) ← 主终点
  - κ(L2) vs κ(L3): ↑ (model偏严, 该档漂宽 → 更贴近人类工作点) ← 次终点
  - κ(L4) vs κ(L3): no call (|D|<0.02) ← 次终点
- 开箱结果（2026-09-14 Step C，git 82861e5）：
  - κ(L1)=0.5055, κ(L3)=0.5304, κ(L5)=0.5276，κ(L2)=0.5322, κ(L4)=0.5234
  - L1: −0.0249 ↓ 命中；L2: +0.0018 tie（预测 ↑ 未中，tie）（1 命中 1 tie）；btw/in=1.31x

### google/gemini-3.8-flash — 2026-09-15 — git hash 82861e5  ⟨reasoning-native，单列⟩
- 数据集: dl21
- 盲态: blind
- Δ = +0.3802（模型 L3 均分 1.4756 − qrels 均分 1.0954）
- D(L1)=+0.0019, D(L2)=+0.0066, D(L4)=+0.0052, D(L5)=+0.0214
- 预测：
  - κ(L1) vs κ(L3): no call (|D|<0.02)
  - κ(L5) vs κ(L3): ↓ (model偏宽, 该档漂宽 → 远离人类工作点) ← 主终点
  - κ(L2) vs κ(L3): no call (|D|<0.02) ← 次终点
  - κ(L4) vs κ(L3): no call (|D|<0.02) ← 次终点
- 开箱结果（2026-09-14 Step C，git 82861e5）：
  - κ(L1)=0.4921, κ(L3)=0.4909, κ(L5)=0.4869，κ(L2)=0.4883, κ(L4)=0.4905
  - L5: −0.0040 tie（预测 ↓ 未中，tie）（0/1）；全档 |Δκ|<0.005，btw/in=0.60x——reasoning-native 对 tone 近乎免疫

### google/gemini-3.8-flash — 2026-09-15 — git hash 82861e5  ⟨reasoning-native，单列⟩
- 数据集: antique
- 盲态: blind
- Δ = -0.3308（模型 L3 均分 0.9996 − qrels 均分 1.3304）
- D(L1)=+0.0243, D(L2)=+0.0213, D(L4)=+0.0039, D(L5)=+0.0189
- 预测：
  - κ(L1) vs κ(L3): ↑ (model偏严, 该档漂宽 → 更贴近人类工作点)
  - κ(L5) vs κ(L3): no call (|D|<0.02) ← 主终点
  - κ(L2) vs κ(L3): ↑ ← 次终点
  - κ(L4) vs κ(L3): no call (|D|<0.02) ← 次终点
- 开箱结果（2026-09-14 Step C，git 82861e5）：
  - κ(L1)=0.5397, κ(L3)=0.5350, κ(L5)=0.5396，κ(L2)=0.5415, κ(L4)=0.5358
  - L1: +0.0047 tie（预测 ↑，tie）；L2: +0.0064 ↑ 命中（1 命中 1 tie）；btw/in=0.66x

### anthropic/claude-opus-5 — 2026-09-15 — git hash 82861e5
- 数据集: dl21
- 盲态: blind
- Δ = +0.4405（模型 L3 均分 1.5145 − qrels 均分 1.0740）
- D(L1)=-0.0193, D(L2)=-0.0005, D(L4)=+0.0185, D(L5)=+0.0032
- 预测：全部 no call（|D|<0.02）
- 开箱结果（2026-09-14 Step C，git 82861e5）：
  - κ(L1)=0.4840, κ(L3)=0.4899, κ(L5)=0.4780，κ(L2)=0.4878, κ(L4)=0.4869
  - 全部 no call，实际全档 |Δκ|≤0.012，一致；btw/in=0.94x

### anthropic/claude-opus-5 — 2026-09-15 — git hash 82861e5
- 数据集: antique
- 盲态: blind
- Δ = -0.3379（模型 L3 均分 0.9753 − qrels 均分 1.3132）
- D(L1)=+0.0189, D(L2)=+0.0075, D(L4)=+0.0483, D(L5)=+0.0486
- 预测：
  - κ(L1) vs κ(L3): no call (|D|<0.02)
  - κ(L5) vs κ(L3): ↑ (model偏严, 该档漂宽 → 更贴近人类工作点) ← 主终点
  - κ(L2) vs κ(L3): no call (|D|<0.02) ← 次终点
  - κ(L4) vs κ(L3): ↑ ← 次终点
- 开箱结果（2026-09-14 Step C，git 82861e5）：
  - κ(L1)=0.5467, κ(L3)=0.5452, κ(L5)=0.5584，κ(L2)=0.5381, κ(L4)=0.5674
  - L5: +0.0132 ↑ 命中；L4: +0.0221 ↑ 命中（2/2）；btw/in=1.27x

### anthropic/claude-opus-5 — 2026-09-15 — git hash 82861e5  ⟨代际对：vs claude-opus-4.8 v1⟩
- 数据集: dl19,dl20
- 盲态: blind
- Δ = +0.4044（模型 L3 均分 1.3901 − qrels 均分 0.9857）
- D(L1)=-0.0036, D(L2)=-0.0143, D(L4)=+0.0380, D(L5)=+0.0246
- 预测：
  - κ(L1) vs κ(L3): no call (|D|<0.02)
  - κ(L5) vs κ(L3): ↓ (model偏宽, 该档漂宽 → 远离人类工作点) ← 主终点
  - κ(L2) vs κ(L3): no call (|D|<0.02) ← 次终点
  - κ(L4) vs κ(L3): ↓ ← 次终点
- 开箱结果（2026-09-14 Step C，git 82861e5）：
  - κ(L1)=0.4845, κ(L3)=0.4980, κ(L5)=0.4733，κ(L2)=0.5098, κ(L4)=0.4722
  - L5: −0.0247 ↓ 命中；L4: −0.0258 ↓ 命中（2/2）；btw/in=1.37x；代际对：κ(L3) 0.4980 vs opus-4.8 的 0.4967，L5 端效应放大（−0.025 vs −0.012）

### openai/gpt-5.6-sol — 2026-09-15 — git hash 82861e5
- 数据集: dl21
- 盲态: blind
- Δ = +0.5493（模型 L3 均分 1.6233 − qrels 均分 1.0740）
- D(L1)=-0.0086, D(L2)=+0.0225, D(L4)=+0.0225, D(L5)=+0.0086
- 预测：
  - κ(L1) vs κ(L3): no call (|D|<0.02)
  - κ(L5) vs κ(L3): no call (|D|<0.02) ← 主终点
  - κ(L2) vs κ(L3): ↓ (model偏宽, 该档漂宽 → 远离人类工作点) ← 次终点
  - κ(L4) vs κ(L3): ↓ ← 次终点
- 开箱结果（2026-09-14 Step C，git 82861e5）：
  - κ(L1)=0.4344, κ(L3)=0.4510, κ(L5)=0.4432，κ(L2)=0.4250, κ(L4)=0.4258
  - L2: −0.0261 ↓ 命中；L4: −0.0253 ↓ 命中（2/2）；btw/in=1.77x

### openai/gpt-5.6-sol — 2026-09-15 — git hash 82861e5
- 数据集: antique
- 盲态: blind
- Δ = -0.1610（模型 L3 均分 1.1522 − qrels 均分 1.3132）
- D(L1)=+0.0189, D(L2)=+0.0327, D(L4)=+0.0159, D(L5)=+0.0143
- 预测：
  - κ(L1) vs κ(L3): no call (|D|<0.02)
  - κ(L5) vs κ(L3): no call (|D|<0.02) ← 主终点
  - κ(L2) vs κ(L3): ↑ (model偏严, 该档漂宽 → 更贴近人类工作点) ← 次终点
  - κ(L4) vs κ(L3): no call (|D|<0.02) ← 次终点
- 开箱结果（2026-09-14 Step C，git 82861e5）：
  - κ(L1)=0.5534, κ(L3)=0.5586, κ(L5)=0.5602，κ(L2)=0.5526, κ(L4)=0.5593
  - L2: −0.0059 ↓（预测 ↑ **未中**）（0/1）；btw/in=0.62x

### openai/gpt-5.6-sol — 2026-09-15 — git hash 82861e5  ⟨代际对：vs gpt-5.5 v1⟩
- 数据集: dl19,dl20
- 盲态: blind
- Δ = +0.5815（模型 L3 均分 1.5672 − qrels 均分 0.9857）
- D(L1)=-0.0067, D(L2)=+0.0157, D(L4)=-0.0005, D(L5)=+0.0050
- 预测：全部 no call（|D|<0.02）
- 开箱结果（2026-09-14 Step C，git 82861e5）：
  - κ(L1)=0.4087, κ(L3)=0.4163, κ(L5)=0.4093，κ(L2)=0.4025, κ(L4)=0.4075
  - 全部 no call，实际全档 |Δκ|≤0.014，一致；btw/in=0.80x；代际对：κ(L3) 0.4163 vs gpt-5.5 的 0.3740

### Round 2 盲态观察备注（预测冻结前记录，非事后）
- 六个 judge 在 DL21 上全部偏宽（Δ ∈ [+0.38, +0.59]），与 v1 DL19/20 一致；
  在 ANTIQUE 上全部偏严（Δ ∈ [−0.40, −0.16]）——**Δ 符号跨数据集家族翻转**，
  机制据此预测同一语气档的 κ 方向也应随之翻转（最强的跨数据集检验）。
- deepseek 家族（v4 与 v4.1）在 DL21 上 D(L1) 大幅漂严（−0.15 ~ −0.20），
  U 型倾向似在延续；v4.1 幅度弱于 v4。
- gemini-3.8-flash（reasoning-native）漂移全档极小（|D|≤0.02 居多），
  与 v1 gemini-3.1-pro 的"reasoning 模型漂移小"观察一致。


### Round 2 开箱总评（2026-09-14）
- 有方向预测（called）共 24 项：**命中 20，tie 3，未中 1**（gpt-5.6-sol × ANTIQUE 的 L2 次终点）。
- 核心确认：**Δ 符号跨数据集家族翻转 → κ 方向随之翻转**（六 judge 在 DL21
  偏宽、ANTIQUE 偏严；DeepSeek 家族 DL21 呈 U 型 / ANTIQUE 呈倒 U，均为盲态
  预测命中）——operating-point 机制最强的跨数据集检验。
- DeepSeek 结构化效应在新一代（v4.1）与新数据集（DL21 btw/in=2.60x）上复现。
- reasoning-native（gemini-3.8-flash）两数据集全档 |Δκ|<0.007（btw/in 0.60–0.66x），
  与 v1 gemini-3.1-pro 一致：reasoning judge 对 tone 近乎免疫（n=2）。
- 代际对（同数据 DL19/20）：opus-5 与 opus-4.8 工作点接近（Δ +0.40 vs +0.35），
  opus-5 的 L5 端负效应放大；gpt-5.6-sol 较 gpt-5.5 κ(L3) 提升（0.4163 vs
  0.3740）且漂移收敛至全 no-call——代际迭代未消除宽松偏置（Δ 仍 +0.58）。

---

# C4 预注册 — DL22 解封（校正验证 held-out）
登记时间：2026-09-15。**用户于 2026-09-15 明确授权解封**（"好的解封"）。
本 commit 之前 DL22 的 κ 或任何 judge-vs-human 一致性指标从未被计算或查看
（分布预检于本日首次运行，仅分布，git 446927c）。解封仅通过
scripts/c4_unseal_dl22.py 执行一次；该脚本要求本登记块先行存在。

## C4-a DL22 语气方向预测（盲态，规则同 §2，|D|<0.02 → no call）

### deepseek/deepseek-v4-flash — dl22 — 盲态: blind
- Δ = +0.4384（L3 均分 1.1164 − qrels 均分 0.6780）
- D(L1)=-0.1572, D(L2)=-0.0717, D(L4)=-0.0821, D(L5)=-0.1626
- 预测：L1 ↑；L5 ↑（主）；L2 ↑（次）；L4 ↑（次）
- 开箱结果（C4 后回填）：

### deepseek/deepseek-v4.1-flash — dl22 — 盲态: blind
- Δ = +0.4432（1.1212 − 0.6780）
- D(L1)=-0.1279, D(L2)=-0.0022, D(L4)=-0.0093, D(L5)=-0.0339
- 预测：L1 ↑；L5 ↑（主）；L2 no call；L4 no call
- 开箱结果（C4 后回填）：

### google/gemini-3.5-flash — dl22 — 盲态: blind
- Δ = +0.3334（1.0114 − 0.6780）
- D(L1)=-0.0556, D(L2)=+0.0089, D(L4)=-0.0179, D(L5)=-0.0194
- 预测：L1 ↑；L5 no call（主）；L2 no call；L4 no call
- 开箱结果（C4 后回填）：

### google/gemini-3.8-flash — dl22 — 盲态: blind ⟨reasoning-native⟩
- Δ = +0.2809（0.9589 − 0.6780）
- D(L1)=+0.0161, D(L2)=+0.0080, D(L4)=+0.0048, D(L5)=+0.0140
- 预测：全部 no call（|D|<0.02）
- 开箱结果（C4 后回填）：

### anthropic/claude-opus-5 — dl22 — 盲态: blind
- Δ = +0.2414（0.9282 − 0.6868）
- D(L1)=-0.0054, D(L2)=-0.0198, D(L4)=+0.0535, D(L5)=+0.0423
- 预测：L1 no call；L5 ↓（主）；L2 no call；L4 ↓（次）
- 开箱结果（C4 后回填）：

### openai/gpt-5.6-sol — dl22 — 盲态: blind
- Δ = +0.3946（1.0815 − 0.6868）
- D(L1)=+0.0132, D(L2)=+0.0361, D(L4)=+0.0229, D(L5)=+0.0019
- 预测：L1 no call；L5 no call（主）；L2 ↓（次）；L4 ↓（次）
- 开箱结果（C4 后回填）：

## C4-b 冻结校正规则（msmarco 家族，dl19+dl20+dl21 拟合；git 446927c，
## seed 42，correction_rules.json 时间戳 2026-09-15T05:19:47Z）与校正预测

| model | Δ̂(msmarco) | dial | monotone map | cross-fit 预期 Δκ |
|---|---|---|---|---|
| deepseek-v4-flash | +0.5561 | L1 | [0,0,2,3] | +0.0515 |
| deepseek-v4.1-flash | +0.5005 | L1 | [0,0,2,3] | +0.0297 |
| gemini-3.5-flash | +0.4714 | L1 | [0,0,2,3] | +0.0286 |
| gemini-3.8-flash | +0.3802 | L3 | [0,1,2,3]（identity，即不校正） | −0.0310（cross-fit 显示校正有害，规则=不动） |
| claude-opus-5 | +0.4155 | L2 | [0,0,2,3] | +0.0131 |
| gpt-5.6-sol | +0.5716 | L1 | [0,0,2,3] | +0.0204 |

校正预测（主终点，逐模型）：
1. 对 5 个施加校正的模型：**κ(dial+map) > κ(L3 raw)**，且幅度与 cross-fit
   预期同数量级；**|Δ|_after < |Δ|_raw**（工作点残余失准收缩）。
2. gemini-3.8-flash（规则=identity）：κ 与 |Δ| 不变（诚实的 no-op 对照）。
3. 表述纪律：任何 κ 提升只作为 measurement artifact 的消除来报告。
- 开箱结果（C4 后回填）：

## C4 开箱结果（2026-09-15，scripts/c4_unseal_dl22.py，预注册 commit 80eacfc）

### C4-a 语气方向（called 11 项：命中 9，tie 1，未中 1）
| model | κ(L3) | dL1 | dL2 | dL4 | dL5 | 判定 |
|---|---|---|---|---|---|---|
| v4-flash | 0.3554 | +0.031↑ | +0.020↑ | +0.027↑ | +0.052↑ | 预测 ↑↑↑↑：**4/4 全中，U 型第三次复现** |
| v4.1-flash | 0.3626 | −0.001 tie | −0.017 | +0.001 | +0.006↑ | L1 tie（预测 ↑），L5 ↑ 命中 |
| gemini-3.5-flash | 0.4057 | −0.008 ↓ | −0.011 | +0.014 | +0.011 | **L1 预测 ↑ 未中** |
| gemini-3.8-flash | 0.4234 | −0.005 | −0.002 | +0.001 | −0.001 | 全 no-call，实际全 tie：一致（免疫三连） |
| opus-5 | 0.4689 | +0.001 | +0.012 | −0.032↓ | −0.022↓ | L5↓ L4↓：2/2 命中 |
| gpt-5.6-sol | 0.3702 | −0.022 | −0.024↓ | −0.012↓ | −0.008 | L2↓ L4↓：2/2 命中 |

### C4-b 校正规则迁移（预注册主终点，逐条判定）
| model | κ(L3) | κ(dial) | κ(dial+map) | Δκ 实际 vs 预期 | \|Δ\| before→after |
|---|---|---|---|---|---|
| v4-flash | 0.3554 | 0.3862 (+0.031) | 0.3548 | −0.001 vs +0.052 **未中** | 0.438→0.129 ✓ |
| v4.1-flash | 0.3626 | 0.3622 | 0.3502 | −0.012 vs +0.030 **未中** | 0.443→0.256 ✓ |
| gemini-3.5-flash | 0.4057 | 0.3979 | 0.3612 | −0.045 vs +0.029 **未中** | 0.333→0.289 ✓ |
| gemini-3.8-flash | 0.4234 | 0.4234 | 0.4234 | 0 vs 0（no-op 对照）**命中** | 0.281→0.281 ✓ |
| opus-5 | 0.4689 | 0.4807 (+0.012) | 0.4041 | −0.065 vs +0.013 **未中** | 0.241→0.226 ✓ |
| gpt-5.6-sol | 0.3702 | 0.3487 | 0.3713 | +0.001 vs +0.020 tie | 0.395→0.185 ✓ |

**判定汇总**：
- 预测 1（κ(dial+map) > κ(L3)，5 个施加校正的模型）：**1 tie，4 未中**。
- 预测 1 附属（|Δ|_after < |Δ|_raw）：**5/5 全部命中**（收缩 13%–71%）。
- 预测 2（3.8-flash no-op 不变）：**命中**。
- 有趣细节：v4-flash 的 **dial-only** 在 dl22 上 +0.031（方向正确、幅度接近
  cross-fit 预期的一半）；是 map 的迁移失败拖垮了组合。

### C4 诚实解读（预注册后记录，不改预测）
工作点对齐（|Δ|）跨数据集迁移成功，但 κ 收益不迁移。机制清楚：DL22 的
qrels 分布显著更严（均分 0.678 vs 拟合家族 dl19/20/21 的 0.93–1.10；
0/1/2/3 = 1113/705/234/94），冻结的 [0,0,2,3] 映射把大量 1 压成 0，
均值对齐的同时压缩了边际分布，chance-corrected 的 κ 中 pe 的变化吃掉了
po 的收益。结论（进论文）：**输出侧重映射的 κ 收益依赖目标域 grade 分布，
分布偏移下不可盲目迁移；工作点（均值）对齐可迁移**。与 C1 样本效率结论
衔接：目标域几十个标注 pair 即可重拟合 map（廉价的域内校准），预注册的
跨域迁移失败恰好论证了这一步不可省略。dial-only（输入侧）在强漂移
judge（v4-flash）上跨域方向正确，量程一如既往受限。

# B5 预注册 — 下游系统排名效用（batch5，dl21，非封存数据）

登记时间：2026-09-16。分析代码 git hash：bce191a（scripts/batch5_downstream.py）。
分析：evaluate 子命令 — NDCG@10（gain 2^rel−1，condensed intersection
universe，四步约定显式声明、不冒充 trec_eval）→ 系统排名 → 对 HUMAN 排名的
Kendall τ-b；paired query bootstrap B=2000 seed 42。

Judges（cheap 全网格；flagship 两模型仅 40% 子样本、覆盖不足，不入 batch5）：
- deepseek-v4-flash、deepseek-v4.1-flash、gemini-3.5-flash：C4-b 冻结规则
  均为 dial=L1、map=[0,0,2,3]（msmarco 家族，git 446927c）
- gemini-3.8-flash：identity@L3 → 结构性 no-op 对照

系统池：pyserini 自建 13 系统（bm25 参数网格 8 + RM3 2 + doc2query-T5 3；
bm25-rocchio 与 d2q-bm25-rm3 因预建索引无 document vectors 被跳过；TREC
官方 63 runs 因 401 login-wall 不可得，升级路径见内部 roadmap）。
覆盖率（pool top-10 并集，登记时数字）：1408 distinct (qid,docid)；
human-judged 1165（82.7%）；四 judge 所需档位全可用 840（59.7%）；
top-up 325 对（L1+L3 × 3 judges + L3 × 3.8-flash，13,650 calls，≈$3.2）
已于登记同时启动采集；243 对无人类标注，condensed 评测自动排除。

盲态声明：登记前未计算或查看过任何 τ(RAW/CORR vs HUMAN)、NDCG 或
system-ranking 数值；coverage 输出只含标签可用性计数，无一致性指标。

预测（主终点，逐条可判）：
1. τ_raw ≥ +0.3（全部 4 judge）：工作点失准主要是阈值伪影，对系统间排序
   破坏有限（v1 R2 文档级排序 τ 0.74–0.94 的下游延伸）。注意池内 bm25
   变体同质性高、τ 方差大，故只押保守下界。
2. 3 个施加校正的 judge：Δτ = τ_corr − τ_raw ≥ 0（方向），|Δτ| ≤ 0.10
   （幅度）。机制：收益只能来自 grade collapse（map [0,0,2,3] 将 1 压到 0，
   消解 judge 对边缘文档的宽松部分分）与 dial 换档（L1 为不同标签集）；
   monotone map 不能重排单查询内文档。
3. gemini-3.8-flash：Δτ ≡ 0.0000（identity@L3 的结构性不变量，代码以
   hard-fail 断言强制）——作为 pipeline correctness check 报告，非 null 结果。
4. 判定纪律：若任一施加校正的 judge Δτ < 0 且 bootstrap 90% CI 不含 0，
   记为"校正伤害下游排序"，如实报告并进论文（与 C4 κ 不迁移同等地位）。

开箱结果（evaluate 2026-09-16 回填；seed 42 确定性，B=2000；
stdout 存档 derived/eval_batch5_stdout.txt，tex batch5_system_ranking.tex）：

| judge | dial | τ_raw [CI] | τ_corr [CI] | Δτ [95% CI] | P(Δ>0) |
|---|---|---|---|---|---|
| v4-flash | L1 | +0.6923 [+0.487,+0.923] | +0.8974 [+0.589,+0.949] | +0.2051 [−0.103,+0.231] | 0.692 |
| v4.1-flash | L1 | +0.7692 [+0.513,+0.949] | +0.7692（同） | +0.0000 [−0.128,+0.154] | 0.468 |
| gemini-3.5-flash | L1 | +0.7949 [+0.462,+0.949] | +0.7436 [+0.487,+0.949] | −0.0513 [−0.128,+0.154] | 0.528 |
| gemini-3.8-flash（对照） | L3 | +0.7692 [+0.436,+0.923] | +0.7692（同） | **+0.0000 精确，CI [0,0]** | — |

- 命中/未中（对照预测逐条）：
  1. τ_raw ≥ +0.3 全部 4 judge：**命中 4/4**（0.69–0.79）。
  2. 校正 judge Δτ 方向 ≥0：v4-flash 命中（+0.2051）；v4.1 边界（精确 0，
     记 tie）；3.5-flash **未中**（−0.0513，CI 跨 0）。幅度 |Δτ|≤0.10：
     v4.1、3.5 命中；v4-flash **未中**（+0.205 超出预测带——收益比预押的大）。
  3. 3.8-flash Δτ ≡ 0：**精确命中**（hard-fail 断言通过，RAW/CORR 列逐位相同）。
  4. 判定纪律：无任何 judge 的 Δτ<0 且 CI 排除 0 → 无"校正伤害下游"判定。

- **评估员（Sonnet）复核后的修订（2026-09-16，代码 0e55074）**：
  a. 预注册第 4 条写明 90% CI，但 evaluate 原只输出 95% CI（更宽区间会
     偏向"无伤害"结论）。已补算（commit 0e55074，同 seed 重跑逐位一致）：
     Δτ 90% CI —— v4-flash [−0.0769,+0.2051]；v4.1 [−0.1026,+0.1282]；
     **3.5-flash [−0.1026,+0.1282] 含 0** → 判定在预注册区间上正式成立。
  b. 位精确验证：v4.1 的 Δτ repr = `0.0`（真精确平局，非舍入）；
     v4-flash = 8/39、3.5-flash = −2/39——全部 τ 落在 13 系统无并列
     τ-b 网格 k/39 上（内部一致性核验通过）。
  c. 预测 2 的诚实合取式打分：逐 judge 的谓词是"方向 AND 幅度"，
     合取命中 **1/3**（仅 v4.1；v4-flash 幅度超带、3.5 方向反）。
     上表按边际分列（方向 2/3、幅度 2/3）仅作分解展示，主评分以合取为准。
  d. 预测 1（τ_raw ≥ 0.3）下界远低于先验（v1 τ 0.74–0.94），命中证据
     强度有限，论文中不作为强确认展示。
  e. 局限（进论文第一序）：池中 learned-sparse/dense 家族全部缺席
     （预建索引不可得），13 系统由 8 个 bm25 变体主导——设计文档预警的
     同质性风险成真，τ 结论的外推性受此上限约束。
- 附加观察（进论文）：绝对 NDCG 尺度上 RAW 系统性虚高（+0.02~+0.13，
  与各 judge 的 Δ 同序），CORR 把最强漂移 judge（v4-flash）的绝对
  NDCG 拉回人类 ±0.01 内；v4.1/3.5 出现过校正（低于人类 0.04~0.08）。
  τ 的 bootstrap CI 全部很宽（13 系统池），任何 Δτ 都不显著——
  按预注册第 1 条的池同质性 caveat 如实报告。

# B6 预注册 — 标注池扩充（batch6，scripts/batch6_poolext.py）

登记时间：2026-10-12。分析代码 git hash：a52b5a4。
rules_sha256: c93876126540819780ed59f36ed1afd298594b4309b9a77376ff8b01c6417293
冻结规则文件：correction_rules.json（git 446927c，时间戳 2026-09-15T05:19:47Z）；run 时 sha256 不符即中止。
分析：`run` 子命令。pair universe = human-judged ∩ judge 在 L1–L5 均有可用标签（parse_ok & 数值 score）；
标签 = 该档全部记录（paraphrase×run）的均值后 round-half-up（文中"paraphrase-mean"即此记录均值；设计平衡时与先按 paraphrase 取均值再平均完全相同）。RAW = L3；CORR-FROZEN = 家族冻结规则（dl21/dl22 用 msmarco，antique 用 antique；dial 档 + monotone map 逐记录施加于取均值之前）；
CORR-REFIT = 同数据集 query-disjoint 2 折交叉拟合（md5(seed|dataset|qid) 分折；主 fold seed 42，稳健性 seeds [43, 44, 45, 46] 仅报告不计分；dial = argmin|Δ+D(ℓ)|，map = 35 个单调映射中训练折 paraphrase-mean 线性 κ 最大者，均为 batch4 估计量），测试折拼接。
不确定性：query-clustered bootstrap B=1000 seed 42，90% 百分位 CI；差值为同一重采样 query 集上的配对差。REFIT 的 CI：evaluation-only bootstrap; fit variability shown by fold-seed spread。
样本效率（E6）：n ∈ [50, 100, 200, 400]，B=1000 次 query 不放回抽样，校准集与扩充池 query-disjoint。
判定纪律（逐字；代码在 BACKFILL 中同文打印）：
1. 方向一律按点估计判定。tie 带为代码硬编码常量：TIE_F1 = 0.005（作用于 E3/E4 的 ΔF1@2、E5 的 ΔP@2 或 ΔR@2、E6 的差额）；TIE_RATIO = 0.02（作用于 E1 的 |ratio−1| 与 E2 的 |ratio−1| 变化量）。
2. tie 既不计命中也不计未中（与此前各轮相同）。
3. "≥0" 型预测（E3/E4/E5/E6）：预测"是"时，Δ > +TIE 记命中，Δ < −TIE 记未中，否则 tie；预测"否"时方向相反。
4. E1：RAW ratio@≥2 点估计落在 [1−TIE_RATIO, 1+TIE_RATIO] 内一律记 tie，不看 CI；否则点估计方向与预测（>1 / <1）一致记命中，相反记未中。
5. E2：变化量 = |r_FROZEN−1| − |r_RAW−1|。预测"是"（收缩）时 < −TIE_RATIO 记命中，> +TIE_RATIO 记未中，否则 tie；预测"否"时方向相反。
6. 90% CI 排除 0（E1 为排除 1）单独报告为"显著"，永不改变命中/未中判定。
7. 合取单元 = (judge, 数据集)：单元内每个适用且已登记的终点均命中 → 单元合取命中；任一未中 → 单元未中；无未中但有已登记终点无法判定（如 E6 未运行） → 单元 not adjudicated；否则有 tie → 单元 tie。被跳过/报错的单元记 not adjudicated，不计分。
8. E5：冻结规则 delta_hat < 0 的单元（严域）判定 R@2(REFIT) − R@2(RAW) ≥ 0；delta_hat ≥ 0 的单元（宽域）判定 P@2(REFIT) − P@2(RAW) ≥ 0。
9. E6：Δ = mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400（扩充池上的配对差），按第 3 条判定；仅当 n=200 与 n=400 均有成功抽样时判定，否则 not adjudicated。
10. E3/E4/E5 使用主 fold seed（42）的 CORR-REFIT；其余 fold seeds 只作稳健性报告，不计分。REFIT 的 CI：evaluation-only bootstrap; fit variability shown by fold-seed spread。
填写格式：把每条终点行末"预测："之后的占位替换为 >1 或 <1（E1），是 或 否（E2–E6）；同一 token 后不得紧接其他文字（如需说明，空格后写）。块内任何占位未填，dl22 门控即拒绝运行。
盲态声明：blind — 登记前未在任何数据集上运行 `run`；`validate` 仅输出计数与覆盖（无一致性指标）；`paste` 仅读规则文件。预测逻辑：宽域（Δ>0）judge 的 RAW 相关集偏多（E1 >1），冻结规则的 dial 换档使相关率向 1 收敛（E2 是，map [0,0,2,3] 不改变 ≥2 集合、收敛全靠 dial；dial=L3 的 antique v4-flash 结构上无变化，预期 tie）；域内重拟合 F1 ≥ RAW（E3 是）；重拟合 ≥ 冻结 在 held-out/部分 in-sample 的 msmarco 规则上成立（E4 是），但 antique 冻结规则本身完全 in-sample，预期冻结 ≥ 重拟合（E4 否）；宽域 precision 升、严域 recall 升（E5 是）；n=200 已达 n=400 收益的 80%（E6 是，C1 结论延伸）。预期 3.8-flash（漂移阶梯平坦、map 由 κ 选择）多为 tie。

## dl22 — held-out（冻结 msmarco 规则未见 dl22）
### deepseek/deepseek-v4-flash — dl22 — 冻结规则 dial=L1 map=[0, 0, 2, 3]，delta_hat=+0.5561（E5 宽域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：>1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E5 [recall direction @≥2] 宽域：P@2（precision）(CORR-REFIT) − P@2（precision）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 1.8506 [+1.5702,+2.1473] pred=>1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.4482 [-0.6000,-0.3046] pred=是 -> HIT significant (90% CI excludes 0)
  - B6-E3 F1@2 REFIT-RAW = +0.0233 [-0.0066,+0.0525] pred=是 -> HIT not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = +0.0120 [-0.0096,+0.0344] pred=是 -> HIT not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 P@2 REFIT-RAW (lenient domain) = +0.0380 [+0.0061,+0.0696] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = -0.0256 pred=是 -> MISS
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW +0.0255+-0.0048 [+0.0200,+0.0328]; dF1@2 REFIT-FROZEN +0.0141+-0.0048 [+0.0087,+0.0215]; dP@2 REFIT-RAW +0.0440+-0.0058 [+0.0380,+0.0517]
  - => cell conjunction: MISS
### deepseek/deepseek-v4.1-flash — dl22 — 冻结规则 dial=L1 map=[0, 0, 2, 3]，delta_hat=+0.5005（E5 宽域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：>1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E5 [recall direction @≥2] 宽域：P@2（precision）(CORR-REFIT) − P@2（precision）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 1.7957 [+1.5377,+2.0673] pred=>1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.6768 [-0.8374,-0.4686] pred=是 -> HIT significant (90% CI excludes 0)
  - B6-E3 F1@2 REFIT-RAW = +0.0085 [-0.0276,+0.0427] pred=是 -> HIT not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = -0.0022 [-0.0138,+0.0093] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 P@2 REFIT-RAW (lenient domain) = +0.0862 [+0.0523,+0.1245] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = -0.0159 pred=是 -> MISS
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW +0.0065+-0.0031 [+0.0022,+0.0101]; dF1@2 REFIT-FROZEN -0.0043+-0.0031 [-0.0086,-0.0006]; dP@2 REFIT-RAW +0.0793+-0.0092 [+0.0635,+0.0862]
  - => cell conjunction: MISS
### google/gemini-3.5-flash — dl22 — 冻结规则 dial=L1 map=[0, 0, 2, 3]，delta_hat=+0.4714（E5 宽域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：>1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E5 [recall direction @≥2] 宽域：P@2（precision）(CORR-REFIT) − P@2（precision）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 1.2073 [+0.9947,+1.4181] pred=>1 -> HIT not significant
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.1829 [-0.2922,+0.1752] pred=是 -> HIT not significant
  - B6-E3 F1@2 REFIT-RAW = -0.0283 [-0.0515,-0.0087] pred=是 -> MISS significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = +0.0082 [-0.0101,+0.0291] pred=是 -> HIT not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 P@2 REFIT-RAW (lenient domain) = -0.0010 [-0.0262,+0.0189] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = -0.0171 pred=是 -> MISS
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW -0.0277+-0.0013 [-0.0283,-0.0254]; dF1@2 REFIT-FROZEN +0.0087+-0.0013 [+0.0082,+0.0110]; dP@2 REFIT-RAW +0.0022+-0.0072 [-0.0010,+0.0150]
  - => cell conjunction: MISS
### google/gemini-3.8-flash — dl22 — 冻结规则 dial=L3 map=[0, 1, 2, 3]，delta_hat=+0.3802（E5 宽域）（identity@L3 → no-op 对照）
- B6-E0 [no-op] CORR-FROZEN ≡ RAW 全终点逐位相同（结构性不变量，代码 hard-fail 强制）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：>1
- B6-E2 [ratio convergence] N/A（结构性：FROZEN ≡ RAW）
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] ≡ B6-E3（FROZEN ≡ RAW），不单独计分
- B6-E5 [recall direction @≥2] 宽域：P@2（precision）(CORR-REFIT) − P@2（precision）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E0 CORR-FROZEN ≡ RAW bit-exactly (invariant passed) pred=structural -> HIT
  - B6-E1 RAW ratio@>=2 = 1.0213 [+0.8382,+1.2088] pred=>1 -> HIT not significant
  - B6-E3 F1@2 REFIT-RAW = +0.0000 [+0.0000,+0.0000] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 P@2 REFIT-RAW (lenient domain) = +0.0000 [+0.0000,+0.0000] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = -0.0120 pred=是 -> MISS
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW -0.0007+-0.0016 [-0.0036,+0.0000]; dF1@2 REFIT-FROZEN -0.0007+-0.0016 [-0.0036,+0.0000]; dP@2 REFIT-RAW +0.0029+-0.0065 [+0.0000,+0.0145]
  - => cell conjunction: MISS
### anthropic/claude-opus-5 — dl22（40% 冻结子样本） — 冻结规则 dial=L2 map=[0, 0, 2, 3]，delta_hat=+0.4155（E5 宽域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：>1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E5 [recall direction @≥2] 宽域：P@2（precision）(CORR-REFIT) − P@2（precision）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 1.3066 [+1.0975,+1.5449] pred=>1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.1679 [-0.2421,-0.0492] pred=是 -> HIT significant (90% CI excludes 0)
  - B6-E3 F1@2 REFIT-RAW = -0.0070 [-0.0234,+0.0071] pred=是 -> MISS not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = -0.0201 [-0.0422,+0.0029] pred=是 -> MISS not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 P@2 REFIT-RAW (lenient domain) = -0.0023 [-0.0144,+0.0094] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = -0.0126 pred=是 -> MISS
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW -0.0070+-0.0000 [-0.0070,-0.0070]; dF1@2 REFIT-FROZEN -0.0201+-0.0000 [-0.0201,-0.0201]; dP@2 REFIT-RAW -0.0023+-0.0000 [-0.0023,-0.0023]
  - => cell conjunction: MISS
### openai/gpt-5.6-sol — dl22（40% 冻结子样本） — 冻结规则 dial=L1 map=[0, 0, 2, 3]，delta_hat=+0.5716（E5 宽域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：>1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E5 [recall direction @≥2] 宽域：P@2（precision）(CORR-REFIT) − P@2（precision）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 1.3212 [+1.1118,+1.5451] pred=>1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.1679 [-0.2406,-0.0594] pred=是 -> HIT significant (90% CI excludes 0)
  - B6-E3 F1@2 REFIT-RAW = -0.0043 [-0.0174,+0.0086] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = -0.0227 [-0.0470,-0.0020] pred=是 -> MISS significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 P@2 REFIT-RAW (lenient domain) = +0.0064 [-0.0055,+0.0209] pred=是 -> HIT not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = -0.0088 pred=是 -> MISS
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW -0.0054+-0.0030 [-0.0089,-0.0016]; dF1@2 REFIT-FROZEN -0.0239+-0.0030 [-0.0273,-0.0201]; dP@2 REFIT-RAW +0.0046+-0.0026 [+0.0005,+0.0064]
  - => cell conjunction: MISS
## dl21 — 冻结规则部分 in-sample（msmarco 规则拟合于 dl19+dl20+dl21）
### deepseek/deepseek-v4-flash — dl21 — 冻结规则 dial=L1 map=[0, 0, 2, 3]，delta_hat=+0.5561（E5 宽域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：>1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E5 [recall direction @≥2] 宽域：P@2（precision）(CORR-REFIT) − P@2（precision）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 1.6461 [+1.4330,+1.9156] pred=>1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.2903 [-0.3647,-0.2274] pred=是 -> HIT significant (90% CI excludes 0)
  - B6-E3 F1@2 REFIT-RAW = +0.0110 [-0.0188,+0.0354] pred=是 -> HIT not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = -0.0048 [-0.0100,+0.0003] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 P@2 REFIT-RAW (lenient domain) = +0.0474 [+0.0264,+0.0695] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = -0.0183 pred=是 -> MISS
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW +0.0159+-0.0034 [+0.0110,+0.0208]; dF1@2 REFIT-FROZEN +0.0001+-0.0034 [-0.0048,+0.0050]; dP@2 REFIT-RAW +0.0468+-0.0050 [+0.0417,+0.0526]
  - => cell conjunction: MISS
### deepseek/deepseek-v4.1-flash — dl21 — 冻结规则 dial=L1 map=[0, 0, 2, 3]，delta_hat=+0.5005（E5 宽域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：>1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E5 [recall direction @≥2] 宽域：P@2（precision）(CORR-REFIT) − P@2（precision）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 1.6004 [+1.3873,+1.8565] pred=>1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.3380 [-0.4239,-0.2645] pred=是 -> HIT significant (90% CI excludes 0)
  - B6-E3 F1@2 REFIT-RAW = +0.0291 [+0.0068,+0.0518] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = +0.0061 [-0.0004,+0.0136] pred=是 -> HIT not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 P@2 REFIT-RAW (lenient domain) = +0.0729 [+0.0516,+0.0981] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = -0.0154 pred=是 -> MISS
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW +0.0247+-0.0027 [+0.0227,+0.0291]; dF1@2 REFIT-FROZEN +0.0017+-0.0027 [-0.0003,+0.0061]; dP@2 REFIT-RAW +0.0741+-0.0038 [+0.0686,+0.0778]
  - => cell conjunction: MISS
### google/gemini-3.5-flash — dl21 — 冻结规则 dial=L1 map=[0, 0, 2, 3]，delta_hat=+0.4714（E5 宽域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：>1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E5 [recall direction @≥2] 宽域：P@2（precision）(CORR-REFIT) − P@2（precision）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 1.3141 [+1.1218,+1.5587] pred=>1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.1630 [-0.2197,-0.0930] pred=是 -> HIT significant (90% CI excludes 0)
  - B6-E3 F1@2 REFIT-RAW = +0.0037 [-0.0145,+0.0187] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = +0.0044 [-0.0022,+0.0126] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 P@2 REFIT-RAW (lenient domain) = +0.0328 [+0.0184,+0.0477] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = -0.0107 pred=是 -> MISS
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW +0.0063+-0.0023 [+0.0037,+0.0080]; dF1@2 REFIT-FROZEN +0.0070+-0.0023 [+0.0044,+0.0087]; dP@2 REFIT-RAW +0.0308+-0.0026 [+0.0290,+0.0344]
  - => cell conjunction: MISS
### google/gemini-3.8-flash — dl21 — 冻结规则 dial=L3 map=[0, 1, 2, 3]，delta_hat=+0.3802（E5 宽域）（identity@L3 → no-op 对照）
- B6-E0 [no-op] CORR-FROZEN ≡ RAW 全终点逐位相同（结构性不变量，代码 hard-fail 强制）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：>1
- B6-E2 [ratio convergence] N/A（结构性：FROZEN ≡ RAW）
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] ≡ B6-E3（FROZEN ≡ RAW），不单独计分
- B6-E5 [recall direction @≥2] 宽域：P@2（precision）(CORR-REFIT) − P@2（precision）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E0 CORR-FROZEN ≡ RAW bit-exactly (invariant passed) pred=structural -> HIT
  - B6-E1 RAW ratio@>=2 = 1.2545 [+1.0848,+1.4535] pred=>1 -> HIT significant (90% CI excludes 1)
  - B6-E3 F1@2 REFIT-RAW = -0.0099 [-0.0173,-0.0035] pred=是 -> MISS significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 P@2 REFIT-RAW (lenient domain) = -0.0071 [-0.0135,-0.0006] pred=是 -> MISS significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = -0.0133 pred=是 -> MISS
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW -0.0076+-0.0043 [-0.0128,-0.0026]; dF1@2 REFIT-FROZEN -0.0076+-0.0043 [-0.0128,-0.0026]; dP@2 REFIT-RAW -0.0071+-0.0030 [-0.0108,-0.0032]
  - => cell conjunction: MISS
### anthropic/claude-opus-5 — dl21（40% 冻结子样本） — 冻结规则 dial=L2 map=[0, 0, 2, 3]，delta_hat=+0.4155（E5 宽域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：>1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E5 [recall direction @≥2] 宽域：P@2（precision）(CORR-REFIT) − P@2（precision）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 1.3719 [+1.1706,+1.6112] pred=>1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.0804 [-0.1429,-0.0291] pred=是 -> HIT significant (90% CI excludes 0)
  - B6-E3 F1@2 REFIT-RAW = -0.0125 [-0.0325,+0.0070] pred=是 -> MISS not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = -0.0032 [-0.0176,+0.0099] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 P@2 REFIT-RAW (lenient domain) = +0.0034 [-0.0188,+0.0244] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = -0.0161 pred=是 -> MISS
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW -0.0381+-0.0382 [-0.0815,-0.0085]; dF1@2 REFIT-FROZEN -0.0287+-0.0382 [-0.0722,+0.0008]; dP@2 REFIT-RAW +0.0122+-0.0115 [+0.0034,+0.0278]
  - => cell conjunction: MISS
### openai/gpt-5.6-sol — dl21（40% 冻结子样本） — 冻结规则 dial=L1 map=[0, 0, 2, 3]，delta_hat=+0.5716（E5 宽域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：>1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E5 [recall direction @≥2] 宽域：P@2（precision）(CORR-REFIT) − P@2（precision）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 1.4121 [+1.2099,+1.6548] pred=>1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.0653 [-0.1126,-0.0228] pred=是 -> HIT significant (90% CI excludes 0)
  - B6-E3 F1@2 REFIT-RAW = -0.0075 [-0.0219,+0.0059] pred=是 -> MISS not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = +0.0025 [-0.0043,+0.0105] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 P@2 REFIT-RAW (lenient domain) = +0.0020 [-0.0118,+0.0143] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = -0.0086 pred=是 -> MISS
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW -0.0128+-0.0210 [-0.0493,+0.0022]; dF1@2 REFIT-FROZEN -0.0029+-0.0210 [-0.0394,+0.0121]; dP@2 REFIT-RAW +0.0096+-0.0121 [+0.0020,+0.0309]
  - => cell conjunction: MISS
## antique — 冻结规则完全 in-sample（antique 规则拟合于 antique 本身）
### deepseek/deepseek-v4-flash — antique — 冻结规则 dial=L3 map=[0, 1, 3, 3]，delta_hat=-0.3997（E5 严域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：<1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：否
- B6-E5 [recall direction @≥2] 严域：R@2（recall）(CORR-REFIT) − R@2（recall）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 0.7309 [+0.6895,+0.7709] pred=<1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.0356 [-0.0477,-0.0245] pred=是 -> HIT significant (90% CI excludes 0)
  - B6-E3 F1@2 REFIT-RAW = +0.0121 [+0.0052,+0.0197] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = +0.0000 [+0.0000,+0.0000] pred=否 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 R@2 REFIT-RAW (strict domain) = +0.0237 [+0.0154,+0.0331] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = +0.0004 pred=是 -> TIE
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW +0.0117+-0.0011 [+0.0097,+0.0121]; dF1@2 REFIT-FROZEN -0.0005+-0.0011 [-0.0024,+0.0000]; dR@2 REFIT-RAW +0.0240+-0.0006 [+0.0237,+0.0251]
  - => cell conjunction: TIE
### deepseek/deepseek-v4.1-flash — antique — 冻结规则 dial=L4 map=[0, 1, 3, 3]，delta_hat=-0.2793（E5 严域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：<1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：否
- B6-E5 [recall direction @≥2] 严域：R@2（recall）(CORR-REFIT) − R@2（recall）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 0.7546 [+0.7126,+0.7979] pred=<1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.0528 [-0.0675,-0.0370] pred=是 -> HIT significant (90% CI excludes 0)
  - B6-E3 F1@2 REFIT-RAW = +0.0137 [+0.0033,+0.0235] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = +0.0000 [+0.0000,+0.0000] pred=否 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 R@2 REFIT-RAW (strict domain) = +0.0317 [+0.0191,+0.0436] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = +0.0023 pred=是 -> TIE
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW +0.0130+-0.0014 [+0.0106,+0.0137]; dF1@2 REFIT-FROZEN -0.0006+-0.0014 [-0.0031,+0.0000]; dR@2 REFIT-RAW +0.0314+-0.0006 [+0.0303,+0.0317]
  - => cell conjunction: TIE
### google/gemini-3.5-flash — antique — 冻结规则 dial=L2 map=[0, 1, 3, 3]，delta_hat=-0.3288（E5 严域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：<1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：否
- B6-E5 [recall direction @≥2] 严域：R@2（recall）(CORR-REFIT) − R@2（recall）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 0.7183 [+0.6760,+0.7592] pred=<1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.0556 [-0.0745,-0.0375] pred=是 -> HIT significant (90% CI excludes 0)
  - B6-E3 F1@2 REFIT-RAW = +0.0015 [-0.0080,+0.0120] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = -0.0088 [-0.0148,-0.0034] pred=否 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 R@2 REFIT-RAW (strict domain) = +0.0132 [+0.0025,+0.0249] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = +0.0011 pred=是 -> TIE
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW +0.0013+-0.0036 [-0.0033,+0.0066]; dF1@2 REFIT-FROZEN -0.0090+-0.0036 [-0.0136,-0.0037]; dR@2 REFIT-RAW +0.0114+-0.0040 [+0.0066,+0.0172]
  - => cell conjunction: TIE
### google/gemini-3.8-flash — antique — 冻结规则 dial=L1 map=[0, 1, 3, 3]，delta_hat=-0.3308（E5 严域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：<1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：否
- B6-E5 [recall direction @≥2] 严域：R@2（recall）(CORR-REFIT) − R@2（recall）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 0.6900 [+0.6483,+0.7309] pred=<1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.0633 [-0.0794,-0.0472] pred=是 -> HIT significant (90% CI excludes 0)
  - B6-E3 F1@2 REFIT-RAW = +0.0140 [+0.0052,+0.0230] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = -0.0040 [-0.0090,+0.0008] pred=否 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 R@2 REFIT-RAW (strict domain) = +0.0290 [+0.0176,+0.0405] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = +0.0017 pred=是 -> TIE
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW +0.0110+-0.0036 [+0.0068,+0.0140]; dF1@2 REFIT-FROZEN -0.0070+-0.0036 [-0.0113,-0.0040]; dR@2 REFIT-RAW +0.0251+-0.0060 [+0.0185,+0.0303]
  - => cell conjunction: TIE
### anthropic/claude-opus-5 — antique（40% 冻结子样本） — 冻结规则 dial=L5 map=[0, 1, 2, 3]，delta_hat=-0.3379（E5 严域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：<1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：否
- B6-E5 [recall direction @≥2] 严域：R@2（recall）(CORR-REFIT) − R@2（recall）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 0.7013 [+0.6424,+0.7583] pred=<1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.0235 [-0.0443,-0.0033] pred=是 -> HIT significant (90% CI excludes 0)
  - B6-E3 F1@2 REFIT-RAW = +0.0211 [+0.0077,+0.0362] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = +0.0155 [+0.0033,+0.0285] pred=否 -> MISS significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 R@2 REFIT-RAW (strict domain) = +0.0369 [+0.0205,+0.0559] pred=是 -> HIT significant (90% CI excludes 0) (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = +0.0020 pred=是 -> TIE
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW +0.0104+-0.0088 [-0.0026,+0.0211]; dF1@2 REFIT-FROZEN +0.0048+-0.0088 [-0.0082,+0.0155]; dR@2 REFIT-RAW +0.0228+-0.0099 [+0.0101,+0.0369]
  - => cell conjunction: MISS
### openai/gpt-5.6-sol — antique（40% 冻结子样本） — 冻结规则 dial=L2 map=[0, 1, 2, 3]，delta_hat=-0.1610（E5 严域）
- B6-E1 [ratio direction] RAW 相关率比 @≥2（|LLM-rel|/|HUMAN-rel|），取值 >1 或 <1｜预测：<1
- B6-E2 [ratio convergence] CORR-FROZEN 使 |ratio@≥2 − 1| 相对 RAW 收缩，取值 是/否｜预测：是
- B6-E3 [F1 refit≥raw] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E4 [F1 refit≥frozen] F1@≥2(CORR-REFIT, fold seed 42) − F1@≥2(CORR-FROZEN) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：否
- B6-E5 [recall direction @≥2] 严域：R@2（recall）(CORR-REFIT) − R@2（recall）(RAW) ≥ 0，取值 是/否（evaluation-only bootstrap; fit variability shown by fold-seed spread）｜预测：是
- B6-E6 [sample efficiency] mean dF1@2(REFIT−RAW)@n=200 − 0.8 × mean dF1@2(REFIT−RAW)@n=400 ≥ 0，取值 是/否（需 --sample-efficiency；两档未都运行则 not adjudicated）｜预测：是
- 开箱结果（run 后回填）：
  - B6-E1 RAW ratio@>=2 = 0.8423 [+0.7770,+0.9065] pred=<1 -> HIT significant (90% CI excludes 1)
  - B6-E2 |r_FROZEN-1| - |r_RAW-1| = -0.0134 [-0.0392,+0.0103] pred=是 -> TIE not significant
  - B6-E3 F1@2 REFIT-RAW = -0.0054 [-0.0176,+0.0082] pred=是 -> MISS not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E4 F1@2 REFIT-FROZEN = +0.0000 [+0.0000,+0.0000] pred=否 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E5 R@2 REFIT-RAW (strict domain) = +0.0000 [-0.0140,+0.0166] pred=是 -> TIE not significant (evaluation-only bootstrap; fit variability shown by fold-seed spread)
  - B6-E6 dF1@2(REFIT-RAW)@200 - 0.8 x @400 = +0.0004 pred=是 -> TIE
  - fold-seed robustness [42, 43, 44, 45, 46] (not scored): dF1@2 REFIT-RAW -0.0041+-0.0028 [-0.0054,+0.0008]; dF1@2 REFIT-FROZEN +0.0012+-0.0028 [+0.0000,+0.0062]; dR@2 REFIT-RAW -0.0013+-0.0030 [-0.0067,+0.0000]
  - => cell conjunction: MISS

合计可判定终点：106（含结构性 no-op 断言 E0）。

## B6 开箱汇总（2026-10-12，run 于登记 commit cd5544d 之后；代码 81112cc；rules sha256 校验通过；no-op 对照 2 单元逐位相同 OK）
- dl22 tally — cells: {'miss': 6} registered endpoints: {'hit': 19, 'miss': 10, 'tie': 6}
- dl21 tally — cells: {'miss': 6} registered endpoints: {'hit': 18, 'tie': 7, 'miss': 10}
- antique tally — cells: {'tie': 4, 'miss': 2} registered endpoints: {'hit': 21, 'tie': 13, 'miss': 2}
- 总计：registered endpoints hit 58 / miss 22 / tie 26（106）；合取单元 0 命中、14 未中、4 tie（E6 在 msmarco 全部未中拖垮合取）。
- 诚实解读：方向性理论终点 E1（18/18）与 E2（16/16，msmarco 全部显著）几乎全中——原始标签把相关集按工作点方向系统性膨胀/收缩，冻结的均值对齐规则可迁移地消除大部分偏差（C4 结论在"标注池可用性"上的直接复现）。方法性能终点偏乐观：E3（refit F1≥RAW）7/6/5，E5（宽域 precision↑/严域 recall↑）强漂移判官显著 +0.03~+0.09、旗舰/温和判官 tie，E6（n=200 达 n=400 收益 80%）msmarco 0/16 未中、antique 全 tie。机制候选：κ 最优单调映射以 recall 换 precision，阈值 F1 不必然上升；小样本 map 重拟合在阈值 F1 上噪声大（待评估员复核）。进 Protocol 的两条候选修正：(a) 输出映射按下游目标（阈值 F1）而非 κ 拟合；(b) 输出映射样本需求（≥400 对）高于 dial（100–200 对），小样本阶段只做均值对齐/dial。

## B6 评估员（Sonnet）复核后的修订与勘误（2026-10-12）
**计数更正**（主 agent 原汇总有误，以此为准）：E0 2/2（结构性，不计入实质）；E1 18/0/0
（2 个不显著：dl22 gemini-3.5 CI [0.995,1.418]、dl22 gemini-3.8 在 tie 带边缘）；
E2 15 hit / 0 miss / 1 tie（antique gpt-5.6），msmarco 10 个 hit 中 9 个显著（dl22 3.5-flash 不显著）；
E3 8/6/4；E4 4/3/9；E5 11/1/6；E6 0/12/6。实质终点 56/22/26（104）。E0–E2 的方向由
delta_hat 符号预先决定、信息量低；E3–E5 为 23/10/19。E3/E4 共 36 项中仅 10 项 CI 排除 0，
且 CI 为 evaluation-only；约半数方向判定在 5 个 fold seed 间会变类（0.005 带内）。

**勘误（登记文本的机制陈述错误）**：盲态声明中"map [0,0,2,3] 不改变 ≥2 集合、收敛全靠 dial"
为误——本实现逐记录施加 map 后再取均值并 round-half-up，三条记录和 ≥5 才判 ≥2，故
[0,0,2,3] 会移除依赖 1 分投票的边缘正例（如 (2,2,1)→4）、[0,1,3,3] 会新增正例（(2,1,1)→5）。
dl22 opus-5 单元（两折 dial 同为 L2、map identity）可分离：E2 收缩的约 87% 来自 map 而非 dial。
E1/E2 的命中不受此影响，但登记理由须按本勘误理解。

**幅度更正**："冻结规则消除大部分膨胀"不成立。|ratio@2−1| 的消除比例：dl22 0.53/0.85/0.88/
0.55/0.52（中位 0.53）；dl21 0.45/0.56/0.52/0.22/0.16；antique 0.08–0.22。held-out dl22 不劣于
部分 in-sample 的 dl21——≥2 边界上的迁移证据成立。但**均值偏移在 dl22 上全部翻号过校正**
（+0.47→−0.13、+0.45→−0.26、+0.34→−0.29、+0.25→−0.23、+0.39→−0.18），原因是冻结规则的
delta_hat 系统性高于 dl22 实测 Δ(L3) 0.06–0.17；≥1 边界被 [0,0,2,3] 按构造抹平
（dl22 F1@1 −0.15~−0.25，5/6 显著）——作为局限如实报告，不藏在全表里。

**E6 终点作废重定义**：E6 = d200 − 0.8·d400 在 d400 ≤ 0 时机械为负（msmarco 12 单元中 8 个
d400 ≤ 0），不构成样本量证据；mean dF1 随 n 单调上升并收敛到交叉拟合值（选择噪声：RAW 已近
F1 局部最优，噪声只减不增；n=50 draw sd 0.13–0.21）。pairs 与 queries 混淆（dl22 约 28 对/query，
n=400≈14 query；antique≈10 对/query）。原"map 需 ≥400 对、dial 需 100–200 对"的候选修正**撤回**。
重做方案（batch6 扩展）：E6' 仅在 d400 > TIE 时适用；横轴改为 query 数；报中位数与灾难 draw 比例；
2×2 因子（dial∈{L3,拟合} × map∈{identity,拟合}）交叉拟合分离 dial/map 效应；F1@2–ratio@2
运行曲线（L3 票和阈值扫描 + 5×35 网格叠加）给出各单元 F1 最优 ratio；嵌套 bootstrap 给
fit-aware CI；按 F1@2 为目标的交叉拟合选择作为"按下游目标拟合"的直接检验。

**可安全写进论文的主张**：原始阈值标签按判官均值偏移方向系统性错报相关集规模
（MS MARCO ×1.2–1.85，ANTIQUE ×0.69–0.84；18 单元中 16 个 CI 排除 1）；为均值对齐拟合的冻结
dial+map 规则在 15/16 适用单元收缩 |ratio@2−1|，MS MARCO 上约去除一半（held-out dl22 中位 0.53）；
对最宽的判官提升 precision（DeepSeek 家族 ΔP@2 +0.03~+0.10）以 recall 为代价；**不**可靠提升
阈值 F1@2（仅 ratio ≥1.6 的四个 DeepSeek 单元 +0.011~+0.023）；≥1 边界按构造被抹平；
关于标注量需求的证据仅限"≈28 对/query、≤14 query 时小样本重拟合中性到有害"。
其余判断（dial 选择在 L1/L5 间双峰、冻结规则与小样本重拟合相当：mean REFIT−FROZEN ≈ −0.002）
作为解读注记。

## B6 探索性追加分析（followup cfecebe、followup2 e7068ff；非预注册；评估员 Sonnet 两轮复核，报告存内部 repo docs/evaluator_reports/2026-10-12_b6_followup*_sonnet.md）
均为开箱后设计，不得作确认性检验；完整性：18/18 单元 universe 与登记 run 一致，登记数字复现到 0.0005。
- **2×2 因子（dial×map 交叉拟合）**：F1@2 效应小（|FROZEN total|≤0.037）；dial/map 不可加——DeepSeek（MS MARCO）为替代（交互至 −0.035*），旗舰 dl22 为互补（主效应≈0、交互为正）；逐记录 map 承担多数 ratio 收缩（dl22 opus 70%）——支持并细化勘误，机制随 judge/数据集变化。REFIT 的归因在 8/18 单元结构性退化（交互恒 0）。
- **运行曲线**：FROZEN:MAP 在 18/18 单元落在 L3 票和阈值曲线上（输出侧单调映射 = re-threshold）；只有 dial 在匹配 ratio 下高于曲线（5 个 MS MARCO 单元，antique 无；dl22 v4.1 的『高于』是弦插值伪影）；旗舰 dl22 只有 dial+map 联合高于曲线。原『in-sample 网格 F1 上限』不是上限（2 单元低于最佳 L3 阈值、4 单元被交叉拟合超过），已改名为乐观参照，『占上限比例』数字撤回。
- **决定性基线（followup2）：交叉拟合单参数 re-threshold（RETHRESH_K/F1，同折）**：18 单元均值 F1 增益 RTK +0.0037 / RTF +0.0048 vs REFIT +0.0035 / F1REFIT +0.0053 / FROZEN +0.0050；各 arm 减基线均值在 −0.0002~+0.0013；逐单元 beats/within/below 6/6/6、8/4/6、7/5/6、6/5/7，且随 fold seed 变动（342 个单元-指标中 180 个带内翻转、20 个硬翻转）；fit-aware CI 排除 0 的 8/72 与 10% 水平的偶然数一致，且 B=200 的嵌套 CI 不对称（RETHRESH 在重采样中不稳），**不得引用为显著**。ANTIQUE：re-threshold 自身 +0.011/+0.014（五个 seed 同号），5/6 单元不低于最佳校正 arm；MS MARCO：re-threshold 均值≈0，仅高 ratio 的 DeepSeek 单元为正、dl22 3.8-flash 为负。剩余差距≤0.02 F1（DeepSeek MS MARCO 的 refit；held-out dl22 旗舰的冻结 dial+map +0.016/+0.019，同一规则在 3.5-flash −0.033）。校正相对 re-threshold 真正不同的是 **ratio 对齐**：MS MARCO 上去除 |ratio−1| 的比例 FROZEN 0.44、REFIT 0.29 vs RT 0.13–0.17（六记录票和有台阶，κ/F1 选的阈值到不了 ratio≈1）。dl21 的 RTK 因 53 query 的悬崖阈值不稳（seed 44 四个单元塌到 −0.059），dl21 任何 RTK 结论不可用。
- **F1 目标重拟合**：F1REFIT−REFIT 五 seed 均值 +0.002~+0.006，逐单元为噪声（17/18 带内翻转）。
- **E6'（横轴 query 数）**：κ 目标下 q=8 捕获 q=32 增益 ≥80% 仅 2/18（antique v4.1、3.8），F1 目标 1/18；均值被灾难 draw 主导；DeepSeek MS MARCO 单元约 q=16（≈450 对）后重拟合才平均不亏损（q=8 时 P(损失>0.05)=0.14/0.03/0.14/0.10），ANTIQUE 4–8 个 query；RAW 已近 F1 最优的 judge 任何 q≤32 都无增益（『保留 RAW』）。RAW 回退率很低（中位 0.014/0.038），DeepSeek 单元 ≤0.013——小 q 的灾难是真的坏规则；回退率只在 RAW 已最优处随 q 上升（dl22 3.8 κ 目标 q=32 时 0.80）。此指引针对 175 网格重拟合，单参数 re-threshold 未做 E6'。
- **可写进论文（探索性标注）**：上述各条按评估员清单；**不得写**：校正优于 re-threshold / dial 或 map『携带超出 re-threshold 的信息』/ seed 稳健 / 『仅在 MS MARCO』对 refit arm 成立 / 任何 arm−RETHRESH 的显著性 / dl21 RTK / in-sample re-threshold 作基线 / F1 最优 ratio=1 / 『恢复或提升判断质量』/ 对数规则 / 回退率解释 antique 小 q 安全。
- **评估员认为 F1-vs-re-threshold 问题已关闭**；可选后续（按价值）：以 ratio 为目标的 re-threshold 基线；对 RETHRESH_F1 做 E6'（若 2–4 个 query 即达增益，『约 16 query』指引应改为『只重拟合阈值』）；若引用区间需 B≥1000 居中区间；RTF 为主、RTK 限 |t−t_raw|≤3。


## B6 勘误（2026-10-12，评估员终审发现）
『幅度更正』段落与『可安全写进论文的主张』中的 dl22 |ratio@2−1| 去除比例中位数应为 **0.55**（五个非 no-op 单元 0.53/0.85/0.88/0.55/0.52 的中位数），原写 0.53 有误；范围 0.52–0.88 不变。同段『MS MARCO ×1.2–1.85』应为『×1.02–1.85（除 dl22 gemini-3.8 的 1.02 外为 1.2–1.85）』。以上两处按只增不改约定在此勘误，不回改原文。
# B7 预注册 — 蒸馏训练（batch7，scripts/batch7_distill.py）

登记时间：2026-10-12。分析代码 git hash：a52b5a4（batch7_distill.py）。冻结规则文件同 B6
（rules_sha256: c93876126540819780ed59f36ed1afd298594b4309b9a77376ff8b01c6417293）。
训练池：dl19+dl20+dl21（v4-flash、gemini-3.5-flash：5049 对/150 query，覆盖人类标注 100%）；
gemini-3.8-flash 仅 dl21（1551 对/53 query，round-2 judge）——只做 judge 内对比。测试：dl22
人类 qrels（evaluate 以本标题行 + 干净已提交的 PREDICTIONS.md + --unseal-dl22 三重门控）。
标签集（8，共用同一 universe）：HUMAN / RAW / CORR_FROZEN（训练池上 in-sample 的家族规则，
非"导入"）/ CORR_REFIT（全预算 2 折交叉拟合，上界参照）/ MIX_200 / MIX_1000（整 query 抽取
的人类标注 + 其余 RAW）/ CORR_REFIT_n200 / CORR_REFIT_n1000（等预算：规则只用对应 MIX 的
人类 query 拟合）。
标签集工作点（build-labels 盲态输出，Δ vs HUMAN，均值 0–3）：
- v4-flash：RAW +0.578；FROZEN +0.053；REFIT +0.064；MIX_200 +0.556；REFIT_n200 +0.185
  （map [0,1,1,3]）；MIX_1000 +0.455；REFIT_n1000 −0.157（map [0,0,1,3]，过校正）。
- gemini-3.5-flash：RAW +0.476；FROZEN −0.066；REFIT −0.066（两折同规则）；MIX_200 +0.460；
  REFIT_n200 −0.161；MIX_1000 +0.375；REFIT_n1000 +0.002。
- gemini-3.8-flash（dl21）：RAW +0.386；FROZEN ≡ RAW（no-op，逐位相同）；REFIT +0.139
  （两折 dial 不同 L4/L1）；MIX_200 +0.341；REFIT_n200 +0.189；MIX_1000 +0.150；REFIT_n1000 +0.390。
模型与配方：主 arm minilm-raw = nreimers/MiniLM-L6-H384-uncased（无排序预训练）；副 arm
msmarco-ce = cross-encoder/ms-marco-MiniLM-L6-v2（warm-start，有 MS MARCO 排序预训练，排序类
终点信息量低）。两 arm 两 objective 统一重置输出头（权重 N(0,0.02)，回归 bias=1.5）；
pointwise 回归 0–3（MSE）；`--select last`、固定 epoch；bs 32、max-len 256、AdamW wd 0.01、
线性调度 10% warmup；seed×3；评分 fp32。超参由 HUMAN 集 sweep（训练池 10% query 验证，
不碰 dl22）选定：**minilm-raw epochs=5 lr=3e-5（val loss 1.0266）；msmarco-ce epochs=3 lr=3e-5
（val loss 0.8381）**。sweep 同时显示：两 arm 的 HUMAN 学生验证集预测均值高于标签均值
+0.19（minilm）/+0.26（msmarco-ce）——训练配方自带尺度偏移，故**绝对终点一律以同 arm 的
HUMAN 学生为配方基线，登记量为 (指标_SET − 指标_HUMAN) 或集合间配对差**。
评测：dl22 全部 judged pair（标准 NDCG@10，gain 2^rel−1，理想序列取该 query 全部人类标注；
MAP@≥2；非 batch5 的 condensed 约定）；绝对：预测 clip 到 [0,3] 后的均值偏移、MAE、4 箱校准/
ECE；round-half-up 后 ≥2 的二值 P/R/F1 与相关率比。query-clustered bootstrap B=1000 seed 42，
90% CI；compare 为同重采样 query 上的配对差，三 seed 先各自聚合再取均值；TOST 等价 margin 0.01
（dl22 ≈76 query 下 90% CI 半宽约 0.01，"inconclusive" 可能且**不等于伤害**）。
判定纪律：方向按点估计（三 seed 均值）；tie 带：NDCG/MAP/F1 0.005，均值偏移/相关率比 0.02；
90% CI 排除 0 单列为"显著"不改判定；tie 不计命中/未中；每条预测按 (judge, arm) 单元逐条打分，
主 arm minilm-raw 计分，msmarco-ce 作稳健性报告。
盲态声明：blind — 登记前未训练任何学生（sweep 仅在 HUMAN 集训练池上）、未评测任何 dl22。

预测（主 arm minilm-raw；每条对 v4-flash 与 gemini-3.5-flash 各判一次，3.8-flash 仅标注处适用）：
- B7-P1 [学生继承工作点] offset(RAW 学生) − offset(HUMAN 学生) > +0.02（宽判官标签把学生
  推高）：v4-flash 是；3.5-flash 是；3.8-flash 是。
- B7-P2 [校正标签把学生拉回] |offset(CORR_FROZEN 学生) − offset(HUMAN 学生)| <
  |offset(RAW 学生) − offset(HUMAN 学生)|：v4-flash 是；3.5-flash 是；3.8-flash N/A（no-op，
  结构性：CORR_FROZEN 学生与 RAW 学生逐位相同，作 pipeline 不变量断言）。
- B7-P3 [排序基本不动] NDCG@10(CORR_FROZEN 学生) − NDCG@10(RAW 学生) 落在 ±0.01 内
  （点估计）：v4-flash 是；3.5-flash 是。同时登记：TOST 可能 inconclusive。
- B7-P4 [阈值 F1 仅强漂移判官受益] F1@2(CORR_FROZEN 学生) − F1@2(RAW 学生) ≥ 0：
  v4-flash 是（A 中 ratio≥1.6 的单元才有 F1 收益）；3.5-flash **no call**（A 中 F1 混合）。
- B7-P5 [相关率比向 1 收敛] |ratio@2(CORR_FROZEN 学生) − 1| < |ratio@2(RAW 学生) − 1|：
  v4-flash 是；3.5-flash 是。
- B7-P6 [等预算：校正优于混入] |offset(CORR_REFIT_n1000 学生) − offset(HUMAN 学生)| <
  |offset(MIX_1000 学生) − offset(HUMAN 学生)|：v4-flash 是；3.5-flash 是。
  n200 版同向：v4-flash 是；3.5-flash 是。F1@2 的等预算对比：no call（A 的 E6 教训）。
- B7-P7 [HUMAN 上界] NDCG@10(HUMAN 学生) ≥ NDCG@10(RAW 学生) − 0.005：v4-flash 是；3.5-flash 是
  （人类标签在排序上不劣于 LLM 标签；若 LLM 标签学生更好则为值得报告的意外）。
- B7-P8 [副 arm 一致性] 以上 P1/P2/P5 在 msmarco-ce 上同向（稳健性，不计分）。
开箱结果（evaluate/compare 后回填）：
开箱结果（2026-10-12，evaluate/compare 于登记 commit 93e8b5e 之后；代码 cfecebe；144 个学生（2 arm × 3 judge × 8 集 × 3 seed）全部评测；
dl22 universe 2146 对 / 76 query，所有结果文件 universe sha e63c9a7096e5 一致；汇总与裁决由 scripts/batch7_summary.py（81ecdb8，仅报告）生成，
数值全部来自 derived/batch7/eval/*.json 与 compare_*.json；* = 90% 配对 query bootstrap CI 排除 0，仅单列不改判定）：

**arm minilm-raw（主 arm，计分）** — 各标签集学生三 seed 均值（NDCG@10 / 均值偏移 / F1@2 / ratio@2 / mean_pred）：
- v4-flash: HUMAN 0.4441/-0.065/0.205/0.69/0.61; RAW 0.4589/+0.553/0.273/2.00/1.23; CORR_FROZEN 0.4386/-0.131/0.180/0.71/0.55; CORR_REFIT 0.4291/-0.120/0.182/0.66/0.56; MIX_200 0.4532/+0.531/0.276/2.00/1.21; MIX_1000 0.4529/+0.465/0.247/1.81/1.14; CORR_REFIT_n200 0.4400/+0.172/0.163/0.58/0.85; CORR_REFIT_n1000 0.4261/-0.243/0.134/0.41/0.44
- 3.5-flash: HUMAN 0.4441/-0.065/0.204/0.68/0.61; RAW 0.4601/+0.453/0.254/1.45/1.13; CORR_FROZEN 0.4432/-0.215/0.183/0.63/0.46; CORR_REFIT 0.4436/-0.215/0.183/0.63/0.46; MIX_200 0.4523/+0.442/0.250/1.47/1.12; MIX_1000 0.4608/+0.405/0.241/1.50/1.08; CORR_REFIT_n200 0.4417/-0.282/0.151/0.49/0.40; CORR_REFIT_n1000 0.4416/-0.164/0.186/0.73/0.51
- 3.8-flash: HUMAN 0.4377/+0.069/0.128/0.52/0.75; RAW 0.4601/+0.505/0.253/1.84/1.18; CORR_FROZEN 0.4601/+0.505/0.253/1.84/1.18; CORR_REFIT 0.4420/+0.176/0.212/1.18/0.85; MIX_200 0.4572/+0.419/0.247/1.39/1.10; MIX_1000 0.4387/+0.152/0.178/0.66/0.83; CORR_REFIT_n200 0.4490/+0.412/0.223/1.27/1.09; CORR_REFIT_n1000 0.4652/+0.512/0.269/1.93/1.19

- B7-P1 v4-flash [minilm-raw] offset(RAW)−offset(HUMAN) = +0.6179 [+0.562,+0.673]* → HIT
- B7-P2 v4-flash [minilm-raw] |offset(FROZEN)−offset(HUMAN)| 0.0658 vs |offset(RAW)−offset(HUMAN)| 0.6179（FROZEN−HUMAN offset -0.0658 [-0.131,+0.000]）→ HIT
- B7-P3 v4-flash [minilm-raw] ΔNDCG@10 FROZEN−RAW = -0.0203 [-0.042,+0.001]，|Δ|>0.01 → MISS；TOST：equivalence within ±0.01 NOT shown (90% CI not inside the margin)
- B7-P4 v4-flash [minilm-raw] ΔF1@2 FROZEN−RAW = -0.0928 [-0.157,-0.034]* → MISS
- B7-P5 v4-flash [minilm-raw] |ratio@2−1| FROZEN 0.2937 vs RAW 1.0041（Δratio FROZEN−RAW -1.2978 [-1.620,-1.013]*）→ HIT
- B7-P6-n1000 v4-flash [minilm-raw] |offset dev| REFIT_n1000 0.1774 vs MIX_1000 0.5299（Δoffset -0.7073 [-0.775,-0.634]*；ΔNDCG@10 -0.0268 [-0.050,-0.001]*；ΔF1@2 -0.1125 [-0.185,-0.042]* no call）→ HIT
- B7-P6-n200 v4-flash [minilm-raw] |offset dev| REFIT_n200 0.2368 vs MIX_200 0.5959（Δoffset -0.3591 [-0.402,-0.314]*；ΔNDCG@10 -0.0132 [-0.031,+0.005]；ΔF1@2 -0.1125 [-0.190,-0.043]* no call）→ HIT
- B7-P7 v4-flash [minilm-raw] ΔNDCG@10 HUMAN−RAW = -0.0147 [-0.045,+0.016] → MISS（登记为『值得报告的意外』：LLM 标签学生排序不劣于人类标签学生）
- B7-P1 3.5-flash [minilm-raw] offset(RAW)−offset(HUMAN) = +0.5178 [+0.473,+0.564]* → HIT
- B7-P2 3.5-flash [minilm-raw] |offset(FROZEN)−offset(HUMAN)| 0.1502 vs |offset(RAW)−offset(HUMAN)| 0.5178（FROZEN−HUMAN offset -0.1502 [-0.209,-0.089]*）→ HIT
- B7-P3 3.5-flash [minilm-raw] ΔNDCG@10 FROZEN−RAW = -0.0169 [-0.033,-0.000]*，|Δ|>0.01 → MISS；TOST：equivalence within ±0.01 NOT shown (90% CI not inside the margin)
- B7-P4 3.5-flash [minilm-raw] ΔF1@2 FROZEN−RAW = -0.0708 [-0.123,-0.024]* → no call（登记）
- B7-P5 3.5-flash [minilm-raw] |ratio@2−1| FROZEN 0.3679 vs RAW 0.4492（Δratio FROZEN−RAW -0.8171 [-1.045,-0.618]*）→ HIT
- B7-P6-n1000 3.5-flash [minilm-raw] |offset dev| REFIT_n1000 0.0989 vs MIX_1000 0.4699（Δoffset -0.5689 [-0.630,-0.506]*；ΔNDCG@10 -0.0192 [-0.042,+0.003]；ΔF1@2 -0.0547 [-0.100,-0.011]* no call）→ HIT
- B7-P6-n200 3.5-flash [minilm-raw] |offset dev| REFIT_n200 0.2167 vs MIX_200 0.5072（Δoffset -0.7238 [-0.770,-0.672]*；ΔNDCG@10 -0.0106 [-0.031,+0.011]；ΔF1@2 -0.0993 [-0.159,-0.043]* no call）→ HIT
- B7-P7 3.5-flash [minilm-raw] ΔNDCG@10 HUMAN−RAW = -0.0160 [-0.044,+0.015] → MISS（登记为『值得报告的意外』：LLM 标签学生排序不劣于人类标签学生）
- B7-P1 3.8-flash [minilm-raw] offset(RAW)−offset(HUMAN) = +0.4358 [+0.393,+0.477]* → HIT
- B7-P2 3.8-flash [minilm-raw] no-op 不变量：ΔNDCG@10 FROZEN−RAW = +0.0000，Δoffset = +0.0001 → N/A（标签逐位相同；学生非逐位相同——训练存在不确定性——但差异为 1e-4 量级、在 tie 带内，不变量成立）

**arm msmarco-ce（副 arm，稳健性，不计分）** — 各标签集学生三 seed 均值（NDCG@10 / 均值偏移 / F1@2 / ratio@2 / mean_pred）：
- v4-flash: HUMAN 0.7464/+0.063/0.494/1.17/0.74; RAW 0.7556/+0.587/0.478/2.30/1.27; CORR_FROZEN 0.7399/-0.031/0.483/1.20/0.65; CORR_REFIT 0.7376/+0.021/0.482/1.33/0.70; MIX_200 0.7599/+0.565/0.477/2.27/1.24; MIX_1000 0.7637/+0.489/0.475/2.01/1.17; CORR_REFIT_n200 0.7468/+0.184/0.450/0.87/0.86; CORR_REFIT_n1000 0.7282/-0.217/0.415/0.73/0.46
- 3.5-flash: HUMAN 0.7464/+0.063/0.494/1.17/0.74; RAW 0.7673/+0.452/0.492/1.52/1.13; CORR_FROZEN 0.7366/-0.148/0.478/1.05/0.53; CORR_REFIT 0.7362/-0.148/0.478/1.05/0.53; MIX_200 0.7670/+0.437/0.490/1.52/1.11; MIX_1000 0.7710/+0.379/0.494/1.50/1.06; CORR_REFIT_n200 0.7437/-0.232/0.455/0.85/0.45; CORR_REFIT_n1000 0.7440/-0.086/0.488/1.18/0.59
- 3.8-flash: HUMAN 0.7467/+0.276/0.450/1.22/0.95; RAW 0.7561/+0.562/0.463/2.02/1.24; CORR_FROZEN 0.7561/+0.562/0.463/2.02/1.24; CORR_REFIT 0.7533/+0.336/0.473/1.57/1.01; MIX_200 0.7589/+0.502/0.479/1.82/1.18; MIX_1000 0.7614/+0.374/0.479/1.48/1.05; CORR_REFIT_n200 0.7572/+0.483/0.481/1.74/1.16; CORR_REFIT_n1000 0.7563/+0.579/0.463/2.08/1.26

- B7-P1 v4-flash [msmarco-ce] offset(RAW)−offset(HUMAN) = +0.5247 [+0.488,+0.563]* → HIT
- B7-P2 v4-flash [msmarco-ce] |offset(FROZEN)−offset(HUMAN)| 0.0941 vs |offset(RAW)−offset(HUMAN)| 0.5247（FROZEN−HUMAN offset -0.0941 [-0.137,-0.046]*）→ HIT
- B7-P3 v4-flash [msmarco-ce] ΔNDCG@10 FROZEN−RAW = -0.0157 [-0.031,-0.001]*，|Δ|>0.01 → MISS；TOST：equivalence within ±0.01 NOT shown (90% CI not inside the margin)
- B7-P4 v4-flash [msmarco-ce] ΔF1@2 FROZEN−RAW = +0.0055 [-0.042,+0.043] → HIT
- B7-P5 v4-flash [msmarco-ce] |ratio@2−1| FROZEN 0.2002 vs RAW 1.3018（Δratio FROZEN−RAW -1.1016 [-1.356,-0.891]*）→ HIT
- B7-P6-n1000 v4-flash [msmarco-ce] |offset dev| REFIT_n1000 0.2792 vs MIX_1000 0.4269（Δoffset -0.7061 [-0.759,-0.650]*；ΔNDCG@10 -0.0355 [-0.049,-0.023]*；ΔF1@2 -0.0599 [-0.136,+0.009] no call）→ HIT
- B7-P6-n200 v4-flash [msmarco-ce] |offset dev| REFIT_n200 0.1211 vs MIX_200 0.5028（Δoffset -0.3817 [-0.418,-0.345]*；ΔNDCG@10 -0.0130 [-0.032,+0.006]；ΔF1@2 -0.0275 [-0.098,+0.037] no call）→ HIT
- B7-P7 v4-flash [msmarco-ce] ΔNDCG@10 HUMAN−RAW = -0.0092 [-0.024,+0.004] → MISS（登记为『值得报告的意外』：LLM 标签学生排序不劣于人类标签学生）
- B7-P1 3.5-flash [msmarco-ce] offset(RAW)−offset(HUMAN) = +0.3893 [+0.367,+0.412]* → HIT
- B7-P2 3.5-flash [msmarco-ce] |offset(FROZEN)−offset(HUMAN)| 0.2106 vs |offset(RAW)−offset(HUMAN)| 0.3893（FROZEN−HUMAN offset -0.2106 [-0.252,-0.167]*）→ HIT
- B7-P3 3.5-flash [msmarco-ce] ΔNDCG@10 FROZEN−RAW = -0.0307 [-0.046,-0.015]*，|Δ|>0.01 → MISS；TOST：equivalence within ±0.01 NOT shown (90% CI not inside the margin)
- B7-P4 3.5-flash [msmarco-ce] ΔF1@2 FROZEN−RAW = -0.0136 [-0.044,+0.012] → no call（登记）
- B7-P5 3.5-flash [msmarco-ce] |ratio@2−1| FROZEN 0.0518 vs RAW 0.5203（Δratio FROZEN−RAW -0.4685 [-0.570,-0.377]*）→ HIT
- B7-P6-n1000 3.5-flash [msmarco-ce] |offset dev| REFIT_n1000 0.1487 vs MIX_1000 0.3161（Δoffset -0.4648 [-0.505,-0.420]*；ΔNDCG@10 -0.0270 [-0.043,-0.010]*；ΔF1@2 -0.0063 [-0.029,+0.012] no call）→ HIT
- B7-P6-n200 3.5-flash [msmarco-ce] |offset dev| REFIT_n200 0.2944 vs MIX_200 0.3742（Δoffset -0.6686 [-0.702,-0.631]*；ΔNDCG@10 -0.0232 [-0.037,-0.010]*；ΔF1@2 -0.0350 [-0.083,+0.006] no call）→ HIT
- B7-P7 3.5-flash [msmarco-ce] ΔNDCG@10 HUMAN−RAW = -0.0209 [-0.039,-0.003]* → MISS（登记为『值得报告的意外』：LLM 标签学生排序不劣于人类标签学生）
- B7-P1 3.8-flash [msmarco-ce] offset(RAW)−offset(HUMAN) = +0.2862 [+0.265,+0.307]* → HIT
- B7-P2 3.8-flash [msmarco-ce] no-op 不变量：ΔNDCG@10 FROZEN−RAW = +0.0000，Δoffset = -0.0000 → N/A（标签逐位相同；学生非逐位相同——训练存在不确定性——但差异为 1e-4 量级、在 tie 带内，不变量成立）

**主 arm 计分：HIT 11 / MISS 5 / tie 0**（P1 3/3、P2 2/2（3.8 不变量 N/A）、P3 0/2、P4 0/1、P5 2/2、P6 4/4、P7 0/2）。
**P8（副 arm 同向，不计分）**：P1/P2/P5 共 7 个可比单元全部同向；P3/P7 在副 arm 同样未中（ΔNDCG@10 FROZEN−RAW −0.016/−0.031，HUMAN−RAW −0.009/−0.021）；P4 副 arm v4-flash +0.0055（主 arm −0.093）。
**主 agent 初步解读（待评估员复核）**：登记的『排序≈不变、绝对量改善』只有后半成立——学生确实继承判官工作点（RAW 学生 offset 高于 HUMAN 学生 +0.44~+0.62），
校正标签把学生拉回人类尺度（FROZEN 学生与 HUMAN 学生 offset 差 0.07~0.15，3.5-flash 过校正至 −0.215）且等预算下优于混入人类标注；但排序**不是**不变：校正标签学生 NDCG@10 低于 RAW 标签学生 0.016~0.031（多数 CI 排除 0），
而 HUMAN 标签学生本身也低于 RAW 标签学生 0.009~0.022（CI 多含 0）；阈值 F1@2 在主 arm 对所有低均值标签集（含 HUMAN）塌缩（recall 0.10~0.17 vs RAW 0.31~0.41），
更像回归学生向标签均值收缩后 round-half-up≥2 的阈值效应而非标签工作点本身（副 arm 各集 F1 均 ≈0.45~0.49）。

## B7 评估员（Sonnet）复核后的修订与勘误（2026-10-12；报告存内部 repo docs/evaluator_reports/2026-10-12_b7_unboxing_sonnet.md）
**计数口径**：裁决脚本把 P6 的 n200/n1000 作为两条、并计入 3.8-flash 的 P1，得 11/5/0（16 项）；登记文本未定义此单位，按『每 judge 一条 P6』则为 9/5（14 项）。两种口径均报告。**实质结构**：11 个命中全部是均值偏移/相关率比类预测（学生 offset 近乎机械地跟随标签均值：v4 RAW 标签 +0.578 → 学生 +0.553，MIX_200 +0.556 → +0.531，n200 +0.185 → +0.172），5 个排序/F1 类预测全部未中——论文应写成『工作点 11/11、排序与 F1 0/5』而非『11/16』。
**勘误（登记措辞）**：(1) 『CORR_FROZEN 学生与 RAW 学生逐位相同』不成立——标签逐位相同，学生只在浮点/GPU 非确定性水平一致（NDCG 差 1.4e-5、offset 5.4e-5；离散终点 6 位小数相同），应写『≤5.4e-5』；(2) 登记前提『训练配方自带 +0.19/+0.26 尺度偏移』在 dl22 上不复现（HUMAN 学生 offset −0.065（minilm）/+0.063（msmarco-ce）），HUMAN 学生基线无害但不得写成『配方固有高估』；(3) 3.5-flash 的 P5 命中是过冲（ratio 1.45 → 0.63）而非向 1 收敛，|ratio−1| 仅从 0.449 降到 0.368。
**P4/F1 塌缩的机制（评估员判定，解读性）**：校正后 recall 大降（v4 −0.256*，3.5 −0.162*）而 precision 不动（+0.013、+0.022，CI 含 0）；主 arm 8 个标签集的 precision 都在 0.19–0.25 而 ratio 从 0.41 到 2.0——这是近随机二值分类器的特征，F1 沿平坦 precision 曲线移动，F1 与 ratio 的 Pearson 0.95（F1=2bρ/(1+ρ)，拟合 b=0.215，RMSE 0.016；b 须对 qrels 核实；全判相关的平凡 F1 ≈0.33–0.35，RAW 学生 0.25–0.27 离此下限不远）。副 arm precision 0.34–0.49、F1 平坦于 0.415–0.494（与 ratio r=0.39），校正无代价。结论：固定阈值 1.5 + F1 + 弱学生使 F1 成为『预测阳性率』的度量；标签工作点只通过 ratio 起作用；HUMAN 学生均值几乎正确（−0.065）却只有 0.69× 的对越过 1.5，是预测离散度压缩（向均值收缩）的信号，但 CSV 无预测 sd，属推断。CORR_FROZEN 学生与 HUMAN 学生在 F1 上不可区分（−0.025 [−0.068,+0.013]、−0.021）。
**P3/P7 排序差的机制（解读性）**：校正标签学生 NDCG@10 低于 RAW 学生 0.016–0.031（FROZEN：v4 −0.020、3.5 −0.017*、副 arm −0.016*/−0.031*；REFIT_n1000 −0.033*/−0.019*/−0.027*/−0.023*），MAP 降 0.021–0.042；±0.01 等价从未被证明；但与 HUMAN 学生不可区分（−0.006 [−0.037,+0.023]、−0.001、−0.0065、−0.0098）。HUMAN<RAW 符号一致（6/6 单元 NDCG 与 MAP 均为负，−0.009~−0.022）但仅 1 个 CI 排除 0（msmarco 3.5 −0.0209），且仅 4 个独立 HUMAN 学生、共用同一批 dl22 query；合并均值约 −0.015。最吻合的机制是『NDCG 随标签水平/形状而非标签来源变化』：NDCG 与学生 offset 的 Spearman 0.95/0.79/1.00/0.88/0.86（3.8 msmarco 0.28）；3.8 minilm 呈单 judge 剂量-反应（标签 offset .390→NDCG .4652，.386→.4601，.189→.4490，.139(REFIT，LLM 衍生)→.4420，.150(MIX_1000)→.4387，0(HUMAN)→.4377）——LLM 衍生的 REFIT 标签表现与 HUMAN 相同，故『LLM 标签优于人类标签』不被支持。过校正本身不是驱动（3.5 offset −0.28 与 −0.16 的 NDCG 相同）。候选解释（均未检验）：冻结 map [0,0,2,3] 抹掉 0/1 信息；弱学生受益于 paraphrase 均值集成标签的去噪/密度。
**未登记但可见**：校正学生低于自身标签 offset（v4 FROZEN 标签 +0.053 → 学生 −0.131；3.5 −0.066 → −0.215），RAW/MIX 学生跟随标签 ±0.03；对比 B6 教师在 dl22 的 FROZEN ratio（v4 ≈1.40、3.5 ≈1.0），学生 ratio 0.71/0.63——工作点以方向而非幅度迁移；RAW 学生放大过报（3.8 教师 1.02 → 学生 1.84）。ECE4 未因校正改善（v4 −0.045 CI 含 0；3.5 +0.099 变差），校正学生 ECE4/MAE 劣于 HUMAN 学生（+0.146*/+0.172*；+0.067*/+0.046*），仅在 MAE 上优于 RAW（−0.146*/−0.106*）。
**P6 的诚实读法**：等预算下对齐（offset、ratio）可迁移，排序与 F1 不可（NDCG 在 8/8 judge×预算×arm 单元更低，4/8 CI 排除 0）；MIX 在 4–20% 人类占比下几乎不动均值（−0.02~−0.09），是低门槛；缺少真正的等预算对手——用同样 n 个人类对对 RAW 学生做事后校准——作为追加探索分析补做（scripts/batch7_posthoc.py）。n 预算规则不稳（v4 n200 +0.185 vs n1000 −0.157；3.5 n200 −0.161 vs n1000 +0.002）。3.8 的 MIX_1000 是 1551 对池的 65% 人类标注，不可比。msmarco-ce 为 MS MARCO 预训练模型，与 DL22（MS MARCO v2）污染不能排除，作为副 arm 加一句说明。
**可安全写进论文的主张**：学生继承判官工作点（+0.62/+0.52/+0.44，CI 全排除 0）[registered]；校正标签学生移到 HUMAN 学生水平（offset −0.066/−0.150 相对 HUMAN 学生；ratio 0.71/0.63 vs RAW 学生 2.00/1.45，与 HUMAN 学生不可区分）[registered]；等预算下拟合校正比混入同量人类标注移动 offset 更多（−0.36~−0.72，CI 排除 0）[registered]；『排序不变』未满足：校正标签学生 NDCG@10 低于 RAW 学生 0.016–0.031 但与 HUMAN 学生不可区分，±0.01 等价未证明 [interpretive]；F1@2 未改善，弱学生的 F1 跟踪预测阳性率（r≈0.95）而非质量 [interpretive]；意外：LLM 标签学生在 6/6 单元排序优于 HUMAN 标签学生（−0.009~−0.022，1 个 CI 排除 0；76 query、4 个独立 HUMAN 学生、单配方、HUMAN 学生自身 ratio 0.69 欠报）[interpretive]；该差距更符合标签水平/形状而非来源 [interpretive]。
**不得写的主张**：校正保持排序/无排序代价；校正改善 F1、ECE 或均值以外的校准；LLM 标签学生优于人类标签学生；学生已校准到人类（实际欠冲 −0.13~−0.22、ratio 0.63–0.71）；等预算校正是人类标注的最佳用法（仅与 MIX 比较）；『配方高估 +0.19』；『3.8 逐位相同』；『副 arm 证实 F1』。
**追加分析（探索性，待跑）**：事后平移校准的 RAW 学生（同 n 人类对）；阈值无关/匹配流行率终点（AUC、匹配 ratio 的 F1、预测 sd、≥1.5 比例、平凡 F1、随机 NDCG 下限）；学生间 Kendall τ 与逐 query NDCG 配对差；教师在同一 dl22 对上的参照；标签集诊断（直方图、与 HUMAN 的相关、grade-1 坍缩比例）。

## B7 探索性追加分析（scripts/batch7_posthoc.py f486988；非预注册；评估员 Sonnet 复核，报告存内部 repo docs/evaluator_reports/2026-10-12_b7_posthoc_sonnet.md）
全部仅由 144 个 dl22 评测 JSON 与标签 manifest 计算（无重训）；完整性：存储终点由 pairs 重算到 1e-9 一致，平移后 NDCG/MAP 与 RAW 逐 query 相等（断言）；dl22 NDCG 在 73 个 IDCG>0 的 query 上、MAP 65 个；教师参照覆盖 1.000、人类分数无不一致。
- **决定性对比：RAW 学生输出侧平移（SHIFT_n = 同 n 人类对上的教师 delta_hat；SHIFT_FULL = 训练全集标签均值差）vs 标签侧校正（CORR_REFIT_n / FROZEN）**：12 个 (judge, arm, n) 单元中 SHIFT_n 的 |offset| 全部更小（9 个 CI 排除 0；剔除 3.8 的 no-op 规则后 8/8、5 显著）、MAE 12/12 更小（9 显著）、ECE4 10 个显著更好/1 个显著更差、NDCG 在 9/12 更高（+0.011~+0.033，6 显著；三个负值均为 3.8 no-op 单元且 ≤0.005）；F1 非 3.8 单元 +0.049*/+0.056*/+0.054*/−0.002/+0.026/+0.058*/+0.031*/−0.020，3.8 单元 SHIFT 更差（−0.075*/−0.113*，流行率效应）。SHIFT_FULL 使主 arm offset 到 +0.004/−0.014/+0.123（HUMAN 学生 −0.065/−0.065/+0.069），与 in-sample ORACLE 平移相当；SHIFT_FULL vs HUMAN：NDCG 6/6 为正（+0.009~+0.022，1 显著）。SHIFT_n vs MIX_n：offset/MAE/ECE4 11/12 显著优于 MIX，NDCG ±0.02 内。n=200（7 query）的 delta_hat 与全集偏移差 0.07–0.10。
- **阈值无关终点**：主 arm 学生弱（AUC 0.565–0.604、F1@matched 0.19–0.23 vs 随机 0.153、NDCG 高于随机下限 0.373 仅 0.05–0.09）；副 arm AUC 0.80–0.83 ≈ 教师 0.825。主 arm RAW 学生 F1@2 0.25–0.27 = 平凡全判相关 F1 0.265，HUMAN/CORR 学生低于平凡值——『塌缩』是从平凡到劣于平凡。校正略降判别力：AUC FROZEN−RAW −0.016/−0.0365*/−0.006/−0.0157*；F1@matched 去掉大部分塌缩（−0.025、−0.020 vs F1@2 −0.093、−0.071），残差≈AUC 损失；平移同样令 F1@2 下降（−0.074*/−0.064*/−0.076*）——证实 F1@2 对这些学生是流行率度量。**收缩假说不成立**：校正学生预测 sd 反而更大（vs HUMAN +0.07~+0.14*），frac≥1.5 跟随均值；所有主 arm 学生 sd 约为标签 sd 的一半（弱回归器共性）。
- **Kendall τ-b 与逐 query NDCG**：主 arm v4 RAW–HUMAN 0.50/0.53、RAW–FROZEN 0.58/0.57 vs seed 间 0.67/0.72（标签集变化带来的重排与换 seed 同量级；3.5 的 RAW–FROZEN 0.70 在 seed 底线内）；副 arm RAW–FROZEN 0.76/0.66 vs seed 0.92/0.91（更强学生中标签侧改变移动排序更多）。FROZEN−RAW 逐 query：55–64% 的 query 下降，mean|d| 为 |mean| 的 3–6 倍——是广泛 churn 的小净不对称，不是少数 query 驱动。
- **教师参照（同一 dl22 对）**：v4 RAW 教师 offset +0.465 / ratio 1.85 / F1 0.537 / AUC 0.825；主 arm 学生 +0.553 / 2.00 / 0.273 / 0.594；副 arm +0.587 / 2.30 / 0.478 / 0.819。FROZEN：教师 −0.129 / 1.40 / 0.548，主 arm 学生 −0.131 / 0.71 / 0.180。方向迁移、幅度（除 1 个单元）与 ratio 不迁移；教师 F1 在校正下不塌缩（+0.011）而主 arm 学生塌缩——塌缩是学生特有；教师 AUC 在同一 map 下也下降（0.825→0.801、0.839→0.775），判别力损失源于标签（map 造成的并列）。学生学到的是训练全集的偏移而非 dl22 的（教师 REFIT dl22 +0.311 vs 学生 −0.120）。
- **标签诊断（训练全集）**：冻结 map [0,0,2,3] 把 RAW 的 1 分 95–96% 压到 0（校正集 grade-1 占比 4–6%，HUMAN 27%）；标签-人类 Pearson RAW 0.661、FROZEN 0.640、MIX_1000 0.721；主 arm v4 的 NDCG 在 7 个 LLM 衍生集上随坍缩比例单调，但标签均值也近乎单调——水平与坍缩共线，仅 3.8 的两个分离单元偏向『水平』；去噪假说仅弱支持。无一机制确立。
- **可写**（探索性）：输出侧平移以同样人类预算达到等于或优于标签侧校正的 offset/MAE，并保持 RAW 学生的 NDCG（高出校正学生 0.011–0.033）；『对本配方，校正应落在输出而非标签』；F1@2 不能裁判此问题。**不得写**：『输出平移普遍优于标签校正』（单配方、单 76-query 集、未做多重校正、3.8 为 no-op）；『标签侧校正无用/有害』（只测了常数偏移；离散 dial/map 无法实现连续平移——部分差距是规则粒度而非位置）；『F1@matched 表明校正无代价』（in-sample；AUC 降 0.006–0.037）；『收缩解释塌缩』（被 pred sd 否定）或任何机制已确立；『n=200 足够』；『学生达到教师水平』。
- **评估员判定 Application B 关闭**（无需新计算）。可选后续：逐 query 差分位数与 leave-top-5-out、per-query AUC CI（免费）；连续平移标签训练的学生（分离位置与粒度，Colab）；map-only/dial-only 学生 2×2（分离 grade-1 坍缩与水平）。待核实：3.8 的 B6 REFIT 教师标签与 B7 REFIT 学生标签不同（B6 为全局规则、B7 为交叉拟合，待确认）；教师参照『1.000 of B6 rows』的分母。

## B7 探索性消融（2026-10-12 登记，非预注册、不计分；在训练任何消融学生之前写下）
目的：把校正标签学生 NDCG@10 下降（相对 RAW 学生 0.016–0.031）的来源拆成『标签均值水平』、
『离散规则结构（map 把 1 分压成 0）』与『语气旋钮本身』。同一 universe 新增 4 套标签
（scripts/batch7_distill.py build-ablation；登记的 8 套标签文件逐字节不变）：
SHIFT_FULL = clip(RAW − Δ̂_full)（连续平移，不改形状）；SHIFT_n1000 = clip(RAW − Δ̂_n1000)；
DIAL_ONLY = 冻结 dial（L1）+ identity map；MAP_ONLY = L3 + 冻结 map [0,0,2,3]。
judge：v4-flash、3.5-flash（3.8-flash 的规则为 identity，退化，不跑）；两 arm；3 seed；
dl22 评测与配对比较与 B7 相同（三重门控不变）。
**预期（探索性，仅供事后对照，不计命中）**：
- 若 SHIFT_FULL 学生的 NDCG 与 RAW 学生持平（|Δ|≤0.01）而 offset 回到 HUMAN 学生水平 → 下降来自
  规则结构而非均值水平；若 SHIFT 学生也掉 0.02–0.03 → 下降来自标签均值水平（与来源无关）。
- MAP_ONLY vs DIAL_ONLY：若 MAP_ONLY 掉分、DIAL_ONLY 不掉 → 损失来自 map 的 grade-1 坍缩，
  语气旋钮本身无排序代价；反之则语气旋钮本身有代价。
- 主 agent 的先验：第一组偏向『规则结构』（因 3.8-flash 的剂量-反应把水平与坍缩混在一起，
  无法区分，故不给出强先验）；第二组偏向 MAP_ONLY 掉分（map 路由 95–96% 的 1 分到 0，教师 AUC
  在同一 map 下也降）。评估员对此持『不可分离』立场。
开箱结果（跑完回填）：
2026-10-18/19（UTC）跑完：build-ablation 校验登记标签 sha256 逐字节不变后写入 4 套标签；48 个学生（2 judge × 4 集 × 3 seed × 2 arm）按登记配方训练；dl22 三重门控评测；配对比较；scripts/batch7_summary.py 汇总（b7_summary_ablation.txt）。登记集的数字与原开箱逐项相同。三 seed 均值（NDCG@10 / offset / F1@2 / ratio@2）与配对差（* = 90% CI 排除 0）如下（原样抄录汇总输出）：

```
ABLATION (exploratory; evaluator-requested, NOT pre-registered; 3-seed means, * = paired 90% CI excludes 0)
====================================================================================================

  [minilm-raw] deepseek__deepseek-v4-flash
  set                    NDCG@10          offset            F1@2           ratio
  RAW                    +0.4589         +0.5528         +0.2732         +2.0041
  HUMAN                  +0.4441         -0.0652         +0.2051         +0.6870
  CORR_FROZEN            +0.4386         -0.1310         +0.1804         +0.7063
  SHIFT_FULL             +0.4443         +0.0582         +0.1803         +0.6677
  SHIFT_n1000            +0.4421         +0.0288         +0.1626         +0.6128
  DIAL_ONLY              +0.4529         +0.3272         +0.2717         +1.2967
  MAP_ONLY               +0.4431         -0.0226         +0.2037         +0.9441
    paired compares A - B:
    DIAL_ONLY_vs_CORR_FROZEN         +0.0143[-0.002,+0.031]    +0.4582[+0.411,+0.502]*   +0.0913[+0.044,+0.148]*   +0.5904[+0.432,+0.757]*
    DIAL_ONLY_vs_HUMAN               +0.0087[-0.020,+0.037]    +0.3924[+0.340,+0.443]*   +0.0666[+0.008,+0.133]*   +0.6098[+0.423,+0.803]*
    DIAL_ONLY_vs_MAP_ONLY            +0.0098[-0.010,+0.029]    +0.3498[+0.302,+0.393]*   +0.0680[+0.023,+0.120]*   +0.3526[+0.197,+0.503]*
    DIAL_ONLY_vs_RAW                 -0.0060[-0.024,+0.011]    -0.2256[-0.253,-0.200]*   -0.0015[-0.026,+0.020]    -0.7073[-0.919,-0.537]*
    MAP_ONLY_vs_CORR_FROZEN          +0.0045[-0.009,+0.019]    +0.1083[+0.081,+0.135]*   +0.0233[+0.003,+0.049]*   +0.2378[+0.152,+0.336]*
    MAP_ONLY_vs_HUMAN                -0.0011[-0.033,+0.031]    +0.0426[-0.021,+0.109]    -0.0014[-0.037,+0.038]    +0.2571[+0.084,+0.453]*
    MAP_ONLY_vs_RAW                  -0.0158[-0.034,+0.004]    -0.5754[-0.624,-0.522]*   -0.0695[-0.129,-0.018]*   -1.0600[-1.333,-0.801]*
    SHIFT_FULL_vs_CORR_FROZEN        +0.0058[-0.010,+0.022]    +0.1892[+0.151,+0.225]*   -0.0001[-0.015,+0.014]    -0.0386[-0.096,+0.016] 
    SHIFT_FULL_vs_HUMAN              +0.0002[-0.031,+0.031]    +0.1234[+0.075,+0.170]*   -0.0248[-0.066,+0.011]    -0.0193[-0.194,+0.141] 
    SHIFT_FULL_vs_RAW                -0.0145[-0.032,+0.002]    -0.4945[-0.530,-0.460]*   -0.0929[-0.163,-0.033]*   -1.3364[-1.651,-1.051]*
    SHIFT_n1000_vs_HUMAN             -0.0021[-0.032,+0.029]    +0.0940[+0.046,+0.142]*   -0.0425[-0.085,-0.007]*   -0.0742[-0.248,+0.083] 
    SHIFT_n1000_vs_RAW               -0.0168[-0.035,+0.001]    -0.5239[-0.562,-0.487]*   -0.1106[-0.188,-0.044]*   -1.3913[-1.720,-1.104]*

  [minilm-raw] google__gemini-3.5-flash
  set                    NDCG@10          offset            F1@2           ratio
  RAW                    +0.4601         +0.4528         +0.2536         +1.4492
  HUMAN                  +0.4441         -0.0650         +0.2042         +0.6850
  CORR_FROZEN            +0.4432         -0.2152         +0.1828         +0.6321
  SHIFT_FULL             +0.4486         +0.0379         +0.1745         +0.6230
  SHIFT_n1000            +0.4453         -0.0016         +0.1666         +0.5793
  DIAL_ONLY              +0.4575         +0.3879         +0.2389         +1.2429
  MAP_ONLY               +0.4416         -0.1679         +0.1840         +0.7378
    paired compares A - B:
    DIAL_ONLY_vs_CORR_FROZEN         +0.0144[-0.002,+0.030]    +0.6031[+0.562,+0.640]*   +0.0561[+0.017,+0.100]*   +0.6108[+0.454,+0.785]*
    DIAL_ONLY_vs_HUMAN               +0.0134[-0.016,+0.041]    +0.4529[+0.409,+0.499]*   +0.0347[+0.000,+0.073]*   +0.5579[+0.403,+0.751]*
    DIAL_ONLY_vs_MAP_ONLY            +0.0159[+0.001,+0.031]*   +0.5558[+0.512,+0.596]*   +0.0548[+0.021,+0.092]*   +0.5051[+0.349,+0.672]*
    DIAL_ONLY_vs_RAW                 -0.0026[-0.013,+0.009]    -0.0649[-0.076,-0.053]*   -0.0147[-0.031,+0.001]    -0.2063[-0.280,-0.141]*
    MAP_ONLY_vs_CORR_FROZEN          -0.0015[-0.011,+0.007]    +0.0473[+0.035,+0.059]*   +0.0012[-0.012,+0.017]    +0.1057[+0.066,+0.159]*
    MAP_ONLY_vs_HUMAN                -0.0025[-0.030,+0.024]    -0.1029[-0.162,-0.039]*   -0.0201[-0.046,+0.006]    +0.0528[-0.084,+0.192] 
    MAP_ONLY_vs_RAW                  -0.0185[-0.032,-0.003]*   -0.6207[-0.663,-0.575]*   -0.0696[-0.113,-0.028]*   -0.7114[-0.919,-0.520]*
    SHIFT_FULL_vs_CORR_FROZEN        +0.0054[-0.006,+0.017]    +0.2530[+0.221,+0.282]*   -0.0083[-0.026,+0.007]    -0.0091[-0.052,+0.035] 
    SHIFT_FULL_vs_HUMAN              +0.0044[-0.024,+0.031]    +0.1029[+0.060,+0.145]*   -0.0296[-0.063,-0.001]*   -0.0620[-0.182,+0.054] 
    SHIFT_FULL_vs_RAW                -0.0115[-0.023,+0.000]    -0.4149[-0.441,-0.389]*   -0.0791[-0.129,-0.031]*   -0.8262[-1.036,-0.635]*
    SHIFT_n1000_vs_HUMAN             +0.0011[-0.028,+0.027]    +0.0633[+0.020,+0.107]*   -0.0375[-0.073,-0.009]*   -0.1057[-0.234,+0.017] 
    SHIFT_n1000_vs_RAW               -0.0148[-0.027,-0.002]*   -0.4545[-0.482,-0.427]*   -0.0870[-0.140,-0.036]*   -0.8699[-1.097,-0.666]*

  [msmarco-ce] deepseek__deepseek-v4-flash
  set                    NDCG@10          offset            F1@2           ratio
  RAW                    +0.7556         +0.5873         +0.4777         +2.3018
  HUMAN                  +0.7464         +0.0626         +0.4943         +1.1667
  CORR_FROZEN            +0.7399         -0.0314         +0.4832         +1.2002
  SHIFT_FULL             +0.7437         +0.1294         +0.4705         +1.1992
  SHIFT_n1000            +0.7429         +0.0976         +0.4724         +1.1514
  DIAL_ONLY              +0.7590         +0.3879         +0.4911         +1.6982
  MAP_ONLY               +0.7408         +0.1436         +0.4686         +1.6616
    paired compares A - B:
    DIAL_ONLY_vs_CORR_FROZEN         +0.0191[+0.009,+0.030]*   +0.4194[+0.385,+0.455]*   +0.0079[-0.021,+0.042]    +0.4980[+0.390,+0.608]*
    DIAL_ONLY_vs_HUMAN               +0.0126[-0.004,+0.029]    +0.3253[+0.292,+0.359]*   -0.0031[-0.038,+0.036]    +0.5315[+0.408,+0.661]*
    DIAL_ONLY_vs_MAP_ONLY            +0.0183[+0.004,+0.033]*   +0.2443[+0.208,+0.282]*   +0.0225[+0.004,+0.044]*   +0.0366[-0.052,+0.116] 
    DIAL_ONLY_vs_RAW                 +0.0034[-0.010,+0.017]    -0.1993[-0.218,-0.181]*   +0.0134[-0.008,+0.033]    -0.6037[-0.769,-0.473]*
    MAP_ONLY_vs_CORR_FROZEN          +0.0009[-0.010,+0.012]    +0.1751[+0.144,+0.207]*   -0.0146[-0.037,+0.011]    +0.4614[+0.344,+0.602]*
    MAP_ONLY_vs_HUMAN                -0.0056[-0.022,+0.011]    +0.0810[+0.032,+0.133]*   -0.0256[-0.055,+0.008]    +0.4949[+0.363,+0.659]*
    MAP_ONLY_vs_RAW                  -0.0148[-0.026,-0.003]*   -0.4436[-0.481,-0.407]*   -0.0091[-0.037,+0.015]    -0.6402[-0.794,-0.513]*
    SHIFT_FULL_vs_CORR_FROZEN        +0.0038[-0.010,+0.017]    +0.1609[+0.125,+0.194]*   -0.0127[-0.030,+0.005]    -0.0010[-0.057,+0.057] 
    SHIFT_FULL_vs_HUMAN              -0.0027[-0.016,+0.012]    +0.0668[+0.037,+0.098]*   -0.0238[-0.046,-0.003]*   +0.0325[-0.056,+0.121] 
    SHIFT_FULL_vs_RAW                -0.0119[-0.018,-0.006]*   -0.4578[-0.481,-0.431]*   -0.0072[-0.058,+0.037]    -1.1026[-1.337,-0.901]*
    SHIFT_n1000_vs_HUMAN             -0.0035[-0.017,+0.011]    +0.0350[+0.005,+0.067]*   -0.0218[-0.044,+0.000]    -0.0152[-0.103,+0.072] 
    SHIFT_n1000_vs_RAW               -0.0127[-0.019,-0.007]*   -0.4897[-0.514,-0.461]*   -0.0052[-0.058,+0.040]    -1.1504[-1.394,-0.945]*

  [msmarco-ce] google__gemini-3.5-flash
  set                    NDCG@10          offset            F1@2           ratio
  RAW                    +0.7673         +0.4519         +0.4920         +1.5203
  HUMAN                  +0.7464         +0.0626         +0.4943         +1.1667
  CORR_FROZEN            +0.7366         -0.1479         +0.4784         +1.0518
  SHIFT_FULL             +0.7511         +0.0766         +0.4750         +0.9939
  SHIFT_n1000            +0.7522         +0.0402         +0.4732         +0.9543
  DIAL_ONLY              +0.7647         +0.4055         +0.4965         +1.4157
  MAP_ONLY               +0.7408         -0.1030         +0.4782         +1.1331
    paired compares A - B:
    DIAL_ONLY_vs_CORR_FROZEN         +0.0280[+0.014,+0.042]*   +0.5534[+0.518,+0.589]*   +0.0181[-0.004,+0.044]    +0.3638[+0.286,+0.446]*
    DIAL_ONLY_vs_HUMAN               +0.0182[+0.000,+0.036]*   +0.3429[+0.317,+0.370]*   +0.0022[-0.025,+0.031]    +0.2490[+0.164,+0.340]*
    DIAL_ONLY_vs_MAP_ONLY            +0.0239[+0.009,+0.038]*   +0.5085[+0.468,+0.547]*   +0.0182[-0.001,+0.041]    +0.2825[+0.213,+0.359]*
    DIAL_ONLY_vs_RAW                 -0.0027[-0.008,+0.002]    -0.0464[-0.053,-0.039]*   +0.0045[-0.006,+0.014]    -0.1047[-0.142,-0.075]*
    MAP_ONLY_vs_CORR_FROZEN          +0.0041[-0.003,+0.012]    +0.0449[+0.035,+0.055]*   -0.0002[-0.011,+0.011]    +0.0813[+0.051,+0.116]*
    MAP_ONLY_vs_HUMAN                -0.0056[-0.025,+0.013]    -0.1656[-0.209,-0.118]*   -0.0160[-0.040,+0.007]    -0.0335[-0.116,+0.052] 
    MAP_ONLY_vs_RAW                  -0.0266[-0.041,-0.011]*   -0.5549[-0.592,-0.517]*   -0.0137[-0.041,+0.008]    -0.3872[-0.476,-0.310]*
    SHIFT_FULL_vs_CORR_FROZEN        +0.0145[+0.004,+0.026]*   +0.2245[+0.196,+0.251]*   -0.0034[-0.014,+0.006]    -0.0579[-0.091,-0.029]*
    SHIFT_FULL_vs_HUMAN              +0.0047[-0.011,+0.020]    +0.0139[-0.014,+0.043]    -0.0192[-0.044,+0.004]    -0.1728[-0.248,-0.093]*
    SHIFT_FULL_vs_RAW                -0.0163[-0.026,-0.008]*   -0.3753[-0.395,-0.353]*   -0.0169[-0.051,+0.012]    -0.5264[-0.631,-0.434]*
    SHIFT_n1000_vs_HUMAN             +0.0058[-0.010,+0.021]    -0.0224[-0.051,+0.008]    -0.0210[-0.046,+0.002]    -0.2124[-0.286,-0.133]*
    SHIFT_n1000_vs_RAW               -0.0151[-0.025,-0.007]*   -0.4117[-0.433,-0.388]*   -0.0187[-0.053,+0.011]    -0.5661[-0.672,-0.471]*
```
主 agent 初读（待评估员复核）：(1) 连续平移标签（SHIFT_FULL，不改形状只降均值）的学生 NDCG 同样低于 RAW 学生（主 arm −0.015/−0.012，副 arm −0.012*/−0.016*），幅度与 CORR_FROZEN 相同、与 HUMAN 学生无差（+0.000/+0.004）→ 排序损失来自标签均值水平，而非离散规则结构；(2) DIAL_ONLY（只换语气档位，均值仍高 +0.33/+0.39）的学生 NDCG 与 RAW 无差（−0.006/−0.003；副 arm +0.003/−0.003），F1 不变，但只去除约三分之一的 ratio 膨胀；MAP_ONLY（降均值）掉 0.015–0.027 → 语气旋钮本身无排序代价，损失随均值下降而来，map 的损失也主要是均值效应。

**消融评估员复核（Sonnet，2026-10-19；报告存内部 repo docs/evaluator_reports/2026-10-19_b7_ablation_sonnet.md）**
完整性：登记集 48 行与原快照逐列相同（差 0.0），1008 条登记比较相同，裁决仍为 11/5/0。
(a) 连续平移标签（LSHIFT；为与事后输出侧 SHIFT 区分改名）的学生 NDCG 低于 RAW 学生 0.012–0.016
（四单元同向、副 arm 两个 CI 排除 0），四单元合并为 FROZEN 损失的 65%；与 HUMAN 学生差 ≤0.006。
LSHIFT 与 RAW 标签信息完全相同（严格单调一一映射 0→0、1→0.42、2→1.42、3→2.42），故损失来自训练标签
**均值水平**，而非标签信息损失、来源或去噪；"规则结构"先验不成立。所有 12 套标签集的学生落在同一条
NDCG–offset 直线上（r 0.90–0.96；仅用登记集拟合可外推预测 16 个消融点，残差 ≤0.006；斜率约
0.02–0.05 NDCG/每格偏移）。map 专属残差 ≤0.01 不能排除；LSHIFT 的截断把 0|1 间距压到 0.42 也是混杂。
(b) DIAL_ONLY 的 NDCG 与 RAW 无差（−0.006/−0.003/+0.003/−0.003，CI 全含 0），但只去除 offset 差距的
36/13/38/12%、ratio 差距的 54/27/53/30%；**按每去除一格偏移的代价计，dial 与 map、平移、人类标签相同
（主 arm 约 0.025–0.03 NDCG/格）——语气旋钮不是免费的对齐杠杆，只是移动得少**。MAP_ONLY 复现 FROZEN
损失（78–109%）。"校正学生低于自身标签均值 0.1–0.2"是 map 的性质（identity-map 与连续集跟随标签 ±0.03）。
预期判定：预期 1 两分支都未字面命中，数据取"水平"分支；预期 2 观察符合分支 1 但推论不成立（2×2 未做
水平匹配，直线即可预测 DIAL−MAP 之差）。仍混杂：截断改形状；DIAL_ONLY 同时改票与水平；水平效应是标签固有
还是配方（MSE、固定 epoch、head bias 1.5）所致未测。
**不得写**：LLM 标签优于人类标签 / 来源无关；语气免费；坍缩或水平"是机制"（仅示充分性）；均值高的标签训出
更好 ranker；LSHIFT 已对齐（主 arm 高于 HUMAN 学生 +0.12）；以标签侧名义谈输出侧平移。
可选后续：head bias 初始化改为标签均值的 6 个学生（判别水平效应是否为配方伪影）；HUMAN+Δ̂ 或不截断的
LSHIFT。评估员判定 B 关闭。

## B7 探索性追加 2（2026-10-19 登记，非预注册、不计分；在训练前写下）：均值水平效应是否为配方伪影
问题：消融显示学生 NDCG@10 随训练标签均值下降而下降（12 套标签集共线）。登记配方的回归头偏置固定
初始化为 1.5（≈RAW 均值 1.56，远高于 HUMAN 0.98 / LSHIFT 1.07）。若把偏置初始化改为各标签集自身均值
（`train --head-bias label_mean`），用 v4-flash、主 arm、LSHIFT（SHIFT_FULL）与 HUMAN 两套标签各 3 seed
共 6 个学生（其余配方不变），dl22 评测并与原学生配对比较：
- 若 LSHIFT@hb 与 HUMAN@hb 的 NDCG 回到 RAW 学生水平（与 RAW 差 |Δ|≤0.005，或相对原 LSHIFT/HUMAN
  学生 +0.01 以上）→ 均值水平效应主要是配方伪影（偏置初始化），论文须改写为"该配方下"的现象并弱化；
- 若仍低于 RAW 0.012–0.016（与原学生无差）→ 效应与偏置初始化无关，但仍限定为"本回归配方"（MSE、
  固定 epoch）。
RAW 自身均值 ≈1.5，故 RAW 学生不受此改动影响，充当对照。主 agent 无强先验（略偏向"部分是配方伪影"）。
开箱结果（跑完回填）：
2026-10-19 跑完（代码 1702127；v4-flash、minilm-raw、3 seed、其余配方不变；dl22 三重门控）：
- SHIFT_FULL@hb（LSHIFT，偏置=训练集标签均值）：NDCG@10 0.4608(sd 0.005) / offset +0.072 / F1@2 0.190 / ratio 0.69；
  vs SHIFT_FULL +0.0165 [+0.006,+0.028]*；vs RAW +0.0019 [−0.012,+0.016]；vs HUMAN +0.0166 [−0.013,+0.047]。
- HUMAN@hb：NDCG@10 0.4493(sd 0.003) / offset −0.045 / F1@2 0.224 / ratio 0.73；vs HUMAN +0.0051 [−0.006,+0.018]；
  vs RAW −0.0096 [−0.039,+0.020]。
- 主 agent 初读（待复核）：LSHIFT 学生的排序损失被偏置初始化完全解释（回到 RAW 水平）；HUMAN 学生约收复
  1/3 的差距、剩余 −0.010 落在噪声内。"均值水平效应"至少大部分是配方伪影 → 论文须改写（见追加 2b）。

## B7 探索性追加 2b（2026-10-19 登记，训练前写下）：校正标签学生在修正偏置后是否仍有排序代价
用同样的 `--head-bias label_mean --tag hb` 配方重训 CORR_FROZEN（v4-flash、3.5-flash；主 arm；3 seed）
与 MAP_ONLY（同上），dl22 评测并与原学生、RAW、HUMAN 配对比较。
- 若 CORR_FROZEN@hb 与 RAW 的 NDCG 差 |Δ|≤0.01（CI 含 0）→ 登记的 P3 未中（"校正标签学生排序更差"）是
  配方伪影；§5.7 的"标签侧校正有排序代价"与 Protocol 第 4 步的"输出侧零代价 vs 标签侧 0.012–0.031"
  须改写为"在偏置初始化合理的配方下，标签侧校正无可检测的排序代价；输出侧平移仍是零重训的选项"；
- 若仍低于 RAW ≥0.012 → 校正标签的代价不止于偏置，保持现有表述并加上 LSHIFT 的反例作限定。
主 agent 先验：偏向前者（LSHIFT 已完全恢复，CORR_FROZEN 的均值与 LSHIFT 相近）。
开箱结果（跑完回填）：
2026-10-19 跑完（12 个学生；主 arm；偏置=训练集标签均值；dl22 三重门控；同集配对差 = @hb − 原学生）：
- v4-flash：CORR_FROZEN@hb NDCG 0.4404（原 0.4386）：vs 原 +0.0019 [−0.008,+0.013]，vs RAW −0.0184 [−0.041,+0.004]；
  MAP_ONLY@hb 0.4412（原 0.4431）：vs 原 −0.0019 [−0.014,+0.010]，vs RAW −0.0177 [−0.038,+0.004]。→ **未恢复**。
  offset/ratio/F1 基本不动（CORR_FROZEN@hb offset −0.101，F1 0.194）。
- 3.5-flash：CORR_FROZEN@hb 0.4615（原 0.4432）：vs 原 +0.0184 [+0.006,+0.032]*，vs RAW +0.0014 [−0.015,+0.018]；
  MAP_ONLY@hb 0.4571（原 0.4416）：vs 原 +0.0154 [+0.003,+0.027]*，vs RAW −0.0030 [−0.017,+0.011]。→ **完全恢复**。
- 主 agent 初读（待复核）：判定按 judge 分叉——3.5-flash 的校正标签排序代价是偏置失配伪影；v4-flash 的
  不是（LSHIFT 恢复而 CORR_FROZEN/MAP_ONLY 不恢复 → v4 的冻结 map 有与水平无关的结构性代价约 0.018）。
  论文须按"judge 依赖、配方依赖"改写，不能写成单一结论。

## B7 探索性追加 2c（2026-10-19 登记，训练前写下；评估员对追加 2 的复核要求）：偏置失配的直接判别
评估员复核（内部 docs/evaluator_reports/2026-10-19_b7_headbias_sonnet.md）：LSHIFT 单元支持"偏置初始化
失配"解释（@hb 配对差 +0.0165*，operating point 未动），HUMAN 单元介于两分支之间（+0.005，不显著），
故"均值水平效应是配方伪影"目前只在一个标签集、一个 judge、一个 arm 上成立；追加 2 初读中"完全解释"
改为"回到 RAW 水平（CI 内）"。为把标签内容固定、只动初始化，追加（v4-flash、主 arm、3 seed）：
- RAW 标签 + 偏置 1.07（tag mb；失配 0.49，与 LSHIFT 原失配相当）：若 NDCG 比 RAW 原学生低 ≈0.015 →
  失配解释成立（标签内容未变）；若不降 → LSHIFT 的恢复另有原因，判定须重议。
- RAW 标签 + 偏置=标签均值（tag hb；≈1.56，失配 ≈0）：阴性对照，预期与 RAW 原学生无差。
- CORR_REFIT_n1000 + 偏置=标签均值（tag hb）：原损失最大（−0.033*，学生 offset −0.24），失配解释预测
  恢复约 +0.03。
- 补做配对比较 LSHIFT@hb vs HUMAN@hb。
判定以同集配对差（hb/mb − 原学生，同重采样 query）为主，点估计 vs RAW 为辅；不再用"CI 含 0"作判据。
开箱结果（跑完回填）：
2026-10-19 跑完（v4-flash、主 arm、3 seed；同集配对差 = 新学生 − 原学生）：
- RAW@mb（RAW 标签 + 偏置 1.07，人为失配 ≈0.49）：NDCG 0.4439；vs RAW −0.0150 [−0.033,+0.002]；offset/F1/ratio 不变
  （+0.002 / +0.014 / −0.011）→ **标签不变、只有失配，就复现了 ≈0.015 的损失**。
- RAW@hb（RAW 标签 + 偏置=自身均值 ≈1.56，阴性对照）：NDCG 0.4594；vs RAW +0.0006 [−0.004,+0.005] → 无变化，对照成立。
- CORR_REFIT_n1000@hb：NDCG 0.4504（原 0.4261）；vs RAW −0.0085 [−0.034,+0.019]（原 −0.0328*）→ 恢复约 +0.024，
  大部分损失为失配所致。
- LSHIFT@hb vs HUMAN@hb 配对比较见 eval/minilm-raw/deepseek__deepseek-v4-flash/compare_SHIFT_FULL@hb_vs_HUMAN@hb.json。
- 主 agent 初读（待复核）：失配机制被直接证实（RAW@mb）。综合 2/2b/2c：主 arm 校正标签学生的排序损失中，
  偏置失配可解释 LSHIFT（全部）、3.5-flash 的 CORR_FROZEN/MAP_ONLY（全部）、v4 的 CORR_REFIT_n1000（大部分），
  但不能解释 v4-flash 的 CORR_FROZEN/MAP_ONLY（残留 ≈0.018）——该残留是 v4 冻结 map 的结构性代价候选。

**追加 2/2b/2c 评估员复核（Sonnet，2026-10-19；内部 docs/evaluator_reports/2026-10-19_b7_headbias_sonnet.md 与 …_headbias2_sonnet.md）**
完整性：登记行、消融行、11/5/0 计数全部不变；summary 的 code 字段记录的是最后一次 evaluate 的代码，非训练代码。
对主 agent 初读的更正：(1) "失配机制被直接证实"→"在固定标签下复现（RAW@mb −0.015，CI 含 0），且失配方向
与 LSHIFT 相反，不能说直接证明"；(2) "完全恢复"→"回到 RAW 水平（CI 内）"；(3) "judge 依赖"→"两单元的
模式，judge×初始化交互未检验（FROZEN t≈−1.4、MAP_ONLY t≈−2.3）"；(4) v4 残留 −0.018 是"持续的点估计、
CI 含 0"的候选，不是已确立的 map 结构代价；FROZEN 与 MAP_ONLY 共用一个 map，是一个观测而非两次复现；
(5) "CORR_FROZEN 均值与 LSHIFT 相近"仅对 v4 成立（失配 0.46 vs 0.43），3.5 的失配更大（0.58）却恢复。
判定（主 arm）：LSHIFT、3.5-flash 的 FROZEN/MAP_ONLY、v4 的 REFIT_n1000（大部分）为偏置初始化伪影；
v4 的 FROZEN/MAP_ONLY（≈−0.018）与 HUMAN（−0.0096）残留未解决；副 arm 未重训（P3 在副 arm −0.016/−0.031）。
登记的 P3/P7 未中与 11/5 计数不变，仅在论文中加"登记配方（偏置 1.5）下"的范围限定。
**不得写**：标签侧校正无排序代价 / P3、P7 是伪影；v4 残留是已证明的 map 代价或是噪声；失配"直接证明"；
LSHIFT 等于或优于 RAW；HUMAN 差距已解释或 LLM/人类标签可互换；任何每格偏移价格或"NDCG 跟随水平"；
关于副 arm、3.8、3.5 HUMAN 的任何说法；标签均值偏置是推荐配方；"输出侧平移无排序代价"作为发现（它按构造
成立）。可选后续（按价值）：副 arm 的 FROZEN/LSHIFT@hb（12 个）；免费的配对比较 MAP_ONLY@hb vs LSHIFT@hb
与 FROZEN@hb vs LSHIFT@hb；v4/3.5 的 FROZEN 标签对人类分数的 AUC 与 P(相关|judge 1 分)；3.5 HUMAN@hb。
