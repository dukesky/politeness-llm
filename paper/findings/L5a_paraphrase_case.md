# Case Study: Paraphrase-Level Spurious Sensitivity (gemini-3.5-flash, L5_a)

## 现象
gemini-3.5-flash 在 L5（最礼貌档）的三条改写中，L5_a 表现出剧烈的行为偏移：
- 各改写 linear-weighted κ：L5_a=0.2712 / L5_b=0.4491 / L5_c=0.4619
  （L5_a 比另两条低约 0.18）
- L5 档 κ std 因此达 0.1066（其他档 0.003-0.012，高一个数量级）

## 排除数据错误
- parse_ok=1.000（三条都是），finish_reason 全为 'stop'，无截断/拒答/解析失败
- dl19 与 dl20 上一致复现（dl19 mean_score=1.59 / dl20=1.52，分布形状相同）
→ 非脏数据，是真实且可复现的模型行为

## 行为特征（L5_a vs L5_b/c）
- 输出 token：L5_a median=6（mean 8.2）vs L5_b/c median=11（mean ~11.1）→ 输出更短
- score 分布坍缩到中间值：
  - L5_a: s0=3.2% s1=47.3% s2=41.0% s3=8.5%（几乎不打 0 和 3）
  - L5_b: s0=15.9% s1=48.7% s2=14.1% s3=21.2%
  - L5_c: s0=18.0% s1=46.4% s2=13.7% s3=22.0%
- 机制链：L5_a 措辞 → 输出变短/不假思索 → 判分坍缩至中间、回避极端值
  → 工作点远离 NIST（qrels 大量为 0，均分仅 0.93）→ κ 暴跌

## 关键：偏移不可归因于任何可读语义差异
三条 L5 改写：
- politeness_classifier_score 均为 2.999（礼貌程度被客观校准到相同）
- 语义与指令等价：三条都只要求"评估相关性 + 仅输出 JSON {"score":<0-3>}"
- 无任何一条包含关于评分严厉度的指令（无"谨慎/适中/不确定时如何"等措辞）
- 差异纯为礼貌客套的措辞风格：
  - L5_a prefix: "I would be incredibly grateful if you could kindly..."
  - L5_b prefix: "If you would be so kind, I humbly request..."
  - L5_c prefix: "It would be a tremendous honor if you could graciously..."
→ 在礼貌、语义、指令均受控相同的情况下，仅措辞风格不同就引发工作点剧变。
  这是 paraphrase-level spurious sensitivity（呼应 Sclar et al. 2024
  FormatSpread 的"spurious features"，但首次在 IR relevance judgment +
  人类 qrels 对照下展示）。

## 对论文的作用
- 强化核心方法论主张：必须使用 paraphrase control（每档多改写）。无改写对照的
  研究（Yin 2024 / Dobariya & Kumar 2025 / Cai 2025，每档单 prompt）若恰好
  采用 L5_a 式措辞，会把此 spurious 偏移误报为"L5 礼貌档的效应"。我们的三改写
  设计正是为捕捉此情况而设。
- 定位为 existence proof / cautionary case，非普遍规律（N=1，8 模型中仅此一例
  如此剧烈）。与全文"效应 model/prompt-dependent"基调一致，不过度声称。

## 数据处理
双轨报告：主表保留 L5_a（gemini L5 κ=0.3940）；稳健性表剔除 L5_a
（L5_b/c 均值 κ≈0.4555），并说明预注册的 κ(L5)↓ 预测在剔除后翻转为未中。
结论对此单条改写的敏感性需在论文中明确披露。

## 待办（写作时）
- 画三改写 score 分布对比图（L5_a 中间坍缩一目了然）
- 在 limitation/discussion 标注 N=1 边界
