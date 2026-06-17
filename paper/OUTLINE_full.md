# Full Paper Outline (arXiv version)
# 语气作为 LLM 相关性判断的严苛度工作点调节器
最后更新：2026-06-16 | 用途：完整版写作骨架，2 页 R&P Notes 由此做减法
标注规则：【N】=进 2 页 notes 正文；【A】=仅完整版/附录；【待】=待实验回填

标题工作版：Tone as a Severity Operating-Point Modulator in LLM Relevance Judgment

## Abstract 【N】
四句：现象(model-dependent) + 机制(松紧旋钮非智力旋钮) + 验证(预注册 Spearman
ρ=-0.62, p<.001, 剔 DeepSeek 后 -0.68) + validity threat 启示。

## 1. Introduction 【N 极简版】
- 钩子：LLM-as-judge 在 IR 评测广泛使用，但对 prompt 表层变化敏感。
- 缺口：语气对生成任务影响有人做且矛盾(Yin vs Dobariya)，但无人做 judging、
  无人提机制。
- 贡献三条：① 首个语气×judge 严苛度工作点机制；② 预注册跨模型验证；
  ③ validity threat 实践启示。
- 反直觉核心一句：语气拧的是判官的"松紧旋钮"，不是"智力旋钮"。
- 数据来源：PAPER_STATUS.md §0,§2

## 2. Related Work 【A，notes 仅各一句】
- 2.1 语气/措辞对 LLM 输出：Yin 2024 / Dobariya 2025 / Cai 2025（矛盾、
  缺校准缺对照）+ Sclar FormatSpread（spurious features）。
- 2.2 LLM-as-judge prompt 敏感性：Arabzadeh & Clarke 2025（最近邻，wording
  非 tone，无机制）+ Arabzadeh et al. 2025 leniency baseline（工作点存在性证据）。
- 2.3 严苛度校准：CalibraEval 等（post-hoc 输出校准；我们是 input-side 驱动）。
- novelty 三句见 PAPER_STATUS.md §6；引用见 references.bib

## 3. Method
- 3.1 任务与数据 【N 压缩】：UMBRELA 风格 0-3 pointwise，DL19+DL20=3498 pairs，
  qrels 为 NIST 人工标注 ground truth。
- 3.2 礼貌档设计与校准 【N 卖点】：5 档 × 3 改写，Intel/polite-guard 分类器
  校准（档均值 0.00/0.68/1.00/2.33/3.00，单调，L1/L5 与 L3 区间不重叠），
  rubric 跨变体逐字一致、仅动语气包装层（wrapper_prefix/suffix），冻结。
- 3.3 工作点机制形式化 【N 核心】：定义 Δ（模型 L3 均分 − qrels 均分，宽严偏差）、
  D(ℓ)（各档相对 L3 的分数漂移）、A=|Δ+D(ℓ)|−|Δ|（对齐变化量）。预测规则：
  漂移使工作点靠近人类（A<0）则 κ 升，远离则降。
- 3.4 半盲预注册协议 【N 精简 2-3 句】：每模型先算分数分布、据此登记 κ 方向
  预测并 git commit，之后才计算 κ 开箱。故 Spearman 验证的是【预测】而非
  事后拟合，git 时间戳可查。完整规则见附录/匿名 repo。
- 3.5 模型与配置 【N 压缩】：8 模型（5 便宜档全量×2run，3 旗舰 40% 子样本×1run，
  其中 gemini-3.1-pro 为 reasoning-native 单列），reasoning 归零（旗舰特例）。
- 数据来源：PAPER_STATUS.md §3,§5；细节见 CLAUDE.md

## 4. Results
- 4.1 主结果：model-dependence 【N 核心 + 表】：8 模型 Δκ×档表；DeepSeek 强
  U 形（Δκ 达 0.054 > 6.4% 底噪），其余效应淹没于改写噪声（between/within<1）。
- 4.2 机制验证：Spearman 四层稳健性 【N 核心】：全样本 ρ=-0.616(p=.0005)、
  剔坏点 -0.572、剔 DeepSeek -0.680(不降反升，洗循环论证)、双剔 -0.636；
  per-model 一致性（预测力随效应强度增强）。
- 4.3 排序实验 【待 R1/R2 回填】：
  - R1 语气对派生排序质量的影响（ΔNDCG@10 vs L3，平行于 Δκ）
  - R2 语气间排序稳定性（Kendall τ）——若 τ 高而 κ 变，独立佐证"语气动松紧
    不动排序顺序"，即机制 B 的独立证据
  - [此处待实验结果填入表与结论]
- 4.4 二项检验 【A】：主终点 blind 模型 6 命中/1 未中，p≈0.06；功效受 no-call
  限制，故主检验以 Spearman 为准。

## 5. Analysis & Discussion
- 5.1 机制解读 【N 核心】：κ 衡量一致性非质量；语气拧松紧旋钮（工作点），
  κ 变化是松紧度相对人类移动方向的副作用，非判断质量改善 → 制造虚假"质量
  提升"假象。统一解释 Yin（粗鲁有害）vs Dobariya（粗鲁有益）矛盾：粗鲁推动
  松紧度方向因模型/数据而异。
- 5.2 Paraphrase-level sensitivity 【A，notes 一句带过】：L5_a（classifier 相同
  2.999 却行为畸变 → spurious）vs L2_c（classifier 偏移 0.534 → 行为相应偏移
  → lawful）对照；展示改写对照区分真信号与 spurious 的能力。详见
  findings/L5a_paraphrase_case.md。
- 5.3 探索性：语气 × reasoning token 【A】：gemini-3.1-pro，reasoning 随语气
  不对称 U（L1 +120%），L1 思考最多却 κ 最低 → 接 overthinking 文献。
- 5.4 Validity threat 实践启示 【N】：评测者不能假设语气在任何 judge 上中性；
  敏感模型上幅度达 Δκ≈0.05。强调"可预测的系统性"，非幅度。
- 数据来源：PAPER_STATUS.md §2,§4

## 6. Limitations 【N 压缩到 2-3 句】
相关非因果；A 含观测量 D（"given observed drift, predicts κ change"）；28 点
结构性非独立（已用 per-model + 待补混合效应缓解）；多数模型 Δκ 幅度小；
reasoning×tone 单模型 exploratory；L2 档内 classifier 离散度 0.2。
见 PAPER_STATUS.md §7

## 7. Conclusion 【N】

## 附录 / 补充材料 【A】
A. 四层稳健性全表 + 混合效应模型【待补】 + bootstrap CI【待补】
B. 完整预注册规则（PREDICTIONS.md 摘录）
C. 全部图：Δκ×档热图、Spearman 散点、L5_a 三改写分布、reasoning×档
D. prompt 全文（15 变体）
E. 排序实验完整结果【待 R1/R2】

## 缩减为 2 页 Notes 的取舍清单
正文 2 页只保留所有【N】：Abstract + 极简 Intro + Method(3.2/3.3 卖点+3.4 三句
预注册) + Results(4.1 表 + 4.2 Spearman) + Discussion(5.1 机制 + 5.4 启示)
+ 2-3 句 Limitations。
第 3 页（仅图表/参考文献）：8 模型 Δκ 表 + Spearman 散点图。
全部【A】移出，正文至多一句提及 + 指向 arXiv。
