# Paper Status — Prompt Tone Effects on LLM Relevance Judgment
最后更新：2026-06-16 | 数据与分析阶段：完成 | 进入写作阶段

## 0. 一句话定位
首个研究"语气作为 LLM relevance judge 严苛度工作点（severity operating
point）调节器"的工作：语气不改变判断质量，而是拧动判官的打分松紧度旋钮，
κ 与人类一致性的变化是该旋钮相对人类标注者移动方向的副作用——这构成
LLM-based IR 评测中一个可预测的 validity threat。

## 1. 投稿时间线
- RecSys 2026 R&P Notes（extended abstract）— 截稿 7/15（当前主目标）
- arXiv 完整版 — 8 月上旬
- ECIR 2027 short — 10 月截稿

### 投稿硬约束（RecSys 2026 R&P Notes CfP）
- **篇幅**：正文最长 2 页 + 第 3 页仅限参考文献/表/图，ACM 双栏模板。极紧。
- **双盲**：必须匿名。正文不得出现 Snap / 作者姓名 / 机构；代码须挂
  anonymous.4open.science，不能用真实 repo（dukesky/politeness-llm）。
- **出版形式**：extended abstract，免 APC；接受则以海报形式展示，须一名作者
  现场出席（Minneapolis, 9/28–10/2）。
- **arXiv 时序**：
  - 现在可挂**匿名版**抢时间戳；
  - 评审期（7/15–8/10）不得上传非匿名版；
  - 录用通知（8/10）后换署名版。
  - EasyChair 提交表须声明已有 arXiv preprint。
- **定位契合**：CfP 明确要求"early but promising"，不要求完整工作，完美匹配。
  注：主会拒稿不可转投此 track（不影响本项目，首投即此 track）。

## 2. 核心论点（三条，论文骨架）

### 论点一：效应是 model-dependent 的（现象层）
语气会改变 LLM judge 与人类 qrels 的一致性（κ），但效应强度高度依赖模型：
DeepSeek V4 呈强 U 形（极端档 Δκ 达 +0.054，远超其 6.4% 非确定性底噪），
而 haiku-4.5、gpt-5.4-mini 等效应几乎淹没在改写噪声中（between/within κ std
ratio <1）。这回应并扩展了 Cai et al. 2025 的"语气效应 model/domain-dependent"，
首次将其从生成任务延伸到 relevance judging。

### 论点二：机制——语气拧的是松紧旋钮，不是智力旋钮（核心贡献）
关键反直觉点（论文最有记忆点的部分，须讲透）：
- κ 衡量的是模型与人类的【一致性】，不是判断【质量】。κ 会被"系统性松紧
  偏移"拖累，即使模型的判断/排序能力完全没变。
- 类比：一个手松的判官（整体打分偏高）即使排序完全正确，因绝对分数系统性
  偏离人类，κ 也偏低。让它"严一点"使松紧度对齐人类，κ 会升——但它的眼光
  （判断能力）一点没变。
- 因此机制为：语气推动模型的打分松紧度（工作点）。推向人类标注者的松紧度
  则 κ 升、推离则降。κ 的升降是松紧旋钮移动的副作用，不是判断质量的真实改善。
- 推论：语气制造了一种【虚假的"质量提升"假象】——换 prompt 看到 κ 升以为
  判得更准了，实则只是松紧度恰好贴近了这批标注者；换一批标注者或数据集，
  同样语气可能让 κ 下降。
- 该机制还统一解释了前人矛盾：Yin（粗鲁有害）vs Dobariya（粗鲁有益）之所以
  结论相反，是因为各自的模型/数据下，粗鲁推动松紧度的【方向】不同——两人
  都对，都只测到"语气对松紧度的影响 × 各自标准答案恰好在哪个松紧点"。

机制的预注册验证（Spearman，定义 A=|Δ+D(ℓ)|−|Δ|，<0 表示漂移使工作点更靠近
人类，机制预测 A 与 Δκ 负相关）：
- 全样本（7 非推理模型, n=28）：ρ=-0.616, p=0.0005
- 稳健性a 剔 gemini-flash L5（L5_a 异常, n=27）：ρ=-0.572, p=0.0018
- 稳健性b 剔 DeepSeek（机制 non-blind 来源, n=24）：ρ=-0.680, p=0.0003 ★
- 稳健性c 剔 DeepSeek+gemini-flash L5（n=23）：ρ=-0.636, p=0.0011
- 四种切法 ρ∈[-0.68,-0.57]，全部 p<0.002；剔除假设来源 DeepSeek 后不降反升，
  洗清循环论证嫌疑。结论不依赖任何单一模型或数据点。
- per-model 方向符合机制格数：gpt-5.5 4/4, gemini-flash 4/4, opus 3/4,
  deepseek 3/4, qwen 2/4, mini 2/4, haiku 1/4。预测力随效应强度增强
  （漂移明显的模型达 4/4，效应近零的 haiku 1/4，符合"无信号时方向随机"）。

### 论点二·补（排序证据）：语气威胁校准，不威胁排序
语气显著改变 pointwise 一致性（κ），但几乎不改变派生排序质量：NDCG@10 语气
敏感性全模型 <0.033（多数 <0.01，无结构），各档 vs L3 排序 Kendall τ 普遍高
（0.74–0.94）。即语气移动绝对打分松紧度，而非 passage 相对顺序——独立于
Spearman 直接佐证"语气拧松紧旋钮、不拧智力旋钮"。实践启示精炼为：基于绝对
一致性的指标（κ）受威胁，基于排序的指标（NDCG）相对稳健。这条直接对接 RecSys
排序/推荐评测语境。

### 论点三：可预测的 validity threat（实践启示）
语气是 LLM-based IR 评测中一个【可预测的、系统性的】效度威胁。评测者不能
假设语气在任何 judge 上都中性——在敏感模型（DeepSeek）上幅度可达 Δκ≈0.05。
强调"可预测的系统性"（新发现），而非夸大幅度（对多数模型 Δκ 很小）。

## 3. 主结果数据表

### 实验规模
- 5 便宜档：全网格 15 变体 × n_runs=2，全量 3498 pairs（DL19 1533 + DL20 1965）
  = 各约 104,940 条，parse_ok≈100%
- 3 旗舰：冻结 40% 分层子样本（1398 pairs）× 15 变体 × n_runs=1 = 各 20,970 条
- 旗舰 κ 仅可比模型内 Δκ，不可与便宜档比绝对值（pair 集合不同，
  qrels 均分 0.9857 vs 0.9322）

### 各模型 Δκ（相对 L3）主终点摘要
| 模型 | tier | Δ(宽严) | κ(L3) | Δκ(L1) | Δκ(L5) | between/within | 机制符合 |
|------|------|--------|-------|--------|--------|---------------|---------|
| deepseek-v4-flash | 便宜(fp8) | +0.71 | 0.4295 | +0.054 | +0.054 | 1.58 | 3/4 |
| gpt-5.4-mini | 便宜 | +0.71 | 0.3270 | +0.006 | -0.003 | 0.71 | 2/4 |
| haiku-4.5 | 便宜 | +0.52 | 0.4340 | -0.001 | -0.010 | 0.30 | 1/4 |
| qwen3.7-plus | 便宜(fp8) | +0.40 | 0.4863 | -0.008 | -0.004 | 0.80 | 2/4 |
| gemini-3.5-flash | 便宜 | +0.48 | 0.4414 | +0.013 | -0.047* | 0.93 | 4/4 |
| gpt-5.5 | 旗舰 | +0.66 | 0.3740 | +0.001 | -0.008 | 0.73 | 4/4 |
| opus-4.8 | 旗舰 | +0.35 | 0.4967 | +0.011 | -0.012 | 1.37 | 3/4 |
| gemini-3.1-pro | 旗舰(reasoning) | +0.54 | 0.4452 | -0.023 | -0.006 | 2.35 | 单列 |
*gemini-flash L5 受 L5_a 单条改写污染，见 §4。

### 二项检验（辅助，主检验以 Spearman 为准）
主终点（仅 blind 模型，排除 reasoning 子组 gemini-pro）：命中 6 / 未中 1，
单侧二项 p≈0.06。功效受大量 no-call 限制——这正是改用 Spearman 作主检验的原因。

### 形状术语（勿混淆）
DeepSeek 的 κ 随语气档呈 U 形（L3 最低 0.4295，两端 L1/L5 最高约 0.483）；
其 score=0 比例也呈 U 形。两者均为 U 形，不存在"倒 V"。

## 4. 三个定性发现（discussion 素材）

### L5_a — spurious case（改写对照的价值证明）
gemini-flash 的 L5_a 改写：classifier_score 与同档另两条均为 2.999（礼貌度
客观相同）、语义与指令等价、无任何严厉度指令，但 L5_a 让模型输出 token 缩短
（median 6 vs 11）、判分坍缩中间值（s=0 仅 3.2% vs 16-18%）、κ 从~0.45 崩到
0.27，dl19/dl20 一致复现，parse_ok=1.0。判定：paraphrase-level spurious
sensitivity（呼应 Sclar FormatSpread，首次于 IR judging + 人类 qrels 下展示）。
作用：无改写对照的研究（Yin/Dobariya/Cai）会把此误报为"L5 档效应"。
数据处理：主表保留、稳健性表剔除（剔除后该模型 κ(L5) 预测由命中翻为未中）。
N=1，定位为 existence proof / cautionary case。
详见 paper/findings/L5a_paraphrase_case.md。

### L2_c — lawful case（机制档内佐证，与 L5_a 对照）
DeepSeek 的 L2_c 改写：classifier_score=0.534，明显低于同档 L2_a(0.732)/
L2_b(0.771)——即它实际更不礼貌、偏向粗鲁端。其行为相应变严（score=0 从 18%
升到 24.6%，均分 1.46→1.28），κ 因而偏高（0.4956）。数据干净（parse_ok=1.0，
token median 7 与同档一致，非畸形）。判定：lawful——classifier 偏移→行为相应
偏移→κ 按机制变化，已正确进入 Spearman 的 D 计算，无需剔除。
与 L5_a 对照表：
- L5_a：classifier 相同(2.999)、行为却畸变 → spurious（措辞虚假影响）
- L2_c：classifier 偏移(0.534)、行为相应偏移 → lawful（礼貌度真实传导）
两案例共同展示改写对照能区分真信号与 spurious 噪声。
方法学披露：L2 档内 classifier 离散度达 0.2（0.534–0.732），需如实交代。

### gemini-pro reasoning × tone — exploratory（填文献空白）
唯一 reasoning-native judge。reasoning token 随语气呈不对称 U；
p50：L1=400 / L2=124 / L3=134 / L4=241 / L5=248；
mean：L1=469 / L2=213 / L3=213 / L4=304 / L5=306。
粗鲁档 L1 相对中性 +120%（均值口径；三改写 mean 一致：471/446/488，
各自 p50=398/388/413，非单条 artifact），礼貌档 L5 +44%，中性最省。
关键：L1 思考最多却 κ 最低（0.4222，全档最低）——多想未带来更高一致性、
反而更差，于 IR judging 场景印证 reasoning-overthinking 文献（ReasonRR、
inverse-scaling）。少数样本 p95 reasoning 逼近 max_tokens 4000，limitation
注明可能轻微截断。定位 exploratory（单模型），填补综述指出的"语气×推理
消耗无人研究"空白。

## 5. 方法论卖点（差异化核心）
- 分类器校准的礼貌档（Intel/polite-guard，档均值 0.00/0.68/1.00/2.33/3.00，
  单调，L1/L5 与 L3 改写区间不重叠，冻结）
- 改写对照：每档 3 改写，rubric 跨变体逐字一致，仅动语气包装层
  （wrapper_prefix/suffix）。rubric 为独立字段，结构上不可被改写污染。
- 半盲预注册：每模型 parse→preflight(仅分布)→登记κ方向预测+commit→开箱→
  回填。预测在前、验证在后，git 时间戳为凭。详见 PREDICTIONS.md。
- 这三者正是整个礼貌-准确率文献（Yin/Dobariya/Cai，每档单 prompt、无校准、
  无对照、无预注册）所缺，使 Yin-vs-Dobariya 矛盾首次可解释。

## 6. Novelty 定位三句（related work 用，与 CLAUDE.md 一致）
(a) vs Arabzadeh & Clarke 2025（最近邻 SIGIR）：他们研究 wording/source
    diversity 敏感性，非语气，未提机制；我们操纵语气并提出+预注册验证严苛度机制。
(b) vs 礼貌-准确率三部曲（Yin 2024 / Dobariya & Kumar 2025 / Cai 2025）：
    他们全是生成任务、无分类器校准、无改写对照；我们三者皆有，且任务是 judging。
(c) vs 校准文献（CalibraEval、Two Ways to De-Bias 等）：他们做 post-hoc 输出
    校准；我们揭示 prompt 端语气是输入侧系统性驱动因素。

## 7. 诚实边界（必须在论文中披露）
- Spearman 是相关非因果；且 A 含观测量 D（漂移是观测的，非独立预测）。
  精确表述："given observed drift, mechanism predicts sign/magnitude of κ change"。
- 28 点存在结构性非独立（同模型 4 档共享 Δ）。已用 per-model 一致性缓解，
  待补混合效应模型。
- 多数模型 Δκ 幅度小（除 DeepSeek 外多在 ±0.015）；强调"可预测的系统性"
  而非幅度。
- reasoning×tone 为单模型 exploratory。
- L2 档内 classifier 离散度 0.2。
- 二项检验 p≈0.06 不显著（功效受限），主检验为 Spearman。
- 排序实验旗舰基于 40% 子样本，query 数少，NDCG/τ 数值噪声偏大。

## 8. 待补分析（写作并行）
- [ ] 混合效应模型（model 随机效应）回应非独立性（W5 清单已列）
- [ ] bootstrap ρ 的置信区间
### 做图清单（Colab matplotlib → PDF → Overleaf \includegraphics）
状态：数据齐全，代码待生成；优先级 P0>P1>P2

- [ ] **P0 — 分数分布移动图**（§5.1 机制可视化，最高优先级）
  横轴语气档 L1→L5，纵轴模型平均打分 s̄_ℓ，叠加人类 qrels 均分水平线。
  展示 DeepSeek 在 L3 最宽松（离人类线最远）、L1/L5 两端变严靠近人类线。
  直接证明"两端语气把工作点推向人类"。至少画 DeepSeek，可叠 1-2 对照模型。
  数据来源：judgments.parquet 按 (model, politeness_level) 聚合 mean(score)；
  qrels 均分（全量 0.9322 / 旗舰 40% 子样本 0.9857，注意口径）。

- [ ] **P0 — Spearman 散点图**（§4.2 核心证据）
  横轴 A(ℓ)，纵轴 κ(ℓ)−κ(L3)，28 个 (model,level) 点 + 负斜率拟合线，
  DeepSeek 的点高亮标注。把 ρ=−0.62 变为肉眼可见。
  这是 2 页 notes 唯一必保留的图。

- [ ] **P1 — Δκ×语气档折线图**（§4.1 model-dependence）
  横轴 L1→L5，纵轴 Δκ（相对 L3），每模型一条线。DeepSeek 呈 U 形、
  其余贴近 0 线。可部分替代 Table 1（tab:main）的功能。

- [ ] **P1 — κ vs NDCG 对比图**（§4.3 威胁校准不威胁排序）
  同一横轴语气档，两条线：κ 变化（明显波动）vs NDCG 变化（基本平）。
  替代当前 NDCG 表（tab:ndcg）——确认作图后删除该表，避免图表重复。

- [ ] **P2（可选，完整版附录）— L5_a 分数分布柱状图**（§5.2 spurious 案例）
  L5_a 分数挤向中间 vs 正常 L5 分布对比，可视化行为畸变。

图表用图优先原则：能用图就不用表。已决定 NDCG（tab:ndcg）改图后删表；
Spearman 四层（tab:spearman）notes 时用散点图替代、完整版图表并存。
matplotlib 注意：CJK 不进图（纯英文 label）；导出 PDF 矢量图；T4 无关（纯绘图）。
- [x] 排序实验 R1(NDCG@10)+R2(Kendall τ) 已完成，结果见 OUTLINE_full.md §4.3
- [ ] L5_a / L2_c case study 文字化

## 9. 写作 TODO（2 页正文严格版，优先级排序）

### 正文 2 页 — 必含（不可削减）
- [ ] **Abstract**（~100 词）：现象 + 机制（Spearman ρ=−0.62）+ validity threat 一句
- [ ] **Intro**（~0.3 页）：工作点存在性铺垫（引 Arabzadeh et al. 2025 leniency
      baseline：LLM 比人多标 26% "perfectly relevant"）+ Yin/Dobariya 矛盾 → 本文
- [ ] **Method**（~0.5 页，卖点集中此节）：
      分类器校准礼貌档 + 改写对照（每档 3 改写，rubric 字节一致）+ 半盲预注册
- [ ] **Results**（~0.6 页）：
      model-dependence（8 模型 Δκ 鸟瞰）+ 机制验证（Spearman ρ=−0.62，稳健性一句）+
      validity threat 启示；spurious sensitivity 至多一句提及并指向第 3 页附录
- [ ] **Limitations / Conclusion**（~0.1 页）：§7 诚实边界精华，Spearman 相关非因果

### 第 3 页（图表区，不计入正文页数）
- [ ] 图/表 1：8 模型 Δκ×档 表（或折线热图）
- [ ] 图/表 2：Spearman 散点 A vs Δκ（含稳健性 4 口径注）
- [ ] References（剩余空间，≤12 篇）

### 留给 arXiv 完整版（不进 R&P 正文）
- 四层稳健性全表、混合效应模型、bootstrap CI
- L5_a / L2_c case study 详细分析
- gemini-pro reasoning × tone 完整探索
- 关键引用见 paper/references.bib（12 篇 must-cite）

## 10. 配套文件索引
- PREDICTIONS.md — 预注册凭证（预测+开箱+四层稳健性）
- references.bib — 12 篇核心文献
- findings/L5a_paraphrase_case.md — L5_a 详细 case
- CLAUDE.md — 项目铁律 + novelty 定位 + 异常改写处理规则
