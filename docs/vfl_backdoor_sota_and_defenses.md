# SOTA: 针对两方 VFL / 纵向拆分学习中后门与投毒攻击的防御

调研时间：本会话。检索范围 2022–2026。**逐条标注核验状态**——多项会议论文（AAAI /
USENIX / NDSS / IEEE / ACM）的 PDF 无法抓取，凡未取得全文的数字均标注为未核验。

本报告服务于一个两方 VFL 赛题，防御方是**持有标签的主动方 A**，可选动作被严格限制在
三个 hook 上：

- `trainingRepresentationHook(cutA, cutB, labels, sampleIds, epoch, batchIndex, context, generator) -> (cutA, cutB)`
- `cutGradientHook(gradientB, passiveCut, labels, sampleIds, epoch, batchIndex, context, generator) -> gradientB`
- `inferenceRepresentationHook(embeddingA, embeddingB, sampleIds, context, generator) -> (embeddingA, embeddingB)`

A 可以新增辅助模块（可含可训练参数），但**不得替换或扩展主任务底模/顶模**，也不得改变
二者约定的输入输出接口；不得触碰 B 方底模、B 方优化器、B 方私有特征。

---

## 1. 两种攻击及其可检测性含义

**LFBA — "Label-Free Backdoor Attacks in Vertical Federated Learning"**，Shen, Huang,
Wan, Ye（武汉大学），AAAI 2025, 39(19):20389–20397。
[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34246) ·
[代码](https://github.com/shentt67/LFBA)

- 不需要任务标签：用**局部 embedding 的梯度**作标签语义代理来构造目标投毒集，再
  **"selectively switching poison samples to disrupt feature learning"**（否则触发器相对
  原始特征会被忽略/欠拟合）。
- 触发器位于 **B 方输入特征空间**，投毒样本的 embedding 由 B 方自己的底模产生，因而
  **天然在分布内**。**因此单纯基于 embedding 范数 / OOD 的检测对 LFBA 很可能失效。**
- 干净标签 + 固定触发器。[BVBench](https://arxiv.org/abs/2608.12962) Table I 归类为：
  标签获取方式 *Gradients*、触发空间 *Input*、模式 *Fixed*、无约束。
- 可利用信号：(i) "选择性替换"造成**跨方语义不一致**；(ii) LFBA **依赖梯度相似度**，
  因此 **cut 梯度 hook 是一个真实可用的杠杆**。

**VILLAIN — "VILLAIN: Backdoor Attacks Against Vertical Split Learning"**，Bai, Chen,
Zhang, Xu, Weng, Goodman（浙江大学 / 蚂蚁集团），USENIX Security 2023, pp. 2743–2760。
[USENIX](https://www.usenix.org/conference/usenixsecurity23/presentation/bai)

- 标签推断 + 数据投毒，随后在**上传的 embedding 中直接注入隐蔽的加性触发器**，并配合
  后门增强。触发器幅度按**高方差 embedding 维度的标准差**构造以换取隐蔽性。
- 关键：BVBench 指出该隐蔽性约束**并不奏效**——"adding such triggers can still lead to
  off-distribution embeddings… can be easily detected using the 99% quantile"。
- 作者声称对"隐私推断防御、后门检测/移除防御以及自适应防御"稳健 → **必须假设 B 知道
  你的防御**。

相关攻击：[BadVFL](https://arxiv.org/abs/2306.10746)（1% 投毒率下 ASR > 93%，含
SDD+SDP 模块）、HijackVFL、BackSplitVFL、LMP/PMP、BAEVFL。

---

## 2. VFL / VSL 专用防御

**D1 · VFLIP** — Cho et al., ESORICS 2024。
[arXiv:2408.15591](https://arxiv.org/abs/2408.15591) ·
[代码](https://github.com/blingcho/VFLIP-esorics24)

推理阶段在服务端用**掩码自编码器（MAE）**。对拼接 embedding 施加两种掩码目标
（"N−1→1" 与 "1→1"）；按参与方计算重构误差 → **参与方级异常检测 + 多数投票**；再做
**净化**：剔除被标记的 embedding，用存活者重构全部 embedding 后送入顶模。
**到两方的移植**：掩蔽 B、从 A 重构 B 是良定义的，但**多数投票退化为无意义**，必须
替换为对 A 自身重构误差的分位数阈值。

**数字（已核验摘要）**：ASR **84.4% → 7.57%**（CIFAR10 / CINIC10 / Imagenette /
NUS-WIDE / Bank-Marketing 平均）。检索到的一行结果表显示 CIFAR10/BadVFL 基线
70–74% 精度、ASR 96–100%，VFLIP **8.12%**（该行未取得完整表头）。
**精度代价存在争议**：ProtoGuard-SL 列 VFLIP 的主任务精度为 0.65（无防御 0.80）；
SecureSplit 称 VFLIP "suffers from high false positives"。

**D2 · SecureSplit** — Dou et al., **WWW 2026**, DOI 10.1145/3774904.3792484。
[arXiv:2601.14054](https://arxiv.org/abs/2601.14054)

算法已完整核验：(1) **UMAP** 降维压缩；(2) **多项式核变换**升维以放大可分性；
(3) 逐坐标**中位数** Λ；(4) **多数半径** R = min{R′ : #{‖Ē_k−Λ‖₂ ≤ R′} ≥ m/2}；
(5) **自适应半径** R_adp = (1 + 1/(α+σ))·R；保留满足 ‖Ē_k−Λ‖₂ ≤ R_adp 的样本。
**"多数"是 batch 内样本多数而非参与方多数 → 完美适配两方。** 四个数据集 × 五种攻击
（含 VILLAIN、BadVFL）× 七个基线。Table 1 首行：CIFAR-10 无防御 ACC 0.84 / ASR 0.76。
**完整表未取得（HTML 截断）。** 报告称 Trimmed-mean / Multi-Krum / HDBSCAN 在拆分学习
中"largely ineffective"。

**D3 · ProtoGuard-SL** — Shui, Jin, Dou, Gao, ICME 2026。
[arXiv:2604.03595](https://arxiv.org/abs/2604.03595)

类别原型 **p_c = Median({E_k | y_k=c})**；一致性表示
**v_k = [cos(E_k,p_1..p_|C|)]**；类别参考 **μ_c = Median{v_k | y_k=c}**；
非一致性得分 **s_k = ‖v_k−μ_{y_k}‖₂**；**共形 p 值**
p_k = (|{s∈S_c : s≥s_k}|+1)/(|S_c|+1)，当 p_k > α 判为良性。
**类条件、非跨参与方 → 两方适配性最好，且用到标签（A 的独有优势）。**
**数字（已核验 Table I，MA/ASR）**：

| 数据集 / 攻击 | 无防御 | ProtoGuard-SL | VFLIP | DP | MP | ANP |
|---|---|---|---|---|---|---|
| CIFAR-10 VILLAIN | 0.80 / 0.92 | **0.83 / 0.06** | 0.65/0.35 | 0.74/0.77 | 0.74/0.67 | 0.77/0.66 |
| CIFAR-10 SplitNN | 0.77 / 0.95 | **0.85 / 0.05** | 0.72/0.33 | 0.79/0.73 | 0.72/0.70 | 0.74/0.67 |
| CIFAR-10 BadVFL | 0.79 / 0.89 | **0.84 / 0.03** | 0.66/0.28 | 0.77/0.68 | 0.72/0.78 | 0.76/0.65 |
| SVHN VILLAIN | 0.83 / 0.95 | **0.87 / 0.05** | 0.76/0.29 | 0.80/0.91 | 0.82/0.88 | 0.80/0.75 |
| Bank Mktg VILLAIN | 0.84 / 0.95 | **0.85 / 0.08** | 0.73/0.39 | 0.82/0.89 | 0.75/0.52 | 0.77/0.67 |

每行主任务精度都**上升**，ASR 相对下降 86–97%。两种攻击都覆盖。
**本赛题的实测反驳见第 5 节**：其跨类近邻规则在本赛题真实数据上会抬高 ASR。

**D4 · "N Truths and a Lie: Consistency-Based Backdoor Defense for VFL"** — Ma, Zhang,
He, Chen, Du（武汉大学），**IEEE TIFS 21:3973–3988, 2026**，
[DOI](https://doi.org/10.1109/TIFS.2026.3675464)

**潜在掩码自编码器（LMAE）**评估**跨参与方语义一致性**；恶意 embedding"较少能从诚实
参与方预测出来"。随后识别攻击方并给出抗后门预测。7 种攻击、5 个数据集、4 个基线。
**数字需付费墙，未核验。**

**D5 · FILTER** — Hu, Chen, Wang（华东师大），**AAAI 2026**, 40(42):35490–35499，
[DOI](https://doi.org/10.1609/aaai.v40i42.40859)

训练阶段的双过滤器（embedding 过滤器 + 损失过滤器），在**训练后段**识别并移除投毒
样本，针对攻击方"掌握足够标签信息"的情形。5 个数据集 × 4 种攻击；检索到的片段显示
超参 `p_init ∈ {0.1, 0.15, 0.2}` 与一行 VILLAIN 配置。**数字未核验（AAAI PDF 无法解码）。**

**D6 · VFLMonitor** — Xu et al., **IEEE TIFS 20:4828–4843, 2025**，
[DOI](https://doi.org/10.1109/TIFS.2025.3564879)

攻击会造成**跨客户端的 embedding–标签关联差异**；通过**多客户端标签估计差异 + 多数
投票**检测，并加入**降低类内 embedding 方差**的训练正则项。
**投票部分在 N=2 时失效，但标签差异部分可移植**为辅助头不一致监控——正是 LFBA 的
"选择性替换"造成的信号。5 个数据集、2 种攻击、3 种场景；**具体数字未核验。**

**D7 · GradSent** — Singh et al., **ACM AsiaCCS 2026**, pp. 1849–1862，
[DOI](https://doi.org/10.1145/3779208.3805965)

用**时序一致性**替代在非 IID 下失效的**空间共识**：轻量**梯度自编码器**学习良性动态
流形，新更新按重构误差打分，配合**迭代自清洗**与**动态阈值**。报告在极端非 IID 下
阻断多数客户端后门且保持主任务；片段显示 **FPR 3.2% @ α=0.1**。
**时序思路可完美移植**——A 看到的是跨 epoch 的单条梯度流，且 hook 暴露 `epoch` /
`batchIndex`。

**D8 · SafeSplit** — Rieger et al.（TU Darmstadt），**NDSS 2025**，
[DOI](https://dx.doi.org/10.14722/ndss.2025.241698) ·
[arXiv:2501.06650](https://arxiv.org/abs/2501.06650)

服务端保存最近 N 个 backbone 的 FIFO。**静态指标**：S_t = DCT_low(B_t − B_{t−1})，
频率表示之间的欧氏距离。**动态指标**：针对参数更新方向的**旋转距离** R_D。两者均小
才判为良性（取最小 N/2+1 名之内）；否则**回滚**到最近良性检查点并跳过该客户端。
**适配性弱**：需要 U 形顺序 ≥3 客户端拆分学习与 head/tail 交换，属于禁止的协议改动，
且会触碰 B。**但两个指标本身是 A 对自己参数的纯计算**，可作为廉价辅助信号复用。
**数字未能取得。**

**D9 · CONFIRM** — Cheng, Zhang, Liu, He, Wang, Ran（中科大 / 中国移动），
**IEEE TIFS 21:4831–4845, 2026**，
[DOI](https://doi.org/10.1109/TIFS.2026.3683603)

训练后阶段，利用"防御方持有标签"的有利位置。核心洞察：**投毒样本在扰动下表现出异常
稳定的预测置信度**。CONFIRM 主动扰动 A 的顶模并测量置信度一致性。
**数字（已核验摘要）**：**AUROC > 99.9%**、F1 比 SOTA 高最多 **21.2 个百分点**，且对
自适应攻击稳健。**与 hook 结构契合度极好——不需要 B 配合，也不需要 embedding 统计量。**

**D10 · 互信息正则化防御** — Zou, Liu et al.，
[IEEE 10825545](https://ieeexplore.ieee.org/document/10825545)

在底模输出与标签之间加互信息正则项，同时降低标签推断与后门成功率。
**不适配**——它修改的是*客户端*的训练目标。会议/年份与数字均未确认。

**D11 · BVBench** — Zhao et al.（HKU），[arXiv:2608.12962](https://arxiv.org/abs/2608.12962)，
2026-08-13，**预印本**。[代码](https://github.com/HKU-TASR/BVBench)

最重要的校准论文：

- 在现实威胁模型下（无类别语义、无目标类标签、无标签空间大小、无顶模后验），
  **"no existing attacks remain feasible"**。从梯度/embedding 推断标签在训练过程中
  **不稳定**。
- **隐蔽性批判（对本赛题直接有用）**：只有 BackSplitVFL 与 VILLAIN 施加 embedding
  约束，而这些约束**"cannot hide trigger-injected embeddings from defenders"**——
  VILLAIN 的触发器可在 **99% 分位**被检出。
- **防御指引**：因为推理时所有参与方都必须在线，**"mitigating backdoor effects is
  more preferred than detection alone"**。Table II 按 Ref.-Free、Adv.-Kn.-Free、
  Proactive?、Multi-Target?、Insensitive-Hyperp? **以及 Utility Recovery** 给防御打分；
  VFLMonitor 与 VFLIP 是其中的 VFL 专用条目，多数在超参不敏感性与效用恢复上得分很差。
- **其防御结果表（含 KUHAR）未能取得**——所有渲染都在 Section VII 之前截断。

---

## 3. 相邻方向与经典防御

**SplitGuard**（Pasquini et al., WPES 2022，
[eprint 2021/1080](https://eprint.iacr.org/2021/1080)）——服务端注入**已知预期行为的
诱饵批次**并测量客户端的梯度响应，以检测训练劫持。**两方场景下用于检测的效费比很高**：
A 拥有自己的训练循环，可以注入诱饵并检查返回的 cut 梯度。

**BVG/VFGNN**（[arXiv:2410.11290](https://arxiv.org/abs/2410.11290)）——图数据，本赛题
不适用，但作为证据有意义：BVG 在**现有防御下仍保持 >99% ASR**。

| 防御 | 出处 | 适配本 hook？ | LFBA | VILLAIN | 数字 |
|---|---|---|---|---|---|
| **Neural Cleanse** | [IEEE S&P 2019](https://ieeexplore.ieee.org/document/8835365) | 部分——可在 embeddingB 子空间反演触发器 | 弱（触发器在 B 的*输入*空间） | 好 | 经典异常指数 ~2.0；未重新核验 |
| **ABL** | [ICLR 2021](https://openreview.net/forum?id=cAw860ncLRW) | **完全适配**——隔离早期损失最小的 ~1% 样本，再梯度上升遗忘 | 可能 | 可能 | ASR 近零、精度略降；**未核验** |
| **Fine-Pruning** | [arXiv:1805.12185](https://arxiv.org/abs/1805.12185) | **不适配**——修改模型参数 | – | – | ASR→0%、精度 −0.4%（已核验） |
| **STRIP** | [arXiv:1902.06531](https://arxiv.org/abs/1902.06531) | 部分——在 hook (c) 扰动 embeddingB 并测输出熵 | 仅代理 | 好 | FAR <1% @ FRR 1%；CIFAR-10/GTSRB 0%/0%（已核验） |
| **SCAn** | Tang et al., NeurIPS 2021 | 原则上可用；逐类、事后 | 中 | 中 | 未核验 |
| **Krum** | [arXiv:1703.02757](https://arxiv.org/abs/1703.02757) | **不适配**——需要 ≥2f+3 个 worker | – | – | SecureSplit 称在 SL 中无效 |
| **Trimmed-mean / median** | [arXiv:1803.01498](https://arxiv.org/abs/1803.01498) | 跨 worker **不适配**；作为**样本级**坐标统计**可用** | ✓ | ✓ | – |
| **FLAME** | [USENIX Sec 2022](https://www.athene-center.de/en/research/publications/flame-taming-backdoors-in-federated-learning-3233) | **不适配**——需要 ≥3 客户端 | – | – | – |
| **Soteria** | [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Soteria_Provable_Defense_Against_Privacy_Leakage_in_Federated_Learning_From_CVPR_2021_paper.html) | 可作为变换——但**反效果**（摧毁信号） | ✗ | ✗ | – |
| **梯度裁剪 / 加噪 / DP-SGD** | 作为防御见 [BadVFL §5.4](https://ar5iv.labs.arxiv.org/html/2306.10746) | **就是 hook (b)** | ✓ | ✓ | **方差 1e-3 → CIFAR-10 ASR 30% 且 TAR 几乎不降；1e-4 → ImageNet ASR 崩但 TAR −20%**（已核验） |
| **白化 / Mahalanobis OOD** | [arXiv:2609.23117](https://arxiv.org/abs/2609.23117)（预印本） | 可用，免费、确定性 | 弱 | **强** | 关键告诫：白化范数是 Mahalanobis 异常度，**高斯尾阈值可能差几个数量级** |

---

## 4. 决定设计方向的结构性结论

**整个 VFL 防御文献建立在"把可疑参与方与其他参与方对比"之上，而两方场景下这是退化的。**
VFLIP、VFLMonitor、"N Truths and a Lie"、SafeSplit、Krum、Trimmed-mean、FLAME 都需要
≥3 方。必须用以下三种"多数"之一替代，且都能在现有 hook 里实现：

1. **类条件多数**——同类样本之间。即 ProtoGuard-SL 的 `Median({E_k | y_k=c})` 与共形秩
   检验。**唯一用到标签的替代方案，而 A 是标签持有者——这是本赛题的比较优势。**
2. **样本级多数**——batch 内坐标中位数（SecureSplit）。与类别无关，更弱。
3. **时序多数**——跨 epoch/batch 的同一梯度流（GradSent 的流形、CONFIRM 的稳定性）。
   hook 已暴露 `epoch` / `batchIndex`。

另外两点：**"修复"优于"剔除"**（BVBench；也因为推理时不能丢样本、且顶模接口固定）——
应当用合理的良性值替换而不是删除。以及**两种攻击需要不同的检测器，应当同时部署**：
VILLAIN 的加性触发器是**范数膨胀/分布外**的；LFBA 的投毒 embedding 是**分布内**的，只能
靠**跨方语义不一致** + **类条件结构**捕捉。由于**两者都用固定触发器**，对残差场做
**共享方向检验**（robust PCA / 首位奇异向量）是一种针对固定触发器的专用检测器——
在公开的 VFL 防御中未见实现。

---

## 5. 本赛题真实数据上的实测结论（对上述文献的修正）

本仓库在真实赛题数据上直接测量并推翻了若干条文献结论：

1. **ProtoGuard-SL 的核心规则不可用。** 跨类近邻规则
   （"到最近其他类原型距离 < margin × 本类内距"）在干净数据上就有约 2% 误报，
   而且在 20 epoch 下**没有任何 ASR 改善**；把被判定样本替换为**本类原型**后，顶模学到的
   是"本类原型 → 本类标签"的强化关联，**ASR 反而抬高、主任务精度下降**
   （干净 0.9675 → 0.9241，LFBA ASR 0.0260 → 0.1526）。
2. **"共模消除"对 VILLAIN 是严格成立的，而不是近似。** 直接测量触发造成的被动方
   embedding 位移：首位奇异方向能量占比 **1.0000**、与首位方向平均 |cos| **1.0000**。
   因此不需要"检测"，直接做**软阈值共模消除**即可。LFBA 的位移能量占比 0.6391、
   平均 |cos| 0.3029，属部分共模，共模消除能带走大部分成分。
3. **无条件共模消除会毁掉主任务。** 推理期偏离范数 2.76 对单样本表征范数 16.65
   （dataset_02），无条件减去等于删掉 17% 的信号，主任务精度从 0.9712 崩到 0.5110。
   必须做**异常判据 + 软阈值**（见 `docs/defense_report.md`）。
4. **梯度统一缩放完全无效。** 把返还给被动方的梯度整体缩小 20 倍，标签恢复率几乎不动
   （0.6772 → 0.6763）——**Adam 的逐参数自适应归一化把统一缩放吸收掉了**。这否定了
   "降低被动方有效学习率来防标签推断"的常见直觉。
5. **逐样本梯度重指派能把标签恢复率压到随机水平**（dataset_03 实测恰好 0.5000），
   验证了"样本—梯度对应关系"是补全攻击的必要条件。
6. **投毒集构造的实现约束必须尊重。** LFBA 用 L1 梯度相似度构造投毒集，若防御使任何
   逐样本梯度**恰好为零**，攻击会直接抛
   `LFBA cannot calculate L1 similarity from zero gradients`。防御实现必须保证逐样本
   梯度不退化。

---

## 6. 核验状态汇总

**已从全文核验**：ProtoGuard-SL Table I；VFLIP 摘要（84.4%→7.57%）与部分结果行
（8.12%）；BadVFL Table 1 与梯度加噪防御数字；SecureSplit 完整算法与 Table 1 首行；
CONFIRM 摘要指标；Fine-Pruning（ASR→0%、精度 −0.4%）；STRIP（FAR<1%）；
BVBench Table I/II（部分）与"99% 分位可检出"结论。

**读到但数字无法核验**：FILTER、"N Truths and a Lie"、VFLMonitor、SafeSplit（除片段
FPR 外）、SplitGuard、BVBench 防御结果表（含 KUHAR）、互信息正则化防御。

**未经公开验证的组合**（需自行在干净切分上标定）：共享方向 robust-PCA 检验；以及
"ProtoGuard-SL / VFLIP 对 LFBA 有效"这一外推（两篇论文都未评估 LFBA）。

**工具限制**：抓取工具无法解码 `application/pdf`，因此所有 AAAI/USENIX/NDSS/IEEE/ACM
的 PDF 数字要么来自 arXiv/ar5iv 的 HTML 版本，要么标注为未核验。arXiv HTML 渲染在靠后
章节会被截断，这正是 BVBench §VII 与 SecureSplit 完整表格只能部分取得的原因。
