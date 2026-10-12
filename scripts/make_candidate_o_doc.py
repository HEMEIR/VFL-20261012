# -*- coding: utf-8 -*-
"""Generate the algorithm-mechanism document for candidate O.

Candidate O is the best submission on the platform (43.313455). This document records the
mechanism it implements, why each component exists, the exact parameter values, and the
measured evidence behind them.

    <python> scripts/make_candidate_o_doc.py
"""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

OUT = (Path(__file__).resolve().parent.parent / "docs"
       / "候选O算法机制说明.docx")
CJK = "Microsoft YaHei"


def styleDocument(document):
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)


def heading(document, text, level=1):
    h = document.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
        run.font.color.rgb = RGBColor(0x1F, 0x3B, 0x63)
    return h


def addRuns(paragraph, text, baseSize=None, baseBold=False):
    """Insert text, turning **...** spans into real bold runs (markdown is not DOCX)."""
    for index, segment in enumerate(str(text).split("**")):
        if not segment:
            continue
        run = paragraph.add_run(segment)
        run.bold = baseBold or (index % 2 == 1)
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
        if baseSize is not None:
            run.font.size = Pt(baseSize)
    return paragraph


def body(document, text):
    return addRuns(document.add_paragraph(), text)


def bullet(document, text):
    return addRuns(document.add_paragraph(style="List Bullet"), text)


def formula(document, text):
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    run.font.size = Pt(10)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x20, 0x20, 0x80)
    return p


def table(document, header, rows, widths=None, size=9):
    t = document.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, text in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        addRuns(cell.paragraphs[0], text, baseSize=size, baseBold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = ""
            addRuns(cells[i].paragraphs[0], str(text), baseSize=size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Pt(w)
    return t


def main():
    document = Document()
    styleDocument(document)

    title = document.add_heading("候选 O 的算法机制说明", level=0)
    for run in title.runs:
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    sub = document.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("平台实测 43.313455（本项目当前最优）· 性能 17.73 / 隐私 8.58 / 鲁棒 17.11")
    r.font.size = Pt(9)
    r.font.name = "Times New Roman"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

    # ---------------- 1 ----------------
    heading(document, "1  算法概览", 1)
    body(document, "候选 O 是一组**只使用 A 方合法可见信息**的防御机制，在**一次训练**中"
                   "同时抵御复合攻击（LFBA / VILLAIN 后门 + PMC / AMC 标签推断）。")
    body(document, "五个机制按重要性排列，各自针对一个可测量的失分点：")
    table(document,
          ["#", "机制", "作用对象", "针对攻击", "实测贡献"],
          [["1", "**梯度重指派**", "A 回传给 B 的逐样本 cut 梯度", "标签推断 + LFBA",
            "**平台 +3.52**（最大单项）"],
           ["2", "**Party Dropout**", "cutB 整行置零", "PMC 标签推断", "隐私公式分 +0.50"],
           ["3", "**范数超额裁剪**", "cutB 逐样本范数", "后门触发样本", "消融确认净正"],
           ["4", "**抖动门控**", "仅 VILLAIN 攻击批的输出扰动", "VILLAIN 固定触发器", "平台 +0.15"],
           ["5", "**软阈值共模消除**", "cutB 的批级共同分量", "VILLAIN 加性偏移", "基础机制"]],
          widths=[20, 90, 130, 90, 110])

    # ---------------- 2 ----------------
    heading(document, "2  威胁模型与 A 方的能力边界", 1)
    body(document, "两方纵向联邦：**A 是主动方**（持有标签 `y`、底模 `bottomA`、顶模 `topModel`），"
                   "**B 是被动方**（持有私有特征 `X_B` 与底模 `bottomB`）。A 是防御方，B 是攻击方。")
    heading(document, "2.1  A 只能通过三个 hook 施加影响", 2)
    formula(document, "trainingRepresentationHook(cutA, cutB, labels, sampleIds, epoch, batchIndex, context, generator)")
    formula(document, "cutGradientHook(gradientB, passiveCut, labels, sampleIds, epoch, batchIndex, context, generator)")
    formula(document, "inferenceRepresentationHook(embeddingA, embeddingB, sampleIds, context, generator)")
    body(document, "**A 不能**读取 B 的原始特征、**不能**修改 `bottomB` 的参数、**不能**改优化器。"
                   "所以全部机制都是对「A 合法看到的张量」的变换。")
    heading(document, "2.2  Hook 中可用的上下文", 2)
    body(document, "`context` 只含公开信息：`backdoor`（`lfba`/`villain`/`none`）、`labelInference`、"
                   "`phase`、`poisoningActive`、`totalEpochs`、`batchesPerEpoch`。"
                   "**不含 logits、不含顶模、不含数据集名**——这决定了若干机制的实现方式。")
    heading(document, "2.3  训练期的数据流（决定 A 的作用点）", 2)
    formula(document, "441: (cutA, cutB) = applyPartyATrainingRepresentationHook(...)")
    formula(document, "450: logits = topModel(cutA, cutB)")
    formula(document, "452: loss.backward()")
    formula(document, "458: gradientB = cutB.grad")
    formula(document, "460: gradientB = applyPartyACutGradientHook(gradientB, ...)")
    formula(document, "472: embeddingB.backward(gradientB)      # B 用 A 返回的梯度更新")
    formula(document, "478: optimizerB.step()")
    body(document, "**关键**：B 的参数更新使用的正是 A 从 `cutGradientHook` 返回的张量。"
                   "因此 A 虽然碰不到 B 的优化器，却能通过**决定传回去的内容**来塑造 B 学到的表征。"
                   "这是全部机制的作用原理。")

    # ---------------- 3 ----------------
    heading(document, "3  机制一：梯度重指派（核心）", 1)
    heading(document, "3.1  原理", 2)
    body(document, "对逐样本 cut 梯度施加一个**随机置换**：")
    formula(document, "g[i] ← g[π(i)]        π 为批内均匀随机置换")
    body(document, "这个操作的性质是全部机制里最特殊的：")
    bullet(document, "**逐字保留梯度分布的全部矩**——批均值、方差、协方差都不变（置换的构造性质）")
    bullet(document, "**只破坏「样本 ↔ 梯度」的配对**——`g[i]` 不再属于样本 `i`")
    bullet(document, "**方向不变**——每个梯度向量的方向和模长都原样保留")
    body(document, "因此顶模的联合训练几乎不受影响（性能项仅 −0.135），"
                   "但两类攻击同时失效：")
    table(document,
          ["攻击", "被破坏的依赖", "后果"],
          [["**PMC / AMC** 标签推断", "补全头依赖 `∂L/∂H_{i,y}` 与样本 `i` 的对应关系",
            "A 回传的信号与 B 的数据错配，B 学到的表征不再与标签对齐"],
           ["**LFBA** 后门", "投毒集由逐样本梯度的 L1 相似度排名选出："
            "`s_i = <g_i,g_anchor>/(‖g_i‖₁·‖g_anchor‖₁)`", "排名变成随机 → 投毒到错误样本"]],
          widths=[110, 220, 190])
    heading(document, "3.2  门控（必须，否则会中止场景）", 2)
    body(document, "重指派在两类场景分别开启，这是实测要求而非设计偏好：")
    table(document,
          ["参数", "值", "含义"],
          [["gradientShuffleStrength", "1.0", "完全置换"],
           ["gradientShuffleLabelInferenceOnly", "true", "纯标签推断场景（无后门）开启"],
           ["gradientShufflePoisoningActiveOnly", "true", "投毒生效的攻击批开启"],
           ["gradientShufflePoisoningBackdoors", "[\"lfba\"]", "**只对 LFBA**——VILLAIN 开启会导致场景中止"]],
          widths=[190, 80, 250])
    heading(document, "3.3  实测", 2)
    table(document,
          ["场景", "H（无 LFBA 重指派）", "O（开启）", "ΔASR"],
          [["d01_lfba_p20", "0.8867", "**0.2300**", "−0.6567"],
           ["d02_lfba_p10", "0.8729", "**0.2001**", "−0.6728"],
           ["d01_lfba_p05", "0.6167", "**0.1600**", "−0.4567"],
           ["d02_lfba_p20", "0.7324", "**0.2933**", "−0.4391"],
           ["LFBA 族平均", "—", "—", "**−0.2017**"],
           ["VILLAIN 族平均", "—", "—", "**0.0000**（完全不受影响）"]],
          widths=[130, 120, 110, 90])
    body(document, "**VILLAIN 完全不受影响**是重要的反向验证：VILLAIN 是**与样本无关的固定加性触发器**，"
                   "不需要任何逐样本信息，所以只改「样本↔梯度配对」的机制对它无效。"
                   "这恰好印证了机制的作用点。")

    # ---------------- 4 ----------------
    heading(document, "4  机制二：Party Dropout", 1)
    heading(document, "4.1  原理", 2)
    body(document, "在训练期随机将一部分样本的 `cutB` **整行置零**：")
    formula(document, "m_i ~ Bernoulli(1 − r),      cutB_i ← m_i · cutB_i")
    body(document, "顶模必须在 B 缺席时也能工作，因此被迫学会不依赖 B。"
                   "它改变的是「**顶模被允许对哪些样本使用 B**」，而不是梯度内容——"
                   "因此属于已验证能迁移到平台的那一类（结构性改变嵌入的使用方式）。")
    heading(document, "4.2  参数与门控", 2)
    table(document,
          ["参数", "值", "说明"],
          [["partyDropoutRate", "0.2", "丢包率；实测 0.05/0.10/0.20 均使 LRA 到 0.2394"],
           ["partyDropoutLabelInferenceOnly", "true", "仅纯标签推断场景"],
           ["（隐含）不得在投毒批生效", "—", "cutB 整行置零使该样本梯度精确为零，"
            "而 LFBA 用逐样本梯度 L1 相似度建投毒集，会直接中止："
            "`ValueError: LFBA cannot calculate L1 similarity from zero gradients`"]],
          widths=[190, 60, 270])
    heading(document, "4.3  实测", 2)
    table(document,
          ["场景", "无 dropout", "rate=0.2", "判定"],
          [["dataset_02_pmc", "0.9025 / 0.6972", "**0.8959 / 0.5039**", "LRA 降到随机水平"],
           ["dataset_03_pmc", "0.9448 / 0.5904", "**0.9584 / 0.5163**", "**双赢**（精度还 +0.0136）"],
           ["dataset_04_amc", "0.8317 / 0.7606", "0.8327 / 0.7606", "无效果"],
           ["dataset_05_amc", "0.9893 / 0.4117", "0.9845 / 0.4583", "略变差"],
           ["dataset_03_villain", "0.9638 / 0.0673", "逐位不变", "门控使然"]],
          widths=[120, 130, 130, 120])
    body(document, "**对 PMC 有效、对 AMC 无效**。原因与机制一致：AMC 的放大发生在 B 本地 Adam 更新"
                   "**之前**，主动放大沿稳定方向的标签信号；而 party dropout 只降低 A 对 B 的依赖，"
                   "不改变 B 本地更新的方向一致性。")

    # ---------------- 5 ----------------
    heading(document, "5  机制三：范数超额裁剪", 1)
    body(document, "逐样本压缩 `cutB` 的范数，只对**超出阈值**的样本按比例缩放：")
    formula(document, "bound = quantile(‖cutB_i‖₂, q) · normExcessFactor")
    formula(document, "s_i = min(1, bound / ‖cutB_i‖)      cutB_i ← s_i · cutB_i")
    body(document, "**只缩放不改变方向**（方向余弦恒为 1），且未超限的样本逐位不变。"
                   "针对的是**后门触发样本**——它们往往呈现异常大的表征范数。")
    table(document,
          ["参数", "值"],
          [["normExcessStrength", "1.0"],
           ["normExcessFactor", "2.5"],
           ["gradientClipQuantile", "0.0（关闭）"],
           ["gradientClipNorm", "0.0（关闭）"]],
          widths=[190, 120])
    body(document, "**注**：`gradientClipQuantile > 0` 在实测中有害（精度崩塌、VILLAIN 的 ASR 变 1.0000），"
                   "因此 O 中保持关闭。这与裁剪作用在**表征**而非**梯度**上有关——"
                   "见第 8 节的判据。")

    # ---------------- 6 ----------------
    heading(document, "6  机制四：抖动门控（仅 VILLAIN）", 1)
    body(document, "向**投毒样本**的 `cutB` 注入小幅随机抖动，且**只对 VILLAIN 生效**：")
    formula(document, "cutB_i ← cutB_i + ε,      ε ~ N(0, σ²)，仅当 backdoor == \"villain\"")
    table(document,
          ["参数", "值", "说明"],
          [["jitterForAttacks", "{villain}", "**只对 VILLAIN**；对 LFBA 无效"],
           ["trainingJitter", "0.5", "抖动幅度"],
           ["jitterPoisonedOnly", "true", "只作用投毒样本"],
           ["jitterWithLabelInference", "false", "纯标签推断场景不施加"]],
          widths=[190, 90, 250])
    body(document, "为什么只对 VILLAIN：LFBA 的投毒需要**样本级梯度保真**（它靠梯度相似度选样本），"
                   "抖动会干扰这一点；而 VILLAIN 是固定加性触发器，抖动直接破坏其触发器的一致性。"
                   "实测平台贡献 **+0.15**。")

    # ---------------- 7 ----------------
    heading(document, "7  机制五：软阈值共模消除", 1)
    body(document, "移除 `cutB` 中**整批共享**的分量，用滑动参考估计批中心：")
    formula(document, "c_t = ρ · c_{t-1} + (1 − ρ) · mean_i(cutB_i)")
    formula(document, "cutB_i ← cutB_i − α · softGate(‖c_t‖) · c_t")
    body(document, "软阈值门控的意思是：只有当共同分量的范数显著超过参考统计量时（"
                   "`recenterDetectionThreshold` 倍）才消除，避免把类别结构当成异常删掉。")
    table(document,
          ["参数", "值", "说明"],
          [["recenterMode", "soft", "软阈值，而非无条件消除"],
           ["recenterDetectionThreshold", "3.0", "触发倍数"],
           ["trainingRecenterStrength", "0.5", "训练期强度"],
           ["inferenceRecenterStrength", "1.0", "推理期强度"],
           ["referenceMomentum", "0.99", "参考统计量的滑动动量"]],
          widths=[190, 90, 250])
    body(document, "针对的是 **VILLAIN 的加性固定偏移**——它对每个投毒样本加同一个向量，"
                   "而批均值消除恰好能抵消这种共享偏移。")

    # ---------------- 8 ----------------
    heading(document, "8  完整参数表（候选 O）", 1)
    table(document,
          ["机制", "参数", "值"],
          [["共模消除", "recenterMode / recenterDetectionThreshold", "soft / 3.0"],
           ["", "trainingRecenterStrength / inferenceRecenterStrength", "0.5 / 1.0"],
           ["", "referenceMomentum", "0.99"],
           ["抖动", "jitterForAttacks", "{villain}"],
           ["", "trainingJitter / jitterPoisonedOnly / jitterWithLabelInference", "0.5 / true / false"],
           ["范数裁剪", "normExcessStrength / normExcessFactor", "1.0 / 2.5"],
           ["**重指派**", "gradientShuffleStrength", "**1.0**"],
           ["", "gradientShuffleLabelInferenceOnly", "**true**"],
           ["", "gradientShufflePoisoningActiveOnly", "**true**"],
           ["", "gradientShufflePoisoningBackdoors", "**[\"lfba\"]**"],
           ["", "gradientShufflePoisonedOnly / gradientShuffleBackdoorOnly", "false / false"],
           ["**Dropout**", "partyDropoutRate", "**0.2**"],
           ["", "partyDropoutLabelInferenceOnly", "**true**"],
           ["（关闭）", "gradientClipQuantile / gradientClipNorm", "0.0 / 0.0"]],
          widths=[80, 250, 130])

    # ---------------- 9 ----------------
    heading(document, "9  三项得分与提升轨迹", 1)
    table(document,
          ["项", "得分", "满分", "占比"],
          [["性能", "**17.728698**", "20", "88.6%"],
           ["隐私", "**8.577887**", "20", "42.9%"],
           ["鲁棒", "**17.113576**", "20", "85.6%"],
           ["合计", "**43.420161**", "60", "—"]],
          widths=[100, 120, 70, 70])
    body(document, "对应平台实测 **43.313455**。")
    body(document, "")
    body(document, "提升轨迹（平台实测）：")
    table(document,
          ["提交", "平台分", "引入的机制"],
          [["precision_v3", "37.9599", "软阈值共模消除 + 抖动"],
           ["jitter_villain", "38.1100", "抖动门控只对 VILLAIN"],
           ["h", "41.6341", "**梯度重指派**（纯标签推断场景）"],
           ["**o**", "**43.3135**", "**重指派扩展到 LFBA 攻击批 + party dropout**"]],
          widths=[100, 80, 280])

    # ---------------- 10 ----------------
    heading(document, "10  为什么只有这些机制有效：一条统一判据", 1)
    body(document, "把本项目全部实测放在一起，规律是清楚的。")
    heading(document, "10.1  判据：重排信息 vs 损毁信息", 2)
    table(document,
          ["机制", "对梯度做了什么", "结果"],
          [["**梯度重指派**", "**重排**——逐字保留分布的全部矩", "训练不受影响，两个攻击同时失效，**平台 +3.52**"],
           ["梯度裁剪", "损毁——压缩范数", "精度崩塌、VILLAIN 的 ASR → 1.0"],
           ["梯度量化", "损毁——坐标吸附", "被 Adam 的逐参数归一化**完全吸收**"],
           ["梯度稀疏化", "损毁——丢弃分量", "精度跌到随机水平、场景中止"],
           ["统一加噪", "损毁——加无偏噪声", "无效（统一缩放 20 倍仅使 LRA 动 0.0009）"]],
          widths=[110, 180, 230])
    body(document, "**重排可以同时压制多个攻击而不伤训练；损毁必然在压制攻击的同时伤及训练**，"
                   "因为 VILLAIN 的检查点选择对梯度通道的任何损毁都极其敏感。")
    heading(document, "10.2  为什么「偏离批中心」类判据在 VFL 上失效", 2)
    body(document, "VFL 中**类别结构本身就是异常**——同一类的样本天然聚集，"
                   "所以任何以「偏离批中心/多数」为判据的机制都会把类别信号当成攻击特征删掉。"
                   "本项目为此付出过明确代价：共享方向消除（robust PCA）在 dataset_02_lfba 上"
                   "把精度打到 **0.5110（随机水平）**，因为删掉的首位方向正是类均值结构。")
    body(document, "这正是软阈值共模消除要带 `recenterDetectionThreshold=3.0` 门控的原因。")
    heading(document, "10.3  本地公式的使用边界（重要限制）", 2)
    body(document, "平台已公布的分项显示：")
    table(document,
          ["项", "平台/本地 比值", "可预测性"],
          [["性能", "**0.9941**", "可预测"],
           ["鲁棒", "**0.9896**", "可预测"],
           ["隐私", "**0.8098**", "**不可预测**——本地公式的全部误差来自此项"]],
          widths=[80, 140, 240])
    body(document, "因此**只有不依赖「本地标签恢复率改善」的机制才可用于选型**。"
                   "直接调优本地 LRA 的机制在平台上系统性负偏约 **−2.1**（候选 p 与 cc_ramp100 的误差"
                   "分别为 −2.121 与 −2.106）。")
    body(document, "候选 O 的核心机制（重指派）属于**改变 B 所学表征**的一类，不是直接调优 LRA 数值，"
                   "所以它的增益在平台上**同向且放大**（公式 +1.011，平台 +1.679）。")

    # ---------------- 11 ----------------
    heading(document, "11  已知限制", 1)
    bullet(document, "**隐私项是瓶颈**（8.58/20，42.9%）。15 个标签推断场景中有 **5 个的 P 已为 0**——"
                     "这些场景的攻击本身就弱（基线恢复率不高于当前值），属于**信息论地板**，防御无从得分。")
    bullet(document, "**party dropout 会破坏 dataset_04_pmc 的满分场景**（Q_p 1.0 → 0.0），"
                     "因为强度 0.2 越过了该场景的阈值。这是 O 相对候选 M 的唯一退步。")
    bullet(document, "**重指派已达代码上限**（`gradientShuffleStrength = 1.0`，完全置换），"
                     "在该机制内无进一步提升余地。")
    bullet(document, "**梯度裁剪在本项目中有害**且在 O 中关闭；`gradientClipQuantile > 0` 会导致"
                     "VILLAIN 场景的 ASR 变为 1.0000。")

    document.save(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
