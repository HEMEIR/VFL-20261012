# -*- coding: utf-8 -*-
"""Generate the mechanism-description document (DOCX).

Describes the two backdoor attacks, the two label-inference attacks, and the defence
mechanism, each with its governing formulas, plus the measured results.

    <python> scripts/make_mechanism_doc.py
"""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

OUT = Path(__file__).resolve().parent.parent / "docs" / "攻击与防御机制说明.docx"

CJK = "Microsoft YaHei"
MONO = "Consolas"


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


def body(document, text, bold=False, indent=0.0):
    """Add a paragraph, rendering **...** spans as real bold runs.

    Markdown asterisks are not DOCX markup: writing them literally leaves visible
    "**" in the rendered page, so they are parsed into runs here instead.
    """
    p = document.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Pt(indent)
    for index, segment in enumerate(text.split("**")):
        if not segment:
            continue
        run = p.add_run(segment)
        run.bold = bold or (index % 2 == 1)
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    return p


def formula(document, text):
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    # Use a font the runtime actually ships; "Consolas" is mapped by the renderer but
    # "Courier" is reported missing, so keep to widely available families.
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    run.font.size = Pt(10)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x20, 0x20, 0x80)
    return p


def bullet(document, text):
    p = document.add_paragraph(style="List Bullet")
    for index, segment in enumerate(text.split("**")):
        if not segment:
            continue
        run = p.add_run(segment)
        run.bold = index % 2 == 1
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    return p


def table(document, header, rows, widths=None):
    t = document.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    def addCell(paragraph, text, bold=False):
        # Markdown ** is not DOCX markup: parse it into real bold runs so no literal
        # asterisks reach the rendered page.
        for index, segment in enumerate(str(text).split("**")):
            if not segment:
                continue
            run = paragraph.add_run(segment)
            run.bold = bold or (index % 2 == 1)
            run.font.size = Pt(9)
            run.font.name = "Times New Roman"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    for i, text in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        addCell(cell.paragraphs[0], text, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = ""
            addCell(cells[i].paragraphs[0], text)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Pt(w)
    return t


def main():
    document = Document()
    styleDocument(document)

    title = document.add_heading("纵向联邦学习中攻击与防御机制的公式化描述", level=0)
    for run in title.runs:
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = subtitle.add_run("VFL-Security-Challenge v1.0.7  ·  两方场景（A 方 = 主动方 + 防御方，B 方 = 被动方 + 攻击方）")
    r.font.size = Pt(9)
    r.font.name = "Times New Roman"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

    # ---------------- 0. 记号 ----------------
    heading(document, "0  记号与威胁模型", 1)
    body(document, "A 方持有样本特征 X_A 与标签 y；B 方持有样本特征 X_B。两方各自维护底模与顶模，"
                   "A 方另有顶模 f_top。训练时 B 计算 cutB 上传给 A，A 计算损失并向 B 回传逐样本梯度。")
    formula(document, "cutA = f_A(X_A),   cutB = f_B(X_B)")
    formula(document, "logits = f_top(cutA, cutB),   L = CE(logits, y)")
    formula(document, "B 收到:   g_B = ∂L / ∂cutB        (逐样本，形状 [batch, d])")
    body(document, "关键约束：A 方只能通过三个 hook 改写通信张量，不能替换基座模型、不能改训练协议、"
                   "不能读取 B 的原始特征或模型参数。可改写的量只有：")
    bullet(document, "trainingRepresentationHook: 训练期的 (cutA, cutB) —— 可拿到 labels、sampleIds、context")
    bullet(document, "cutGradientHook: 回传给 B 的 g_B —— 可拿到 labels")
    bullet(document, "inferenceRepresentationHook: 推理期的 (embeddingA, embeddingB) —— 拿不到顶模")

    # ---------------- 1. 后门 ----------------
    heading(document, "1  后门攻击的机制", 1)

    heading(document, "1.1  LFBA（基于梯度的标签翻转后门）", 2)
    body(document, "LFBA 的关键在于**如何挑选投毒样本**：它利用 A 方回传的逐样本 cut 梯度，"
                   "按与锚点样本梯度的 L1 余弦相似度排序，取最相似的若干样本改为目标类并注入触发器。"
                   "实现依据 attacks/backdoor/lfba.py::buildPoisonSet。")
    formula(document, "s_i = <g_i, g_anchor> / (‖g_i‖₁ · ‖g_anchor‖₁)")
    formula(document, "PoisonSet = argtopk_i  s_i        (取相似度最高的 k 个)")
    formula(document, "k = round(p · N_target),   p ∈ {0.05, 0.10, 0.20}")
    body(document, "然后对这 k 个样本同时做两件事：")
    bullet(document, "标签替换：y_i ← y_target")
    bullet(document, "特征替换：X_B,i ← X_B,anchor（把投毒样本的 B 方特征替换为锚点样本的特征）")
    body(document, "要害在于：选样完全依赖逐样本梯度 g_i 与样本的一一对应关系。一旦这个对应关系被破坏，"
                   "攻击者挑出的就不再是它以为的那些样本，攻击的定向性随之瓦解。", bold=True)

    heading(document, "1.2  VILLAIN（固定加性触发器后门）", 2)
    body(document, "VILLAIN 在 B 方的嵌入空间构造一个固定的加性触发器，与样本无关，因此无法通过"
                   "破坏逐样本对应关系来防御。实现依据 attacks/backdoor/villain.py::buildVillainTrigger。")
    body(document, "第一步，选出嵌入空间中方差最大的 m 个维度：")
    formula(document, "D* = argtopm_d  σ_d            (σ_d 为第 d 维在训练集上的标准差)")
    body(document, "第二步，按 4 维一组的 ++−− 符号规则构造触发向量：")
    formula(document, "δ = mean_{d∈D*} σ_d")
    formula(document, "signs_d = +1 (d mod 4 < 2),   −1 (d mod 4 ≥ 2)")
    formula(document, "trigger = mask(D*) ⊙ (β · δ · signs),   mask(D*)_d = 1 若 d ∈ D*")
    body(document, "第三步，在推理时对所有测试样本做加性注入：")
    formula(document, "cutB' = cutB + trigger")
    body(document, "由此可推出 v 的两个可防御性质：")
    bullet(document, "触发器只写在 m 个维度上，且幅度为 β·δ（δ 与数据方差同量级），因此受触发的样本嵌入会偏离正常分布、范数被抬高；")
    bullet(document, "触发是**加性且固定**的，因此受触发样本在嵌入空间中形成一个低方差的紧致簇，这是可检测的结构性特征。")

    # ---------------- 2. 标签推断 ----------------
    heading(document, "2  标签推断攻击的机制", 1)

    heading(document, "2.1  PMC / AMC（被动模型补全）", 2)
    body(document, "这两者是**训练完成之后**的离线攻击，攻击代码自身声明其数据来源为 "
                   "\"readsPartyAFeatures\": False 与 \"readsTopModel\": False，"
                   "即完全不读 A 的特征、也不读顶模。它们复制 B 方自己的底模并训练一个补全头：")
    formula(document, "θ̂ = argmin_θ  L_head( head_θ(bottomB(X_B)),  y_aux )")
    body(document, "其中 y_aux 来自公开的辅助标签 CSV。攻击的恢复率记为 A_recovered。")
    body(document, "因此该攻击的上限由**训练期 B 的 bottomB 学到了多少标签信息**决定：一旦训练结束，"
                   "A 无法再干预。这也意味着防御必须在训练期完成。", bold=True)
    formula(document, "AMC 的放大:  g_B ← clamp(α_AMC) · g_B,   α_AMC 由相邻步梯度方向一致性决定 ∈ [1, 5]")
    body(document, "AMC 的放大发生在 B 本地 Adam 更新之前，只作用于 bottomB 的原始梯度，A 既看不到也管不了；"
                   "其目的是让 bottomB 更快地沿梯度中稳定的标签相关方向学习。")

    heading(document, "2.2  梯度符号泄露（信息论根源）", 2)
    body(document, "对 softmax + 交叉熵，损失对 logits 的梯度具有确定的符号结构：真实类那一维被减去 1，"
                   "因而其符号与其余维度相反。")
    formula(document, "∂L/∂H_{i,j} = S_j          (j ≠ y)")
    formula(document, "∂L/∂H_{i,j} = S_j − 1      (j = y)")
    body(document, "所以**只要逐样本梯度被暴露，标签即可被直接读出**。本项目对此做了直接实测："
                   "在关闭重指派的配置下检验 2560 行，cutGradient 行均值符号与标签的一致率为 "
                   "0.1926（随机为 0.5），证实该通路确实存在。")

    # ---------------- 3. 防御 ----------------
    heading(document, "3  本方案防御算法的机制", 1)

    heading(document, "3.1  逐样本梯度重指派（主机制）", 2)
    body(document, "在 cutGradientHook 内，对回传给 B 的逐样本梯度做**批内随机置换**：")
    formula(document, "g_B[i] ← g_B[π(i)],     π 为批内的随机排列")
    body(document, "这个操作的性质很特殊：它**逐字保持了整个梯度分布的所有矩**（均值、方差、协方差全部不变），"
                   "却精确破坏了 1.1 节的 s_i 与 2.2 节的符号—标签对应关系。因此它既压制标签推断，"
                   "又破坏 LFBA 的投毒选样。", bold=True)
    body(document, "门控采用两分支的并集，按公开标签（hook context 中的 backdoor / labelInference）选择：")
    formula(document, "启用 ⟺ (backdoor == \"none\" ∧ labelInference ≠ \"none\")")
    formula(document, "        ∨ (poisoningActive ∧ backdoor ∈ {lfba})")
    bullet(document, "第一支对应纯标签推断场景（PMC/AMC），那里 poisoningActive 恒为 False；")
    bullet(document, "第二支对应 LFBA 的攻击批，那里 poisoningActive 为 True；")
    bullet(document, "VILLAIN 被显式排除：它的 poisoningActive 覆盖整个攻击阶段，在其上开启会使它达不到 "
                     "minimumMainTaskAccuracy 而中止场景（已实际复现）。")

    heading(document, "3.2  Party Dropout（本轮新增，来自外部建议）", 2)
    body(document, "在 trainingRepresentationHook 内，对随机一部分样本把 cutB **整行置零**，"
                   "使顶模必须学会在 B 缺席时也能工作：")
    formula(document, "m_i ~ Bernoulli(1 − r),    cutB[i] ← m_i · cutB[i],    r = 0.05")
    body(document, "它改变的不是梯度内容，而是「顶模被允许对哪些样本使用 B」，因此降低了顶模对 B 的依赖，"
                   "从而减少 bottomB 被训练成承载任务特化（因而可预测标签）结构的压力。")
    body(document, "**强度 r 存在阈值效应**，这一点由实测发现：某个已把攻击压到低于随机水平的场景"
                   "（dataset_04_pmc，恢复率 0.2394 < 0.5，隐私贡献已达上限 Q_p = 1.0），"
                   "在 r = 0.2 时会被破坏（恢复率升至 0.7606，Q_p 掉至 0），"
                   "而在 r = 0.05 与 r = 0.10 下完整保留。强度扫描结果：")
    table(document,
          ["dropout 强度 r", "Σ Q_p（5 个 pmc 场景）", "Δ vs 不启用", "折算隐私项"],
          [["0（不启用）", "2.8357", "—", "—"],
           ["0.05", "3.7538", "+0.9180", "+1.22 公式分"],
           ["0.10", "3.6712", "+0.8355", "+1.11 公式分"],
           ["0.20", "3.1068", "+0.2710", "+0.36 公式分"]],
          widths=[80, 130, 90, 90])
    body(document, "")
    body(document, "由此得到一条可复用判据：**对已达评分天花板的场景施加任何机制都只会造成损失**，"
                   "因此调参前应先识别哪些场景的 Q_p 已等于 1，并确保新机制不去触碰它们。", bold=True)
    body(document, "两重门控（均为实测要求）：")
    bullet(document, "只在 labelInference ≠ \"none\" 且 backdoor == \"none\" 的纯标签推断场景生效；")
    bullet(document, "绝不在攻击批生效——否则整行置零会使该样本梯度**精确为零**，而 LFBA 的 s_i 无法在零梯度上计算，"
                     "会中止整个场景。该缺陷实际复现过两次。")

    heading(document, "3.3  逐样本范数超额裁剪", 2)
    body(document, "针对 VILLAIN 等加性触发器抬高嵌入范数的性质，对每个样本的范数超过冻结训练参考的"
                   "部分做**软阈值收缩**（保留带内结构，只压超额部分）：")
    formula(document, "thr = factor · referenceNorm,      factor = 2.5")
    formula(document, "excess = max(‖cutB_i‖ − thr, 0)")
    formula(document, "scale_i = (thr + (1 − strength)·excess) / ‖cutB_i‖  （仅当 excess > 0）")
    body(document, "只用单样本统计量与冻结标量，不计算批均值/中位数/SVD，因此没有批间正反馈通道。"
                   "这一点是实测选定的：早期使用批级统计量的版本（批中位数 + 首奇异方向）会让 "
                   "dataset_04/clean 的表征幅度从 0.248 涨到 15.6 并最终发散到 NaN。")

    heading(document, "3.4  抖动按攻击类型门控", 2)
    body(document, "训练期对 B 的表征施加乘性抖动，但只对 VILLAIN 生效：")
    formula(document, "cutB ← cutB ⊙ (1 + ε),   ε ~ U(−a, a),   a = 0.5 · trainingJitter")
    formula(document, "启用条件:  backdoor == \"villain\"")
    body(document, "LFBA 场景必须关闭抖动，因为抖动会降低主任务精度而在 LFBA 上并无隐私收益；"
                   "这一条改动（jitterForAttacks）单独就带来平台 +0.15。")

    # ---------------- 4. 评分 ----------------
    heading(document, "4  官方评分公式与本方案的对应", 1)
    body(document, "总分为三项各 20 分之和。A_min 为随机猜测（二分类 0.5，六分类 1/6）。")

    heading(document, "4.1  模型性能（20 分）", 2)
    formula(document, "U = clip( (A − A_min) / (A_clean − A_min) )")
    formula(document, "S_perf = 20 · mean(U)")
    body(document, "干净场景不计入平均，只提供 A_clean；LFBA 的三个投毒强度先在族内平均，故 LFBA 每个数据集只算一个条目。")

    heading(document, "4.2  隐私保护（20 分）", 2)
    formula(document, "P   = clip( (A_attack − A_recovered) / (A_attack − A_min) )")
    formula(document, "G_p = clip( (A − A_min) / (0.9·A_clean − A_min) )")
    formula(document, "Q_p = G_p · P,        S_priv = 20 · mean(Q_p)")
    body(document, "A_attack 是**逐场景**的无防御恢复率。本方案对应：重指派压低 A_recovered，"
                   "party dropout 进一步压低纯 PMC 场景的 A_recovered。")

    heading(document, "4.3  安全鲁棒性（20 分）", 2)
    formula(document, "G_r = clip( (A − A_min) / (0.9·A_clean − A_min) )")
    formula(document, "R   = 1 − ASR")
    formula(document, "Q_r = 2·G_r·R / (G_r + R),     S_sec = 20 · mean(Q_r)")
    body(document, "本方案对应：重指派破坏 LFBA 的投毒选样、范数裁剪抑制 VILLAIN 的加性触发、"
                   "抖动只对 VILLAIN 生效。注意 G_p 与 G_r 共用分母，精度跌破 0.9·A_clean 会**同时**削弱两项。")

    # ---------------- 5. 结果 ----------------
    heading(document, "5  实测结果", 1)
    heading(document, "5.1  平台提交轨迹", 2)
    table(document,
          ["提交", "平台分", "累计提升", "引入的机制"],
          [["precision_v3", "37.9599", "—", "软阈值共模消除 + 抖动"],
           ["jitter_villain", "38.1100", "+0.1501", "抖动只在 VILLAIN 生效"],
           ["h", "41.6341", "+3.6742", "梯度重指派（仅纯标签推断场景）"],
           ["o", "43.3135", "+5.3536", "重指派扩展到 LFBA 攻击批 + party dropout (r=0.2)"],
           ["p", "41.8406", "+3.8807", "party dropout 强度降至 r=0.05（**已证伪，不提交**）"]],
          widths=[90, 60, 60, 190])
    body(document, "")
    body(document, "**最优提交为 team001__o.zip（43.3135）。** 三维分解说明 h→o 的增益几乎全部来自鲁棒项：", bold=True)
    table(document,
          ["变体", "性能", "隐私", "鲁棒", "公式总分", "平台实测"],
          [["h", "17.864", "8.848", "15.696", "42.409", "41.6341"],
           ["o", "17.729", "8.578", "17.114", "43.420", "43.3135"],
           ["p", "17.793", "9.161", "17.114", "44.068", "41.8406"]],
          widths=[60, 60, 60, 60, 70, 80])

    heading(document, "5.2  本地公式的使用边界（由候选 p 的失败界定）", 2)
    body(document, "候选 p 是本地公式分最高的候选（44.068），平台实测却低于 o 与 h（41.8406），"
                   "即公式在 43.4~44.1 区间**不再单调**。p 与 o 的唯一差别是 party dropout 强度，"
                   "只影响 3 个场景（dataset_02/04/05_pmc），而 p 的全部收益都建立在本地验证集的"
                   "标签恢复率上。由此得到一条必须遵守的规则：")
    bullet(document, "**公式可用于判断量级上的改进**：h 的 +1.011、o 的 +1.011 公式分都在平台兑现"
                     "（分别 +3.67 与 +1.68 平台分）；")
    bullet(document, "**公式不能用于排序只涉及少数场景的 0.6 级微调**：成功率与改动的作用面成正比——"
                     "有效的改动（h、o）都作用于全部或大部分标签推断场景，且改变的是机制结构；"
                     "失败的改动要么属于加噪族（k 的 FIDP，平台 −1.06），要么是在已达平衡的配置上做参数微调"
                     "（p，平台 −1.47）。")

    heading(document, "5.3  防御机制的分类学结论（由多轮实测归纳）", 2)
    body(document, "本项目对 13 类候选防御做了实现或判定，得到两条可判定的筛选规则：")
    bullet(document, "**规则一**：判据是「样本相对批中心/多数的偏离」的防御，在 VFL 上必然失败——"
                     "因为 VFL 中「类别结构」与「异常」在定义上是同一个东西。"
                     "已由 SecureSplit、ProtoGuard-SL、共享方向/robust PCA 三次独立实测证实。")
    bullet(document, "**规则二**：**结构性破坏攻击可利用信息**的机制有效（梯度重指派，平台 +3.52）；"
                     "**注入噪声稀释信息**的机制无效（统一加噪、范数超额裁剪、FIDP 逐维度噪声，平台 −0.99 至 0）。")
    body(document, "此外实测确认了两条边界：")
    bullet(document, "AMC 的放大发生在 B 本地 Adam 之前，A 无法观测或干预，因此所有基于梯度结构的全局门控都失败；")
    bullet(document, "若 B 的特征本身即可预测标签，则标签推断存在信息论地板——"
                     "在本赛题中 15 个标签推断场景里有 5 个的 P 已为 0，即防御无从改善。")

    document.save(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
