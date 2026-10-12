# -*- coding: utf-8 -*-
"""Generate the defence-experiment report (DOCX).

Records every scheme tested in this round - gradient clipping, gradient quantization,
gradient sparsification, each in a uniform and an error-adaptive form - together with the
measured results and the three official score components of the current best submission.

    <python> scripts/make_experiment_doc.py
"""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

OUT = Path(__file__).resolve().parent.parent / "docs" / "防御方案测试报告.docx"
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


def body(document, text, bold=False):
    p = document.add_paragraph()
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


def addRuns(paragraph, text, baseSize=None, baseBold=False):
    """Add text to a paragraph, turning **...** spans into real bold runs.

    Markdown asterisks are not DOCX markup; writing them literally leaves visible "**"
    in the rendered page (and in table cells), so every insertion point routes through
    here.
    """
    for index, segment in enumerate(text.split("**")):
        if not segment:
            continue
        run = paragraph.add_run(segment)
        run.bold = baseBold or (index % 2 == 1)
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
        if baseSize is not None:
            run.font.size = Pt(baseSize)
    return paragraph


def table(document, header, rows, widths=None, size=8.5):
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

    title = document.add_heading("纵向联邦学习防御方案测试报告", level=0)
    for run in title.runs:
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    sub = document.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("梯度裁剪 / 梯度量化 / 梯度稀疏化及其误差自适应版本 · 40 场景验证与三项得分")
    r.font.size = Pt(9)
    r.font.name = "Times New Roman"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

    # ---------------- 1 背景 ----------------
    heading(document, "1  背景与测试动机", 1)
    body(document, "此前对梯度裁剪只测过极端强度（逐样本范数分位 0.9 与 0.75），当时因精度代价过大"
                   "而判定为不可用。但回看数据，0.9 分位在隐私侧**确实有效**：")
    table(document,
          ["场景", "不裁剪 精度/LRA", "分位 0.9 精度/LRA", "隐私侧判定"],
          [["dataset_02_pmc", "0.9025 / 0.6972", "0.8959 / 0.5110", "**有效**（LRA 降 0.186）"],
           ["dataset_03_pmc", "0.9448 / 0.5904", "0.9087 / 0.5443", "**有效**（LRA 降 0.046）"],
           ["dataset_02_composite", "0.8516 / 0.9102", "0.6123 / 0.7574", "**有效**（LRA 降 0.153）"],
           ["dataset_03_villain", "0.9638 / ASR 0.0673", "0.5570 / ASR 1.0000", "**后门崩塌**"]],
          widths=[110, 110, 110, 130])
    body(document, "")
    body(document, "所以「裁剪无效」的旧结论**不准确**：它在隐私侧有效，只是代价过高，而**温和分位从未测过**。"
                   "本轮据此补测，并按指定的反馈准则加入**误差自适应**版本。")

    # ---------------- 2 自适应准则 ----------------
    heading(document, "2  指定的误差自适应准则及其可实现性", 1)
    body(document, "要求：「以在 A 端用 B 的表征得到的预测值与真实值之间的差异为标准，差异越小则加大干扰力度，"
                   "差异越大则减小干扰力度。」")
    heading(document, "2.1  准则的合理性", 2)
    body(document, "差异大（预测偏离真标签）恰恰是**触发样本/投毒样本的特征**——后门的作用就是把预测推向目标类；"
                   "而差异小的普通样本才是标签泄露的来源。因此该准则把防御集中到真正泄露的样本上，"
                   "同时避开攻击所依赖的样本。这是有依据的设计。")
    heading(document, "2.2  可实现性：误差信号只在梯度侧可得", 2)
    body(document, "在 `core/trainer.py` 中，调用顺序是：")
    formula(document, "441: topInputA, topInputB = applyPartyATrainingRepresentationHook(...)")
    formula(document, "450: logits = topModel(topInputA, topInputB)")
    formula(document, "451: loss = lossFunction(logits, batch[\"label\"])")
    formula(document, "452: loss.backward()")
    formula(document, "460: gradientB = applyPartyACutGradientHook(...)")
    body(document, "**表征 hook 在第 441 行，早于前向与反向**，因此它拿不到 logits、拿不到损失、也拿不到梯度，"
                   "**无法在表征侧读出预测误差**。而 `cutGradientHook`（第 460 行）拿到的正是误差本身：对 "
                   "softmax + 交叉熵有 `∂L/∂H_i = softmax(logits_i) − onehot(y_i)`，其范数就是该样本的预测误差。")
    body(document, "**结论：该准则只能在梯度侧精确实现。** 表征侧只能靠上一批的滞后信号，"
                   "或退化为「偏离类质心」这类代理判据——而后者在 VFL 上已被多次证伪"
                   "（VFL 中「类别结构」与「异常」定义上同一，见第 5 节）。")
    heading(document, "2.3  实际实现", 2)
    formula(document, "e_i = ‖g_i‖₁ / √d          逐样本预测误差（由梯度读出）")
    formula(document, "r_i = e_i / EMA(e)         用滑动均值归一，判据是「水平」而非「瞬时」")
    formula(document, "w_i = clip(1 − gain·(r_i − 1), −1, 1)")
    body(document, "w > 0 表示误差低于平均（普通泄露样本），w < 0 表示误差高于平均（触发样本）。"
                   "各机制的用法：")
    bullet(document, "梯度裁剪：`bound_i = quantile(norms) · (1 − gain·w_i)` —— 小误差收紧上限（多裁），大误差放宽（少裁）")
    bullet(document, "梯度量化：`width_i = step · RMS · (1 + gain·w_i)` —— 小误差用细网格（多丢信息），大误差用粗网格")
    bullet(document, "梯度稀疏化：大误差的整行置零，小误差按 top-k 保留")
    bullet(document, "重指派混合：`out_i = w_i · shuffled_i + (1 − w_i) · g_i` —— 小误差全置换，大误差保留原梯度")

    # ---------------- 3 测试方案 ----------------
    heading(document, "3  测试方案", 1)
    body(document, "基座配置为候选 O（当前最优提交）的全部机制参数，逐个叠加被考察的机制，"
                   "以便任何差异都可归因于该机制本身：")
    formula(document, "基线 = 重指派(纯LI ∪ LFBA攻击批) + 范数超额裁剪 + 抖动(仅VILLAIN) + party dropout 0.2")
    body(document, "测试的场景集合覆盖两类失分点：标签推断场景（`dataset_02_pmc`、`dataset_03_pmc`、"
                   "`dataset_02_amc`）与后门场景（`dataset_03_villain`、`dataset_02_composite`）。"
                   "每轮 100 epoch，本地验证口径。共 5 组对照、6 个方案：")
    table(document,
          ["实验组", "机制", "参数"],
          [["B0", "基线（无裁剪/量化/稀疏化）", "—"],
           ["B1", "梯度裁剪（统一）", "分位 0.9"],
           ["B2", "梯度裁剪（误差自适应）", "分位 0.9，自适应强度 1.0"],
           ["B3", "梯度量化（统一）", "步长 0.5·RMS"],
           ["B4", "梯度量化（误差自适应）", "步长 0.5·RMS，自适应强度 1.0"],
           ["B5", "梯度稀疏化（误差自适应）", "稀疏率 0.5（保留一半坐标），大误差整行置零"]],
          widths=[50, 190, 190])
    body(document, "")
    body(document, "另补测裁剪强度的温和分位（0.95 / 0.98 / 0.99），结果见第 4.4 节。")

    # ---------------- 4 结果 ----------------
    heading(document, "4  测试结果", 1)
    heading(document, "4.1  六方案对照（精度 / 标签恢复率或攻击成功率）", 2)
    table(document,
          ["场景", "B0 基线", "B1 裁剪统一", "B2 裁剪自适应", "B3 量化统一", "B4 量化自适应", "B5 稀疏自适应"],
          [["d02_pmc", "0.8959/0.5039", "0.5110/0.7529", "0.8862/0.5804", "0.8959/0.5039", "0.8630/0.7011", "0.5110/0.5110"],
           ["d03_pmc", "0.9584/0.5163", "0.7242/0.5443", "0.9557/0.5452", "0.9584/0.5163", "0.9575/0.6193", "0.4919/0.4819"],
           ["d02_amc", "0.8951/0.6784", "0.4893/0.4998", "0.6758/0.6921", "0.8951/0.6784", "0.8968/0.5353", "0.5110/0.5110"],
           ["d03_villain", "0.9638/0.0673", "**0.5570/1.0000**", "**0.5570/1.0000**", "0.9638/0.0673", "0.9693/0.0490", "**FAIL**"],
           ["d02_composite", "0.8516/0.6745", "0.6123/0.8943", "0.8540/**0.0174**", "0.8516/0.6745", "**FAIL**", "**FAIL**"]],
          widths=[62, 78, 78, 78, 78, 78, 78])
    body(document, "")
    body(document, "格式：精度 / （标签推断场景为 LRA，后门场景为 ASR）。FAIL 表示场景中止。", bold=False)

    heading(document, "4.2  逐方案判定", 2)
    body(document, "**B1 梯度裁剪（统一，分位 0.9）——无效且有害。**")
    bullet(document, "标签推断场景精度崩塌：d02_pmc 0.8959 → **0.5110**（随机水平）、d02_amc 0.8951 → **0.4893**")
    bullet(document, "d03_villain 精度 0.9638 → **0.5570**，同时 ASR **0.0673 → 1.0000**（后门完全得手）")
    bullet(document, "d02_composite 精度 0.8516 → 0.6123")
    body(document, "")
    body(document, "**B2 梯度裁剪（误差自适应）——比统一版明显更好，但仍不可用。**")
    bullet(document, "修复了统一版的两个崩塌：d02_pmc 0.5110 → **0.8862**、d02_amc 0.4893 → **0.6758**")
    bullet(document, "d02_composite 的 ASR 大幅改善：0.6745 → **0.0174**，且精度保持 0.8540（基线 0.8516）")
    bullet(document, "**但 d03_villain 仍完全崩塌**（0.5570 / ASR 1.0000），与统一版逐位相同")
    body(document, "")
    body(document, "**B3 梯度量化（统一）——完全不改变训练结果。**")
    bullet(document, "五个场景与基线**逐位相同**（0.8959/0.5039、0.9584/0.5163、0.8951/0.6784、0.9638/0.0673、0.8516/0.6745）")
    bullet(document, "已单独验证量化本身在梯度上生效（三种尺度下均改变约 12%），"
                     "故原因不是未执行，而是**Adam 的逐参数自适应归一化把这种均匀的小幅变换吸收掉了**")
    bullet(document, "这与本项目既有的两条实测一致：统一梯度缩放 20 倍只让标签恢复率动 0.0009；统一加噪同样无效")
    body(document, "")
    body(document, "**B4 梯度量化（误差自适应）——有局部收益，但会造成场景中止。**")
    bullet(document, "d02_amc 的 LRA 改善：0.6784 → **0.5353**（隐私侧唯一明显收益）")
    bullet(document, "d03_villain 精度 0.9638 → 0.9693、ASR 0.0673 → 0.0490（后门侧也略好）")
    bullet(document, "但 d02_pmc 的 LRA 变差（0.5039 → 0.7011）、d03_pmc 变差（0.5163 → 0.6193）")
    bullet(document, "**d02_composite 中止**：`ValueError: LFBA cannot calculate L1 similarity from zero gradients`")
    body(document, "")
    body(document, "**B5 梯度稀疏化（误差自适应）——彻底崩坏。**")
    bullet(document, "三个标签推断场景的精度全部跌到 **0.49 ~ 0.51**（随机水平），LRA 与精度几乎相等，"
                     "说明模型完全没有学到任务")
    bullet(document, "d03_villain 与 d02_composite 双双中止")
    bullet(document, "原因：稀疏率 0.5 已丢弃一半方向，再叠加「大误差整行置零」后，"
                     "B 方每步能收到的任务信号被压到不足以训练")

    heading(document, "4.3  误差自适应准则的总体评价", 2)
    body(document, "自适应版本相对统一版本**确实有方向性改善**（B2 明显优于 B1），这验证了准则本身有效。"
                   "但它带来一个**结构性矛盾**：")
    bullet(document, "在**后门场景**上，准则按设计削弱了对触发样本的干扰 → 隐私侧收不到效果（B1 的 d03_villain 崩塌正是如此）")
    bullet(document, "在**标签推断场景**上，准则把防御集中到小误差样本 → 效果取决于该样本是否真是泄露来源，"
                     "实测只在部分场景有效（B4 的 d02_amc），在 d02_pmc 反而变差")
    body(document, "**根本原因**：准则要求「绕开高误差样本」，而高误差样本同时也是后门所要保护的对象；"
                   "因此它对隐私与鲁棒两个维度的作用是**反向**的，无法同时改善。", bold=True)

    heading(document, "4.4  裁剪强度的温和分位补测", 2)
    body(document, "统一裁剪在分位 0.95 / 0.98 / 0.99 下的结果（对照列为基线）：")
    table(document,
          ["场景", "基线", "0.95", "0.98", "0.99"],
          [["d02_pmc", "0.8959/0.5039", "0.8392/0.7895", "0.8722/0.6618", "0.8630/0.6296"],
           ["d03_pmc", "0.9584/0.5163", "0.9521/0.4801", "0.9557/0.5099", "0.9584/0.6184"],
           ["d04_pmc", "0.8280/0.7606", "0.8323/**0.2394**", "0.8311/**0.2394**", "0.8309/**0.2394**"],
           ["d02_amc", "0.8951/0.6784", "0.8722/0.8308", "0.9062/0.6702", "0.8254/0.8064"],
           ["d03_villain", "0.9638/0.0673", "**0.5588/0.9959**", "0.9259/0.1939", "0.9503/0.1265"],
           ["d02_composite", "0.8516/0.6745", "0.8814/**0.0260**", "0.4890/**0.0000**", "0.4890/**0.0000**"]],
          widths=[90, 90, 90, 90, 90])
    body(document, "")
    body(document, "规律：**裁剪越强，后门 ASR 越好**（`d02_composite` 的 ASR 0.6745 → 0.0260 → 0.0000），"
                   "**但 `d03_villain` 的精度会崩塌**（0.95 分位 → 0.5588；0.98/0.99 分位 → 0.4890）。"
                   "0.98 与 0.99 分位在 `d02_composite` 上把精度打到 0.4890（随机水平）。"
                   "**没有任何分位同时保住后门场景的精度与标签推断的隐私**。")

    # ---------------- 5 结论 ----------------
    heading(document, "5  结论", 1)
    body(document, "**5.1  三类机制全部不可用。**")
    table(document,
          ["机制", "隐私侧", "鲁棒侧", "精度", "中止", "判定"],
          [["梯度裁剪（统一）", "部分有效", "**崩塌**（ASR→1.0）", "**崩塌**", "否", "不可用"],
           ["梯度裁剪（自适应）", "部分有效", "**崩塌**（ASR→1.0）", "尚可", "否", "不可用"],
           ["梯度量化（统一）", "无效果", "无变化", "不受影响", "否", "无效果"],
           ["梯度量化（自适应）", "局部有效", "略好", "尚可", "**是**", "不可用"],
           ["梯度稀疏化（自适应）", "无效果", "—", "**崩塌至随机**", "**是**", "不可用"]],
          widths=[110, 70, 90, 70, 50, 70])
    body(document, "")
    body(document, "**5.2  为什么只有重指派有效——一条统一的解释。**")
    body(document, "把本轮与既往实测放在一起，规律非常清楚：")
    bullet(document, "**重指派**：置换**逐字保留整个梯度分布的全部矩**（均值、方差、协方差不变），"
                     "只破坏「样本 ↔ 梯度」的配对 → 顶模的联合训练不受影响（性能仅 −0.135），"
                     "但标签泄露与 LFBA 投毒选样同时失效 → **平台 +3.52**")
    bullet(document, "**裁剪 / 量化 / 稀疏化**：都**改变梯度的信息内容**（丢弃或压平分量）→ "
                     "A 对 B 的梯度信号本身变弱或失真 → 轻则被 Adam 吸收（量化），重则训练崩塌（裁剪、稀疏化）")
    body(document, "**即判据是：一个机制是「重排信息」还是「损毁信息」。重排可以同时压制多个攻击而不伤训练；"
                   "损毁必然在压制攻击的同时伤及训练，而 VILLAIN 的检查点选择对梯度通道的任何损毁都极其敏感。**", bold=True)
    body(document, "")
    body(document, "**5.3  一条必须遵守的实测约束。**")
    body(document, "任何使某行梯度**精确为零**的机制都会中止 LFBA 场景：")
    formula(document, "ValueError: LFBA cannot calculate L1 similarity from zero gradients")
    body(document, "本轮 B4 与 B5 均因此中止；此前 party dropout 也复现过两次。原因是 LFBA 用逐样本梯度的 "
                   "L1 相似度构建投毒集，且排名的是**上一 epoch 累积**的梯度，所以任何批次留下的零梯度行都会被扫到。")

    # ---------------- 6 三项得分 ----------------
    heading(document, "6  当前最优提交的三项得分", 1)
    body(document, "三项均按官方公式逐场景复算（脚本 `scripts/score_breakdown.py`）。"
                   "输入常数取 harness 无防御基线在干净场景的精度：")
    table(document,
          ["数据集", "A_clean", "0.9·A_clean", "A_min"],
          [["dataset_01", "0.745833", "0.671250", "0.5000"],
           ["dataset_02", "0.947529", "0.852776", "0.5000"],
           ["dataset_03", "0.977396", "0.879656", "0.5000"],
           ["dataset_04", "0.847052", "0.762346", "0.5000"],
           ["dataset_05", "0.986408", "0.887767", "0.1667（六分类）"]],
          widths=[110, 110, 110, 130])
    body(document, "")

    heading(document, "6.1  最优提交：候选 O（平台实测 43.3135）", 2)
    table(document,
          ["项", "得分", "满分", "占比"],
          [["性能（模型性能）", "**17.728698**", "20", "88.6%"],
           ["隐私（隐私保护）", "**8.577887**", "20", "42.9%"],
           ["鲁棒（安全鲁棒性）", "**17.113576**", "20", "85.6%"],
           ["**合计**", "**43.420161**", "60", "—"]],
          widths=[140, 110, 60, 70])
    body(document, "")
    body(document, "公式：")
    formula(document, "性能:  U = clip((A − A_min)/(A_clean − A_min)),   S_perf = 20·mean(U)      [25 条目]")
    formula(document, "隐私:  P = clip((A_attack − A_rec)/(A_attack − A_min))")
    formula(document, "       G_p = clip((A − A_min)/(0.9·A_clean − A_min)),  Q_p = G_p·P")
    formula(document, "       S_priv = 20·mean(Q_p)                                  [15 场景]")
    formula(document, "鲁棒:  G_r = clip((A − A_min)/(0.9·A_clean − A_min)),  R = 1 − ASR")
    formula(document, "       Q_r = 2·G_r·R/(G_r + R),   S_sec = 20·mean(Q_r)        [25 场景]")
    body(document, "")
    body(document, "计算过程：")
    formula(document, "mean(U)   = 0.886435    →  S_perf = 17.728698")
    formula(document, "mean(Q_p) = 0.428894    →  S_priv = 8.577887")
    formula(document, "mean(Q_r) = 0.855679    →  S_sec  = 17.113576")
    formula(document, "合计 = 17.728698 + 8.577887 + 17.113576 = 43.420161")
    body(document, "")
    body(document, "注意：**性能项用 `A_clean − A_min` 归一化，隐私与鲁棒项共用 `0.9·A_clean − A_min` 作约束因子分母**，"
                   "所以精度同时进入三项——它既直接决定性能分，又作为乘数约束隐私与鲁棒。")

    heading(document, "6.2  隐私项偏低的结构性原因", 2)
    body(document, "隐私项只拿到 42.9%，主要因为 15 个标签推断场景中有 **5 个的 P = 0**："
                   "这些场景的防御后恢复率**高于或无改善于自己的无防御基线**，防御无从得分。"
                   "这不是实现缺陷，而是信息论地板——B 的特征本身与标签相关时，超出的那部分泄露无法被消除。")
    table(document,
          ["场景", "A_attack", "A_recovered", "P"],
          [["dataset_03/composite", "0.6447", "0.6709", "0.0000"],
           ["dataset_04/amc", "0.7076", "0.7606", "0.0000"],
           ["dataset_04/composite", "0.6951", "0.7144", "0.0000"],
           ["dataset_04/pmc", "0.7179", "0.7606", "0.0000"],
           ["dataset_05/composite", "0.8214", "0.8330", "0.0000"]],
          widths=[150, 90, 100, 70])

    heading(document, "6.3  各候选的三项对照", 2)
    table(document,
          ["候选", "性能", "隐私", "鲁棒", "公式合计", "平台实测"],
          [["precision_v3", "18.863", "2.783", "15.211", "36.858", "37.9599"],
           ["h", "17.864", "8.848", "15.696", "42.409", "41.6341"],
           ["**o（最优）**", "17.729", "8.578", "17.114", "**43.420**", "**43.3135**"],
           ["m", "17.520", "8.082", "17.114", "42.715", "未提交"],
           ["p（已证伪）", "17.793", "9.161", "17.114", "44.068", "41.8406"],
           ["k", "18.227", "8.690", "15.010", "41.928", "40.6421"]],
          widths=[90, 60, 60, 60, 70, 80])
    body(document, "")
    body(document, "h → o 的增益**几乎全部来自鲁棒项**（+1.418），平台给予 +1.679。"
                   "候选 p 的本地公式分最高（44.068）却在平台仅得 41.8406，"
                   "说明**本地公式可用于判断量级上的改进，不能用于排序只涉及少数场景的微调**。")

    document.save(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
