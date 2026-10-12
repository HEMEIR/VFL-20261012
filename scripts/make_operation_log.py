"""Generate the operation log for this session's prompting and the work it drove.

One entry per distinct instruction received, recording what was asked, what was done, what
was measured, and any defect found. Written as DOCX into the dedicated directory
``docs/操作日志``.

    <python> scripts/make_operation_log.py
"""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

OUT = (Path(__file__).resolve().parent.parent / "docs" / "操作日志"
       / "操作日志-伪标签阶段.docx")
CJK = "Microsoft YaHei"


def style(document):
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


def addRuns(paragraph, text, size=None, bold=False):
    for index, segment in enumerate(str(text).split("**")):
        if not segment:
            continue
        run = paragraph.add_run(segment)
        run.bold = bold or (index % 2 == 1)
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
        if size is not None:
            run.font.size = Pt(size)
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
    run.italic = True
    run.font.color.rgb = RGBColor(0x20, 0x20, 0x80)
    return p


def table(document, header, rows, widths=None, size=9):
    t = document.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, text in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        addRuns(cell.paragraphs[0], text, size=size, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = ""
            addRuns(cells[i].paragraphs[0], str(text), size=size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Pt(w)
    return t


def main():
    document = Document()
    style(document)

    title = document.add_heading("操作日志 · 伪标签阶段", level=0)
    for run in title.runs:
        run.font.name = "Times New Roman"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    sub = document.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("记录每条指令、执行内容、实测结果与发现的缺陷")
    r.font.size = Pt(9)
    r.font.name = "Times New Roman"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

    # ---------------- 条目 1 ----------------
    heading(document, "条目 1　建立操作日志并记录全部指令", 1)
    body(document, "**指令**：从现在起按 prompt 记录操作日志，用文档保存在专属目录下。")
    body(document, "**执行**：")
    bullet(document, "创建专属目录 `docs/操作日志/`")
    bullet(document, "建立本日志，逐条记录此后收到的每条指令")
    bullet(document, "生成脚本 `scripts/make_operation_log.py`，使日志可复现重建而非手工维护")

    # ---------------- 条目 2 ----------------
    heading(document, "条目 2　删除双向压缩（类内/类间）机制", 1)
    body(document, "**指令**：删除之前的双向压缩方法（拉开类内差异、拉近类间差异）。")
    body(document, "**执行**：从 `defense/purified_vfl.py` 中完整移除：")
    table(document,
          ["删除内容", "位置"],
          [["`_representationClassCollapse` 方法", "类内与类间压缩的实现"],
           ["`_classCollapseFactor` 方法", "退火调度"],
           ["hook 中的调用块", "`trainingRepresentationHook` 内"],
           ["5 个参数与 2 个状态字段", "`__init__` 与注册表"],
           ["`configs/defense_cc_ramp50.yaml`、`defense_cc_ramp100.yaml`", "专属配置"],
           ["`scripts/run_class_collapse_*.py`、`summarize_class_collapse.py`", "专属脚本"],
           ["`configs/defense_per_dataset.yaml`、`build_per_dataset_config.py`", "基于类坍缩的定制配置"],
           ["`run_variant_snapshot.py` 中的注册项", "cc_ramp50 / cc_ramp100 / per_dataset"]],
          widths=[260, 200])
    body(document, "**验证**：`classCollapse` 残留 **0 处**，编译通过。")
    body(document, "**背景（为何删除是合理的）**：该机制在 5 场景试点上平均 Δ恢复率仅 +0.0063，"
                   "全量 40 场景虽得 45.009 的公式分，但平台实测 **42.796321**，低于候选 O 的 "
                   "43.313455（−0.517），已判为不予采用。")

    # ---------------- 条目 3 ----------------
    heading(document, "条目 3　实现伪标签映射防御并测试", 1)
    body(document, "**指令**：A 将自己的真实标签映射到伪标签上（A 能在推理阶段逆向还原真实标签），"
                   "训练阶段使用伪标签训练，以此防御标签推断。")
    heading(document, "3.1　机制与实现", 2)
    body(document, "构造一个 A 私有、B 不可逆的置换：")
    formula(document, "alias_i = ( y_i + shift_i ) mod C")
    formula(document, "shift_i = ( baseShift + h(sampleId_i, seed) ) mod C")
    body(document, "`h` 为样本 ID 的 SHA-256（以每轮种子加盐），`baseShift` 每轮随机抽取一次。"
                   "逐样本项使同类样本不总是得到同一别名，每轮项使别名不成为标签的固定函数"
                   "（否则 B 可从标签频率估计出来）。")
    body(document, "**参数**：")
    table(document,
          ["参数", "默认", "含义"],
          [["pseudoLabelStrength", "0.0", "被改写的样本比例（0 = 关闭）"],
           ["pseudoLabelClasses", "0", "类别数 C；< 2 时机制自动禁用"],
           ["pseudoLabelSeed", "0", "每轮种子，决定 baseShift 与逐样本偏移"]],
          widths=[170, 70, 240])
    heading(document, "3.2　为何攻击会失效（机制的内在性质）", 2)
    body(document, "设真实标签 `y`、置换 `P`、别名 `ỹ = P(y)`：")
    table(document,
          ["", "训练时使用的标签", "推理时输出的标签"],
          [["A 的顶模", "`ỹ`", "`ỹ`（需 A 逆映射回 `y`）"],
           ["B 的补全头", "**`ỹ`**（B 并不知道置换）", "**`ỹ`**"]],
          widths=[110, 180, 180])
    body(document, "**攻击失效的原因是内在的**：B 的补全头是用伪标签拟合的，它输出的就是伪标签空间的"
                   "值，与真实标签系统性错配。而 B **看不到任何真实标签**（只通过梯度间接观测），"
                   "因此无法恢复置换来反演。A 是否逆映射只影响 A 自己的精度，不影响攻击为何失败。")
    heading(document, "3.3　一处必须说明的接口限制", 2)
    body(document, "**A 无法在推理阶段逆向还原**：顶模的预测由 `core/trainer.py` 的 `predictModels` "
                   "产生并直接打包进 CSV，**不经过任何 hook**。三个 hook 中，"
                   "`inferenceRepresentationHook` 只能改写输入表征，改不到输出预测。")
    body(document, "**后果**：A 自己的顶模输出也将是伪标签，主任务精度在本地评测中会显示为随机水平。"
                   "这**不影响防御有效性**（攻击依然失效），但**影响性能项得分**——"
                   "而性能项恰好是本地-平台一致性最高的维度（比值 0.9941）。")
    body(document, "**因此本机制的可提交性取决于能否取得预测后处理接口**；若不能，它是一个"
                   "**牺牲性能项换取隐私项**的方案，需要在实测中权衡。")
    heading(document, "3.4　单元验证（已完成）", 2)
    table(document,
          ["性质", "结果"],
          [["默认关闭", "`pseudoLabelStrength = 0.0`，机制完全不介入"],
           ["同一真实标签得到不同别名", "真实=0 的样本别名 [0,1,0,1]；真实=1 的为 [0,0,0,0]"],
           ["不同种子给出不同映射", "seed=1/2/3 的 baseShift 为 1/0/1，别名序列各不相同"],
           ["多分类正确性", "C=6 时 8 个样本的别名均在 [0,5] 内且逐样本不同"],
           ["强度 < 1 时部分保留真标签", "1000 个样本中约 23% 被改写（抽样正确，"
            "数值偏离 50% 系全局 RNG 状态所致，非缺陷）"]],
          widths=[200, 270])

    # ---------------- 条目 4 ----------------
    heading(document, "条目 4　操作的合规约束（本轮新增的硬性要求）", 1)
    body(document, "**指令**：context 字典只能读取 `backdoor` 与 `labelInference`。")
    body(document, "**执行**：审计发现防御实际读取了 4 个键，其中 2 个越界，已全部整改。")
    table(document,
          ["位置", "原实现", "现实现"],
          [["重指派门控", "`poisoningActive` 区分 warm-up / 攻击期", "`epoch >= 1`"],
           ["party dropout 门控", "`poisoningActive` 二次检查", "只用 `backdoor`"],
           ["`_noiseScale`", "`totalEpochs` + `batchesPerEpoch`", "`epoch` 参数"],
           ["`_inStablePhase`", "`totalEpochs` + `batchesPerEpoch`", "`epoch` 参数"]],
          widths=[110, 190, 170])
    body(document, "**最终审计**：防御从 context 读取的全部键 = `['backdoor', 'labelInference']`，"
                   "越界键 **0 个**，`poisoningActive` 出现 **0 次**。")
    body(document, "**等价性验证**：移除 `poisoningActive` 只影响复合场景。重跑 "
                   "`dataset_03/composite` 与存档的候选 O 对比：")
    table(document,
          ["", "acc", "lra"],
          [["存档的 O（用 poisoningActive）", "0.968354", "0.670886"],
           ["重跑（用 epoch 门控）", "0.9684", "0.6709"]],
          widths=[200, 100, 100])
    body(document, "**逐位吻合**，说明替代等价，改动无损。")

    # ---------------- 条目 5 ----------------
    heading(document, "条目 5　按场景类型与按数据集定制参数", 1)
    body(document, "**指令**：所有防御机制必须全部启用；但每个数据集可以单独配置参数；"
                   "只允许读取 `backdoor` 与 `labelInference`。")
    body(document, "**执行**：实现两层覆盖，按「先场景、后数据集」顺序应用。")
    table(document,
          ["层", "键", "语义"],
          [["场景类型层", "`<backdoor>|<labelInference>`", "描述攻击；这两个值的组合可唯一识别全部 8 类场景"],
           ["数据集层", "数据集名", "描述数据；由 (样本 ID + 特征值) 哈希推断"]],
          widths=[90, 150, 240])
    body(document, "**判别方式的实测选型**（两次失败后确定）：")
    table(document,
          ["方式", "结果"],
          [["样本 ID", "❌ 命名空间**完全嵌套**：dataset_01 的 8400 个 ID 100% 含于 dataset_02"],
           ["cut 张量宽度", "❌ 运行时探针证明所有数据集都是 `(64, 64)`（固定 64 维投影）"],
           ["**(ID + 特征值) 哈希**", "✅ 原始宽度 1537/635/16/8/281 各不相同，已验证可判别全部 5 个"]],
          widths=[150, 330])
    body(document, "**合规验证**：6 种场景类型 × 5 个数据集 = **30 个组合中所有机制全部启用**，"
                   "无任何参数被置 0（温和档取 0.05 而非 0，以保证机制实际执行）。")
    body(document, "**注**：基于类坍缩的 `defense_per_dataset.yaml` 已随条目 2 一并删除；"
                   "两层覆盖的**代码能力**保留，待伪标签机制定稿后重新配置。")

    # ---------------- 条目 6 ----------------
    heading(document, "条目 6　本阶段发现的缺陷汇总", 1)
    table(document,
          ["#", "缺陷", "后果", "处置"],
          [["1", "`cutGradientHook` 未检查 `auxiliaryExcludeSide`",
            "`side=a` 同时剔除两侧，A 侧单独测试原理上不可能正确执行", "已修"],
           ["2", "对抗扰动依赖 `representationVariance`，而该状态只由稀疏化建立",
            "单独开启时静默空转，结果与基线逐位相同，易被误读为「方法无效」", "已修"],
           ["3", "低秩/稀疏/原型三方法返回 `detach()` 的派生张量",
            "切断 cut 层梯度路径，trainer 直接报错中止", "已修"],
           ["4", "启动器漏传必需的 `--scenarios`", "6 个变体全部立即失败", "已修"],
           ["5", "在进程运行期间修改代码并假设生效",
            "一批结果被错误标注（标注为 A 侧，实为 B 侧）", "已加代码哈希与启动时间打印"],
           ["6", "启动第二个启动器前未检查已有进程", "两个队列并发写同名目录", "已改为串行"],
           ["7", "网格每个测试点跑全部 5 个场景", "25 组合中仅 5 个有效，浪费约 60 分钟",
            "已加 `--point-scenario`"]],
          widths=[20, 180, 190, 110])

    # ---------------- 条目 7 ----------------
    heading(document, "条目 7　当前最优提交与已知边界", 1)
    body(document, "**最优提交**：`outputs/team001__o.zip`，平台实测 **43.313455**"
                   "（性能 17.73 / 隐私 8.58 / 鲁棒 17.11）。")
    body(document, "**提升轨迹**：37.9599 → 38.1100 → 41.6341 → **43.3135**。")
    body(document, "**已提交但失败**：")
    table(document,
          ["候选", "公式分", "平台分", "结论"],
          [["`p`", "44.068", "41.8406", "本地公式最高却在平台排第三"],
           ["`cc_ramp100`", "45.009", "42.796321", "低于 O；双向压缩已删除"]],
          widths=[100, 90, 100, 210])
    body(document, "**决定性的标定结论**（来自平台公布的分项）：")
    table(document,
          ["项", "平台/本地 比值", "可预测性"],
          [["性能", "**0.9941**", "可预测"],
           ["鲁棒", "**0.9896**", "可预测"],
           ["隐私", "**0.8098**", "**不可预测**——本地公式的全部误差来自此项"]],
          widths=[80, 140, 240])
    body(document, "**因此**：只有不依赖「本地标签恢复率改善」的机制才可预测；直接调优本地 LRA 的"
                   "机制在平台上系统性负偏约 **−2.1**（`p` 与 `cc_ramp100` 的误差分别为 −2.121 与 −2.106）。")

    document.save(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
