# PFNS 防御策略：软编码伪标签（Soft Alias Encoding）

> 生成时间：本阶段 PFNS 试点。配置文件：`configs/defense_purified_pfns.yaml`

## 一、机制定义

把真实标签映射到**别名块的均匀软分布**，而不是硬映射到单个别名。

| 项 | 硬编码（旧） | 软编码（PFNS） |
|---|---|---|
| 目标形式 | one-hot 单个别名 | 块内均匀概率分布 |
| 类 c 的目标 | 随机 pick 块内 1 个别名 | 块内每个别名各 `1/K` |
| 损失 | CrossEntropyLoss（整数标签） | `-(target·log_softmax(logits)).sum(1).mean()`（等价 KL） |
| 逆映射 | `(alias − base) // K` | 同左（argmax 落在块内） |

**别名布局**（二分类、K=2、输出 8 维）：

| 真实类 | 别名块 |
|---|---|
| 0 | {4, 5} 各 0.5 |
| 1 | {6, 7} 各 0.5 |

`输出维度 = 2 × 真实类数 × K`，偏移 `base = 真实类数 × K`（别名**不含 0/1**）。

## 二、为什么软编码优于硬编码

玩具实测（`scripts/probe_alias_soft.py`，逆映射后精度，基线 0.9514）：

| K | 硬编码 | 软编码 | 软−硬 |
|---|---|---|---|
| 2 | 0.9443 | **0.9471** | +0.003 |
| 4 | 0.9271 | **0.9500** | **+0.023** |

**K=4 软编码几乎无损**（0.9500 ≈ 基线 0.9514），而硬编码 K=4 掉到 0.9271。

原因：硬编码要求模型在块内做**随机 pick 的硬选择**（pick 与输入无关 → 优化困难）；
软编码让模型学"块内均匀"，目标平滑、优化稳定。

## 三、软编码的额外优势（梯度通道）

软编码不仅改标签，还改了回传给 B 的梯度内容：

```
硬编码: gradientB[i] = softmax(logits_i) − onehot(alias_i)   ← 尖锐，指向单类别
软编码: gradientB[i] = softmax(logits_i) − 块内均匀分布        ← 平滑，指向整个块
```

B 收到的梯度不再对应单一类别，可能进一步混淆标签推断——**这是软编码相比硬编码的独立增益通道，尚未在真实 VFL 中量化**。

## 四、参数

```yaml
defense:
  name: purified_vfl
  parameters:
    pseudoLabelStrength: 1.0      # 启用伪标签
    labelAliasFactor: 2           # 每类 K 个别名（输出 = 2*C*K）
    labelAliasSoft: true          # 软编码（false = 硬编码）
    pseudoLabelSeed: 11           # 块内随机性种子（软编码下块内均匀，种子仅影响硬编码）
    # …候选 O 的重指派 / party dropout / jitter / norm excess 保持不变
```

## 五、实现位置（3 处）

| 文件 | 改动 |
|---|---|
| `defense/purified_vfl.py` | `labelAliasSoft` 参数 + `_softLabel()`（生成 2D soft target） |
| `core/trainer.py` | 损失处检测 2D target → KL 散度形式 |
| `core/experiment.py` | 输出维度 `2*C*K` 注入（`setPseudoLabelClasses`） |

## 六、验证状态

| 项 | 结果 |
|---|---|
| 玩具可学性 | ✅ K=2 0.9471 / K=4 0.9500 |
| 端到端（真实代码） | ✅ K=2 0.9529 / K=4 0.9500（`probe_alias_soft_e2e.py`） |
| 配置加载 | ✅ YAML 解析 + 防御构建 + soft target 形状正确 |
| 真实 VFL | 🔄 PFNS 试点运行中（PFNS1_K2 / PFNS2_K4，无重指派对照） |

## 七、待验证 / 已知限制

1. **全局参数**：`labelAliasSoft`/`labelAliasFactor` 影响所有场景的输出维度（2 → 2C·K·2）。
   软编码当前**未按场景门控**，对后门（VILLAIN/LFBA）场景的影响需全量确认。
2. **真实 VFL 对照**：PFNS（软）与 PFN（硬）同基线，待结果确认软编码能否修复
   硬编码的两个实测问题（`d02_amc` K=2 精度暴跌、`d05_pmc` 六分类恶化）。
3. **K 值选择**：K=2 保守（精度损失小）、K=4 在 `d02_amc` 更接近随机但 `d03/d05` 更差，
   最终取值待 PFNS/PFN 结果后决定。
