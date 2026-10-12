# 两方 VFL 赛题选手基座

本仓库提供两方纵向联邦学习（VFL）的无防御基座、正式攻击场景和统一提交工具。
参赛者可以在 A 方既定模型结构的训练或推理流程中接入防御，但不得替换基座模型，
也不得改变公开数据 ID、正式攻击参数、测试协议或提交格式。最终结果将由组织方
复现核验。

## 快速开始

统一公开参考环境使用 Python 3.12.11、PyTorch 2.8.0 和 torchvision 0.23.0。
组织方复现服务器可根据评测硬件采用与上述接口兼容的厂商构建版本；其他直接依赖
以 `requirements.txt` 中固定的版本为准。
克隆选手仓库：

```powershell
git clone https://github.com/Yanio9/VFL-Security-Challenge.git
cd VFL-Security-Challenge
python -m pip install -r requirements.txt
```

公开数据包由赛题平台单独提供。获取后，请将压缩包保存到：

```text
data/formal_public_datasets_v1.0.7.zip
```

请在仓库根目录执行以下命令，将压缩包直接解压到 `data/datasets/`。

在 Windows PowerShell 中解压：

```powershell
New-Item -ItemType Directory -Force data/datasets | Out-Null
Expand-Archive `
  data/formal_public_datasets_v1.0.7.zip `
  data/datasets -Force
```

在 Linux 中解压：

```bash
mkdir -p data/datasets
unzip data/formal_public_datasets_v1.0.7.zip -d data/datasets
```

解压后的主要目录结构如下。五个 `dataset_XX/` 目录内部结构相同：

```text
VFL-Security-Challenge/
├── data/
│   └── datasets/                       # 公开数据包必须解压到这里
│       ├── README.md
│       ├── manifest.json
│       ├── auxiliary_labels/
│       │   ├── dataset_01_pmc.csv
│       │   ├── ...
│       │   └── dataset_05_pmc.csv
│       ├── dataset_01/
│       │   ├── train/
│       │   │   ├── party_A_train.csv
│       │   │   └── party_B_train.csv
│       │   ├── validation/
│       │   │   ├── party_A_validation.csv
│       │   │   └── party_B_validation.csv
│       │   ├── test_hidden/
│       │   │   ├── party_A_test.csv
│       │   │   └── party_B_test.csv
│       │   └── test_backdoor/
│       │       ├── lfba/
│       │       │   ├── party_A_test.csv
│       │       │   └── party_B_test.csv
│       │       └── villain/
│       │           ├── party_A_test.csv
│       │           └── party_B_test.csv
│       ├── dataset_02/
│       ├── dataset_03/
│       ├── dataset_04/
│       └── dataset_05/
├── configs/
├── main.py
└── requirements.txt
```

可以在仓库根目录检查五个数据集目录是否齐全：

```powershell
Get-ChildItem data/datasets -Directory -Filter "dataset_*" | Select-Object -ExpandProperty Name
```

正常情况下应依次看到 `dataset_01` 至 `dataset_05`。不要把压缩包额外解压到
`data/datasets/formal_public_datasets_v1.0.7/` 子目录中，否则正式配置无法找到数据。

## 正式数据集

五个匿名数据集均为正式赛题数据：

| 数据集 | 类别数 |
|---|---:|
| `dataset_01` | 2 |
| `dataset_02` | 2 |
| `dataset_03` | 2 |
| `dataset_04` | 2 |
| `dataset_05` | 6 |

每个数据集包含训练集、验证集、无标签干净测试集，以及 LFBA 和 VILLAIN
各自的后门测试视图。训练集和验证集标签仅位于 A 方文件；所有测试文件均不包含
标签。两个公开后门测试视图均为完整干净测试集的逐字节副本；攻击在运行时向全部
样本注入触发器，组织方再使用私有标签仅对非目标类样本计算 ASR。

```text
data/datasets/dataset_XX/
├── train/
├── validation/
├── test_hidden/
└── test_backdoor/
    ├── lfba/
    └── villain/
```

A、B 两方通过 `ID` 对齐。运行时不会重新切分数据或改变测试样本顺序。

## 正式场景

每个数据集包含八个场景：

| 场景 | 攻击类型 | 场景说明 |
|---|---|---|
| `clean` | 无攻击 | 按标准纵向联邦学习流程训练，用于评测干净主任务表现。 |
| `lfba_p05` | LFBA 后门攻击 | 使用 LFBA 构造固定毒化样本集合，训练投毒率为 5%。 |
| `lfba_p10` | LFBA 后门攻击 | 攻击方式与 `lfba_p05` 相同，训练投毒率提高为 10%。 |
| `lfba_p20` | LFBA 后门攻击 | 攻击方式与 `lfba_p05` 相同，训练投毒率提高为 20%。 |
| `villain` | VILLAIN 后门攻击 | 恶意 B 方在中间表示中构造并注入加性触发器，使触发样本趋向指定目标类。 |
| `pmc` | 被动标签推断攻击 | B 方在不主动改变联合训练过程的情况下，利用合法可见信息和辅助标签执行模型补全，恢复隐藏测试标签。 |
| `amc` | 主动标签推断攻击 | B 方在训练阶段主动调整本地更新，再执行模型补全并恢复隐藏测试标签。 |
| `composite` | 复合攻击 | 在同一次训练中组合一个后门攻击与一个标签推断攻击，具体组合按数据集固定。 |

`composite` 使用以下固定复合方式：

| 数据集 | 复合场景 |
|---|---|
| `dataset_01`、`dataset_02`、`dataset_04` | LFBA + AMC |
| `dataset_03`、`dataset_05` | VILLAIN + PMC |

### 攻击方法背景

- LFBA：[Label-Free Backdoor Attacks in Vertical Federated Learning（AAAI 2025）](https://ojs.aaai.org/index.php/AAAI/article/view/34246)
- VILLAIN：[VILLAIN: Backdoor Attacks Against Vertical Split Learning（USENIX Security 2023）](https://www.usenix.org/conference/usenixsecurity23/presentation/bai)
- PMC/AMC：[Label Inference Attacks Against Vertical Federated Learning（USENIX Security 2022）](https://www.usenix.org/conference/usenixsecurity22/presentation/fu-chong)

以上论文用于理解攻击背景与基本原理；赛事中的正式攻击行为和参数以本仓库代码及
`configs/scenarios/` 下的场景配置为准。

正式配置位于 `configs/scenarios/`，命名格式为：

```text
configs/scenarios/dataset_XX_<scenario>.yaml
```

## 参赛权限边界

本赛题不规定防御方法在训练或推理流程中的具体接入位置。参赛者扮演持有标签的
A 方，可以使用和修改：

- 在保持 A 方底模和顶模基座结构不变的前提下，修改 A 方本地训练和推理逻辑；
- A 方持有的损失函数、优化过程和梯度处理；
- A 方合法持有的数据、标签、模型状态及协议中可见的通信信息；
- 新增辅助防御模块和防御配置。辅助模块可以包含可训练参数，但不得替换或扩展
  主任务底模、顶模，也不得改变二者约定的输入输出接口。

正式 YAML 展开后的数据、模型、训练、攻击及标签推断字段属于冻结评测协议，不得
修改或绕过。具体包括：

- A/B 两方底模和 A 方顶模的基座结构，包括模型类型、层数、层宽、embedding
  维度及输入输出维度；
- B 方底模、B 方本地优化流程、B 方私有特征和内部状态；
- LFBA、VILLAIN、PMC、AMC 的攻击实现及其正式参数；
- 触发器、投毒率、目标类和各数据集的复合场景定义；
- 公开测试数据、样本 ID、场景配置中的固定评测字段和提交格式。

防御参数使用独立文件配置，默认读取 `configs/defense.yaml`。不要在正式场景 YAML 中
增加或修改防御参数；正式场景运行和打包时会依据 `configs/formal_config_lock.json`
校验展开后的数据、模型、训练、攻击及标签推断协议。`dataRoot`、`device`、进度显示和
`output` 等部署运行字段不计入协议锁，但也不能通过防御配置覆盖。当前共享的
`model.*` 和 `training.*` 会同时改变 B 方或攻击训练过程，不能作为防御参数使用。
该本地锁用于尽早发现协议偏移，最终仍以组织方只读配置副本的复现结果为准。

防御配置示例：

```yaml
defense:
  name: my_defense
  parameters:
    strength: 0.25
```

该文件的根节点只能是 `defense`，`defense` 下只能包含 `name` 和 `parameters`。
`name` 必须是非空字符串，`parameters` 必须是映射；其内部字段由参赛者的防御代码
自行定义，值应使用可确定序列化的字符串键映射、列表、有限数值、布尔值、字符串或
空值。加载后可从 `config["defense"]` 读取这些参数，用于注册防御 hook 或初始化防御
模块；仅填写配置不会自动实现或启用防御。

也可以在仓库 `configs/` 目录下新建 `.yaml` 或 `.yml` 文件，并通过
`--defense-config` 选择。防御配置不能声明或覆盖攻击、训练、模型、数据和标签推断等
正式字段；出现额外根字段或额外 `defense` 字段时，程序会在运行前拒绝该配置。

直接关闭攻击、修改 B 方行为、读取越权信息或伪造预测结果均视为规避评测。组织方
会在复现阶段检查提交代码、运行过程和结果是否遵守上述边界。

### 统一 A 方防御入口

`VflTrainer.configurePartyADefenseHooks(...)` 提供三个 A 方防御扩展入口：

- `trainingRepresentationHook`：处理训练阶段 A 方可见的 cut-layer 表征；
- `cutGradientHook`：处理 A 方返回给 B 方的 cut-layer 梯度；
- `inferenceRepresentationHook`：处理推理阶段 A 方可见的表征。

普通训练、LFBA、VILLAIN 及复合场景均统一经过这些入口。攻击协议规定的通信变换
先执行，再由 A 方防御处理其合法可见的信息；默认不注册 hook 时保持无防御基线。

两个表征 hook 分别返回 `(cutA, cutB)` 和 `(embeddingA, embeddingB)`；梯度 hook
返回 `gradientB`。返回张量必须保持原有的 shape、dtype 和 device，且不能包含
NaN 或 Inf。

## 运行场景

在项目根目录运行一个正式场景：

```powershell
python main.py --config configs/scenarios/dataset_03_clean.yaml
```

上述命令默认读取 `configs/defense.yaml`。使用另一份防御配置时执行：

```powershell
python main.py --config configs/scenarios/dataset_03_clean.yaml `
  --defense-config configs/my_defense.yaml
```

运行复合场景：

```powershell
python main.py --config configs/scenarios/dataset_03_composite.yaml
```

需要关闭进度条时增加 `--no-progress`。

## 生成提交 ZIP

运行全部 40 个场景并生成提交文件：

```powershell
python scripts/build_submission.py --run-all `
  --team-id team001 `
  --model-version v03
```

不指定 `--defense-config` 时默认使用 `configs/defense.yaml`。修改防御配置会使对应的
场景缓存失效；如果运行时选择了自定义路径，后续断点续跑和 `--package-only` 必须继续
传入同一路径。

> ⚠️ **重要**：`configs/defense.yaml` 指向的是仓库自带的 `robust_vfl` 基线，**不是**
> 本项目开发并提交的防御。要复现本项目的结果，必须显式传入
> `--defense-config configs/defense_purified.yaml`。`build_submission.py` 在使用裸默认值
> 时会打印同样的提醒。

## 本仓库实现的防御（快速开始）

本项目在 A 方三个合法可见的通信接口上实现了一个防御方法 `purified_vfl`
（代码 `defense/purified_vfl.py`，报告 `docs/defense_report.md`）。复现与提交所需的最小命令：

```powershell
# 1) 复现已提交的配置（约 1.5-2 小时，40 场景 x 100 epoch）
python scripts/run_variant_snapshot.py --variant precision_v3

# 2) 或仅从已有缓存重新打包（无需重训）
python scripts/build_submission.py --package-only `
  --defense-config configs/defense_purified.yaml `
  --results-dir outputs/submission_cache_precision `
  --team-id <team_id> --model-version <version>
```

已记录的四个工作点（均跑满 40 场景 x 100 epoch、各 40/40 指标快照、SHA-256 零不符）。
按本地精度排序，**推荐先提交 nojitter**（评分对精度最敏感，见 docs/SOTA_comparison.md）：

| 配置 | 机制要点 | 本地 40 场景均值（精度 / LRA） | 提交包 |
|---|---|---|---|
| `configs/defense_purified_nojitter.yaml` | 关闭注入批抖动 | **0.8875** / 0.7239 | `team001__nojitter.zip` |
| `configs/defense_purified_jitter_villain.yaml` | 抖动只保留在 VILLAIN 场景 | 0.8867 / 0.7239 | `team001__jitter_villain.zip` |
| `configs/defense_purified.yaml` | 软阈值共模消除（阈值 3.0）+ 注入批抖动 0.5 | 0.8844 / 0.7239 | `team001__precision_v3.zip` |
| `configs/defense_purified_villain_gate.yaml` | 增加"仅 VILLAIN 场景无条件共模消除" | 0.8837 / 0.7283 | `team001__villain_gate.zip` |

自带 `robust_vfl` 基线为 0.8737 / 0.7332，即四个工作点在本地可复现的两个维度上均优于基线。

详见 `docs/SOTA_comparison.md`（与自带 SOTA 的量化对比）、`docs/variants.md`（全部变体、
被否决机制与原因、量纲警告）与 `docs/DELIVERY.md`（交付与复现说明）。

最终文件默认保存为：

```text
outputs/<team_id>__<model_version>.zip
```

ZIP 包含根目录 `manifest.json` 和 80 个预测 CSV。打包脚本会检查数据集、场景、
文件名、列名、ID、行序、预测类别和 SHA-256，不需要手工整理目录或编辑清单。

如果已经生成完整缓存，只重新校验并打包：

```powershell
python scripts/build_submission.py --package-only `
  --team-id team001 `
  --model-version v03
```

如使用自定义防御配置，应在上述两条命令中均增加
`--defense-config configs/my_defense.yaml`。

单场景更新、断点续跑、缓存保留和常见错误处理见
[选手结果生成与提交指南](docs/submission_guide.md)。完整格式示例位于
`examples/submission_example.zip`。

## 提交要求

- 每个测试 ID 必须且只能出现一次，并保持公开文件中的顺序；
- `dataset_01` 至 `dataset_04` 的预测类别为 `0–1`；
- `dataset_05` 的预测类别为 `0–5`；
- PMC/AMC 的标签推断文件仅包含隐藏测试样本的 `ID,Prediction`；
- 不得向提交 ZIP 添加 checkpoint、日志、指标、标签或其他调试文件；
- 应保留生成提交文件时使用的代码、配置和缓存，供组织方复现。

自动评分由性能、隐私和鲁棒性三部分组成，每部分 20 分。自动评测 JSON 不包含工程分；工程分及其他赛制分项由组织方另行评定。
