# 选手结果生成与提交指南

本指南说明如何运行正式评测场景、复用已有结果、单独更新某个场景，并生成组织方
可以直接校验的统一提交压缩包。

正式提交统一通过以下脚本生成：

```text
scripts/build_submission.py
```

选手不需要手工创建目录、复制 CSV 或编辑 `manifest.json`。手工修改缓存中的预测
文件会造成校验值不一致，应重新运行对应场景生成结果。

组织方同时提供 `submission_example.zip` 作为完整格式示例。它包含五个数据集、
40 个场景和 80 个预测文件；其中预测统一为合法占位类别，仅用于说明目录、列名、
ID 和行序，不代表任何基线成绩。

## 1. 脚本会生成什么

脚本会生成三类本地输出。

### 1.1 完整场景运行结果

默认位置：

```text
outputs/scenarios/dataset_XX/scenario_name/
```

每个场景默认保留最近两次成功运行：

```text
outputs/scenarios/dataset_01/lfba_p05/
├── current/
│   ├── checkpoint.pt
│   ├── metrics.json
│   ├── attack_audit.json
│   ├── main_task_predictions.csv
│   ├── backdoor_predictions.csv
│   └── run_manifest.json
└── previous/
    ├── checkpoint.pt
    ├── metrics.json
    ├── attack_audit.json
    ├── main_task_predictions.csv
    ├── backdoor_predictions.csv
    └── run_manifest.json
```

`current` 是最近一次校验成功的完整运行结果，提交缓存始终从 `current` 更新。
`previous` 是更新前的上一版 `current`，用于比较防御效果、恢复 checkpoint 或
回退预测结果。场景第一次成功运行时只有 `current`，第二次成功运行后才会出现
`previous`。

运行失败、预测校验失败或提交缓存更新失败时，原有 `current` 和 `previous` 均
保持不变。

### 1.2 场景结果缓存

默认位置：

```text
outputs/submission_cache/
```

缓存按照“数据集/场景”组织。例如：

```text
outputs/submission_cache/
└── dataset_01/
    ├── clean/
    │   ├── main_task_predictions.csv
    │   └── run_manifest.json
    ├── lfba_p05/
    │   ├── main_task_predictions.csv
    │   ├── backdoor_predictions.csv
    │   └── run_manifest.json
    └── composite/
        ├── main_task_predictions.csv
        ├── backdoor_predictions.csv
        ├── label_inference_predictions.csv
        └── run_manifest.json
```

缓存用于避免每次打包都重新运行全部场景。`run_manifest.json` 记录该场景的随机
种子、完整配置哈希、正式协议配置/锁哈希、防御配置路径及哈希、数据指纹、源码内容
校验值以及预测文件校验值。

缓存目录不是最终提交包，请勿直接提交。

### 1.3 最终提交压缩包

默认位置按照队伍编号和模型版本自动生成：

```text
outputs/<team_id>__<model_version>.zip
```

最终 ZIP 只包含：

- 根目录 `manifest.json`；
- 40 个正式场景下的 80 个预测 CSV。

ZIP 不包含：

- checkpoint；
- 训练日志；
- `metrics.json`；
- `attack_audit.json`；
- 每个场景缓存中的 `run_manifest.json`；
- 训练集预测；
- 组织方私有标签。

这些文件不进入 ZIP，但仍保留在 `outputs/scenarios/.../current` 和
`previous` 中供选手本地分析。

根目录 `manifest.json` 由脚本自动生成，主要结构如下：

```json
{
  "schemaVersion": "1.0",
  "team_id": "team001",
  "model_version": "v03",
  "datasetCount": 5,
  "scenarioCount": 40,
  "predictionFileCount": 80,
  "scenarios": [
    {
      "dataset": "dataset_01",
      "scenario": "clean",
      "predictionFiles": {
        "main_task_predictions.csv": {
          "sha256": "对应CSV文件的SHA-256校验值"
        }
      }
    }
  ]
}
```

清单不重复记录 `rowCount`。打包脚本和评分器会直接读取 CSV，根据官方 ID 清单
检查实际行数、ID 完整性、唯一性和顺序；SHA-256 用于检查文件内容是否被修改。
源码版本、配置哈希等复现信息只保存在本地 `run_manifest.json` 中。

## 2. 正式场景清单

每个数据集包含以下 8 个场景：

```text
clean
lfba_p05
lfba_p10
lfba_p20
villain
pmc
amc
composite
```

`composite` 是统一的场景类型标识，实际复合攻击按数据集固定：
dataset_01/02/04 为 `LFBA + AMC`（dataset_04 的 AMC 从第 2 轮启用），
dataset_03/05 为 `VILLAIN + PMC`。

五个数据集共 40 个场景。

### 2.1 正式配置协议锁

正式 YAML 展开后的数据、模型、训练、攻击及标签推断字段均为冻结协议字段。正式
场景运行和打包时会依据 `configs/formal_config_lock.json` 进行校验，协议字段发生
变化会阻止正式运行或打包。

参赛者不得在正式场景 YAML 中增加或调整防御参数。`dataRoot`、`device`、进度显示
和 `output` 等仅用于部署运行的字段不计入协议锁；其中 `dataRoot` 只能改变正式数据
的存放位置，不能改变数据内容。当前共享的 `model.*` 和 `training.*` 会同时影响 B 方
或攻击训练过程，不能借此配置防御，也不能通过防御配置覆盖。

本地锁文件用于在运行前检查协议一致性，不是最终信任依据；最终以组织方保存的
只读配置副本进行复现和核验。

### 2.2 独立防御配置

防御参数使用独立文件配置，默认路径为 `configs/defense.yaml`：

```yaml
defense:
  name: my_defense
  parameters:
    strength: 0.25
```

文件根节点只能是 `defense`，`defense` 下只能包含 `name` 和 `parameters`。
`name` 必须是非空字符串，`parameters` 必须是映射；内部字段可由防御方法自行定义，
值应使用可确定序列化的字符串键映射、列表、有限数值、布尔值、字符串或空值。
配置加载后可由选手代码从 `config["defense"]` 读取，用于注册防御 hook 或初始化防御
模块；仅填写配置不会自动实现或启用防御。

如需保留多组参数，可以在仓库 `configs/` 目录下新建 `.yaml` 或 `.yml` 文件，并在
命令中指定：

```powershell
python main.py --config configs/scenarios/dataset_03_clean.yaml `
  --defense-config configs/my_defense.yaml
```

防御配置不能声明或覆盖攻击、训练、模型、数据、标签推断及其他正式场景字段；出现
额外根字段或额外 `defense` 字段时会在运行前报错。路径也必须位于本仓库的
`configs/` 目录内。运行、断点续跑和 `--package-only` 必须使用同一防御配置路径。

提交包中的预测文件数量为：

| 类型 | 数量 | 适用范围 |
|---|---:|---|
| `main_task_predictions.csv` | 40 | 每个场景都必须生成 |
| `backdoor_predictions.csv` | 25 | 启用 LFBA 或 VILLAIN 的场景 |
| `label_inference_predictions.csv` | 15 | 启用 PMC 或 AMC 的场景 |
| 合计 | 80 | 不包含根目录 `manifest.json` |

## 3. 首次生成完整提交

在项目根目录执行：

```powershell
python scripts/build_submission.py --run-all `
  --team-id team001 `
  --model-version v03
```

`--team-id` 填写竞赛平台分配的队伍编号，`--model-version` 由参赛队伍自行填写，
用于区分不同提交版本。凡是生成 ZIP 的命令都必须提供这两个参数；使用
`--no-package` 仅更新场景缓存时可以省略。两个值均限 1–64 个英文字母、数字、点、
下划线或连字符，并且必须以字母或数字开头。

命令默认读取 `configs/defense.yaml`。如使用自定义文件，在所有运行和打包命令中均
增加 `--defense-config configs/my_defense.yaml`。

脚本会依次：

1. 检查 40 个场景的缓存；
2. 复用配置未变化且文件校验通过的场景；
3. 运行缺失、配置已变化或缓存无效的场景；
4. 校验全部 80 个预测 CSV；
5. 生成 `outputs/team001__v03.zip`（名称由参数自动生成）。

正式场景包含完整训练和攻击评测，运行时间可能较长，尤其是图像场景。
建议在 CUDA 环境中执行正式运行。

### 3.1 运行中断后继续

如果运行在中途被终止，再次执行相同命令即可：

```powershell
python scripts/build_submission.py --run-all `
  --team-id team001 `
  --model-version v03
```

已经完整生成并通过校验的场景会显示：

```text
[reuse] dataset_01/clean
```

需要重新运行的场景会显示：

```text
[run] dataset_01/lfba_p05
```

因此，中断后不需要删除缓存，也不需要从第一个场景重新开始。

### 3.2 保留一代或两代完整结果

默认保留两代：

```powershell
python scripts/build_submission.py --run-all --retention 2 `
  --team-id team001 --model-version v03
```

这等价于不显式填写 `--retention`。每个场景最多保存 `current` 和 `previous`
两份完整结果，不会继续累积更早版本。

磁盘空间不足时可以只保留最新一代：

```powershell
python scripts/build_submission.py --run-all --retention 1 `
  --team-id team001 --model-version v03
```

此时每次成功更新后只保留 `current`，旧 `current` 和已有 `previous` 会在新结果
与提交缓存均更新成功后删除。该参数只影响完整本地运行结果，不影响提交缓存和 ZIP
格式。

## 4. 单独更新一个场景

假设选手只优化了 `dataset_01` 的 `lfba_p05` 场景，可以执行：

```powershell
python scripts/build_submission.py `
  --dataset dataset_01 `
  --scenario lfba_p05 `
  --team-id team001 `
  --model-version v03
```

脚本会：

1. 在临时目录重新运行该场景；
2. 校验该场景要求的全部预测文件；
3. 将旧 `current` 轮换为 `previous`；
4. 将新完整结果保存为 `current`；
5. 从新 `current` 整体更新该场景的提交缓存；
6. 保留其余 39 个场景的已有结果和缓存；
7. 使用更新后的完整缓存重新生成队伍版本 ZIP。

不能只替换该场景的某一个 CSV。例如，后门场景的主任务预测和后门预测必须来自
同一次场景运行。

### 4.1 其他场景尚未准备完整

如果缓存中还没有其余 39 个完整场景，直接打包会因场景缺失而失败。此时使用：

```powershell
python scripts/build_submission.py `
  --dataset dataset_01 `
  --scenario lfba_p05 `
  --no-package
```

该命令只更新指定场景的缓存，不尝试生成完整 ZIP。

即使忘记添加 `--no-package`，指定场景只要已经运行并校验成功，其缓存仍然会被
保存；随后只是完整打包阶段因其他场景缺失而报错。

## 5. 仅校验和重新打包

如果 40 个场景缓存已经齐全，只需要重新生成 ZIP：

```powershell
python scripts/build_submission.py --package-only `
  --team-id team001 `
  --model-version v03
```

该命令不会训练模型，也不会重新运行攻击，只会：

1. 校验场景清单；
2. 校验 CSV 格式、ID 和行序；
3. 校验配置与文件哈希；
4. 生成新的 `submission.zip`。

它不会修改 `outputs/scenarios/` 下的 `current` 或 `previous`。

若生成缓存时使用了自定义防御配置，本命令也必须通过 `--defense-config` 指定同一路径；
路径或内容不一致都会阻止复用该缓存。

适用情况包括：

- 原 ZIP 被误删；
- 需要重新确认提交格式；
- 已更新单个缓存，但此前使用了 `--no-package`；
- 需要将结果输出到另一个 ZIP 路径。

## 6. 强制重跑全部场景

以下命令忽略已有有效缓存，重新运行全部 40 个场景：

```powershell
python scripts/build_submission.py --run-all --force --retention 2 `
  --team-id team001 --model-version v03
```

仅在以下情况建议使用：

- 修改了所有场景共享的防御训练或推理流程；
- 修改了所有场景共享的辅助防御模块；
- 修改了全局数据预处理；
- 需要从同一份源码状态重新生成所有结果。

如果只修改了一个场景，不要使用 `--force`，直接按第 4 节单独更新即可。

## 7. 源码变化与缓存复用

完整配置哈希包含独立防御配置。修改防御名称或任意参数会使使用该配置生成的场景
缓存失效，`--run-all` 会自动重跑受影响场景；切换防御配置路径时也必须在后续命令中
保持一致。

源码内容校验值不一致只会产生复现警告，不会自动重跑全部旧场景。这是为了
允许选手只优化某一个场景，然后保留其他场景已经完成的结果。

因此：

- 仅影响某个场景的修改：只重新运行该场景；
- 影响若干场景的修改：逐个重新运行受影响场景；
- 影响全部场景的公共修改：使用 `--run-all --force`。

最终审核时组织方会复跑代码。选手应确保没有继续使用实际已被公共代码修改影响的
旧缓存。

## 8. 自定义缓存和输出路径

可以自定义缓存目录：

```powershell
python scripts/build_submission.py `
  --run-all `
  --results-dir D:\vfl_results\submission_cache `
  --team-id team001 `
  --model-version v03
```

`--results-dir` 只改变提交缓存的位置，不改变完整场景结果的位置。完整结果仍按照
场景 YAML 的 `output.directory` 保存，并在该目录下使用 `current/previous`。

可以自定义 ZIP 路径：

```powershell
python scripts/build_submission.py `
  --package-only `
  --results-dir D:\vfl_results\submission_cache `
  --output D:\vfl_results\team001__v03.zip `
  --team-id team001 `
  --model-version v03
```

相对路径以项目根目录为基准，绝对路径会直接使用。
可以通过 `--output` 更改保存目录，但 ZIP 文件名仍必须与
`<team_id>__<model_version>.zip` 一致。

不要让两个进程同时写入同一个缓存目录。

## 9. CSV 格式

### 9.1 主任务预测

文件名：

```text
main_task_predictions.csv
```

格式：

```csv
ID,Prediction
TEST_000001,0
TEST_000002,1
```

### 9.2 后门预测

文件名：

```text
backdoor_predictions.csv
```

格式：

```csv
ID,Prediction
TEST_000244,1
TEST_000754,0
```

`main_task_predictions.csv` 始终对应完整、未修改的干净测试集，用于计算主任务
Accuracy。LFBA 和 VILLAIN 的公开后门测试视图都是该干净测试集的完整副本，ID、
顺序及公开 A/B 特征必须完全一致；攻击代码在运行时向全部样本注入对应触发器，
`backdoor_predictions.csv` 因而也必须覆盖完整测试集。组织方使用私有标签排除原标签
等于该攻击 `targetLabel` 的样本，只在其余样本上计算 ASR。两种攻击的目标类可以
不同，但公开后门测试文件的 ID 集合不再因此变化。

### 9.3 PMC/AMC 标签推理预测

文件名：

```text
label_inference_predictions.csv
```

格式：

```csv
ID,Prediction
TEST_000001,0
TEST_000002,1
```

该文件的 ID 必须与官方隐藏测试集 ID 及顺序完全一致，不能写入训练集或验证集的
标签恢复结果。

所有 `Prediction` 必须是对应数据集的合法整数类别编号：dataset_01、02、03、04
使用 `0–1`，dataset_05 使用 `0–5`。正式提交不要求概率列。

## 10. 最终 ZIP 目录结构

ZIP 解压后，`manifest.json` 和五个正式数据集目录必须直接位于根目录：

```text
manifest.json
dataset_01/
├── clean/
│   └── main_task_predictions.csv
├── lfba_p05/
│   ├── main_task_predictions.csv
│   └── backdoor_predictions.csv
├── pmc/
│   ├── main_task_predictions.csv
│   └── label_inference_predictions.csv
└── composite/
    ├── main_task_predictions.csv
    ├── backdoor_predictions.csv
    └── label_inference_predictions.csv
...
dataset_05/
```

正确结构如下。打开 ZIP 后，第一层应直接看到 `manifest.json` 和五个数据集目录：

```text
submission.zip
├── manifest.json
├── dataset_01/
├── dataset_02/
├── dataset_03/
├── dataset_04/
└── dataset_05/
```

以下结构是错误的，因为 ZIP 第一层额外增加了 `team_name/`：

```text
submission.zip
└── team_name/
    ├── manifest.json
    ├── dataset_01/
    ├── dataset_02/
    ├── dataset_03/
    ├── dataset_04/
    └── dataset_05/
```

选手不需要手工调整目录。使用 `scripts/build_submission.py` 生成 ZIP 时，会自动采用
第一种正确结构。

## 11. 打包前执行的校验

脚本会检查：

- 五个数据集是否齐全；
- 每个数据集的 8 个场景是否齐全；
- 80 个预测文件是否齐全；
- 文件名和目录名是否正确；
- CSV 列名及列顺序是否正确；
- ID 是否重复；
- ID 集合和行序是否与公开文件完全一致；
- `Prediction` 是否只包含对应数据集的合法类别编号；
- PMC/AMC 标签推断文件是否与隐藏测试集 ID 及顺序完全一致；
- 正式协议配置是否与 `formal_config_lock.json` 一致；
- 防御配置路径和内容哈希是否与场景运行时一致；
- 场景配置哈希是否发生变化；
- CSV 是否仍与场景运行清单中的 SHA-256 一致；
- 最终根清单是否覆盖全部场景和预测文件。

任何正式错误都会阻止打包。代码版本差异属于复现警告，允许打包，但应由选手确认
旧结果是否仍然有效。

新场景运行和缓存更新采用同一轮换事务。任何步骤失败时，脚本会恢复更新前的
`current`、`previous` 和提交缓存。

## 12. 常见问题

### 12.1 `missing_prediction_file`

场景没有生成其要求的某个预测文件。应重新运行该场景，不要手工创建空文件。

### 12.2 `id_order_mismatch`

预测 CSV 的 ID 数量、内容或行序与公开数据不一致。应检查推理 DataLoader 是否
设置了 `shuffle=False`，以及生成 CSV 时是否进行了额外排序。

### 12.3 `invalid_columns`

CSV 列名或列顺序不正确。不要增加概率、标签、索引或其他调试列。

### 12.4 `invalid_predictions`

预测列包含了对应数据集类别范围之外的值。概率需要先转换为最终硬标签。

### 12.5 `stale_scenario_config`

该缓存由旧配置生成。重新运行对应数据集和场景即可。

### 12.6 `formal_config_lock_mismatch`

正式 YAML 中的数据、模型、训练或攻击协议字段已经偏离官方锁。请恢复对应正式
配置；防御参数应写在独立的 `configs/defense.yaml` 或所选防御配置文件中。不要重新
生成锁文件来掩盖协议变化。

### 12.7 `defense_config_path_mismatch`

当前命令选择的防御配置路径与生成缓存时不同。请改回运行该场景时使用的
`--defense-config` 路径，或使用新路径重新运行该场景。

### 12.8 `stale_defense_config`

防御配置内容在场景运行后发生变化。请使用当前配置重新运行对应场景，不要手工修改
`run_manifest.json` 中的哈希。

### 12.9 `prediction_checksum_mismatch`

缓存中的 CSV 在场景运行完成后被修改。重新运行对应场景，不要手工更新
`run_manifest.json`。

### 12.10 `different_code_revision` 或 `different_code_checksum`

部分缓存由另一份源码生成。该警告不会阻止打包。请判断代码修改影响哪些场景，并
重新运行所有受影响的场景。

### 12.11 完整打包提示仍缺少其他场景

当前缓存不完整。可以继续逐场景使用 `--no-package`，或者执行：

```powershell
python scripts/build_submission.py --run-all `
  --team-id team001 `
  --model-version v03
```

脚本会复用已有有效缓存，只补充缺失或配置失效的场景。

## 13. 推荐工作流程

建议按以下顺序开发和提交：

1. 在 `configs/defense.yaml`（或 `configs/` 下的自定义防御配置）中设置参数，并使用
   短周期运行调试防御方法；
2. 确认代码稳定后运行 `--run-all` 建立第一版完整缓存；
3. 修改某个场景后，仅重新运行受影响场景；
4. 公共代码变化较大时，使用 `--run-all --force`；
5. 使用与运行时相同的 `--defense-config` 执行 `--package-only`，做最后一次完整校验；
6. 提交自动生成的 `outputs/<team_id>__<model_version>.zip`；
7. 保留生成该 ZIP 时的代码、配置和缓存，供最终复跑核验。

## 14. 提交前检查清单

提交前请确认：

- [ ] 使用正式场景配置而非 debug 配置；
- [ ] 防御参数只写在独立防御配置中，且运行和打包使用同一路径；
- [ ] 最终命令执行成功且没有打包错误；
- [ ] 已填写平台分配的 `team_id` 和本次提交的 `model_version`；
- [ ] 提交文件名符合 `<team_id>__<model_version>.zip`；
- [ ] ZIP 中包含根目录 `manifest.json`；
- [ ] ZIP 中包含 40 个正式场景；
- [ ] ZIP 中包含 80 个预测 CSV；
- [ ] 没有手工修改生成后的预测 CSV；
- [ ] 没有把 checkpoint、日志或私有数据放入 ZIP；
- [ ] 已确认需要保留两代结果时使用默认 `--retention 2`；
- [ ] 已处理所有与本次代码修改相关的源码版本警告；
- [ ] 已保存提交对应的完整代码版本，能够供组织方复跑。

查看脚本参数：

```powershell
python scripts/build_submission.py --help
```
