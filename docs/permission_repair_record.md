# Workspace file-permission repair record

记录时间：本会话开始时。

## 现象

沙箱模式下工作区只能写根目录，`data/`、`outputs/`、`defense/`、`core/` 等全部子目录写入被
拒绝，报 `SetNamedSecurityInfoW failed (Win32 5)` / `Permission denied` / `[sandbox: file
access denied under workspace-write mode]`。

## 根本原因

沙箱 shell 运行在 Windows **低完整性级别**（`S-1-16-4096`，Low Mandatory Level）。工作区根
目录被设置了一条 `Mandatory Label\Low Mandatory Level:(OI)(CI)(NW)` 强制完整性标签，因此
根目录及其下对象可被低完整性进程写入；而各子目录缺少该标签，默认取父级 `D:\` 的中等完整性
级别，受到 no-write-up 规则限制，低完整性进程一律写入被拒。这与访问权限项无关，属于强制完整
性控制（MAC）。

## 已做的改动

1. `D:\MakeCode\VisualStudio\WorkPlace\VFL\_dsh_acl_recovery\` 下保存了两份完整备份与回滚脚本：
   - `acl-backup-5dff35429bc240bc8fe1883a91b0e931.json`（对象：工作区根目录）
   - `acl-backup-fe1a964ccd864518a4f9b7b3eba57841.json`（对象：`outputs`）
2. 工作区根目录与 `outputs` 目录：补入当前用户的完全控制允许项（由上述技能脚本完成，已备份）。
3. 其余子目录（`attacks`、`configs`、`core`、`data`、`datasets`、`defense`、`docs`、
   `examples`、`metrics`、`models`、`outputs`、`scripts`）：
   - 补入当前用户的完全控制允许项；
   - 设置与根目录一致的 `Low` 强制完整性标签。

**未改动**：任何文件内容、所有者、以及已有的拒绝项。

## 回滚

技能脚本生成的两份备份可用其自带脚本回滚，例如：

```powershell
powershell -NoProfile -File 'D:\MakeCode\VisualStudio\WorkPlace\VFL\_dsh_acl_recovery\acl-backup-5dff35429bc240bc8fe1883a91b0e931.json.ps1' -Path 'D:\MakeCode\VisualStudio\WorkPlace\VFL\VFL-Security-Challenge-v1.0.7' -AllowRoot 'D:\MakeCode\VisualStudio\WorkPlace\VFL\VFL-Security-Challenge-v1.0.7' -Restore 'D:\MakeCode\VisualStudio\WorkPlace\VFL\_dsh_acl_recovery\acl-backup-5dff35429bc240bc8fe1883a91b0e931.json'
```

其余子目录的标签与授权项可逐目录撤销：

```powershell
$root = 'D:\MakeCode\VisualStudio\WorkPlace\VFL\VFL-Security-Challenge-v1.0.7'
$me = "$env:USERDOMAIN\$env:USERNAME"
foreach ($d in @('attacks','configs','core','data','datasets','defense','docs','examples','metrics','models','outputs','scripts')) {
  $p = Join-Path $root $d
  icacls $p /remove:g "${me}" | Out-Null
  icacls $p /setintegritylevel '(OI)(CI)MEDIUM' | Out-Null
}
```

## 验证

修复后沙箱内对 11 个代表性路径逐个做写入并删除探针测试，全部 `WRITE_OK`。

## 说明

技能自带的诊断/修复脚本只处理访问权限项与第三方应用包权限项，不处理强制完整性标签，因此单靠
它无法解决本场景；标签部分需要额外处理。
