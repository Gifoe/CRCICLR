# OpenBMI PERSIST-EEG Stage-0：seed-0 结果汇总

## 1. 完成状态

- 数据：OpenBMI **offline/train only**（NEMAR `nm000273`），未下载 GigaDB 原始 MAT。
- 实验：5-fold、单 seed（seed-0）；没有运行多 seed。
- 远端调度器已输出 `SCHEDULER_COMPLETE`。
- 三种表示（handcrafted、EEGNet、ConvTransformer）均完成 5/5 folds。
- 每个表示、每个 fold 均有：verification、event、subject-erasure、rank sensitivity、closed-set、negative-control、variance decomposition 和 `COMPLETE.json`。
- 总体官方 gate 显示 `NOT_EVALUATED`，原因是冻结 evaluator 要求多 seed 的 `COMPLETE.json`；这不是 seed-0 计算失败。

远端结果根目录：

`/root/autodl-tmp/persist_eeg_stage0_repo/outputs/persist_eeg_stage0/results/openbmi/`

## 2. 核心结果（5-fold 平均）

### 2.1 Cross-session subject persistence

AUROC 越高越好；每列是同一 paradigm 的 session-1 → session-2。

| 表示 | ERP | MI | SSVEP |
|---|---:|---:|---:|
| handcrafted | **0.9658** | **0.9854** | **0.9821** |
| EEGNet | 0.5596 | 0.6411 | 0.6412 |
| ConvTransformer | 0.5564 | 0.6293 | 0.6356 |

handcrafted 的跨 session subject persistence 明显最强；两个神经表示接近中等或偏弱。

### 2.2 Cross-paradigm subject persistence

列出的三项分别是 MI–ERP、MI–SSVEP、ERP–SSVEP 的 long-stage raw AUROC。

| 表示 | MI–ERP | MI–SSVEP | ERP–SSVEP |
|---|---:|---:|---:|
| handcrafted | **0.8210** | **0.8789** | **0.8417** |
| EEGNet | 0.6948 | 0.6789 | 0.6632 |
| ConvTransformer | 0.6250 | 0.6428 | 0.5881 |

### 2.3 Event decoding（raw balanced accuracy）

| 表示 | ERP | MI | SSVEP |
|---|---:|---:|---:|
| handcrafted | 0.5762 | 0.6525 | 0.5186 |
| EEGNet | 0.6500 | 0.7394 | **0.9245** |
| ConvTransformer | 0.5298 | 0.6266 | 0.5982 |

EEGNet 的 SSVEP event decoding 最强；handcrafted 的 subject persistence 最强，但 event decoding 并不占优。

### 2.4 Subject-subspace erasure

以下是 `subject_erased − raw` 的 balanced-accuracy 变化；负值表示擦除 subject subspace 后 event decoding 下降。

| 表示 | ERP | MI | SSVEP |
|---|---:|---:|---:|
| handcrafted | +0.0003 | −0.0017 | +0.0040 |
| EEGNet | +0.0000 | **−0.0476** | −0.0104 |
| ConvTransformer | −0.0001 | −0.0212 | −0.0227 |

handcrafted 的擦除影响接近零；神经表示在 MI/SSVEP 上出现可见性能下降，说明其 event decoding 与 subject-related subspace 有一定耦合。

## 3. 诊断结果

### 3.1 Rank sensitivity：subject-verification macro AUROC

| rank | handcrafted | EEGNet | ConvTransformer |
|---:|---:|---:|---:|
| 1 | **0.8386** | 0.6216 | 0.6058 |
| 2 | 0.8164 | 0.5913 | 0.5857 |
| 4 | 0.8080 | 0.5527 | 0.5692 |
| 8 | 0.7607 | 0.5458 | 0.5338 |
| 16 | 0.6336 | 0.5060 | 0.5127 |
| 32 | 0.3689 | 0.5002 | 0.4932 |

subject signal 主要集中在低秩方向；handcrafted 随 rank 增大下降最明显，rank-32 已低于随机水平附近。EEGNet/ConvTransformer 在 rank≥16 时接近随机。

### 3.2 Closed-set diagnostics

下表为 session-1 → session-2 的 balanced accuracy；对角线是同 paradigm，非对角线是 cross-paradigm。最后一列是 6 个 cross-paradigm 方向的平均值。

| 表示 | ERP→ERP | MI→MI | SSVEP→SSVEP | cross-paradigm 平均 |
|---|---:|---:|---:|---:|
| handcrafted | 0.5895 | **0.6946** | **0.7253** | **0.4927** |
| EEGNet | 0.2578 | 0.5261 | 0.5287 | 0.1864 |
| ConvTransformer | 0.2841 | 0.4518 | 0.4882 | 0.2256 |

handcrafted 保留了较强的同-paradigm session transfer，但跨 paradigm closed-set transfer 接近或低于 0.5；这与它的高 cross-paradigm AUROC 并不矛盾，因为两者是不同的诊断任务和分类设置。

### 3.3 Negative controls

| 诊断 | handcrafted | EEGNet | ConvTransformer |
|---|---:|---:|---:|
| global-session-only | 0.4860 | 0.5148 | 0.5015 |
| paradigm-only | **0.7894** | 0.6697 | 0.5978 |
| trial-label permutation：MI | 0.5036 | 0.5024 | 0.4907 |
| trial-label permutation：ERP | 0.5003 | 0.4993 | 0.5000 |
| trial-label permutation：SSVEP | 0.2492 | 0.2546 | 0.2568 |

global-session-only 基本接近随机；MI/ERP label permutation 也接近 0.5，SSVEP permutation 接近四分类机会水平 0.25。另一方面，paradigm-only 分数明显高于随机，尤其 handcrafted，说明数据中存在较强的 paradigm-specific 可分性，不能把所有 persistence 分数都解释成纯 subject identity。

### 3.4 Hierarchical variance decomposition

方差占比（long / medium / fast）：

| 表示 | long | medium | fast |
|---|---:|---:|---:|
| handcrafted | 15.36% | 4.27% | **80.37%** |
| EEGNet | 20.27% | 2.48% | **77.26%** |
| ConvTransformer | 16.02% | 3.17% | **80.81%** |

三种表示都由 fast（记录内/试次层面）方差主导；long（subject/persistence 相关）方差仅约 15–20%。

## 4. 结论与限制

1. **最强 subject persistence：handcrafted。** 它在 cross-session、cross-paradigm 和低秩 rank sensitivity 上都明显优于两个神经表示。
2. **最强 SSVEP event decoding：EEGNet（0.9245）。** 但 EEGNet 的 MI/SSVEP subject-erasure 下降，表明 event decoding 与 subject subspace 存在耦合。
3. **ConvTransformer 当前结果最弱。** Cross-session 和 cross-paradigm persistence 仅中等，rank 增大后接近随机。
4. **paradigm confound 明显。** paradigm-only control 高于随机，尤其 handcrafted（0.7894）；因此不能把 handcrafted 的高 AUROC 直接等同于完全去除 paradigm 信息后的 subject persistence。
5. **数据限制：** 当前只有 OpenBMI offline/train；没有 online/test 部分，也没有补下载 GigaDB MAT。所有数值是单 seed、5-fold 结果，不应当作为多 seed 稳健性结论。

## 5. 机器可读结果

远端每个表示的 fold 目录下保存了完整 CSV 和 `COMPLETE.json`。核心摘要位于：

`/root/autodl-tmp/persist_eeg_stage0_repo/delivery/persist_eeg_stage0/OPENBMI_STAGE0_SEED0_CORE_SUMMARY.json`

