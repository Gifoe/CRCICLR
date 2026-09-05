# Incremental Relation frozen-feature pilot：停止

本轮只做 seed 0、OpenBMI/WBCIC outer fold 0，并比较四个固定方法：
SUBJECT_BALANCED_ERM、GENERIC_RESIDUAL、GENERIC_PROTOTYPE、
CROSS_SESSION_RELATION。canonical EEGNet checkpoint 和 normalizer 来自既有
source-only fold0 cache；没有重训 backbone。relation 使用 full 64 维 latent，
显式采用 leave-one-source-subject-out 方向构造。结果是单 seed、单 fold 的
方向性筛查，不是正式的多折结论。

## 结果

|数据集|SB-ERM BA|Generic Residual BA|Generic Prototype BA|Cross-session Relation BA|Relation ΔBA|Relation ΔMacro-F1|Relation 相对最强 generic|终端条件|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|OpenBMI|79.727%|79.909%|80.091%|80.000%|+0.273 pp|+0.429 pp|−0.091 pp|不满足|
|WBCIC|80.278%|80.222%|80.444%|80.778%|+0.500 pp|+0.506 pp|+0.333 pp|不满足|

继续条件预先固定为：两个数据集的 relation 相对 SB-ERM 都至少 +0.5 pp
BA、Macro-F1 不下降、至少一半被试 BA 不下降，并且 relation 相对最强
generic control 仍至少领先 +0.5 pp。OpenBMI 的 relation 只有 +0.273 pp，
且 Generic Prototype 达到 +0.364 pp；WBCIC relation 虽为 +0.500 pp，
但相对最强 generic 只多 +0.333 pp。因此两个数据集都没有通过。

逐被试 relation BA 方向：OpenBMI 4 人提高、4 人持平、3 人下降，最差 −6 pp；
WBCIC 5 人提高、2 人持平、2 人下降，最差 −2 pp。完整差值见
`results/INCREMENTAL_RELATION_SUBJECT_DELTAS.csv`。

Exact terminal：`INCREMENTAL_RELATION_STOP_NO_CLEAR_GAIN`

## 判断

该 pilot 没有证明 cross-session relation 产生了足够的增量判别能力。OpenBMI
更接近普通 prototype 的小幅收益，WBCIC 的收益也没有稳定超过 generic control。
按锁定规则停止，不运行 fold 1–4、其他 seed、stacking、selector、GeoSR 或
其它 ablation。不能把本轮结果写成最终论文 claim。

## 工程与合规

训练阶段只读取 source/model-fit 数据；所有 checkpoint、frozen feature 和
relation spec 在 outcome 前写入 pre-outcome lock。outcome 仅在两个数据集
完成、hash lock 和 access lock 写入后读取。WBCIC outer 10 和 OpenBMI sealed
holdout 均未打开。已有 cache 未删除。

frozen-feature extraction 使用 GPU `cuda:0`，OpenBMI 约 2.91 s、WBCIC 约
3.95 s；四个方法为矩阵/阈值评估，无额外 neural training。训练和 outcome
日志未持续采集 GPU utilization，因此报告不虚构该指标；显存峰值沿用同一
GPU 数据视图验证中 OpenBMI 10,353 MiB、WBCIC 16,995 MiB 的独立上界记录。

`INCREMENTAL_RELATION_EXECUTION_LOCK.json` 固定了 cuda:0、顺序单 GPU、
full-latent view 和代码哈希；`validate_incremental_relation.py` 会重算
summary/delta、验证锁链及 runtime artifact hashes。`results/VALIDATION.json`
和 independent validation 均要求 pass=true。
