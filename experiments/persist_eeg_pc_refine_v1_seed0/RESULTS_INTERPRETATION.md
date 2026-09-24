# PC-Refine EEGNet V1 结果解读

状态：**EXPLORATORY_POST_HELDOUT**。20 个正式 heldout 预测单元均在
`FINAL_EVAL_LOCK.json` 下完成，但锁定的聚合程序在跨 fold 平均不同维度
的 Protected 能量向量时失败。后续恢复程序只修改该事后机制指标的汇总方式；
BA、F1、NLL、五折概率平均和受试者配对 bootstrap 公式未改。
按原协议，恢复后的结果不能称为未经修改的 confirmatory V1。

主指标为未来 session 的 subject-equal BA。OpenBMI 为 14 名正式 heldout
受试者；WBCIC 为 10 名 true-outer 受试者。seed=0，每任务五折概率平均。
置信区间基于 20,000 次 biological-subject 配对 bootstrap。

| 任务 | EEGNet | Protected-PC | ΔBA (95% CI) | Random-PC ΔBA (95% CI) |
| --- | ---: | ---: | ---: | ---: |
| OpenBMI MI | 0.7114 | 0.7136 | +0.0021 [−0.0021, +0.0064] | +0.0021 [−0.0021, +0.0064] |
| OpenBMI ERP | 0.8645 | 0.8624 | −0.0021 [−0.0045, −0.0003] | 未触发 |
| OpenBMI SSVEP | 0.9200 | 0.9214 | +0.0014 [+0.0000, +0.0036] | +0.0000 [−0.0021, +0.0021] |
| WBCIC MI | 0.7905 | 0.8115 | +0.0210 [+0.0075, +0.0360] | +0.0225 [+0.0120, +0.0350] |

Random-PC 仅因相应任务的 Protected-PC 点估计为正而启动，属于预先约定的
条件式探索分析。WBCIC 上 Random-PC 的增益略高于 Protected-PC；
Random-PC 减 Protected-PC 为 +0.0015，95% CI [−0.0080, +0.0085]。
MI 两者持平，SSVEP 两者差异的区间也跨零。

**结论：** WBCIC 上的 adapter 相对匹配 EEGNet 有明确的 BA 提升，但当前
实验无法将其归因于 Protected 坐标或 P×C 机制。MI 和 SSVEP 的正差值很小，
ERP 的主 BA 下降。跨任务的机制特异性性能提升未得到支持。
