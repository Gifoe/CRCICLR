# GeoSR 快速筛查：STOP

本轮未通过预先锁定的继续条件。WBCIC 有正信号，但 OpenBMI 明显退步，
两个数据集方向相反。因此停止本轮，不启动 folds1–4、其他方法或正式完整运行。

这是 seed0、每个数据集仅 fold0、保留 discovery 最佳 checkpoint、跳过
final-refit 的方向性筛查，不能当作正式五折 seed0 结果，也不能据此断言
GeoSR 在所有情形下无效。

|数据集|SB-ERM BA|GeoSR BA|ΔBA|SB-ERM Macro-F1|GeoSR Macro-F1|ΔMacro-F1|
|---|---:|---:|---:|---:|---:|---:|
|OpenBMI|79.73%|77.82%|−1.91 pp|79.38%|77.39%|−1.99 pp|
|WBCIC|80.28%|82.17%|+1.89 pp|80.17%|82.08%|+1.91 pp|

指标为 fold0 outcome 被试的等权平均，不是三 seed 平均。
OpenBMI 11 人中 3 人提高、3 人持平、5 人下降；最差下降 10 pp。
WBCIC 9 人中 5 人提高、1 人持平、3 人下降；最差下降 1.5 pp。
逐被试 BA/Macro-F1 和差值见 `results/RAPID_TRIAGE_SUBJECT_DELTAS.csv`。

锁定的继续条件要求两个数据集都达到平均 BA 至少 +0.5 pp、Macro-F1
非负提升、至少半数被试 BA 不下降。OpenBMI 不满足，因此不能仅凭 WBCIC
的正结果继续原协议，也不能在看过此次 outcome 后修改本轮判定规则。

Exact terminal: `RAPID_TRIAGE_STOP_INCONCLUSIVE_OR_MIXED`

## 运行速度

已修复每个 batch 重复上传整份源数据的设备索引错误。独立源数据整 epoch
验证：OpenBMI 25.40 → 1.45 秒（17.5 倍），WBCIC 70.30 → 1.90 秒（36.9 倍）；
训练损失和完整参数 SHA-256 均逐位一致。

加速后的实际 GeoSR student 训练：OpenBMI 36 epochs 共 56.03 秒；
WBCIC 38 epochs 共 76.89 秒，分别选择 epoch28 和 epoch30 的 checkpoint。
SB-ERM 从已有进度恢复后分别补跑 12 和 10 epochs，耗时 20.92 和 23.12 秒。
两个数据集总计 10 个 initial-selection teacher 缓存全部复用；本次续跑没有新训 teacher。

2026-09-05 05:16:48–05:20:41 UTC，加速续跑、锁定及 outcome 评估合计约
3 分 53 秒。这不包含此前慢速训练、两数据集性能验证和最终打包核验时间。
父进程训练阶段 221.56 秒，GPU 采样平均利用率 70.57%，峰值显存 14,961 MiB；
该阶段包含加载和换任务，不能与纯 epoch 利用率直接等同。

完整协议入口 `code/run_geosr_accelerated.py` 已采用同一设备索引修复。
本次 STOP 后未执行完整协议，所以不存在实测的完整 seed0 新总耗时。

## 核验与封存

协议 amendment 在读取 outcome 前锁定，SHA-256：
`a60ace47470d77b41a9f15da2baf1df556839fe25658fd1f11e3ad8d57fea6dc`。

两 worker 均完成后才写预评估锁、访问锁并评估。WBCIC outer10 和 OpenBMI
sealed holdout 未打开。服务器独立核验重新计算逐被试表、均值、判定规则，
校验锁链和 4 个 checkpoint 文件哈希：`INDEPENDENT_VALIDATION.json` pass=true。
原 `VALIDATION.json` 亦 pass=true；回归测试 7 passed。

已禁用 `PERSIST_EEG_GEOSR_RAPID_TRIAGE` 计划任务；没有 GeoSR 训练/评估
进程继续运行。既有 cache、teacher、student 和原子进度文件保留在服务器，
未纳入 Git。Git 属性保留锁定 JSON 和执行代码的原始字节，避免 Windows
换行转换破坏已记录的 SHA-256。
