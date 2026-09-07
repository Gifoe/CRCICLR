# HSC-TTA EEG：历史交接说明（已被替代）

> 本文记录旧的 upper-risk 原型状态，不能作为当前正式方法说明。当前 CPU 方法、验证结果和下一阶段接口请以 `CPU_CRITICAL_INDEX_REPAIR_REPORT.md`、`docs/THEORY_SPEC.md`、`THEORY_IMPLEMENTATION_AUDIT.md` 和 `LEAKAGE_AUDIT_REPORT.md` 为准。

更新时间：2026-08-01（Asia/Shanghai）

## 1. 请 ChatGPT 完成的任务

请基于本文档审查当前 HSC-TTA EEG 项目的真实状态，并给出下一阶段的明确要求和执行步骤。你的输出应当：

1. 先判断当前 CPU-only 阶段是否满足项目要求，并列出仍需修复的 CPU 问题；
2. 重点审查当前统计证书在 `alpha=0.10/0.20` 下是否具有非平凡可行性；
3. 给出后续 GPU 阶段的严格执行顺序、输入输出、数据隔离规则和验收标准；
4. 明确哪些选择必须由研究者决定，哪些可以直接交给 Codex 实现；
5. 最后输出一段可以直接交给 Codex 执行的完整提示词；
6. 不要求或复述任何 SSH、GitHub、Hugging Face 密码或 token。

不要把 synthetic coverage 或 `q=1` 解读为方法已经取得科学成功。当前最重要的问题是证书虽然保守有效，但可能完全没有效用。

## 2. 项目位置与范围

- 服务器项目根目录：`/root/autodl-tmp/hsc_tta_eeg`
- Git 代码仓库：`/root/autodl-tmp/hsc_tta_eeg/repo`
- Conda 环境：`hsc_cpu`，Python 3.11
- 当前阶段严格为 CPU-only；所有正式命令均设置 `CUDA_VISIBLE_DEVICES=""`
- 未下载 CBraMod 或任何 foundation-model checkpoint
- 未调用 CUDA、未提取真实 backbone embedding、未训练真实任务分类头、未在真实 EEG 上运行 TTA
- 原始数据、处理缓存、split、episode、日志和 mock 输出均位于数据盘，不进入 Git

## 3. 当前已经完成的工作

### 3.1 工程和统计方法代码

仓库已实现：

- HMC、CAP、EEGMMIDB 的统一 `DatasetAdapter`；
- HMC/CAP 五分类睡眠标签映射；
- EEGMMIDB runs 4/6/8/10/12/14 的四分类 motor-imagery 事件映射；
- 睡眠与 MI 的滤波、重采样、切窗、HDF5 原子缓存和配置哈希；
- 严格 subject-disjoint split；
- HMC/CAP 时钟连续 90 分钟 context 与 future episode；
- EEGMMIDB context runs 4/6、future runs 8/10/12/14；
- `no_tta`、`t3a`、`entropy_adapter` 的统一动作接口；
- prediction sets；
- empirical-Bernstein-style subject-internal block risk bound；
- `HistGradientBoostingRegressor` meta-risk predictor、GroupKFold、lambda 单调修正和序列化；
- finite-sample simultaneous conformal residual quantile；
- deterministic safe action selection；
- CSR、NAR、HER 等 subject-level metrics 和 subject bootstrap；
- simulations A–E；
- 未来 GPU 阶段三个 parquet schema 及 Pydantic validator；
- 完整 artifact/leakage 验收脚本和报告生成脚本；
- 可恢复的 `scripts/run_full_cpu_phase.sh`。

主要代码目录：

```text
repo/
├── configs/
├── docs/
├── scripts/
├── src/hsc_tta/
├── tests/
├── CPU_PHASE_REPORT.md
├── DATA_AUDIT_REPORT.md
├── PREPROCESSING_REPORT.md
├── SPLIT_AND_EPISODE_REPORT.md
└── NEXT_GPU_PHASE.md
```

### 3.2 原始数据下载与完整性

全部数据来自官方 PhysioNet 公共地址。实际使用官方 S3 payload，并用 aria2 断点续传。下载结束后重新计算本地 SHA256，并对 manifest 做了第二次独立校验。

| 数据集 | 已验证文件 | 缺失/失败 |
| --- | ---: | ---: |
| EEGMMIDB | 654 | 0 |
| HMC | 453 | 0 |
| CAP | 324 | 0 |

数据下载过程中的实际处理：

- AutoDL 学术代理对 PhysioNet/S3 实测更慢，因此下载期间关闭代理；
- aria2 从 `8 files × 8 connections` 调整为 `16 files × 4 connections`；
- EEGMMIDB 使用 109 名被试的 6 个目标 MI runs；
- HMC 每条记录同时下载主 EDF、scoring EDF、scoring TXT；
- CAP 每条记录同时下载 EDF、TXT、EDF.ST；
- 下载完成后 `.aria2` 临时分片数量为 0；
- 原始文件没有删除、修改或加入 Git。

### 3.3 数据审计

| 数据集 | 可读 recordings | eligible subjects | 排除 subjects |
| --- | ---: | ---: | ---: |
| EEGMMIDB | 654 | 109 | 0 |
| HMC | 151 | 151 | 0 |
| CAP | 108 | 103 | 5 |

CAP 排除项为 `cap:n13`、`cap:n14`、`cap:n15`、`cap:nfle25`、`cap:nfle33`，原因均为缺少规定的中央通道。没有为了凑 split 数量放宽通道规则。

生成的正式 manifest：

```text
data/manifests/subjects.parquet
data/manifests/recordings.parquet
data/manifests/channels.parquet
data/manifests/annotations.parquet
data/manifests/exclusions.parquet
data/manifests/eegmmidb_download_manifest.parquet
data/manifests/hmc_download_manifest.parquet
data/manifests/cap_download_manifest.parquet
data/manifests/dataset_audit.json
```

### 3.4 全量 CPU 预处理

| 数据集 | 完整 HDF5 caches | windows/epochs | subject failures |
| --- | ---: | ---: | ---: |
| EEGMMIDB | 109 | 9,837 | 0 |
| HMC | 151 | 137,243 | 0 |
| CAP | 103 | 103,021 | 0 |

合计 363 名 eligible subjects、250,101 个窗口。

缓存至少包含：

```text
signal
label
window_start
window_end
channel_names
channel_mask
sampling_rate
recording_id
run_id
quality_flags
preprocessing_config_hash
```

预处理过程中修复了以下真实问题：

1. CAP TXT 时间字段混用 `HH.MM.SS` 与 `HH:MM:SS`，解析器现支持两者并有回归测试；
2. HMC scoring EDF 被错误当成主 recording 的风险已排除；
3. CAP annotation 使用 EDF recording start 与跨午夜时钟对齐；
4. 睡眠数据在 `get_data()` 前先选择 1–2 条规定中央通道，避免加载所有 PSG 通道；
5. 缓存增加逐窗口 `quality_flags`；
6. resume 检查移到 EDF 读取前，避免已完成缓存被重新滤波；
7. AutoDL 容器 cgroup 内存上限实际为 2 GiB。HMC/CAP 使用单 worker，CAP 后半段采用每名被试一个新 Python 进程以彻底释放内存；
8. 配置哈希改变时生成新临时缓存并原子替换旧处理缓存，绝不复用错误配置。

### 3.5 Subject splits

seeds 0–4 均已生成并验证。

| 数据集 | role counts（每个 seed） |
| --- | --- |
| HMC | task_head_train=70, meta_risk_train=35, conformal_calibration=20, final_test=26 |
| EEGMMIDB | task_head_train=45, meta_risk_train=30, conformal_calibration=15, final_test=19 |
| CAP | target_site_calibration=25, external_final_test=78 |

CAP calibration 根据官方记录名中的病理前缀执行确定性比例分层；容量允许时每个已出现病理类别至少进入 1 名 calibration subject。

验证结果：

- 所有 role subject 集合互斥；
- 每个 split 精确覆盖全部 eligible subjects；
- 同一 subject 不跨 role；
- 重复 seed 生成确定性结果；
- CAP test 不进入 CAP calibration。

### 3.6 Deployment episodes

共生成 15 个 episode parquet：3 datasets × 5 seeds。

| 数据集 | episodes/seed | context 数量范围 | future 数量范围 | episode exclusions |
| --- | ---: | ---: | ---: | ---: |
| EEGMMIDB | 109 | 24–38 | 48–76 | 0 |
| HMC | 151 | 180–180 | 290–1,131 | 0 |
| CAP | 103 | 175–180 | 248–1,540 | 0 |

独立 validator 的结果：

- artifact failures：0
- split leakage：0
- U_s/V_s index overlap：0
- sleep clock-boundary violations：0
- MI run-protocol violations：0
- future minimum violations：0

验证摘要位于：

```text
outputs/cpu_validation/validation_summary.json
```

### 3.7 测试、覆盖率和 mock GPU 接口

- `pytest -q`：25 passed，1 个预期的 conservative-quantile warning；
- `pytest --cov=src/hsc_tta`：71% line coverage；
- Git diff check：通过；
- 仓库凭据扫描：0 命中；
- 仓库内没有大于 5 MiB 的文件；
- 当前数据盘剩余约 284 GiB。

生成并逐行验证的 mock GPU 接口：

| 文件 | rows |
| --- | ---: |
| `mock_features/subject_context_features.parquet` | 30 |
| `mock_features/subject_action_surface.parquet` | 1,800 |
| `mock_features/subject_decisions.parquet` | 30 |

这些文件仅用于冻结 schema，不是真实 CBraMod 结果。

## 4. 当前最重要的科学问题

最终 synthetic summary：

```text
n_subjects=120
n_calibration=30
q=1.0
surface_coverage=1.0
certified_subject_rate=0.0
```

这意味着：

- simultaneous certificate 在模拟中覆盖，但主要因为上界饱和；
- 没有任何非平凡 action-lambda 通过 `alpha=0.20`；
- 当前结果不能支持“方法具有有效 safety-utility trade-off”的论文主张；
- 按当前 empirical-Bernstein-style bound，`3*log(3/eta)/B` 在常见 block 数下已经很大；
- 对睡眠 future 使用 10 分钟 blocks 时，典型整夜记录的 B 很难让该加法项低于 0.20；
- 单纯增加同一 subject 的窗口不能解决独立 block/subject 数不足问题。

ChatGPT 需要重点判断：

1. 当前 bound 公式是否与计划中的理论证明一致；
2. 是否应采用更紧且有严格依据的 bounded-risk concentration bound；
3. `eta_within`、block 长度、两层误差预算和 conformal `delta` 应如何联合设计；
4. 是否应将 within-subject bound 与 across-subject conformal residual 分工重新定义；
5. 在不使用测试被试调参的前提下，如何预先设定能产生非平凡 CSR 的方案；
6. 如果理论上无法在现有数据长度下达到 `alpha=0.10/0.20`，论文目标是否必须修改。

## 5. 已生成的报告

代码仓库中已有：

```text
INITIAL_INSPECTION.md
DATA_AUDIT_REPORT.md
SMOKE_DOWNLOAD_REPORT.md
SMOKE_TEST_REPORT.md
PREPROCESSING_REPORT.md
SPLIT_AND_EPISODE_REPORT.md
CPU_PHASE_REPORT.md
NEXT_GPU_PHASE.md
docs/DATASETS.md
docs/DATA_SCHEMA.md
docs/CPU_PIPELINE.md
docs/METHOD_SPEC.md
docs/GPU_INTERFACE_SPEC.md
```

ChatGPT 在提出下一步前应优先阅读：

1. `CPU_PHASE_REPORT.md`
2. `docs/METHOD_SPEC.md`
3. `docs/GPU_INTERFACE_SPEC.md`
4. `SPLIT_AND_EPISODE_REPORT.md`
5. `NEXT_GPU_PHASE.md`

## 6. Git 状态

- 当前分支：`main`
- CPU 完整阶段提交：`e54ff07 complete CPU data pipeline and validation`
- 原始数据和所有大型 artifact 均被 `.gitignore` 排除
- GitHub 远端：`Gifoe/CRCICLR`
- 尚未推送：服务器没有配置 GitHub HTTPS PAT 或 SSH 私钥，`git push --dry-run` 因无法读取 GitHub username 而失败

GitHub 推送是认证问题，不是代码或 CPU pipeline 问题。不要使用 GitHub 账户密码；应使用具有仓库写权限的 fine-grained PAT 或 SSH key。

## 7. 明确未完成的内容

以下内容是后续 GPU/科学实验阶段，当前没有执行：

- CBraMod checkpoint 下载和校验；
- 冻结 backbone 的真实 EEG embedding 提取；
- HMC/EEGMMIDB task-head 训练；
- HMC/EEGMMIDB meta-risk predictor 的真实训练；
- HMC→CAP 外部 shift 的真实 action surface；
- 在真实 U_s 上运行 T3A 或 entropy adapter；
- 真实 calibration residual quantile；
- 真实 final-test 指标、置信区间和论文结论；
- 统计界的理论修订和证明；
- GitHub push。

## 8. 希望 ChatGPT 输出的下一阶段方案格式

请按以下结构回复：

1. **CPU 阶段验收结论**：通过/有条件通过/不通过，并给证据；
2. **必须先解决的理论问题**：尤其是 `q=1、CSR=0`；
3. **是否允许进入 GPU 阶段**：给出明确的 go/no-go 条件；
4. **GPU 阶段逐步执行清单**：每一步输入、输出、角色数据、允许访问的字段、禁止项；
5. **资源预算**：显存、系统内存、数据盘、预计时长；
6. **实验矩阵**：datasets、seeds、actions、lambda、alpha、消融和 baselines；
7. **验收标准**：测试、泄漏、schema、统计有效性、CSR 和风险指标；
8. **失败/回退规则**：哪些情况必须停止，哪些可恢复；
9. **给 Codex 的完整下一阶段提示词**：可直接复制执行，不能隐含使用测试集调参。
