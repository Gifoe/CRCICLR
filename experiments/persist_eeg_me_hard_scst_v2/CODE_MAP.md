# ME-HardSCST V2 — SCST-V1 code map

The requested base object `39e803178e25403a64ff14a46eb3592ebec643ae` is the
tree of commit `f692883a391fcb9df76bb02ecd76934fbaad61f3`; it is not a
commit itself. V2 branches from that unique matching commit.

| Component | Exact file and symbol | Inputs | Outputs/configuration | V2 decision |
| --- | --- | --- | --- | --- |
| Training entrypoint | `experiments/persist_eeg_scst_utility_stage1/code/train_utility.py::main` | committed lock, model name, five folds, three seeds | 75 model/method/fold/seed evaluations per model | Replaced by V2 gated entrypoints |
| Model wrapper | `stage1_common.py::{build_model,model_features,feature_logits,model_logits}` | model id, raw trials/checkpoint | pre-classifier representation and two-class logits | Reused without editing V1 |
| ATCNet-CleanRoom | `persist_eeg_scst_competence_generality_v1/code/specialist_models.py::ATCNet` | `[batch, channels, 1000]` | 32-D feature and linear head | Reused as discovery backbone |
| Dataset loaders | `stage1_common.py::load_data`; upstream P2/P3 `common.py::load_data` | authorized cache only | memmap raw array, subject/session/label metadata | Reused behind stricter V2 sentinel |
| Split loaders | `stage1_common.py::roles`; upstream P2/P3 `common.py::frozen_fold` | dataset, fold 0–4 | disjoint model-fit/validation/outcome subject IDs | Reused; source bank restricted to model-fit |
| Session/subject indexing | `stage1_common.py::{subject_sort,row_indices}` | metadata, subject set, session set | deterministic row indices | Reused |
| Representation extraction | `stage1_common.py::{model_features,infer_model}` | model, raw indices | indices/features/logits/labels/subjects/sessions | Reused; V2 Scope B adds EMA rebuild |
| Subject-class centroids | `train_utility.py::centroids` | combined source representations, session 0 | `(subject,class) -> mean feature` | Replaced: anchor-excluded, count-aware mixed-effects bank |
| Transport vectors | `train_utility.py::frozen_transport` | class-conditional residuals | fixed per-row residual difference and Gaussian norm control | Replaced: `b_t-b_s`, dynamic bank, residual-subspace HardRandom |
| Support projection | `audit_stage1.py::{support_distance,support_radius,solve_alpha}` | query/direction/support | largest V1 alpha within centroid 3NN radius | Only distance/radius semantics reused; V2 alpha is fixed and gated |
| RandomTransport | `train_utility.py::{frozen_transport,train_method}` | isotropic latent vector matched in Euclidean norm | transported CE | Reused only as frozen V1 baseline M2 |
| SCST-NoConsistency | `train_utility.py::train_method` | fixed structured delta | clean CE + 0.5 transported CE | Historical baseline only; V2 M3 is separately implemented |
| Full-SCST loss | `train_utility.py::train_method`; `stage1_common.py::symmetric_kl` | clean and transported logits | CE + transported CE + symmetric KL | Never reused in V2 primary loss |
| Optimizer/scheduler | `train_utility.py::train_method` | all model parameters | AdamW, lr `1e-4`, wd `1e-3`, 15 epochs; no scheduler | Reused as matched budget unless scope masks parameters |
| Fold/seed handling | `train_utility.py::main`; `stage1_common.py::{FOLDS,SEEDS,stable_seed,set_seed}` | folds 0–4, seeds 0–2 | deterministic units | Reused |
| Per-subject evaluation | `train_utility.py::evaluate_future` | outcome subjects, session 2 | subject BA, macro-F1, CE | Reused with explicit session argument |
| Bootstrap | `train_utility.py::aggregate` | subject-averaged paired BA | 10,000 biological-subject draws and fold signs | Reused, extended to all V2 paired controls |
| Aggregation | `train_utility.py::aggregate`; `finalize_stage1.py::main` | unit CSVs | summary/statistics/reports/figures | Replaced by V2-specific aggregation |
| Data-access gates | `stage1_common.py` module contract; upstream cache audits; `SCST_STAGE1_TRAINING_LOCK.json` | authorized development caches and committed lock | fail-closed cache/split checks | Strengthened with V2 locks and S3 sentinel |

## Recovered limitation

`ShuffleSameClass=0.805928` is present only as an immutable value in the V2
request. No tracked or runtime code, per-fold file, or per-subject file on the
server contains that method name. It is retained as a historical report-only
fact and is not represented as regenerated evidence. The five methods actually
implemented by `train_utility.py` are fully paired and reproducible.
