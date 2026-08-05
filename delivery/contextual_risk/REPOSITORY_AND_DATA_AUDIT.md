# Repository and data audit

## Runtime and repository

- Project root: `/root/autodl-tmp/hsc_tta_eeg`
- Repository: `/root/autodl-tmp/hsc_tta_eeg/repo`
- Starting commit: `4469a5b3132a51386e8c0b364446147225b07940`
- Branch: `v4-contextual-risk-select-and-run`
- Python: 3.11.15; PyTorch: 2.7.0+cu128; CUDA: 12.8
- GPU: NVIDIA GeForce RTX 5090

Only CBraMod frozen embeddings and the ten frozen V2 source heads were used. Their exact checkpoint paths and SHA-256 values are in `outputs/v3_probecert/source_models/SOURCE_MODEL_MANIFEST.json`. No TTA action or adapter was executed.

## Data inventory

| Dataset | Subjects | Classes | Role in this run |
|---|---:|---:|---|
| HMC | 151 | 5 | internal screening source |
| EEGMMIDB | 109 | 4 | internal screening source |
| CAP | 99 | 5 | reserved external replication; not opened |

Input episodes are under `/root/autodl-tmp/hsc_tta_eeg/data/episodes_v3`. New immutable episodes are under `/root/autodl-tmp/hsc_tta_eeg/data/episodes_contextual_risk`; U is the sorted union of V3 adapt/probe indices and V is the unchanged V3 Future. Source logits/probabilities were cached offline under `outputs/contextual_risk/source_cache` and shared by A/B. Formal-calibration and internal-final cache members were not loaded by the surface/screening stage; only their unlabeled context features were constructed.

Master cohorts are 90/30/31 for HMC and 65/21/23 for EEGMMIDB (development/formal calibration/internal final). All five source-head seeds share the same master cohorts and hash folds.

## Historical access and isolation status

HMC, EEGMMIDB, and CAP were already accessed by V1--V3 experiments. Consequently, no result here is a historically untouched confirmation. The new formal-calibration and internal-final cohorts were reserved for the current method only, but the STOP decision means neither was opened.

There is a second, material limitation: the existing frozen source heads use different task-head subject splits per seed. On average, source-head-training overlap is 42.44% of HMC development rows and 40.00% of EEGMMIDB development rows; overlap for every seed/cohort is recorded in `outputs/contextual_risk/provenance/SOURCE_HEAD_SUBJECT_OVERLAP.csv`. Thus the screening evidence must not be described as uniformly unseen-subject source-head evaluation. Correcting this would require new commonly split source heads, which is outside the frozen-source contract.

CAP would use the seed-matched HMC source model, as in V3. CAP did not participate in branch selection, feature selection, hyperparameter selection, or this stopped run.

## Reuse and rewrites

Reused: frozen token embeddings, frozen source-head checkpoints, V3 chronological episodes, and source-model factory. Rewritten: contextual cohorts, episodes, access controller, TPS/APS/RAPS families, unified quantiles, 12-D context signature, source cache, cross-fitted screening, statistical gates, run state, and delivery.

No required input file was missing or corrupted. Both datasets met the fixed 60/20/20 minimum counts, so no nested-protocol deviation was triggered.
