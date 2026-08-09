# ProbeCert-V3 implementation plan

Baseline: `v2-joint-risk-benefit@f576ea249603f112c71f3825a7eb1707e4008591`.

## Reuse without modification

- Frozen token embeddings under `data/embeddings_tokens_v2` and the qualified V2 source-head checkpoints.
- Subject pools and outer subject isolation from `data/splits_v2_dev`; V3 copies these into a separately hashed `data/splits_v3_dev` namespace.
- The original V2 `future_indices`, CBraMod token loader, prediction-set implementation, fixed 20-threshold plus sentinel grid, and subject-level metrics.
- The formal action names `no_tta`, `official_t3a`, and `robust_residual_adapter`.
- V1/V2 source, decisions, outputs, and delivery files remain read-only.

## Replace in the V3 path

- Replace the single context with chronological, disjoint Adapt/Probe contexts while preserving Future exactly.
- Replace high-dimensional risk/benefit regressors and actionwise simultaneous calibration with a deterministic low-capacity Probe policy and a scalar policy-level joint critical index.
- Replace the V2 adapter collapse objective and hidden-only Gaussian consistency with a V3-only corrected adapter using nonnegative collapse loss and deterministic EEG nuisance augmentations.
- Replace fixed action parameters with subject-grouped, meta-only successive-halving search.
- Replace seed-as-independent aggregation with subject-cluster bootstrap after averaging repeated seeds.

## New code and configuration

- `src/hsc_tta/v3/access.py`: four-phase runtime state machine and hash-verified future gate.
- `src/hsc_tta/v3/episodes.py`, `pseudo_episodes.py`: A/P/V protocols and grouped pseudo-episodes.
- `src/hsc_tta/v3/actions.py`, `augmentations.py`, `action_search.py`: frozen action lifecycle, corrected adapter, augmentation audit, and bounded search.
- `src/hsc_tta/v3/oracle_headroom.py`: set-efficiency Safe-Oracle gate with subject bootstrap.
- `src/hsc_tta/v3/probe_metrics.py`, `cross_context_surfaces.py`: four Probe diagnostics plus three hard gates.
- `src/hsc_tta/v3/probe_policy.py`: deterministic threshold policy and meta-only threshold selection.
- `src/hsc_tta/v3/policy_certificate.py`: exact finite-sample policy-level joint critical-index calibration.
- `src/hsc_tta/v3/nested_policy_evaluation.py`, `statistics.py`: leakage-safe outer evaluation and clustered inference.
- `src/hsc_tta/v3/baselines.py`, `ablations.py`, `simulation.py`, `reporting.py`, `artifacts.py`.
- CLI stages under `scripts/v3/`; immutable configs under `configs/v3/`; tests under `tests/v3/`.

## Execution gates

1. Implement and test access, episode, action-freeze, Probe-metric, policy, certificate, and statistical invariants.
2. Build V3 episodes and copy/validate subject splits.
3. Audit source manifests, run bounded nested action search, then compute Oracle headroom.
4. If the preregistered Oracle gate fails, stop selector development and emit `delivery/v3_probecert/ORACLE_HEADROOM_NO_GO.md`.
5. If it passes, build cross-context surfaces; fit Probe policies only on meta subjects; calibrate one score per calibration subject; freeze and hash every outer decision before opening Future.
6. Run equal-budget baselines, ablations, CAP external-site replication, 5000-repetition simulations, 2000-repetition subject-cluster bootstrap, reports, coverage, provenance, and artifact hashing.

No old final outcome or CAP label is permitted to influence V3 method selection. Large subject-level parquet files, checkpoints, token caches, and EEG data remain server-only.
