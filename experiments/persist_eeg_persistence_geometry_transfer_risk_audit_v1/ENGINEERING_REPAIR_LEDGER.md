# Engineering repair ledger

| ID | Problem / cause | Fix | Scientific definition changed? |
|---|---|---|---|
| R001 | Target branch had no implementation for this audit. | Added two-phase A-only EEGNet runner with explicit role and lock checks. | No |
| R002 | Server is CPU-only and older scripts assume GPU or load complete-role metadata. | CPU-safe batching; A/B predicate reads; mmap source arrays; no outcome-role index. | No |
| R003 | Historical persistence spectra use incompatible fold matrices. | Rebuild spectrum per current canonical fold from A-only embeddings. | No |
| R004 | JSON cannot losslessly carry runtime ndarray operators. | Persist runtime spectrum in `spec.npz` and keep JSON provenance separate. | No |
| R005 | Matched null draw provenance must survive the phase boundary. | Save deterministic draw/subject novelty rows under ignored runtime and recompute draw-level null rho after lock. | No |

All changes above are implementation/provenance repairs; none were selected from
or conditioned on transfer outcomes.

- problem: OpenBMI SafeData.batch expected internal signal/cache columns that metadata loader did not expose
  cause: loader retained manifest names signal_cache_path/cache_index while batch uses _signal_path/_cache_index
  fix: add explicit internal aliases with validated dtypes before returning A/B metadata
  scientific_definition_changed: false
