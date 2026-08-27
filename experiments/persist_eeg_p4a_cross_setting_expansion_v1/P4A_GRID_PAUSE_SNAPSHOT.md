# P4A Grid Pause Snapshot

- Timestamp (UTC): `2026-08-27T08:43:18.419978+00:00`
- Label: `OPTIONAL_PARTIAL_INVARIANCE_GRID`
- Branch / commit: `codex/persist-eeg-p4a-cross-setting-expansion-v1` / `4cf2493d9c859e569ef8a3209feae632075e7261`
- Mandatory ERM: **45/45 complete**
- Optional non-ERM grid: **205/405 complete**
- Per-setting grid: S4=135/135, S5=70/135, S6=0/135
- Last completed atomic configuration: `S5/fold-2/seed-1/mmd__lambda-0.01`
- Current training process: none
- Scheduler: `Ready`; next trigger `2099-01-01T00:00:00.0000000+08:00`

The in-flight candidate was allowed to finish and atomically write its candidate JSON, checkpoint, and source freeze. The launcher was then stopped before it could begin another non-ERM configuration. Existing runtime artifacts were retained without deletion or renaming.

This is a **computational-scope pause**, not a scientific-outcome decision. The partial invariance grid is excluded from P4B hypothesis, predictor, normalization, threshold, and setting-selection decisions. Invariance outcome deltas and direction-level future utilities remain sealed.

Exact completed/incomplete configuration IDs, checkpoint paths, scheduler state, dirty Git state, and hashes are recorded in `P4A_GRID_PAUSE_SNAPSHOT.json`.
