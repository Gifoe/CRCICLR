# PERSIST-EEG P3 Closure and P4 Adaptive Pilot

This directory contains the code and result artifacts from the seed-0 P3
closure and P4 adaptive-pilot evaluation.

## Scope

- `p3_closure_v2.py` closes and freezes the P3 trajectory analysis.
- `p4_persist_pb.py` implements the four permitted P4 development variants.
- `p4_finalize_development.py`, `p4_update_adaptation_log.py`, and
  `p4_verify_final.py` finalize and verify the protocol outputs.
- `outputs/persist_eeg_p3closure_p4/` contains the reports, tables, development
  curves, integrity checks, and final decision.

No raw EEG data, embedding cache, model checkpoint, Hugging Face credential, or
server credential is included.

## Outcome

- P3: `P3_CLOSED_AND_FROZEN`
- P4: `P4_MAIN_METHOD_NOT_SUPPORTED`
- Method lock: refused
- Outer-test access: not started

See `outputs/persist_eeg_p3closure_p4/P3P4_FINAL_DECISION.md` and
`outputs/persist_eeg_p3closure_p4/p4/P4_FINAL_REPORT.md` for the evidence and
decision rationale.
