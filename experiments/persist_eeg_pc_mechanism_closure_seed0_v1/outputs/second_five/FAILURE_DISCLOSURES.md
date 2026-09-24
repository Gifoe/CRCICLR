# Failure disclosures for cells 6-10

All failures below were orchestration or fail-closed recovery events. They produced no accepted scientific output; the successful version-forward artifacts are the ones represented in the compact checkpoint.

- Cell 10 replay V3 refused to relaunch because its duplicate guard detected a stale log. No new scientific output was produced. Replay V4 completed and its audit digest is recorded in fold 4 provenance.
- Cell 10 endpoint/native V5 invoked PowerShell's reserved `$args` variable incorrectly and opened an empty interactive Python process. It was stopped without scientific output. V6 completed the intended audit.
- Cell 10 subject/compact V2 waited on an incorrect prerequisite log path. It was stopped without scientific output. The corrected version completed the subject/session controls.
- Cell 10 compact V7 failed closed because it expected a nonexistent `FINAL_P_BACKTRACE_V1.json`. No compact result was accepted. The V9 recovery path used the validated CSV artifact and completed successfully.
- An empty cell 10 V1 subject/compact wrapper was created by a failed deployment command. It was never used to generate scientific output; version-forward wrappers were used instead.

Earlier failed attempts and their logs remain preserved in the runtime environment and are intentionally excluded from this compact Git checkpoint.
