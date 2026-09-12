# LiteBN seed-0 outer-to-heldout access amendment

The parent seven-backbone SEARCH freeze prohibits fixed historical-heldout
materialisation until a second, result-bearing SEARCH freeze has been committed
and pushed.  The requested scope here is limited to the documented
`V8_INTERNAL_HOLDOUT` membership (OpenBMI 14 subjects; WBCIC 10 subjects), not
any final/true-outer cohort.

Before any heldout labels are opened, this experiment writes and commits
`OUTER_TO_HELDOUT_LOCK.json` and a SHA-256 sidecar containing the twenty frozen
LiteBN seed-0 selected-checkpoint hashes and all outer-development records.
It attempts a GitHub push first.  If GitHub is unreachable, that failed push is
persisted in the external runtime log and status artifact.  The one-shot
internal-heldout run proceeds under the user's explicit instruction, but the
result is reported as locally committed and push-pending rather than remotely
published.  There is no checkpoint choice, refit, calibration, normalizer fit,
or protocol adaptation based on heldout data.
