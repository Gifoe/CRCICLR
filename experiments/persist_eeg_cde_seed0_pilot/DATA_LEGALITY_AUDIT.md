# DATA LEGALITY AUDIT

This file is predeclared before the seed-0 pilot.  The runner replaces it with
the observed row and role counts after preflight.  The experiment is limited
to the canonical OpenBMI 54-subject scope and the WBCIC 41-subject authorized
development scope.  The WBCIC scope lock must assert
`outer_subject_ids_present=false`; the sealed outer ten are not enumerated or
opened.  OpenBMI sealed/internal holdout data are not opened.  No outcome
history is used for fitting, normalization, epoch choice, adapter training or
fusion selection.
