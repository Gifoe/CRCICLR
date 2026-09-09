# Seven-backbone four-task three-seed benchmark

This independent experiment evaluates seven *single* EEG backbones under the
existing CRCICLR frozen future-session subject-equal protocol.  It is
deliberately staged: source and adapter fidelity are audited first, TeCh is
selected using inner validation only, the full protocol is committed before
any new SEARCH outer-dev prediction, and fixed historical held-out evaluation
is inaccessible until a second committed freeze.

Runtime checkpoints and signal caches are external to Git.  This directory
contains only code, provenance, locks, compact metric tables, and reports.
