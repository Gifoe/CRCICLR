# PERSIST-SA V6 method record

## Intended mechanism

PERSIST-SA tests the rule: preserve persistent task-useful structure, adapt
session-sensitive task structure from legal target history, and suppress only a
prospectively certified harmful component. No harmful real-EEG component passed
the required persistence, signed-utility, decision-relevance, intervention, and
stability checks, so suppression remained exactly zero.

## Implemented families

The study first reconstructs fold-compatible representations and matched
history baselines. It then evaluates low-capacity affine/bilinear conditional
adapters, supervised prototypes, geometry controls, partial encoder fine-tuning,
and diagonal empirical-Fisher protection against paired uniform and random
controls. A larger MI-specific EEGNet family and a shallow-conv candidate are
selected only on discovery subjects, refit on all non-outcome subjects, and
adapted using the outcome subject's historical labels. History-only selective
heads use internal S1 validation for OpenBMI and bidirectional S1/S2
pseudo-deployment for WBCIC.

For WBCIC, the best legal result is a fixed-logit blend of V5 and the
future-session target-adapted candidate. The blend coefficient is fixed before
outcome evaluation. For OpenBMI, the best result is the discovery-selected
MI-specific backbone with a discovery-selected frozen/head/tail/full adaptation
rule.

## Controls and deterministic repair

Every personalized method is compared with controls allowed the same target
history labels. The initial Fisher comparison was invalid because dropout and
minibatch randomness were not paired across generic/Fisher controls; after
pairing, the apparent PERSIST gain disappeared. Outcome adaptation was then
replayed from saved fold checkpoints with an independent deterministic CPU/CUDA
seed per fold and subject. Only the replayed tables are authoritative.

## Empirical conclusion

The OpenBMI MI-specific generic model reaches 83.20% BA, but the best PERSIST
candidate reaches 82.02%. On WBCIC the strongest generic candidate reaches
82.08%, while the strongest PERSIST candidate reaches 82.02%. Thus the data do
not support an incremental PERSIST-SA performance or safety claim. The code
supports K=1 and K=2 legal histories, but a unified successful method was not
found.
