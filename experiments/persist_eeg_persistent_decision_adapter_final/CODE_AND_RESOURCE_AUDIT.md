# Code and resource audit

* **ATCNet-CleanRoom:** `specialist_representations/ATCNet`; `features` are the
  frozen representation and `logits` are the frozen population head output.
* **ATCNet-Official / EEGNeX:** no PDA evaluation was opened because the
  primary ATCNet source gate failed.
* **Feature/head boundary:** `pda_core.load_rep` reads only the CleanRoom
  archive; `fit_shared_basis` and `fit_block_adapter` never mutate logits or
  features.
* **OpenBMI/WBCIC source resources:** `model_fit` historical sessions are used
  for the source basis; validation/outcome subjects are subject-disjoint and
  evaluated by historical-session-to-later-session transitions.
* **Biological IDs/session IDs/trial order:** loaded from archive metadata;
  temporal blocks are deterministic contiguous index blocks.
* **Future/sealed resources:** WBCIC S2 utility metrics, outer-10 resources,
  and OpenBMI sealed holdout were not accessed.
* **Existing ERM/PERSIST-RE artifacts:** preserved in their original
  directories; this package does not overwrite them.
