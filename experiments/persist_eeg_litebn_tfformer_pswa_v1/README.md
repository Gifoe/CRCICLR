# LiteBN / TFFormer PSWA v1

This experiment measures whether the Protected canonical coordinates selected
by the corrected Appendix-L PEEH analysis are sufficient for stronger
worst-session decoding than equal-rank random canonical coordinates.

The primary result uses seed 0 and five folds for LiteBN and TFFormer on
OpenBMI MI, ERP, SSVEP, and WBCIC MI. OpenBMI uses the established 14-subject
internal-heldout diagnostic cohort on S1/S2. WBCIC uses the established
10-subject true-outer diagnostic cohort on S0/S1/S2.

No model training, checkpoint selection, Protected selection, persistence
permutation, target adaptation, or hyperparameter search is performed. The
corrected PEEH canonical arrays omitted from disk are recovered deterministically
and checked by exact Protected-erasure BA reproduction. Missing evaluation
session representations are extracted once in eval mode, then only compact
canonical arrays are retained in the external runtime cache.

Run the primary analysis with:

```bash
python code/run_pswa.py --device cuda --seeds 0
```

Scientific outputs are in `outputs/pswa_v1/`. The optional seeds 1/2 stage is
separate and does not block the primary seed0 report.
