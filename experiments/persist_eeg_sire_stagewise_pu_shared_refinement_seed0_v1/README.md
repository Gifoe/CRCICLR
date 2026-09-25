# Stagewise Protected utility shared refinement (seed 0)

Five-fold OpenBMI MI pilot on the canonical SIRE-EEG / CompactLite checkpoint. The two trainable arms use the same four shared convolution weights, initialization, eval-mode model, AdamW settings, paired-session batches, and fixed epoch-20 endpoint.

- `SHARED_CE_CONTINUATION`: CE on full native logits.
- `STAGEWISE_PU_SHARED`: Shared1 cross-session P growth, Shared1 variance floor, Shared2 P-only hybrid CE through the frozen suffix, complement preservation, and a reference-correct prediction anchor.

The fixed Shared1 and Shared2 PathFit geometry is loaded from the completed layerwise audit runtime and checked against its committed SHA-256 hashes. All fitting and training use inner-train only. Discovery is read only after both arms finish epoch 20. Outer-dev and final-heldout EEG are never read.

On the server, with the prior layerwise runtime and canonical source assets available:

```bash
python experiments/persist_eeg_sire_stagewise_pu_shared_refinement_seed0_v1/code/run.py --fold 0
python experiments/persist_eeg_sire_stagewise_pu_shared_refinement_seed0_v1/code/run.py --fold 1
python experiments/persist_eeg_sire_stagewise_pu_shared_refinement_seed0_v1/code/run.py --fold 2
python experiments/persist_eeg_sire_stagewise_pu_shared_refinement_seed0_v1/code/run.py --fold 3
python experiments/persist_eeg_sire_stagewise_pu_shared_refinement_seed0_v1/code/run.py --fold 4
python experiments/persist_eeg_sire_stagewise_pu_shared_refinement_seed0_v1/code/run.py --aggregate
```

Per-fold checkpoints and intermediate CSVs live in `SIRE_STAGEWISE_RUNTIME` (default: sibling of repository). Compact aggregate results, protocol, and report are committed under this experiment directory.
