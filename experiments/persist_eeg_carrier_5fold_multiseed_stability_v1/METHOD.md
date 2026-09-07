# Method

The implementation reuses the canonical `EEGNet` and `CompactLite` BN definitions from `persist_eeg_carrier_dualdataset_screen_v1`. OpenBMI trains on S1 and evaluates S2; WBCIC trains on S1+S2 and evaluates S3. Normalization is fit on inner-training source sessions only. Both models in every cell use the same fold, cache, manifest, optimizer, epoch budget, and checkpoint rule.

The primary endpoint is mean subject-balanced accuracy difference (LiteBN minus EEGNet). The subject is the bootstrap unit; 10,000 resamples use seed 0. Epoch 60 is retained only as a checkpoint-selection diagnostic.
