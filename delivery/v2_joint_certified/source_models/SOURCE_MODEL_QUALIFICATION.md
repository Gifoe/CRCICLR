# Source model qualification

| dataset   | architecture             |   macro_f1_mean |   macro_f1_std |   balanced_accuracy_mean |   parameters |   classes_predicted_min |
|:----------|:-------------------------|----------------:|---------------:|-------------------------:|-------------:|------------------------:|
| eegmmidb  | channel_temporal_head    |        0.262053 |      0.0156197 |                 0.274736 |       387284 |                       4 |
| eegmmidb  | official_downstream_head |        0.42151  |      0.0231785 |                 0.425722 |     10241004 |                       4 |
| eegmmidb  | old_mean_mlp             |        0.186006 |      0.0175522 |                 0.264227 |        52484 |                       1 |
| hmc       | old_mean_mlp             |        0.424595 |      0.0497831 |                 0.50235  |        52741 |                       4 |
| hmc       | temporal_attention_head  |        0.506423 |      0.0729352 |                 0.549311 |       213941 |                       4 |

EEGMMIDB qualification: **PASS**.
Official audit: CBraMod commit `0ff6be9`. The official PhysioNet 64-channel `all_patch_reps_twolayer` classifier is implemented as `official_downstream_head`. The official ISRUC head assumes six channels and 30-epoch sequences and is incompatible with the frozen formal single-C4 episode, so it is not mislabeled as an official HMC implementation. No partial fine-tuning was selected; all compared heads use the frozen backbone.
