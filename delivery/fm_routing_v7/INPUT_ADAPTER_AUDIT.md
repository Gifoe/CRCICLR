# Input adapter audit

```json
{
  "cbramod": {
    "source": "existing audited token cache",
    "pooling": "mean over valid channel-patch tokens",
    "embedding_dim": 200
  },
  "labram": {
    "sampling_rate": 200,
    "unit": "microvolts",
    "hmc_subwindows_seconds": 10,
    "hmc_pooling": "mean of three deterministic subwindow embeddings",
    "eegmmidb_window_seconds": 3.2,
    "channel_mapping": "normalized 10-20 names",
    "embedding_dim": 200
  },
  "biot": {
    "sampling_rate": 200,
    "unit": "microvolts",
    "hmc_channels": [
      "C3-M2",
      "C4-M1"
    ],
    "hmc_channel_tokens": [
      10,
      14
    ],
    "eegmmidb_montages": [
      "FP1-F7",
      "F7-T7",
      "T7-P7",
      "P7-O1",
      "FP2-F8",
      "F8-T8",
      "T8-P8",
      "P8-O2",
      "FP1-F3",
      "F3-C3",
      "C3-P3",
      "P3-O1",
      "FP2-F4",
      "F4-C4",
      "C4-P4",
      "P4-O2"
    ],
    "n_fft": 200,
    "hop_length": 100,
    "pooling": "official sequence mean",
    "embedding_dim": 256
  }
}
```

All rules were frozen before task-performance computation. No evaluation label is used by an adapter.
