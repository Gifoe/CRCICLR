# Model architecture

Each pathway is EEGNet-derived: temporal convolution, depthwise spatial
convolution across all 62 channels, average pooling, separable temporal
convolution, compact embedding, and a two-class linear head.  Pathways share
no target-trainable parameters or normalization buffers.

The total logits are `logits_P + logits_A`.  The dual width is selected solely
by parameter-count distance to the fold's legal B1 EEGNet from the two frozen
candidates in `PROTOCOL_FROZEN.json`, subject to the 1.25x cap.  FLOPs/MACs and
exact parameters are measured by the runner and written to
`results/efficiency.csv`.
