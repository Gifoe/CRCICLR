# PRD on a fixed carrier

For each support and query subject, PRD forms a normalized binary class-direction
from the canonical EEGNet 64-dimensional embedding. Query directions are compared
with the normalized mean support direction. Gradients flow through both support
and query embeddings; no centroid, prototype, absolute-location or auxiliary
objective is used. The comparison uses identical initial weights and CE batches.
