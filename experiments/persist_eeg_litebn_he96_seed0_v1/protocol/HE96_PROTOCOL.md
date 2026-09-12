# HE96 frozen protocol

Exact CompactLite with only backend width changed from 48->64->64 to 48->96->64.
All branches, pooling, final64 channels, embedding, head, normalizer, loss,
optimizer, manifest, AMP, 60 epochs and selection are historical. Historical
weights copy into channels0:64; point1/depth2 extras use canonical Conv defaults;
point2 extra columns are exactly zero. No other architecture/training change.
The widened first-backend dropout is split into a contiguous historical first
64-channel tensor and the new 32 channels. The latter is evaluated in a forked
RNG context so its random draws cannot perturb the historical downstream
dropout stream. This is required for exact train-mode, same-seed function
preservation at initialization; it changes neither the historical 64-channel
calculation nor the candidate's learnable parameters.
Likewise, `point2` is evaluated as the sum of its 64-input historical kernel
and its 32-input new kernel. This is mathematically the same 96-to-64
convolution, but prevents zero-initialized new columns from changing the AMP
reduction path of the historical columns.
OpenBMI MI five folds run first; compare outer before any next task. If HE96 outer
mean delta is nonpositive, stop without starting WBCIC/ERP/SSVEP.
