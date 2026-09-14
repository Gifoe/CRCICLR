# LiteBN-G2 protocol

This locked experiment inserts one 4Q/2KV, d=64 GQA+RoPE block after historical LiteBN backend block 2 and before Pool8. It has two zero-initialized ReZero scalars. Stem, backend1, and every BatchNorm state are frozen; only the late global block plus declared backend2/readout parameters train. Exposed benchmarks are opened only after each task has five frozen checkpoints.
