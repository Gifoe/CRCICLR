# LiteBN-G protocol

One 4Q/2KV, d=48 GQA+RoPE block is inserted after the 48-channel stem concat and before historical backend block 1. It uses two zero-initialized ReZero scalars. Early stem and all BatchNorm states are permanently frozen; only the global block and declared low-LR backend/readout parameters train. Exposed benchmarks are accessed only after each task has five frozen checkpoints.
