# Structured representation protocol

- CBraMod: retain final `[channel, patch, 200]` tokens and mask.
- LaBraM: retain final patch tokens; concatenate HMC 10-second subwindows in chronological order without pre-head averaging.
- BIOT: retain the transformer output sequence before `.mean(dim=1)`, including checkpoint channel-token indices and temporal positions.
- Atomic cache identity, if Gate F permits full extraction, is dataset × model × outer fold × subject. Tokens are float16 on disk and float32 for statistics. No cache is indexed by head seed.
