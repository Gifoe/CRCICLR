# Optimizer-aligned source harm audit

The primary certificate is `g_guard^T Delta_A`, where `Delta_A` is obtained by one real AdamW step on a disposable in-memory identity clone. Only source sessions from canonical inner-training subjects are read. The base LiteBN is frozen and eval-mode throughout. Target B has five deterministic, class-balanced, mutually disjoint source blocks; B1..B4 form K=1/2/4 guards and B_future is reserved for actual CE harm.
