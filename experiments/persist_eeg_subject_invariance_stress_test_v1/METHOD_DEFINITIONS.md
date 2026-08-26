# Frozen method definitions

ERM minimizes ordinary binary task cross-entropy. DANN adds a fixed `Linear(64,128)-ReLU-Dropout(0.2)-Linear(128,K)` source-subject classifier after unit-coefficient gradient reversal; only inner-train source subject IDs enter that objective. CORAL averages the standard covariance discrepancy `||C_s-C_t||_F^2/(4d^2)` over source-subject pairs represented in a minibatch. MMD averages biased nonnegative multi-kernel RBF MMD over the same source domains. The RBF bandwidths are fixed from a deterministic initial-embedding source sample using the median-distance rule in the frozen JSON.

The only method-dependent term is the declared invariance objective. Shared backbone initialization, raw rows, train-only normalizer, task loss, optimizer family, learning rate, epoch ceiling, validation rule, and minibatch ordering are matched within backbone/fold/seed.

The fixed strength grid is `{0.01, 0.1, 1.0}` for DANN, CORAL, and MMD. ERM is the unique `lambda=0` reference. No strength may be added after outcome inspection.
