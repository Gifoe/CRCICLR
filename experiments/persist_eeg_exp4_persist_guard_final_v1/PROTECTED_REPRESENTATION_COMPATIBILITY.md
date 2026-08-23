# Protected representation compatibility

Both Generic adaptation and protected certification operate on the same 64-dimensional embedding emitted by each clean fold/seed StandardEEGNet checkpoint. The population head, target logistic head, interpolation update, P/U/D bank, gradients, and decisions are all expressed in that standardized coordinate system. Mean protected ranks by fold: `{"0": 8.0, "1": 8.0, "2": 8.0, "3": 8.0, "4": 8.0}`. A zero-rank fold is forced to Generic. No V6/V7 protected basis is reused.
