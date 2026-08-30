# Bank staleness audit

Scope A used fixed encoder coordinates. Scope B used EMA decay 0.99 and rebuilt the detached bank exactly once at each epoch start. Fully trainable encoder with permanently frozen bank was not used.
