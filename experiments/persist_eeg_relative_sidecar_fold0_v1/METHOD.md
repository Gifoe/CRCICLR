# Method

Stage A fits a `64 -> 32 -> 64` offset estimator from frozen EEGNet embeddings
to detached leave-one-out centers within training subject/session episode groups.
Stage B retains the frozen EEGNet logits and adds a zero-initialized `64 -> 32
-> 2` residual-logit sidecar.  Raw-Sidecar receives `h`; Relative-Sidecar
receives `h - g(h)`.  The two sidecars share initial weights, stream, optimizer,
steps, and selection procedure.  Oracle centering is an after-freeze
transductive diagnostic, never a legal primary method.
