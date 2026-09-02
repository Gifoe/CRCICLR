# PERSIST-RME design audit

"
        "The subject-gradient signature is a task-risk derivative (embedding, LayerNorm and task head only), not an identity probe: subject IDs index legal source losses and never enter inference. Every risk expert is a full canonical EEGNet trained on every legal source subject with non-zero weights; no subject subset or target information is used. The method is task-agnostic and uses only EEG, labels and source subject IDs.

"
        "It is distinct from random-seed ensembles (fixed risk modes and matched ERM control), subject-subset curricula (all subjects remain), GroupDRO/CVaR (fixed low-rank marginalization rather than one worst group), DANN/CORAL (no representation invariance loss), and stacking (no learned trial-dependent combiner). The canonical EEGNet and preprocessing/evaluator are imported unchanged.
