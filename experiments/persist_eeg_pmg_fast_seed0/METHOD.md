# PMG fast seed-0 mechanism pilot

PMG simulates prospective subject transfer inside model-fit subjects. Each step uses four pseudo-seen subject folds A and one disjoint pseudo-future fold B, a first-order virtual update with `alpha_inner=1e-4`, `lambda_meta=1.0`, and positive-harm penalty `mu_harm=0.5`. M1 is a five-epoch matched ERM refinement. Both start from one canonical model-fit-only M0 checkpoint. Only source model-fit and discovery sessions are used; no outcome index is constructed.

This is a mechanism pilot, not a multi-seed or outcome confirmation.
