# V2 decision-dependence bug

V2 compared uncentered erased logits with centered raw logits for candidate interventions, while random controls were centered on both sides. This asymmetric definition can rank candidates by arbitrary class-independent offsets.
