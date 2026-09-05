# Second-backbone geometry check

Candidate: DeepConvNet, because it is competent in the existing multi-backbone closure.

Decision: `DO_NOT_IMPROVISE`. The existing DeepConvNet closure uses a different representation/block and candidate-roster protocol than Shared Geometry V1.2. There is no unambiguous one-to-one mapping for protected coordinates, random controls, and geometry gates. A new mapping would change the frozen protocol, so no second-backbone training is executed.
