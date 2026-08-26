# Invariance manipulation audit

- CORAL λ=0.01: S_I=0.0011 [-0.0015, 0.0049], ΔBA=-0.0024 [-0.0082, 0.0021], meaningful=False, counterexample=NONE.
- CORAL λ=0.1: S_I=-0.0048 [-0.0092, 0.0004], ΔBA=-0.0013 [-0.0068, 0.0030], meaningful=False, counterexample=NONE.
- CORAL λ=1: S_I=-0.0062 [-0.0093, -0.0026], ΔBA=0.0005 [-0.0027, 0.0033], meaningful=False, counterexample=NONE.
- DANN λ=0.01: S_I=0.0105 [0.0069, 0.0143], ΔBA=-0.0026 [-0.0081, 0.0007], meaningful=False, counterexample=NONE.
- DANN λ=0.1: S_I=0.0129 [0.0092, 0.0166], ΔBA=-0.0025 [-0.0080, 0.0020], meaningful=False, counterexample=NONE.
- DANN λ=1: S_I=0.0094 [0.0049, 0.0142], ΔBA=-0.0152 [-0.0265, -0.0049], meaningful=False, counterexample=NONE.
- MMD λ=0.01: S_I=0.0026 [-0.0007, 0.0067], ΔBA=-0.0016 [-0.0067, 0.0032], meaningful=False, counterexample=NONE.
- MMD λ=0.1: S_I=0.0030 [-0.0002, 0.0070], ΔBA=-0.0021 [-0.0046, 0.0005], meaningful=False, counterexample=NONE.
- MMD λ=1: S_I=0.0017 [-0.0039, 0.0069], ΔBA=-0.0012 [-0.0060, 0.0033], meaningful=False, counterexample=NONE.

Meaningful reduction requires mean S_I ≥ max(0.05, 10% of matched ERM absolute identity skill) and paired hierarchical CI lower > 0. Meaningful families: none.
