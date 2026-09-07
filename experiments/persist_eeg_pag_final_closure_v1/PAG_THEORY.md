# PAG theory: insufficiency and transportability

Let environments e generate (X,Y), let z_p(X) be a persistent representation component, f a fixed predictor, and I a fixed intervention. Define P(z_p)>0 as a source persistence property, C_s(z_p)=R_s(f o I)-R_s(f), and A_t(z_p)=R_t(f)-R_t(f o I). A positive C_s means the source intervention increases risk; positive A_t means it helps target risk.

## Proposition 1 (insufficiency)

Without a condition linking target conditional distributions to source conditional distributions, source-observable P, C_s, and G cannot determine the sign of A_t. Two target environments can agree on every source-observable premise while giving opposite A_t signs.

### Proof

Take binary Y in {-1,+1}. On the source set z_p=Y and f(x)=sign(z_p). Let I replace z_p by zero. Then R_s(f)=0 and R_s(f o I)=1/2 (ties predict +1), so C_s=1/2 != 0. The source has perfect persistence and perfect task geometry. Define target e1 by z_p=Y and target e2 by z_p=-Y. In e1, R(f)=0 and R(f o I)=1/2, so A_e1=-1/2. In e2, R(f)=1 and R(f o I)=1/2, so A_e2=+1/2. Both targets are compatible with the same source premises; therefore (P,C,G) does not identify the sign of A_t. QED.

## Proposition 2 (sufficient transportability condition)

Let Delta_e=R_e(f)-R_e(f o I). If every admissible target satisfies Delta_e=Delta_s for the fixed f and I, and source evidence establishes Delta_s>=delta>0, then every admissible target has A_e>=delta>0.

### Proof

Transportability gives A_e=Delta_e=Delta_s. The source lower bound therefore gives A_e>=delta in every admissible target. QED.

This is an additional cross-environment assumption, not a consequence of persistence, consequence, or shared geometry alone.
