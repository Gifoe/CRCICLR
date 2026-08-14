# PERSIST-SI P4 Closure

- Decision: `P4_SELECTIVE_INVARIANCE_NOT_SUPPORTED`
- Formal method lock: `REFUSED`
- Outer-test used: `false`
- Formal outer evaluation authorized: `false`

## Development versions

| Version | MI ΔBA | ERP ΔBA | SSVEP ΔBA | A–D | Decision |
|---|---:|---:|---:|:---:|---|
| SI_V0 | -0.0024 | +0.0094 | -0.0000 | true | P4_SI_REPRESENTATION_ONLY |
| SI_V1 | -0.0015 | +0.0096 | -0.0018 | true | P4_SI_REPRESENTATION_ONLY |
| SI_V2 | -0.0012 | +0.0089 | -0.0022 | true | P4_SI_REPRESENTATION_ONLY |
| SI_V3 | +0.0009 | +0.0080 | -0.0007 | false | P4_SELECTIVE_INVARIANCE_NOT_SUPPORTED |
| SI_V4 | -0.0006 | +0.0079 | +0.0004 | false | P4_SELECTIVE_INVARIANCE_NOT_SUPPORTED |

## Closure

Repeated-measure persistence and target relevance are detectable, and protected-vs-nuisance intervention audits are often valid, but the current selective representation intervention does not establish a stable validation generalization gain or the preregistered nuisance-suppression gate across the development panel.

Five scientifically distinct development versions (SI-V0 through SI-V4) were evaluated on 3 folds x 2 seeds. No version met the preregistered lock/generalization requirements; further tuning would be unbounded search.

All method changes were fit/evaluated with TRAIN/VALIDATION only. No outer-test signal, label, embedding, or metric was accessed.
