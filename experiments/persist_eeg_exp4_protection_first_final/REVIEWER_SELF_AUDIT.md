# Reviewer self-audit

* **Strong Generic baseline?** Yes on the prespecified S2 subject-held-out selection and S3 development endpoint: +7.07 pp over Frozen. It is not a weak straw man.
* **Capacity mismatch?** No intentional mismatch: all adapters are the same zero-initialized 32×32 linear residual with the same data, optimizer, epochs, and seeds. V1 is the only method difference.
* **Leakage?** No outcome S3 labels enter basis construction or generic selection. The development loader enforces 41 allowed subjects and checks the sealed locks.
* **Dimensionality explanation?** Controls use the same rank and include Random/PCA/persistence-only/identity guards. They explain the tiny V1 effect; specificity is not established.
* **One-subject/seed effect?** The primary unit is 41 subjects and the exact sign and bootstrap summaries are retained. RandomGuard has three deterministic draws collapsed before inference. Still, only one optimization seed was used for the primary methods, so seed robustness is limited.
* **Mechanism?** V1 coordinate preservation is real, but decision-response preservation is incomplete. V2 fixes the latter and loses performance; V3 is intermediate and still fails. This inconsistency is a substantive weakness, not hidden.
* **Frozen-model artifact?** No: Generic has substantial headroom and all guards adapt outside the protected subspace. The issue is that protection does not improve safety here.
* **Second backbone?** Not run because the primary EEGNet gate failed. This is correct protocol behavior, but leaves no replication.
* **Outer one-shot?** Outer was never opened. `OUTER_ACCESS_LOCK.json` remains sealed and no outer files exist.
* **Causal chain?** Experiments 1–3 motivate the protected block, but Exp4 does not validate the constructive consequence. The appropriate conclusion is a negative/falsifying result for this formulation, not a broad invariance claim.

The main reviewer weakness remains external validity: only EEGNet development data were tested. A second weakness is that the frozen P01_04 assignment comes from a prior audit/model family and may not align perfectly with an S1-only anchor. Resolving that would require a new prospective protocol, not post hoc tuning of the sealed outer set.
