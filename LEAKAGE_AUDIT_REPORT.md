# Leakage Audit Report

Result: **ZERO DETECTED LEAKAGE** in CPU artifacts and formal APIs.

- Artifact validator failures: 0.
- All split roles remain subject-disjoint.
- All main120 U/V index intersections are empty.
- Sleep future episodes begin strictly after context and contain exactly 240 valid epochs.
- MI context/future runs remain 4/6 versus 8/10/12/14.
- Context and pre-outcome Pydantic schemas forbid undeclared future fields.
- Selector rejects `future_*`, future classification metrics, and harmful-adaptation outcomes.
- Decision/outcome tables join one-to-one on dataset, seed, episode, subject, and alpha.
- Final-test gate rejects missing or changed configuration and decision hashes.
- CAP internal splits contain no task-head or predictor fitting subjects.

This report audits the CPU protocol and interfaces. It does not claim that future GPU code is leak-free until that code and its outputs pass the same gates.
