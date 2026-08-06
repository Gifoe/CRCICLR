# Final decision

**Verdict: `V7_STOP_NO_ADMISSIBLE_EXPERT_POOL`**

Scientific stop at Phase A. No pre-registered sleep-plus-motor dataset pair supports CBraMod and at least two additional checkpoint-faithful model families. HMC has only CBraMod; SleepEDFFull is absent.

## Phase A Gate

- A1: `False`
- A2: `False`
- A3: `False`
- A4: `False`
- A5: `False`
- A6: `False`
- A7: `True`
- A8: `False`

## Downstream status

All Phase B–I outputs are **NOT RUN**. No `CORE_BENCHMARK_FREEZE.json` was created. No expert, Oracle, router, abstention, PARES, formal-calibration, internal-final, or external experiment was run.

Historical V7/V7R results were not overwritten. Channel semantics were not fabricated. No model or dataset was replaced based on performance. No backbone was fine-tuned, no evaluation leakage occurred, protected subjects were not opened, and CAP was not opened.
