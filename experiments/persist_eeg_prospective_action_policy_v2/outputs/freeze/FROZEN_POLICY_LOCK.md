# Frozen policy lock

`OUTER_TEST_USED = false`

- Lock hash: `e679c7a955ccf3745bb35ce6c86a61c57705557f3eed8917b724b0e5613b5fd4`
- Split hash: `f033c6ce4b6c79dcbba60581423b812ed32a921595c740396aac62163edac944`
- Frozen candidates: `I003_CROSS_RUN_FULL, I003_CROSS_RUN_PROTECTED_SAFE`
- Learned checkpoints: none (both policies are deterministic)
- Allowed development-holdout openings: one
- Post-holdout tuning: forbidden
- WBCIC outer evaluation: unauthorized

The full policy maximizes recovered exploration headroom but may invoke ERASE.
The protected-safe policy forbids ERASE and is frozen as the safety comparator.
