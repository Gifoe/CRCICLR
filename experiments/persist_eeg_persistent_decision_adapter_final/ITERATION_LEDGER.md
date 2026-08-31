# Iteration ledger

1. Reused the frozen CleanRoom representation archive; no raw EEG was copied.
2. Implemented deterministic temporal block construction, frozen population
   logits, low-rank correction, diagonal-Fisher pooling, cross-fit estimates,
   and correct/wrong/shuffled controls.
3. Repaired a numerical metric edge case for one-class historical blocks by
   using an explicit two-class balanced-accuracy definition.
4. Ran the fixed 12-recipe source search. It failed the declared gate.
5. Sealed WBCIC S2 and all cross-backbone/outer work; no favorable search was
   attempted after the gate failure.
