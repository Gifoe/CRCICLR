# Data legality audit

"
        "PASS before training. OpenBMI uses the frozen 54-subject MI manifest, S1+S2 source and S2 future-session roles. WBCIC uses only the 41 subjects in the frozen development scope lock, S1+S2 source and S3 future session. The sealed WBCIC outer ten and any OpenBMI sealed/internal holdout are not enumerated or opened. Outcome labels are not used for basis construction, training, repair or selection; the only permitted pre-score outcome read is checkpoint equivalence (UIDs, labels and probabilities, no BA).
