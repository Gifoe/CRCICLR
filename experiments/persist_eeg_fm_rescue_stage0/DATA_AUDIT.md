# Data audit

Only the materialized OpenBMI 40-subject development cache and WBCIC 41-subject development cache are addressable. OpenBMI has shape `[8000, 62, 1000]` and two sessions; WBCIC has shape `[24591, 58, 1000]` and three sessions. The WBCIC outer 10 and OpenBMI sealed holdout identifiers are absent and were not enumerated. Frozen five-fold roles are copied exactly from the historical protocols. Target/outcome subjects never enter anchor training.
