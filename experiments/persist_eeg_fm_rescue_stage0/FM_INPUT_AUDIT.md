# FM input audit

Every one of the 62 OpenBMI and 58 WBCIC channels occurs in LaBraM's official `standard_1020` vocabulary after case normalization; therefore no channel is dropped. CBraMod has no fixed channel vocabulary and receives the same maximal dataset order. Both inputs are four 200-sample patches at 200 Hz. See `protocol/FM_INPUT_PROTOCOL_LOCK.json` for the frozen unit/filter/resampling details.
