# Persistence-geometry transfer-risk audit

Terminal: `TRANSFER_RISK_BRIDGE_PARTIAL_OR_NONSPECIFIC`

This is an A-only EEGNet, seed-0, canonical 5-fold audit. Discovery query labels were read only after PRE_OUTCOME_PROTOCOL_LOCK.json; canonical outcome subjects, WBCIC outer-10 and OpenBMI sealed/outer data were not opened.

|Dataset|n subjects|Protected rho|95% CI|LOFO positive|
|---|---:|---:|---|---:|
|OpenBMI|36|0.4539|[0.1090, 0.7122]|5/5|
|WBCIC|41|0.4328|[0.1654, 0.6300]|5/5|

G0: `{'OpenBMI': True, 'WBCIC': True}`
G1: `True`
G2: `False`
G3: `True`

PGEG_AUTHORIZED = `False`
PGEG_TRAINING_STARTED = `False`

Controls and alternative-explanation audit are in the compact CSV tables.
