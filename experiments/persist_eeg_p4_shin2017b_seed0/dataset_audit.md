# Shin2017B P4 dataset audit

- Dataset: Shin2017B / nm000268, EEG only.
- Subjects: 29 (sub-01 through sub-29).
- Sessions: 1arithmetic, 3arithmetic, 5arithmetic (chronological S1/S2/S3).
- Task labels: raw 3=subtraction/mental arithmetic, 4=rest; cached binary 1/0.
- Sampling: 200 Hz; no resampling; each whole task trial is 10 s = 2,000 samples.
- Tensor: every session `(20, 30, 2000)`, with 10 trials per class.
- Channels (preserved order): `F7, AFF5h, F3, AFp1, AFp2, AFF6h, F4, F8, AFF1h, AFF2h, Cz, Pz, FCC5h, FCC3h, CCP5h, CCP3h, T7, P7, P3, PPO1h, POO1, POO2, PPO2h, P4, FCC4h, FCC6h, CCP4h, CCP6h, P8, T8`.
- Missing/invalid recordings: none after cache validation.
