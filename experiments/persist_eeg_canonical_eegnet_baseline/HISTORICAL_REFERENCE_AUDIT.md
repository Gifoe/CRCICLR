# Historical reference audit

## CANONICAL BASELINE STATUS

Historical values are context only and were not used for model construction, epoch selection, tuning, or stopping. The nearest stored references are OpenBMI BA=0.7719 and WBCIC BA=0.7884 from prior project artifacts with different subject/cache scopes. If a canonical result differs by more than five percentage points from its reference, the result is reported as-is and triggers an audit of split, subject roles, session usage, preprocessing, channel handling, normalization, checkpoint selection and metric aggregation; no parameter is changed to match the reference.
