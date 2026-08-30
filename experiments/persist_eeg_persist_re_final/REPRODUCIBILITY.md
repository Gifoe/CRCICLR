# Reproducibility

Use Python 3.11/torch CUDA environment `Benchmark_TTA_Win`, set `PYTHONPATH` to
`code`, and run `python -u code/run_source.py --device cuda`.  Seeds are derived
from SHA-256 labels recorded in the code.  The source fit cache is resumable but
is not a deliverable.  Tests run without EEG data and cover gradient,
centering, partition, balancing, optimizer, bootstrap, and authorization
contracts.

