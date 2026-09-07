# Initial inspection

- Timestamp: 2026-08-01 (Asia/Shanghai task date)
- Host: `autodl-container-a6a040afb3-b82784d4`
- Current directory at inspection: `/root`
- Project code: `/root/autodl-tmp/hsc_tta_eeg/repo`
- Data root: `/root/autodl-tmp/hsc_tta_eeg`
- CPU: 2 × Intel Xeon Platinum 8470Q sockets, 52 cores/socket, 2 threads/core, 208 logical CPUs
- Memory: 754 GiB total, approximately 626 GiB available at inspection; no swap
- Data volume: XFS `/dev/md0`, 350 GiB total, approximately 350 GiB available at inspection
- System volume: 30 GiB overlay; large data are prohibited there
- Git: 2.34.1; target repository was empty and cloned into the code directory
- Python: Miniconda base Python 3.12.3; isolated `hsc_cpu` Python 3.11 environment created
- CUDA: explicitly disabled with `CUDA_VISIBLE_DEVICES=""`; no GPU work or model checkpoint download is authorized
- Existing reusable files: no prior HSC-TTA code, dataset, cache, manifest, or output was present
- Writable: `/root/autodl-tmp` confirmed writable

## Risks and controls

The initial shell PATH omitted Miniconda, so all environment commands use absolute paths or activation. The 350GB disk leaves limited headroom for three raw datasets plus processed caches: full downloads require all smoke checks and at least 300GB free; new writes stop below 60GB; full preprocessing does not start below 150GB. CAP subject identity may not be recoverable beyond stable filename prefixes and must remain explicitly documented. GitHub push authentication is not assumed and will be tested only after local commits are ready.

## Execution stages

CPU-1 implements schemas, prediction sets, block bounds, meta-risk prediction, simultaneous conformal certification, deterministic selection, actions, metrics, and synthetic simulations. CPU-2 smoke-downloads official EEGMMIDB, HMC, and CAP files and validates readability. CPU-3+ proceed only if storage and smoke gates pass. The GPU stage and all foundation-model downloads remain prohibited.

