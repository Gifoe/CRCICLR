# Reproduce

```bash
cd /root/autodl-tmp/hsc_tta_eeg/repo
/root/miniconda3/envs/hsc_gpu/bin/python scripts/fm_routing_v7_full/run_all.py --repo-root /root/autodl-tmp/hsc_tta_eeg/repo --resume
/root/miniconda3/envs/hsc_gpu/bin/python -m pytest -q
```

Terminal resume rebuilds manifests/reports only and cannot enter a downstream gate.
