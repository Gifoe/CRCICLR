# HSC-TTA EEG

CPU-only research implementation of **Hierarchical Subject-Certified Test-Time Adaptation**. The repository separates subject-visible context `U_s` from offline-only future data `V_s`, implements simultaneous subject-level conformal certification across the complete action/threshold surface, and provides reproducible dataset, split, episode, simulation, and reporting pipelines.

The CPU phase never downloads a foundation-model checkpoint and never invokes CUDA. Raw EEG, processed caches, manifests, outputs, logs, state, and credentials live outside Git under `/root/autodl-tmp/hsc_tta_eeg`.

```bash
conda activate hsc_cpu
pip install -e '.[eeg,test,report]'
export CUDA_VISIBLE_DEVICES=""
pytest -q
python scripts/run_mock_pipeline.py --config configs/method/hsc_tta.yaml --device cpu
```

See `docs/CPU_PIPELINE.md` and `docs/METHOD_SPEC.md` for protocols and statistical assumptions.

