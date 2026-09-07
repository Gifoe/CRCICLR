# PERSIST-ICG P5

`code/p5_icg.py` implements the bounded PERSIST-ICG method-development
program from V0 through V2, with V3 entered only when the predeclared
progression rule authorizes it. It uses only TRAIN/development-validation
subjects and writes `outer_test_used: false` into every artifact.

Run from the repository root in the GPU environment with `src` on
`PYTHONPATH`:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -u experiments/persist_eeg_p5_icg/code/p5_icg.py --phase v2 --device cuda
```

The runner is resumable by `implementation_id`, uses deterministic fold/seed
streams shared across versions, and stores the required reports under
`experiments/persist_eeg_p5_icg/outputs/`.
