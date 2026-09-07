# PERSIST-EEG multi-backbone final closure

This experiment prospectively tests whether the negative EEGNet actionability
result is representation-dependent.  The five-family roster is frozen to
EEGNet, FBCNet, EEGConformer, DeepConvNet, and TeCh.  EEGNet is imported only
as read-only published evidence; B1--B4 are the new prospective experiments.

The WBCIC development cache, 41/10 subject split, S1+S2 -> S3 session roles,
four rank blocks, competence gate, and H1--H5 definitions are inherited from
`persist_eeg_wbcic_actionability_v2`.  The ten outer subjects remain sealed.

## Execution

```powershell
$python = 'D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe'
$code = 'experiments\persist_eeg_multibackbone_final_closure\code'
& $python "$code\freeze_protocol.py"
& $python "$code\pipeline.py" all --device cuda --workers 0
& $python "$code\finalize.py"
& $python "$code\figures.py"
```

`pipeline.py all` is resumable at content-validated checkpoints.  It completes
task-only search first and runs H1--H5 only for backbones that pass the frozen
competence gate.  `finalize.py` performs the prospective 16-candidate Holm
correction and terminates constructive search unless a candidate passes every
effect, stability, and multiplicity gate.

Large epoch caches, checkpoints, and fold embeddings are intentionally ignored
by Git.  Compact locks, results, reports, and figures are versioned.
