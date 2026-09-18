# Frozen EEGConformer/FBCNet PEEH (seed 0)

This extension replays 40 previously selected frozen checkpoints. It does not train models, change a checkpoint, update batch-normalization, or reselect a model.

|Model|Task|Params|D|Intact BA|Protected BA|Random BA|PEEH pp|95% CI|Coverage|Significant|
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
|EEGConformer|OpenBMI_MI|853,506|32|70.40|58.39|66.30|7.911|[5.900, 10.469]|5/5|YES|
|EEGConformer|OpenBMI_ERP|341,506|32|79.35|74.49|76.45|1.959|[1.315, 2.643]|3/5|YES|
|EEGConformer|OpenBMI_SSVEP|853,572|32|82.30|65.37|75.05|9.678|[7.732, 11.860]|5/5|YES|
|EEGConformer|WBCIC_MI|847,106|32|77.91|65.44|75.94|10.499|[6.241, 14.965]|5/5|YES|
|FBCNet|OpenBMI_MI|21,026|1,152|62.23|59.04|60.74|1.699|[0.329, 3.081]|4/5|YES|
|FBCNet|OpenBMI_ERP|21,026|1,152|66.00|55.33|62.02|6.684|[5.089, 8.389]|5/5|YES|
|FBCNet|OpenBMI_SSVEP|23,332|1,152|69.23|38.93|59.39|20.462|[15.149, 25.883]|5/5|YES|
|FBCNet|WBCIC_MI|19,874|1,152|52.16|52.06|52.09|0.032|[-0.321, 0.364]|1/5|NO|

Interpretation is restricted to persistence-consequence consistency. Absolute PEEH is not a model-quality ranking.
