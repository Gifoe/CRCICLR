# Frozen cross-backbone PEEH (seed 0)

TFFormer was excluded by user instruction. SGN is deferred because 6/20 seed0 frozen checkpoints are absent and this analysis is forbidden to train.

Ninety-nine admissible cells completed. CBraMod / WBCIC_MI / fold4 is excluded because its numerical active rank is below the locked minimum of four; the threshold was not weakened.

|Model|Task|Params|D|Intact BA|Protected BA|Random BA|PEEH pp|95% CI|Coverage|Significant|
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
|EEGNet|OpenBMI_MI|34,162|64|67.57|58.33|64.87|6.542|[4.766, 8.206]|5/5|YES|
|EEGNet|OpenBMI_ERP|9,586|64|84.04|65.29|75.79|10.503|[8.869, 12.110]|5/5|YES|
|EEGNet|OpenBMI_SSVEP|34,292|64|89.67|70.30|81.04|10.744|[7.994, 13.361]|5/5|YES|
|EEGNet|WBCIC_MI|34,098|64|78.44|56.12|76.02|19.898|[13.993, 25.726]|5/5|YES|
|CBraMod|OpenBMI_MI|4,884,002|200|50.67|50.11|50.14|0.027|[-0.924, 0.872]|3/5|NO|
|CBraMod|OpenBMI_ERP|4,884,002|200|74.60|69.98|71.55|1.574|[0.929, 2.239]|4/5|YES|
|CBraMod|OpenBMI_SSVEP|4,884,404|200|85.04|74.14|77.15|3.011|[2.533, 3.462]|1/5|YES|
|CBraMod|WBCIC_MI|4,884,002|200|NA|NA|NA|NA|NA|3/4|NOT_ESTIMATED|
|TeCh|OpenBMI_MI|793,026|128|71.29|57.88|66.71|8.829|[5.119, 14.424]|5/5|YES|
|TeCh|OpenBMI_ERP|793,026|128|79.22|64.46|72.72|8.254|[6.485, 9.984]|5/5|YES|
|TeCh|OpenBMI_SSVEP|6,203,140|256|83.93|60.36|71.64|11.283|[9.874, 12.734]|5/5|YES|
|TeCh|WBCIC_MI|789,954|128|71.94|56.72|66.73|10.008|[5.974, 13.795]|5/5|YES|
|ModernTCN|OpenBMI_MI|17,391,170|246,016|60.56|60.54|60.59|0.056|[-0.024, 0.141]|1/5|NO|
|ModernTCN|OpenBMI_ERP|17,026,114|63,488|76.88|70.71|75.81|5.092|[4.296, 5.857]|5/5|YES|
|ModernTCN|OpenBMI_SSVEP|17,883,204|246,016|85.80|79.21|84.54|5.323|[3.639, 7.051]|5/5|YES|
|ModernTCN|WBCIC_MI|15,914,050|230,144|66.84|66.88|66.84|-0.034|[-0.597, 0.426]|3/5|NO|
|Medformer|OpenBMI_MI|3,890,178|72,576|67.63|59.02|65.74|6.723|[2.268, 13.455]|5/5|YES|
|Medformer|OpenBMI_ERP|3,781,890|18,432|76.54|58.37|68.03|9.656|[8.419, 10.881]|5/5|YES|
|Medformer|OpenBMI_SSVEP|4,035,332|72,576|87.50|36.87|67.39|30.521|[24.381, 36.613]|5/5|YES|
|Medformer|WBCIC_MI|3,853,314|72,576|72.41|60.16|70.03|9.875|[6.797, 12.999]|5/5|YES|

Interpretation is restricted to persistence-consequence consistency. Absolute PEEH is not a model-quality ranking.
