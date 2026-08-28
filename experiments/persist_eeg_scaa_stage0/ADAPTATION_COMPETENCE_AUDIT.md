# Adaptation competence audit

Only anchor/source information and target S1 were accessed. S2 and S3 were not
loaded. The tested intervention was supervised classifier-head-only adaptation
from the out-of-fold ERM checkpoint. The encoder and all normalization state
were frozen.

Global LR candidates were `[0.0001, 0.0003, 0.001]`. Selection used mean target S1
validation BA across 41 subjects, both primary backbones, and three matched
anchor seeds; ties used lower NLL then smaller LR. The frozen S1 split was the
first 70% versus final 30% within each class in cache chronology.

Selected LR: `0.001`. Mean anchor/adapted S1-validation BA:
`0.75637` / `0.77161`;
delta `+0.01524`. Mean prediction-change
rate `0.07161`; mean relative head change
`0.23237`; catastrophic fraction
`0.00000`.

Competence terminal: `HEAD_ONLY_COMPETENCE_PASS`.
This is not evidence of S2-to-S3 utility transfer.
