# Final report

terminal = CUMULATIVE_SMOOTH_HARM_SUPPORTED_DECISION_NOT_SUPPORTED

Source/refit-only fixed-sentinel seed-0 EEGNet one-epoch cumulative audit. No outcome, WBCIC outer-10, OpenBMI sealed cohort, seed 1/2 or second backbone was opened.

|dataset|H_BER events|harmful subjects|CE decision AUROC|CE CI|BBR decision AUROC|BBR CI|selected Spearman|BBR same-different AUROC|CE same-different AUROC|BBR Q5-Q1|CE Q5-Q1|
|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|
|OpenBMI|63|41|0.5524431995020229|[0.45377966435841194, 0.6497848765903633]|0.4828301691046789|[0.3919763180350918, 0.5740496664513801]|0.18433975555514973|-0.10499014420583042|-0.02334267040149407|0.031183932346723064|0.09936575052854124|
|WBCIC|58|32|0.46421600520494466|[0.37760840847900673, 0.5551956922858562]|0.44079375406636306|[0.3496629452418926, 0.5321571373114462]|0.004100916490691128|-0.04001301236174365|-0.03708523096942096|-0.14393939393939392|-0.08143939393939392|

## Required answers

- One-epoch exact decision power sufficient: True
- Selected cumulative certificate: NONE
- Final model development scientifically justified: False
- Same-subject specificity: {'BBR': False, 'CE': False}
- Fold robustness: {'BBR': False, 'CE': True}
- Prospective-gradient family closure recommendation: True
- No new horizon or model was started automatically.
