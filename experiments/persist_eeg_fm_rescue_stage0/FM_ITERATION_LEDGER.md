# FM iteration ledger

## V0 official-checkpoint full fine-tuning

- Diagnosis: official checkpoints and final 200-D representations load correctly; dataset adapters must repair only sampling, unit and channel-index requirements.
- Change: maximal legal channels, 200-Hz four-patch input, official checkpoint, full-model AdamW fine-tuning and a new two-class head.
- Evidence available: repository/checkpoint/input audits and source-validation only.
- Prediction: competent source validation without layer or outcome search.
- Outcome evidence inspected: NO.
- Keep/reject: pending the frozen source-validation search.
