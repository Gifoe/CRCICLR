# SCAA Stage-0 protocol

The primary intervention is supervised classifier-head-only adaptation from
the subject's out-of-fold ERM anchor. The encoder and normalization state are
frozen. Target S1 is split within class in cache chronology: the first 70% is
training and the final 30% is validation. One global learning rate is selected
from `{1e-4, 3e-4, 1e-3}` using mean S1-validation balanced accuracy across all
41 development subjects, both backbones, and three seeds. Checkpoint ties use
lower validation NLL and then earlier epoch.

The frozen M1 is evaluated without further training on S2 and S3. Primary
utility is the balanced-accuracy difference from the same frozen M0 anchor.
Seeds are averaged within subject/backbone. Pooled analysis first averages the
two backbones within subject, leaving 41 independent subject units.

Primary analyses are Pearson/Spearman transfer with 10,000 subject bootstrap
resamples, prospectively defined sign concordance, the fixed certificate
`Delta_S2 > 0`, future harm, coverage, and Anchor/Always/S2-Gated S3 policy BA.
The exact Strong Support gates and terminal logic are machine-readable in
`protocol/SCAA_STAGE0_PROTOCOL_LOCK.json`. A fixed 90% one-sided paired S2 LCB
is secondary and cannot rescue primary failure.

No S2/S3 utility is accessed until all code is committed, the lock is generated,
and the lock itself is committed. After outcome access, adapter, hyperparameters,
subjects, folds, backbones, metric, and certificate are immutable.
