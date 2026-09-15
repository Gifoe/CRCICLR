# TeCh P2 selection-factor replication

This is the serial successor to the frozen EEGNet P2 experiment.  It starts
only after the EEGNet runtime completion marker exists and the EEGNet
protocol-only validation passes.

The selector definitions, exact-rank dynamic program, tie breaking, corrected
PSWA probe, ridge alpha, random-draw handling, session minimum, biological-
subject aggregation, and 20,000-resample paired bootstrap are identical to the
EEGNet P2 implementation.  The only scientific change is the frozen backbone:
TeCh replaces EEGNet.  No neural model is trained or fine-tuned.
