# LiteBN-CM-GA final report

Terminal: `CM_GA_GRADIENT_TRANSFER_NOT_SUPPORTED`.

The single candidate failed its mandatory Stage-0 transfer gate. Per protocol, no residual training, benchmark evaluation, or multiseed run was performed.

| Dataset | AUROC | 95% CI | Spearman | 95% CI | control AUROC | AUROC advantage 95% CI | pass |
|---|---:|---|---:|---|---:|---|---|
| OpenBMI | 0.6432 | [0.5738, 0.7130] | 0.2599 | [0.1180, 0.4009] | 0.5450 | [+0.0019, +0.1879] | False |
| WBCIC | 0.6035 | [0.5314, 0.6746] | 0.2214 | [0.0585, 0.3769] | 0.5285 | [-0.0179, +0.1686] | False |

## Required questions

### 1. Does the signal transfer for OpenBMI MI?

No under the locked gate. Primary AUROC and Spearman were positive, but the Spearman advantage over the different-subject control had a confidence interval crossing zero.

### 2. Does it transfer for WBCIC MI?

No under the locked gate. Primary AUROC and Spearman were positive, but the AUROC advantage over control had a confidence interval crossing zero.

### 3. Fraction of residual updates rejected?

Not applicable; Stage 1 was not authorized.

### 4. Does WBCIC remain at least as good as matched LiteBN?

Not evaluated; the exposed benchmark was not opened.

### 5. Does OpenBMI MI exceed 75.55%?

Not evaluated.

### 6. Does OpenBMI ERP exceed 85.57%?

Not evaluated.

### 7. Does OpenBMI SSVEP exceed 92.34%?

Not evaluated.

### 8. Does WBCIC MI exceed 79.18%?

Not evaluated.

### 9. Are all thresholds exceeded simultaneously?

No evidence; Stage 0 stopped the experiment.

### 10. Do three-seed means exceed all thresholds?

Not applicable; Stage 2 did not run.

### 11. Epoch-0 fallback frequency?

Not applicable; no Stage-1 checkpoint selection occurred.

### 12. Learned C/M amplitudes?

No learning occurred. lambda_channel and all three mixer gamma parameters remain exactly zero at the audited identity initialization.

### 13. Enough evidence to freeze LiteBN-CM-GA?

No.

EXPOSED_BENCHMARK_EVALUATION_ACCESSED = NO

NEW_SEALED_TEST_ACCESSED = NO

FINAL_MODEL_CANDIDATE_FAIL
