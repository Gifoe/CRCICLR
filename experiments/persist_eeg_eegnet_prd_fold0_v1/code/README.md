# EEGNet PRD fold-0 screening

This is a seed-0, OpenBMI fold-0-only matched control: unchanged canonical
EEGNet trained with CE versus the same EEGNet with `CE + 0.25 * PRD(z)`. Both use
the same frozen SEARCH-only cache, normalizer and episode manifest. The recorded
split-SHA amendment means this is a screening control, not a historically exact
baseline reproduction. No holdout or WBCIC data are used.
