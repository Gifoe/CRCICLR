# Confirmatory dataset requirements

The present HMC, CAP, and EEGMMIDB final outcomes are tainted by prior inspection. They cannot support a confirmatory claim for HSC-TTA v2.

## Required datasets

1. One untouched sleep dataset or site with subject-level calibration and test partitions fixed before labels are opened.
2. One untouched motor-imagery or second EEG task dataset, also excluded from all present method and hyperparameter choices.

For each task, target at least 30 calibration subjects and 50 test subjects. The certifiability audit must be rerun before acquisition if fewer are available. Subjects, not windows, are the exchangeability and resampling units.

## Compatibility and episode protocol

- Sleep must supply a real C4-compatible derivation or a mapping approved before labels are inspected. MI must preserve the declared electrode order; missing channels are explicit, never silently duplicated.
- Label mapping is frozen in the acquisition manifest and audited against the source annotation specification.
- Each subject yields a chronological, disjoint context set U and future set V. Actions and decisions use U only; V labels remain inaccessible until the decision file is atomically written and hashed.
- Dataset normalization cannot use V statistics.
- Audit whether any subject/site overlaps CBraMod pretraining data. Ambiguous overlap disqualifies the dataset from the confirmatory claim.

## Storage, licensing, and ethics

Estimate compressed download, extracted data, token cache, and temporary conversion space before acquisition; reserve at least 2.5 times the extracted size plus 60 GB safety headroom. Do not automatically download a large dataset from this template. Record license terms, redistribution restrictions, consent/ethics basis, data-use approval, and deletion obligations.

## Freeze workflow

1. Publish the method-freeze hash, dataset identity, channel/label maps, sample sizes, exclusion rules, seeds, and analysis plan.
2. Populate `configs/v2_confirmation_template.json`; require disjoint subject IDs, license approval, pretraining-overlap audit, and the exact method-freeze hash.
3. Implement `ConfirmatoryDatasetAdapter`, run `scripts/run_v2_confirmation.py --dry-run`, and verify hashes without opening test outcomes.
4. Fit only source-allowed components, fit predictors only on the specified meta subjects, compute q only on calibration subjects, freeze U-only decisions, then open V exactly once.
5. Report all seeds and exclusions, marginal validity, CSR/fallback, set size, utility, TTA rate, and PPV regardless of outcome. Never revise the method using this test.
