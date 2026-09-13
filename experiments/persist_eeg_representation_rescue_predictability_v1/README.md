# PERSIST-EEG representation rescue predictability v1

Development-only audit testing whether ordered class coordinates and full frozen pre-head embeddings make Rescue/Harm routing observable beyond confidence summaries.

Run in order:

```bash
python code/extract_full_representations.py --repo "$REPO" --runtime "$RUNTIME"
python code/build_feature_families.py --repo "$REPO" --runtime "$RUNTIME"
python code/representation_crossfit.py --repo "$REPO" --runtime "$RUNTIME"
python code/representation_policy.py --repo "$REPO" --runtime "$RUNTIME"
python code/aggregate_representation_audit.py --repo "$REPO"
```

The runtime directory stores compressed trial-level representations, feature matrices, and out-of-fold scores. Audits and aggregate outputs are committed under `outputs/`. No internal/final heldout or test artifact is used.
