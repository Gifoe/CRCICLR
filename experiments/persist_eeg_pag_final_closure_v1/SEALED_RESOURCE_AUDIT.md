# Sealed resource audit

Status before preregistration: `SEALED_RESOURCE_UNAVAILABLE`.

The repository's frozen data audit states that only the OpenBMI 40-subject development cache and WBCIC 41-subject development cache are materialized; OpenBMI sealed identifiers and WBCIC outer identifiers are absent and unenumerated. The V8 outer lock independently records all WBCIC outer flags as false. The historical PUD-Aux loader explicitly imports no internal-holdout or WBCIC-outer loader.

This is a resource-availability finding, not a sealed outcome. No sealed labels, subjects, features, predictions, or outcomes are read. The preregistered confirmation is therefore fail-closed until an independently audited sealed resource is supplied.
