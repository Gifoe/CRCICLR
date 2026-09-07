# Cache gap report

Metadata-only audit. No training, PAG sealed confirmation, sealed outcome, label value, prediction value, or performance outcome was read. Sensitive outer/sealed paths were not opened.

### OpenBMI

- cache currently has: 54 subjects
- frozen search subjects present: 40/40
- frozen internal holdout present: 14/14
- missing internal holdout IDs: []
- sessions available: ['session-1', 'session-2']
- tasks available: ['ERP', 'MI', 'SSVEP']
- raw EEG available: NO
- labels/codes embedded in cache: YES
- can sealed holdout be run from current cache alone: YES (input-ready MI cache + split manifest; not executed)

### WBCIC

- cache currently has: 41 subjects
- V8 development subjects covered: 41/41
- V8 internal holdout present: 10/10
- missing V8 internal holdout IDs: []
- any extra non-development subjects found: NO
- sessions available: ['session-0', 'session-1', 'session-2']
- tasks available: ['MI']
- raw EEG available: NO
- any evidence true outer data are physically present: NO
- can true outer evaluation run from current cache alone: NO

### Minimum missing data

OpenBMI needs no additional subject files for an input-only replay, but raw source would be needed to regenerate preprocessing. WBCIC needs authorized outer-10 MI data (raw or independently verified compatible 58-channel × 1000-sample preprocessed cache), sessions 0/1/2, subject/session metadata, and separately authorized scoring labels.
