# Stage-0 v2 protocol freeze

This file is written before Gate-0 performance. The binary task is fixed to EEGMMIDB runs 4/8/12 (T1/T2 left/right imagery), with subject-level split seed 2027 and three training seeds. The matched reference is CAR64 (`source_car64`), frozen before any result is read. All B2–B6 methods share the same Stage0Transformer, temporal stem, optimizer, schedule, and seeds. B4 uses only continuous active/reference/count metadata; no family category or signed B vector. B2–B6 have no channel-row positional identity; canonical component identity is reserved for B7/B8. Gate A aggregates all held-out non-polarity operators and requires both an overall mean and at least two family means. Gate B compares B6 against the strongest B2–B5 model overall and on composite-only operators.

No protocol, operator, threshold, seed, or matched-reference choice may be changed after performance inspection.
