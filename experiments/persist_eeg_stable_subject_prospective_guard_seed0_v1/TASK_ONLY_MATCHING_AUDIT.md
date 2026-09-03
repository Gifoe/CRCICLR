# Task-only matching audit

TASK_ONLY_MATCHED, SSPG, CROSS_SUBJECT_K4_GUARD and RANDOM_DIRECTION_GUARD start from the exact canonical seed-0 checkpoint and use identical candidate-independent A schedules, dropout-keyed RNG, AdamW settings, clipping and two-epoch horizon. Only the registered post-AdamW correction differs.
