# WBCIC S0→S1 full legal action-bank audit

Only S0 labels train the experts; S1 held-subject rows receive six predictions per expert. S2 and outer resources remain inaccessible.

{
  "schema": "CGRSTACK_WBCIC_FULL_LEGAL_ACTION_BANK_V1",
  "dataset": "WBCIC",
  "subjects": 41,
  "subject_ids": [
    "1",
    "2",
    "3",
    "5",
    "6",
    "7",
    "9",
    "11",
    "12",
    "13",
    "14",
    "16",
    "17",
    "18",
    "19",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "41",
    "42",
    "44",
    "45",
    "47",
    "48",
    "49",
    "50"
  ],
  "sample_count": 8198,
  "run_count_per_expert": 6,
  "expert_count": 3,
  "experts": {
    "E0": "ERM EEGNet",
    "E1": "subject-adversarial EEGNet (GRL coefficient 0.10)",
    "E2": "CORAL geometry-robust EEGNet (coefficient 0.10)"
  },
  "partition_definition": "two independent fixed hash-salted biological-subject bipartitions; each orientation predicts only the excluded group",
  "partition_salts": [
    "CGRSTACK_WBCIC_PARTITION_R1_B0",
    "CGRSTACK_WBCIC_PARTITION_R1_B1"
  ],
  "complete_case_filter": false,
  "S2_accessed": false,
  "outer_accessed": false,
  "rows": 147564,
  "bank_sha256": "7ef0fdf16433df84f1acbb7a35808fb3c86aa1ca7d44f1c69a2ce97579636abe",
  "source_sessions": [
    0
  ],
  "evaluation_session": 1,
  "samples": 8198,
  "source_samples": 8198,
  "structural_identity_only": true
}
