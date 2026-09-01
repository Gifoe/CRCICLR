# R1 handoff audit

CPV2 reads the committed R1 tables only; no R1 scientific computation is rerun.

{
  "r1_terminal": "CGRSTACK_SOURCE_NOT_SUPPORTED",
  "expected_terminal": "CGRSTACK_SOURCE_NOT_SUPPORTED",
  "terminal_verified": true,
  "key_results": [
    {
      "dataset": "OpenBMI",
      "method": "B3_STRONGEST_KEEP_STACK",
      "BA": 0.6213888888888888,
      "macro_f1": 0.5918995971213649,
      "delta_vs_STRONGEST_KEEP_STACK": 0.0,
      "median_subject_delta_BA": 0.0,
      "bootstrap_CI95_L": 0.0,
      "bootstrap_CI95_U": 0.0,
      "positive_subject_fraction": 0.0,
      "nonnegative_subject_fraction": 1.0,
      "subjects": 54,
      "actual_decision_change_rate": 0.0,
      "unsafe_decision_change_rate": 0.0,
      "soft_fusion_mass": 0.0,
      "rescue_precision": 0.0,
      "OUTER_TEST_USED": false
    },
    {
      "dataset": "OpenBMI",
      "method": "B6_LOGISTIC_ALL_EXPERT",
      "BA": 0.659537037037037,
      "macro_f1": 0.640268590493185,
      "delta_vs_STRONGEST_KEEP_STACK": 0.0381481481481481,
      "median_subject_delta_BA": 0.035,
      "bootstrap_CI95_L": 0.021574074074074,
      "bootstrap_CI95_U": 0.0547245370370369,
      "positive_subject_fraction": 0.7962962962962963,
      "nonnegative_subject_fraction": 0.7962962962962963,
      "subjects": 54,
      "actual_decision_change_rate": 0.1994444444444444,
      "unsafe_decision_change_rate": 0.404363974001857,
      "soft_fusion_mass": 0.0,
      "rescue_precision": 0.595636025998143,
      "OUTER_TEST_USED": false
    },
    {
      "dataset": "OpenBMI",
      "method": "B7_ALL_EXPERT_STACK",
      "BA": 0.6374074074074074,
      "macro_f1": 0.6097835722911173,
      "delta_vs_STRONGEST_KEEP_STACK": 0.0160185185185185,
      "median_subject_delta_BA": 0.0174999999999999,
      "bootstrap_CI95_L": 0.0037037037037036,
      "bootstrap_CI95_U": 0.0274999999999999,
      "positive_subject_fraction": 0.6851851851851852,
      "nonnegative_subject_fraction": 0.7222222222222222,
      "subjects": 54,
      "actual_decision_change_rate": 0.1299074074074074,
      "unsafe_decision_change_rate": 0.4383464005702067,
      "soft_fusion_mass": 1.0,
      "rescue_precision": 0.5616535994297933,
      "OUTER_TEST_USED": false
    },
    {
      "dataset": "OpenBMI",
      "method": "B12_CGRFUSE_R1",
      "BA": 0.6219444444444444,
      "macro_f1": 0.5924855607846033,
      "delta_vs_STRONGEST_KEEP_STACK": 0.0005555555555555,
      "median_subject_delta_BA": 0.0,
      "bootstrap_CI95_L": -0.0006481481481481,
      "bootstrap_CI95_U": 0.0017592592592592,
      "positive_subject_fraction": 0.1481481481481481,
      "nonnegative_subject_fraction": 0.9444444444444444,
      "subjects": 54,
      "actual_decision_change_rate": 0.0066666666666666,
      "unsafe_decision_change_rate": 0.4583333333333333,
      "soft_fusion_mass": 0.0173611111111111,
      "rescue_precision": 0.5416666666666666,
      "OUTER_TEST_USED": false
    },
    {
      "dataset": "WBCIC",
      "method": "B3_STRONGEST_KEEP_STACK",
      "BA": 0.5088950480413895,
      "macro_f1": 0.4101232680549143,
      "delta_vs_STRONGEST_KEEP_STACK": 0.0,
      "median_subject_delta_BA": 0.0,
      "bootstrap_CI95_L": 0.0,
      "bootstrap_CI95_U": 0.0,
      "positive_subject_fraction": 0.0,
      "nonnegative_subject_fraction": 1.0,
      "subjects": 41,
      "actual_decision_change_rate": 0.0,
      "unsafe_decision_change_rate": 0.0,
      "soft_fusion_mass": 0.0,
      "rescue_precision": 0.0,
      "OUTER_TEST_USED": false
    },
    {
      "dataset": "WBCIC",
      "method": "B6_LOGISTIC_ALL_EXPERT",
      "BA": 0.5173946784922395,
      "macro_f1": 0.4328841526954481,
      "delta_vs_STRONGEST_KEEP_STACK": 0.0084996304508499,
      "median_subject_delta_BA": 0.0,
      "bootstrap_CI95_L": -0.0023540280857353,
      "bootstrap_CI95_U": 0.0194013303769401,
      "positive_subject_fraction": 0.4146341463414634,
      "nonnegative_subject_fraction": 0.6585365853658537,
      "subjects": 41,
      "actual_decision_change_rate": 0.2612832398145889,
      "unsafe_decision_change_rate": 0.4836601307189542,
      "soft_fusion_mass": 0.0,
      "rescue_precision": 0.5163398692810458,
      "OUTER_TEST_USED": false
    },
    {
      "dataset": "WBCIC",
      "method": "B7_ALL_EXPERT_STACK",
      "BA": 0.5105974377925597,
      "macro_f1": 0.3947587588170297,
      "delta_vs_STRONGEST_KEEP_STACK": 0.0017023897511702,
      "median_subject_delta_BA": 0.0,
      "bootstrap_CI95_L": -0.0074439517122443,
      "bootstrap_CI95_U": 0.0101023651145602,
      "positive_subject_fraction": 0.3658536585365853,
      "nonnegative_subject_fraction": 0.6097560975609756,
      "subjects": 41,
      "actual_decision_change_rate": 0.1851671139302269,
      "unsafe_decision_change_rate": 0.4953886693017127,
      "soft_fusion_mass": 1.0,
      "rescue_precision": 0.5046113306982872,
      "OUTER_TEST_USED": false
    },
    {
      "dataset": "WBCIC",
      "method": "B12_CGRFUSE_R1",
      "BA": 0.5088950480413895,
      "macro_f1": 0.4101232680549143,
      "delta_vs_STRONGEST_KEEP_STACK": 0.0,
      "median_subject_delta_BA": 0.0,
      "bootstrap_CI95_L": 0.0,
      "bootstrap_CI95_U": 0.0,
      "positive_subject_fraction": 0.0,
      "nonnegative_subject_fraction": 1.0,
      "subjects": 41,
      "actual_decision_change_rate": 0.0,
      "unsafe_decision_change_rate": 0.0,
      "soft_fusion_mass": 0.0,
      "rescue_precision": 0.0,
      "OUTER_TEST_USED": false
    }
  ],
  "positive_subject_fractions": [
    {
      "dataset": "OpenBMI",
      "method": "B3_STRONGEST_KEEP_STACK",
      "positive_subject_fraction": 0.0
    },
    {
      "dataset": "OpenBMI",
      "method": "B6_LOGISTIC_ALL_EXPERT",
      "positive_subject_fraction": 0.7962962962962963
    },
    {
      "dataset": "OpenBMI",
      "method": "B7_ALL_EXPERT_STACK",
      "positive_subject_fraction": 0.6851851851851852
    },
    {
      "dataset": "OpenBMI",
      "method": "B12_CGRFUSE_R1",
      "positive_subject_fraction": 0.1481481481481481
    },
    {
      "dataset": "WBCIC",
      "method": "B3_STRONGEST_KEEP_STACK",
      "positive_subject_fraction": 0.0
    },
    {
      "dataset": "WBCIC",
      "method": "B6_LOGISTIC_ALL_EXPERT",
      "positive_subject_fraction": 0.4146341463414634
    },
    {
      "dataset": "WBCIC",
      "method": "B7_ALL_EXPERT_STACK",
      "positive_subject_fraction": 0.3658536585365853
    },
    {
      "dataset": "WBCIC",
      "method": "B12_CGRFUSE_R1",
      "positive_subject_fraction": 0.0
    }
  ],
  "bootstrap_CI": [
    {
      "dataset": "OpenBMI",
      "method": "B3_STRONGEST_KEEP_STACK",
      "bootstrap_CI95_L": 0.0,
      "bootstrap_CI95_U": 0.0
    },
    {
      "dataset": "OpenBMI",
      "method": "B6_LOGISTIC_ALL_EXPERT",
      "bootstrap_CI95_L": 0.021574074074074,
      "bootstrap_CI95_U": 0.0547245370370369
    },
    {
      "dataset": "OpenBMI",
      "method": "B7_ALL_EXPERT_STACK",
      "bootstrap_CI95_L": 0.0037037037037036,
      "bootstrap_CI95_U": 0.0274999999999999
    },
    {
      "dataset": "OpenBMI",
      "method": "B12_CGRFUSE_R1",
      "bootstrap_CI95_L": -0.0006481481481481,
      "bootstrap_CI95_U": 0.0017592592592592
    },
    {
      "dataset": "WBCIC",
      "method": "B3_STRONGEST_KEEP_STACK",
      "bootstrap_CI95_L": 0.0,
      "bootstrap_CI95_U": 0.0
    },
    {
      "dataset": "WBCIC",
      "method": "B6_LOGISTIC_ALL_EXPERT",
      "bootstrap_CI95_L": -0.0023540280857353,
      "bootstrap_CI95_U": 0.0194013303769401
    },
    {
      "dataset": "WBCIC",
      "method": "B7_ALL_EXPERT_STACK",
      "bootstrap_CI95_L": -0.0074439517122443,
      "bootstrap_CI95_U": 0.0101023651145602
    },
    {
      "dataset": "WBCIC",
      "method": "B12_CGRFUSE_R1",
      "bootstrap_CI95_L": 0.0,
      "bootstrap_CI95_U": 0.0
    }
  ],
  "instability_gates_verified": true,
  "action_bank_manifests": {
    "OPENBMI_FULL_LEGAL_ACTION_BANK_MANIFEST.json": {
      "schema": "CGRSTACK_OpenBMI_FULL_LEGAL_ACTION_BANK_V1",
      "dataset": "OpenBMI",
      "subjects": 54,
      "subject_ids": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
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
        "39",
        "40",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "52",
        "53",
        "54"
      ],
      "sample_count": 10800,
      "run_count_per_expert": 6,
      "expert_count": 3,
      "experts": {
        "E0": "ERM EEGNet",
        "E1": "subject-adversarial EEGNet (GRL coefficient 0.10)",
        "E2": "CORAL geometry-robust EEGNet (coefficient 0.10)"
      },
      "partition_definition": "two independent fixed hash-salted biological-subject bipartitions; each orientation predicts only the excluded group",
      "partition_salts": [
        "CGRSTACK_OpenBMI_PARTITION_R1_B0",
        "CGRSTACK_OpenBMI_PARTITION_R1_B1"
      ],
      "complete_case_filter": false,
      "S2_accessed": false,
      "outer_accessed": false,
      "rows": 194400,
      "bank_sha256": "051d24d37a2b10ada348ec28566a015206d12e7720a9797db858ddbddfa53cd4",
      "paradigm": "mi",
      "sessions": [
        1,
        2
      ],
      "samples": 10800,
      "historical_bank_subjects": 52,
      "historical_subjects_missing_from_cache": [
        "17",
        "46"
      ]
    },
    "WBCIC_S0_S1_FULL_LEGAL_ACTION_BANK_MANIFEST.json": {
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
  },
  "S2_untouched": true,
  "outer_untouched": true,
  "rerun": false
}
