# PERSIST-EEG P2 Comprehensive Multi-seed Report

Decision: `P2_PASS_MULTI_SEED_PERSISTENCE_UTILITY`

## Table A - Per-seed raw BA

|   seed |    ERP |     MI |   SSVEP |
|-------:|-------:|-------:|--------:|
|      0 | 0.6528 | 0.757  |  0.9373 |
|      1 | 0.6479 | 0.7533 |  0.9342 |
|      2 | 0.6542 | 0.7621 |  0.9346 |
|      3 | 0.6556 | 0.7616 |  0.9331 |
|      4 | 0.6508 | 0.7534 |  0.9348 |

## Table B - Long utility (Erase U_L - Raw)

|   seed |   ERP delta UL |   MI delta UL |   SSVEP delta UL |
|-------:|---------------:|--------------:|-----------------:|
|      0 |        -0.0009 |       -0.0524 |          -0.0075 |
|      1 |        -0.0047 |       -0.0358 |          -0.0114 |
|      2 |        -0.0034 |       -0.0591 |          -0.0079 |
|      3 |        -0.0026 |       -0.0388 |          -0.011  |
|      4 |        -0.0046 |       -0.0481 |          -0.0113 |

## Table C - Medium utility (Erase U_M - Raw)

|   seed |   ERP delta UM |   MI delta UM |   SSVEP delta UM |
|-------:|---------------:|--------------:|-----------------:|
|      0 |        -0.0047 |       -0.0704 |          -0.0163 |
|      1 |        -0.007  |       -0.0452 |          -0.0165 |
|      2 |        -0.0074 |       -0.0741 |          -0.0232 |
|      3 |        -0.0059 |       -0.0533 |          -0.0252 |
|      4 |        -0.0061 |       -0.0527 |          -0.0186 |

## Table D - Random controls

| task   | target   |    mean |    std |   median |      5% |    95% |
|:-------|:---------|--------:|-------:|---------:|--------:|-------:|
| erp    | U_L      |  0      | 0.0011 |   0.0001 | -0.0018 | 0.0017 |
| erp    | U_M      |  0      | 0.001  |   0.0001 | -0.0018 | 0.0016 |
| mi     | U_L      | -0      | 0.0023 |   0      | -0.0036 | 0.0036 |
| mi     | U_M      |  0.0001 | 0.0025 |   0      | -0.0036 | 0.0041 |
| ssvep  | U_L      | -0.0001 | 0.0017 |  -0      | -0.0027 | 0.0026 |
| ssvep  | U_M      | -0.0001 | 0.0017 |  -0      | -0.0025 | 0.0025 |

## Table E - PCA controls

|   seed |   ERP delta PCA |   MI delta PCA |   SSVEP delta PCA |
|-------:|----------------:|---------------:|------------------:|
|      0 |         -0.0234 |        -0.1906 |           -0.0581 |
|      1 |         -0.0394 |        -0.1791 |           -0.0889 |
|      2 |         -0.0313 |        -0.1747 |           -0.0925 |
|      3 |         -0.0342 |        -0.1963 |           -0.0917 |
|      4 |         -0.0298 |        -0.1863 |           -0.0685 |

## Table F - Cross-session Long verification before/after erasure

|   seed | source   |   erase_UL |    raw |   delta |
|-------:|:---------|-----------:|-------:|--------:|
|      0 | erp      |     0.5382 | 0.707  | -0.1688 |
|      0 | mi       |     0.6946 | 0.8082 | -0.1136 |
|      0 | ssvep    |     0.687  | 0.7685 | -0.0815 |
|      1 | erp      |     0.5555 | 0.7075 | -0.152  |
|      1 | mi       |     0.6729 | 0.8045 | -0.1316 |
|      1 | ssvep    |     0.6801 | 0.7698 | -0.0897 |
|      2 | erp      |     0.5396 | 0.7018 | -0.1622 |
|      2 | mi       |     0.7005 | 0.7948 | -0.0944 |
|      2 | ssvep    |     0.6678 | 0.7717 | -0.1039 |
|      3 | erp      |     0.5415 | 0.7065 | -0.1649 |
|      3 | mi       |     0.6982 | 0.7835 | -0.0853 |
|      3 | ssvep    |     0.6804 | 0.7685 | -0.0881 |
|      4 | erp      |     0.5521 | 0.7189 | -0.1668 |
|      4 | mi       |     0.7113 | 0.7957 | -0.0845 |
|      4 | ssvep    |     0.6938 | 0.7748 | -0.081  |

## Table G - Medium verification

|   seed | task   |   erase_UM |    raw |   delta |
|-------:|:-------|-----------:|-------:|--------:|
|      0 | erp    |     0.5355 | 0.5418 | -0.0063 |
|      0 | mi     |     0.604  | 0.6159 | -0.0119 |
|      0 | ssvep  |     0.6173 | 0.6209 | -0.0036 |
|      1 | erp    |     0.5201 | 0.5457 | -0.0256 |
|      1 | mi     |     0.6258 | 0.6406 | -0.0148 |
|      1 | ssvep  |     0.6131 | 0.6215 | -0.0084 |
|      2 | erp    |     0.5219 | 0.5368 | -0.0149 |
|      2 | mi     |     0.6242 | 0.6393 | -0.0152 |
|      2 | ssvep  |     0.6192 | 0.6257 | -0.0066 |
|      3 | erp    |     0.5359 | 0.5485 | -0.0126 |
|      3 | mi     |     0.605  | 0.6299 | -0.0249 |
|      3 | ssvep  |     0.6163 | 0.6228 | -0.0065 |
|      4 | erp    |     0.5216 | 0.5427 | -0.0211 |
|      4 | mi     |     0.6184 | 0.6193 | -0.0009 |
|      4 | ssvep  |     0.6168 | 0.6213 | -0.0045 |

## Table H - Residualized/refit Long utility

|   seed |   ERP residual delta UL |   MI residual delta UL |   SSVEP residual delta UL |
|-------:|------------------------:|-----------------------:|--------------------------:|
|      0 |                 -0.0013 |                -0.0525 |                   -0.0067 |
|      1 |                 -0.0043 |                -0.0377 |                   -0.0118 |
|      2 |                 -0.0034 |                -0.0578 |                   -0.0078 |
|      3 |                 -0.0032 |                -0.04   |                   -0.011  |
|      4 |                 -0.0043 |                -0.0485 |                   -0.0117 |

## Table I - U_L/U_M overlap

|   seed |   U_L_rank |   U_M_rank |   normalized_overlap |
|-------:|-----------:|-----------:|---------------------:|
|      0 |        4.4 |        6   |               0.9447 |
|      1 |        5   |        6.2 |               0.9619 |
|      2 |        5.2 |        6.4 |               0.9403 |
|      3 |        5.2 |        6.4 |               0.9281 |
|      4 |        4.6 |        5.8 |               0.9611 |

## Required conclusions

1. MI Long utility is stable: mean delta=-0.0465, CI=-0.0570 to -0.0360, negative seeds=5/5.
2. SSVEP is weaker but nonzero: mean delta=-0.0098, CI=-0.0137 to -0.0061.
3. ERP is near-neutral in magnitude: mean delta=-0.0032, despite a narrow nonzero aggregate CI.
4. Real U_L erasure differs from random: Gate D=PASS.
5. U_L erasure removes Long signal: cross-session delta=-0.1108, CI=[-0.12214668556495949, -0.10008145367657698].
6. MI survives fold-local paradigm residualization/refit: delta=-0.0469, Gate E=PASS.
7. Medium status: MEDIUM_WEAK_OR_REDUNDANT.
8. U_L/U_M are not independent enough: mean normalized overlap=0.9472.
9. P2 decision: P2_PASS_MULTI_SEED_PERSISTENCE_UTILITY.

Cross-paradigm verification remains supporting-only because paradigm composition can affect identity verification. PCA erasure is a structured high-variance control, not evidence that PCA is or is not persistence.
