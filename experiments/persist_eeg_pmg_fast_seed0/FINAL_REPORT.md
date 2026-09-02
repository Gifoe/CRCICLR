# PMG fast seed-0 mechanism pilot

## Discovery summary

| Dataset | Anchor BA | ERM BA | PMG BA | PMG-anchor pp | PMG-ERM pp |
|---|---:|---:|---:|---:|---:|
| OpenBMI | 80.2222% | 78.8889% | 78.7556% | -1.4667 | -0.1333 |
| WBCIC | 79.6079% | 78.9593% | 73.4954% | -6.1125 | -5.4639 |

## Prospective mechanism

| Dataset | Anchor harm | ERM harm | PMG harm | PMG vs ERM reduction | PMG fraction harmed | PMG cosine |
|---|---:|---:|---:|---:|---:|---:|
| OpenBMI | 0.0005928 | 0.0005955 | 0.0006014 | -0.99% | 43.29% | 0.09462 |
| WBCIC | 0.0000949 | 0.0000780 | 0.0000719 | +7.77% | 38.40% | 0.22272 |

## Gate decision
{
  "ba_anchor_ok": false,
  "ba_erm_ok": false,
  "one_dataset_erm_win": false,
  "harm_reduced_both": false,
  "harm_reduction_at_least_one_10pct": false,
  "no_ba_loss_over_0_5pp_both": false,
  "harm_reduction_fraction": {
    "OpenBMI": -0.009932051422670618,
    "WBCIC": 0.07770739601564813
  }
}

## Validity
- source-only: yes; outcome indices constructed: no
- WBCIC sealed outer ten accessed: no
- OpenBMI sealed/internal cohort accessed: no
- seed 1/2 run: no
- mathematical audit: PASS

terminal = PMG_FAST_NOT_SUPPORTED
