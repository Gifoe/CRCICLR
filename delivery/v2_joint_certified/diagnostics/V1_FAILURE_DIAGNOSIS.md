# V1 failure diagnosis

| dataset   |   beneficial_headroom |   safe_beneficial |   oracle_gain |   safe_gain |   certified_gain |     hsc_gain |     regret |   no_tta_error |
|:----------|----------------------:|------------------:|--------------:|------------:|-----------------:|-------------:|-----------:|---------------:|
| cap       |              0.383784 |         0.0310811 |     0.0380293 |  0.00118243 |       0.00132883 | -0.000123874 | 0.00130631 |       0.369741 |
| eegmmidb  |              0.410526 |         0         |     0.0173246 |  0          |       0          |  0           | 0          |       0.747851 |
| hmc       |              0.446154 |         0.2       |     0.0256731 |  0.0063141  |       0.00932692 | -0.00766026  | 0.0139744  |       0.420833 |

Interpretation rules: oracle minus safe-oracle gain estimates loss to the risk constraint; safe-oracle minus certified-oracle gain estimates certificate conservatism; certified-oracle minus HSC gain estimates selector weakness. Low No-TTA quality, especially EEGMMIDB, is a source-model qualification failure rather than a selector result.
