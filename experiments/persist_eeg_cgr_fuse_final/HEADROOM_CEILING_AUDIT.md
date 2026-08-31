# Headroom ceiling audit

Oracle rows use labels only as a diagnostic. H2 is restricted to non-unanimous KEEP votes. H4/H5 use per-action ensemble evidence or a per-sample convex oracle and are not deployable.

| dataset   | oracle                    |       BA |   delta_vs_STRONGEST_KEEP |   residual_headroom |   sample_fraction |   oracle_rescuable_fraction_in_region | OUTER_TEST_USED   |
|:----------|:--------------------------|---------:|--------------------------:|--------------------:|------------------:|--------------------------------------:|:------------------|
| OpenBMI   | H0_ALL                    | 0.883077 |                0.0238462  |          0.0238462  |          1        |                             0.0238462 | False             |
| OpenBMI   | H1_I003_REGION            | 0.883077 |                0.0238462  |          0.0238462  |          0.292308 |                             0.0238462 | False             |
| OpenBMI   | H2_NONUNANIMOUS           | 0.883077 |                0.0238462  |          0.0238462  |          0.292308 |                             0.0238462 | False             |
| OpenBMI   | H3_CONTINUOUS_INSTABILITY | 0.855769 |               -0.00346154 |         -0.00346154 |          1        |                             0.0280769 | False             |
| OpenBMI   | H4_ACTION_ENSEMBLE        | 0.883077 |                0.0238462  |          0.0238462  |          1        |                             0.0238462 | False             |
| OpenBMI   | H5_CONVEX_ORACLE          | 0.883077 |                0.0238462  |          0.0238462  |          1        |                             0.0238462 | False             |
| WBCIC     | H0_ALL                    | 0.805814 |                0.0395208  |          0.0395208  |          1        |                             0.0395218 | False             |
| WBCIC     | H1_I003_REGION            | 0.804717 |                0.0384233  |          0.0384233  |          0.384484 |                             0.038424  | False             |
| WBCIC     | H2_NONUNANIMOUS           | 0.804717 |                0.0384233  |          0.0384233  |          0.384484 |                             0.038424  | False             |
| WBCIC     | H3_CONTINUOUS_INSTABILITY | 0.768845 |                0.00255112 |          0.00255112 |          1        |                             0.0393999 | False             |
| WBCIC     | H4_ACTION_ENSEMBLE        | 0.805204 |                0.0389111  |          0.0389111  |          1        |                             0.0401317 | False             |
| WBCIC     | H5_CONVEX_ORACLE          | 0.805204 |                0.0389111  |          0.0389111  |          1        |                             0.0401317 | False             |