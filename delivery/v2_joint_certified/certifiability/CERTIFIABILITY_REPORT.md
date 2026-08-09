# Certifiability and sample-size audit

Counts above the observed calibration budget are residual-bootstrap projections, not new independent subjects. Action counts 5 and 10 use conservative multiplicity stress scaling while evaluating the observed three-action library; no synthetic TTA utility is claimed.

| Dataset | alpha | m | q | CSR | fallback | positive TTA | joint validity |
|---|---:|---:|---:|---:|---:|---:|---:|
| eegmmidb | 0.10 | 10 | 1.559 | 0.316 | 0.684 | 0.000 | 1.000 |
| eegmmidb | 0.10 | 15 | 1.462 | 0.354 | 0.646 | 0.000 | 0.998 |
| eegmmidb | 0.10 | 20 | 1.346 | 0.424 | 0.576 | 0.001 | 0.996 |
| eegmmidb | 0.10 | 25 | 1.430 | 0.379 | 0.621 | 0.000 | 0.997 |
| eegmmidb | 0.10 | 30 | 1.325 | 0.429 | 0.571 | 0.000 | 0.996 |
| eegmmidb | 0.10 | 50 | 1.326 | 0.423 | 0.577 | 0.001 | 0.996 |
| eegmmidb | 0.10 | 75 | 1.380 | 0.404 | 0.596 | 0.000 | 0.998 |
| eegmmidb | 0.10 | 100 | 1.277 | 0.456 | 0.544 | 0.001 | 0.997 |
| eegmmidb | 0.20 | 10 | 1.599 | 0.539 | 0.461 | 0.000 | 0.999 |
| eegmmidb | 0.20 | 15 | 1.509 | 0.578 | 0.422 | 0.000 | 0.999 |
| eegmmidb | 0.20 | 20 | 1.395 | 0.638 | 0.362 | 0.001 | 0.996 |
| eegmmidb | 0.20 | 25 | 1.473 | 0.592 | 0.408 | 0.000 | 0.998 |
| eegmmidb | 0.20 | 30 | 1.384 | 0.631 | 0.369 | 0.000 | 0.997 |
| eegmmidb | 0.20 | 50 | 1.379 | 0.642 | 0.358 | 0.000 | 0.998 |
| eegmmidb | 0.20 | 75 | 1.385 | 0.636 | 0.364 | 0.000 | 0.997 |
| eegmmidb | 0.20 | 100 | 1.324 | 0.668 | 0.332 | 0.000 | 0.996 |
| hmc | 0.10 | 10 | 1.579 | 0.135 | 0.865 | 0.000 | 0.983 |
| hmc | 0.10 | 15 | 1.551 | 0.156 | 0.844 | 0.000 | 0.978 |
| hmc | 0.10 | 20 | 1.452 | 0.212 | 0.788 | 0.001 | 0.975 |
| hmc | 0.10 | 25 | 1.505 | 0.173 | 0.827 | 0.001 | 0.980 |
| hmc | 0.10 | 30 | 1.367 | 0.240 | 0.760 | 0.001 | 0.974 |
| hmc | 0.10 | 50 | 1.366 | 0.241 | 0.759 | 0.001 | 0.972 |
| hmc | 0.10 | 75 | 1.354 | 0.220 | 0.780 | 0.000 | 0.976 |
| hmc | 0.10 | 100 | 1.328 | 0.221 | 0.779 | 0.000 | 0.974 |
| hmc | 0.20 | 10 | 1.506 | 0.211 | 0.789 | 0.001 | 0.988 |
| hmc | 0.20 | 15 | 1.472 | 0.226 | 0.774 | 0.002 | 0.980 |
| hmc | 0.20 | 20 | 1.354 | 0.265 | 0.735 | 0.004 | 0.979 |
| hmc | 0.20 | 25 | 1.438 | 0.226 | 0.774 | 0.001 | 0.986 |
| hmc | 0.20 | 30 | 1.332 | 0.286 | 0.714 | 0.005 | 0.975 |
| hmc | 0.20 | 50 | 1.280 | 0.288 | 0.712 | 0.004 | 0.977 |
| hmc | 0.20 | 75 | 1.293 | 0.290 | 0.710 | 0.003 | 0.979 |
| hmc | 0.20 | 100 | 1.290 | 0.295 | 0.705 | 0.003 | 0.981 |

## Interpretation

The current m=12/14 nested calibration folds are sufficient for conservative validity but not for certifying positive adaptation: the empirical positive-TTA rate is zero. The requested m=15/20/25 range improves risk CSR in bootstrap projections but cannot solve weak U-to-benefit predictability by sample size alone.