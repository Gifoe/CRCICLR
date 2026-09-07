# Measurement repair audit

The V2 candidate/random decision-dependence asymmetry was repaired. Both paths now use `D(u)=mean(abs(center(z_erased_u)-center(z_raw)))`, with `center(z)=z-mean_class(z)`. See `results/DECISION_METRIC_AUDIT.json`.
