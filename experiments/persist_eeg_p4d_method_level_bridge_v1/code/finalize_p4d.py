from __future__ import annotations

import pandas as pd

import p4d_common as c


def main() -> None:
    validation_path = c.RESULTS / "P4D_FINAL_VALIDATION.json"
    validation = c.read_json(validation_path)
    if validation.get("pass") is not True:
        raise RuntimeError("final report is blocked until core validator pass=true")
    p4c = c.read_json(c.P4C / "results" / "P4C_SAFETY_FINAL_VALIDATION.json")
    auth = c.read_json(c.EXP / "P4D_AUTHORIZATION.json")
    canonical = c.read_json(c.EXP / "CANONICAL_INVARIANCE_CONFIGS.json")
    protocol = c.read_json(c.EXP / "P4D_PROTOCOL_FROZEN.json")
    completion = c.read_json(c.RESULTS / "P4D_S6_CANONICAL_TRAINING_COMPLETE.json")
    summary = c.read_json(c.RESULTS / "bridge_model_summary.json")
    burden = pd.read_csv(c.RESULTS / "P4D_SOURCE_UNSAFE_BURDEN.csv")
    inventory = pd.read_csv(c.RESULTS / "invariance_grid_inventory.csv")
    per_setting = pd.read_csv(c.RESULTS / "per_setting_bridge.csv")
    per_method = pd.read_csv(c.RESULTS / "per_method_bridge.csv")
    coefficients = summary["coefficients"]
    cis = summary["coefficient_CIs"]
    slopes = summary["simple_slopes"]
    head = summary["headroom"]
    burden_summary = burden.groupby("setting_id").R_unsafe.agg(["min", "median", "mean", "max"]).reset_index()
    method_configs = {row["method"]: row for row in canonical["methods"]}
    status_counts = inventory.groupby(["setting_id", "status"]).size().rename("cells").reset_index()
    setting_points = {row.setting_id: row.beta_zSI_x_Runsafe for row in per_setting.itertuples()}
    method_points = {row.method: row.beta_zSI_x_Runsafe for row in per_method.itertuples()}
    report = f"""# P4D Final Report — Mechanism-to-Method Bridge

## Validated terminal

`{summary['P4D_terminal']}`

`P4E_MODEL_AUTHORIZATION = {summary['P4E_MODEL_AUTHORIZATION']}`

This is a constrained result. Only DANN passed the frozen source-side manipulation competence gate. Therefore P4D cannot reach strong support, regardless of the numerical interaction, because the prespecified two-method replication gate fails. P4E is independently blocked because P4C was partial rather than strong.

## Required answers

1. **P4C exact safety terminal:** `{p4c['SAFETY_STATUS']}`.
2. **Why P4D was authorized:** conditional authorization; pooled DeltaRegime={auth['pooled_DeltaRegime_BA']:+.9f}, pooled U_high={auth['pooled_U_high_BA']:+.9f}, S4/S6 directions agree, and purity passes.
3. **P4C validator:** PASS.
4. **P4C low-E actionability:** `{p4c['LOW_E_ACTIONABILITY_STATUS']}`.
5. **Exact R_unsafe:** count(High-I AND High-E) / count(High-I), within frozen P4C ERM setting/fold/seed assignments.
6. **S4/S6 R_unsafe distribution:** shown below; thresholds were source-only frozen.
7. **Historically observed outcomes:** S1/S2/S3 grids and ERM competence outcomes; inventory records every cell.
8. **Trained but sealed before P4D:** all 135 S4 non-ERM cells, 70 partial S5 cells, and then only frozen S6 canonical DANN cells.
9. **Untrained:** remaining S5 and all noncanonical S6 method/lambda cells.
10. **Manipulation-competent methods:** `{', '.join(summary['competent_methods'])}` only.
11. **Canonical lambdas:** DANN={method_configs['DANN']['lambda_star']}; MMD and CORAL have no canonical lambda because they failed competence.
12. **Lambda selection purity:** yes; S4 source identity reduction only, no future BA/F1/CE.
13. **S6 training added:** yes, canonical-only.
14. **New training runs:** {completion['new_training_runs']} (not 135).
15. **Exact z_SI:** `{protocol['identity_normalization']}`.
16. **beta z_SI:** {coefficients['beta_z_SI']:+.9f}.
17. **beta R_unsafe:** {coefficients['beta_R_unsafe']:+.9f}.
18. **beta interaction:** {coefficients['beta_zSI_x_Runsafe']:+.9f}.
19. **Interaction 95% CI:** [{cis['beta_zSI_x_Runsafe'][0]:+.9f}, {cis['beta_zSI_x_Runsafe'][1]:+.9f}].
20. **slope_low:** {slopes['slope_low']:+.9f}.
21. **slope_high:** {slopes['slope_high']:+.9f}.
22. **DeltaSlope_bridge:** {slopes['DeltaSlope_bridge']:+.9f}, CI [{slopes['DeltaSlope_CI'][0]:+.9f}, {slopes['DeltaSlope_CI'][1]:+.9f}].
23. **S4 bridge direction:** beta interaction {setting_points.get('S4', float('nan')):+.9f}.
24. **S6 bridge direction:** beta interaction {setting_points.get('S6', float('nan')):+.9f}.
25. **DANN result:** primary competent method; beta interaction {method_points.get('DANN', float('nan')):+.9f}.
26. **MMD result:** identity-manipulation incompetent under the frozen rule; excluded from primary outcome evaluation.
27. **CORAL result:** identity-manipulation incompetent under the frozen rule; excluded from primary outcome evaluation.
28. **Cross-method consistency:** not established; only one competent method, so G5 fails.
29. **Low-unsafe DeltaG_BA:** {head['low_unsafe_DeltaG_BA']:+.9f}, CI [{head['low_unsafe_CI'][0]:+.9f}, {head['low_unsafe_CI'][1]:+.9f}].
30. **High-unsafe DeltaG_BA:** {head['high_unsafe_DeltaG_BA']:+.9f}, CI [{head['high_unsafe_CI'][0]:+.9f}, {head['high_unsafe_CI'][1]:+.9f}].
31. **HeadroomContrast:** {head['HeadroomContrast']:+.9f}, CI [{head['HeadroomContrast_CI'][0]:+.9f}, {head['HeadroomContrast_CI'][1]:+.9f}].
32. **Canonical headroom method:** `{head['method']}`, selected by source identity suppression before outcomes.
33. **P4D terminal:** `{summary['P4D_terminal']}`.
34. **P4E model authorization:** `{summary['P4E_MODEL_AUTHORIZATION']}`.
35. **Sealed outer holdouts:** untouched; OpenBMI internal holdout untouched and WBCIC outer 10 unenumerated.
36. **Outcome-driven protocol modification:** none.
37. **P4A 405-grid:** remained paused; no mechanical restart.
38. **Scientific interpretation:** the data test whether run-level task-entangled identity burden moderates global invariance. The result is limited to the competent DANN manipulation and cannot establish a method-general bridge. The exact sign and uncertainty above determine whether even that narrow bridge is partial or unsupported; no selective-invariance model is justified.

## Unsafe burden summary

{c.markdown_table(burden_summary, digits=9)}

## Grid inventory summary

{c.markdown_table(status_counts)}

## Prospective setting bridge

{c.markdown_table(per_setting, digits=9)}

## Prospective method bridge

{c.markdown_table(per_method, digits=9)}

## Gate audit

{c.markdown_table(pd.DataFrame([{'gate': key, 'pass': value} for key, value in summary['gates'].items()]))}

## Final limitation

The bridge is scientifically underpowered at the method level because two of three standard methods did not manipulate identity reliably in the frozen S4 audit. Treating their non-effect as usable suppression would be invalid. Expanding S6 training or changing lambdas after seeing outcomes would not repair that identification problem; it would introduce selection bias.
"""
    c.write_text(c.EXP / "P4D_FINAL_REPORT.md", report)
    print("P4D_FINAL_REPORT_WRITTEN_AFTER_CORE_VALIDATOR_PASS")


if __name__ == "__main__":
    main()
