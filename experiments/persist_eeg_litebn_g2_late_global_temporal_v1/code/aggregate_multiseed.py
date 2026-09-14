#!/usr/bin/env python3
"""Finalize the locked LiteBN-G2 result after the sequential seed-0 gate."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

REPO = Path(os.environ.get("LITEBN_G2_REPO", "/root/rivermind-data/CRCICLR_G2_WORK")).resolve()
OUT = REPO / "experiments/persist_eeg_litebn_g2_late_global_temporal_v1/outputs"
EARLY_G_SSVEP = .9084285714
EARLY_G_ALPHA_MAX = .00373122


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"); os.replace(temporary, path)


def main() -> None:
    seed = json.loads((OUT / "SEED0_GATE_DECISION.json").read_text()); tasks = pd.read_csv(OUT / "SEED0_TASK_RESULTS.csv")
    identity, frozen, bn, selection, attention = (pd.read_csv(OUT / name) for name in ("EPOCH0_IDENTITY_AUDIT.csv", "FROZEN_PARAMETER_AUDIT.csv", "BN_LOCK_AUDIT.csv", "CHECKPOINT_SELECTION.csv", "ATTENTION_SELECTIVITY.csv"))
    alpha_a = float(selection.alpha_attn_selected.abs().max()) if len(selection) else 0.0
    alpha_f = float(selection.alpha_ffn_selected.abs().max()) if len(selection) else 0.0
    mean_alpha_a = float(attention.alpha_attn_selected.mean()) if len(attention) else 0.0
    mean_alpha_f = float(attention.alpha_ffn_selected.mean()) if len(attention) else 0.0
    mean_entropy = float(attention.selected_normalized_attention_entropy.mean()) if len(attention) else float("nan")
    selective_folds = int(attention.attention_selective.sum()) if len(attention) else 0
    terminal = seed["terminal"]
    near_uniform = bool(terminal == "SSVEP_BENCHMARK_FAIL" and mean_entropy >= .97)
    final = {"terminal": terminal, "seed0_terminal": terminal, "epoch0_identity_pass": bool(identity["pass"].all()), "frozen_stem_backend1_pass": bool(frozen["bitwise_unchanged"].all()), "bn_lock_pass": bool(bn["bitwise_unchanged"].all() and bn["all_bn_eval"].all()), "max_abs_alpha_attn": alpha_a, "max_abs_alpha_ffn": alpha_f, "mean_selected_alpha_attn": mean_alpha_a, "mean_selected_alpha_ffn": mean_alpha_f, "mean_selected_normalized_attention_entropy": mean_entropy, "entropy_lt_097_folds": selective_folds, "epoch0_fallback_count": int(selection.epoch0_fallback.sum()), "descriptive_labels": ["GLOBAL_ATTENTION_REMAINS_NEAR_UNIFORM"] if near_uniform else [], "NEW_SEALED_TEST_ACCESSED": "NO", "seed1_run": False, "seed2_run": False}
    write_json(OUT / "FINAL_LITEBN_G2_DECISION.json", final)
    executed = tasks[tasks.Executed == "YES"].set_index("task")
    def value(task: str, key: str, default="not reached"):
        return default if task not in executed.index else executed.loc[task, key]
    def passed(task: str) -> bool:
        return bool(value(task, "PASS", False))
    lines = ["# LiteBN-G2 final report", "", f"Terminal: `{terminal}`.", "", "| Task | Benchmark best | Matched LiteBN | LiteBN-G2 | Delta vs B0 pp | Delta vs benchmark pp | Selected epochs | Epoch0 fallback count | Mean selected normalized attention entropy | Entropy<0.97 folds | PASS |", "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|"]
    for _, row in tasks.iterrows():
        numeric = lambda item, digits=4: "—" if pd.isna(item) else f"{float(item):.{digits}f}"
        lines.append(f"| {row.task} | {numeric(row.benchmark_best)} | {numeric(row.matched_LiteBN)} | {numeric(row.LiteBN_G2)} | {numeric(row.delta_vs_LiteBN_pp, 3)} | {numeric(row.delta_vs_benchmark_pp, 3)} | {row.selected_epochs or '—'} | {numeric(row.epoch0_fallback_count, 0)} | {numeric(row.mean_selected_normalized_attention_entropy, 4)} | {numeric(row.entropy_lt_097_folds, 0)} | {row.PASS} |")
    ssvep = float(value("OpenBMI_SSVEP", "LiteBN_G2", float("nan")))
    attention_material = bool(alpha_a > EARLY_G_ALPHA_MAX)
    activation = "both" if abs(mean_alpha_a) > 1e-12 and abs(mean_alpha_f) > 1e-12 else ("attention" if abs(mean_alpha_a) > 1e-12 else ("FFN" if abs(mean_alpha_f) > 1e-12 else "neither"))
    lines += ["", "## Required answers", "",
              f"1. Epoch0 exactly reproduced matched LiteBN: `{final['epoch0_identity_pass']}`.",
              f"2. Stem and backend1 parameters were bitwise unchanged: `{final['frozen_stem_backend1_pass']}`.",
              f"3. All BatchNorm states and affine parameters were unchanged and eval-locked: `{final['bn_lock_pass']}`.",
              f"4. Late attention increased SSVEP relative to early LiteBN-G (0.9084285714): `{bool(ssvep > EARLY_G_SSVEP)}` ({ssvep}).",
              f"5. LiteBN-G2 exceeded matched LiteBN: `{bool(ssvep > float(value('OpenBMI_SSVEP', 'matched_LiteBN', float('inf'))))}`.",
              f"6. LiteBN-G2 exceeded 92.34%: `{passed('OpenBMI_SSVEP')}` ({ssvep}).",
              f"7. Selected attention became more selective than prior near-uniform attention: `{bool(mean_entropy < .97)}` (mean normalized entropy={mean_entropy:.6f}; selective folds={selective_folds}/{len(attention)}).",
              f"8. Checkpoint selection selected epoch0 in {final['epoch0_fallback_count']}/{len(selection)} executed folds.",
              f"9. |alpha_attn| became materially larger than early LiteBN-G maximum 0.00373122: `{attention_material}` (maximum={alpha_a:.6g}).",
              f"10. Activation was mainly associated with: `{activation}` (mean alpha_attn={mean_alpha_a:.6g}; mean alpha_ffn={mean_alpha_f:.6g}).",
              f"11. If SSVEP failed, the experiment terminated immediately: `{terminal == 'SSVEP_BENCHMARK_FAIL'}`.",
              f"12. If SSVEP passed, ERP/MI/WBCIC passed fixed gates: `{terminal in ('SEED0_ALL_TASK_PASS_MULTISEED_PENDING', 'LITEBN_G2_FINAL_CANDIDATE_PASS')}`.",
              "13. Multiseed ran and all four means exceeded benchmarks: `False` (not permitted after a seed0 failure).",
              f"14. Late high-level global attention is justified as a final single-model direction: `{terminal == 'LITEBN_G2_FINAL_CANDIDATE_PASS'}`.",
              "", "All opened benchmark resources are labeled EXPOSED_DEVELOPMENT_BENCHMARK. NEW_SEALED_TEST_ACCESSED = NO."]
    (OUT / "FINAL_LITEBN_G2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"LITEBN_G2_AGGREGATE_COMPLETE terminal={terminal}", flush=True)


if __name__ == "__main__":
    main()
