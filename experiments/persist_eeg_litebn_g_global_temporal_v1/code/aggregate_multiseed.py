#!/usr/bin/env python3
"""Finalize the LiteBN-G result after the locked seed-0 sequential gate."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

REPO = Path(os.environ.get("LITEBN_G_REPO", "/root/rivermind-data/CRCICLR_G_WORK")).resolve()
OUT = REPO / "experiments/persist_eeg_litebn_g_global_temporal_v1/outputs"


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".part"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"); os.replace(tmp, path)


def main() -> None:
    seed = json.loads((OUT / "SEED0_GATE_DECISION.json").read_text()); tasks = pd.read_csv(OUT / "SEED0_TASK_RESULTS.csv")
    identity, stem, bn, selection = (pd.read_csv(OUT / name) for name in ("EPOCH0_IDENTITY_AUDIT.csv", "FROZEN_STEM_AUDIT.csv", "BN_LOCK_AUDIT.csv", "CHECKPOINT_SELECTION.csv"))
    alpha_a = selection.alpha_attn_selected.abs().max() if len(selection) else 0.0; alpha_f = selection.alpha_ffn_selected.abs().max() if len(selection) else 0.0
    executed = tasks[tasks.Executed == "YES"].set_index("task")
    def cell(task, key, default="not reached"):
        return default if task not in executed.index else executed.loc[task, key]
    terminal = seed["terminal"]
    final = {"terminal": terminal, "seed0_terminal": terminal, "epoch0_identity_pass": bool(identity["pass"].all()), "frozen_stem_pass": bool(stem["bitwise_unchanged"].all()), "bn_lock_pass": bool(bn["bitwise_unchanged"].all() and bn["all_bn_eval"].all()), "max_abs_alpha_attn": float(alpha_a), "max_abs_alpha_ffn": float(alpha_f), "epoch0_fallback_count": int(selection.epoch0_fallback.sum()), "NEW_SEALED_TEST_ACCESSED": "NO", "seed1_run": False, "seed2_run": False}
    write_json(OUT / "FINAL_LITEBN_G_DECISION.json", final)
    lines = ["# LiteBN-G final report", "", f"Terminal: `{terminal}`.", "", "| Task | Benchmark best | Benchmark model | Matched LiteBN | LiteBN-G | Delta vs LiteBN pp | Delta vs benchmark pp | Selected epochs | Epoch0 fallback count | PASS |", "|---|---:|---|---:|---:|---:|---:|---|---:|---|"]
    for _, row in tasks.iterrows():
        f = lambda value, digits=4: "—" if pd.isna(value) else f"{float(value):.{digits}f}"
        lines.append(f"| {row.task} | {f(row.benchmark_best)} | {row.benchmark_model} | {f(row.matched_LiteBN)} | {f(row.LiteBN_G)} | {f(row.delta_vs_LiteBN_pp,3)} | {f(row.delta_vs_benchmark_pp,3)} | {row.selected_epochs or '—'} | {f(row.epoch0_fallback_count,0)} | {row.PASS} |")
    ssvep_pass = bool(cell("OpenBMI_SSVEP", "PASS", False)); erp_pass = bool(cell("OpenBMI_ERP", "PASS", False)); mi_pass = bool(cell("OpenBMI_MI", "PASS", False)); wbcic_pass = bool(cell("WBCIC_MI", "PASS", False))
    lines += ["", "## Required answers", "",
              f"1. Epoch0 reproduced exact historical LiteBN: `{final['epoch0_identity_pass']}`.",
              f"2. All early-stem weights were bitwise unchanged: `{final['frozen_stem_pass']}`.",
              f"3. All BatchNorm parameters/buffers were bitwise unchanged and eval-locked: `{final['bn_lock_pass']}`.",
              f"4. Seed0 SSVEP exceeded 92.34%: `{ssvep_pass}` ({cell('OpenBMI_SSVEP','LiteBN_G')}).",
              f"5. If SSVEP failed, immediate termination occurred: `{terminal == 'SSVEP_BENCHMARK_FAIL'}`.",
              f"6. ERP exceeded 85.57%: `{erp_pass}` ({cell('OpenBMI_ERP','LiteBN_G')}).",
              f"7. OpenBMI MI exceeded 75.55%: `{mi_pass}` ({cell('OpenBMI_MI','LiteBN_G')}).",
              f"8. WBCIC exceeded 79.18%: `{wbcic_pass}` ({cell('WBCIC_MI','LiteBN_G')}).",
              f"9. WBCIC preserved or exceeded matched LiteBN: `{('WBCIC_MI' in executed.index and float(executed.loc['WBCIC_MI','LiteBN_G']) >= float(executed.loc['WBCIC_MI','matched_LiteBN']))}`.",
              f"10. Maximum selected |alpha_attn|={alpha_a:.6g}; |alpha_ffn|={alpha_f:.6g}.",
              f"11. Checkpoint selection fell back to epoch0 in {final['epoch0_fallback_count']}/{len(selection)} executed folds.",
              f"12. The global block became active: `{bool(alpha_a != 0 or alpha_f != 0)}`.",
              "13. Multiseed ran: `False` (only permitted after all seed0 task gates pass).",
              f"14. LiteBN-G is justified as final single-model candidate: `{terminal == 'LITEBN_G_FINAL_CANDIDATE_PASS'}`.",
              "", "All opened benchmark resources are labeled EXPOSED_DEVELOPMENT_BENCHMARK. NEW_SEALED_TEST_ACCESSED = NO."]
    (OUT / "FINAL_LITEBN_G_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"LITEBN_G_AGGREGATE_COMPLETE terminal={terminal}", flush=True)


if __name__ == "__main__": main()
