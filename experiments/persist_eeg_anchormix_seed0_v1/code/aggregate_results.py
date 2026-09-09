"""Regenerate the compact AnchorMix result tables from completed subject rows."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


REPO = Path(os.environ.get("R2EEG_REPO", Path(__file__).resolve().parents[3])).resolve()
OUT = REPO / "experiments" / "persist_eeg_anchormix_seed0_v1" / "outputs"


def main() -> None:
    subject = pd.read_csv(OUT / "SUBJECT_METRICS.csv")
    if len(subject) != 71 or set(subject.dataset) != {"OpenBMI", "WBCIC"}:
        raise RuntimeError("expected the completed 40-subject OpenBMI and 31-subject WBCIC result rows")

    folds, summaries = [], []
    for dataset in ("OpenBMI", "WBCIC"):
        ds = subject[subject.dataset == dataset]
        best = "EEGNet" if ds.EEGNet_BA.mean() >= ds.LiteBN_BA.mean() else "LiteBN"
        for fold, group in ds.groupby("fold"):
            row = {"Dataset": dataset, "Fold": int(fold)}
            for name in ("EEGNet", "LiteBN", "REF50", "AnchorMix"):
                row[name] = float(group[f"{name}_BA"].mean())
                row[f"{name}_macro_F1"] = float(group[f"{name}_macro_F1"].mean())
            row["Delta_vs_best_pp"] = (row["AnchorMix"] - max(row["EEGNet"], row["LiteBN"])) * 100
            row["Delta_vs_REF50_pp"] = (row["AnchorMix"] - row["REF50"]) * 100
            folds.append(row)

        best_ba = float(ds[f"{best}_BA"].mean())
        row = {
            "Dataset": dataset,
            "EEGNet": float(ds.EEGNet_BA.mean()),
            "LiteBN": float(ds.LiteBN_BA.mean()),
            "REF50": float(ds.REF50_BA.mean()),
            "AnchorMix": float(ds.AnchorMix_BA.mean()),
            "BestSingle": best,
        }
        row["DeltaBest_pp"] = (row["AnchorMix"] - best_ba) * 100
        row["DeltaREF50_pp"] = (row["AnchorMix"] - row["REF50"]) * 100
        for name in ("EEGNet", "LiteBN", "REF50", "AnchorMix"):
            row[f"{name}_macro_F1"] = float(ds[f"{name}_macro_F1"].mean())
        summaries.append(row)

    fold_df = pd.DataFrame(folds).sort_values(["Dataset", "Fold"])
    summary_df = pd.DataFrame(summaries)
    fold_df.to_csv(OUT / "FOLD_METRICS.csv", index=False)
    summary_df.to_csv(OUT / "DATASET_SUMMARY.csv", index=False)

    ds_ok = bool((summary_df.DeltaBest_pp >= 0).all())
    ref_gap_ok = bool((summary_df.DeltaREF50_pp >= -0.5).all())
    very_strong = bool((summary_df.DeltaREF50_pp >= 0).all())
    terminal = "VERY_STRONG" if very_strong else "STRONG" if ds_ok and ref_gap_ok else "PROMISING" if ds_ok else "MIXED" if bool((summary_df.DeltaBest_pp >= 0).any()) else "FAIL"

    lines = [
        "# AnchorMix-EEG seed-0 result",
        "",
        f"Final terminal: **{terminal}**",
        "",
        "| Dataset | EEGNet | LiteBN | REF50 | AnchorMix | DeltaBest | DeltaREF50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(f"| {row.Dataset} | {row.EEGNet * 100:.2f}% | {row.LiteBN * 100:.2f}% | {row.REF50 * 100:.2f}% | {row.AnchorMix * 100:.2f}% | {row.DeltaBest_pp:+.2f} pp | {row.DeltaREF50_pp:+.2f} pp |")
    lines += [
        "",
        "Macro-F1 for every method is in `DATASET_SUMMARY.csv` and `FOLD_METRICS.csv`.",
        "",
        "## Fold metrics",
        "",
        "| Dataset | Fold | EEGNet | LiteBN | REF50 | AnchorMix | Delta vs best single | Delta vs REF50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fold_df.itertuples(index=False):
        lines.append(f"| {row.Dataset} | {row.Fold} | {row.EEGNet * 100:.2f}% | {row.LiteBN * 100:.2f}% | {row.REF50 * 100:.2f}% | {row.AnchorMix * 100:.2f}% | {row.Delta_vs_best_pp:+.2f} pp | {row.Delta_vs_REF50_pp:+.2f} pp |")
    lines += [
        "",
        "1. AnchorMix exceeds the strongest fixed single model on both datasets: " + ("YES" if ds_ok else "NO"),
        "2. AnchorMix is within 0.5 pp of REF50 on both datasets: " + ("YES" if ref_gap_ok else "NO"),
        "3. OpenBMI/WBCIC direction is consistent: " + ("YES" if ds_ok else "NO"),
        "4. Parameter count: see `MODEL_COST.json`.",
        "5. Next step 3 seeds + ERP/SSVEP: " + ("CONSIDER" if terminal in {"VERY_STRONG", "STRONG"} else "NO; stop."),
        "",
        "AnchorMix uses one latent representation and one classifier (`self.head`); it has no prediction-level fusion or auxiliary prediction head.",
    ]
    (OUT / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(terminal)


if __name__ == "__main__":
    main()
