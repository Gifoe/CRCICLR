from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import common as c


def main() -> None:
    lock = c.read_json(c.PROTOCOL / "DATA_ACCESS_LOCK.json")
    if lock.get("pass") is not True or lock.get("S2_or_S3_adaptation_utility_inspected") is not False:
        raise RuntimeError("data audit is not complete and outcome-clean")
    data = c.load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("competence audit requires the server GPU")
    raw = torch.from_numpy(np.asarray(data.x)).to(device)
    rows: list[dict] = []
    for backbone in c.BACKBONES:
        for fold in c.FOLDS:
            target_subjects = c.roles(fold)["outcome"]
            s1_indices = c.row_indices(data.metadata, target_subjects, (0,))
            for seed in c.SEEDS:
                model, mean, std, unit_protocol = c.load_anchor(backbone, fold, seed, device)
                extracted = c.extract(model, raw, data.metadata, s1_indices, mean, std)
                initial_weight = model.head.weight.detach().float().cpu().numpy()
                initial_bias = model.head.bias.detach().float().cpu().numpy()
                for subject in target_subjects:
                    mask = extracted["subjects"].astype(str) == subject
                    features = extracted["features"][mask]
                    labels = extracted["labels"][mask]
                    anchor_logits = extracted["logits"][mask]
                    train_idx, val_idx = c.chronological_class_split(labels)
                    anchor_val = c.metrics(labels[val_idx], anchor_logits[val_idx])
                    anchor_confidence = float(c.softmax_np(anchor_logits[val_idx]).max(axis=1).mean())
                    for lr in c.LR_GRID:
                        adapted = c.adapt_head(
                            features,
                            labels,
                            train_idx,
                            val_idx,
                            initial_weight,
                            initial_bias,
                            lr,
                            c.stable_seed("SCAA-S1-head", backbone, fold, seed, subject, lr),
                        )
                        adapted_val = c.metrics(labels[val_idx], adapted["logits"][val_idx])
                        rows.append(
                            {
                                "backbone": backbone,
                                "fold": fold,
                                "seed": seed,
                                "subject_id": subject,
                                "lr": lr,
                                "S1_train_rows": len(train_idx),
                                "S1_validation_rows": len(val_idx),
                                "anchor_S1_validation_BA": anchor_val["BA"],
                                "adapted_S1_validation_BA": adapted_val["BA"],
                                "S1_validation_BA_delta": adapted_val["BA"] - anchor_val["BA"],
                                "adapted_S1_validation_macro_F1": adapted_val["macro_F1"],
                                "adapted_S1_validation_NLL": adapted_val["NLL"],
                                "anchor_S1_validation_confidence": anchor_confidence,
                                "prediction_change_rate": float((adapted["logits"][val_idx].argmax(1) != anchor_logits[val_idx].argmax(1)).mean()),
                                "centered_logit_change_L2": float(np.linalg.norm((adapted["logits"][val_idx] - anchor_logits[val_idx]) - (adapted["logits"][val_idx] - anchor_logits[val_idx]).mean(axis=1, keepdims=True), axis=1).mean()),
                                "parameter_relative_change": adapted["parameter_relative_change"],
                                "best_epoch": adapted["best_epoch"],
                                "target_seen_by_anchor": False,
                                "S2_or_S3_accessed": False,
                            }
                        )
                del model
                torch.cuda.empty_cache()
                print(f"[competence] {backbone} fold={fold} seed={seed}", flush=True)
    frame = pd.DataFrame(rows)
    c.write_csv(c.RUNTIME / "ADAPTATION_COMPETENCE_CELLS.csv", frame)
    summary = (
        frame.groupby("lr", as_index=False)
        .agg(
            cells=("subject_id", "size"),
            anchor_S1_validation_BA=("anchor_S1_validation_BA", "mean"),
            adapted_S1_validation_BA=("adapted_S1_validation_BA", "mean"),
            S1_validation_BA_delta=("S1_validation_BA_delta", "mean"),
            adapted_S1_validation_NLL=("adapted_S1_validation_NLL", "mean"),
            prediction_change_rate=("prediction_change_rate", "mean"),
            parameter_relative_change=("parameter_relative_change", "mean"),
            median_best_epoch=("best_epoch", "median"),
        )
    )
    collapse = frame.assign(catastrophic=frame.S1_validation_BA_delta < -0.10).groupby("lr").catastrophic.mean()
    summary["catastrophic_fraction"] = summary.lr.map(collapse)
    selected = summary.sort_values(
        ["adapted_S1_validation_BA", "S1_validation_BA_delta", "adapted_S1_validation_NLL", "lr"],
        ascending=[False, False, True, True],
    ).iloc[0]
    competence = bool(
        selected.adapted_S1_validation_BA >= 0.60
        and selected.S1_validation_BA_delta >= 0.005
        and selected.prediction_change_rate >= 0.01
        and selected.parameter_relative_change > 1e-4
        and selected.catastrophic_fraction <= 0.10
    )
    c.write_csv(c.RESULTS / "ADAPTATION_COMPETENCE_GRID.csv", summary)
    selection = {
        "schema": "SCAA_STAGE0_ADAPTATION_RECIPE_SELECTION_V1",
        "selected_before_S2_or_S3_utility": True,
        "S2_or_S3_utility_accessed": False,
        "adapter": "classifier_head_only_supervised",
        "feature_encoder_frozen": True,
        "lr_candidates": list(c.LR_GRID),
        "selected_lr": float(selected.lr),
        "weight_decay": c.WEIGHT_DECAY,
        "maximum_epochs": c.MAX_EPOCHS,
        "minimum_epochs": c.MIN_EPOCHS,
        "patience": c.PATIENCE,
        "S1_split": "within-class chronological first 70% train / final 30% validation",
        "checkpoint_rule": "highest S1-validation BA, then lower NLL, then earlier epoch",
        "global_recipe_across_subjects_backbones_seeds": True,
        "cells_per_candidate": int(selected.cells),
        "selected_summary": selected.to_dict(),
        "competence_gate_pass": competence,
        "last_block_switch_used": False,
    }
    c.write_json(c.PROTOCOL / "ADAPTATION_RECIPE_SELECTION.json", selection)
    report = f"""# Adaptation competence audit

Only anchor/source information and target S1 were accessed. S2 and S3 were not
loaded. The tested intervention was supervised classifier-head-only adaptation
from the out-of-fold ERM checkpoint. The encoder and all normalization state
were frozen.

Global LR candidates were `{list(c.LR_GRID)}`. Selection used mean target S1
validation BA across 41 subjects, both primary backbones, and three matched
anchor seeds; ties used lower NLL then smaller LR. The frozen S1 split was the
first 70% versus final 30% within each class in cache chronology.

Selected LR: `{float(selected.lr):.6g}`. Mean anchor/adapted S1-validation BA:
`{float(selected.anchor_S1_validation_BA):.5f}` / `{float(selected.adapted_S1_validation_BA):.5f}`;
delta `{float(selected.S1_validation_BA_delta):+.5f}`. Mean prediction-change
rate `{float(selected.prediction_change_rate):.5f}`; mean relative head change
`{float(selected.parameter_relative_change):.5f}`; catastrophic fraction
`{float(selected.catastrophic_fraction):.5f}`.

Competence terminal: `{'HEAD_ONLY_COMPETENCE_PASS' if competence else 'HEAD_ONLY_NO_HEADROOM'}`.
This is not evidence of S2-to-S3 utility transfer.
"""
    c.write_text(c.EXP / "ADAPTATION_COMPETENCE_AUDIT.md", report)
    ledger = f"""# Adaptation iteration ledger

## V0 head-only supervised adaptation

- Diagnosed requirement: test an ordinary target-S1 adaptation action without
  introducing a selector or new architecture.
- Proposed before S2/S3: freeze the encoder and adapt only the two-class head;
  select one global LR from `{list(c.LR_GRID)}` using S1 validation only.
- Predicted competence signature: nonzero parameter/prediction change, mean S1
  validation improvement >=0.5 pp, mean adapted BA >=0.60, and catastrophic
  fraction <=0.10.
- Actual S1-only result: selected LR `{float(selected.lr):.6g}`, mean BA delta
  `{float(selected.S1_validation_BA_delta):+.5f}`, competence pass `{competence}`.
- Decision: `{'freeze head-only recipe; no last-block repair' if competence else 'head-only has no objective headroom; one pre-outcome last-block repair is required'}`.
- S2/S3 utility inspected: NO.
"""
    c.write_text(c.EXP / "ADAPTATION_ITERATION_LEDGER.md", ledger)
    print(f"SCAA_STAGE0_ADAPTATION_COMPETENCE_COMPLETE pass={competence} selected_lr={float(selected.lr):.6g}", flush=True)


if __name__ == "__main__":
    main()

