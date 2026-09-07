from __future__ import annotations

import argparse

import pandas as pd
import torch

import specialist_train as s


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("model", choices=s.MODELS); args = parser.parse_args(); model = args.model; device = torch.device("cuda")
    for dataset in s.DATASETS:
        rows = []
        for config in s.CONFIGS[model]:
            for fold in s.FOLDS:
                rows.append(s.train(model, dataset, config, fold, 0, s.checkpoint_path(model, dataset, config["id"], fold, 0), device))
        grid = pd.DataFrame(rows).groupby("config", as_index=False).agg(mean_validation_BA=("validation_BA", "mean"), mean_validation_NLL=("validation_NLL", "mean"), minimum_fold_BA=("validation_BA", "min"))
        selected = grid.sort_values(["mean_validation_BA", "mean_validation_NLL", "minimum_fold_BA", "config"], ascending=[False, True, False, True]).iloc[0]
        config = next(value for value in s.CONFIGS[model] if value["id"] == selected["config"])
        print(f"[pretrain-selection] {model} {dataset} {config['id']} BA={selected['mean_validation_BA']:.6f}", flush=True)
        for fold in s.FOLDS:
            for seed in (1, 2): s.train(model, dataset, config, fold, seed, s.checkpoint_path(model, dataset, config["id"], fold, seed), device)
    print(f"SPECIALIST_PRETRAIN_COMPLETE {model}", flush=True)


if __name__ == "__main__":
    main()
