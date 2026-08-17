"""Publication figures generated only from frozen compact result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT, RESULTS


FIG = OUT / "figures"
ORDER = ["EEGNet", "FBCNet", "EEGConformer", "DeepConvNet", "TeCh"]
BLOCKS = ["P01_04", "P05_08", "P09_16", "P17_32"]


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def conceptual() -> None:
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.axis("off")
    labels = ["Persistence", "Signed utility", "Decision dependence", "Actionability", "Action"]
    for index, label in enumerate(labels):
        x = index / (len(labels) - 1)
        ax.text(x, 0.52, label, ha="center", va="center", fontsize=11, bbox={"boxstyle": "round,pad=.45", "fc": "#e8f1f8", "ec": "#315b7d"})
        if index:
            ax.annotate("", (x - 0.11, 0.52), (x - 0.20, 0.52), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(0.5, 0.08, "Each arrow is an empirical gate; failure implies NO-OP, not nuisance", ha="center", fontsize=10)
    save(fig, "figure1_persist_conceptual_chain")


def competence(backbones: pd.DataFrame) -> None:
    frame = backbones.set_index("Backbone").loc[ORDER]
    x = np.arange(len(frame))
    mean = frame["Best_task_BA"].to_numpy(float)
    low = frame["Task_BA_CI_L"].to_numpy(float)
    high = frame["Task_BA_CI_U"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    color = ["#3a6f8f" if bool(value) else "#b85c5c" for value in frame["Competence"]]
    ax.bar(x, mean, color=color, width=.7)
    ax.errorbar(x, mean, yerr=np.vstack((mean - low, high - mean)), fmt="none", color="black", capsize=4)
    ax.axhline(.60, color="#555", ls="--", lw=1, label="mean-BA gate")
    ax.axhline(.50, color="#999", ls=":", lw=1, label="chance")
    ax.set_xticks(x, ORDER, rotation=15)
    ax.set_ylim(.45, 1.0)
    ax.set_ylabel("Unseen-subject S3 balanced accuracy")
    ax.legend(frameon=False)
    save(fig, "figure2_backbone_task_competence")


def gate_matrix(blocks: pd.DataFrame) -> None:
    frame = blocks.set_index(["Backbone", "Block"])
    gates = ["H1", "H2", "H3", "H4", "H5"]
    values, labels = [], []
    for backbone in ORDER:
        for block in BLOCKS:
            row = frame.loc[(backbone, block)]
            values.append([float(bool(row[value])) for value in gates])
            labels.append(f"{backbone} · {block}")
    fig, ax = plt.subplots(figsize=(6.2, 10))
    image = ax.imshow(values, cmap=matplotlib.colors.ListedColormap(["#eeeeee", "#2e7d67"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(gates)), gates)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title("Prospective gate matrix")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, "PASS" if value else "—", ha="center", va="center", fontsize=7, color="white" if value else "#777")
    save(fig, "figure3_cross_backbone_gate_matrix")


def utility_decision(blocks: pd.DataFrame) -> None:
    frame = blocks[blocks["Competence"].astype(bool)].copy()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {name: plt.cm.tab10(index) for index, name in enumerate(ORDER)}
    markers = {"P01_04": "o", "P05_08": "s", "P09_16": "^", "P17_32": "D"}
    for _, row in frame.iterrows():
        ax.scatter(row["u_spec"], row["Finite_decision_ratio"], s=90 if row["Globally_qualified_actionable"] else 55, color=colors[row["Backbone"]], marker=markers[row["Block"]], edgecolor="black" if row["Protected"] else "none", linewidth=.9)
    ax.axvline(0, color="#777", lw=1)
    ax.axhline(1, color="#777", lw=1)
    ax.set_xlabel("Specific signed utility (u_spec; harmful ←)")
    ax.set_ylabel("Finite decision-dependence ratio")
    ax.set_title("Utility is not decision dependence")
    for backbone in ORDER:
        ax.scatter([], [], color=colors[backbone], label=backbone)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, "figure4_utility_vs_decision_dependence")


def funnel(blocks: pd.DataFrame) -> None:
    new = blocks[blocks["Backbone"] != "EEGNet"]
    counts = [
        int(new["H1"].astype(bool).sum()),
        int((new["H1"].astype(bool) & new["H2"].astype(bool)).sum()),
        int((new["H1"].astype(bool) & new["H2"].astype(bool) & new["H3"].astype(bool)).sum()),
        int((new["H1"].astype(bool) & new["H2"].astype(bool) & new["H3"].astype(bool) & new["H4"].astype(bool)).sum()),
        int(new["Preliminary_actionable"].astype(bool).sum()),
        int(new["Globally_qualified_actionable"].astype(bool).sum()),
    ]
    names = ["Persistent", "+ harmful", "+ decision active", "+ actionable", "+ stable", "+ global FWER"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(np.arange(len(names)), counts, color=plt.cm.Blues(np.linspace(.4, .85, len(names))))
    ax.set_yticks(np.arange(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("New backbone × block candidates")
    ax.set_xlim(0, 16.5)
    for index, value in enumerate(counts):
        ax.text(value + .2, index, str(value), va="center")
    save(fig, "figure5_selection_funnel")


def action_matrix(blocks: pd.DataFrame) -> None:
    frame = blocks.set_index(["Backbone", "Block"])
    code = {"NO_OP": 0, "PRESERVE": 1, "REPLICATION_REQUIRED": 2, "SUPPRESS": 2}
    values = []
    for backbone in ORDER:
        row = []
        for block in BLOCKS:
            action = str(frame.loc[(backbone, block), "Action"])
            row.append(code.get(action, 0))
        values.append(row)
    cmap = matplotlib.colors.ListedColormap(["#eeeeee", "#4c78a8", "#d95f02"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.imshow(values, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(4), BLOCKS)
    ax.set_yticks(range(5), ORDER)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, ["NO-OP", "PRESERVE", "SUPPRESS*"][value], ha="center", va="center", fontsize=8, color="white" if value else "#555")
    ax.set_title("Final representation × block action matrix\n*SUPPRESS requires completed replication/authorization")
    save(fig, "figure8_final_action_matrix")


def main() -> None:
    backbones = pd.read_csv(RESULTS / "MASTER_BACKBONE_RESULTS.csv")
    blocks = pd.read_csv(RESULTS / "MASTER_BLOCK_RESULTS.csv")
    conceptual()
    competence(backbones)
    gate_matrix(blocks)
    utility_decision(blocks)
    funnel(blocks)
    action_matrix(blocks)
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
