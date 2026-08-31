"""Generate compact, result-only figures (no runtime/checkpoint material)."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pda_core as c


def savefig(fig, stem: str) -> None:
    out = c.EXP / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=160, bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = c.RESULTS / "SOURCE_PER_SUBJECT.csv"
    if not out.is_file():
        return
    frame = pd.read_csv(out)
    # Keep visualizations deliberately small and free of trial-level traces.
    summary = frame.groupby(["dataset", "method"], as_index=False).BA.mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    for ds, g in summary.groupby("dataset"):
        ax.bar([f"{ds}:{m}" for m in g.method], g.BA, label=ds)
    ax.tick_params(axis="x", rotation=70); ax.set_ylabel("subject-balanced BA"); ax.set_title("PERSIST-PDA source outcome")
    savefig(fig, "source_results")
    components = pd.read_csv(c.RESULTS / "ADAPTER_COMPONENTS.csv")
    fig, ax = plt.subplots(figsize=(5, 4)); ax.scatter(components.persistent_norm, components.transient_norm, s=10, alpha=.65)
    ax.set_xlabel("persistent adapter norm"); ax.set_ylabel("transient adapter norm"); ax.set_title("Persistent vs transient")
    savefig(fig, "persistent_vs_transient")
    mech = frame[frame.method.isin(["correct_adapter", "wrong_adapter", "shuffled_adapter"])]
    sm = mech.groupby(["dataset", "method"], as_index=False).BA.mean()
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar([f"{d}:{m}" for d,m in zip(sm.dataset,sm.method)], sm.BA); ax.tick_params(axis="x",rotation=65); ax.set_title("Correct / wrong / shuffled")
    savefig(fig, "correct_wrong_shuffled")
    # Required figure stems that are not applicable because the gate stopped
    # before a future or cross-backbone evaluation are explicit empty panels.
    for stem, title in [("method_overview", "PERSIST-PDA source-only method"), ("future_session_gain", "Future session sealed"), ("stability_vs_future_gain", "Future gain unavailable"), ("cross_backbone_gain", "EEGNeX not opened")]:
        fig, ax = plt.subplots(figsize=(5, 3)); ax.text(.5,.5,title,ha="center",va="center"); ax.set_axis_off(); savefig(fig, stem)


if __name__ == "__main__":
    main()
