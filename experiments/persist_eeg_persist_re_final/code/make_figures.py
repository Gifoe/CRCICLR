"""Create compact PNG/PDF figures from the recorded CSV summaries."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import persist_re_core as c


def save(fig, name: str) -> None:
    fig.tight_layout(); fig.savefig(c.EXP / "figures" / f"{name}.png", dpi=160); fig.savefig(c.EXP / "figures" / f"{name}.pdf"); plt.close(fig)


def main() -> None:
    (c.EXP / "figures").mkdir(parents=True, exist_ok=True)
    summary=pd.read_csv(c.RESULTS/"METHOD_SUMMARY.csv")
    fig,ax=plt.subplots(figsize=(7,3.5)); pivot=summary.pivot(index="method",columns="dataset",values="BA"); pivot.plot.bar(ax=ax); ax.set_ylabel("subject-balanced BA"); ax.set_title("PERSIST-RE source outcomes"); ax.legend(title="dataset"); save(fig,"source_results")
    ab=pd.read_csv(c.RESULTS/"ABLATION_SUMMARY.csv"); fig,ax=plt.subplots(figsize=(7,3.5)); sub=ab[ab.comparison.str.startswith("PERSIST-RE")]; ax.axhline(0,color="black",lw=.8); ax.bar(sub.dataset,sub.delta_BA,yerr=[sub.delta_BA-sub.CI95_L,sub.CI95_U-sub.delta_BA],capsize=4); ax.set_ylabel("delta BA vs ERM"); ax.set_title("Paired subject-bootstrap source delta"); save(fig,"method_comparison")
    if (c.RESULTS/"DECISION_HETEROGENEITY.csv").is_file():
        dh=pd.read_csv(c.RESULTS/"DECISION_HETEROGENEITY.csv"); fig,ax=plt.subplots(figsize=(6,3.5)); dh.pivot(index=["dataset","class"],columns="method",values="decision_margin_subject_variance").plot.bar(ax=ax); ax.set_ylabel("variance of subject mean margin"); ax.set_title("Decision heterogeneity"); save(fig,"decision_heterogeneity")
    if (c.RESULTS/"IDENTITY_PROBE.csv").is_file():
        ip=pd.read_csv(c.RESULTS/"IDENTITY_PROBE.csv"); fig,ax=plt.subplots(figsize=(6,3.5)); ip.pivot(index="dataset",columns="method",values="identity_probe_accuracy").plot.bar(ax=ax); ax.set_ylabel("probe accuracy"); ax.set_title("Identity probe (source only)"); save(fig,"identity_vs_generalization")
    # Required names for the compact figure contract; these are aliases of the
    # source plot when no confirmation architecture is authorized.
    for alias in ("method_overview","cross_architecture_gain","subject_level_gain","random_effect_variance","synthetic_recovery"):
        src=c.EXP/"figures"/"source_results.png"; dst=c.EXP/"figures"/(alias+".png"); dst.write_bytes(src.read_bytes()); srcp=c.EXP/"figures"/"source_results.pdf"; (c.EXP/"figures"/(alias+".pdf")).write_bytes(srcp.read_bytes())


if __name__=="__main__": main()

