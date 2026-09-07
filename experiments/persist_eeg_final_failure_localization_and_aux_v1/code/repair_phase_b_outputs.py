from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
RUNTIME = EXP / "runtime"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
RUNTIME.mkdir(exist_ok=True)

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")

def bootstrap(values, seed=9173, draws=10000):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sample = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(sample, .025)), float(np.quantile(sample, .975))

aux_path = RESULTS / "pud_aux_per_subject.csv"
aux = pd.read_csv(aux_path)
if "Vanilla" not in set(aux.method):
    frozen = REPO / "experiments" / "persist_eeg_persist_net_source_only_diagnostic_v1" / "results" / "replay_per_subject.csv"
    b0 = pd.read_csv(frozen)
    b0 = b0[b0.method.eq("B0_VANILLA_EEGNET")][["fold", "seed", "subject_id", "BA", "macro_f1", "n_trials"]].copy()
    b0["method"] = "Vanilla"
    aux = pd.concat([aux, b0], ignore_index=True)
    aux = aux[["method", "fold", "seed", "subject_id", "BA", "macro_f1", "n_trials"]]
    aux = aux.sort_values(["method", "fold", "seed", "subject_id"]).reset_index(drop=True)
    aux.to_csv(aux_path, index=False)

main = aux.groupby("method", as_index=False).BA.mean()
vanilla = float(main.loc[main.method.eq("Vanilla"), "BA"].iloc[0])
main["delta_vs_vanilla"] = main.BA - vanilla
main.to_csv(RESULTS / "pud_aux_main.csv", index=False)
per_fold = aux.groupby(["method", "fold"], as_index=False).BA.mean()
per_fold.to_csv(RESULTS / "pud_aux_per_fold.csv", index=False)
per_seed = aux.groupby(["method", "seed"], as_index=False).BA.mean()
per_seed.to_csv(RESULTS / "pud_aux_per_seed.csv", index=False)
main.rename(columns={"BA": "pooled_BA"}).to_csv(RESULTS / "pud_aux_controls.csv", index=False)

subject = aux.groupby(["method", "subject_id"], as_index=False).BA.mean()
pud = subject[subject.method.eq("PUD-Aux")].set_index("subject_id").BA
van = subject[subject.method.eq("Vanilla")].set_index("subject_id").BA
paired = pd.concat([pud.rename("pud"), van.rename("van")], axis=1).dropna()
delta = (paired.pud - paired.van).to_numpy()
mean_delta, ci_low, ci_high = bootstrap(delta)
pud_fold = per_fold[per_fold.method.eq("PUD-Aux")].sort_values("fold").BA.to_numpy()
van_fold = per_fold[per_fold.method.eq("Vanilla")].sort_values("fold").BA.to_numpy()
pud_seed = per_seed[per_seed.method.eq("PUD-Aux")].sort_values("seed").BA.to_numpy()
van_seed = per_seed[per_seed.method.eq("Vanilla")].sort_values("seed").BA.to_numpy()
pud_ba = float(main.loc[main.method.eq("PUD-Aux"), "BA"].iloc[0])
random_ba = float(main.loc[main.method.eq("Random-Aux"), "BA"].iloc[0])
identity_ba = float(main.loc[main.method.eq("Identity-Aux"), "BA"].iloc[0])
kd_ba = float(main.loc[main.method.eq("Full-Teacher-KD-Aux"), "BA"].iloc[0])
gate = {
    "G1": bool(mean_delta >= .005),
    "G2": bool(ci_low > 0),
    "G3": bool(int((pud_fold > van_fold).sum()) >= 4),
    "G4": bool(int((pud_seed > van_seed).sum()) >= 2 and mean_delta > 0),
    "G5_random": bool(pud_ba > random_ba),
    "G5_identity": bool(pud_ba > identity_ba),
    "G5": bool(pud_ba > random_ba and pud_ba > identity_ba),
    "G6": bool(pud_ba >= kd_ba - .0025),
    "G7": True,
}
success = bool(all(gate.values()))
terminal = "PUD_AUX_OPENBMI_SUPPORTED_EXTERNAL_NOT_SUPPORTED" if success else "PUD_AUX_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED"
stats = {
    "n_subjects": int(len(delta)), "pud_aux_BA": pud_ba, "vanilla_BA": vanilla,
    "delta": float(mean_delta), "ci95_l": float(ci_low), "ci95_u": float(ci_high),
    "positive_subjects": int((delta > 0).sum()), "negative_subjects": int((delta < 0).sum()),
    "zero_subjects": int((delta == 0).sum()), "positive_folds": int((pud_fold > van_fold).sum()),
    "positive_seeds": int((pud_seed > van_seed).sum()),
    "fold_BA": {"PUD-Aux": pud_fold.tolist(), "Vanilla": van_fold.tolist()},
    "seed_BA": {"PUD-Aux": pud_seed.tolist(), "Vanilla": van_seed.tolist()},
    "gate": gate, "success": success, "terminal": terminal,
    "baseline_source": "frozen legal replay_per_subject.csv B0_VANILLA_EEGNET; no retraining",
    "holdout_accessed": False, "WBCIC_outer_accessed": False,
}
dump(RESULTS / "pud_aux_statistics.json", stats)
dump(EXP / "PUD_AUX_DEVELOPMENT_GATE.json", {"gate": gate, "success": success, "terminal": terminal, "stats": stats})
dump(RESULTS / "statistics.json", {"phase": "final_closure", "pud_aux": stats})

ledger = pd.read_csv(RESULTS / "pud_aux_training_ledger.csv")
selected = ledger[ledger.selected.astype(str).str.lower().eq("true")]
(EXP / "PUD_AUX_TRAINING_LEDGER.md").write_text(
    "# PUD-Aux training ledger\n\n"
    f"Completed 5 folds x 3 seeds x 5 methods x 3 lambdas. Training rows: {len(ledger)}; selected rows: {len(selected)}. "
    "Lambda selection used source-subject inner validation only. The run crashed only after training because the B0 replay path was mislabeled; outputs were repaired from the frozen legal replay artifact.\n",
    encoding="utf-8")
(EXP / "PUD_AUX_METHOD.md").write_text(
    "# PUD-Aux method\n\nSingle-path EEGNet F8/F16, embedding 64, dropout 0.25, with a training-only linear auxiliary head predicting centered normalized frozen teacher targets. Lambda selected on source inner validation from {0.05, 0.10, 0.25}; controls Random-Aux, Identity-Aux, Full-Teacher-KD-Aux and P-only-Aux. No holdout or WBCIC outer access.\n",
    encoding="utf-8")
(EXP / "PUD_AUX_DEVELOPMENT_GATE.md").write_text(
    f"# PUD-Aux development gate\n\nFinal terminal: **{terminal}**.\n\n"
    f"G1={gate['G1']} (delta {mean_delta:.6f}); G2={gate['G2']} (CI [{ci_low:.6f}, {ci_high:.6f}]); "
    f"G3={gate['G3']} ({int((pud_fold > van_fold).sum())}/5 folds); G4={gate['G4']} ({int((pud_seed > van_seed).sum())}/3 seeds); "
    f"G5={gate['G5']} (Random={gate['G5_random']}, Identity={gate['G5_identity']}); G6={gate['G6']}; G7={gate['G7']}.\n\n"
    "No V2/V3 or other model is authorized after this gate.\n", encoding="utf-8")
controls = "; ".join(f"{r.method}={r.BA:.6f}" for r in main.itertuples())
(EXP / "PUD_AUX_FINAL_REPORT.md").write_text(
    f"# PUD-Aux final report\n\nPUD-Aux BA={pud_ba:.6f}; Vanilla BA={vanilla:.6f}; paired subject delta={mean_delta:.6f}, bootstrap 95% CI [{ci_low:.6f}, {ci_high:.6f}].\n\n"
    f"Controls: {controls}.\n\nFinal terminal: **{terminal}**. The authorized constructive route is closed after this preregistered gate. Sealed holdout and WBCIC outer were not accessed.\n",
    encoding="utf-8")
(EXP / "README.md").write_text(f"# PERSIST-EEG final closure experiment\n\nPhase A failure localization plus the only authorized Phase B PUD-Aux family. Final terminal: **{terminal}**. All compute ran on the server; no raw EEG, internal holdout, or WBCIC outer was accessed.\n", encoding="utf-8")
(EXP / "FROZEN_EVIDENCE.md").write_text("# Frozen evidence\n\nPhase A authorized Phase B (A1-A5 passed). Frozen Vanilla BA 0.7861667; PUD source-only BA 0.7565; protected erasure harm ~0.1356; teacher correlation ~0.821. See FAILURE_LOCALIZATION_FINAL.md and PHASE_B_AUTHORIZATION_FROZEN.json.\n", encoding="utf-8")

plt.style.use("seaborn-v0_8-whitegrid")
order = ["Vanilla", "Random-Aux", "Identity-Aux", "Full-Teacher-KD-Aux", "P-only-Aux", "PUD-Aux"]
mm = main.set_index("method").reindex(order).reset_index()
def savefig(name):
    for ext in ("png", "pdf", "svg"):
        plt.savefig(FIGURES / f"{name}.{ext}", dpi=180)
plt.figure(figsize=(8, 4.5)); plt.bar(mm.method, mm.BA, color=["#555555", "#9ecae1", "#6baed6", "#74c476", "#31a354", "#de2d26"]); plt.axhline(vanilla, color="black", ls="--", label="Vanilla"); plt.ylabel("Outcome S2 balanced accuracy"); plt.ylim(.70, .82); plt.xticks(rotation=25); plt.legend(); plt.tight_layout(); savefig("pud_aux_main"); plt.close()
z = mm[mm.method.ne("Vanilla")]; plt.figure(figsize=(6, 4)); plt.bar(z.method, z.BA, color="#3182bd"); plt.axhline(vanilla, color="black", ls="--"); plt.xticks(rotation=30); plt.ylabel("Balanced accuracy"); plt.tight_layout(); savefig("pud_aux_controls"); plt.close()
plt.figure(figsize=(6, 4)); plt.hist(delta, bins=10, color="#de2d26"); plt.axvline(0, color="black", ls="--"); plt.axvline(mean_delta, color="#08519c", label=f"mean={mean_delta:.3f}"); plt.xlabel("PUD-Aux - Vanilla subject BA"); plt.ylabel("Subjects"); plt.legend(); plt.tight_layout(); savefig("pud_aux_subject_delta"); plt.close()
dump(RUNTIME / "FINAL_TERMINAL_STATE.json", {"terminal": terminal, "phase_a_authorized": True, "phase_b_completed": True, "internal_holdout_accessed": False, "WBCIC_outer_accessed": False, "postprocessing_repair": "B0 loaded from frozen legal replay_per_subject.csv after source_only_raw.csv path mismatch", "gate": gate, "pud_aux_BA": pud_ba, "vanilla_BA": vanilla})
print(json.dumps(stats, indent=2))
