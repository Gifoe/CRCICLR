#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from _common import config_hash, load_yaml, project_root


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="."); parser.add_argument("--config", default="configs/v3/main.yaml")
    args = parser.parse_args(); root = project_root(args.root); repo = root / "repo"; config = load_yaml(args.config)
    action = pd.read_csv(root / "outputs/v3_probecert/action_search/ACTION_CONFIG_RESULTS.csv")
    oracle = pd.read_csv(root / "outputs/v3_probecert/oracle_headroom/ORACLE_HEADROOM_SUMMARY.csv")
    gate = json.loads((root / "outputs/v3_probecert/oracle_headroom/ORACLE_GATE.json").read_text())
    selected = json.loads((root / "outputs/v3_probecert/action_search/SELECTED_ACTION_CONFIGS.json").read_text())
    delivery = repo / "delivery/v3_probecert"; delivery.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    method_freeze = {"method": "ProbeCert-V3 stage-0", "stage": "oracle_gate", "oracle_gate": gate,
                     "code_commit": commit, "main_config_sha256": config_hash(config),
                     "action_configs": selected, "created_utc": datetime.now(timezone.utc).isoformat(),
                     "future_work_authorized": bool(gate["go"])}
    (delivery / "V3_METHOD_FREEZE.json").write_text(json.dumps(method_freeze, indent=2, sort_keys=True) + "\n")
    (delivery / "V3_METHOD_SPEC.md").write_text("""# ProbeCert-V3 method specification

## Target and access protocol

ProbeCert-V3 targets subject-level selection among No-TTA and frozen test-time interventions. Each episode is split chronologically into Adapt (A), Probe (P), and unchanged Future (V). Candidate actions update on A only and then call `freeze_state()`. Unlabeled P is used only for action diagnosis. V inputs and labels are inaccessible until the policy decision, action-state hash, configuration hash, and prediction-set index have been persisted and verified by the runtime access controller.

## Action library and Stage 0

The finite action library contains No-TTA, official T3A, and a corrected residual adapter. Every action starts from the same source checkpoint and is reinitialized per subject. The adapter uses a nonnegative collapse penalty. Stage 0 screens 64 T3A and 81 adapter configurations by subject-grouped successive halving on meta-development subjects, retains at most one configuration per action/dataset/seed, and measures label-informed Safe-Oracle headroom on development V. Oracle results are an upper bound, not deployment evidence.

## Probe policy

For each frozen action, the policy measures expected-set-size gain, nuisance-augmentation consistency, three-block temporal stability, source drift/class quality, and update magnitude on P. Thresholds and deterministic tie breaking are fit only on meta-fit subjects. The deployed policy selects an eligible action or falls back to No-TTA; no P labels, calibration outcomes, or outer outcomes enter this decision.

## Joint certificate

Each calibration subject contributes one policy-level critical index: the first lambda-grid index satisfying both future-risk control at alpha and noninferiority degradation at epsilon. The split-conformal order statistic uses `ceil((m+1)(1-delta))`; if this rank exceeds the calibration sample size, the method returns the full-set sentinel. Calibration occurs after the complete policy is frozen and contributes one scalar per subject, not one row per candidate action.

## Evaluation

Five seeded subject-level outer splits contain disjoint meta-fit, calibration, and outer-evaluation roles. Hyperparameter/action search and policy fitting use meta-fit subjects; calibration freezes the joint index; outer V is opened only after decisions are persisted. HMC and EEGMMIDB are development tasks. CAP transfers the HMC policy and action configurations and recalibrates only the conformal quantile at the target site; it is external-site replication, not untouched confirmation.
""")
    (delivery / "V3_DATA_PROTOCOL.md").write_text("""# V3 data protocol

- V3 inherits V2 subject partitions and records the SHA256 of each inherited split.
- The original Future indices are preserved byte-for-value at the array level.
- Original context is split chronologically into A and P. EEGMMIDB is split at whole-run boundaries; HMC and CAP use the same chronological 1:1 rule.
- A/P/V are nonempty, pairwise disjoint, ordered, and A+P exactly reconstructs context.
- Action search and Oracle Stage 0 use only `meta_risk_train`. Calibration, final/outer evaluation, and CAP labels are excluded from selection.
- CAP remains an external-site replication because it was observed in earlier project versions; it is not an untouched confirmation set.
""")
    chosen = action[action.stage == "full"].sort_values("mean_safe_gain", ascending=False).groupby(["dataset", "seed", "action"]).head(1)
    (delivery / "V3_ACTION_SEARCH_REPORT.md").write_text("# V3 action search report\n\nThe required finite grids contained 64 T3A and 81 adapter configurations. Five sorted meta subjects were used for screening; four configurations per action then advanced to all meta subjects. Rows are grouped by subject, and one configuration per dataset/seed/action is frozen.\n\n" +
        table(chosen, ["dataset", "seed", "action", "config_id", "availability_rate", "mean_safe_gain", "positive_subject_rate", "harm_rate"]) + "\n")
    main = oracle[oracle.epsilon == float(config["epsilon"])]
    oracle_text = "# V3 Oracle headroom report\n\nInternal gate: **{}**. The gate is a development decision rule, not a statistical theorem. CI values use subject clustering after averaging repeated seeds.\n\n{}\n".format(
        "GO" if gate["go"] else "NO-GO", table(main, ["dataset", "alpha", "positive_subject_rate", "mean_gain", "relative_set_size_reduction", "relative_ci_lower", "relative_ci_upper", "maximum_single_action_positive_rate", "minimum_seed_positive_rate", "harm_rate"]))
    (delivery / "V3_ORACLE_HEADROOM_REPORT.md").write_text(oracle_text)
    limitations = """# V3 limitations

- Stage-0 Oracle outcomes use meta-development Future labels and establish only whether the action library has exploitable headroom; they are not deployable-policy results.
- Nuisance transforms are applied to cached token representations because the current frozen source head consumes CBraMod tokens. They are controlled representation-space proxies, not a claim that raw-waveform invariance was verified end to end.
- Repeated seeds reuse some subjects; inference must average seed results per subject before bootstrap.
- HMC and EEGMMIDB remain development tasks. CAP is an observed external-site replication, not untouched confirmation.
- The policy-level conformal result is marginal over exchangeable subjects and episodes; it is not an individual-subject or action-conditional guarantee.
"""
    (delivery / "V3_LIMITATIONS.md").write_text(limitations)
    readiness = "# V3 ICLR readiness assessment\n\nDecision: **{}**.\n\n".format("CONTINUE AFTER ORACLE GO" if gate["go"] else "NO-GO")
    readiness += ("The action library clears the internal Stage-0 headroom gate. This is necessary but insufficient; nested policy evaluation, policy-level calibration, baselines, ablations, simulations, and external-site replication remain required.\n" if gate["go"] else
                  "The action library does not clear the predeclared Oracle headroom gate. Developing a more complex selector would be scientifically unjustified because even a label-informed Safe Oracle lacks reliable utility under the stated constraints.\n")
    (delivery / "V3_ICLR_READINESS_ASSESSMENT.md").write_text(readiness)
    if not gate["go"]:
        (delivery / "ORACLE_HEADROOM_NO_GO.md").write_text("# Oracle headroom NO-GO\n\n" + oracle_text + "\nSelector development stopped as required.\n")
    provenance = "# Provenance\n\n- Base branch: `v2-joint-risk-benefit`\n- V3 branch: `v3-probecert-policy-crc`\n- Code commit at Stage-0 freeze: `{}`\n- All source checkpoints are enumerated in `outputs/v3_probecert/source_models/SOURCE_MODEL_MANIFEST.json`.\n- Episode and split validation hashes are in `outputs/v3_probecert/provenance/`.\n".format(commit)
    (delivery / "PROVENANCE.md").write_text(provenance)
    artifacts = []
    commands = {"episodes": "python scripts/v3/build_v3_episodes.py --root . --config configs/v3/episode_protocol.yaml",
                "action_search": "python scripts/v3/run_action_search.py --root . --config configs/v3/action_search.yaml --device cuda --resume",
                "oracle_headroom": "python scripts/v3/run_oracle_headroom.py --root . --config configs/v3/main.yaml --device cuda --resume"}
    for path in sorted((root / "outputs/v3_probecert").rglob("*")):
        if path.is_file():
            section = path.relative_to(root / "outputs/v3_probecert").parts[0]
            artifacts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path),
                              "generation_command": commands.get(section, "generated by V3 pipeline")})
    (root / "outputs/v3_probecert/ARTIFACT_MANIFEST.json").write_text(json.dumps({"artifacts": artifacts}, indent=2, sort_keys=True) + "\n")
    print({"gate": "GO" if gate["go"] else "NO-GO", "delivery": str(delivery), "artifacts": len(artifacts)})


if __name__ == "__main__": main()
