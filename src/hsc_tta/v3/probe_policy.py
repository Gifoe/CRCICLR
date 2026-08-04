from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import pandas as pd


@dataclass(frozen=True)
class ProbeThresholds:
    tau_class: float
    tau_update: float
    tau_set: float
    tau_aug_margin: float
    tau_positive_blocks: float
    tau_time_mad: float
    tau_drift: float

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class ProbePolicy:
    required = {"action", "action_available", "r_class", "normalized_update_magnitude", "g_set", "g_aug",
                "positive_probe_block_fraction", "temporal_mad", "d_src", "action_cost"}

    def __init__(self, thresholds: ProbeThresholds):
        self.thresholds = thresholds

    def decide(self, diagnostics: pd.DataFrame) -> dict[str, object]:
        missing = self.required - set(diagnostics)
        if missing:
            raise ValueError(f"missing Probe fields: {sorted(missing)}")
        if "no_tta" in set(diagnostics.action):
            raise ValueError("Probe candidates contain only intervention actions")
        t = self.thresholds; frame = diagnostics.copy()
        gates = {
            "unavailable": ~frame.action_available.astype(bool),
            "class_collapse": frame.r_class < t.tau_class,
            "update_magnitude": frame.normalized_update_magnitude > t.tau_update,
            "set_gain": frame.g_set < t.tau_set,
            "augmentation": frame.g_aug < -t.tau_aug_margin,
            "temporal_blocks": frame.positive_probe_block_fraction < t.tau_positive_blocks,
            "temporal_mad": frame.temporal_mad > t.tau_time_mad,
            "source_drift": frame.d_src > t.tau_drift,
        }
        eligible = pd.Series(True, index=frame.index)
        reasons = []
        for index in frame.index:
            failed = [name for name, mask in gates.items() if bool(mask.loc[index])]
            reasons.append("eligible" if not failed else ";".join(failed)); eligible.loc[index] = not failed
        frame["gate_reason"] = reasons; accepted = frame[eligible]
        if accepted.empty:
            return {"selected_action": "no_tta", "intervention": False,
                    "selection_reason": "no_eligible_intervention", "threshold_hash": t.config_hash,
                    "gate_reasons": dict(zip(frame.action, frame.gate_reason))}
        row = accepted.sort_values(["g_set", "d_src", "action_cost", "action"],
                                   ascending=[False, True, True, True], kind="stable").iloc[0]
        return {"selected_action": str(row.action), "intervention": True,
                "selection_reason": "probe_eligible_max_set_gain", "threshold_hash": t.config_hash,
                "gate_reasons": dict(zip(frame.action, frame.gate_reason))}
