from __future__ import annotations

import math

import numpy as np
import pandas as pd


SENTINEL = 20


def _experiment(repetitions: int, m: int, delta: float, actions: int, correlation: float,
                harm_rate: float, site_shift: float, rng: np.random.Generator) -> dict[str, float]:
    probe = rng.normal(size=(repetitions, m, actions)); noise = rng.normal(size=probe.shape)
    future = correlation * probe + np.sqrt(max(0.0, 1 - correlation ** 2)) * noise
    harm = rng.random(probe.shape) < harm_rate
    indices = np.clip(np.rint(10 - 2 * future), 0, SENTINEL - 1).astype(np.int16)
    indices[harm] = SENTINEL
    selected = probe.argmax(2); intervene = probe.max(2) > .5
    policy_indices = np.take_along_axis(indices, selected[..., None], axis=2)[..., 0]
    policy_indices = np.where(intervene, policy_indices, 10)
    k = math.ceil((m + 1) * (1 - delta))
    if k > m:
        j_star = np.full(repetitions, SENTINEL, dtype=np.int16); insufficient = 1.0
    else:
        j_star = np.partition(policy_indices, k - 1, axis=1)[:, k - 1]; insufficient = 0.0
    test_probe = rng.normal(size=(repetitions, actions)); test_noise = rng.normal(size=test_probe.shape)
    test_future = correlation * test_probe + np.sqrt(max(0.0, 1 - correlation ** 2)) * test_noise - site_shift
    test_harm = rng.random(test_probe.shape) < min(.99, harm_rate + .25 * site_shift)
    test_indices = np.clip(np.rint(10 - 2 * test_future), 0, SENTINEL - 1).astype(np.int16); test_indices[test_harm] = SENTINEL
    test_selected = test_probe.argmax(1); test_intervene = test_probe.max(1) > .5
    test_policy = test_indices[np.arange(repetitions), test_selected]; test_policy = np.where(test_intervene, test_policy, 10)
    action_k = math.ceil((m + 1) * (1 - delta / max(actions, 1)))
    if action_k > m: actionwise = np.full(repetitions, SENTINEL)
    else: actionwise = np.partition(indices, action_k - 1, axis=1)[:, action_k - 1, :].max(1)
    selected_future = test_future[np.arange(repetitions), test_selected]
    no_information_gain = np.where(test_intervene, selected_future, 0.0).mean()
    return {"joint_validity": float(np.mean(test_policy <= j_star)), "nominal_validity": 1 - delta,
            "validity_gap": float(np.mean(test_policy <= j_star) - (1 - delta)),
            "sentinel_probability": float(np.mean(j_star == SENTINEL)), "calibration_insufficient_rate": insufficient,
            "policy_efficiency": float(np.mean(1 - j_star / SENTINEL)),
            "actionwise_efficiency": float(np.mean(1 - actionwise / SENTINEL)),
            "policy_minus_actionwise_efficiency": float(np.mean(actionwise - j_star) / SENTINEL),
            "intervention_rate": float(test_intervene.mean()), "test_harm_rate": float(test_harm[np.arange(repetitions), test_selected].mean()),
            "selected_future_utility": float(np.where(test_intervene, selected_future, 0.0).mean()),
            "no_information_apparent_gain": float(no_information_gain if correlation == 0 else np.nan)}


def run_simulations(config: dict[str, object]) -> pd.DataFrame:
    repetitions = int(config["repetitions"]); rng = np.random.default_rng(int(config["seed"])); rows = []
    for m in config["calibration_sizes"]:
        for delta in config["deltas"]:
            for actions in config["action_counts"]:
                for rho in config["probe_future_correlations"]:
                    result = _experiment(repetitions, int(m), float(delta), int(actions), float(rho), .15, 0.0, rng)
                    rows.append({"scenario":"exchangeable_grid","repetitions":repetitions,"m":m,"delta":delta,
                                 "actions":actions,"probe_future_correlation":rho,"harm_rate":.15,"site_shift":0.0,**result})
    for harm in config["harm_rate_sensitivity"]:
        result=_experiment(repetitions,30,.1,3,.5,float(harm),0.0,rng)
        rows.append({"scenario":"harm_sensitivity","repetitions":repetitions,"m":30,"delta":.1,"actions":3,
                     "probe_future_correlation":.5,"harm_rate":harm,"site_shift":0.0,**result})
    for shift in config["site_shift_sensitivity"]:
        result=_experiment(repetitions,30,.1,3,.5,.15,float(shift),rng)
        rows.append({"scenario":"site_shift","repetitions":repetitions,"m":30,"delta":.1,"actions":3,
                     "probe_future_correlation":.5,"harm_rate":.15,"site_shift":shift,**result})
    return pd.DataFrame(rows)

