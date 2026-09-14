"""Locked LiteBN-G2 seed-0 task gates."""
from __future__ import annotations


TASK_ORDER = ("OpenBMI_SSVEP", "OpenBMI_ERP", "OpenBMI_MI", "WBCIC_MI")


def gate(task: str, result: dict) -> tuple[bool, str]:
    g = float(result["LiteBN_G2_BA"])
    if task == "OpenBMI_SSVEP": return g > .9234, "SSVEP_PASS" if g > .9234 else "SSVEP_BENCHMARK_FAIL"
    if task == "OpenBMI_ERP": return g > .8557, "ERP_PASS" if g > .8557 else "ERP_BENCHMARK_FAIL"
    if task == "OpenBMI_MI": return g > .7555, "OPENBMI_MI_PASS" if g > .7555 else "OPENBMI_MI_BENCHMARK_FAIL"
    if task == "WBCIC_MI":
        if g <= .7918: return False, "WBCIC_BENCHMARK_FAIL"
        if g < float(result["matched_LiteBN_BA"]): return False, "WBCIC_BENCHMARK_PASS_PROTECTION_FAIL"
        return True, "WBCIC_PASS"
    raise ValueError(task)
