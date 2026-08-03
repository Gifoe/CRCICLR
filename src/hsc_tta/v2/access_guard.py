from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


TAINTED_FILENAMES = frozenset({
    "ALL_COUNTERFACTUAL_ACTION_OUTCOMES.parquet",
    "ALL_SUBJECT_DECISIONS.parquet",
    "final_counterfactual_action_outcomes",
    "final_test_outcomes",
    "joined_decisions",
})


class OldFinalAccessGuard:
    """Program-level gate around already-observed v1 final-test artifacts."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.freeze = self.root / "outputs" / "v2_joint_certified" / "freeze" / "V2_METHOD_FREEZE.json"

    @staticmethod
    def is_tainted(path: str | Path) -> bool:
        value = Path(path)
        return any(part in TAINTED_FILENAMES for part in value.parts) or value.name in TAINTED_FILENAMES

    def assert_access(self, path: str | Path, *, purpose: str) -> None:
        if not self.is_tainted(path):
            return
        if purpose == "v1_oracle_diagnostic":
            return
        if purpose == "exploratory_replication" and self.freeze.is_file():
            payload = json.loads(self.freeze.read_text(encoding="utf-8"))
            if payload.get("methods_frozen") is True:
                return
        raise PermissionError(
            f"old final-test access denied for purpose={purpose!r}; "
            "only the one-time oracle diagnostic or post-freeze exploratory replication is allowed"
        )

    def read_parquet(self, path: str | Path, *, purpose: str) -> pd.DataFrame:
        self.assert_access(path, purpose=purpose)
        return pd.read_parquet(path)
