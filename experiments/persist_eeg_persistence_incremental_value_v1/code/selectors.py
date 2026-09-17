"""Frozen rank-matched selector rules; no evaluation data are accepted here."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Choice:
    block_ids: tuple[int, ...]
    coordinates: tuple[int, ...]
    score: float

    @property
    def rank(self) -> int:
        return len(self.coordinates)


def _coordinates(blocks: list[list[int]], ids: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(i) for block_id in ids for i in blocks[block_id]}))


def exact_rank_choice(blocks: list[list[int]], scores: list[float], budget: int) -> Choice:
    """Exact 0/1 rank-knapsack with lexicographic block-ID tie breaking."""
    if len(blocks) != len(scores):
        raise ValueError("block/score count mismatch")
    if budget < 0 or budget > sum(map(len, blocks)):
        raise ValueError("invalid rank budget")
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for block_id, (block, score) in enumerate(zip(blocks, scores)):
        width = len(block)
        if not width or width > 4:
            raise ValueError("candidate block rank must be 1..4")
        for old_rank, (old_score, old_ids) in sorted(list(states.items()), reverse=True):
            rank = old_rank + width
            if rank > budget:
                continue
            candidate = (old_score + float(score), old_ids + (block_id,))
            incumbent = states.get(rank)
            if incumbent is None or candidate[0] > incumbent[0] or (
                candidate[0] == incumbent[0] and candidate[1] < incumbent[1]
            ):
                states[rank] = candidate
    if budget not in states:
        raise RuntimeError(f"exact-rank selection infeasible: {budget}")
    score, ids = states[budget]
    coordinates = _coordinates(blocks, ids)
    if len(coordinates) != budget:
        raise RuntimeError("candidate blocks overlap or rank is inconsistent")
    return Choice(ids, coordinates, score)


def choose(spec: dict[str, Any], assignment: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose PU/U/P solely from TRAIN spectrum and TRAIN erasure evidence."""
    blocks = [list(map(int, block)) for block in spec["blocks"]]
    if len(blocks) != len(assignment) or len(blocks) != len(spec["support"]):
        raise ValueError("candidate block mismatch")
    pu_ids = tuple(i for i, row in enumerate(assignment) if row["protected"])
    for i, row in enumerate(assignment):
        if int(row["block"]) != i or int(row["dimensions"]) != len(blocks[i]):
            raise ValueError("utility evidence/block mismatch")
        gate = (bool(spec["support"][i]["persistence_supported"])
                and float(row["absolute_CI_low"]) > 0
                and float(row["excess_CI_low"]) > 0)
        if bool(row["protected"]) != gate:
            raise ValueError("PU differs from formal Protected union")
    pu = Choice(pu_ids, _coordinates(blocks, pu_ids), 0.0)
    k = pu.rank
    if k == 0:
        return {"PU_empty": True, "k": 0, "PU": pu, "U_only": None, "P_only": None,
                "utility_scores": [], "persistence_scores": []}
    utility_scores = [min(float(row["absolute_CI_low"]), float(row["excess_CI_low"])) for row in assignment]
    persistence_scores = [float(s["rho"]) - float(s["null_p95"]) for s in spec["support"]]
    u = exact_rank_choice(blocks, utility_scores, k)
    p = exact_rank_choice(blocks, persistence_scores, k)
    if not pu.rank == u.rank == p.rank == k:
        raise RuntimeError("rank match failed")
    return {"PU_empty": False, "k": k, "PU": pu, "U_only": u, "P_only": p,
            "utility_scores": utility_scores, "persistence_scores": persistence_scores}


def self_test() -> None:
    blocks = [[0, 1], [2], [3, 4], [5]]
    assert exact_rank_choice(blocks, [1, 2, 4, 2], 3).block_ids == (1, 2)
    assert exact_rank_choice(blocks, [0, 0, 0, 0], 3).block_ids == (0, 1)
    assert exact_rank_choice(blocks, [-1, -2, -3, -4], 3).rank == 3
    assert exact_rank_choice(blocks, [1, 2, 3, 4], 0).block_ids == ()
    spec = {"blocks": blocks, "support": [
        {"persistence_supported": False, "rho": 0.1, "null_p95": 0.2},
        {"persistence_supported": True, "rho": 0.5, "null_p95": 0.2},
        {"persistence_supported": True, "rho": 0.4, "null_p95": 0.2},
        {"persistence_supported": False, "rho": 0.0, "null_p95": 0.2},
    ]}
    evidence = [
        {"block": 0, "dimensions": 2, "absolute_CI_low": 1.0, "excess_CI_low": 1.0, "protected": False},
        {"block": 1, "dimensions": 1, "absolute_CI_low": 2.0, "excess_CI_low": 2.0, "protected": True},
        {"block": 2, "dimensions": 2, "absolute_CI_low": 3.0, "excess_CI_low": 3.0, "protected": True},
        {"block": 3, "dimensions": 1, "absolute_CI_low": 4.0, "excess_CI_low": 4.0, "protected": False},
    ]
    result = choose(spec, evidence)
    assert result["k"] == 3 and result["PU"].block_ids == (1, 2)
    assert result["U_only"].block_ids == (2, 3)
    assert result["P_only"].block_ids == (1, 2)
    assert result["PU"].rank == result["U_only"].rank == result["P_only"].rank
    for row in evidence:
        row["protected"] = False
        row["absolute_CI_low"] = -1.0
    empty = choose(spec, evidence)
    assert empty["PU_empty"] and empty["k"] == 0 and empty["U_only"] is None


if __name__ == "__main__":
    self_test()
    print("SELECTOR_SELF_TEST_PASS")
