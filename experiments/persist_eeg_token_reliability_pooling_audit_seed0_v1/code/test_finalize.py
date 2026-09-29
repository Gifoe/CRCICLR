import pandas as pd
import pytest

from finalize import METHODS, bootstrap, selected_pivot


def test_subject_unit_pivot_and_paired_bootstrap():
    rows = []
    for subject, offset in (("s1", 0.0), ("s2", .1)):
        for method in METHODS:
            rows.append({"fold": 0, "subject": subject, "method": method,
                         "BA": .6 + offset + (.02 if method == "RELIABLE_UTILITY_TOPK" else 0)})
    random = pd.DataFrame([{"fold": 0, "subject": subject, "repeat": repeat, "BA": .55 + offset}
                           for subject, offset in (("s1", 0.0), ("s2", .1)) for repeat in range(3)])
    effects, biological, fold_ba = selected_pivot({
        "OUTER_POOLING_RESULTS.csv": pd.DataFrame(rows),
        "RANDOM_TWO_STREAM_SUBJECT.csv": random,
    })
    assert len(effects) == 2
    assert len(biological) == 2
    assert fold_ba.loc[0, "RELIABLE_UTILITY_TOPK"] == pytest.approx(.67)
    ci = bootstrap(biological)
    row = ci.query("method_a == 'RELIABLE_UTILITY_TOPK' and method_b == 'MEAN_ALL'").iloc[0]
    assert abs(row.mean_difference - .02) < 1e-12
    assert row.subjects == 2
    assert row.draws == 20_000
