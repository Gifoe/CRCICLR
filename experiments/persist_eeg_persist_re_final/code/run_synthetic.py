"""Small clustered simulation used only for mechanism illustration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import persist_re_core as c


def main() -> None:
    rng = np.random.default_rng(20260831)
    subjects = np.array([str(i) for i in range(24)])
    rows = []
    test_rows = []
    for sid in subjects:
        label = rng.integers(0, 2, 80)
        x = rng.normal(size=(80, 8)).astype("float32")
        x[:, 0] += 1.4 * (2 * label - 1)  # shared task direction
        x[:, 1] += (0.8 + 0.5 * np.sin(int(sid))) * (2 * label - 1)  # subject slope
        x[:, 2] += 1.5 * (int(sid) % 2)  # identity feature
        frame = {"features": x, "labels": label.astype("int64"), "subjects": np.repeat(sid, 80), "indices": np.arange(80) + int(sid) * 1000, "sessions": np.zeros(80, dtype="int64")}
        (rows if int(sid) < 18 else test_rows).append(frame)
    def join(values):
        return {k: np.concatenate([v[k] for v in values]) for k in values[0]}
    train, test = join(rows), join(test_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for method in ("ERM", "AdversarialMixed", "ProspectiveOnly", "RandomEffectOnly", "PERSIST-RE"):
        model, diag = c.fit_model(method, train, 2, 1e-3, 0.5, c.stable_seed("synthetic", method), device=device)
        pred = c.predict(model, test, {}, device)
        ba = float(np.mean([((pred["population_logits"][test["subjects"] == s].argmax(1) == test["labels"][test["subjects"] == s]).mean()) for s in np.unique(test["subjects"])]))
        out.append({"method": method, "held_subject_population_accuracy": ba, "random_effect_parameter_norm": diag.get("random_effect_parameter_norm", 0.0), "subjects_train": 18, "subjects_test": 6})
    c.write_csv(c.RESULTS / "SYNTHETIC_RESULTS.csv", pd.DataFrame(out))
    print(pd.DataFrame(out).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

