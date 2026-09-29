"""Create reviewed compact hash index; excludes runtime NPZ/checkpoints/raw logs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target = HERE / "protocol" / "OUTPUT_HASHES.json"
    if target.exists(): raise FileExistsError(target)
    output = {p.name: sha(p) for p in sorted((HERE / "outputs").iterdir()) if p.is_file()}
    code = {p.name: sha(p) for p in sorted((HERE / "code").iterdir()) if p.is_file() and p.suffix in (".py", ".ps1")}
    runtime = {}
    for phase in ("train_lock", "train_pairing", "outer"):
        runtime[phase] = {f"fold{f}": sha(HERE / "runtime_inputs" / phase / f"fold{f}.json") for f in range(5)}
    record = {"schema": "TRIAL_CONTEXT_INTERACTION_COMPACT_HASH_INDEX_V1",
              "protocol_lock_sha256": sha(HERE / "protocol" / "PROTOCOL_LOCK.json"),
              "source_provenance_sha256": sha(HERE / "protocol" / "SOURCE_PROVENANCE.json"),
              "representation_provenance_sha256": sha(HERE / "protocol" / "REPRESENTATION_PROVENANCE.json"),
              "code_sha256": code, "compact_outputs_sha256": output,
              "server_runtime_json_sha256_not_pushed": runtime,
              "excluded": ["feature NPZ caches", "checkpoints", "raw task logs"],
              "formal_final_heldout_eeg_reads": 0}
    with target.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    print(sha(target))


if __name__ == "__main__": main()
