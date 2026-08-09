#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from hsc_tta.v3.episodes import EpisodeProtocol, build_v3_episodes
from hsc_tta.v3.provenance import source_model_manifest, validate_episode_artifacts

from _common import config_hash, load_yaml, project_root


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="."); parser.add_argument("--config", required=True)
    args = parser.parse_args(); root = project_root(args.root); config = load_yaml(args.config)
    protocol = EpisodeProtocol(adapt_probe_ratio=str(config["adapt_probe_ratio"]),
        candidate_ratios=tuple(config["candidate_ratios"]), sleep_window_seconds=float(config["sleep_window_seconds"]),
        min_adapt=int(config["min_adapt"]), min_probe=int(config["min_probe"]))
    manifest = build_v3_episodes(root, protocol, tuple(config["datasets"]))
    split_hashes = {}
    for dataset in ("hmc", "eegmmidb"):
        source = root / "data/splits_v2_dev" / dataset; target = root / "data/splits_v3_dev" / dataset
        for path in sorted(source.rglob("*.json")):
            destination = target / path.relative_to(source); destination.parent.mkdir(parents=True, exist_ok=True)
            payload = json.loads(path.read_text()); payload["version"] = "v3-probecert-development-v1"
            payload["inherits_v2_subject_partition_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            part = destination.with_suffix(".json.part"); part.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n"); os.replace(part,destination)
            split_hashes[str(destination)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    output = root / "outputs/v3_probecert/provenance"; output.mkdir(parents=True, exist_ok=True)
    (output/"V3_SPLIT_HASHES.json").write_text(json.dumps({"config_hash":config_hash(config),"files":split_hashes},indent=2,sort_keys=True)+"\n")
    validation = validate_episode_artifacts(root, tuple(config["datasets"]))
    (output/"V3_EPISODE_VALIDATION.json").write_text(json.dumps(validation,indent=2,sort_keys=True)+"\n")
    source_output = root / "outputs/v3_probecert/source_models"; source_output.mkdir(parents=True,exist_ok=True)
    (source_output/"SOURCE_MODEL_MANIFEST.json").write_text(json.dumps(source_model_manifest(root),indent=2,sort_keys=True)+"\n")
    print({"subjects":len(manifest),"episode_files":validation["validated_files"],"split_files":len(split_hashes),
           "protocol_hash":protocol.config_hash,"future_preserved":True})


if __name__ == "__main__": main()
