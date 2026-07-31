from __future__ import annotations

import json
import os
from pathlib import Path
import h5py
import numpy as np


def write_subject_hdf5(path: str | Path, arrays: dict[str, np.ndarray], metadata: dict[str, object], config_hash: str, resume: bool = True) -> str:
    path = Path(path)
    if path.exists() and resume:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("complete", False) and handle.attrs.get("preprocessing_config_hash") == config_hash:
                return "resumed"
        raise RuntimeError("existing cache is incomplete or has a different configuration hash")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with h5py.File(temporary, "w") as handle:
        for name, array in arrays.items():
            handle.create_dataset(name, data=array, compression="gzip", compression_opts=1, chunks=True)
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        handle.attrs["preprocessing_config_hash"] = config_hash
        handle.attrs["complete"] = True
    os.replace(temporary, path)
    return "written"

