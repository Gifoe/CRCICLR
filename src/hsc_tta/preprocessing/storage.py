from __future__ import annotations

import json
import os
from pathlib import Path
import h5py
import numpy as np


def write_subject_hdf5(path: str | Path, arrays: dict[str, np.ndarray], metadata: dict[str, object], config_hash: str, resume: bool = True) -> str:
    path = Path(path)
    replacing = path.exists()
    if path.exists() and resume:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("complete", False) and handle.attrs.get("preprocessing_config_hash") == config_hash:
                return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with h5py.File(temporary, "w") as handle:
        for name, array in arrays.items():
            value = np.asarray(array)
            if value.ndim == 0:
                handle.create_dataset(name, data=value)
            else:
                handle.create_dataset(name, data=value, compression="gzip", compression_opts=1, chunks=True)
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        handle.attrs["preprocessing_config_hash"] = config_hash
        handle.attrs["complete"] = True
    os.replace(temporary, path)
    return "rewritten_config_changed" if replacing else "written"
