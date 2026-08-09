import hashlib
import json

import numpy as np
import pytest

from hsc_tta.v2.confirmation import ConfirmatoryEpisode, file_sha256, validate_manifest


def _manifest(freeze_hash):
    return {"dataset": "new_site", "dataset_hash": "abc", "license_approved": True,
            "pretraining_overlap_audited": True, "calibration_subjects": ["c1"],
            "test_subjects": ["t1"], "method_freeze_sha256": freeze_hash}


def test_confirmatory_manifest_requires_freeze_hash_and_isolation(tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text("frozen")
    digest = hashlib.sha256(b"frozen").hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(digest)))
    assert validate_manifest(manifest, file_sha256(freeze))["dataset"] == "new_site"
    payload = _manifest(digest); payload["test_subjects"] = ["c1"]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="disjoint"):
        validate_manifest(manifest, digest)


def test_confirmatory_manifest_requires_approvals(tmp_path):
    path = tmp_path / "manifest.json"
    payload = _manifest("x"); payload["license_approved"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(PermissionError):
        validate_manifest(path, "x")
    with pytest.raises(ValueError, match="does not reference"):
        validate_manifest(path, "wrong")


def test_confirmatory_episode_rejects_u_v_overlap_and_duplicate_channels():
    valid = ConfirmatoryEpisode("d", "s", np.array([0, 1]), np.array([2, 3]), ("C3", "C4"), {"a": 0}, "hash")
    valid.validate()
    with pytest.raises(ValueError, match="U/V overlap"):
        ConfirmatoryEpisode("d", "s", np.array([0]), np.array([0]), ("C3",), {"a": 0}, "hash").validate()
    with pytest.raises(ValueError, match="duplicate"):
        ConfirmatoryEpisode("d", "s", np.array([0]), np.array([1]), ("C3", "C3"), {"a": 0}, "hash").validate()
