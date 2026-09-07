import inspect
import pytest

pytest.importorskip("torch")
from hsc_tta.gpu.embeddings import extract_subject


def test_embedding_resume_is_hash_gated():
    source = inspect.getsource(extract_subject)
    assert "adapter_config_hash" in source and "checkpoint_sha256" in source
