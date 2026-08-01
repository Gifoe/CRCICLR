from pathlib import Path
import pytest

pytest.importorskip("torch")
from hsc_tta.gpu.experiment import head_path


def test_cap_head_path_is_hmc():
    assert "/hmc/seed_2/" in str(head_path(Path("/x"), "cap", 2)).replace("\\", "/")
