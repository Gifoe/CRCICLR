from pathlib import Path


def test_preflight_contains_fixed_checkpoint_hash():
    text = (Path(__file__).parents[1] / "scripts" / "run_gpu_preflight.py").read_text()
    assert "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178" in text
