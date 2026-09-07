from pathlib import Path


def test_full_runner_supports_required_controls():
    text = (Path(__file__).parents[1] / "scripts" / "run_full_gpu_experiment.sh").read_text()
    for option in ("--resume", "--start-stage", "--stop-after-stage", "--seeds", "--datasets",
                   "--device", "--num-workers", "--batch-size", "--dry-run"):
        assert option in text
