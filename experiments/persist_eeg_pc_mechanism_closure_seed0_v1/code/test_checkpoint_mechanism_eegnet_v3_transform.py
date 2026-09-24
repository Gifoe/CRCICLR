from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("audit_checkpoint_mechanism_eegnet_v3.py")
SPEC = importlib.util.spec_from_file_location("audit_checkpoint_mechanism_eegnet_v3", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v3_adapter_changes_only_backtrace_and_fresh_output_paths() -> None:
    source = "\n".join((
        'result = cell / "checkpoint_mechanism_v1" / f"epoch_{epoch:03d}.json"',
        'failure = cell / "checkpoint_mechanism_v1" / f"epoch_{epoch:03d}.FAIL_CLOSED.json"',
        'backtrace_path = cell / "FINAL_P_BACKTRACE_V1.json"',
        'print("CHECKPOINT_MECHANISM_V1_COMPLETE", task, fold, epoch)',
        'cap = 8; projector = "TRAIN_ONLY_FROZEN"; metric = adjacent_mediation(...)',
    ))
    transformed = MODULE.transform_source(source)
    assert transformed.count('cell / "checkpoint_mechanism_v2"') == 2
    assert 'cell / "FINAL_P_BACKTRACE_V2.json"' in transformed
    assert 'CHECKPOINT_MECHANISM_V2_COMPLETE' in transformed
    assert 'cap = 8; projector = "TRAIN_ONLY_FROZEN"; metric = adjacent_mediation(...)' in transformed
    assert 'checkpoint_mechanism_v1' not in transformed
    assert 'FINAL_P_BACKTRACE_V1.json' not in transformed


def test_v3_adapter_fails_closed_if_locked_source_shape_changes() -> None:
    try:
        MODULE.transform_source('cell / "checkpoint_mechanism_v1"')
    except RuntimeError as exc:
        assert "locked source transformation mismatch" in str(exc)
    else:
        raise AssertionError("adapter must reject an unexpected source shape")
