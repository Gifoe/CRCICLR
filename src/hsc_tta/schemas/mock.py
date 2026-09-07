"""Deprecated placeholder for the retired mixed GPU/statistics mock schema."""


def write_mock_gpu_interface(*args: object, **kwargs: object) -> None:
    raise RuntimeError(
        "The mixed mock GPU interface was retired; use the leakage-separated formal schemas"
    )
