from .no_tta import NoTTA
from .t3a import T3A
from .entropy_mock import EntropyAdapterMock

try:
    from .entropy_adapter import EntropyAdapter, ResidualAdapter
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    EntropyAdapter = None
    ResidualAdapter = None

__all__ = ["NoTTA", "T3A", "EntropyAdapter", "EntropyAdapterMock", "ResidualAdapter"]
