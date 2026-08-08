"""Stage-0 falsification tools for measurement-operator-aware EEG."""

from .basis import CanonicalBasis, normalize_electrode_name
from .lifting import lifting_operators
from .measurement import MeasurementFunctional, build_measurement_matrix
from .operators import OperatorView, generate_eegmmidb_operators

__all__ = [
    "CanonicalBasis",
    "MeasurementFunctional",
    "OperatorView",
    "build_measurement_matrix",
    "generate_eegmmidb_operators",
    "lifting_operators",
    "normalize_electrode_name",
]
