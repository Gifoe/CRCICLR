import numpy as np

from hsc_tta.moa_stage0.basis import CanonicalBasis, normalize_electrode_name
from hsc_tta.moa_stage0.lifting import lifting_operators, numerical_audit
from hsc_tta.moa_stage0.measurement import MeasurementFunctional, build_measurement_matrix


def test_name_normalization_is_syntactic_and_explicit():
    assert normalize_electrode_name(" Fc3. ") == "FC3"
    assert normalize_electrode_name("T3") == "T7"
    assert normalize_electrode_name("M1") != normalize_electrode_name("A1")


def test_basis_gauge_and_gram():
    audit = CanonicalBasis.fixed().audit()
    assert audit["constant_mode_max_abs"] < 1e-12
    assert audit["gram_identity_relative_error"] < 1e-6


def test_measurement_sign_is_preserved():
    basis = CanonicalBasis.fixed()
    coordinates = {"C3": np.array([-0.5, 0.0, 0.866]), "P3": np.array([-0.4, -0.5, 0.768])}
    forward = build_measurement_matrix([MeasurementFunctional.bipolar("C3", "P3")], coordinates, basis)
    reverse = build_measurement_matrix([MeasurementFunctional.bipolar("P3", "C3")], coordinates, basis)
    np.testing.assert_allclose(forward, -reverse)


def test_lifting_projector_resolution_and_generative_identity():
    rng = np.random.default_rng(7)
    b = rng.normal(size=(9, 32))
    values = lifting_operators(b, alpha=1e-2)
    assert values["L"].shape == (32, 9)
    assert values["Q"].shape == values["R"].shape == (32, 32)
    x = rng.normal(size=(32, 41)); y = b @ x
    np.testing.assert_allclose(values["L"] @ y, values["R"] @ x, atol=1e-10)
    audit = numerical_audit(b)
    assert audit["q_symmetry_relative_error"] < 1e-10
    assert audit["q_idempotence_relative_error"] < 1e-10
    assert audit["r_symmetry_relative_error"] < 1e-10
    assert audit["r_eigenvalue_min"] > -1e-10
    assert audit["r_eigenvalue_max"] < 1 + 1e-10


def test_operator_and_null_space_sanity():
    rng = np.random.default_rng(11)
    b0 = rng.normal(size=(16, 32)); a = rng.normal(size=(7, 16)); bt = a @ b0
    assert np.linalg.norm(bt - a @ b0) / np.linalg.norm(bt) < 1e-14
    _, _, vh = np.linalg.svd(bt, full_matrices=True)
    null_vector = vh[-1]
    assert np.linalg.norm(bt @ null_vector) < 1e-10


def test_nested_operator_with_projector_residual():
    rng = np.random.default_rng(19)
    broad = rng.normal(size=(12, 32)); narrow = broad[[1, 4, 8]]
    row_projector = np.linalg.pinv(broad) @ broad
    residual = np.linalg.norm(narrow - narrow @ row_projector) / np.linalg.norm(narrow)
    assert residual < 1e-10
