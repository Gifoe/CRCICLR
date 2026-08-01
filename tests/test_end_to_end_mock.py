from hsc_tta.simulation import run_simulations
from hsc_tta.schemas import write_mock_gpu_interface
from hsc_tta.schemas.models import ActionSurfaceRow, ContextFeatureRow, SubjectDecisionRow
import pandas as pd


def test_end_to_end_mock_outputs_are_computed(tmp_path):
    out=run_simulations(tmp_path,seed=3,n_subjects=40)
    assert set(out)=={"simulation_summary","certificate_coverage","pseudo_sample_size","post_selection_validity","safety_utility","risk_predictor_misspecification"}
    assert all((tmp_path/f"{name}.csv").exists() for name in out)


def test_mock_gpu_interface_round_trip(tmp_path):
    paths = write_mock_gpu_interface(tmp_path, seed=4, n_subjects=40)
    context = pd.read_parquet(paths["subject_context_features"])
    surface = pd.read_parquet(paths["subject_action_surface"])
    decisions = pd.read_parquet(paths["subject_decisions"])
    assert len(context) == len(decisions) == 10
    assert len(surface) == 10 * 3 * 20
    ContextFeatureRow.model_validate(context.iloc[0].to_dict())
    ActionSurfaceRow.model_validate(surface.iloc[0].to_dict())
    SubjectDecisionRow.model_validate(decisions.iloc[0].to_dict())
