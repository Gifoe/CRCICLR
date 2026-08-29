"""Outcome-blind tests for the locked subject-level D-versus-I reanalysis."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from itertools import product
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import run_subject_level_d_vs_i_reanalysis as analysis  # noqa: E402


def formal_grid() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dataset_specs = {
        "OPENBMI_STRESS": {
            "fold_sizes": [8, 8, 8, 8, 8],
            "backbones": ("eegnet", "eegconformer"),
            "configurations": analysis.CONFIGURATIONS,
            "prefix": "OB",
        },
        "WBCIC_REPLICATION": {
            "fold_sizes": [9, 8, 8, 8, 8],
            "backbones": ("eegnet",),
            "configurations": (("ERM", 0.0),),
            "prefix": "WB",
        },
    }
    for dataset, spec in dataset_specs.items():
        subject_offset = 0
        for fold, fold_size in enumerate(spec["fold_sizes"]):
            subjects = [f"{spec['prefix']}{subject_offset + index:03d}" for index in range(fold_size)]
            subject_offset += fold_size
            for subject, backbone, seed, configuration, direction in product(
                subjects,
                spec["backbones"],
                range(3),
                spec["configurations"],
                range(8),
            ):
                method, lam = configuration
                method_value = {"ERM": 0.0, "DANN": 1.0, "CORAL": 2.0, "MMD": 3.0}[method]
                rows.append(
                    {
                        "dataset": dataset,
                        "subject_id_internal": subject,
                        "backbone": backbone,
                        "fold": fold,
                        "seed": seed,
                        "method": method,
                        "lambda": float(lam),
                        "direction_id": direction,
                        "persistence": 0.1 * direction,
                        "geometry_strength": 1.0 + 0.01 * direction,
                        "rank_feature": 1.0,
                        "identity_score": 0.2 + 0.01 * method_value,
                        "D_finite": 0.3 + 0.02 * direction + 0.01 * seed,
                        "subject_CE_effect": 0.05 * direction + 0.001 * subject_offset,
                        "subject_trial_count": 100,
                    }
                )
    return pd.DataFrame(rows)


def compact_cross_fit_grid() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, backbones, configurations in (
        ("OPENBMI_STRESS", ("eegnet", "eegconformer"), analysis.CONFIGURATIONS),
        ("WBCIC_REPLICATION", ("eegnet",), (("ERM", 0.0),)),
    ):
        for fold in range(5):
            for subject_index in range(3):
                subject = f"{dataset[:2]}-{fold}-{subject_index}"
                subject_term = 0.03 * subject_index - 0.01 * fold
                for backbone, seed, configuration, direction in product(
                    backbones, range(3), configurations, range(2)
                ):
                    method, lam = configuration
                    method_term = {"ERM": 0.0, "DANN": 0.01, "CORAL": -0.01, "MMD": 0.02}[method]
                    d_value = 0.2 + 0.04 * direction + 0.01 * seed + method_term
                    i_value = 0.5 - 0.03 * direction + 0.02 * seed
                    rows.append(
                        {
                            "dataset": dataset,
                            "subject_id_internal": subject,
                            "backbone": backbone,
                            "fold": fold,
                            "seed": seed,
                            "method": method,
                            "lambda": float(lam),
                            "direction_id": direction,
                            "persistence": 0.1 + 0.01 * direction,
                            "geometry_strength": 1.0 + 0.02 * direction,
                            "rank_feature": 1.0,
                            "identity_score": i_value,
                            "D_finite": d_value,
                            "subject_CE_effect": 0.8 * d_value + subject_term + 0.002 * seed,
                            "subject_trial_count": 10,
                        }
                    )
    return pd.DataFrame(rows)


def literal_nonuniform_refit_delta(
    observations: pd.DataFrame,
    spec: analysis.AnalysisSpec,
    *,
    fold: int,
    backbone: str,
    multiplicity: np.ndarray,
) -> np.ndarray:
    """Slow, direct ridge refit used only to test the optimized bootstrap algebra."""

    block = analysis.analysis_filter(observations, spec)
    block = block[(block.fold == fold) & (block.backbone == backbone)]
    subjects = analysis.subject_sort(block.subject_id_internal.unique())
    multiplicity = np.asarray(multiplicity, dtype=np.float64)
    families = sorted(block.method.astype(str).unique()) if spec.subject_aggregation_mode == "EQUAL_METHOD_FAMILY" else ["ALL"]
    squared = {
        model: np.zeros((len(subjects), len(families)), dtype=np.float64)
        for model in ("MI", "MD")
    }
    counts = np.zeros((len(subjects), len(families)), dtype=np.int64)
    for held_seed in sorted(map(int, block.seed.unique())):
        training = block[block.seed != held_seed]
        features = (
            training.groupby(analysis.CELL_COLUMNS, as_index=False)
            .agg(**{column: (column, "first") for column in analysis.PREDICTOR_COLUMNS})
            .sort_values(analysis.CELL_COLUMNS)
            .reset_index(drop=True)
        )
        training_index = pd.MultiIndex.from_frame(features[analysis.CELL_COLUMNS])
        effects = np.stack(
            [
                training[training.subject_id_internal.astype(str) == subject]
                .set_index(analysis.CELL_COLUMNS)
                .reindex(training_index)
                .subject_CE_effect.to_numpy(np.float64)
                for subject in subjects
            ]
        )
        cell_weight = analysis.training_cell_weights(features, spec.training_weight_mode)
        for subject_index, subject in enumerate(subjects):
            denominator = float(multiplicity.sum() - multiplicity[subject_index])
            target = (
                multiplicity @ effects - multiplicity[subject_index] * effects[subject_index]
            ) / denominator
            test = block[
                (block.seed == held_seed) & (block.subject_id_internal.astype(str) == subject)
            ].sort_values(analysis.CELL_COLUMNS)
            for family_index, family in enumerate(families):
                mask = np.ones(len(test), dtype=bool) if family == "ALL" else test.method.astype(str).to_numpy() == family
                counts[subject_index, family_index] += int(mask.sum())
                for model in ("MI", "MD"):
                    prediction = analysis.ridge_predict(
                        features[list(analysis.MODELS[model])].to_numpy(np.float64),
                        target,
                        test[list(analysis.MODELS[model])].to_numpy(np.float64),
                        sample_weight=cell_weight,
                        alpha=analysis.RIDGE_ALPHA,
                    )
                    squared[model][subject_index, family_index] += np.square(
                        prediction[mask] - test.subject_CE_effect.to_numpy(np.float64)[mask]
                    ).sum()
    rmse = {model: np.sqrt(value / counts) for model, value in squared.items()}
    return (rmse["MI"] - rmse["MD"]).mean(axis=1)


class LockedReanalysisTests(unittest.TestCase):
    def test_protocol_binding_and_global_fold_rules(self) -> None:
        protocol_path = REPOSITORY / "paper_closure" / "protocol" / "SUBJECT_LEVEL_D_VS_I_REANALYSIS_LOCK.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        self.assertEqual(analysis.verify_protocol_binding(protocol)["status"], "PASS")
        binding = analysis.implementation_binding()
        specs = analysis.artifact_specs(
            analysis.SourceRoots(Path("C:/synthetic/openbmi"), Path("C:/synthetic/wbcic"))
        )
        self.assertEqual(len(specs), binding["expected_manifest_artifact_count"])
        self.assertEqual(Counter(row["source_alias"] for row in specs), binding["expected_manifest_counts_by_alias"])
        self.assertEqual(Counter(row["role"] for row in specs), binding["expected_manifest_counts_by_role"])
        openbmi = analysis.global_protocol_roles(
            REPOSITORY
            / "experiments"
            / "persist_eeg_subject_invariance_stress_test_v1"
            / "STRESS_TEST_PROTOCOL_FROZEN.json",
            dataset="OPENBMI_STRESS",
        )
        wbcic = analysis.global_protocol_roles(
            REPOSITORY
            / "experiments"
            / "persist_eeg_wbcic_independent_replication_v1"
            / "WBCIC_REPLICATION_PROTOCOL_FROZEN.json",
            dataset="WBCIC_REPLICATION",
        )
        self.assertEqual(len({subject for row in openbmi.values() for subject in row["outcome"]}), 40)
        self.assertEqual(len({subject for row in wbcic.values() for subject in row["outcome"]}), 41)

    def test_exhaustive_gates_and_canonical_one_shot_paths(self) -> None:
        cases = [
            (0.10, (0.01, 0.20), "SUPPORTED_CONDITIONAL"),
            (0.10, (-0.01, 0.20), "PARTIAL"),
            (-0.01, (-0.20, 0.20), "NOT_SUPPORTED"),
            (-0.10, (-0.20, -0.01), "REVERSED"),
            (0.10, (-0.20, -0.01), "POINT_CI_DIRECTION_CONFLICT"),
            (-0.10, (0.01, 0.20), "POINT_CI_DIRECTION_CONFLICT"),
        ]
        for observed, ci95, expected in cases:
            self.assertEqual(analysis.classify_dataset_gate(observed, ci95), expected)
        with self.assertRaises(RuntimeError):
            analysis.interval(np.asarray([0.0, np.inf]))
        self.assertEqual(
            analysis.classify_cross_dataset_terminal(
                ["SUPPORTED_CONDITIONAL", "POINT_CI_DIRECTION_CONFLICT"],
                points_positive=True,
                openbmi_both_backbone_points_positive=True,
            ),
            "CROSS_DATASET_POINT_CI_DIRECTION_CONFLICT",
        )

        paths = analysis.canonical_paths(REPOSITORY)
        checked = analysis.verify_canonical_invocation(
            REPOSITORY,
            protocol=paths["protocol"],
            manifest=paths["manifest"],
            output_directory=paths["output_directory"],
        )
        self.assertEqual(
            checked["repository_paths"]["output_directory"],
            analysis.CANONICAL_REPOSITORY_PATHS["output_directory"],
        )
        analysis.verify_canonical_invocation(
            REPOSITORY,
            protocol=paths["protocol"],
            manifest_output=paths["manifest"],
        )
        with self.assertRaises(RuntimeError):
            analysis.verify_canonical_invocation(
                REPOSITORY,
                protocol=paths["protocol"],
                manifest=paths["manifest"],
                output_directory=paths["output_directory"].with_name("alternate-result"),
            )
        with self.assertRaises(RuntimeError):
            analysis.verify_canonical_invocation(
                REPOSITORY,
                protocol=paths["protocol"],
                manifest_output=paths["manifest"].with_name("alternate-manifest.csv"),
            )

    def test_manifest_publication_stays_staged_on_prepublication_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "manifest.csv"
            rows = [{column: "" for column in analysis.MANIFEST_COLUMNS}]
            with mock.patch.object(
                analysis,
                "sha256_file",
                side_effect=PermissionError("injected prepublication hash failure"),
            ):
                with self.assertRaises(PermissionError):
                    analysis.write_manifest_atomic(output, rows)
            self.assertFalse(output.exists())
            self.assertTrue(output.with_name(f".{output.name}.staging").exists())

    def test_matched_float64_d_finite(self) -> None:
        clean = np.zeros((2, 2), dtype=np.float64)
        erased = np.asarray([[1.0, -1.0], [2.0, -2.0]], dtype=np.float64)
        self.assertAlmostEqual(analysis.exact_d_finite(clean, erased), np.sqrt(5.0), places=14)

    def test_effect_rows_uses_corrected_D_after_legacy_integrity_check(self) -> None:
        rng = np.random.default_rng(17)
        source_features = rng.normal(size=(8, 8))
        outcome_features = rng.normal(size=(8, 8))
        weight = rng.normal(size=(2, 8))
        bias = rng.normal(size=2)
        center = source_features.mean(axis=0)
        basis = np.eye(8)
        corrected_source_logits = source_features @ weight.T + bias
        corrected_outcome_logits = outcome_features @ weight.T + bias
        stored_source_logits = corrected_source_logits + np.asarray([0.03, -0.02])
        stored_outcome_logits = corrected_outcome_logits + np.column_stack(
            [np.linspace(0.0, 0.02, 8), np.linspace(0.01, -0.01, 8)]
        )
        labels = np.asarray([0, 1] * 4)
        subjects = np.asarray(["a"] * 4 + ["b"] * 4)
        rows = []
        for direction_id in range(8):
            erased_source = analysis.erase_direction(source_features, center, basis[:, direction_id])
            erased_outcome = analysis.erase_direction(outcome_features, center, basis[:, direction_id])
            erased_source_logits = erased_source @ weight.T + bias
            erased_outcome_logits = erased_outcome @ weight.T + bias
            rows.append(
                {
                    "direction_id": direction_id,
                    "persistence": 0.1,
                    "geometry_strength": 1.0,
                    "rank": 1,
                    "identity_score": 0.2,
                    "D_finite": analysis.exact_d_finite(stored_source_logits, erased_source_logits),
                    "outcome_CE_effect": float(
                        np.mean(
                            analysis.numpy_cross_entropy(erased_outcome_logits, labels)
                            - analysis.numpy_cross_entropy(stored_outcome_logits, labels)
                        )
                    ),
                    "outcome_subject_count": 2,
                }
            )
        emitted, integrity = analysis.effect_rows(
            dataset="OPENBMI_STRESS",
            backbone="eegnet",
            fold=0,
            seed=0,
            method="ERM",
            lam=0.0,
            direction_table=pd.DataFrame(rows),
            embeddings={
                "source_features": source_features,
                "source_logits": stored_source_logits,
                "outcome_features": outcome_features,
                "outcome_logits": stored_outcome_logits,
                "outcome_labels": labels,
                "outcome_subjects": subjects,
            },
            weight=weight,
            bias=bias,
            center=center,
            basis=basis,
            legacy_uses_stored_clean_logits=True,
            legacy_d_uses_stored_source_logits=True,
        )
        corrected_first = analysis.exact_d_finite(
            corrected_source_logits,
            analysis.erase_direction(source_features, center, basis[:, 0]) @ weight.T + bias,
        )
        self.assertAlmostEqual(emitted[0]["D_finite"], corrected_first, places=14)
        corrected_erased_logits = (
            analysis.erase_direction(outcome_features, center, basis[:, 0]) @ weight.T + bias
        )
        corrected_trial_effect = analysis.numpy_cross_entropy(corrected_erased_logits, labels) - analysis.numpy_cross_entropy(
            corrected_outcome_logits, labels
        )
        self.assertAlmostEqual(
            emitted[0]["subject_CE_effect"],
            float(corrected_trial_effect[:4].mean()),
            places=14,
        )
        self.assertGreater(integrity["max_legacy_to_float64_D_finite_abs_difference"], 0.0)
        self.assertGreater(integrity["max_legacy_to_float64_aggregate_abs_difference"], 0.0)

    def test_exact_cartesian_grid_rejects_offsetting_omission_and_duplicate(self) -> None:
        grid = formal_grid()
        audit = analysis.validate_reconstructed_grid(grid)
        self.assertEqual(audit["status"], "PASS")
        corrupted = pd.concat([grid.iloc[1:], grid.iloc[[1]]], ignore_index=True)
        with self.assertRaises(RuntimeError):
            analysis.validate_reconstructed_grid(corrupted)

    def test_embedding_key_and_index_guards(self) -> None:
        source_subjects = np.repeat(["1", "2"], 80)
        source_sessions = np.tile(np.repeat([0, 1], 40), 2)
        source_labels = np.tile(np.repeat([0, 1], 20), 4)
        outcome_subjects = np.repeat(["3"], 40)
        outcome_sessions = np.repeat([2], 40)
        outcome_labels = np.repeat([0, 1], 20)
        archive = {
            "source_features": np.zeros((160, 4), dtype=np.float32),
            "source_logits": np.zeros((160, 2), dtype=np.float32),
            "source_labels": source_labels,
            "source_subjects": source_subjects,
            "source_sessions": source_sessions,
            "source_indices": np.arange(160),
            "outcome_features": np.zeros((40, 4), dtype=np.float32),
            "outcome_logits": np.zeros((40, 2), dtype=np.float32),
            "outcome_labels": outcome_labels,
            "outcome_subjects": outcome_subjects,
            "outcome_sessions": outcome_sessions,
            "outcome_indices": np.arange(1000, 1040),
        }
        roles = {"source": ("1", "2"), "outcome": ("3",)}
        guarded = analysis.embedding_partition_guard(
            archive,
            dataset="WBCIC_REPLICATION",
            roles=roles,
            context="synthetic",
        )
        analysis.assert_same_index_arrays(guarded, guarded, context="synthetic")
        with_extra = dict(archive, unexpected=np.asarray([1]))
        with self.assertRaises(RuntimeError):
            analysis.embedding_partition_guard(
                with_extra,
                dataset="WBCIC_REPLICATION",
                roles=roles,
                context="synthetic-extra",
            )
        mutated = {key: value.copy() for key, value in guarded.items()}
        mutated["outcome_indices"][0] += 1
        with self.assertRaises(RuntimeError):
            analysis.assert_same_index_arrays(guarded, mutated, context="synthetic-mutated")

    def test_wbcic_source_freeze_grid_and_same_handle_npz_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            unit = Path(temporary)
            basis_path = unit / "source_freeze" / "erm_persistence_basis.npz"
            checkpoint_path = unit / "checkpoints" / f"{analysis.config_slug('ERM', 0.0)}.pt"
            selection_path = unit / "LAMBDA_SELECTION_FROZEN.json"
            basis_path.parent.mkdir(parents=True)
            checkpoint_path.parent.mkdir(parents=True)
            basis_path.write_bytes(b"frozen-basis-file")
            checkpoint_path.write_bytes(b"frozen-erm-checkpoint")
            selection_path.write_bytes(b"frozen-selection")
            center = np.zeros(8, dtype=np.float64)
            basis = np.eye(8, dtype=np.float64)
            checkpoints = []
            for index, (method, lam) in enumerate(analysis.CONFIGURATIONS):
                digest = analysis.sha256_file(checkpoint_path) if (method, lam) == ("ERM", 0.0) else f"{index + 1:064x}"
                checkpoints.append(
                    {"method": method, "lambda": lam, "checkpoint_sha256": digest}
                )
            payload = {
                "schema": "WBCIC_REPLICATION_RUN_SOURCE_FREEZE_V1",
                "pass": True,
                "backbone": "eegnet",
                "fold": 0,
                "seed": 0,
                "model_fit_subjects": ["1", "2"],
                "validation_discovery_subjects": ["3"],
                "outcome_subjects_not_loaded": True,
                "outcome_S3_labels_used": False,
                "sealed_WBCIC_outer_accessed": False,
                "OpenBMI_holdout_accessed": False,
                "direction_count": 8,
                "persistence_basis_file_sha256": analysis.sha256_file(basis_path),
                "persistence_basis_array_sha256": analysis.sha256_array(basis),
                "persistence_center_array_sha256": analysis.sha256_array(center),
                "selection_file_sha256": analysis.sha256_file(selection_path),
                "checkpoint_count": len(checkpoints),
                "checkpoints": checkpoints,
            }
            guard_path = unit / "SOURCE_FREEZE_COMPLETE.json"
            analysis.write_json(guard_path, payload)
            guarded = analysis.wbcic_source_freeze_guard(
                guard_path,
                roles={
                    "model_fit": ("1", "2"),
                    "validation_discovery": ("3",),
                },
                backbone="eegnet",
                fold=0,
                seed=0,
                basis_path=basis_path,
                checkpoint_path=checkpoint_path,
                center=center,
                basis=basis,
            )
            self.assertEqual(guarded["status"], "PASS")
            malformed = json.loads(guard_path.read_text(encoding="utf-8"))
            malformed["checkpoints"][1]["method"] = "ERM"
            malformed["checkpoints"][1]["lambda"] = 0.0
            analysis.write_json(guard_path, malformed)
            with self.assertRaises(RuntimeError):
                analysis.wbcic_source_freeze_guard(
                    guard_path,
                    roles={"model_fit": ("1", "2"), "validation_discovery": ("3",)},
                    backbone="eegnet",
                    fold=0,
                    seed=0,
                    basis_path=basis_path,
                    checkpoint_path=checkpoint_path,
                    center=center,
                    basis=basis,
                )

            archive_path = unit / "object-array.npz"
            np.savez(archive_path, subjects=np.asarray(["1", "2"], dtype=object))
            key = ("SYNTHETIC", archive_path.relative_to(unit).as_posix())
            hashes = {key: analysis.sha256_file(archive_path)}
            with analysis.load_manifest_verified_npz(
                archive_path,
                source_alias="SYNTHETIC",
                source_root=unit,
                manifest_hashes=hashes,
                allow_pickle=True,
            ) as archive:
                self.assertEqual(list(archive["subjects"].astype(str)), ["1", "2"])
            np.savez(archive_path, subjects=np.asarray(["changed"], dtype=object))
            with self.assertRaises(RuntimeError):
                with analysis.load_manifest_verified_npz(
                    archive_path,
                    source_alias="SYNTHETIC",
                    source_root=unit,
                    manifest_hashes=hashes,
                    allow_pickle=True,
                ):
                    pass
            with self.assertRaises(ValueError):
                analysis.write_json(unit / "nonfinite.json", {"value": float("nan")})

    def test_equal_family_weights_and_refitted_uniform_point(self) -> None:
        reconstructed = compact_cross_fit_grid()
        predicted, split = analysis.add_doubly_cross_fitted_predictions(reconstructed)
        self.assertFalse(split.other_historical_folds_used.any())
        self.assertFalse(split.other_backbones_used.any())
        self.assertFalse(split.held_subject_outcomes_used_for_fit.any())
        self.assertFalse(split.held_seed_outcomes_used_for_fit.any())
        public = analysis.pseudonymize(predicted)
        subjects, backbone_subjects = analysis.subject_summaries(public)
        spec = next(item for item in analysis.ANALYSES if item.name == "OPENBMI_EQUAL_FAMILY_PRIMARY")
        train_example = reconstructed[
            (reconstructed.dataset == "OPENBMI_STRESS")
            & (reconstructed.fold == 0)
            & (reconstructed.backbone == "eegnet")
            & (reconstructed.seed != 0)
        ].groupby(analysis.CELL_COLUMNS, as_index=False).first()
        weights = analysis.training_cell_weights(train_example, spec.training_weight_mode)
        family_totals = pd.DataFrame({"method": train_example.method, "weight": weights}).groupby("method").weight.sum()
        self.assertTrue(np.allclose(family_totals, family_totals.iloc[0]))
        bootstrap = analysis.refitted_subject_bootstrap(
            reconstructed,
            spec,
            draws=50,
            seed=1234,
        )
        check = analysis.verify_uniform_refit_points(
            subjects[subjects.analysis == spec.name],
            backbone_subjects[backbone_subjects.analysis == spec.name],
            bootstrap,
            analysis=spec.name,
        )
        self.assertEqual(check["status"], "PASS")
        self.assertLess(max(check["absolute_differences"].values()), 1e-12)
        block = next(
            item
            for item in analysis.prepare_refit_blocks(reconstructed, spec)
            if item["fold"] == 0 and item["backbone"] == "eegnet"
        )
        multiplicity = np.asarray([2.0, 0.0, 1.0])
        optimized = analysis.refit_block_delta(block, multiplicity)
        literal = literal_nonuniform_refit_delta(
            reconstructed,
            spec,
            fold=0,
            backbone="eegnet",
            multiplicity=multiplicity,
        )
        self.assertTrue(np.allclose(optimized, literal, atol=1e-14, rtol=1e-14))


if __name__ == "__main__":
    unittest.main(verbosity=2)
