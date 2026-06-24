from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from recovery_score.components import biological_identity
from recovery_score.scorer import GaitNormalityScorer
from recovery_score.validation import pair_aware_repeated_splits


class GaitNormalityScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.ha = rng.normal(0.0, 0.25, size=(8, 3, 2, 9, 101))
        self.acl = rng.normal(1.5, 0.25, size=(4, 3, 2, 9, 101))
        self.matrices = np.concatenate([self.ha, self.acl], axis=0)
        self.groups = np.array(["HA"] * len(self.ha) + ["ACLD"] * 2 + ["ACLR"] * 2)
        self.subject_ids = np.array(
            [f"HA{i}" for i in range(len(self.ha))] + ["ACLD1", "ACLD2", "ACLR1", "ACLR2"]
        )

    def test_score_is_finite_and_acl_is_farther_from_ha(self) -> None:
        scorer = GaitNormalityScorer().fit(self.matrices, self.groups, self.subject_ids)
        scores = scorer.score_matrices(self.matrices)
        self.assertTrue(np.isfinite(scores.to_numpy()).all())
        self.assertLess(
            scores.loc[self.groups != "HA", "normality_score"].mean(),
            scores.loc[self.groups == "HA", "normality_score"].mean(),
        )
        self.assertTrue(
            {"hip_score", "knee_score", "ankle_score", "fast_score", "asymmetry_score"}
            <= set(scores.columns)
        )

    def test_model_json_roundtrip(self) -> None:
        scorer = GaitNormalityScorer().fit(self.matrices, self.groups, self.subject_ids)
        expected = scorer.score_matrices(self.matrices[:2])
        with tempfile.TemporaryDirectory() as directory:
            path = scorer.save(Path(directory) / "model.json")
            actual = GaitNormalityScorer.load(path).score_matrices(self.matrices[:2])
        np.testing.assert_allclose(expected.to_numpy(), actual.to_numpy(), rtol=1e-10, atol=1e-10)

    def test_pair_aware_split_has_no_identity_overlap(self) -> None:
        metadata = pd.DataFrame(
            {
                "subject_id": self.subject_ids,
                "group": self.groups,
                "identity_id": [biological_identity(subject) for subject in self.subject_ids],
            }
        )
        for _, _, train_idx, test_idx in pair_aware_repeated_splits(
            metadata, n_splits=2, n_repeats=2, seed=13
        ):
            train_ids = set(metadata.loc[train_idx, "identity_id"])
            test_ids = set(metadata.loc[test_idx, "identity_id"])
            self.assertFalse(train_ids & test_ids)
            for patient in ("ACL::1", "ACL::2"):
                self.assertNotEqual(patient in train_ids, patient in test_ids)


if __name__ == "__main__":
    unittest.main()
