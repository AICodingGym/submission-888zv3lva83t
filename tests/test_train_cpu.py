import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_cpu import (  # noqa: E402
    apply_thresholds,
    optimize_thresholds,
    qwk,
    round_and_clip,
    text_statistics,
    validate_submission,
)


class TrainCpuTests(unittest.TestCase):
    def test_qwk_is_one_for_perfect_predictions(self):
        labels = np.array([1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(qwk(labels, labels), 1.0)

    def test_qwk_matches_sklearn(self):
        labels = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        predictions = np.array([1, 2, 2, 3, 2, 3, 4, 5, 4, 6, 5, 6])
        expected = cohen_kappa_score(labels, predictions, weights="quadratic")
        self.assertAlmostEqual(qwk(labels, predictions), expected)

    def test_score_conversion_stays_in_valid_range(self):
        predictions = np.array([-3.0, 1.49, 2.6, 8.0])
        np.testing.assert_array_equal(round_and_clip(predictions), [1, 1, 3, 6])
        thresholded = apply_thresholds(predictions, [1.5, 2.5, 3.5, 4.5, 5.5])
        self.assertTrue(np.all((thresholded >= 1) & (thresholded <= 6)))

    def test_threshold_optimizer_returns_monotonic_thresholds(self):
        labels = np.repeat(np.arange(1, 7), 20)
        predictions = labels.astype(float) + np.linspace(-0.4, 0.4, len(labels))
        thresholds, score = optimize_thresholds(labels, predictions, passes=2, candidate_quantiles=51)
        self.assertEqual(len(thresholds), 5)
        self.assertTrue(np.all(np.diff(thresholds) > 0))
        self.assertGreater(score, 0.9)

    def test_text_statistics_are_finite(self):
        features = text_statistics(["Hello world!", "A longer essay.\n\nIt has two paragraphs."])
        self.assertEqual(features.shape[0], 2)
        self.assertTrue(np.isfinite(features).all())

    def test_submission_validation(self):
        test = pd.DataFrame({"essay_id": ["a", "b"], "full_text": ["x", "y"]})
        submission = pd.DataFrame({"essay_id": ["a", "b"], "score": [2, 4]})
        validate_submission(submission, test)

if __name__ == "__main__":
    unittest.main()
