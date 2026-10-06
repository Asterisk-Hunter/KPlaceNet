"""Small deterministic tests for split-conformal order statistics."""

from __future__ import annotations

import math
import unittest

import torch

from src.uncertainty import fit_conformal_quantile


class FitConformalQuantileTests(unittest.TestCase):
    @staticmethod
    def probs_for_scores(scores: list[float], *, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        true_probs = torch.tensor([1.0 - score for score in scores], device=device)
        probs = torch.stack((true_probs, 1.0 - true_probs), dim=1)
        labels = torch.zeros(len(scores), dtype=torch.long, device=device)
        return probs, labels

    def test_uses_discrete_order_statistic_instead_of_interpolation(self):
        # n=4, alpha=.4 => ceil(5*.6)=3, so q is the third sorted score .6.
        probs, labels = self.probs_for_scores([0.1, 0.4, 0.6, 0.9])
        self.assertAlmostEqual(fit_conformal_quantile(probs, labels, alpha=0.4), 0.6)

    def test_ties_at_the_selected_rank_are_preserved(self):
        # n=5, alpha=.5 => rank 3; the tied third order statistic is .4.
        probs, labels = self.probs_for_scores([0.4, 0.2, 0.4, 0.8, 0.1])
        self.assertAlmostEqual(fit_conformal_quantile(probs, labels, alpha=0.5), 0.4)

    def test_single_example_returns_its_score_when_rank_is_one(self):
        probs, labels = self.probs_for_scores([0.25])
        self.assertAlmostEqual(fit_conformal_quantile(probs, labels, alpha=0.5), 0.25)

    def test_small_calibration_set_uses_infinite_threshold_when_rank_exceeds_n(self):
        probs, labels = self.probs_for_scores([0.2, 0.7])
        self.assertEqual(fit_conformal_quantile(probs, labels, alpha=0.1), math.inf)

    def test_extreme_alpha_values(self):
        probs, labels = self.probs_for_scores([0.1, 0.4, 0.8])
        self.assertEqual(fit_conformal_quantile(probs, labels, alpha=1e-12), math.inf)
        self.assertAlmostEqual(
            fit_conformal_quantile(probs, labels, alpha=1.0 - 1e-12), 0.1
        )

    def test_labels_are_aligned_to_probability_device(self):
        probs, _ = self.probs_for_scores([0.2, 0.4, 0.8])
        labels = torch.zeros(3, dtype=torch.long, device="cpu")
        if torch.cuda.is_available():
            probs = probs.cuda()
        self.assertAlmostEqual(fit_conformal_quantile(probs, labels, alpha=0.5), 0.4)

    def test_rejects_invalid_alpha(self):
        probs, labels = self.probs_for_scores([0.2])
        for alpha in (0, 1, -0.1, 1.1, math.nan, math.inf):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                fit_conformal_quantile(probs, labels, alpha=alpha)

    def test_rejects_empty_calibration_data(self):
        with self.assertRaises(ValueError):
            fit_conformal_quantile(torch.empty((0, 2)), torch.empty((0,), dtype=torch.long))

    def test_rejects_shape_label_and_probability_errors(self):
        with self.assertRaises(ValueError):
            fit_conformal_quantile(torch.ones((2, 2)) / 2, torch.zeros((2, 1), dtype=torch.long))
        with self.assertRaises(ValueError):
            fit_conformal_quantile(torch.tensor([[0.5, 0.5]]), torch.tensor([2]))
        with self.assertRaises(ValueError):
            fit_conformal_quantile(torch.tensor([[0.2, 0.2]]), torch.tensor([0]))
        with self.assertRaises(ValueError):
            fit_conformal_quantile(torch.tensor([[math.nan, math.nan]]), torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
