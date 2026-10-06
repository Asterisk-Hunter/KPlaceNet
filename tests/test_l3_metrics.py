"""Tests for the L3 geodesic-distance evaluation helper."""

from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from scripts.run_l3_uncertainty import eval_haversine


class EvalHaversineTests(unittest.TestCase):
    def test_reports_known_geodesic_threshold_metrics(self):
        probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        centroids = np.array([[0.0, 0.0], [0.0, 1.0]])
        metrics = eval_haversine(
            probs,
            torch.tensor([0, 0]),
            centroids,
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        )

        self.assertEqual(metrics["within_1km"], 50.0)
        self.assertEqual(metrics["within_25km"], 50.0)
        self.assertEqual(metrics["within_200km"], 100.0)
        self.assertAlmostEqual(metrics["mean_km"], 55.597, places=2)
        self.assertEqual(metrics["n"], 2)

    def test_empty_abstention_subset_is_valid_json_with_explicit_missing_distance(self):
        probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        metrics = eval_haversine(
            probs,
            torch.tensor([0, 1]),
            np.array([[0.0, 0.0], [0.0, 1.0]]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            mask=torch.tensor([False, False]),
        )

        self.assertIsNone(metrics["mean_km"])
        self.assertIsNone(metrics["median_km"])
        self.assertEqual(metrics["n"], 0)
        self.assertNotIn("NaN", json.dumps(metrics))

    def test_rejects_mismatched_coordinate_or_mask_lengths(self):
        probs = torch.tensor([[1.0, 0.0]])
        centroids = np.array([[0.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            eval_haversine(
                probs, torch.tensor([0]), centroids, np.array([]), np.array([0.0])
            )
        with self.assertRaises(ValueError):
            eval_haversine(
                probs,
                torch.tensor([0]),
                centroids,
                np.array([0.0]),
                np.array([0.0]),
                mask=torch.tensor([True, False]),
            )


if __name__ == "__main__":
    unittest.main()
