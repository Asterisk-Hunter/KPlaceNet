"""Tests for deterministic, auditable calibration/evaluation splits."""

from __future__ import annotations

import unittest

from src.reproducibility import make_l3_split, split_manifest, stable_sample_ids


class ReproducibilityTests(unittest.TestCase):
    def test_split_is_seeded_disjoint_and_exhaustive(self):
        first = make_l3_split(10, 2, 4, 42)
        self.assertEqual(first, make_l3_split(10, 2, 4, 42))
        self.assertNotEqual(first[0], make_l3_split(10, 2, 4, 43)[0])
        temperature, calibration, evaluation = first
        self.assertFalse(set(temperature) & set(calibration))
        self.assertFalse(set(temperature) & set(evaluation))
        self.assertFalse(set(calibration) & set(evaluation))
        self.assertEqual(
            set(temperature) | set(calibration) | set(evaluation), set(range(10))
        )
        self.assertEqual((len(temperature), len(calibration), len(evaluation)), (2, 4, 4))

    def test_split_rejects_empty_or_exhaustive_partition(self):
        for values in ((1, 1, 1, 1), (5, 0, 1, 1), (5, 2, 3, 1), (5, 1, 1, -1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                make_l3_split(*values)

    def test_stable_sample_ids_use_source_id_or_metadata_hash(self):
        rows = [
            {"id": "source-1", "image_path": "a.jpg", "lat": 10, "lon": 20},
            {"id": float("nan"), "image_path": "b.jpg", "lat": 11, "lon": 21},
        ]
        ids = stable_sample_ids(rows)
        self.assertEqual(ids[0], "id:source-1")
        self.assertTrue(ids[1].startswith("sha256:"))
        self.assertEqual(ids[1], stable_sample_ids([rows[1]])[0])

    def test_duplicate_identities_are_rejected(self):
        row = {"image_path": "a.jpg", "lat": 10, "lon": 20}
        with self.assertRaises(ValueError):
            stable_sample_ids([row, dict(row)])

    def test_manifest_records_ids_and_is_stable(self):
        ids = [f"id:{i}" for i in range(5)]
        temperature, cal, evaluation = make_l3_split(5, 1, 2, 7)
        manifest = split_manifest(
            temperature_indices=temperature,
            calibration_indices=cal,
            evaluation_indices=evaluation,
            sample_ids=ids,
            seed=7,
            csv_sha256="csv-hash",
            checkpoint_sha256="checkpoint-hash",
        )
        self.assertEqual(
            manifest["temperature_sample_ids"],
            [ids[index] for index in temperature],
        )
        self.assertEqual(manifest["calibration_sample_ids"], [ids[index] for index in cal])
        self.assertEqual(manifest["evaluation_sample_ids"], [ids[index] for index in evaluation])
        self.assertEqual(
            manifest["split_sha256"],
            split_manifest(
                temperature_indices=temperature,
                calibration_indices=cal,
                evaluation_indices=evaluation,
                sample_ids=ids,
                seed=7,
                csv_sha256="csv-hash",
                checkpoint_sha256="checkpoint-hash",
            )["split_sha256"],
        )

    def test_manifest_rejects_overlap_and_incomplete_indices(self):
        ids = ["a", "b", "c"]
        base = {
            "sample_ids": ids,
            "seed": 1,
            "csv_sha256": "csv",
            "checkpoint_sha256": "checkpoint",
        }
        for temp, cal, evaluation in (
            ([0], [0, 1], [2]),
            ([0], [1], []),
            ([0], [1, 1], [2]),
        ):
            with self.subTest(cal=cal, evaluation=evaluation), self.assertRaises(ValueError):
                split_manifest(
                    temperature_indices=temp,
                    calibration_indices=cal,
                    evaluation_indices=evaluation,
                    **base,
                )


if __name__ == "__main__":
    unittest.main()
