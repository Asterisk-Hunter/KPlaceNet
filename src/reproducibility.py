"""Deterministic calibration/evaluation split and sample provenance helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def make_l3_split(
    sample_count: int, temperature_size: int, calibration_size: int, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Return deterministic, disjoint temperature/calibration/evaluation indices.

    The split is a seeded random permutation of the input CSV rows. It does
    not make geographically dependent examples exchangeable; it only removes
    dependence on CSV row ordering and makes the exact partition reproducible.
    Temperature fitting and conformal threshold fitting use separate samples,
    so the score function is fixed independently of conformal calibration data.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2:
        raise ValueError("sample_count must be an integer >= 2")
    for name, size in (("temperature_size", temperature_size), ("calibration_size", calibration_size)):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"{name} must be a positive integer")
    if temperature_size + calibration_size >= sample_count:
        raise ValueError("temperature and calibration splits must leave evaluation samples")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(sample_count, generator=generator).tolist()
    temperature = sorted(permutation[:temperature_size])
    calibration = sorted(permutation[temperature_size:temperature_size + calibration_size])
    used = set(temperature) | set(calibration)
    evaluation = [index for index in range(sample_count) if index not in used]

    if (
        set(temperature) & set(calibration)
        or set(temperature) & set(evaluation)
        or set(calibration) & set(evaluation)
        or len(temperature) + len(calibration) + len(evaluation) != sample_count
    ):
        raise RuntimeError("temperature/calibration/evaluation split is not disjoint and exhaustive")
    return temperature, calibration, evaluation


def stable_sample_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Build unique IDs from source IDs or canonical image/location metadata.

    If the CSV has a non-empty ``id`` column, it is preserved with an ``id:``
    prefix. Otherwise each identifier is a SHA-256 digest of the path and
    coordinates, making the manifest useful without exposing raw metadata.
    Duplicate identities are rejected instead of silently making the split
    manifest ambiguous.
    """
    sample_ids: list[str] = []
    for row_number, row in enumerate(rows):
        source_id = row.get("id")
        source_id_text = "" if source_id is None else str(source_id).strip()
        source_id_is_missing = source_id_text.lower() in {"", "nan", "<na>", "none"}
        if not source_id_is_missing:
            identifier = f"id:{source_id_text}"
        else:
            try:
                identity = {
                    "image_path": str(row["image_path"]),
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                }
                payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"row {row_number} has invalid sample identity fields") from error
            identifier = "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
        sample_ids.append(identifier)

    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample identifiers must be unique to audit split overlap")
    return sample_ids


def sha256_file(path: str) -> str:
    """Hash a file incrementally, without reading it all into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_manifest(
    *,
    temperature_indices: Sequence[int],
    calibration_indices: Sequence[int],
    evaluation_indices: Sequence[int],
    sample_ids: Sequence[str],
    seed: int,
    csv_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Create a JSON-ready manifest that records exact split membership."""
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample identifiers must be unique")
    temperature_list = list(temperature_indices)
    calibration_list = list(calibration_indices)
    evaluation_list = list(evaluation_indices)
    if (
        len(set(temperature_list)) != len(temperature_list)
        or len(set(calibration_list)) != len(calibration_list)
        or len(set(evaluation_list)) != len(evaluation_list)
    ):
        raise ValueError("split index lists must not contain duplicates")
    temperature_set, calibration_set, evaluation_set = (
        set(temperature_list), set(calibration_list), set(evaluation_list)
    )
    if (
        temperature_set & calibration_set
        or temperature_set & evaluation_set
        or calibration_set & evaluation_set
    ):
        raise ValueError("temperature, calibration, and evaluation indices must be disjoint")
    all_indices = temperature_set | calibration_set | evaluation_set
    if all_indices != set(range(len(sample_ids))):
        raise ValueError("split indices must cover every sample exactly once")

    temperature_ids = [sample_ids[index] for index in temperature_list]
    calibration_ids = [sample_ids[index] for index in calibration_list]
    evaluation_ids = [sample_ids[index] for index in evaluation_list]
    if (
        set(temperature_ids) & set(calibration_ids)
        or set(temperature_ids) & set(evaluation_ids)
        or set(calibration_ids) & set(evaluation_ids)
    ):
        raise ValueError("temperature, calibration, and evaluation sample IDs must be disjoint")
    manifest = {
        "method": "torch.randperm(cpu)",
        "seed": seed,
        "csv_sha256": csv_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "temperature_indices": temperature_list,
        "calibration_indices": calibration_list,
        "evaluation_indices": evaluation_list,
        "temperature_sample_ids": temperature_ids,
        "calibration_sample_ids": calibration_ids,
        "evaluation_sample_ids": evaluation_ids,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["split_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return manifest
