"""Spatially stratified metrics for uncertainty-aware geolocation.

The equal-area grid is a descriptive diagnostic: it exposes geographic
heterogeneity that a single global coverage number can hide. It does not
provide conditional coverage guarantees, especially when locations are
dependent or evaluation data differ from calibration data.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from numbers import Integral
from typing import Any


_EARTH_RADIUS_KM = 6371.0
_WILSON_Z_95 = 1.959963984540054


def _wilson_interval(successes: int, count: int) -> list[float]:
    """Return a two-sided 95% Wilson interval for a binomial proportion."""
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("successes and count must define a non-empty binomial sample")
    z2 = _WILSON_Z_95**2
    proportion = successes / count
    denominator = 1.0 + z2 / count
    center = (proportion + z2 / (2 * count)) / denominator
    margin = (
        _WILSON_Z_95
        * math.sqrt(proportion * (1.0 - proportion) / count + z2 / (4 * count**2))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _equal_area_region(
    lat: float, lon: float, lat_bands: int, lon_bands: int
) -> tuple[int, int]:
    """Map WGS84 coordinates to a deterministic equal-area latitude/longitude grid."""
    if not math.isfinite(lat) or not -90.0 <= lat <= 90.0:
        raise ValueError("latitude must be finite and in [-90, 90]")
    if not math.isfinite(lon) or not -180.0 <= lon <= 180.0:
        raise ValueError("longitude must be finite and in [-180, 180]")

    # Uniform bands in sin(latitude), combined with uniform longitude bands,
    # have equal spherical area (up to the grid's discretization).
    area_fraction = (math.sin(math.radians(lat)) + 1.0) / 2.0
    lat_index = min(lat_bands - 1, int(area_fraction * lat_bands))
    wrapped_lon = (lon + 180.0) % 360.0
    lon_index = min(lon_bands - 1, int(wrapped_lon / 360.0 * lon_bands))
    return lat_index, lon_index


def spatial_stratified_metrics(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    prediction_sets: Sequence[Sequence[bool]],
    true_lats: Sequence[float],
    true_lons: Sequence[float],
    centroids: Sequence[Sequence[float]],
    *,
    lat_bands: int = 6,
    lon_bands: int = 12,
    min_count: int = 1,
) -> dict[str, Any]:
    """Report accuracy, set coverage/size, and geodesic error per equal-area cell.

    ``prediction_sets`` and probabilities use the same class order. Every
    non-empty grid cell is returned, including cells with fewer than
    ``min_count`` rows; those entries get ``metrics: None`` so sparse strata
    remain visible without encouraging unstable interpretation. Coverage
    intervals are descriptive Wilson intervals and do not account for spatial
    dependence or selection of the grid.
    """
    sample_count = len(probabilities)
    if not isinstance(lat_bands, int) or isinstance(lat_bands, bool) or lat_bands < 1:
        raise ValueError("lat_bands must be a positive integer")
    if not isinstance(lon_bands, int) or isinstance(lon_bands, bool) or lon_bands < 1:
        raise ValueError("lon_bands must be a positive integer")
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    if sample_count == 0:
        raise ValueError("evaluation data must contain at least one sample")
    aligned = (labels, prediction_sets, true_lats, true_lons)
    if any(len(values) != sample_count for values in aligned):
        raise ValueError("all per-sample inputs must have matching lengths")
    if len(centroids) == 0:
        raise ValueError("centroids must contain at least one [latitude, longitude] pair")
    class_count = len(centroids)
    for point in centroids:
        if len(point) != 2:
            raise ValueError("each centroid must contain [latitude, longitude]")
        if not all(math.isfinite(float(value)) for value in point):
            raise ValueError("centroids must contain finite coordinates")
        if (
            not -90.0 <= float(point[0]) <= 90.0
            or not -180.0 <= float(point[1]) <= 180.0
        ):
            raise ValueError("centroids must use valid latitude and longitude ranges")
    if any(len(row) != class_count for row in probabilities):
        raise ValueError("probability rows must match the centroid class count")
    if any(len(row) != class_count for row in prediction_sets):
        raise ValueError("prediction-set rows must match the centroid class count")

    groups: dict[tuple[int, int], dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "correct": 0, "covered": 0, "set_size": 0, "distances": []}
    )
    for probs, label, prediction_set, lat, lon in zip(
        probabilities, labels, prediction_sets, true_lats, true_lons, strict=True
    ):
        if (
            isinstance(label, bool)
            or not isinstance(label, Integral)
            or not 0 <= label < class_count
        ):
            raise ValueError("labels must be integer class indices in range")
        label = int(label)
        if (
            len(probs) == 0
            or any(not math.isfinite(float(p)) or not 0.0 <= float(p) <= 1.0 for p in probs)
            or not math.isclose(sum(map(float, probs)), 1.0, rel_tol=1e-5, abs_tol=1e-6)
        ):
            raise ValueError("each probability row must be finite, in [0, 1], and sum to 1")
        if not all(math.isfinite(float(value)) for value in (lat, lon)):
            raise ValueError("evaluation coordinates must be finite")
        region = _equal_area_region(float(lat), float(lon), lat_bands, lon_bands)
        group = groups[region]
        prediction = max(range(class_count), key=lambda index: float(probs[index]))
        group["n"] += 1
        group["correct"] += int(prediction == label)
        group["covered"] += int(bool(prediction_set[label]))
        group["set_size"] += sum(bool(value) for value in prediction_set)

        pred_lat, pred_lon = map(float, centroids[prediction])
        true_lat, true_lon = float(lat), float(lon)
        phi1, phi2 = math.radians(pred_lat), math.radians(true_lat)
        dphi = phi2 - phi1
        dlam = math.radians(true_lon - pred_lon)
        hav = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        hav = min(1.0, max(0.0, hav))
        group["distances"].append(2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(hav)))

    result: dict[str, Any] = {}
    for (lat_index, lon_index), group in sorted(groups.items()):
        n = group["n"]
        key = f"lat{lat_index:02d}_lon{lon_index:02d}"
        if n < min_count:
            result[key] = {"n": n, "metrics": None}
            continue
        distances = group["distances"]
        result[key] = {
            "n": n,
            "metrics": {
                "top1_accuracy": group["correct"] / n,
                "conformal_coverage": group["covered"] / n,
                "conformal_coverage_wilson_95": _wilson_interval(group["covered"], n),
                "mean_prediction_set_size": group["set_size"] / n,
                "mean_distance_km": sum(distances) / n,
                "within_1km": sum(distance <= 1.0 for distance in distances) / n,
                "within_25km": sum(distance <= 25.0 for distance in distances) / n,
                "within_200km": sum(distance <= 200.0 for distance in distances) / n,
            },
        }
    return {
        "grid": {
            "method": "equal_area_latitude_bands_x_uniform_longitude_bands",
            "latitude_bands": lat_bands,
            "longitude_bands": lon_bands,
            "minimum_count_for_metrics": min_count,
        },
        "regions": result,
        "limitations": (
            "Descriptive regional diagnostics only; marginal split-conformal guarantees "
            "do not imply conditional or region-wise coverage. Wilson intervals assume "
            "independent Bernoulli observations and are not corrected for spatial dependence."
        ),
    }
