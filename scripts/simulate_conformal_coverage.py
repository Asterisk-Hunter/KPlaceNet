"""Synthetic exchangeability check for the finite-sample conformal rank.

This is a mathematical smoke check, not a geolocation/model evaluation.
Run with: python scripts/simulate_conformal_coverage.py --trials 20000 --seed 2026
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")

    n = 9
    alpha = 0.01
    rank = int(torch.ceil(torch.tensor((n + 1) * (1 - alpha))).item())
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    scores = torch.rand((args.trials, n + 1), generator=generator)
    # The finite-sample rule has rank n+1 here, hence q=+infinity and coverage 1.
    discrete_coverage = 1.0
    # The previous code clamped the fractional quantile level to 1, selecting
    # max(calibration scores); compare that threshold on each held-out score.
    interpolated_cutoffs = scores[:, :n].max(dim=1).values
    old_coverage = (scores[:, n] <= interpolated_cutoffs).float().mean().item()

    print(f"Synthetic iid Uniform scores: trials={args.trials}, n={n}, alpha={alpha}, seed={args.seed}")
    print(f"Finite-sample rank k={rank}; discrete threshold=+infinity; coverage={discrete_coverage:.4f}")
    print(f"Previous clamped maximum-score cutoff; empirical coverage={old_coverage:.4f}")
    print("Synthetic exchangeable-score diagnostic only; not a real-data result or shift guarantee.")


if __name__ == "__main__":
    main()
