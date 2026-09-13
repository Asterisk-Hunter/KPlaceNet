"""Uncertainty — stub for L3 (Gap 3).

Locked plan Section 4 L3:
  - Temperature scaling on L2 winner
  - Conformal prediction for prediction sets / intervals
  - Abstention threshold + region-wise reliability
  - Metrics: ECE (target <0.1), coverage vs accuracy tradeoff

L0: stubs only. No training logic, no extra deps. Keeps imports working.
"""

from __future__ import annotations

from typing import Tuple, Optional
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Temperature scaling — Guo et al. 2017
# ---------------------------------------------------------------------------


class TemperatureScaler(nn.Module):
    """Temperature scaling stub.

    TODO L3: learn temperature T on held-out val set by minimizing NLL.

    Usage (L3):
        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels)  # optimizes T via LBFGS
        scaled_logits = scaler(logits)      # logits / T
        probs = softmax(scaled_logits)

    L0: identity (T=1.0).
    """

    def __init__(self, init_temp: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(init_temp)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """TODO L3: optimize temperature on val set.

        Stub returns current temperature without optimization.

        Args:
            logits: (N, K) unscaled logits
            labels: (N,) true cell ids

        Returns:
            fitted temperature as float
        """
        # TODO L3: implement NLL minimization w.r.t. temperature
        #   - Use torch.optim.LBFGS on self.temperature
        #   - Loss = CrossEntropy(logits / T, labels)
        #   - Freeze model, only optimize T
        #   - Expected: ECE < 0.1 after scaling
        return float(self.temperature.item())


# ---------------------------------------------------------------------------
# Conformal prediction — stub
# ---------------------------------------------------------------------------


def conformal_prediction_set(
    probs: torch.Tensor,
    alpha: float = 0.1,
    quantile: Optional[float] = None,
) -> torch.Tensor:
    """Conformal prediction set stub.

    TODO L3: implement split-conformal with calibration quantile.

    Args:
        probs: (N, K) softmax probabilities
        alpha: miscoverage rate (target coverage = 1 - alpha)
        quantile: precomputed quantile threshold (from cal set). If None, stub uses top-1.

    Returns:
        mask: (N, K) bool tensor indicating prediction set per sample
              L0 stub: top-1 only.

    TODO L3:
        - Compute nonconformity scores on cal set: s = 1 - prob_true_class
        - q = quantile(s, 1 - alpha)
        - Prediction set = {k : 1 - prob_k <= q}  (or cumulative prob variant)
        - Report coverage vs set size tradeoff
    """
    # L0 stub: single prediction (argmax)
    n, k = probs.shape
    top1 = probs.argmax(dim=1)
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[torch.arange(n), top1] = True
    return mask


# ---------------------------------------------------------------------------
# Abstention — stub
# ---------------------------------------------------------------------------


def abstention_mask(
    probs: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Decide which samples to abstain from.

    TODO L3: threshold on max softmax (confidence) or on temperature-scaled
             confidence, with region-wise reliability analysis.

    Args:
        probs: (N, K) softmax probabilities (ideally temperature-scaled)
        threshold: abstain if max_prob < threshold

    Returns:
        should_predict: (N,) bool — True if model should emit prediction

    Metrics to log (L3 exit):
        - coverage = mean(should_predict)
        - accuracy_when_predicting
        - abstention rate vs ECE
    """
    max_prob = probs.max(dim=1).values
    return max_prob >= threshold


# ---------------------------------------------------------------------------
# ECE — Expected Calibration Error
# ---------------------------------------------------------------------------


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """ECE stub.

    TODO L3: implement standard binned ECE.
        - Bin by confidence (max prob)
        - Per bin: |acc - conf| weighted by bin size
        - Target < 0.1 (Section 9), stretch < 0.05

    L0: returns 0.0 placeholder so eval pipeline imports cleanly.
    """
    # TODO L3: real implementation:
    #   conf, pred = probs.max(dim=1)
    #   correct = pred.eq(labels)
    #   bins = torch.linspace(0, 1, n_bins+1)
    #   ece = 0.
    #   for b in range(n_bins):
    #       mask = (conf > bins[b]) & (conf <= bins[b+1])
    #       if mask.any():
    #           ece += mask.float().mean() * abs(correct[mask].float().mean() - conf[mask].mean())
    #   return float(ece)
    return 0.0
