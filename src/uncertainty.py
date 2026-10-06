"""Uncertainty — L3 implementations.

Locked plan Section 4 L3:
  - Temperature scaling on L2 winner (Guo et al. 2017)
  - Conformal prediction for prediction sets
  - Abstention threshold + region-wise reliability
  - Metrics: ECE (target <0.1), coverage vs accuracy tradeoff

References:
  - Guo et al. "On Calibration of Modern Neural Networks" (2017)
  - Sadinle et al. "Least-favorable testing for conformal prediction" (2019)
  - Angelopoulos & Bates "A Gentle Introduction to Conformal Prediction" (2023)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Temperature scaling — Guo et al. 2017
# ---------------------------------------------------------------------------


class TemperatureScaler(nn.Module):
    """Temperature scaling for post-hoc calibration.

    Learns a single scalar T > 0 on a held-out calibration set by minimizing
    NLL (cross-entropy) of the temperature-scaled logits.  The model forward
    is simply ``logits / T``, so probabilities become ``softmax(logits / T)``.

    Usage:
        scaler = TemperatureScaler()
        scaler.fit(cal_logits, cal_labels)    # optimizes T via LBFGS
        scaled_logits = scaler(eval_logits)   # logits / T
        probs = F.softmax(scaled_logits, dim=1)

    After ``fit()`` the temperature attribute is a learned ``nn.Parameter``
    with an optimised value; call ``.item()`` to get the float.
    """

    def __init__(self, init_temp: float = 1.0) -> None:
        super().__init__()
        self.temperature: nn.Parameter = nn.Parameter(torch.tensor(float(init_temp)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature.clamp(min=1e-3)

    @torch.no_grad()
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
        device: Optional[torch.device] = None,
    ) -> float:
        """Optimize temperature T on calibration logits via LBFGS.

        Minimizes cross-entropy ``CE(logits / T, labels)`` w.r.t. T only.
        The caller's model is untouched; only ``self.temperature`` is updated.

        Args:
            logits: (N, K) raw model outputs on calibration set.
            labels: (N,) ground-truth cell indices (long).
            lr: learning rate for LBFGS (0.01 default).
            max_iter: max LBFGS iterations (100 default, typically converges
                      in <30).
            device: optional device to place tensors on (uses logits device
                    if None).

        Returns:
            Fitted temperature as a float.
        """
        if device is None:
            device = logits.device

        logits = logits.to(device)
        labels = labels.to(device)

        # Ensure float32 for optimizer
        logits_f = logits.float()
        labels_l = labels.long()

        # Reset temperature to init (default 1.0)
        self.temperature.data.fill_(1.0)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=lr,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def eval_step() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = logits_f / self.temperature.clamp(min=1e-3)
            loss = nll_criterion(scaled, labels_l)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        # Clamp to reasonable range for safety
        T_val = float(self.temperature.item())
        if T_val < 0.01:
            T_val = 0.01
            self.temperature.data.fill_(T_val)
        elif T_val > 100.0:
            T_val = 100.0
            self.temperature.data.fill_(T_val)

        return T_val


# ---------------------------------------------------------------------------
# ECE — Expected Calibration Error
# ---------------------------------------------------------------------------


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) with uniform confidence bins.

    Bins predictions by confidence (max probability).  For each bin, measures
    the absolute gap between mean confidence and mean accuracy, weighted by
    the fraction of samples in that bin.

    Args:
        probs: (N, K) softmax probabilities (temperature-scaled preferred).
        labels: (N,) ground-truth cell indices (long).
        n_bins: number of equal-width bins over [0, 1] (default 15).

    Returns:
        ECE as a float in [0, 1].  Target < 0.1 (Section 9); stretch < 0.05.
    """
    probs = probs.float()
    labels = labels.long()

    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels)

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = 0.0

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        # Inclusive left, exclusive right except last bin is inclusive right
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        if mask.any():
            frac = mask.float().mean().item()
            bin_acc = correct[mask].float().mean().item()
            bin_conf = conf[mask].mean().item()
            ece += frac * abs(bin_acc - bin_conf)

    return float(ece)


# ---------------------------------------------------------------------------
# Conformal prediction — split conformal
# ---------------------------------------------------------------------------


def fit_conformal_quantile(
    probs_cal: torch.Tensor,
    labels_cal: torch.Tensor,
    alpha: float = 0.1,
) -> float:
    """Fit a conformal quantile threshold from calibration data.

    Uses the nonconformity score ``s_i = 1 - p_i[y_i]`` (probability of the
    true class). Let ``k = ceil((n + 1) * (1 - alpha))``. The threshold is the
    *k-th smallest observed score*, not a linearly interpolated quantile. If
    ``k > n``, it is ``+inf`` (the conservative finite-sample convention).
    Under exchangeability of calibration and future examples this gives
    marginal coverage at least ``1 - alpha`` (Angelopoulos & Bates, 2023,
    Eq. 3.3). This guarantee does not imply conditional or region-wise
    coverage, and need not hold under geographic distribution shift.

    Args:
        probs_cal: (N, K) softmax probabilities on calibration set.
        labels_cal: (N,) ground-truth labels for calibration set.
        alpha: miscoverage rate, strictly between 0 and 1 (default 0.1).

    Returns:
        Quantile threshold ``q`` as a float.  Use with ``conformal_prediction_set``.
    """
    if probs_cal.ndim != 2 or probs_cal.shape[1] == 0:
        raise ValueError("probs_cal must have shape (N, K) with K > 0")
    if labels_cal.ndim != 1 or labels_cal.shape[0] != probs_cal.shape[0]:
        raise ValueError("labels_cal must have shape (N,) matching probs_cal")
    if probs_cal.shape[0] == 0:
        raise ValueError("calibration data must contain at least one example")
    if labels_cal.dtype == torch.bool or labels_cal.is_floating_point() or labels_cal.is_complex():
        raise ValueError("labels_cal must contain integer class indices")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha must be a finite number strictly between 0 and 1") from error
    if not torch.isfinite(torch.tensor(alpha)) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be a finite number strictly between 0 and 1")

    probs_cal = probs_cal.float()
    labels_cal = labels_cal.to(device=probs_cal.device, dtype=torch.long)
    if not torch.isfinite(probs_cal).all():
        raise ValueError("probs_cal must contain only finite values")
    if ((probs_cal < 0) | (probs_cal > 1)).any():
        raise ValueError("probs_cal values must be in [0, 1]")
    if not torch.allclose(
        probs_cal.sum(dim=1),
        torch.ones(probs_cal.shape[0], device=probs_cal.device),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("each row of probs_cal must sum to 1")

    n = probs_cal.shape[0]
    if (labels_cal < 0).any() or (labels_cal >= probs_cal.shape[1]).any():
        raise ValueError("labels_cal contains a class index outside probs_cal")
    # Nonconformity: 1 - prob of true class
    true_probs = probs_cal[torch.arange(n, device=probs_cal.device), labels_cal]
    scores = 1.0 - true_probs

    # Split-conformal finite-sample order statistic. torch.quantile's default
    # linear interpolation does not implement the required discrete rank.
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")
    q = float(torch.kthvalue(scores, rank).values.item())

    return q


def conformal_prediction_set(
    probs: torch.Tensor,
    alpha: float = 0.1,
    quantile: Optional[float] = None,
) -> torch.Tensor:
    """Construct conformal prediction sets via split conformal method.

    For each sample, includes class ``k`` in the prediction set if its
    nonconformity score ``1 - p_k`` is at most the calibrated quantile
    threshold ``q``:

        ``C(x) = { k : 1 - p_k(x) <= q }``

    When ``quantile`` is None (no calibration), falls back to top-1 prediction
    only (L0 stub behavior).

    Args:
        probs: (N, K) softmax probabilities (temperature-scaled preferred).
        alpha: miscoverage rate; default 0.1.  Ignored when ``quantile`` is set.
        quantile: pre-fitted threshold from ``fit_conformal_quantile``.
                  If None, returns top-1 only (no conformal guarantee).

    Returns:
        mask: (N, K) bool tensor — True where class k is in the prediction set.
    """
    probs = probs.float()
    n, k = probs.shape

    if quantile is None:
        # Fallback: top-1 only (no conformal guarantee)
        mask = torch.zeros(n, k, dtype=torch.bool, device=probs.device)
        top1 = probs.argmax(dim=1)
        mask[torch.arange(n, device=probs.device), top1] = True
        return mask

    # Split-conformal: include class k if 1 - p_k <= q
    nonconformity = 1.0 - probs
    mask = nonconformity <= quantile

    # Guarantee at least one class per sample (the max prob class)
    max_prob_idx = probs.argmax(dim=1)
    mask[torch.arange(n, device=probs.device), max_prob_idx] = True

    return mask


# ---------------------------------------------------------------------------
# Abstention — confidence threshold
# ---------------------------------------------------------------------------


def abstention_mask(
    probs: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Decide which samples the model should predict vs abstain.

    A sample is predicted when its maximum softmax probability exceeds the
    threshold; otherwise the model abstains (outputs "don't know").

    Args:
        probs: (N, K) softmax probabilities (ideally temperature-scaled).
        threshold: abstain if max_prob < threshold (default 0.5).

    Returns:
        should_predict: (N,) bool — True if model should emit prediction.
    """
    max_prob = probs.max(dim=1).values
    return max_prob >= threshold
