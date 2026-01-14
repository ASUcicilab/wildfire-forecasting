"""Evaluation metrics for wildfire spread prediction."""

from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        y_pred: Predicted probabilities (flattened)
        y_true: Ground truth labels (flattened)
        threshold: Classification threshold

    Returns:
        Dict of metric names to values
    """
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    y_pred_binary = (y_pred_flat > threshold).astype(int)

    metrics = {
        "ap": average_precision_score(y_true_flat, y_pred_flat),
        "f1": f1_score(y_true_flat, y_pred_binary),
        "precision": precision_score(y_true_flat, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true_flat, y_pred_binary, zero_division=0),
        "iou": jaccard_score(y_true_flat, y_pred_binary, zero_division=0),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_binary)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        metrics["tp"] = int(tp)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
        metrics["tn"] = int(tn)

    return metrics


def compute_per_sample_ap(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> List[float]:
    """Compute Average Precision for each sample.

    Args:
        y_pred: Predictions of shape (N, H, W)
        y_true: Targets of shape (N, H, W)

    Returns:
        List of AP values per sample
    """
    aps = []
    for pred, gt in zip(y_pred, y_true):
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()

        if gt_flat.sum() > 0:  # Only if there are positive pixels
            ap = average_precision_score(gt_flat, pred_flat)
        else:
            ap = float("nan")
        aps.append(ap)

    return aps


def compute_stratified_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    group_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics stratified by groups.

    Args:
        y_pred: Predictions of shape (N, H, W)
        y_true: Targets of shape (N, H, W)
        groups: Group assignment of shape (N,)
        group_names: Optional mapping from group id to name

    Returns:
        Dict of group name to metrics dict
    """
    unique_groups = np.unique(groups)
    results = {}

    for g in unique_groups:
        mask = groups == g
        g_name = group_names[g] if group_names and g in group_names else str(g)

        g_pred = y_pred[mask]
        g_true = y_true[mask]

        results[g_name] = compute_metrics(g_pred, g_true)
        results[g_name]["n_samples"] = int(mask.sum())

    return results


def compute_channel_nmi(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute Normalized Mutual Information between each channel and targets.

    Args:
        x: Input features of shape (N, T, C, H, W) or (N, C, H, W)
        y_true: Ground truth masks of shape (N, H, W)
        y_pred: Optional predicted masks of shape (N, H, W)

    Returns:
        Dict with 'relevance' (NMI with GT) and optionally 'reliance' (NMI with pred)
    """
    # Flatten temporal dimension if present
    if x.ndim == 5:
        x = x[:, -1]  # Use last timestep

    N, C, H, W = x.shape

    # Compute NMI for each channel
    nmi_gt = np.zeros(C)
    nmi_pred = np.zeros(C) if y_pred is not None else None

    for c in range(C):
        # Flatten spatial dimensions
        x_c = x[:, c].flatten()
        gt_flat = y_true.flatten()

        # Discretize continuous features for NMI
        x_c_binned = np.digitize(x_c, np.percentile(x_c, np.linspace(0, 100, 21)))

        nmi_gt[c] = normalized_mutual_info_score(gt_flat, x_c_binned)

        if y_pred is not None:
            pred_flat = (y_pred > 0.5).astype(int).flatten()
            nmi_pred[c] = normalized_mutual_info_score(pred_flat, x_c_binned)

    result = {"relevance": nmi_gt}
    if nmi_pred is not None:
        result["reliance"] = nmi_pred

        # Spearman correlation between relevance and reliance
        corr, pval = spearmanr(nmi_gt, nmi_pred)
        result["spearman_corr"] = corr
        result["spearman_pval"] = pval

    return result


def compute_integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    n_steps: int = 50,
    baseline: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Integrated Gradients attribution.

    Args:
        model: Model to explain
        x: Input tensor of shape (B, T, C, H, W)
        target: Target for attribution (not used for binary segmentation)
        n_steps: Number of integration steps
        baseline: Baseline input (default: zeros)

    Returns:
        Attribution tensor of same shape as x
    """
    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(x)

    # Create interpolation path
    alphas = torch.linspace(0, 1, n_steps, device=x.device)

    # Accumulate gradients
    gradients = torch.zeros_like(x)

    x.requires_grad_(True)

    for alpha in alphas:
        # Interpolated input
        x_interp = baseline + alpha * (x - baseline)
        x_interp = x_interp.requires_grad_(True)

        # Forward pass
        output = model(x_interp)

        # Sum output for gradient
        output.sum().backward()

        gradients += x_interp.grad
        x_interp.grad.zero_()

    # Average and multiply by input difference
    attributions = gradients / n_steps * (x - baseline)

    return attributions.detach()
