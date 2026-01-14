"""Visualization utilities for CMPF experiments."""

from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_prediction_comparison(
    gt: np.ndarray,
    predictions: Dict[str, np.ndarray],
    event_id: str,
    save_path: Optional[str] = None,
    show_metrics: bool = True,
    figsize: Tuple[int, int] = (16, 4),
) -> plt.Figure:
    """Plot ground truth vs multiple model predictions.

    Args:
        gt: Ground truth mask (H, W)
        predictions: Dict of model_name -> prediction (H, W)
        event_id: Event identifier for title
        save_path: Path to save figure
        show_metrics: Whether to show per-sample AP
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_cols = 1 + len(predictions)  # GT + predictions
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    # Plot GT
    axes[0].imshow(gt, cmap="Reds", vmin=0, vmax=1)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    # Plot predictions
    for i, (name, pred) in enumerate(predictions.items(), 1):
        axes[i].imshow(pred, cmap="Reds", vmin=0, vmax=1)

        if show_metrics:
            from .metrics import compute_metrics

            metrics = compute_metrics(pred, gt)
            title = f"{name}\nAP: {metrics['ap']:.3f}"
        else:
            title = name

        axes[i].set_title(title)
        axes[i].axis("off")

    plt.suptitle(f"Event: {event_id}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_tp_fp_fn_map(
    gt: np.ndarray,
    pred: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 6),
) -> plt.Figure:
    """Plot TP/FP/FN color-coded map.

    Args:
        gt: Ground truth binary mask (H, W)
        pred: Predicted probabilities (H, W)
        threshold: Classification threshold
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    pred_binary = pred > threshold

    # Create TP/FP/FN map
    # 0: TN (background), 1: TP (green), 2: FP (red), 3: FN (blue)
    result = np.zeros_like(gt, dtype=np.uint8)
    result[(gt == 1) & (pred_binary == 1)] = 1  # TP
    result[(gt == 0) & (pred_binary == 1)] = 2  # FP
    result[(gt == 1) & (pred_binary == 0)] = 3  # FN

    # Custom colormap
    colors = ["white", "green", "red", "blue"]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(result, cmap=cmap, vmin=0, vmax=3)

    # Legend
    patches = [
        mpatches.Patch(color="green", label="TP"),
        mpatches.Patch(color="red", label="FP"),
        mpatches.Patch(color="blue", label="FN"),
    ]
    ax.legend(handles=patches, loc="upper right")
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Plot training curves.

    Args:
        metrics: Dict of metric_name -> list of values per epoch
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, metrics.items()):
        ax.plot(values)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_efficiency_comparison(
    results: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot training efficiency comparison (AP vs epochs).

    Args:
        results: Dict of method_name -> {'epochs': [...], 'ap': [...]}
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, data in results.items():
        ax.plot(data["epochs"], data["ap"], marker="o", label=name)

    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Average Precision (AP)")
    ax.set_title("Training Efficiency Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_stratified_results(
    results: Dict[str, Dict[str, float]],
    metric: str = "ap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot stratified evaluation results as grouped bar chart.

    Args:
        results: Dict of model_name -> {group_name -> {metric: value}}
        metric: Metric to plot
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    models = list(results.keys())
    groups = list(results[models[0]].keys())

    x = np.arange(len(groups))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=figsize)

    for i, model in enumerate(models):
        values = [results[model][g][metric] for g in groups]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model)

    ax.set_xlabel("Group")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Stratified {metric.upper()} by Group")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_relevance(
    nmi_values: np.ndarray,
    channel_names: List[str],
    pred_nmi: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot feature relevance/reliance analysis.

    Args:
        nmi_values: NMI with ground truth (relevance)
        channel_names: Names of channels
        pred_nmi: Optional NMI with predictions (reliance)
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(channel_names))

    if pred_nmi is not None:
        width = 0.35
        ax.bar(x - width / 2, nmi_values, width, label="Relevance (GT)")
        ax.bar(x + width / 2, pred_nmi, width, label="Reliance (Pred)")
        ax.legend()
    else:
        ax.bar(x, nmi_values)

    ax.set_xlabel("Channel")
    ax.set_ylabel("Normalized Mutual Information")
    ax.set_title("Feature Relevance Analysis")
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_ig_attribution(
    attributions: np.ndarray,
    channel_names: List[str],
    original_image: Optional[np.ndarray] = None,
    gt: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    n_top_channels: int = 6,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    """Plot Integrated Gradients attribution maps.

    Args:
        attributions: Attribution tensor (T, C, H, W)
        channel_names: Names of channels
        original_image: Optional original input for overlay
        gt: Optional ground truth mask
        save_path: Path to save figure
        n_top_channels: Number of top channels to display
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Use last timestep
    if attributions.ndim == 4:
        attributions = attributions[-1]

    # Get channel importance (mean absolute attribution)
    channel_importance = np.abs(attributions).mean(axis=(1, 2))
    top_channels = np.argsort(channel_importance)[-n_top_channels:][::-1]

    n_cols = n_top_channels + (1 if gt is not None else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    # Plot GT if provided
    idx = 0
    if gt is not None:
        axes[0].imshow(gt, cmap="Reds")
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        idx = 1

    # Plot top channels
    for i, ch in enumerate(top_channels):
        ax = axes[idx + i]
        attr = attributions[ch]

        # Normalize for visualization
        vmax = np.abs(attr).max()
        ax.imshow(attr, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{channel_names[ch]}\n(Imp: {channel_importance[ch]:.3f})")
        ax.axis("off")

    plt.suptitle("Integrated Gradients Attribution", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
