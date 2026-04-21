#!/usr/bin/env python3
"""
N-Day Mean Baseline Model Validation and Comparison with LSTM.
Tests baseline models from 1-7 days and compares R² performance.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.progress import track
from sklearn.metrics import mean_squared_error, r2_score

from src.lstm.data_utils import (
    apply_normalization,
    apply_ocean_mask,
    load_currents_data,
    temporal_train_test_split,
)


console = Console()


class NDayMeanBaseline:
    """Simple N-day mean baseline model."""

    def __init__(self, n_days=3):
        self.n_days = n_days
        self.name = f"{n_days}-Day Mean"

    def predict(self, sequences):
        """Predict using mean of last N days."""
        # sequences shape: [batch, seq_len, channels, points]
        # Take mean of last n_days
        predictions = np.mean(sequences[:, -self.n_days :, :, :], axis=1)
        return predictions


def load_validation_data():
    """Load and prepare validation data (same as LSTM validation)."""
    console.print("[cyan]Loading validation data for baseline comparison...[/cyan]")

    # Load LSTM checkpoint for data preprocessing parameters
    checkpoint = torch.load(
        "lstm_results/best_point_by_point_model.pth", map_location="cpu", weights_only=False
    )

    # Load raw data
    uo_path = "data/raw/adriatic_currents_uo_detided_2023_2024.nc"
    vo_path = "data/raw/adriatic_currents_vo_detided_2023_2024.nc"
    uo_data, vo_data, time_coords = load_currents_data(uo_path, vo_path)

    # Use same split as LSTM training
    uo_train, vo_train, uo_test, vo_test = temporal_train_test_split(
        uo_data, vo_data, train_fraction=0.8
    )

    # Apply same normalization
    norm_stats = checkpoint["normalization_stats"]
    uo_test_norm, vo_test_norm = apply_normalization(uo_test, vo_test, norm_stats)

    # Apply same ocean mask
    ocean_mask = checkpoint["ocean_mask"]
    data = np.stack([uo_test_norm, vo_test_norm], axis=1)
    data_masked, ocean_indices = apply_ocean_mask(data, ocean_mask)

    sequence_length = checkpoint["model_config"]["sequence_length"]

    # Create sequences
    sequences, targets = [], []
    for i in range(len(data_masked) - sequence_length):
        x = data_masked[i : i + sequence_length]
        y = data_masked[i + sequence_length]

        if not (np.isnan(x).any() or np.isnan(y).any()):
            sequences.append(x)
            targets.append(y)

    sequences = np.array(sequences)
    targets = np.array(targets)

    return sequences, targets, ocean_mask, ocean_indices, norm_stats, checkpoint


def calculate_baseline_performance(sequences, targets, n_days_list=None):
    """Calculate R² for different N-day baseline models."""
    if n_days_list is None:
        n_days_list = [1, 2, 3, 4, 5, 6, 7]

    console.print("[cyan]Testing N-day baseline models...[/cyan]")

    results = {}

    for n_days in track(n_days_list, description="Testing baselines"):
        baseline = NDayMeanBaseline(n_days=n_days)
        predictions = baseline.predict(sequences)

        # Global R² scores
        pred_flat = predictions.flatten()
        true_flat = targets.flatten()
        valid_mask = np.isfinite(pred_flat) & np.isfinite(true_flat)

        r2_global = r2_score(true_flat[valid_mask], pred_flat[valid_mask])
        mse_global = mean_squared_error(true_flat[valid_mask], pred_flat[valid_mask])

        # Component-wise R² scores
        pred_uo_flat = predictions[:, 0, :].flatten()
        pred_vo_flat = predictions[:, 1, :].flatten()
        true_uo_flat = targets[:, 0, :].flatten()
        true_vo_flat = targets[:, 1, :].flatten()

        valid_uo = np.isfinite(pred_uo_flat) & np.isfinite(true_uo_flat)
        valid_vo = np.isfinite(pred_vo_flat) & np.isfinite(true_vo_flat)

        r2_uo = r2_score(true_uo_flat[valid_uo], pred_uo_flat[valid_uo])
        r2_vo = r2_score(true_vo_flat[valid_vo], pred_vo_flat[valid_vo])

        # Per-point statistics
        n_points = predictions.shape[2]
        r2_per_point = []

        for point_idx in range(n_points):
            # Combined R² for this point
            pred_point = np.concatenate(
                [predictions[:, 0, point_idx], predictions[:, 1, point_idx]]
            )
            true_point = np.concatenate([targets[:, 0, point_idx], targets[:, 1, point_idx]])
            valid_point = np.isfinite(pred_point) & np.isfinite(true_point)

            if np.sum(valid_point) > 1:
                r2_point = r2_score(true_point[valid_point], pred_point[valid_point])
            else:
                r2_point = np.nan
            r2_per_point.append(r2_point)

        r2_per_point = np.array(r2_per_point)

        results[n_days] = {
            "predictions": predictions,
            "r2_global": r2_global,
            "r2_uo": r2_uo,
            "r2_vo": r2_vo,
            "mse_global": mse_global,
            "r2_per_point": r2_per_point,
            "r2_per_point_min": np.nanmin(r2_per_point),
            "r2_per_point_mean": np.nanmean(r2_per_point),
            "r2_per_point_max": np.nanmax(r2_per_point),
            "poor_points_count": np.sum(r2_per_point < 0.7),
            "model_name": f"{n_days}-Day Mean",
        }

        console.print(
            f"[green]{n_days}-Day Mean: R² = {r2_global:.4f}, "
            f"Per-point R² min/mean/max = {np.nanmin(r2_per_point):.3f}/"
            f"{np.nanmean(r2_per_point):.3f}/{np.nanmax(r2_per_point):.3f}[/green]"
        )

    return results


def plot_baseline_comparison(baseline_results, save_path="lstm_results/baseline_comparison"):
    """Create baseline comparison table."""
    console.print("[cyan]Creating baseline comparison table...[/cyan]")

    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    n_days_list = list(baseline_results.keys())

    # Create simple figure with just the table
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.axis("tight")
    ax.axis("off")

    # Create summary table - just Global R²
    table_data = []
    for n_days in n_days_list:
        table_data.append([f"{n_days}-Day Mean", f"{baseline_results[n_days]['r2_global']:.3f}"])

    table = ax.table(
        cellText=table_data, colLabels=["Model", "Global R²"], cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    ax.set_title("Baseline Model Performance Comparison", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(save_path / "baseline_comparison_table.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Find best performing baseline
    global_r2_values = [baseline_results[n]["r2_global"] for n in n_days_list]
    best_idx = np.argmax(global_r2_values)
    best_n_days = n_days_list[best_idx]

    return best_n_days


def create_best_baseline_validation_plots(
    baseline_results,
    best_n_days,
    targets,
    ocean_mask,
    ocean_indices,
    save_path="plots/baseline_validation",
):
    """Create detailed validation plots for the best baseline model."""
    console.print(
        f"[cyan]Creating detailed plots for best baseline ({best_n_days}-Day Mean)...[/cyan]"
    )

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    predictions = baseline_results[best_n_days]["predictions"]

    # Use same plotting functions as LSTM validation
    plot_predictions_vs_actual_baseline(predictions, targets, f"{best_n_days}-Day Mean", save_path)
    plot_spatial_predictions_baseline(
        predictions, targets, ocean_mask, ocean_indices, f"{best_n_days}-Day Mean", save_path
    )
    plot_time_series_baseline(predictions, targets, f"{best_n_days}-Day Mean", save_path)
    plot_error_statistics_baseline(predictions, targets, f"{best_n_days}-Day Mean", save_path)


def plot_predictions_vs_actual_baseline(predictions, targets, model_name, save_path):
    """Create scatter plots for baseline model."""
    console.print(f"[cyan]Creating prediction vs actual plots for {model_name}...[/cyan]")

    save_path = Path(save_path)

    # Flatten data for scatter plot
    pred_uo = predictions[:, 0, :].flatten()
    pred_vo = predictions[:, 1, :].flatten()
    true_uo = targets[:, 0, :].flatten()
    true_vo = targets[:, 1, :].flatten()

    # Remove NaN values
    valid_uo = np.isfinite(pred_uo) & np.isfinite(true_uo)
    valid_vo = np.isfinite(pred_vo) & np.isfinite(true_vo)

    # Calculate R² scores
    r2_uo = r2_score(true_uo[valid_uo], pred_uo[valid_uo])
    r2_vo = r2_score(true_vo[valid_vo], pred_vo[valid_vo])
    r2_overall = r2_score(
        np.concatenate([true_uo[valid_uo], true_vo[valid_vo]]),
        np.concatenate([pred_uo[valid_uo], pred_vo[valid_vo]]),
    )

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # UO component
    axes[0].scatter(true_uo[valid_uo], pred_uo[valid_uo], alpha=0.5, s=1)
    axes[0].plot(
        [true_uo[valid_uo].min(), true_uo[valid_uo].max()],
        [true_uo[valid_uo].min(), true_uo[valid_uo].max()],
        "r--",
        lw=2,
    )
    axes[0].set_xlabel("True UO (m/s)")
    axes[0].set_ylabel("Predicted UO (m/s)")
    axes[0].set_title(f"UO Component\nR² = {r2_uo:.4f}")
    axes[0].grid(True, alpha=0.3)

    # VO component
    axes[1].scatter(true_vo[valid_vo], pred_vo[valid_vo], alpha=0.5, s=1)
    axes[1].plot(
        [true_vo[valid_vo].min(), true_vo[valid_vo].max()],
        [true_vo[valid_vo].min(), true_vo[valid_vo].max()],
        "r--",
        lw=2,
    )
    axes[1].set_xlabel("True VO (m/s)")
    axes[1].set_ylabel("Predicted VO (m/s)")
    axes[1].set_title(f"VO Component\nR² = {r2_vo:.4f}")
    axes[1].grid(True, alpha=0.3)

    # Combined
    all_true = np.concatenate([true_uo[valid_uo], true_vo[valid_vo]])
    all_pred = np.concatenate([pred_uo[valid_uo], pred_vo[valid_vo]])
    axes[2].scatter(all_true, all_pred, alpha=0.5, s=1)
    axes[2].plot([all_true.min(), all_true.max()], [all_true.min(), all_true.max()], "r--", lw=2)
    axes[2].set_xlabel("True Current (m/s)")
    axes[2].set_ylabel("Predicted Current (m/s)")
    axes[2].set_title(f"Combined (UO + VO)\nR² = {r2_overall:.4f}")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} Baseline: Predictions vs Actual Values", fontsize=16, y=1.02)
    plt.tight_layout()

    filename = f"baseline_{model_name.replace('-', '').replace(' ', '_').lower()}_predictions_vs_actual.png"
    plt.savefig(save_path / filename, dpi=300, bbox_inches="tight")
    plt.show()

    return r2_uo, r2_vo, r2_overall


def plot_spatial_predictions_baseline(
    predictions, targets, ocean_mask, ocean_indices, model_name, save_path
):
    """Create spatial plots for baseline model."""
    import matplotlib.pyplot as plt

    from src.lstm.validate_model import plot_spatial_predictions

    plot_spatial_predictions(predictions, targets, ocean_mask, ocean_indices, 0, save_path)

    # Add title by modifying the last figure
    plt.suptitle(f"{model_name} Baseline: Spatial Predictions vs Ground Truth", fontsize=16, y=1.02)
    plt.tight_layout()

    # Rename the saved file
    old_path = save_path / "spatial_predictions_t0.png"
    new_path = (
        save_path
        / f"baseline_{model_name.replace('-', '').replace(' ', '_').lower()}_spatial_predictions.png"
    )
    if old_path.exists():
        old_path.rename(new_path)


def plot_time_series_baseline(predictions, targets, model_name, save_path):
    """Create time series plots for baseline model."""
    import matplotlib.pyplot as plt

    from src.lstm.validate_model import plot_time_series

    plot_time_series(predictions, targets, 0, save_path)

    # Add title by modifying the last figure
    plt.suptitle(
        f"{model_name} Baseline: Time Series Comparison for Ocean Point 0", fontsize=16, y=1.02
    )
    plt.tight_layout()

    # Rename the saved file
    old_path = save_path / "time_series_point0.png"
    new_path = (
        save_path
        / f"baseline_{model_name.replace('-', '').replace(' ', '_').lower()}_time_series.png"
    )
    if old_path.exists():
        old_path.rename(new_path)


def plot_error_statistics_baseline(predictions, targets, model_name, save_path):
    """Create error statistics plots for baseline model."""
    import matplotlib.pyplot as plt

    from src.lstm.validate_model import plot_error_statistics

    plot_error_statistics(predictions, targets, save_path)

    # Add title by modifying the last figure
    plt.suptitle(f"{model_name} Baseline: Error Distribution Analysis", fontsize=16, y=1.02)
    plt.tight_layout()

    # Rename the saved file
    old_path = save_path / "error_analysis.png"
    new_path = (
        save_path
        / f"baseline_{model_name.replace('-', '').replace(' ', '_').lower()}_error_analysis.png"
    )
    if old_path.exists():
        old_path.rename(new_path)


def main():
    """Main baseline validation script."""
    console.print("[bold cyan]N-Day Mean Baseline Models Validation[/bold cyan]")

    # Load validation data
    sequences, targets, ocean_mask, ocean_indices, norm_stats, checkpoint = load_validation_data()

    # Test baseline models (1-7 days)
    baseline_results = calculate_baseline_performance(
        sequences, targets, n_days_list=[1, 2, 3, 4, 5, 6, 7]
    )

    # Create comparison table
    best_n_days = plot_baseline_comparison(baseline_results)

    # Create detailed plots for best baseline
    create_best_baseline_validation_plots(
        baseline_results, best_n_days, targets, ocean_mask, ocean_indices
    )

    console.print("[bold green]Baseline validation complete![/bold green]")
    console.print(f"[green]Best baseline: {best_n_days}-Day Mean[/green]")
    console.print("[green]Check lstm_results/ for all plots[/green]")


if __name__ == "__main__":
    main()
