#!/usr/bin/env python3
"""
Validate Point-by-Point LSTM model with comprehensive visualizations.
This script loads the trained model and creates validation plots to verify performance.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.progress import track
from sklearn.metrics import r2_score

from src.lstm.data_utils import (
    apply_normalization,
    apply_ocean_mask,
    load_currents_data,
    reconstruct_full_grid,
    temporal_train_test_split,
)
from src.lstm.point_by_point import PointByPointLSTM


console = Console()


def load_trained_model(model_path="lstm_results/best_point_by_point_model.pth"):
    """Load the trained model and all artifacts."""
    console.print(f"[cyan]Loading model from {model_path}...[/cyan]")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]

    # Recreate model
    model = PointByPointLSTM(
        input_channels=model_config["input_channels"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=0.0,  # No dropout during inference
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def prepare_validation_data(checkpoint):
    """Prepare validation data using same preprocessing as training."""
    console.print("[cyan]Preparing validation data...[/cyan]")

    # Load raw data (use default paths - adjust if needed)
    uo_path = "data/raw/adriatic_currents_uo_detided_2023_2024.nc"
    vo_path = "data/raw/adriatic_currents_vo_detided_2023_2024.nc"

    uo_data, vo_data, time_coords = load_currents_data(uo_path, vo_path)

    # Use same split as training
    uo_train, vo_train, uo_test, vo_test = temporal_train_test_split(
        uo_data, vo_data, train_fraction=0.8
    )

    # Apply same normalization
    norm_stats = checkpoint["normalization_stats"]
    uo_test_norm, vo_test_norm = apply_normalization(uo_test, vo_test, norm_stats)

    # Apply same ocean mask
    ocean_mask = checkpoint["ocean_mask"]

    # Prepare sequences for validation split only
    data = np.stack([uo_test_norm, vo_test_norm], axis=1)  # [time, 2, lat, lon]
    data_masked, ocean_indices = apply_ocean_mask(data, ocean_mask)

    sequence_length = checkpoint["model_config"]["sequence_length"]

    # Create sequences from test data
    sequences, targets = [], []
    for i in range(len(data_masked) - sequence_length):
        x = data_masked[i : i + sequence_length]  # [seq_len, 2, n_ocean_points]
        y = data_masked[i + sequence_length]  # [2, n_ocean_points]

        if not (np.isnan(x).any() or np.isnan(y).any()):
            sequences.append(x)
            targets.append(y)

    sequences = np.array(sequences)
    targets = np.array(targets)

    console.print(f"[green]Validation sequences: {sequences.shape}[/green]")

    return sequences, targets, ocean_mask, ocean_indices, norm_stats, time_coords


def generate_predictions(model, sequences):
    """Generate predictions for validation sequences."""
    console.print("[cyan]Generating predictions...[/cyan]")

    predictions = []
    batch_size = 32  # Process in batches to avoid memory issues

    with torch.no_grad():
        for i in track(range(0, len(sequences), batch_size), description="Predicting"):
            batch = sequences[i : i + batch_size]
            batch_tensor = torch.FloatTensor(batch)
            pred = model(batch_tensor).numpy()
            predictions.append(pred)

    return np.concatenate(predictions, axis=0)


def plot_predictions_vs_actual(predictions, targets, save_path="plots/lstm_validation"):
    """Create scatter plots of predictions vs actual values."""
    console.print("[cyan]Creating prediction vs actual plots...[/cyan]")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

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

    plt.suptitle("LSTM Model: Predictions vs Actual Values", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path / "predictions_vs_actual.png", dpi=300, bbox_inches="tight")
    plt.show()

    return r2_uo, r2_vo, r2_overall


def plot_spatial_predictions(
    predictions,
    targets,
    ocean_mask,
    ocean_indices,
    time_idx=0,
    save_path="plots/lstm_validation",
):
    """Create spatial maps of predictions vs actual values."""
    console.print("[cyan]Creating spatial prediction maps...[/cyan]")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Reconstruct full spatial grids
    pred_full = reconstruct_full_grid(predictions[[time_idx]], ocean_mask, ocean_indices)
    true_full = reconstruct_full_grid(targets[[time_idx]], ocean_mask, ocean_indices)

    pred_full = pred_full[0]  # Remove batch dimension
    true_full = true_full[0]  # Remove batch dimension

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Define symmetric colorbar limits for better visualization
    uo_abs_max = max(
        abs(np.nanmin(true_full[0])),
        abs(np.nanmax(true_full[0])),
        abs(np.nanmin(pred_full[0])),
        abs(np.nanmax(pred_full[0])),
    )
    vo_abs_max = max(
        abs(np.nanmin(true_full[1])),
        abs(np.nanmax(true_full[1])),
        abs(np.nanmin(pred_full[1])),
        abs(np.nanmax(pred_full[1])),
    )

    uo_min, uo_max = -uo_abs_max, uo_abs_max
    vo_min, vo_max = -vo_abs_max, vo_abs_max

    # UO component (flip vertically for proper geographic orientation)
    im1 = axes[0, 0].imshow(true_full[0], vmin=uo_min, vmax=uo_max, cmap="RdBu_r", origin="lower")
    axes[0, 0].set_title("True UO (Eastward)")
    axes[0, 0].set_xlabel("Longitude")
    axes[0, 0].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[0, 0], label="m/s")

    im2 = axes[0, 1].imshow(pred_full[0], vmin=uo_min, vmax=uo_max, cmap="RdBu_r", origin="lower")
    axes[0, 1].set_title("Predicted UO (Eastward)")
    axes[0, 1].set_xlabel("Longitude")
    axes[0, 1].set_ylabel("Latitude")
    plt.colorbar(im2, ax=axes[0, 1], label="m/s")

    # UO error with symmetric scaling
    uo_error = true_full[0] - pred_full[0]
    uo_error_abs_max = np.nanmax(np.abs(uo_error))
    im3 = axes[0, 2].imshow(
        uo_error, vmin=-uo_error_abs_max, vmax=uo_error_abs_max, cmap="RdBu_r", origin="lower"
    )
    axes[0, 2].set_title("UO Error (True - Pred)")
    axes[0, 2].set_xlabel("Longitude")
    axes[0, 2].set_ylabel("Latitude")
    plt.colorbar(im3, ax=axes[0, 2], label="m/s")

    # VO component (flip vertically for proper geographic orientation)
    im4 = axes[1, 0].imshow(true_full[1], vmin=vo_min, vmax=vo_max, cmap="RdBu_r", origin="lower")
    axes[1, 0].set_title("True VO (Northward)")
    axes[1, 0].set_xlabel("Longitude")
    axes[1, 0].set_ylabel("Latitude")
    plt.colorbar(im4, ax=axes[1, 0], label="m/s")

    im5 = axes[1, 1].imshow(pred_full[1], vmin=vo_min, vmax=vo_max, cmap="RdBu_r", origin="lower")
    axes[1, 1].set_title("Predicted VO (Northward)")
    axes[1, 1].set_xlabel("Longitude")
    axes[1, 1].set_ylabel("Latitude")
    plt.colorbar(im5, ax=axes[1, 1], label="m/s")

    # VO error with symmetric scaling
    vo_error = true_full[1] - pred_full[1]
    vo_error_abs_max = np.nanmax(np.abs(vo_error))
    im6 = axes[1, 2].imshow(
        vo_error, vmin=-vo_error_abs_max, vmax=vo_error_abs_max, cmap="RdBu_r", origin="lower"
    )
    axes[1, 2].set_title("VO Error (True - Pred)")
    axes[1, 2].set_xlabel("Longitude")
    axes[1, 2].set_ylabel("Latitude")
    plt.colorbar(im6, ax=axes[1, 2], label="m/s")

    plt.suptitle(
        f"LSTM Model: Spatial Predictions vs Ground Truth (Time Step {time_idx})",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path / f"spatial_predictions_t{time_idx}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_time_series(predictions, targets, point_idx=0, save_path="plots/lstm_validation"):
    """Plot time series for a specific ocean point."""
    console.print("[cyan]Creating time series plots...[/cyan]")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    time_steps = np.arange(len(predictions))

    # UO time series
    axes[0].plot(time_steps, targets[:, 0, point_idx], "b-", label="True UO", linewidth=2)
    axes[0].plot(time_steps, predictions[:, 0, point_idx], "r--", label="Predicted UO", linewidth=2)
    axes[0].set_ylabel("UO Current (m/s)")
    axes[0].set_title(f"Time Series for Ocean Point {point_idx}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # VO time series
    axes[1].plot(time_steps, targets[:, 1, point_idx], "b-", label="True VO", linewidth=2)
    axes[1].plot(time_steps, predictions[:, 1, point_idx], "r--", label="Predicted VO", linewidth=2)
    axes[1].set_ylabel("VO Current (m/s)")
    axes[1].set_xlabel("Time Step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"LSTM Model: Time Series Comparison for Ocean Point {point_idx}", fontsize=16, y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path / f"time_series_point{point_idx}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_error_statistics(predictions, targets, save_path="plots/lstm_validation"):
    """Create error distribution and statistics plots."""
    console.print("[cyan]Creating error analysis plots...[/cyan]")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Calculate errors
    errors_uo = predictions[:, 0, :] - targets[:, 0, :]
    errors_vo = predictions[:, 1, :] - targets[:, 1, :]

    # Flatten and remove NaNs
    errors_uo_flat = errors_uo.flatten()
    errors_vo_flat = errors_vo.flatten()
    errors_uo_valid = errors_uo_flat[np.isfinite(errors_uo_flat)]
    errors_vo_valid = errors_vo_flat[np.isfinite(errors_vo_flat)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # UO error distribution
    axes[0].hist(errors_uo_valid, bins=50, alpha=0.7, density=True)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_title(
        f"UO Error Distribution\nMean: {np.mean(errors_uo_valid):.4f}, Std: {np.std(errors_uo_valid):.4f}"
    )
    axes[0].set_xlabel("Error (m/s)")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.3)

    # VO error distribution
    axes[1].hist(errors_vo_valid, bins=50, alpha=0.7, density=True)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_title(
        f"VO Error Distribution\nMean: {np.mean(errors_vo_valid):.4f}, Std: {np.std(errors_vo_valid):.4f}"
    )
    axes[1].set_xlabel("Error (m/s)")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("LSTM Model: Error Distribution Analysis", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path / "error_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main validation script."""
    console.print("[bold cyan]Point-by-Point LSTM Model Validation[/bold cyan]")

    # Load model
    model, checkpoint = load_trained_model()

    # Prepare validation data
    sequences, targets, ocean_mask, ocean_indices, norm_stats, time_coords = (
        prepare_validation_data(checkpoint)
    )

    # Generate predictions
    predictions = generate_predictions(model, sequences)

    console.print(f"[green]Predictions shape: {predictions.shape}[/green]")
    console.print(f"[green]Targets shape: {targets.shape}[/green]")

    # Create validation plots
    r2_uo, r2_vo, r2_overall = plot_predictions_vs_actual(predictions, targets)

    console.print("[bold green]Validation R² Scores:[/bold green]")
    console.print(f"[green]  UO: {r2_uo:.4f}[/green]")
    console.print(f"[green]  VO: {r2_vo:.4f}[/green]")
    console.print(f"[green]  Overall: {r2_overall:.4f}[/green]")

    plot_spatial_predictions(predictions, targets, ocean_mask, ocean_indices)
    plot_time_series(predictions, targets)
    plot_error_statistics(predictions, targets)

    console.print("[bold green]Validation complete! Check plots/ for plots.[/bold green]")


if __name__ == "__main__":
    main()
