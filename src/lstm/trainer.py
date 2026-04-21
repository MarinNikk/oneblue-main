"""Point-by-point LSTM training."""

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

from .data_utils import (
    apply_normalization,
    create_ocean_mask_train_only,
    load_currents_data,
    normalize_currents_train_only,
    temporal_train_test_split,
)
from .point_by_point import PointByPointLSTM


def train_point_by_point_lstm(
    uo_path: str,
    vo_path: str,
    sequence_length: int = 7,
    prediction_horizon: int = 3,
    hidden_size: int = 64,
    num_layers: int = 2,
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 0.001,
    min_valid_fraction: float = 0.8,
    output_dir: str = "lstm_results",
) -> dict[str, Any]:
    """Train point-by-point LSTM model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console = Console()

    # Start timing
    training_start_time = time.time()

    # Data preparation
    uo_data, vo_data, _ = load_currents_data(uo_path, vo_path)
    uo_train, vo_train, uo_test, vo_test = temporal_train_test_split(
        uo_data, vo_data, train_fraction=0.8
    )

    norm_stats = normalize_currents_train_only(uo_train, vo_train)
    uo_train_norm, vo_train_norm = apply_normalization(uo_train, vo_train, norm_stats)
    uo_test_norm, vo_test_norm = apply_normalization(uo_test, vo_test, norm_stats)

    ocean_mask = create_ocean_mask_train_only(uo_train_norm, vo_train_norm, min_valid_fraction)

    # Save preprocessing artifacts
    with open(output_dir / "normalization_stats.pkl", "wb") as f:
        pickle.dump(norm_stats, f)
    with open(output_dir / "ocean_mask.pkl", "wb") as f:
        pickle.dump(ocean_mask, f)

    # Prepare sequences
    (train_sequences, train_targets), (test_sequences, test_targets), ocean_indices = (
        PointByPointLSTM.prepare_sequences_temporal_split(
            np.concatenate([uo_train_norm, uo_test_norm], axis=0),
            np.concatenate([vo_train_norm, vo_test_norm], axis=0),
            ocean_mask,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            train_fraction=len(uo_train_norm) / (len(uo_train_norm) + len(uo_test_norm)),
        )
    )

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_sequences), torch.FloatTensor(train_targets)
    )
    val_dataset = TensorDataset(torch.FloatTensor(test_sequences), torch.FloatTensor(test_targets))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = PointByPointLSTM(
        input_channels=2,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0,
        prediction_horizon=prediction_horizon,
    )
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    train_losses = []
    val_losses = []
    r2_history = []
    best_val_loss = float("inf")
    best_r2 = -np.inf
    best_r2_uo = 0.0
    best_r2_vo = 0.0
    early_stop_count = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        with Progress() as progress:
            task = progress.add_task(
                f"[green]Epoch {epoch + 1}/{epochs} - Training",
                total=len(train_loader),
            )
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                progress.advance(task)
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets_list = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_predictions.append(outputs.cpu().numpy())
                val_targets_list.append(targets.cpu().numpy())

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Calculate R² — shapes are always [n, horizon, 2, pts]
        preds = np.concatenate(val_predictions, axis=0)[:, -1]  # [n, 2, pts]
        tgts = np.concatenate(val_targets_list, axis=0)[:, -1]
        p_flat, t_flat = preds.flatten(), tgts.flatten()
        mask = np.isfinite(p_flat) & np.isfinite(t_flat)
        if mask.sum() > 0:
            r2_overall = r2_score(t_flat[mask], p_flat[mask])
            r2_uo = r2_score(tgts[:, 0].flatten(), preds[:, 0].flatten())
            r2_vo = r2_score(tgts[:, 1].flatten(), preds[:, 1].flatten())
        else:
            r2_overall = r2_uo = r2_vo = 0.0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        r2_history.append({"overall": r2_overall, "uo": r2_uo, "vo": r2_vo})

        # Print epoch results
        table = Table(
            show_header=True,
            header_style="bold magenta",
            title=f"Epoch {epoch + 1} Results",
        )
        table.add_column("Metric", style="cyan", width=12)
        table.add_column("Value", justify="right", width=15)

        table.add_row("Train Loss", f"{train_loss:.6f}")
        table.add_row("Val Loss", f"{val_loss:.6f}")
        table.add_row("R²", f"{r2_overall:.4f}")
        table.add_row("R² UO", f"{r2_uo:.4f}")
        table.add_row("R² VO", f"{r2_vo:.4f}")
        table.add_row("LR", f"{optimizer.param_groups[0]['lr']:.2e}")

        console.print(table)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_r2 = r2_overall
            best_r2_uo = r2_uo
            best_r2_vo = r2_vo
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "r2_scores": {"overall": r2_overall, "uo": r2_uo, "vo": r2_vo},
                    "model_config": {
                        "model_type": "point_by_point",
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "sequence_length": sequence_length,
                        "prediction_horizon": prediction_horizon,
                        "input_channels": 2,
                    },
                    "normalization_stats": norm_stats,
                    "ocean_mask": ocean_mask,
                    "ocean_indices": ocean_indices,
                },
                output_dir / "best_point_by_point_model.pth",
            )
            console.print(
                f"[green]New best model: R² = {r2_overall:.4f} (UO: {r2_uo:.4f}, VO: {r2_vo:.4f})[/green]"
            )
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= 10:
            console.print(f"[yellow]Early stopping at epoch {epoch + 1}[/yellow]")
            break

        console.print()

    # Calculate total training time
    total_training_time = time.time() - training_start_time

    # Save results
    results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "r2_history": r2_history,
        "best_val_loss": best_val_loss,
        "final_r2": r2_history[-1]["overall"] if r2_history else 0.0,
        "model_config": {
            "model_type": "point_by_point",
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "sequence_length": sequence_length,
            "prediction_horizon": prediction_horizon,
        },
        "normalization_stats": norm_stats,
        "ocean_mask": ocean_mask,
        "training_time_seconds": total_training_time,
    }

    with open(output_dir / "point_by_point_training_results.pkl", "wb") as f:
        pickle.dump(results, f)

    return {
        "best_val_loss": best_val_loss,
        "best_r2": best_r2,
        "best_r2_uo": best_r2_uo,
        "best_r2_vo": best_r2_vo,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "total_params": sum(p.numel() for p in model.parameters()),
        "output_dir": str(output_dir),
        "training_time_seconds": total_training_time,
    }
