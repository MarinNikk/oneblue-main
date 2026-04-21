"""Training CLI commands."""

import random

import numpy as np
import torch
import typer
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

from copernicus_data import CopernicusDataHandler
from src.configs.config import Config
from src.model.adriatic_dataset import (
    AdriaticDataset,
    normalize_data,
)
from src.model.conv_lstm import AdriaticPredictor
from src.model.trainer import train_model

from .utils import set_seed, slice_data


console = Console()
app = typer.Typer(rich_markup_mode="rich")


@app.command()
def train(
    config_file: str = typer.Option("config.yaml", help="Configuration file path"),
    secrets_file: str = typer.Option("secrets.yaml", help="Secrets file path"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
):
    """Train the Adriatic current prediction model."""
    set_seed(seed)

    config = Config(config_path=config_file, secrets_path=secrets_file)

    handler = CopernicusDataHandler(config)
    data = handler.load_data()
    if data is None:
        console.print("[red]Failed to load CMEMS data! Check your configuration.[/red]")
        raise typer.Exit(1)

    # Split RAW data chronologically (70/15/15 split)
    total_days = len(data["dates"])
    train_end = int(0.70 * total_days)
    val_end = int(0.85 * total_days)

    train_data_raw = slice_data(data, 0, train_end)
    val_data_raw = slice_data(data, train_end, val_end)
    test_data_raw = slice_data(data, val_end, total_days)

    # Normalize: compute stats from training data, apply to val/test
    train_data_norm, train_stats = normalize_data(
        train_data_raw,
        stats_file="normalization_stats.pkl",
        train_stats=None,
    )
    val_data_norm, _ = normalize_data(
        val_data_raw,
        train_stats=train_stats,
    )
    test_data_norm, _ = normalize_data(
        test_data_raw,
        train_stats=train_stats,
    )

    # Create datasets
    train_dataset = AdriaticDataset(
        train_data_norm,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
    )
    val_dataset = AdriaticDataset(
        val_data_norm,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
    )
    test_dataset = AdriaticDataset(
        test_data_norm,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
    )

    # Create data loaders
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn,
    )

    # Create model
    model = AdriaticPredictor(
        input_dim=6,  # u_curr, v_curr, u_wind, v_wind, ssh, temp
        hidden_dims=config.hidden_dims,
        seasonal_dim=2,  # sin/cos encoding
        output_dim=2,  # u_curr, v_curr predictions
        prediction_horizon=config.prediction_horizon,
    )

    # Train model
    full_grid_size = val_dataset.full_grid_size
    uniform_step = val_dataset.uniform_step

    best_r2, best_mae, best_rmse, best_r2_full = train_model(
        model,
        train_loader,
        val_loader,
        config,
        train_stats,
        full_grid_size=full_grid_size,
        uniform_step=uniform_step,
    )

    # Final results summary
    result_table = Table(
        title="Final Training Results", show_header=True, header_style="bold blue"
    )
    result_table.add_column("Metric", style="cyan", width=20)
    result_table.add_column("Best Value", justify="right", width=15)
    result_table.add_column("Unit", style="dim", width=8)

    result_table.add_row("R²", f"{best_r2:.4f}", "")
    if full_grid_size:
        result_table.add_row("R² (full grid)", f"{best_r2_full:.4f}", "")
    result_table.add_row("MAE Magnitude", f"{best_mae:.4f}", "m/s")
    result_table.add_row("RMSE Magnitude", f"{best_rmse:.4f}", "m/s")

    console.print(result_table)
