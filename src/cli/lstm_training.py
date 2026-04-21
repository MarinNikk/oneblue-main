"""CLI for training LSTM models on ocean current data."""

import typer
from rich.console import Console
from rich.table import Table

from src.lstm.trainer import train_point_by_point_lstm


console = Console()
app = typer.Typer()


@app.command()
def point_by_point(
    uo_path: str = typer.Option(
        "data/raw/adriatic_currents_uo_detided_2023_2024.nc", help="Path to UO data file"
    ),
    vo_path: str = typer.Option(
        "data/raw/adriatic_currents_vo_detided_2023_2024.nc", help="Path to VO data file"
    ),
    sequence_length: int = typer.Option(7, help="Input sequence length"),
    prediction_horizon: int = typer.Option(1, help="Number of days ahead to predict"),
    hidden_size: int = typer.Option(64, help="LSTM hidden size"),
    num_layers: int = typer.Option(2, help="Number of LSTM layers"),
    batch_size: int = typer.Option(16, help="Batch size"),
    epochs: int = typer.Option(50, help="Number of training epochs"),
    lr: float = typer.Option(0.001, help="Learning rate"),
    min_valid_fraction: float = typer.Option(
        0.8, help="Minimum valid data fraction for ocean mask"
    ),
    output_dir: str = typer.Option("lstm_results", help="Output directory for results"),
):
    """Train Point-by-Point LSTM model for ocean current prediction."""
    results = train_point_by_point_lstm(
        uo_path=uo_path,
        vo_path=vo_path,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        min_valid_fraction=min_valid_fraction,
        output_dir=output_dir,
    )

    # Final results summary
    result_table = Table(
        title="Final Training Results", show_header=True, header_style="bold blue"
    )
    result_table.add_column("Metric", style="cyan", width=20)
    result_table.add_column("Best Value", justify="right", width=15)

    result_table.add_row("R²", f"{results['best_r2']:.4f}")
    result_table.add_row("R² UO", f"{results['best_r2_uo']:.4f}")
    result_table.add_row("R² VO", f"{results['best_r2_vo']:.4f}")
    result_table.add_row("Val Loss", f"{results['best_val_loss']:.6f}")

    console.print(result_table)
