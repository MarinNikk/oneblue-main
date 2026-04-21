"""Baseline models CLI commands."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.baselines import run_mean_baseline_experiment
from src.baselines.xgboost_baseline import run_xgboost_baseline_experiment
from src.rbf_lstm.utils.io_utils import load_ocean_data


console = Console()
app = typer.Typer(rich_markup_mode="rich")


def print_metrics_table(
    metrics: dict[str, float], split_name: str, model_name: str = "7-Day Mean Baseline"
) -> None:
    """
    Pretty-print evaluation metrics using Rich tables.

    Args:
        metrics: Dictionary with evaluation metrics
        split_name: Name of the split (train/val/test)
        model_name: Name of the model for the table title
    """
    table = Table(
        title=f"📊 {model_name} - {split_name.capitalize()} Set",
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Metric", style="cyan", width=20)
    table.add_column("U Component", justify="right", width=15)
    table.add_column("V Component", justify="right", width=15)
    table.add_column("Magnitude", justify="right", width=15)
    table.add_column("Unit", style="dim", width=8)

    # Add rows
    table.add_row(
        "RMSE",
        f"{metrics['rmse_u']:.6f}",
        f"{metrics['rmse_v']:.6f}",
        f"{metrics['rmse_magnitude']:.6f}",
        "m/s",
    )
    table.add_row(
        "MAE",
        f"{metrics['mae_u']:.6f}",
        f"{metrics['mae_v']:.6f}",
        f"{metrics['mae_magnitude']:.6f}",
        "m/s",
    )
    table.add_row("R²", f"{metrics['r2_u']:.6f}", f"{metrics['r2_v']:.6f}", "—", "")

    console.print(table)


@app.command()
def mean_nday(
    window_size: int = typer.Option(7, "--window-size", help="Number of days used for averaging."),
):
    """
    Run 7-day mean baseline experiment.

    This establishes the minimum performance threshold - it predicts
    tomorrow's currents as the simple average of the past 7 days.

    No machine learning, no training - just pure temporal averaging.
    Any ML model should significantly outperform this baseline.
    """
    console.print("[bold blue]🌊 7-Day Mean Baseline Experiment[/bold blue]\n")

    # Load U and V currents using existing utility
    console.print("[cyan]Loading U/V current data...[/cyan]")

    # Paths for detided current data
    data_dir = "data/raw"
    uo_file = "adriatic_currents_uo_detided_2023_2024.nc"
    vo_file = "adriatic_currents_vo_detided_2023_2024.nc"

    try:
        uo_dataset, vo_dataset = load_ocean_data(data_dir, uo_file, vo_file)

        # Extract the data arrays from the datasets
        # Using the correct variable names for detided data
        u_data = uo_dataset.uo_detided.values
        v_data = vo_dataset.vo_detided.values

    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        console.print("[yellow]Please ensure detided data files exist at:[/yellow]")
        console.print(f"  {data_dir}/{uo_file}")
        console.print(f"  {data_dir}/{vo_file}")
        return

    console.print("[green]✓ Loaded data:[/green]")
    console.print(f"  U currents shape: {u_data.shape}")
    console.print(f"  V currents shape: {v_data.shape}")
    console.print(f"  Total timesteps: {len(u_data)}")
    console.print(f"  Spatial resolution: {u_data.shape[1]} × {u_data.shape[2]}")

    # Show split information
    n_total = len(u_data)
    train_end = int(n_total * 0.7)
    val_end = train_end + int(n_total * 0.15)

    console.print("\n[cyan]Dataset split:[/cyan]")
    console.print(f"  Train: timesteps 0-{train_end} ({train_end} samples)")
    console.print(f"  Val:   timesteps {train_end}-{val_end} ({val_end - train_end} samples)")
    console.print(f"  Test:  timesteps {val_end}-{n_total} ({n_total - val_end} samples)\n")

    # Run experiment
    console.print("[yellow]Running 7-day mean baseline experiment...[/yellow]\n")
    results = run_mean_baseline_experiment(
        u_data=u_data,
        v_data=v_data,
        window_size=window_size,
        split_ratios=(0.7, 0.15, 0.15),
    )

    # Print results for each split
    for split_name, metrics in results.items():
        print_metrics_table(metrics, split_name)

    # Summary comparison table
    console.print("\n[bold cyan]═══ Summary Across Splits ═══[/bold cyan]\n")

    summary_table = Table(
        title="📊 7-Day Mean Baseline - Performance Summary",
        show_header=True,
        header_style="bold magenta",
    )

    summary_table.add_column("Split", style="cyan", width=12)
    summary_table.add_column("R² (U)", justify="right", width=12)
    summary_table.add_column("R² (V)", justify="right", width=12)
    summary_table.add_column("MAE (mag)", justify="right", width=12)
    summary_table.add_column("RMSE (mag)", justify="right", width=12)
    summary_table.add_column("Status", width=10)

    for split_name, metrics in results.items():
        r2_u = metrics["r2_u"]
        r2_v = metrics["r2_v"]
        mae_mag = metrics["mae_magnitude"]
        rmse_mag = metrics["rmse_magnitude"]

        status = "✅" if r2_u > 0.3 and r2_v > 0.3 else "⚠️"

        summary_table.add_row(
            split_name.capitalize(),
            f"{r2_u:.4f}",
            f"{r2_v:.4f}",
            f"{mae_mag:.6f}",
            f"{rmse_mag:.6f}",
            status,
        )

    console.print(summary_table)

    # Save results
    import json

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / "baseline_7day_mean.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]✓ Results saved to: {output_file}[/green]")


@app.command()
def xgboost(
    window_size: int = typer.Option(
        7, "--window-size", help="Number of days used as lag features."
    ),
    n_estimators: int = typer.Option(100, "--n-estimators", help="Number of boosting rounds."),
):
    """
    Run XGBoost baseline experiment.

    This uses gradient boosting with 7-day lag features to predict
    tomorrow's currents. Tests whether non-linear temporal patterns
    exist that can be exploited by tree-based models.

    The model treats each spatial location independently and uses
    flattened temporal windows as input features.
    """
    console.print("[bold blue]🚀 XGBoost Baseline Experiment[/bold blue]\n")

    # Load U and V currents using existing utility
    console.print("[cyan]Loading U/V current data...[/cyan]")

    # Paths for detided current data
    data_dir = "data/raw"
    uo_file = "adriatic_currents_uo_detided_2023_2024.nc"
    vo_file = "adriatic_currents_vo_detided_2023_2024.nc"

    try:
        uo_dataset, vo_dataset = load_ocean_data(data_dir, uo_file, vo_file)

        # Extract the data arrays from the datasets
        u_data = uo_dataset.uo_detided.values
        v_data = vo_dataset.vo_detided.values

    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        console.print("[yellow]Please ensure detided data files exist at:[/yellow]")
        console.print(f"  {data_dir}/{uo_file}")
        console.print(f"  {data_dir}/{vo_file}")
        return

    console.print("[green]✓ Loaded data:[/green]")
    console.print(f"  U currents shape: {u_data.shape}")
    console.print(f"  V currents shape: {v_data.shape}")
    console.print(f"  Total timesteps: {len(u_data)}")
    console.print(f"  Spatial resolution: {u_data.shape[1]} × {u_data.shape[2]}")

    # Show split information
    n_total = len(u_data)
    train_end = int(n_total * 0.7)
    val_end = train_end + int(n_total * 0.15)

    console.print("\n[cyan]Dataset split:[/cyan]")
    console.print(f"  Train: timesteps 0-{train_end} ({train_end} samples)")
    console.print(f"  Val:   timesteps {train_end}-{val_end} ({val_end - train_end} samples)")
    console.print(f"  Test:  timesteps {val_end}-{n_total} ({n_total - val_end} samples)\n")

    # Show XGBoost configuration
    console.print("[cyan]XGBoost configuration:[/cyan]")
    console.print(f"  Window size: {window_size} days")
    console.print(f"  Trees (n_estimators): {n_estimators}")
    console.print(f"  Features per sample: {window_size * u_data.shape[1] * u_data.shape[2]:,}")
    console.print(f"  Spatial locations: {u_data.shape[1] * u_data.shape[2]:,}\n")

    # Run experiment
    console.print("[yellow]Training XGBoost models and evaluating...[/yellow]\n")
    results = run_xgboost_baseline_experiment(
        u_data=u_data,
        v_data=v_data,
        window_size=window_size,
        n_estimators=n_estimators,
        split_ratios=(0.7, 0.15, 0.15),
    )

    # Print results for each split
    for split_name, metrics in results.items():
        print_metrics_table(metrics, split_name, f"XGBoost ({window_size}-day features)")

    # Summary comparison table
    console.print("\n[bold cyan]═══ Summary Across Splits ═══[/bold cyan]\n")

    summary_table = Table(
        title="🚀 XGBoost Baseline - Performance Summary",
        show_header=True,
        header_style="bold magenta",
    )

    summary_table.add_column("Split", style="cyan", width=12)
    summary_table.add_column("R² (U)", justify="right", width=12)
    summary_table.add_column("R² (V)", justify="right", width=12)
    summary_table.add_column("MAE (mag)", justify="right", width=12)
    summary_table.add_column("RMSE (mag)", justify="right", width=12)
    summary_table.add_column("Status", width=10)

    for split_name, metrics in results.items():
        r2_u = metrics["r2_u"]
        r2_v = metrics["r2_v"]
        mae_mag = metrics["mae_magnitude"]
        rmse_mag = metrics["rmse_magnitude"]

        status = "✅" if r2_u > 0.5 and r2_v > 0.5 else "⚠️" if r2_u > 0.3 and r2_v > 0.3 else "❌"

        summary_table.add_row(
            split_name.capitalize(),
            f"{r2_u:.4f}",
            f"{r2_v:.4f}",
            f"{mae_mag:.6f}",
            f"{rmse_mag:.6f}",
            status,
        )

    console.print(summary_table)

    # Save results
    import json

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / f"baseline_xgboost_w{window_size}_t{n_estimators}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]✓ Results saved to: {output_file}[/green]")

    # Performance comparison guidance
    console.print("\n[dim]💡 Interpretation:[/dim]")
    console.print("[dim]• XGBoost should outperform the 7-day mean baseline[/dim]")
    console.print("[dim]• R² > 0.5 indicates the model captures >50% of variance[/dim]")
    console.print("[dim]• Compare with mean baseline results to assess improvement[/dim]")
