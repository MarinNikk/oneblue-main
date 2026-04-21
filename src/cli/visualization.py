"""Visualization CLI commands."""

import typer
from rich.console import Console


# from src.visualization.learning_time_vs_clusters import plot_learning_time_vs_clusters

# from src.visualization.r2_vs_clusters import plot_r2_vs_clusters
# from src.visualization.r2_vs_days import plot_r2_vs_days


console = Console()
app = typer.Typer(rich_markup_mode="rich")


@app.command()
def r2_vs_days(
    max_days: int = typer.Option(30, help="Maximum number of days to analyze"),
    n_clusters: int = typer.Option(100, help="Number of clusters"),
    spred: float = typer.Option(0.5, help="Spread parameter"),
):
    """Plot R² score vs days for RBF network."""
    console.print("[bold blue]🔍 R² Score vs Days Analysis[/bold blue]")


@app.command()
def learning_time_vs_clusters(
    min_clusters: int = typer.Option(10, help="Minimum number of clusters"),
    max_clusters: int = typer.Option(500, help="Maximum number of clusters"),
    step: int = typer.Option(50, help="Step size for cluster range"),
    time_day: int = typer.Option(0, help="Day to analyze"),
    spred: float = typer.Option(0.5, help="Spread parameter"),
):
    """Plot learning time vs number of clusters for RBF network."""
    console.print("[bold blue]⏱️ Learning Time vs Clusters Analysis[/bold blue]")


@app.command()
def r2_vs_clusters(
    min_clusters: int = typer.Option(10, help="Minimum number of clusters"),
    max_clusters: int = typer.Option(500, help="Maximum number of clusters"),
    step: int = typer.Option(50, help="Step size for cluster range"),
    time_day: int = typer.Option(0, help="Day to analyze"),
    spred: float = typer.Option(0.5, help="Spread parameter"),
):
    """Plot R² score vs number of clusters for RBF network."""
    console.print("[bold blue]📊 R² Score vs Clusters Analysis[/bold blue]")


if __name__ == "__main__":
    app()
