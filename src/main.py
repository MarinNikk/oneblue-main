import warnings

import typer
from rich.console import Console

from src.cli import baseline_models_cli, copernicus, lstm_training, training, visualization


warnings.filterwarnings("ignore")

console = Console()
app = typer.Typer(rich_markup_mode="rich")

# Add sub-commands
app.add_typer(copernicus.app, name="copernicus", help="Copernicus data download commands")
app.add_typer(training.app, name="convlstm", help="ConvLSTM model training commands")
app.add_typer(lstm_training.app, name="lstm", help="LSTM model training commands")
app.add_typer(visualization.app, name="visualization", help="Visualization commands")
app.add_typer(baseline_models_cli.app, name="baselines", help="Baseline average")


if __name__ == "__main__":
    app()
