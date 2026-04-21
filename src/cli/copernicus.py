"""Copernicus data download CLI commands."""

from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from copernicus_data import CopernicusDataHandler
from src.configs.config import Config


console = Console()
app = typer.Typer(rich_markup_mode="rich")


@app.command()
def download_specific(
    dataset_id: str = typer.Argument(..., help="Dataset ID to download from"),
    variable: str = typer.Argument(..., help="Variable name to download"),
    output_name: str = typer.Option("test_download", help="Output filename (without .nc)"),
):
    """Download a specific variable from a specific dataset for testing."""
    config = Config()
    handler = CopernicusDataHandler(config)

    if not handler.check_credentials():
        return

    console.print(f"[yellow]Downloading {variable} from {dataset_id}[/yellow]")

    output_file = Path(f"{output_name}.nc")

    try:
        handler.copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=[variable],
            minimum_longitude=config.spatial_bounds.lon_min,
            maximum_longitude=config.spatial_bounds.lon_max,
            minimum_latitude=config.spatial_bounds.lat_min,
            maximum_latitude=config.spatial_bounds.lat_max,
            start_datetime=f"{config.date_range.start_date}T00:00:00",
            end_datetime=f"{config.date_range.end_date}T23:59:59",
            minimum_depth=config.depth.min,
            maximum_depth=config.depth.max,
            output_filename=str(output_file),
            username=config.cmems.username,
            password=config.cmems.password,
        )

        console.print(f"[green]✅ Downloaded to {output_file}[/green]")

        # Show what's in the file
        try:
            import xarray as xr

            ds = xr.open_dataset(output_file)
            console.print("\n[cyan]File contents:[/cyan]")
            console.print(f"[dim]Variables: {list(ds.data_vars)}[/dim]")
            console.print(f"[dim]Coordinates: {list(ds.coords)}[/dim]")
            console.print(f"[dim]Shape: {ds.dims}[/dim]")

            # Show data quality
            if list(ds.data_vars):
                var = list(ds.data_vars)[0]
                data = ds[var].values
                console.print(f"\n[cyan]Data quality for {var}:[/cyan]")
                console.print(f"[dim]Total values: {data.size}[/dim]")
                console.print(f"[dim]NaN values: {np.isnan(data).sum()}[/dim]")
                if np.isfinite(data).any():
                    valid_data = data[np.isfinite(data)]
                    console.print(
                        f"[dim]Range: {valid_data.min():.4f} to {valid_data.max():.4f}[/dim]"
                    )
                else:
                    console.print("[red]No valid data found![/red]")

            ds.close()
        except Exception as e:
            console.print(f"[yellow]Could not inspect file: {str(e)}[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Download failed: {str(e)}[/red]")


@app.command()
def test_download():
    """Test downloading a small sample of CMEMS data."""
    config = Config()
    handler = CopernicusDataHandler(config)

    if not handler.check_credentials():
        console.print("[red]Please update your CMEMS credentials in secrets.yaml[/red]")
        return

    console.print("[yellow]Testing download with configured dataset IDs...[/yellow]")

    # Access dataset with dot notation
    test_dataset = config.cmems_datasets.currents
    console.print(f"[cyan]Checking variables in: {test_dataset}[/cyan]")

    try:
        info = handler.copernicusmarine.describe(dataset_id=test_dataset)

        console.print("[green]Available variables:[/green]")
        actual_variables = []
        for var in info.variables:
            var_name = getattr(var, "short_name", "unknown")
            var_long = getattr(var, "long_name", "N/A")
            actual_variables.append(var_name)
            console.print(f"[cyan]• {var_name}: {var_long}[/cyan]")

        # Try to find current-related variables
        current_vars = [
            v
            for v in actual_variables
            if any(term in v.lower() for term in ["current", "velocity", "u", "v"])
        ]

        if current_vars:
            console.print(f"\n[green]Found current variables: {current_vars}[/green]")
            test_var = current_vars[0]
            console.print(f"[yellow]Testing download with: {test_var}[/yellow]")

            # Test download with actual variable name
            output_file = Path("test_download.nc")

            handler.copernicusmarine.subset(
                dataset_id=test_dataset,
                variables=[test_var],
                minimum_longitude=config.spatial_bounds.lon_min,
                maximum_longitude=config.spatial_bounds.lon_max,
                minimum_latitude=config.spatial_bounds.lat_min,
                maximum_latitude=config.spatial_bounds.lat_max,
                start_datetime=f"{config.date_range.start_date}T00:00:00",
                # Just 1 day for testing
                end_datetime=f"{config.date_range.start_date}T00:00:00",
                minimum_depth=config.depth.min,
                maximum_depth=config.depth.max,
                output_filename=str(output_file),
                username=config.cmems.username,
                password=config.cmems.password,
            )

            console.print("[green]✅ Test download successful![/green]")
            console.print(f"[green]File saved: {output_file}[/green]")

            # Clean up test file
            if output_file.exists():
                output_file.unlink()
                console.print("[dim]Test file cleaned up[/dim]")
        else:
            console.print("[red]No current variables found in dataset[/red]")

    except Exception as e:
        console.print(f"[red]❌ Test failed: {str(e)}[/red]")
