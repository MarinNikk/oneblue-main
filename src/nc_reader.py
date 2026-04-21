"""NetCDF File Reader for Currents Data
Simple reader for NC files focusing on ocean currents data
"""

from pathlib import Path

import numpy as np
import xarray as xr
from rich.console import Console
from rich.table import Table


console = Console()


class NCReader:
    """Simple NetCDF reader for currents and oceanographic data."""

    def __init__(self):
        self.console = console

    def load_file(self, filepath: str | Path) -> xr.Dataset | None:
        """Load a NetCDF file."""
        filepath = Path(filepath)

        if not filepath.exists():
            console.print(f"[red]File not found: {filepath}[/red]")
            return None

        if filepath.suffix.lower() not in [".nc", ".netcdf"]:
            console.print(f"[red]Not a NetCDF file: {filepath}[/red]")
            return None

        try:
            console.print(f"[cyan]Loading NetCDF file: {filepath.name}[/cyan]")
            dataset = xr.open_dataset(filepath)
            console.print(f"[green]✅ Successfully loaded {filepath.name}[/green]")
            return dataset
        except Exception as e:
            console.print(f"[red]Error loading {filepath}: {str(e)}[/red]")
            return None

    def inspect_file(self, filepath: str | Path) -> None:
        """Inspect the contents of a NetCDF file."""
        dataset = self.load_file(filepath)
        if dataset is None:
            return

        console.print(f"\n[bold cyan]NetCDF File Information: {Path(filepath).name}[/bold cyan]")

        # Basic info
        console.print("\n[yellow]Dataset Info:[/yellow]")
        console.print(f"Size: {dataset.nbytes / (1024**2):.2f} MB")

        # Dimensions
        console.print("\n[yellow]Dimensions:[/yellow]")
        for dim_name, dim_size in dataset.dims.items():
            console.print(f"  {dim_name}: {dim_size}")

        # Coordinates
        console.print("\n[yellow]Coordinates:[/yellow]")
        for coord_name, coord in dataset.coords.items():
            dtype_str = str(coord.dtype)
            shape_str = "x".join(map(str, coord.shape))
            if coord.size > 0:
                if np.issubdtype(coord.dtype, np.number):
                    min_val = float(coord.min().values)
                    max_val = float(coord.max().values)
                    console.print(
                        f"  {coord_name} ({dtype_str}) [{shape_str}]: {min_val:.4f} to {max_val:.4f}"
                    )
                else:
                    console.print(f"  {coord_name} ({dtype_str}) [{shape_str}]")
            else:
                console.print(f"  {coord_name} ({dtype_str}) [{shape_str}]: empty")

        # Data variables
        console.print("\n[yellow]Data Variables:[/yellow]")
        for var_name, var in dataset.data_vars.items():
            dtype_str = str(var.dtype)
            shape_str = "x".join(map(str, var.shape))

            # Get data statistics if numeric
            if np.issubdtype(var.dtype, np.number) and var.size > 0:
                values = var.values
                finite_mask = np.isfinite(values)
                finite_count = finite_mask.sum()
                total_count = values.size

                if finite_count > 0:
                    finite_values = values[finite_mask]
                    min_val = finite_values.min()
                    max_val = finite_values.max()
                    mean_val = finite_values.mean()
                    std_val = finite_values.std()

                    console.print(f"  {var_name} ({dtype_str}) [{shape_str}]:")
                    console.print(f"    Range: {min_val:.4f} to {max_val:.4f}")
                    console.print(f"    Mean: {mean_val:.4f}, Std: {std_val:.4f}")
                    console.print(
                        f"    Valid: {finite_count}/{total_count} ({100 * finite_count / total_count:.1f}%)"
                    )
                else:
                    console.print(f"  {var_name} ({dtype_str}) [{shape_str}]: no valid data")
            else:
                console.print(f"  {var_name} ({dtype_str}) [{shape_str}]")

            # Show attributes if any
            if var.attrs:
                for attr_name, attr_value in var.attrs.items():
                    console.print(f"    @{attr_name}: {attr_value}")

        # Global attributes
        if dataset.attrs:
            console.print("\n[yellow]Global Attributes:[/yellow]")
            for attr_name, attr_value in dataset.attrs.items():
                console.print(f"  {attr_name}: {attr_value}")

        dataset.close()

    def extract_currents(
        self, filepath: str | Path, u_var: str = None, v_var: str = None
    ) -> dict | None:
        """Extract currents data (u and v components) from NetCDF file."""
        dataset = self.load_file(filepath)
        if dataset is None:
            return None

        try:
            # Auto-detect current variable names if not provided
            if u_var is None or v_var is None:
                u_var, v_var = self._detect_current_variables(dataset)
                if u_var is None or v_var is None:
                    console.print(
                        "[red]Could not detect current variables. Please specify u_var and v_var.[/red]"
                    )
                    dataset.close()
                    return None

            console.print(f"[cyan]Extracting currents: u={u_var}, v={v_var}[/cyan]")

            # Extract data
            u_data = dataset[u_var]
            v_data = dataset[v_var]

            # Get coordinates
            coords = self._extract_coordinates(dataset)

            # Convert to numpy arrays
            u_values = u_data.values
            v_values = v_data.values

            # Basic validation
            if u_values.shape != v_values.shape:
                console.print(f"[red]Shape mismatch: u{u_values.shape} vs v{v_values.shape}[/red]")
                dataset.close()
                return None

            console.print(f"[green]✅ Extracted currents data: {u_values.shape}[/green]")

            # Data quality info
            u_valid = np.isfinite(u_values).sum()
            v_valid = np.isfinite(v_values).sum()
            total = u_values.size

            console.print(
                f"[dim]U component: {u_valid}/{total} valid ({100 * u_valid / total:.1f}%)[/dim]"
            )
            console.print(
                f"[dim]V component: {v_valid}/{total} valid ({100 * v_valid / total:.1f}%)[/dim]"
            )

            result = {
                "u": u_values,
                "v": v_values,
                "coordinates": coords,
                "u_attrs": dict(u_data.attrs),
                "v_attrs": dict(v_data.attrs),
                "shape": u_values.shape,
                "dims": list(u_data.dims),
            }

            dataset.close()
            return result

        except Exception as e:
            console.print(f"[red]Error extracting currents: {str(e)}[/red]")
            dataset.close()
            return None

    def _detect_current_variables(self, dataset: xr.Dataset) -> tuple:
        """Auto-detect current variable names."""
        common_u_names = ["uo", "u", "eastward_sea_water_velocity", "u_velocity", "U"]
        common_v_names = ["vo", "v", "northward_sea_water_velocity", "v_velocity", "V"]

        u_var = None
        v_var = None

        # Look for u component
        for name in common_u_names:
            if name in dataset.data_vars:
                u_var = name
                break

        # Look for v component
        for name in common_v_names:
            if name in dataset.data_vars:
                v_var = name
                break

        if u_var and v_var:
            console.print(f"[green]Auto-detected current variables: u={u_var}, v={v_var}[/green]")
        else:
            console.print("[yellow]Available variables:[/yellow]")
            for var in dataset.data_vars:
                console.print(f"  {var}")

        return u_var, v_var

    def _extract_coordinates(self, dataset: xr.Dataset) -> dict:
        """Extract coordinate information."""
        coords = {}

        # Common coordinate names
        coord_mappings = {
            "time": ["time", "t"],
            "latitude": ["latitude", "lat", "y"],
            "longitude": ["longitude", "lon", "x"],
            "depth": ["depth", "z", "level", "lev"],
        }

        for coord_type, possible_names in coord_mappings.items():
            for name in possible_names:
                if name in dataset.coords:
                    coords[coord_type] = {
                        "name": name,
                        "values": dataset.coords[name].values,
                        "attrs": dict(dataset.coords[name].attrs),
                    }
                    break

        return coords

    def extract_variable(self, filepath: str | Path, variable_name: str) -> dict | None:
        """Extract any variable from NetCDF file."""
        dataset = self.load_file(filepath)
        if dataset is None:
            return None

        if variable_name not in dataset.data_vars:
            console.print(f"[red]Variable '{variable_name}' not found in dataset[/red]")
            console.print("[yellow]Available variables:[/yellow]")
            for var in dataset.data_vars:
                console.print(f"  {var}")
            dataset.close()
            return None

        try:
            console.print(f"[cyan]Extracting variable: {variable_name}[/cyan]")

            var_data = dataset[variable_name]
            coords = self._extract_coordinates(dataset)
            values = var_data.values

            console.print(f"[green]✅ Extracted {variable_name}: {values.shape}[/green]")

            # Data quality info
            if np.issubdtype(values.dtype, np.number):
                valid = np.isfinite(values).sum()
                total = values.size
                console.print(
                    f"[dim]Valid data: {valid}/{total} ({100 * valid / total:.1f}%)[/dim]"
                )

                if valid > 0:
                    finite_values = values[np.isfinite(values)]
                    console.print(
                        f"[dim]Range: {finite_values.min():.4f} to {finite_values.max():.4f}[/dim]"
                    )

            result = {
                "data": values,
                "coordinates": coords,
                "attrs": dict(var_data.attrs),
                "shape": values.shape,
                "dims": list(var_data.dims),
                "dtype": str(values.dtype),
            }

            dataset.close()
            return result

        except Exception as e:
            console.print(f"[red]Error extracting variable: {str(e)}[/red]")
            dataset.close()
            return None

    def list_files(self, directory: str | Path, pattern: str = "*.nc") -> list[Path]:
        """List NetCDF files in directory."""
        directory = Path(directory)
        if not directory.exists():
            console.print(f"[red]Directory not found: {directory}[/red]")
            return []

        files = list(directory.glob(pattern))
        console.print(f"[cyan]Found {len(files)} NetCDF files in {directory}[/cyan]")

        if files:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Filename", style="cyan")
            table.add_column("Size", style="yellow")
            table.add_column("Modified", style="green")

            for file in sorted(files):
                size_mb = file.stat().st_size / (1024**2)
                mtime = file.stat().st_mtime
                from datetime import datetime

                mod_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                table.add_row(file.name, f"{size_mb:.1f} MB", mod_time)

            console.print(table)

        return files


def main():
    """Example usage of NCReader."""
    import sys

    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python nc_reader.py <nc_file_path> [variable_name][/yellow]")
        console.print("[yellow]Examples:[/yellow]")
        console.print("  python nc_reader.py data.nc                    # Inspect file")
        console.print("  python nc_reader.py data.nc currents           # Extract currents")
        console.print(
            "  python nc_reader.py data.nc ssh                # Extract specific variable"
        )
        return

    reader = NCReader()
    filepath = sys.argv[1]

    if len(sys.argv) == 2:
        # Just inspect the file
        reader.inspect_file(filepath)
    elif sys.argv[2].lower() == "currents":
        # Extract currents
        currents = reader.extract_currents(filepath)
        if currents:
            console.print("\n[green]Successfully extracted currents data![/green]")
            console.print(f"U component shape: {currents['u'].shape}")
            console.print(f"V component shape: {currents['v'].shape}")
            console.print(f"Dimensions: {currents['dims']}")
    else:
        # Extract specific variable
        variable = sys.argv[2]
        result = reader.extract_variable(filepath, variable)
        if result:
            console.print(f"\n[green]Successfully extracted {variable}![/green]")
            console.print(f"Shape: {result['shape']}")
            console.print(f"Dimensions: {result['dims']}")
            console.print(f"Data type: {result['dtype']}")


if __name__ == "__main__":
    main()
