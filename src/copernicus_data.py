"""Minimal Copernicus Marine Service (CMEMS) Data Handler
Only downloads and processes the configured datasets
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress


console = Console()


class CopernicusDataHandler:
    """Minimal handler for downloading and processing CMEMS data."""

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.data_dir) / "raw"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import copernicusmarine
            import xarray as xr

            self.copernicusmarine = copernicusmarine
            self.xr = xr
        except ImportError:
            console.print("[red]Run: pip install copernicusmarine xarray netcdf4[/red]")
            raise

    def check_credentials(self):
        """Check if CMEMS credentials are configured."""
        if not hasattr(self.config, "cmems") or not self.config.cmems:
            console.print("[red]Please add CMEMS credentials to config.yaml[/red]")
            return False
        return True

    def download_data(self):
        """Download configured datasets."""
        if not self.check_credentials():
            return None

        console.print(
            f"[yellow]Downloading CMEMS data ({self.config.date_range.start_date} to {self.config.date_range.end_date})...[/yellow]"
        )

        # Use dot notation to access config
        datasets_config = self.config.cmems_datasets
        downloaded_files = {}

        try:
            with Progress() as progress:
                # Get dict items - handle both DotDict and regular dict
                dataset_items = (
                    datasets_config.__dict__.items()
                    if hasattr(datasets_config, "__dict__")
                    else datasets_config.items()
                )

                task = progress.add_task("[green]Downloading...", total=len(list(dataset_items)))

                for data_type, dataset_id in dataset_items:
                    # Access variable mapping with dot notation
                    variables = getattr(self.config.variable_mapping, data_type, [])

                    for var in variables:
                        # Create filename with date range
                        start_year = self.config.date_range.start_date.split("-")[0]
                        end_year = self.config.date_range.end_date.split("-")[0]
                        output_file = (
                            self.output_dir
                            / f"adriatic_{data_type}_{var}_{start_year}_{end_year}.nc"
                        )

                        try:
                            self.copernicusmarine.subset(
                                dataset_id=dataset_id,
                                variables=[var],
                                minimum_longitude=self.config.spatial_bounds.lon_min,
                                maximum_longitude=self.config.spatial_bounds.lon_max,
                                minimum_latitude=self.config.spatial_bounds.lat_min,
                                maximum_latitude=self.config.spatial_bounds.lat_max,
                                start_datetime=f"{self.config.date_range.start_date}T00:00:00",
                                end_datetime=f"{self.config.date_range.end_date}T23:59:59",
                                minimum_depth=self.config.depth.min,
                                maximum_depth=self.config.depth.max,
                                output_filename=str(output_file),
                                username=self.config.cmems.username,
                                password=self.config.cmems.password,
                            )
                            downloaded_files[f"{data_type}_{var}"] = output_file
                        except Exception as e:
                            console.print(f"[yellow]Failed {data_type}/{var}: {str(e)}[/yellow]")

                    progress.advance(task)

            return downloaded_files if downloaded_files else None

        except Exception as e:
            console.print(f"[red]Download error: {str(e)}[/red]")
            return None

    def load_data(self):
        """Load downloaded data."""
        # Download if no files exist
        if not self.output_dir.exists() or len(list(self.output_dir.glob("*.nc"))) == 0:
            downloaded_files = self.download_data()
            if not downloaded_files:
                return None
        else:
            # Find existing files
            downloaded_files = {}
            dataset_items = (
                self.config.cmems_datasets.__dict__.items()
                if hasattr(self.config.cmems_datasets, "__dict__")
                else self.config.cmems_datasets.items()
            )

            for data_type, _ in dataset_items:
                variables = getattr(self.config.variable_mapping, data_type, [])
                for var in variables:
                    files = list(self.output_dir.glob(f"adriatic_{data_type}_{var}_*.nc"))
                    if files:
                        downloaded_files[f"{data_type}_{var}"] = files[0]

        # Load NetCDF files
        datasets = {}
        for file_key, filepath in downloaded_files.items():
            try:
                datasets[file_key] = self.xr.open_dataset(filepath)
            except Exception as e:
                console.print(f"[red]Error loading {file_key}: {str(e)}[/red]")
                return None

        return self._process_data(datasets)

    def _process_data(self, datasets):
        """Process datasets into model format."""
        # Get reference dataset for dimensions
        first_dataset = list(datasets.values())[0]
        time = first_dataset.time.values
        lat_full = (
            first_dataset.latitude.values
            if "latitude" in first_dataset.coords
            else first_dataset.lat.values
        )
        lon_full = (
            first_dataset.longitude.values
            if "longitude" in first_dataset.coords
            else first_dataset.lon.values
        )

        # Store original full resolution
        full_H, full_W = len(lat_full), len(lon_full)

        # Subsample with UNIFORM step to preserve aspect ratio
        # This ensures upsampling back to full resolution is spatially accurate
        target_H, target_W = self.config.spatial_size
        target_pixels = target_H * target_W
        full_pixels = full_H * full_W

        # Calculate uniform step based on target pixel count
        # step² ≈ full_pixels / target_pixels
        import math

        uniform_step = max(1, int(math.sqrt(full_pixels / target_pixels)))

        # Apply uniform step to both dimensions
        lat = lat_full[::uniform_step]
        lon = lon_full[::uniform_step]

        n_times, H, W = len(time), len(lat), len(lon)

        # Initialize arrays (downsampled)
        currents_u = np.zeros((n_times, H, W))
        currents_v = np.zeros((n_times, H, W))
        ssh = np.zeros((n_times, H, W))
        temp = np.zeros((n_times, H, W))
        wind_speed = np.zeros((n_times, H, W))
        wind_direction = np.zeros((n_times, H, W))

        # Initialize full-resolution arrays for currents (for evaluation)
        currents_u_full = np.zeros((n_times, full_H, full_W))
        currents_v_full = np.zeros((n_times, full_H, full_W))

        # Extract data from datasets
        for file_key, ds in datasets.items():
            all_vars = list(ds.data_vars)
            data_var = all_vars[0]
            data = ds[data_var]

            # Handle depth dimension
            if "depth" in data.dims:
                data = data.isel(depth=0)

            # For currents, extract full-resolution data first (for evaluation)
            is_current_u = "currents_uo" in file_key or (
                "currents_u" in file_key and "uo" not in file_key
            )
            is_current_v = "currents_vo" in file_key or (
                "currents_v" in file_key and "vo" not in file_key
            )

            if is_current_u or is_current_v:
                full_values = data.values.copy()
                full_values = np.where(np.isinf(full_values), np.nan, full_values)
                if is_current_u:
                    currents_u_full[:] = full_values
                elif is_current_v:
                    currents_v_full[:] = full_values

            # Subsample spatially for training data (uniform step preserves aspect ratio)
            lat_name = "latitude" if "latitude" in data.dims else "lat"
            lon_name = "longitude" if "longitude" in data.dims else "lon"
            if len(getattr(data, lat_name)) > H:
                data = data.isel(
                    {
                        lat_name: slice(0, None, uniform_step),
                        lon_name: slice(0, None, uniform_step),
                    }
                )
                data = data.isel({lat_name: slice(0, H), lon_name: slice(0, W)})

            # Get values and check for issues
            values = data.values

            # Keep NaN/Inf values as NaN for proper masking
            # Only convert Inf to NaN, keep NaN as NaN
            values = np.where(np.isinf(values), np.nan, values)

            # Map to output arrays
            if "currents_uo" in file_key:
                currents_u[:] = values
            elif "currents_u" in file_key and "uo" not in file_key:
                currents_u[:] = values
            elif "currents_vo" in file_key:
                currents_v[:] = values
            elif "currents_v" in file_key and "vo" not in file_key:
                currents_v[:] = values
            elif "ssh_" in file_key:
                ssh[:] = values
            elif "temperature_" in file_key:
                temp[:] = values
            elif "wind_wind_speed" in file_key:
                # Handle time dimension mismatch for wind data
                min_time = min(len(values), n_times)
                wind_speed[:min_time] = values[:min_time]
            elif "wind_wind_to_direction" in file_key:
                # Handle time dimension mismatch for wind data
                min_time = min(len(values), n_times)
                wind_direction[:min_time] = values[:min_time]

        # Convert wind speed/direction to u/v components
        valid_speed = np.isfinite(wind_speed).sum() > 0
        valid_direction = np.isfinite(wind_direction).sum() > 0

        if valid_speed and valid_direction:
            wind_u = wind_speed * np.cos(np.radians(wind_direction))
            wind_v = wind_speed * np.sin(np.radians(wind_direction))
        else:
            console.print("[red]No valid wind data available![/red]")
            return None

        # Validate final arrays have data
        for name, arr in [
            ("currents_u", currents_u),
            ("currents_v", currents_v),
            ("ssh", ssh),
            ("temp", temp),
            ("wind_u", wind_u),
            ("wind_v", wind_v),
        ]:
            valid = np.isfinite(arr).sum()
            if valid == 0:
                console.print(f"[red]{name}: NO VALID DATA![/red]")
                return None

        # Apply wind lag
        wind_u_lagged = np.roll(wind_u, 1, axis=0)
        wind_v_lagged = np.roll(wind_v, 1, axis=0)

        # Seasonal features
        dates = [pd.to_datetime(t).to_pydatetime() for t in time]
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        seasonal = np.stack(
            [
                np.sin(2 * np.pi * day_of_year / 365.25),
                np.cos(2 * np.pi * day_of_year / 365.25),
            ],
            axis=1,
        )

        # Close datasets
        for ds in datasets.values():
            ds.close()

        final_data = {
            "currents_u": currents_u,
            "currents_v": currents_v,
            "wind_u": wind_u_lagged,
            "wind_v": wind_v_lagged,
            "ssh": ssh,
            "temp": temp,
            "dates": dates,
            "seasonal": seasonal,
            # Full-resolution data for evaluation
            "currents_u_full": currents_u_full,
            "currents_v_full": currents_v_full,
            "full_grid_size": (full_H, full_W),
            "downsampled_grid_size": (H, W),
            "uniform_step": uniform_step,
        }

        return final_data
