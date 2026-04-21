"""
Error Analysis & Diagnostics Module for Ocean Current Prediction Models.

This module provides comprehensive spatial and temporal error analysis tools
for evaluating model performance across different regions, seasons, and
environmental conditions in the Adriatic Sea.

Features:
- Spatial error distribution heatmaps (MAE/RMSE across basin)
- Geographic region analysis (coastal, deep water, complex bathymetry)
- Seasonal performance analysis (winter vs summer patterns)
- Monthly breakdown of prediction accuracy
- Event detection for specific phenomena (storms, current reversals)
"""

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.nc_reader import NCReader


class SpatialErrorAnalyzer:
    """Analyzes spatial distribution of prediction errors."""

    def __init__(
        self,
        u_pred: np.ndarray,
        v_pred: np.ndarray,
        u_true: np.ndarray,
        v_true: np.ndarray,
        coordinates: dict,
        time_coords: np.ndarray | None = None,
    ):
        """
        Initialize spatial error analyzer.

        Args:
            u_pred: Predicted U currents, shape (T, H, W) or (H, W)
            v_pred: Predicted V currents, shape (T, H, W) or (H, W)
            u_true: True U currents, shape (T, H, W) or (H, W)
            v_true: True V currents, shape (T, H, W) or (H, W)
            coordinates: Dict with lat/lon coordinate information
            time_coords: Time coordinates if available
        """
        self.u_pred = np.array(u_pred)
        self.v_pred = np.array(v_pred)
        self.u_true = np.array(u_true)
        self.v_true = np.array(v_true)
        self.coordinates = coordinates
        self.time_coords = time_coords

        # Handle both temporal and non-temporal data
        if self.u_pred.ndim == 3:
            self.temporal = True
            self.n_timesteps, self.height, self.width = self.u_pred.shape
        else:
            self.temporal = False
            self.height, self.width = self.u_pred.shape
            self.n_timesteps = 1

        # Extract coordinates
        self.lat = coordinates.get("latitude", {}).get("values", None)
        self.lon = coordinates.get("longitude", {}).get("values", None)

    def calculate_spatial_errors(self) -> dict[str, np.ndarray]:
        """Calculate spatial error metrics at each grid point."""
        if self.temporal:
            # Average over time for spatial analysis
            u_pred_mean = np.nanmean(self.u_pred, axis=0)
            v_pred_mean = np.nanmean(self.v_pred, axis=0)
            u_true_mean = np.nanmean(self.u_true, axis=0)
            v_true_mean = np.nanmean(self.v_true, axis=0)
        else:
            u_pred_mean = self.u_pred
            v_pred_mean = self.v_pred
            u_true_mean = self.u_true
            v_true_mean = self.v_true

        # Point-wise errors
        mae_u = np.abs(u_pred_mean - u_true_mean)
        mae_v = np.abs(v_pred_mean - v_true_mean)

        mse_u = (u_pred_mean - u_true_mean) ** 2
        mse_v = (v_pred_mean - v_true_mean) ** 2

        rmse_u = np.sqrt(mse_u)
        rmse_v = np.sqrt(mse_v)

        # Magnitude errors
        mag_pred = np.sqrt(u_pred_mean**2 + v_pred_mean**2)
        mag_true = np.sqrt(u_true_mean**2 + v_true_mean**2)
        mae_magnitude = np.abs(mag_pred - mag_true)
        rmse_magnitude = np.sqrt((mag_pred - mag_true) ** 2)

        return {
            "mae_u": mae_u,
            "mae_v": mae_v,
            "rmse_u": rmse_u,
            "rmse_v": rmse_v,
            "mae_magnitude": mae_magnitude,
            "rmse_magnitude": rmse_magnitude,
            "magnitude_pred": mag_pred,
            "magnitude_true": mag_true,
        }

    def create_spatial_heatmaps(self, save_dir: str = "results/spatial_analysis") -> None:
        """Create spatial error distribution heatmaps."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        errors = self.calculate_spatial_errors()

        # Set up the plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Spatial Error Distribution Across Adriatic Basin", fontsize=16)

        # Define plots
        plots = [
            ("mae_u", "MAE U-Component (m/s)", "Reds"),
            ("mae_v", "MAE V-Component (m/s)", "Blues"),
            ("mae_magnitude", "MAE Current Magnitude (m/s)", "Oranges"),
            ("rmse_u", "RMSE U-Component (m/s)", "Reds"),
            ("rmse_v", "RMSE V-Component (m/s)", "Blues"),
            ("rmse_magnitude", "RMSE Current Magnitude (m/s)", "Oranges"),
        ]

        for idx, (error_key, title, cmap) in enumerate(plots):
            row, col = divmod(idx, 3)
            ax = axes[row, col]

            error_data = errors[error_key]
            # Mask out NaN values for visualization
            masked_data = np.ma.masked_invalid(error_data)

            im = ax.imshow(masked_data, cmap=cmap, aspect="auto", origin="lower")
            ax.set_title(title)

            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Set coordinate labels if available
            if self.lat is not None and self.lon is not None:
                # Sample ticks to avoid overcrowding
                y_ticks = np.linspace(0, self.height - 1, 5).astype(int)
                x_ticks = np.linspace(0, self.width - 1, 5).astype(int)

                ax.set_yticks(y_ticks)
                ax.set_xticks(x_ticks)
                ax.set_yticklabels([f"{self.lat[i]:.1f}°N" for i in y_ticks])
                ax.set_xticklabels([f"{self.lon[i]:.1f}°E" for i in x_ticks])

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        plt.tight_layout()
        plt.savefig(Path(save_dir) / "spatial_error_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Spatial error heatmaps saved to {save_dir}/spatial_error_heatmaps.png")

    def identify_problem_regions(self, threshold_percentile: float = 90) -> dict[str, np.ndarray]:
        """Identify geographic regions with systematic prediction challenges."""
        errors = self.calculate_spatial_errors()

        # Define thresholds based on percentiles
        mae_threshold = np.nanpercentile(errors["mae_magnitude"], threshold_percentile)
        rmse_threshold = np.nanpercentile(errors["rmse_magnitude"], threshold_percentile)

        # Identify problem regions
        high_mae_regions = errors["mae_magnitude"] > mae_threshold
        high_rmse_regions = errors["rmse_magnitude"] > rmse_threshold

        # Combined problem regions
        problem_regions = high_mae_regions | high_rmse_regions

        return {
            "high_mae_regions": high_mae_regions,
            "high_rmse_regions": high_rmse_regions,
            "problem_regions": problem_regions,
            "mae_threshold": mae_threshold,
            "rmse_threshold": rmse_threshold,
        }

    def analyze_geographic_zones(self) -> dict[str, dict]:
        """Analyze prediction challenges by geographic zones."""
        errors = self.calculate_spatial_errors()

        if self.lat is None or self.lon is None:
            print("Warning: No coordinate information available for geographic analysis")
            return {}

        zones = {}

        # Define geographic zones for Adriatic Sea
        # Northern Adriatic (shallow water)
        north_mask = self.lat > 44.5  # Venice and above

        # Central Adriatic (intermediate depth)
        central_mask = (self.lat >= 42.0) & (self.lat <= 44.5)

        # Southern Adriatic (deep water)
        south_mask = self.lat < 42.0

        # Coastal zones (near land boundaries)
        # Approximate coastal areas by identifying high gradient regions
        # This is a simplification - ideally use bathymetry data

        zone_masks = {
            "Northern_Adriatic": north_mask,
            "Central_Adriatic": central_mask,
            "Southern_Adriatic": south_mask,
        }

        for zone_name, mask in zone_masks.items():
            if not isinstance(mask, np.ndarray):
                # Convert scalar mask to spatial mask
                mask = np.broadcast_to(mask[:, None], (self.height, self.width))

            zone_errors = {}
            for error_type, error_data in errors.items():
                if "magnitude" in error_type or "mae" in error_type or "rmse" in error_type:
                    zone_values = error_data[mask]
                    zone_errors[error_type] = {
                        "mean": np.nanmean(zone_values),
                        "std": np.nanstd(zone_values),
                        "median": np.nanmedian(zone_values),
                        "percentile_90": np.nanpercentile(zone_values, 90),
                        "n_points": np.sum(~np.isnan(zone_values)),
                    }

            zones[zone_name] = zone_errors

        return zones


class TemporalErrorAnalyzer:
    """Analyzes temporal patterns in prediction errors."""

    def __init__(
        self,
        u_pred: np.ndarray,
        v_pred: np.ndarray,
        u_true: np.ndarray,
        v_true: np.ndarray,
        time_coords: np.ndarray,
    ):
        """
        Initialize temporal error analyzer.

        Args:
            u_pred: Predicted U currents, shape (T, H, W)
            v_pred: Predicted V currents, shape (T, H, W)
            u_true: True U currents, shape (T, H, W)
            v_true: True V currents, shape (T, H, W)
            time_coords: Time coordinates array
        """
        self.u_pred = np.array(u_pred)
        self.v_pred = np.array(v_pred)
        self.u_true = np.array(u_true)
        self.v_true = np.array(v_true)
        self.time_coords = time_coords

        # Convert time coordinates to datetime objects if needed
        if isinstance(time_coords[0], (int, float)):
            # Assume days since some epoch - this may need adjustment based on your data
            self.datetime_coords = [
                datetime(2023, 1, 1) + timedelta(days=float(t)) for t in time_coords
            ]
        else:
            self.datetime_coords = time_coords

    def calculate_temporal_errors(self) -> dict[str, np.ndarray]:
        """Calculate error metrics for each timestep."""
        n_timesteps = len(self.u_pred)

        temporal_errors = {
            "mae_u": np.zeros(n_timesteps),
            "mae_v": np.zeros(n_timesteps),
            "rmse_u": np.zeros(n_timesteps),
            "rmse_v": np.zeros(n_timesteps),
            "mae_magnitude": np.zeros(n_timesteps),
            "rmse_magnitude": np.zeros(n_timesteps),
        }

        for t in range(n_timesteps):
            # Calculate spatial mean errors for this timestep
            u_pred_t = self.u_pred[t]
            v_pred_t = self.v_pred[t]
            u_true_t = self.u_true[t]
            v_true_t = self.v_true[t]

            # Remove NaN values
            valid_mask = (
                ~np.isnan(u_pred_t)
                & ~np.isnan(v_pred_t)
                & ~np.isnan(u_true_t)
                & ~np.isnan(v_true_t)
            )

            if np.any(valid_mask):
                temporal_errors["mae_u"][t] = np.mean(
                    np.abs(u_pred_t[valid_mask] - u_true_t[valid_mask])
                )
                temporal_errors["mae_v"][t] = np.mean(
                    np.abs(v_pred_t[valid_mask] - v_true_t[valid_mask])
                )

                temporal_errors["rmse_u"][t] = np.sqrt(
                    np.mean((u_pred_t[valid_mask] - u_true_t[valid_mask]) ** 2)
                )
                temporal_errors["rmse_v"][t] = np.sqrt(
                    np.mean((v_pred_t[valid_mask] - v_true_t[valid_mask]) ** 2)
                )

                mag_pred_t = np.sqrt(u_pred_t[valid_mask] ** 2 + v_pred_t[valid_mask] ** 2)
                mag_true_t = np.sqrt(u_true_t[valid_mask] ** 2 + v_true_t[valid_mask] ** 2)

                temporal_errors["mae_magnitude"][t] = np.mean(np.abs(mag_pred_t - mag_true_t))
                temporal_errors["rmse_magnitude"][t] = np.sqrt(
                    np.mean((mag_pred_t - mag_true_t) ** 2)
                )
            else:
                # Fill with NaN if no valid data
                for key in temporal_errors:
                    temporal_errors[key][t] = np.nan

        return temporal_errors

    def seasonal_analysis(self, save_dir: str = "results/temporal_analysis") -> dict[str, dict]:
        """Compare winter vs summer prediction accuracy."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        temporal_errors = self.calculate_temporal_errors()

        # Extract months from datetime coordinates
        months = [dt.month for dt in self.datetime_coords]

        # Define seasons
        winter_months = [12, 1, 2]  # DJF
        summer_months = [6, 7, 8]  # JJA

        winter_mask = np.isin(months, winter_months)
        summer_mask = np.isin(months, summer_months)

        seasonal_stats = {}

        for season, mask in [("Winter", winter_mask), ("Summer", summer_mask)]:
            season_stats = {}
            for error_type, error_values in temporal_errors.items():
                season_values = error_values[mask]
                season_values = season_values[~np.isnan(season_values)]

                if len(season_values) > 0:
                    season_stats[error_type] = {
                        "mean": np.mean(season_values),
                        "std": np.std(season_values),
                        "median": np.median(season_values),
                        "min": np.min(season_values),
                        "max": np.max(season_values),
                        "n_samples": len(season_values),
                    }
                else:
                    season_stats[error_type] = {
                        "mean": np.nan,
                        "std": np.nan,
                        "median": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                        "n_samples": 0,
                    }

            seasonal_stats[season] = season_stats

        # Create seasonal comparison plot
        self._plot_seasonal_comparison(seasonal_stats, save_dir)

        return seasonal_stats

    def monthly_breakdown(self, save_dir: str = "results/temporal_analysis") -> dict[int, dict]:
        """Identify which months have highest/lowest prediction error."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        temporal_errors = self.calculate_temporal_errors()
        months = [dt.month for dt in self.datetime_coords]

        monthly_stats = {}

        for month in range(1, 13):
            month_mask = np.array(months) == month
            month_stats = {}

            for error_type, error_values in temporal_errors.items():
                month_values = error_values[month_mask]
                month_values = month_values[~np.isnan(month_values)]

                if len(month_values) > 0:
                    month_stats[error_type] = {
                        "mean": np.mean(month_values),
                        "std": np.std(month_values),
                        "median": np.median(month_values),
                        "n_samples": len(month_values),
                    }
                else:
                    month_stats[error_type] = {
                        "mean": np.nan,
                        "std": np.nan,
                        "median": np.nan,
                        "n_samples": 0,
                    }

            monthly_stats[month] = month_stats

        # Create monthly breakdown plot
        self._plot_monthly_breakdown(monthly_stats, save_dir)

        return monthly_stats

    def detect_extreme_events(self, threshold_percentile: float = 95) -> dict[str, list]:
        """Detect specific phenomena like storms or current reversals."""
        temporal_errors = self.calculate_temporal_errors()

        # Define thresholds for extreme errors
        mae_threshold = np.nanpercentile(temporal_errors["mae_magnitude"], threshold_percentile)
        rmse_threshold = np.nanpercentile(temporal_errors["rmse_magnitude"], threshold_percentile)

        extreme_events = {
            "high_error_events": [],
            "temporal_patterns": {},
            "thresholds": {"mae_threshold": mae_threshold, "rmse_threshold": rmse_threshold},
        }

        # Identify timesteps with extreme errors
        for t, datetime_t in enumerate(self.datetime_coords):
            if (
                temporal_errors["mae_magnitude"][t] > mae_threshold
                or temporal_errors["rmse_magnitude"][t] > rmse_threshold
            ):

                extreme_events["high_error_events"].append(
                    {
                        "datetime": datetime_t,
                        "timestep": t,
                        "mae_magnitude": temporal_errors["mae_magnitude"][t],
                        "rmse_magnitude": temporal_errors["rmse_magnitude"][t],
                    }
                )

        return extreme_events

    def _plot_seasonal_comparison(self, seasonal_stats: dict, save_dir: str) -> None:
        """Plot seasonal comparison of error metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Seasonal Error Analysis: Winter vs Summer", fontsize=14)

        metrics = ["mae_magnitude", "rmse_magnitude", "mae_u", "mae_v"]
        titles = ["MAE Magnitude", "RMSE Magnitude", "MAE U-Component", "MAE V-Component"]

        for idx, (metric, title) in enumerate(zip(metrics, titles, strict=False)):
            row, col = divmod(idx, 2)
            ax = axes[row, col]

            winter_mean = seasonal_stats["Winter"][metric]["mean"]
            summer_mean = seasonal_stats["Summer"][metric]["mean"]
            winter_std = seasonal_stats["Winter"][metric]["std"]
            summer_std = seasonal_stats["Summer"][metric]["std"]

            seasons = ["Winter", "Summer"]
            means = [winter_mean, summer_mean]
            stds = [winter_std, summer_std]

            bars = ax.bar(seasons, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_ylabel(f"{title} (m/s)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, mean_val in zip(bars, means, strict=False):
                if not np.isnan(mean_val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 0.01,
                        f"{mean_val:.4f}",
                        ha="center",
                        va="bottom",
                    )

        plt.tight_layout()
        plt.savefig(Path(save_dir) / "seasonal_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_monthly_breakdown(self, monthly_stats: dict, save_dir: str) -> None:
        """Plot monthly breakdown of error metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Monthly Error Breakdown", fontsize=14)

        months = list(range(1, 13))
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        metrics = ["mae_magnitude", "rmse_magnitude", "mae_u", "mae_v"]
        titles = ["MAE Magnitude", "RMSE Magnitude", "MAE U-Component", "MAE V-Component"]

        for idx, (metric, title) in enumerate(zip(metrics, titles, strict=False)):
            row, col = divmod(idx, 2)
            ax = axes[row, col]

            monthly_means = [monthly_stats[month][metric]["mean"] for month in months]
            monthly_stds = [monthly_stats[month][metric]["std"] for month in months]

            ax.errorbar(
                month_names, monthly_means, yerr=monthly_stds, marker="o", capsize=3, alpha=0.7
            )
            ax.set_ylabel(f"{title} (m/s)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(Path(save_dir) / "monthly_breakdown.png", dpi=300, bbox_inches="tight")
        plt.close()


class ComprehensiveErrorAnalysis:
    """Main class that orchestrates comprehensive error analysis."""

    def __init__(self, data_path: str = "data/raw"):
        """Initialize comprehensive error analyzer."""
        self.data_path = Path(data_path)
        self.reader = NCReader()

    def run_baseline_error_analysis(
        self, baseline_results: dict, save_dir: str = "results/error_analysis"
    ) -> dict:
        """Run comprehensive error analysis on baseline model results."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        print("Loading current data for error analysis...")

        # Load actual current data
        uo_path = self.data_path / "adriatic_currents_uo_detided_2023_2024.nc"
        vo_path = self.data_path / "adriatic_currents_vo_detided_2023_2024.nc"

        uo_data_dict = self.reader.extract_variable(uo_path, "uo")
        vo_data_dict = self.reader.extract_variable(vo_path, "vo")

        if uo_data_dict is None or vo_data_dict is None:
            raise ValueError("Could not load current data")

        # Extract data and coordinates
        uo_data = uo_data_dict["data"]
        vo_data = vo_data_dict["data"]
        coordinates = uo_data_dict["coordinates"]

        print(f"Loaded current data: {uo_data.shape}")
        print(f"Coordinates available: {list(coordinates.keys())}")

        # Use best baseline model results
        best_model = max(baseline_results.keys(), key=lambda k: baseline_results[k]["r2_global"])
        print(f"Analyzing best baseline model: {best_model}-Day Mean")

        predictions = baseline_results[best_model]["predictions"]

        # For demonstration, we'll use a subset of the data that matches predictions
        n_pred_timesteps = predictions.shape[0]
        u_true_subset = uo_data[-n_pred_timesteps:]
        v_true_subset = vo_data[-n_pred_timesteps:]

        # Create time coordinates (simplified - would need actual time data in practice)
        time_coords = np.arange(n_pred_timesteps)

        print("Running spatial error analysis...")

        # Spatial Analysis
        spatial_analyzer = SpatialErrorAnalyzer(
            predictions[:, 0, :],
            predictions[:, 1, :],  # U, V predictions
            u_true_subset,
            v_true_subset,  # U, V true values
            coordinates,
            time_coords,
        )

        spatial_analyzer.create_spatial_heatmaps(save_dir)
        problem_regions = spatial_analyzer.identify_problem_regions()
        geographic_zones = spatial_analyzer.analyze_geographic_zones()

        print("Running temporal error analysis...")

        # Temporal Analysis
        temporal_analyzer = TemporalErrorAnalyzer(
            predictions[:, 0, :], predictions[:, 1, :], u_true_subset, v_true_subset, time_coords
        )

        seasonal_stats = temporal_analyzer.seasonal_analysis(save_dir)
        monthly_stats = temporal_analyzer.monthly_breakdown(save_dir)
        extreme_events = temporal_analyzer.detect_extreme_events()

        # Compile comprehensive results
        comprehensive_results = {
            "model_analyzed": f"{best_model}-Day Mean Baseline",
            "spatial_analysis": {
                "problem_regions": problem_regions,
                "geographic_zones": geographic_zones,
            },
            "temporal_analysis": {
                "seasonal_stats": seasonal_stats,
                "monthly_stats": monthly_stats,
                "extreme_events": extreme_events,
            },
            "data_info": {
                "n_timesteps": n_pred_timesteps,
                "spatial_shape": (u_true_subset.shape[1], u_true_subset.shape[2]),
                "coordinates_available": list(coordinates.keys()),
            },
        }

        # Save comprehensive results
        import json

        results_file = Path(save_dir) / "comprehensive_error_analysis.json"

        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist() if obj.size < 1000 else "Large array - not saved"
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj

        json_safe_results = convert_for_json(comprehensive_results)

        with open(results_file, "w") as f:
            json.dump(json_safe_results, f, indent=2, default=str)

        print("Comprehensive error analysis complete!")
        print(f"Results saved to {save_dir}")
        print("- Spatial heatmaps: spatial_error_heatmaps.png")
        print("- Seasonal analysis: seasonal_comparison.png")
        print("- Monthly breakdown: monthly_breakdown.png")
        print("- Detailed results: comprehensive_error_analysis.json")

        return comprehensive_results


def run_comprehensive_baseline_analysis():
    """Main function to run comprehensive baseline error analysis."""
    print("=== Comprehensive Baseline Model Error Analysis ===")

    # First, run baseline models to get predictions
    print("Step 1: Running baseline models...")

    try:
        from src.baselines.baseline_nday_validation import (
            calculate_baseline_performance,
            load_validation_data,
        )

        # Load validation data
        sequences, targets, ocean_mask, ocean_indices, norm_stats, checkpoint = (
            load_validation_data()
        )

        # Calculate baseline performance for different N-day models
        baseline_results = calculate_baseline_performance(
            sequences, targets, n_days_list=[1, 2, 3, 4, 5, 6, 7]
        )

        print("Step 2: Running comprehensive error analysis...")

        # Run comprehensive error analysis
        analyzer = ComprehensiveErrorAnalysis()
        comprehensive_results = analyzer.run_baseline_error_analysis(
            baseline_results, save_dir="results/comprehensive_error_analysis"
        )

        print("=== Analysis Complete ===")
        return comprehensive_results

    except Exception as e:
        print(f"Error running analysis: {e}")
        print("Please ensure baseline validation data is available")
        raise


if __name__ == "__main__":
    run_comprehensive_baseline_analysis()
