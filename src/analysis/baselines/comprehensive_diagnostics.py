"""
Comprehensive Error Analysis & Diagnostics for Adriatic Ocean Current Prediction.

This module provides detailed, publication-quality analysis including:
- Seasonal analysis (Winter/Spring/Summer/Autumn)
- Geographic regions (North/Central/South Adriatic)
- Coastal vs Deep vs Complex bathymetry zones
- Monthly breakdown with clear patterns
- Circulation feature analysis
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


warnings.filterwarnings("ignore")

# Set plotting style
plt.style.use("default")
sns.set_palette("husl")


class ComprehensiveAdriaticAnalyzer:
    """Comprehensive analysis of prediction errors across the Adriatic Sea."""

    def __init__(
        self,
        u_pred: np.ndarray,
        v_pred: np.ndarray,
        u_true: np.ndarray,
        v_true: np.ndarray,
        time_coords: np.ndarray,
        coordinates: dict,
    ):
        """
        Initialize comprehensive analyzer.

        Args:
            u_pred: Predicted U currents (T, H, W)
            v_pred: Predicted V currents (T, H, W)
            u_true: True U currents (T, H, W)
            v_true: True V currents (T, H, W)
            time_coords: Time coordinate strings
            coordinates: Coordinate dictionary with lat/lon
        """
        self.u_pred = np.array(u_pred)
        self.v_pred = np.array(v_pred)
        self.u_true = np.array(u_true)
        self.v_true = np.array(v_true)
        self.time_coords = time_coords
        self.coordinates = coordinates

        # Extract coordinate grids
        self.lat = coordinates.get("latitude", {}).get("values", None)
        self.lon = coordinates.get("longitude", {}).get("values", None)

        # Create coordinate meshgrids if needed
        if self.lat is not None and self.lon is not None:
            if self.lat.ndim == 1 and self.lon.ndim == 1:
                self.lon_grid, self.lat_grid = np.meshgrid(self.lon, self.lat)
            else:
                self.lat_grid = self.lat
                self.lon_grid = self.lon

        # Extract dates and create date-based indices
        self.dates = self._extract_dates()
        self.seasonal_indices = self._create_seasonal_indices()
        self.monthly_indices = self._create_monthly_indices()

        print(f"Initialized analyzer: {len(self.u_pred)} timesteps")
        print(f"Date range: {self.dates[0]} to {self.dates[-1]}")
        print(f"Seasons covered: {list(self.seasonal_indices.keys())}")

    def _extract_dates(self) -> list[datetime]:
        """Extract datetime objects from time coordinates."""
        dates = []
        for time_str in self.time_coords:
            try:
                if isinstance(time_str, str):
                    # Parse ISO format: YYYY-MM-DDTHH:MM:SS
                    date_part = time_str.split("T")[0]
                    year, month, day = map(int, date_part.split("-"))
                    dates.append(datetime(year, month, day))
                else:
                    # Fallback for other formats
                    dates.append(datetime(2024, 1, 1))  # Default
            except (ValueError, IndexError, TypeError):
                dates.append(datetime(2024, 1, 1))  # Default fallback
        return dates

    def _create_seasonal_indices(self) -> dict[str, list[int]]:
        """Create indices for each season."""
        seasonal_indices = {
            "Winter": [],  # Dec, Jan, Feb
            "Spring": [],  # Mar, Apr, May
            "Summer": [],  # Jun, Jul, Aug
            "Autumn": [],  # Sep, Oct, Nov
        }

        for i, date in enumerate(self.dates):
            month = date.month
            if month in [12, 1, 2]:
                seasonal_indices["Winter"].append(i)
            elif month in [3, 4, 5]:
                seasonal_indices["Spring"].append(i)
            elif month in [6, 7, 8]:
                seasonal_indices["Summer"].append(i)
            else:  # [9, 10, 11]
                seasonal_indices["Autumn"].append(i)

        return seasonal_indices

    def _create_monthly_indices(self) -> dict[int, list[int]]:
        """Create indices for each month."""
        monthly_indices = {month: [] for month in range(1, 13)}

        for i, date in enumerate(self.dates):
            monthly_indices[date.month].append(i)

        return monthly_indices

    def _define_geographic_regions(self) -> dict[str, np.ndarray]:
        """Define geographic regions of the Adriatic Sea."""
        if self.lat_grid is None or self.lon_grid is None:
            return {}

        regions = {}

        # === MAIN ADRIATIC REGIONS ===
        # Northern Adriatic (Venice area, shallow)
        regions["North_Adriatic"] = self.lat_grid > 44.5

        # Central Adriatic (intermediate depth)
        regions["Central_Adriatic"] = (self.lat_grid >= 42.0) & (self.lat_grid <= 44.5)

        # Southern Adriatic (deep basin)
        regions["South_Adriatic"] = self.lat_grid < 42.0

        # === COASTAL VS OPEN SEA ===
        # Italian (western) coast - within ~30km of coast
        regions["Italian_Coast"] = self.lon_grid < 14.5

        # Croatian (eastern) coast - within ~30km of coast
        regions["Croatian_Coast"] = self.lon_grid > 16.0

        # Open sea (central Adriatic)
        regions["Open_Sea"] = (self.lon_grid >= 14.5) & (self.lon_grid <= 16.0)

        # === DEPTH-BASED REGIONS (approximate) ===
        # Shallow water (typically <50m)
        regions["Shallow_Water"] = self.lat_grid > 44.0  # Northern regions

        # Intermediate depth (50-200m)
        regions["Intermediate_Depth"] = (self.lat_grid >= 42.5) & (self.lat_grid <= 44.0)

        # Deep water (>200m)
        regions["Deep_Water"] = self.lat_grid < 42.5

        # === COMPLEX BATHYMETRY REGIONS ===
        # Strait of Otranto (complex topography)
        regions["Otranto_Strait"] = (self.lat_grid < 40.5) & (self.lon_grid > 18.0)

        # Central basin channels (between Italian/Croatian coasts)
        regions["Central_Channels"] = (
            (self.lat_grid >= 42.0)
            & (self.lat_grid <= 44.0)
            & (self.lon_grid >= 14.5)
            & (self.lon_grid <= 16.0)
        )

        # Croatian archipelago (complex coastline)
        regions["Croatian_Islands"] = (
            (self.lat_grid >= 42.5)
            & (self.lat_grid <= 45.0)
            & (self.lon_grid >= 15.0)
            & (self.lon_grid <= 17.5)
        )

        return regions

    def calculate_seasonal_errors(self, save_dir: str) -> dict:
        """Calculate detailed seasonal error statistics."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        seasonal_results = {}
        regions = self._define_geographic_regions()

        # Colors for seasons
        season_colors = {
            "Winter": "#1f77b4",  # Blue
            "Spring": "#2ca02c",  # Green
            "Summer": "#ff7f0e",  # Orange
            "Autumn": "#d62728",  # Red
        }

        # Calculate errors for each season
        for season, indices in self.seasonal_indices.items():
            if len(indices) == 0:
                continue

            print(f"Analyzing {season}: {len(indices)} days")

            # Extract seasonal data
            u_pred_season = self.u_pred[indices]
            v_pred_season = self.v_pred[indices]
            u_true_season = self.u_true[indices]
            v_true_season = self.v_true[indices]

            # Calculate seasonal mean spatial patterns
            u_pred_mean = np.nanmean(u_pred_season, axis=0)
            v_pred_mean = np.nanmean(v_pred_season, axis=0)
            u_true_mean = np.nanmean(u_true_season, axis=0)
            v_true_mean = np.nanmean(v_true_season, axis=0)

            # Calculate error fields
            np.abs(u_pred_mean - u_true_mean)
            np.abs(v_pred_mean - v_true_mean)
            mae_magnitude = np.abs(
                np.sqrt(u_pred_mean**2 + v_pred_mean**2) - np.sqrt(u_true_mean**2 + v_true_mean**2)
            )

            # Calculate temporal error evolution
            temporal_mae = []
            for t in range(len(indices)):
                valid_mask = (
                    ~np.isnan(u_pred_season[t])
                    & ~np.isnan(v_pred_season[t])
                    & ~np.isnan(u_true_season[t])
                    & ~np.isnan(v_true_season[t])
                )

                if np.sum(valid_mask) > 100:
                    mag_pred = np.sqrt(
                        u_pred_season[t][valid_mask] ** 2 + v_pred_season[t][valid_mask] ** 2
                    )
                    mag_true = np.sqrt(
                        u_true_season[t][valid_mask] ** 2 + v_true_season[t][valid_mask] ** 2
                    )
                    temporal_mae.append(np.mean(np.abs(mag_pred - mag_true)))
                else:
                    temporal_mae.append(np.nan)

            # Regional analysis for this season
            regional_stats = {}
            for region_name, region_mask in regions.items():
                if np.any(region_mask):
                    region_mae = mae_magnitude[region_mask]
                    valid_mae = region_mae[~np.isnan(region_mae)]

                    if len(valid_mae) > 0:
                        regional_stats[region_name] = {
                            "mean": float(np.mean(valid_mae)),
                            "std": float(np.std(valid_mae)),
                            "median": float(np.median(valid_mae)),
                            "n_points": len(valid_mae),
                        }

            # Store seasonal results
            seasonal_results[season] = {
                "mae_magnitude_field": mae_magnitude,
                "temporal_mae": temporal_mae,
                "regional_stats": regional_stats,
                "n_days": len(indices),
                "overall_mae": float(
                    np.nanmean([mae for mae in temporal_mae if not np.isnan(mae)])
                ),
            }

        # Create seasonal comparison plots
        self._create_seasonal_plots(seasonal_results, season_colors, save_dir)

        return seasonal_results

    def _create_seasonal_plots(self, seasonal_results: dict, colors: dict, save_dir: str):
        """Create comprehensive seasonal comparison plots."""

        # === PLOT 1: Seasonal spatial heatmaps ===
        seasons_with_data = [s for s in seasonal_results if seasonal_results[s]["n_days"] > 0]
        n_seasons = len(seasons_with_data)

        if n_seasons > 0:
            fig, axes = plt.subplots(2, n_seasons, figsize=(4 * n_seasons, 8))
            if n_seasons == 1:
                axes = axes.reshape(-1, 1)
            elif axes.ndim == 1:
                axes = axes.reshape(2, -1)

            # Top row: MAE magnitude heatmaps
            for i, season in enumerate(seasons_with_data):
                mae_field = seasonal_results[season]["mae_magnitude_field"]

                ax = axes[0, i] if n_seasons > 1 else axes[0]
                im = ax.imshow(
                    mae_field,
                    cmap="Reds",
                    vmin=0,
                    vmax=np.nanpercentile(mae_field, 95),
                    origin="lower",
                    aspect="auto",
                )
                ax.set_title(
                    f'{season} MAE\n({seasonal_results[season]["n_days"]} days)', fontsize=12
                )
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                plt.colorbar(im, ax=ax, label="MAE (m/s)", shrink=0.8)

                # Add geographic annotations
                if self.lat_grid is not None:
                    ax.contour(
                        self.lat_grid, levels=[42, 44.5], colors="black", alpha=0.3, linewidths=1
                    )
                    ax.text(
                        0.02,
                        0.98,
                        "N",
                        transform=ax.transAxes,
                        fontsize=10,
                        bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "alpha": 0.8},
                    )
                    ax.text(
                        0.02,
                        0.50,
                        "C",
                        transform=ax.transAxes,
                        fontsize=10,
                        bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "alpha": 0.8},
                    )
                    ax.text(
                        0.02,
                        0.02,
                        "S",
                        transform=ax.transAxes,
                        fontsize=10,
                        bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "alpha": 0.8},
                    )

            # Bottom row: Regional comparison
            for i, season in enumerate(seasons_with_data):
                ax = axes[1, i] if n_seasons > 1 else axes[1]

                # Plot North/Central/South comparison
                regions_to_plot = ["North_Adriatic", "Central_Adriatic", "South_Adriatic"]
                region_labels = ["North", "Central", "South"]

                regional_stats = seasonal_results[season]["regional_stats"]
                means = []
                stds = []
                labels = []

                for region, label in zip(regions_to_plot, region_labels, strict=False):
                    if region in regional_stats:
                        means.append(regional_stats[region]["mean"])
                        stds.append(regional_stats[region]["std"])
                        labels.append(label)

                if means:
                    bars = ax.bar(
                        labels, means, yerr=stds, capsize=5, alpha=0.7, color=colors[season]
                    )
                    ax.set_ylabel("MAE (m/s)")
                    ax.set_title(f"{season} Regional Errors")
                    ax.grid(True, alpha=0.3)

                    # Add value labels
                    for bar, mean_val in zip(bars, means, strict=False):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            bar.get_height() + bar.get_height() * 0.05,
                            f"{mean_val:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=10,
                        )

            plt.tight_layout()
            plt.savefig(
                Path(save_dir) / "seasonal_spatial_analysis.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        # === PLOT 2: Seasonal overview comparison ===
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Seasonal Error Analysis Overview", fontsize=16, fontweight="bold")

        # Top-left: Overall seasonal comparison
        ax1 = axes[0, 0]
        seasons = []
        overall_maes = []
        n_days_list = []
        season_colors_list = []

        for season in ["Winter", "Spring", "Summer", "Autumn"]:
            if season in seasonal_results and seasonal_results[season]["n_days"] > 0:
                seasons.append(season)
                overall_maes.append(seasonal_results[season]["overall_mae"])
                n_days_list.append(seasonal_results[season]["n_days"])
                season_colors_list.append(colors[season])

        if seasons:
            bars = ax1.bar(seasons, overall_maes, color=season_colors_list, alpha=0.7)
            ax1.set_ylabel("Overall MAE (m/s)")
            ax1.set_title("Seasonal Error Comparison")
            ax1.grid(True, alpha=0.3)

            # Add annotations
            for bar, mae, days in zip(bars, overall_maes, n_days_list, strict=False):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.0005,
                    f"{mae:.3f}\n({days} days)",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        # Top-right: North vs South comparison across seasons
        ax2 = axes[0, 1]
        north_maes = []
        south_maes = []

        for season in seasons:
            if season in seasonal_results:
                regional_stats = seasonal_results[season]["regional_stats"]
                north_mae = regional_stats.get("North_Adriatic", {}).get("mean", np.nan)
                south_mae = regional_stats.get("South_Adriatic", {}).get("mean", np.nan)
                north_maes.append(north_mae if not np.isnan(north_mae) else 0)
                south_maes.append(south_mae if not np.isnan(south_mae) else 0)

        if seasons:
            x = np.arange(len(seasons))
            width = 0.35

            bars1 = ax2.bar(
                x - width / 2,
                north_maes,
                width,
                label="North Adriatic",
                color="lightblue",
                alpha=0.8,
            )
            bars2 = ax2.bar(
                x + width / 2, south_maes, width, label="South Adriatic", color="darkred", alpha=0.8
            )

            ax2.set_ylabel("MAE (m/s)")
            ax2.set_title("North vs South Adriatic Seasonal Comparison")
            ax2.set_xticks(x)
            ax2.set_xticklabels(seasons)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    if bar.get_height() > 0:
                        ax2.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            bar.get_height() + 0.0001,
                            f"{bar.get_height():.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                        )

        # Bottom-left: Temporal variability
        ax3 = axes[1, 0]
        for season in seasons:
            if season in seasonal_results:
                temporal_mae = seasonal_results[season]["temporal_mae"]
                valid_mae = [mae for mae in temporal_mae if not np.isnan(mae)]
                if valid_mae:
                    ax3.plot(
                        range(len(valid_mae)),
                        valid_mae,
                        "o-",
                        color=colors[season],
                        label=f"{season} ({len(valid_mae)} days)",
                        alpha=0.7,
                        linewidth=2,
                        markersize=4,
                    )

        ax3.set_xlabel("Day within Season")
        ax3.set_ylabel("Daily MAE (m/s)")
        ax3.set_title("Temporal Error Evolution by Season")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Bottom-right: Regional error distribution
        ax4 = axes[1, 1]

        # Create box plot of regional errors
        regions_for_box = [
            "North_Adriatic",
            "Central_Adriatic",
            "South_Adriatic",
            "Italian_Coast",
            "Croatian_Coast",
            "Open_Sea",
        ]
        region_labels_box = ["North", "Central", "South", "IT Coast", "HR Coast", "Open Sea"]

        box_data = []
        box_labels = []

        for region, label in zip(regions_for_box, region_labels_box, strict=False):
            region_values = []
            for season in seasons:
                if (
                    season in seasonal_results
                    and region in seasonal_results[season]["regional_stats"]
                ):
                    region_values.append(seasonal_results[season]["regional_stats"][region]["mean"])

            if region_values:
                box_data.append(region_values)
                box_labels.append(label)

        if box_data:
            box_plot = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)

            # Color boxes
            for patch in box_plot["boxes"]:
                patch.set_facecolor("lightblue")
                patch.set_alpha(0.7)

            ax4.set_ylabel("MAE (m/s)")
            ax4.set_title("Regional Error Distribution Across Seasons")
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(Path(save_dir) / "seasonal_overview.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Seasonal analysis plots saved to {save_dir}/")

    def calculate_monthly_patterns(self, save_dir: str) -> dict:
        """Calculate detailed monthly error patterns."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        monthly_results = {}
        regions = self._define_geographic_regions()

        # Calculate for each month
        for month in range(1, 13):
            indices = self.monthly_indices[month]
            if len(indices) == 0:
                continue

            print(f"Analyzing month {month}: {len(indices)} days")

            # Extract monthly data
            u_pred_month = self.u_pred[indices]
            v_pred_month = self.v_pred[indices]
            u_true_month = self.u_true[indices]
            v_true_month = self.v_true[indices]

            # Calculate temporal MAE for this month
            temporal_mae = []
            for t in range(len(indices)):
                valid_mask = (
                    ~np.isnan(u_pred_month[t])
                    & ~np.isnan(v_pred_month[t])
                    & ~np.isnan(u_true_month[t])
                    & ~np.isnan(v_true_month[t])
                )

                if np.sum(valid_mask) > 100:
                    mag_pred = np.sqrt(
                        u_pred_month[t][valid_mask] ** 2 + v_pred_month[t][valid_mask] ** 2
                    )
                    mag_true = np.sqrt(
                        u_true_month[t][valid_mask] ** 2 + v_true_month[t][valid_mask] ** 2
                    )
                    temporal_mae.append(np.mean(np.abs(mag_pred - mag_true)))
                else:
                    temporal_mae.append(np.nan)

            # Regional analysis for this month
            u_pred_mean = np.nanmean(u_pred_month, axis=0)
            v_pred_mean = np.nanmean(v_pred_month, axis=0)
            u_true_mean = np.nanmean(u_true_month, axis=0)
            v_true_mean = np.nanmean(v_true_month, axis=0)

            mae_magnitude = np.abs(
                np.sqrt(u_pred_mean**2 + v_pred_mean**2) - np.sqrt(u_true_mean**2 + v_true_mean**2)
            )

            regional_stats = {}
            for region_name, region_mask in regions.items():
                if np.any(region_mask):
                    region_mae = mae_magnitude[region_mask]
                    valid_mae = region_mae[~np.isnan(region_mae)]

                    if len(valid_mae) > 0:
                        regional_stats[region_name] = {
                            "mean": float(np.mean(valid_mae)),
                            "std": float(np.std(valid_mae)),
                            "n_points": len(valid_mae),
                        }

            monthly_results[month] = {
                "temporal_mae": temporal_mae,
                "regional_stats": regional_stats,
                "n_days": len(indices),
                "overall_mae": float(
                    np.nanmean([mae for mae in temporal_mae if not np.isnan(mae)])
                ),
            }

        # Create monthly plots
        self._create_monthly_plots(monthly_results, save_dir)

        return monthly_results

    def _create_monthly_plots(self, monthly_results: dict, save_dir: str):
        """Create detailed monthly analysis plots."""

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

        months_with_data = [month for month in range(1, 13) if month in monthly_results]

        # === MAIN MONTHLY OVERVIEW ===
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Monthly Error Analysis - Adriatic Sea Current Prediction",
            fontsize=16,
            fontweight="bold",
        )

        # Top-left: Overall monthly progression
        ax1 = axes[0, 0]

        if months_with_data:
            month_labels = [month_names[m - 1] for m in months_with_data]
            overall_maes = [monthly_results[m]["overall_mae"] for m in months_with_data]
            n_days_list = [monthly_results[m]["n_days"] for m in months_with_data]

            # Color by season
            colors = []
            for month in months_with_data:
                if month in [12, 1, 2]:
                    colors.append("#1f77b4")  # Winter - Blue
                elif month in [3, 4, 5]:
                    colors.append("#2ca02c")  # Spring - Green
                elif month in [6, 7, 8]:
                    colors.append("#ff7f0e")  # Summer - Orange
                else:
                    colors.append("#d62728")  # Autumn - Red

            bars = ax1.bar(month_labels, overall_maes, color=colors, alpha=0.7)
            ax1.set_ylabel("Overall MAE (m/s)")
            ax1.set_title("Monthly Error Progression")
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis="x", rotation=45)

            # Add value labels and sample counts
            for bar, mae, days in zip(bars, overall_maes, n_days_list, strict=False):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.0005,
                    f"{mae:.3f}\n({days}d)",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Top-right: North vs South monthly comparison
        ax2 = axes[0, 1]

        north_monthly = []
        south_monthly = []

        for month in months_with_data:
            regional_stats = monthly_results[month]["regional_stats"]
            north_mae = regional_stats.get("North_Adriatic", {}).get("mean", np.nan)
            south_mae = regional_stats.get("South_Adriatic", {}).get("mean", np.nan)
            north_monthly.append(north_mae if not np.isnan(north_mae) else 0)
            south_monthly.append(south_mae if not np.isnan(south_mae) else 0)

        if month_labels:
            x = np.arange(len(month_labels))
            width = 0.35

            ax2.bar(
                x - width / 2,
                north_monthly,
                width,
                label="North Adriatic",
                color="lightblue",
                alpha=0.8,
            )
            ax2.bar(
                x + width / 2,
                south_monthly,
                width,
                label="South Adriatic",
                color="darkred",
                alpha=0.8,
            )

            ax2.set_ylabel("MAE (m/s)")
            ax2.set_title("North vs South Adriatic Monthly Comparison")
            ax2.set_xticks(x)
            ax2.set_xticklabels(month_labels, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Bottom-left: Coastal vs Open Sea monthly patterns
        ax3 = axes[1, 0]

        italian_coast_monthly = []
        croatian_coast_monthly = []
        open_sea_monthly = []

        for month in months_with_data:
            regional_stats = monthly_results[month]["regional_stats"]
            italian = regional_stats.get("Italian_Coast", {}).get("mean", 0)
            croatian = regional_stats.get("Croatian_Coast", {}).get("mean", 0)
            open_sea = regional_stats.get("Open_Sea", {}).get("mean", 0)

            italian_coast_monthly.append(italian)
            croatian_coast_monthly.append(croatian)
            open_sea_monthly.append(open_sea)

        if month_labels:
            ax3.plot(
                month_labels,
                italian_coast_monthly,
                "o-",
                label="Italian Coast",
                color="green",
                linewidth=2,
                markersize=6,
            )
            ax3.plot(
                month_labels,
                croatian_coast_monthly,
                "s-",
                label="Croatian Coast",
                color="orange",
                linewidth=2,
                markersize=6,
            )
            ax3.plot(
                month_labels,
                open_sea_monthly,
                "^-",
                label="Open Sea",
                color="blue",
                linewidth=2,
                markersize=6,
            )

            ax3.set_ylabel("MAE (m/s)")
            ax3.set_title("Coastal vs Open Sea Monthly Patterns")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis="x", rotation=45)

        # Bottom-right: Depth zones monthly comparison
        ax4 = axes[1, 1]

        shallow_monthly = []
        intermediate_monthly = []
        deep_monthly = []

        for month in months_with_data:
            regional_stats = monthly_results[month]["regional_stats"]
            shallow = regional_stats.get("Shallow_Water", {}).get("mean", 0)
            intermediate = regional_stats.get("Intermediate_Depth", {}).get("mean", 0)
            deep = regional_stats.get("Deep_Water", {}).get("mean", 0)

            shallow_monthly.append(shallow)
            intermediate_monthly.append(intermediate)
            deep_monthly.append(deep)

        if month_labels:
            ax4.plot(
                month_labels,
                shallow_monthly,
                "o-",
                label="Shallow (<50m)",
                color="lightgreen",
                linewidth=2,
                markersize=6,
            )
            ax4.plot(
                month_labels,
                intermediate_monthly,
                "s-",
                label="Intermediate (50-200m)",
                color="yellow",
                linewidth=2,
                markersize=6,
            )
            ax4.plot(
                month_labels,
                deep_monthly,
                "^-",
                label="Deep (>200m)",
                color="darkblue",
                linewidth=2,
                markersize=6,
            )

            ax4.set_ylabel("MAE (m/s)")
            ax4.set_title("Depth-Based Monthly Error Patterns")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            Path(save_dir) / "monthly_comprehensive_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # === DETAILED MONTHLY HEATMAP ===
        if len(months_with_data) > 6:  # Only create if we have sufficient data
            fig, ax = plt.subplots(figsize=(14, 8))

            # Create matrix for heatmap
            regions_for_heatmap = [
                "North_Adriatic",
                "Central_Adriatic",
                "South_Adriatic",
                "Italian_Coast",
                "Croatian_Coast",
                "Open_Sea",
                "Shallow_Water",
                "Intermediate_Depth",
                "Deep_Water",
            ]
            region_labels_heatmap = [
                "North Adriatic",
                "Central Adriatic",
                "South Adriatic",
                "Italian Coast",
                "Croatian Coast",
                "Open Sea",
                "Shallow Water",
                "Intermediate Depth",
                "Deep Water",
            ]

            heatmap_data = []
            for region in regions_for_heatmap:
                row_data = []
                for month in range(1, 13):
                    if (
                        month in monthly_results
                        and region in monthly_results[month]["regional_stats"]
                    ):
                        row_data.append(monthly_results[month]["regional_stats"][region]["mean"])
                    else:
                        row_data.append(np.nan)
                heatmap_data.append(row_data)

            heatmap_data = np.array(heatmap_data)

            # Create heatmap
            sns.heatmap(
                heatmap_data,
                xticklabels=month_names,
                yticklabels=region_labels_heatmap,
                cmap="Reds",
                annot=True,
                fmt=".3f",
                cbar_kws={"label": "MAE (m/s)"},
                ax=ax,
            )

            ax.set_title(
                "Monthly Regional Error Heatmap - Adriatic Sea", fontsize=14, fontweight="bold"
            )
            ax.set_xlabel("Month")
            ax.set_ylabel("Geographic Region")

            plt.tight_layout()
            plt.savefig(
                Path(save_dir) / "monthly_regional_heatmap.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        print(f"Monthly analysis plots saved to {save_dir}/")

    def create_comprehensive_summary(
        self, seasonal_results: dict, monthly_results: dict, save_dir: str
    ) -> dict:
        """Create comprehensive analysis summary."""

        summary = {
            "analysis_overview": {
                "total_timesteps": len(self.u_pred),
                "date_range": f"{self.dates[0].strftime('%Y-%m-%d')} to {self.dates[-1].strftime('%Y-%m-%d')}",
                "seasons_analyzed": list(seasonal_results.keys()),
                "months_analyzed": list(monthly_results.keys()),
            }
        }

        # Seasonal summary
        seasonal_summary = {}
        for season, results in seasonal_results.items():
            regional_stats = results["regional_stats"]

            seasonal_summary[season] = {
                "n_days": results["n_days"],
                "overall_mae": results["overall_mae"],
                "north_adriatic_mae": regional_stats.get("North_Adriatic", {}).get("mean", np.nan),
                "south_adriatic_mae": regional_stats.get("South_Adriatic", {}).get("mean", np.nan),
                "coastal_vs_open": {
                    "italian_coast": regional_stats.get("Italian_Coast", {}).get("mean", np.nan),
                    "croatian_coast": regional_stats.get("Croatian_Coast", {}).get("mean", np.nan),
                    "open_sea": regional_stats.get("Open_Sea", {}).get("mean", np.nan),
                },
            }

        summary["seasonal_analysis"] = seasonal_summary

        # Monthly summary
        monthly_summary = {}
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

        for month, results in monthly_results.items():
            regional_stats = results["regional_stats"]

            monthly_summary[month_names[month - 1]] = {
                "n_days": results["n_days"],
                "overall_mae": results["overall_mae"],
                "north_south_ratio": (
                    regional_stats.get("South_Adriatic", {}).get("mean", 1)
                    / max(regional_stats.get("North_Adriatic", {}).get("mean", 1), 0.001)
                ),
            }

        summary["monthly_analysis"] = monthly_summary

        # Key findings
        if seasonal_results:
            # Best and worst seasons
            season_maes = [
                (season, data["overall_mae"]) for season, data in seasonal_results.items()
            ]
            season_maes.sort(key=lambda x: x[1])

            summary["key_findings"] = {
                "best_season": season_maes[0][0] if season_maes else None,
                "worst_season": season_maes[-1][0] if season_maes else None,
                "seasonal_variation": (
                    (season_maes[-1][1] - season_maes[0][1]) if len(season_maes) > 1 else 0
                ),
            }

        # Save summary
        with open(Path(save_dir) / "comprehensive_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def run_full_analysis(self, save_dir: str = "results/comprehensive_diagnostics") -> dict:
        """Run complete comprehensive analysis."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        print("=== COMPREHENSIVE ADRIATIC ERROR ANALYSIS ===")
        print(f"Analysis period: {self.dates[0]} to {self.dates[-1]}")
        print(f"Total timesteps: {len(self.u_pred)}")

        # Run seasonal analysis
        print("\n1. Running seasonal analysis...")
        seasonal_results = self.calculate_seasonal_errors(save_dir + "/seasonal")

        # Run monthly analysis
        print("\n2. Running monthly analysis...")
        monthly_results = self.calculate_monthly_patterns(save_dir + "/monthly")

        # Create comprehensive summary
        print("\n3. Creating comprehensive summary...")
        summary = self.create_comprehensive_summary(seasonal_results, monthly_results, save_dir)

        print("\n=== ANALYSIS COMPLETE ===")
        print(f"Results saved to: {save_dir}/")
        print("Generated files:")
        print("  - seasonal/seasonal_spatial_analysis.png")
        print("  - seasonal/seasonal_overview.png")
        print("  - monthly/monthly_comprehensive_analysis.png")
        print("  - monthly/monthly_regional_heatmap.png")
        print("  - comprehensive_summary.json")

        return {
            "seasonal_results": seasonal_results,
            "monthly_results": monthly_results,
            "summary": summary,
        }


def run_comprehensive_analysis():
    """Main function to run comprehensive analysis on baseline model."""
    import sys

    sys.path.append("src")

    from nc_reader import NCReader
    from src.baselines.mean_baseline import SevenDayMeanBaseline

    print("=== COMPREHENSIVE ADRIATIC BASELINE ANALYSIS ===")

    # Load data
    reader = NCReader()

    uo_data_dict = reader.extract_variable(
        "data/raw/adriatic_currents_uo_detided_2023_2024.nc", "uo_detided"
    )
    vo_data_dict = reader.extract_variable(
        "data/raw/adriatic_currents_vo_detided_2023_2024.nc", "vo_detided"
    )

    if uo_data_dict is None or vo_data_dict is None:
        print("Error: Could not load data")
        return None

    u_data = uo_data_dict["data"]
    v_data = vo_data_dict["data"]
    coordinates = uo_data_dict["coordinates"]
    time_coords = coordinates["time"]["values"]

    print(f"Loaded dataset: {u_data.shape[0]} days ({time_coords[0]} to {time_coords[-1]})")

    # Split data (70/30)
    n_total = len(u_data)
    train_size = int(0.7 * n_total)

    _u_train, u_test = u_data[:train_size], u_data[train_size:]
    _v_train, v_test = v_data[:train_size], v_data[train_size:]
    test_time_coords = time_coords[train_size:]

    # Run baseline model
    baseline = SevenDayMeanBaseline(window_size=7)

    u_predictions, v_predictions = [], []
    valid_indices = []

    for t in range(7, len(u_test)):
        u_window = u_test[t - 7 : t]
        v_window = v_test[t - 7 : t]

        if np.sum(~np.isnan(u_window)) / u_window.size > 0.1:
            u_pred, v_pred = baseline.predict(u_window, v_window)
            u_predictions.append(u_pred)
            v_predictions.append(v_pred)
            valid_indices.append(t)

    u_predictions = np.array(u_predictions)
    v_predictions = np.array(v_predictions)
    u_true = u_test[np.array(valid_indices)]
    v_true = v_test[np.array(valid_indices)]
    valid_time_coords = test_time_coords[np.array(valid_indices)]

    print(f"Generated {len(u_predictions)} predictions")

    # Initialize comprehensive analyzer
    analyzer = ComprehensiveAdriaticAnalyzer(
        u_predictions, v_predictions, u_true, v_true, valid_time_coords, coordinates
    )

    # Run full analysis
    results = analyzer.run_full_analysis("results/comprehensive_diagnostics")

    return results


if __name__ == "__main__":
    run_comprehensive_analysis()
