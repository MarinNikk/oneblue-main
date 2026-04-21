"""
Adriatic Sea Geographic Analysis Module.

Provides detailed geographic and bathymetric analysis specific to the Adriatic Sea,
including identification of coastal zones, deep vs shallow water areas, complex
bathymetry regions, and circulation features.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class AdriaticGeographyAnalyzer:
    """Analyzes Adriatic Sea-specific geographic features and bathymetry."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        """
        Initialize with coordinate arrays.

        Args:
            lat: Latitude array (1D or 2D)
            lon: Longitude array (1D or 2D)
        """
        self.lat = lat if lat.ndim == 2 else np.meshgrid(lon, lat)[1]
        self.lon = lon if lon.ndim == 2 else np.meshgrid(lon, lat)[0]

        # Adriatic Sea geographic boundaries
        self.adriatic_bounds = {
            "north": 45.8,  # Trieste area
            "south": 39.5,  # Otranto Strait
            "west": 12.0,  # Italian coast
            "east": 20.0,  # Croatian/Albanian coast
        }

    def create_bathymetry_zones(self) -> dict[str, np.ndarray]:
        """
        Create bathymetry-based zones for the Adriatic Sea.

        Returns masks for different depth zones based on typical Adriatic bathymetry.
        Note: This uses geographic approximations. For accurate analysis,
        actual bathymetry data should be used.
        """

        # Approximate bathymetry zones based on Adriatic Sea characteristics
        zones = {}

        # Northern Adriatic (generally shallow, <50m)
        # Venice area and northern regions
        northern_shallow = self.lat > 44.5
        zones["northern_shallow"] = northern_shallow

        # Central Adriatic West (Italian side, moderate depth 50-200m)
        central_west = (self.lat >= 42.0) & (self.lat <= 44.5) & (self.lon <= 15.5)
        zones["central_west"] = central_west

        # Central Adriatic East (Croatian side, varied bathymetry)
        central_east = (self.lat >= 42.0) & (self.lat <= 44.5) & (self.lon > 15.5)
        zones["central_east"] = central_east

        # Southern Adriatic Deep Basin (>1000m in central part)
        # Deepest part of the Adriatic
        southern_deep = (self.lat < 42.0) & (self.lon >= 14.0) & (self.lon <= 18.0)
        zones["southern_deep"] = southern_deep

        # Otranto Strait region (connection to Ionian Sea)
        otranto_strait = (self.lat < 40.5) & (self.lon >= 18.0)
        zones["otranto_strait"] = otranto_strait

        return zones

    def identify_coastal_zones(self, coastal_width_km: float = 30) -> dict[str, np.ndarray]:
        """
        Identify coastal boundary zones.

        Args:
            coastal_width_km: Width of coastal zone in kilometers

        Returns:
            Dictionary with coastal zone masks for different sides of the Adriatic
        """
        # Approximate coastal zones using lat/lon boundaries
        # This is simplified - ideally would use coastline data

        coastal_zones = {}

        # Convert km to approximate degrees (rough approximation)
        lat_degree_km = 111.0  # km per degree latitude
        coastal_width_deg = coastal_width_km / lat_degree_km

        # Italian (western) coast
        italian_coast = self.lon <= (self.adriatic_bounds["west"] + coastal_width_deg)
        coastal_zones["italian_coast"] = italian_coast

        # Croatian/Balkan (eastern) coast
        eastern_coast = self.lon >= (self.adriatic_bounds["east"] - coastal_width_deg)
        coastal_zones["eastern_coast"] = eastern_coast

        # Northern coast (around Venice/Trieste)
        northern_coast = self.lat >= (self.adriatic_bounds["north"] - coastal_width_deg)
        coastal_zones["northern_coast"] = northern_coast

        # Southern coast (Puglia/Albania)
        southern_coast = self.lat <= (self.adriatic_bounds["south"] + coastal_width_deg)
        coastal_zones["southern_coast"] = southern_coast

        # Combined coastal areas
        all_coastal = italian_coast | eastern_coast | northern_coast | southern_coast
        coastal_zones["all_coastal"] = all_coastal

        # Open sea (non-coastal areas)
        open_sea = ~all_coastal
        coastal_zones["open_sea"] = open_sea

        return coastal_zones

    def identify_complex_bathymetry_regions(self) -> dict[str, np.ndarray]:
        """
        Identify regions with complex bathymetry features.

        Returns:
            Dictionary with masks for straits, channels, and complex topography
        """
        complex_regions = {}

        # Strait of Otranto (narrow passage to Ionian Sea)
        otranto_strait = (
            (self.lat >= 39.5) & (self.lat <= 40.5) & (self.lon >= 18.0) & (self.lon <= 19.5)
        )
        complex_regions["otranto_strait"] = otranto_strait

        # Palagruža Sill (important topographic feature in central Adriatic)
        # Shallow ridge between central basins
        palagruza_sill = (
            (self.lat >= 42.2) & (self.lat <= 43.0) & (self.lon >= 15.5) & (self.lon <= 16.5)
        )
        complex_regions["palagruza_sill"] = palagruza_sill

        # Jabuka Pit area (deep depression)
        jabuka_pit = (
            (self.lat >= 42.8) & (self.lat <= 43.3) & (self.lon >= 15.0) & (self.lon <= 15.8)
        )
        complex_regions["jabuka_pit"] = jabuka_pit

        # Croatian island channels (complex coastline with many islands)
        croatian_channels = (
            (self.lat >= 42.5) & (self.lat <= 45.0) & (self.lon >= 15.0) & (self.lon <= 17.5)
        )
        complex_regions["croatian_channels"] = croatian_channels

        # Venice Lagoon area (complex shallow water environment)
        venice_lagoon = (
            (self.lat >= 45.2) & (self.lat <= 45.8) & (self.lon >= 12.0) & (self.lon <= 12.8)
        )
        complex_regions["venice_lagoon"] = venice_lagoon

        return complex_regions

    def identify_circulation_features(self) -> dict[str, np.ndarray]:
        """
        Identify areas influenced by specific circulation features.

        Returns:
            Dictionary with masks for different circulation patterns
        """
        circulation_features = {}

        # Eastern Adriatic Current (flows along eastern coast)
        # Typically stronger current along Croatian/Albanian coast
        eastern_adriatic_current = (self.lon >= 15.5) & (self.lat >= 40.0) & (self.lat <= 45.0)
        circulation_features["eastern_adriatic_current"] = eastern_adriatic_current

        # Western Adriatic Coastal Current (along Italian coast)
        western_adriatic_current = (self.lon <= 15.0) & (self.lat >= 40.0) & (self.lat <= 44.5)
        circulation_features["western_adriatic_current"] = western_adriatic_current

        # South Adriatic Gyre (cyclonic circulation in southern basin)
        # Typically located in the deepest part of southern Adriatic
        south_adriatic_gyre = (
            (self.lat >= 40.5) & (self.lat <= 42.5) & (self.lon >= 15.0) & (self.lon <= 18.0)
        )
        circulation_features["south_adriatic_gyre"] = south_adriatic_gyre

        # Middle Adriatic Jet (fast current in central region)
        middle_adriatic_jet = (
            (self.lat >= 42.5) & (self.lat <= 44.0) & (self.lon >= 14.5) & (self.lon <= 16.5)
        )
        circulation_features["middle_adriatic_jet"] = middle_adriatic_jet

        # Bora Wind Influence Zones (areas strongly affected by Bora winds)
        # Northeast Adriatic where Bora winds are strongest
        bora_influence_north = (self.lat >= 44.0) & (self.lon >= 13.0) & (self.lon <= 16.0)
        circulation_features["bora_influence_north"] = bora_influence_north

        # Central Adriatic Bora influence
        bora_influence_central = (
            (self.lat >= 42.5) & (self.lat <= 44.0) & (self.lon >= 14.0) & (self.lon <= 17.0)
        )
        circulation_features["bora_influence_central"] = bora_influence_central

        return circulation_features

    def analyze_zone_statistics(
        self, error_data: np.ndarray, zones: dict[str, np.ndarray]
    ) -> dict[str, dict]:
        """
        Calculate statistics for each geographic zone.

        Args:
            error_data: Error data array (same shape as lat/lon grids)
            zones: Dictionary of zone masks

        Returns:
            Dictionary with statistics for each zone
        """
        zone_stats = {}

        for zone_name, zone_mask in zones.items():
            # Extract values for this zone
            zone_values = error_data[zone_mask]

            # Remove NaN values
            valid_values = zone_values[~np.isnan(zone_values)]

            if len(valid_values) > 0:
                stats = {
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values)),
                    "median": float(np.median(valid_values)),
                    "min": float(np.min(valid_values)),
                    "max": float(np.max(valid_values)),
                    "percentile_25": float(np.percentile(valid_values, 25)),
                    "percentile_75": float(np.percentile(valid_values, 75)),
                    "percentile_90": float(np.percentile(valid_values, 90)),
                    "percentile_95": float(np.percentile(valid_values, 95)),
                    "n_points": len(valid_values),
                    "coverage_percent": float(len(valid_values) / len(zone_values) * 100),
                }
            else:
                stats = {
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "percentile_25": np.nan,
                    "percentile_75": np.nan,
                    "percentile_90": np.nan,
                    "percentile_95": np.nan,
                    "n_points": 0,
                    "coverage_percent": 0.0,
                }

            zone_stats[zone_name] = stats

        return zone_stats

    def create_geographic_zones_plot(self, save_dir: str = "results/geographic_analysis"):
        """Create visualization of all geographic zones."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Get all zone types
        bathymetry_zones = self.create_bathymetry_zones()
        coastal_zones = self.identify_coastal_zones()
        complex_regions = self.identify_complex_bathymetry_regions()
        circulation_features = self.identify_circulation_features()

        # Create comprehensive plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Adriatic Sea Geographic Zones for Error Analysis", fontsize=16)

        # Plot 1: Bathymetry zones
        ax1 = axes[0, 0]
        self._plot_zone_overlay(bathymetry_zones, ax1, "Bathymetry Zones")

        # Plot 2: Coastal zones
        ax2 = axes[0, 1]
        self._plot_zone_overlay(coastal_zones, ax2, "Coastal Zones")

        # Plot 3: Complex bathymetry
        ax3 = axes[1, 0]
        self._plot_zone_overlay(complex_regions, ax3, "Complex Bathymetry Regions")

        # Plot 4: Circulation features
        ax4 = axes[1, 1]
        self._plot_zone_overlay(circulation_features, ax4, "Circulation Features")

        plt.tight_layout()
        plt.savefig(Path(save_dir) / "adriatic_geographic_zones.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Geographic zones visualization saved to {save_dir}/adriatic_geographic_zones.png")

    def _plot_zone_overlay(self, zones: dict[str, np.ndarray], ax, title: str):
        """Plot multiple zones with different colors."""
        # Create base map
        ax.contour(self.lon, self.lat, self.lat, levels=10, colors="lightgray", alpha=0.3)

        # Color map for zones
        colors = plt.cm.tab10(np.linspace(0, 1, len(zones)))

        # Plot each zone
        for idx, (_zone_name, zone_mask) in enumerate(zones.items()):
            if np.any(zone_mask):
                # Create masked array for plotting
                zone_plot = np.ma.masked_where(~zone_mask, np.ones_like(zone_mask))
                ax.contourf(
                    self.lon,
                    self.lat,
                    zone_plot,
                    levels=[0.5, 1.5],
                    colors=[colors[idx]],
                    alpha=0.6,
                )

        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Set Adriatic Sea bounds
        ax.set_xlim(self.adriatic_bounds["west"], self.adriatic_bounds["east"])
        ax.set_ylim(self.adriatic_bounds["south"], self.adriatic_bounds["north"])

        # Add legend
        legend_elements = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=colors[idx],
                alpha=0.6,
                label=zone_name.replace("_", " ").title(),
            )
            for idx, zone_name in enumerate(zones.keys())
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    def run_comprehensive_geographic_analysis(
        self, error_data: np.ndarray, save_dir: str = "results/geographic_analysis"
    ) -> dict:
        """
        Run comprehensive geographic analysis.

        Args:
            error_data: Error data array (H, W)
            save_dir: Directory to save results

        Returns:
            Comprehensive analysis results
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        print("Running comprehensive geographic analysis...")

        # Get all zone types
        bathymetry_zones = self.create_bathymetry_zones()
        coastal_zones = self.identify_coastal_zones()
        complex_regions = self.identify_complex_bathymetry_regions()
        circulation_features = self.identify_circulation_features()

        # Calculate statistics for each zone type
        results = {
            "bathymetry_zones": self.analyze_zone_statistics(error_data, bathymetry_zones),
            "coastal_zones": self.analyze_zone_statistics(error_data, coastal_zones),
            "complex_regions": self.analyze_zone_statistics(error_data, complex_regions),
            "circulation_features": self.analyze_zone_statistics(error_data, circulation_features),
        }

        # Create visualizations
        self.create_geographic_zones_plot(save_dir)

        # Create detailed statistics plots
        self._create_zone_statistics_plots(results, save_dir)

        # Save detailed results
        with open(Path(save_dir) / "adriatic_geographic_analysis.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Geographic analysis complete! Results saved to {save_dir}")
        return results

    def _create_zone_statistics_plots(self, results: dict, save_dir: str):
        """Create bar plots comparing statistics across zones."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Error Statistics by Geographic Zones", fontsize=16)

        zone_types = [
            "bathymetry_zones",
            "coastal_zones",
            "complex_regions",
            "circulation_features",
        ]
        titles = ["Bathymetry Zones", "Coastal Zones", "Complex Regions", "Circulation Features"]

        for idx, (zone_type, title) in enumerate(zip(zone_types, titles, strict=False)):
            row, col = divmod(idx, 2)
            ax = axes[row, col]

            zone_data = results[zone_type]
            zone_names = list(zone_data.keys())
            mean_errors = [zone_data[name]["mean"] for name in zone_names]
            std_errors = [zone_data[name]["std"] for name in zone_names]

            # Filter out NaN values
            valid_indices = [i for i, val in enumerate(mean_errors) if not np.isnan(val)]
            if valid_indices:
                valid_names = [zone_names[i].replace("_", "\n") for i in valid_indices]
                valid_means = [mean_errors[i] for i in valid_indices]
                valid_stds = [std_errors[i] for i in valid_indices]

                bars = ax.bar(valid_names, valid_means, yerr=valid_stds, capsize=3, alpha=0.7)
                ax.set_ylabel("Mean Error (m/s)")
                ax.set_title(title)
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, mean_val in zip(bars, valid_means, strict=False):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 0.001,
                        f"{mean_val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        plt.tight_layout()
        plt.savefig(Path(save_dir) / "zone_statistics_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()


def run_adriatic_geographic_analysis(
    coordinates: dict, error_data: np.ndarray, save_dir: str = "results/adriatic_analysis"
) -> dict:
    """
    Main function to run Adriatic-specific geographic analysis.

    Args:
        coordinates: Dictionary with latitude/longitude data
        error_data: Error data array
        save_dir: Directory to save results

    Returns:
        Comprehensive geographic analysis results
    """
    # Extract coordinate arrays
    lat_data = coordinates.get("latitude", {}).get("values", None)
    lon_data = coordinates.get("longitude", {}).get("values", None)

    if lat_data is None or lon_data is None:
        raise ValueError("Latitude and longitude coordinates are required")

    # Initialize analyzer
    analyzer = AdriaticGeographyAnalyzer(lat_data, lon_data)

    # Run comprehensive analysis
    results = analyzer.run_comprehensive_geographic_analysis(error_data, save_dir)

    return results
