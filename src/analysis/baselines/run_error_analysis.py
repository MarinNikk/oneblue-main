"""
Standalone script to run comprehensive error analysis on baseline models.

This script loads the raw Adriatic current data, runs baseline models,
and performs comprehensive spatial and temporal error analysis.
"""

import pickle
import sys
from pathlib import Path

import numpy as np


# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.baselines.error_analysis import SpatialErrorAnalyzer, TemporalErrorAnalyzer
from src.baselines.mean_baseline import SevenDayMeanBaseline
from src.nc_reader import NCReader


def load_adriatic_data(data_dir: str = "data/raw"):
    """Load Adriatic Sea current data."""
    data_path = Path(data_dir)
    reader = NCReader()

    print("Loading Adriatic Sea current data...")

    # Load U and V components
    uo_path = data_path / "adriatic_currents_uo_detided_2023_2024.nc"
    vo_path = data_path / "adriatic_currents_vo_detided_2023_2024.nc"

    uo_data_dict = reader.extract_variable(uo_path, "uo")
    vo_data_dict = reader.extract_variable(vo_path, "vo")

    if uo_data_dict is None or vo_data_dict is None:
        raise ValueError("Could not load current data files")

    u_data = uo_data_dict["data"]  # Shape: (T, H, W)
    v_data = vo_data_dict["data"]
    coordinates = uo_data_dict["coordinates"]

    print(f"Loaded data shapes: U={u_data.shape}, V={v_data.shape}")
    print(f"Available coordinates: {list(coordinates.keys())}")

    # Get time coordinates if available
    time_coords = coordinates.get("time", {}).get("values", np.arange(u_data.shape[0]))

    return u_data, v_data, coordinates, time_coords


def run_baseline_models_on_data(u_data, v_data, test_fraction=0.3):
    """Run baseline models on the data and return predictions."""
    print("Running baseline models...")

    n_total = len(u_data)
    test_start = int(n_total * (1 - test_fraction))

    # Split data
    u_train, u_test = u_data[:test_start], u_data[test_start:]
    _v_train, v_test = v_data[:test_start], v_data[test_start:]

    print(f"Train/Test split: {len(u_train)}/{len(u_test)} timesteps")

    # Run 7-day mean baseline
    print("Running 7-day mean baseline...")
    mean_baseline = SevenDayMeanBaseline(window_size=7)

    # Generate predictions for test set
    u_predictions = []
    v_predictions = []

    window_size = 7
    for t in range(window_size, len(u_test)):
        u_window = u_test[t - window_size : t]
        v_window = v_test[t - window_size : t]

        u_pred, v_pred = mean_baseline.predict(u_window, v_window)
        u_predictions.append(u_pred)
        v_predictions.append(v_pred)

    u_predictions = np.array(u_predictions)
    v_predictions = np.array(v_predictions)
    u_true = u_test[window_size:]
    v_true = v_test[window_size:]

    print(f"Generated predictions: {u_predictions.shape}")

    return u_predictions, v_predictions, u_true, v_true


def run_spatial_analysis(u_pred, v_pred, u_true, v_true, coordinates, save_dir):
    """Run spatial error analysis."""
    print("Running spatial error analysis...")

    # For this analysis, average over time to get spatial patterns
    if u_pred.ndim == 3:
        u_pred_spatial = np.nanmean(u_pred, axis=0)
        v_pred_spatial = np.nanmean(v_pred, axis=0)
        u_true_spatial = np.nanmean(u_true, axis=0)
        v_true_spatial = np.nanmean(v_true, axis=0)
    else:
        u_pred_spatial = u_pred
        v_pred_spatial = v_pred
        u_true_spatial = u_true
        v_true_spatial = v_true

    # Initialize spatial analyzer
    spatial_analyzer = SpatialErrorAnalyzer(
        u_pred_spatial, v_pred_spatial, u_true_spatial, v_true_spatial, coordinates
    )

    # Create spatial heatmaps
    spatial_analyzer.create_spatial_heatmaps(save_dir)

    # Identify problem regions
    problem_regions = spatial_analyzer.identify_problem_regions()

    # Analyze geographic zones
    geographic_zones = spatial_analyzer.analyze_geographic_zones()

    return {
        "spatial_analyzer": spatial_analyzer,
        "problem_regions": problem_regions,
        "geographic_zones": geographic_zones,
    }


def run_temporal_analysis(u_pred, v_pred, u_true, v_true, time_coords, save_dir):
    """Run temporal error analysis."""
    print("Running temporal error analysis...")

    # Initialize temporal analyzer
    temporal_analyzer = TemporalErrorAnalyzer(u_pred, v_pred, u_true, v_true, time_coords)

    # Seasonal analysis
    seasonal_stats = temporal_analyzer.seasonal_analysis(save_dir)

    # Monthly breakdown
    monthly_stats = temporal_analyzer.monthly_breakdown(save_dir)

    # Extreme events detection
    extreme_events = temporal_analyzer.detect_extreme_events()

    return {
        "temporal_analyzer": temporal_analyzer,
        "seasonal_stats": seasonal_stats,
        "monthly_stats": monthly_stats,
        "extreme_events": extreme_events,
    }


def run_adriatic_specific_analysis(coordinates, error_data, save_dir):
    """Run Adriatic Sea-specific geographic analysis."""
    print("Running Adriatic-specific geographic analysis...")

    try:
        from src.evaluations.adriatic_geography import run_adriatic_geographic_analysis

        # Run comprehensive geographic analysis
        adriatic_results = run_adriatic_geographic_analysis(coordinates, error_data, save_dir)

        return adriatic_results
    except Exception as e:
        print(f"Warning: Could not run Adriatic-specific analysis: {e}")
        return {}


def main():
    """Main function to run comprehensive error analysis."""
    print("=== Comprehensive Baseline Error Analysis for Adriatic Sea ===\n")

    save_dir = "results/comprehensive_error_analysis"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Load data
        print("Step 1: Loading Adriatic current data...")
        u_data, v_data, coordinates, time_coords = load_adriatic_data()

        # Step 2: Run baseline models
        print("\nStep 2: Running baseline models...")
        u_pred, v_pred, u_true, v_true = run_baseline_models_on_data(u_data, v_data)

        # Step 3: Spatial error analysis
        print("\nStep 3: Spatial error analysis...")
        spatial_results = run_spatial_analysis(
            u_pred, v_pred, u_true, v_true, coordinates, save_dir
        )

        # Step 4: Temporal error analysis
        print("\nStep 4: Temporal error analysis...")
        test_time_coords = (
            time_coords[-len(u_pred) :] if len(time_coords) > len(u_pred) else time_coords
        )
        temporal_results = run_temporal_analysis(
            u_pred, v_pred, u_true, v_true, test_time_coords, save_dir
        )

        # Step 5: Adriatic-specific geographic analysis
        print("\nStep 5: Adriatic-specific geographic analysis...")

        # Calculate magnitude error for geographic analysis
        mag_pred = np.sqrt(u_pred**2 + v_pred**2)
        mag_true = np.sqrt(u_true**2 + v_true**2)
        mag_error = np.abs(mag_pred - mag_true)

        # Average over time for spatial analysis
        spatial_mag_error = np.nanmean(mag_error, axis=0)

        adriatic_results = run_adriatic_specific_analysis(
            coordinates, spatial_mag_error, save_dir + "/adriatic_geography"
        )

        # Step 6: Save comprehensive results
        print("\nStep 6: Saving comprehensive results...")

        comprehensive_results = {
            "model_info": {
                "model_type": "7-Day Mean Baseline",
                "test_data_shape": u_pred.shape,
                "time_period": f"{len(u_true)} timesteps",
            },
            "spatial_analysis": {
                "problem_regions_summary": {
                    "mae_threshold": float(spatial_results["problem_regions"]["mae_threshold"]),
                    "rmse_threshold": float(spatial_results["problem_regions"]["rmse_threshold"]),
                },
                "geographic_zones": spatial_results["geographic_zones"],
            },
            "temporal_analysis": {
                "seasonal_comparison": temporal_results["seasonal_stats"],
                "monthly_breakdown": temporal_results["monthly_stats"],
                "extreme_events_count": len(
                    temporal_results["extreme_events"]["high_error_events"]
                ),
            },
            "adriatic_specific": adriatic_results,
        }

        # Save to pickle for detailed analysis
        with open(Path(save_dir) / "comprehensive_results.pkl", "wb") as f:
            pickle.dump(comprehensive_results, f)

        # Print summary
        print("\n=== ANALYSIS COMPLETE ===")
        print(f"Results saved to: {save_dir}")
        print("- Spatial heatmaps: spatial_error_heatmaps.png")
        print("- Seasonal analysis: seasonal_comparison.png")
        print("- Monthly breakdown: monthly_breakdown.png")

        if adriatic_results:
            print("- Adriatic geographic zones: adriatic_geography/adriatic_geographic_zones.png")
            print("- Zone statistics: adriatic_geography/zone_statistics_comparison.png")

        print("- Comprehensive results: comprehensive_results.pkl")

        # Print key findings
        print("\n=== KEY FINDINGS ===")

        # Seasonal comparison
        if (
            "Winter" in temporal_results["seasonal_stats"]
            and "Summer" in temporal_results["seasonal_stats"]
        ):
            winter_mae = temporal_results["seasonal_stats"]["Winter"]["mae_magnitude"]["mean"]
            summer_mae = temporal_results["seasonal_stats"]["Summer"]["mae_magnitude"]["mean"]
            print("Seasonal Analysis:")
            print(f"  Winter MAE: {winter_mae:.4f} m/s")
            print(f"  Summer MAE: {summer_mae:.4f} m/s")
            if not np.isnan(winter_mae) and not np.isnan(summer_mae):
                seasonal_diff = ((winter_mae - summer_mae) / summer_mae) * 100
                print(f"  Winter vs Summer: {seasonal_diff:+.1f}% difference")

        # Geographic zones with highest errors
        if spatial_results["geographic_zones"]:
            print("\nGeographic Analysis:")
            for zone_name, zone_stats in spatial_results["geographic_zones"].items():
                if "mae_magnitude" in zone_stats and not np.isnan(
                    zone_stats["mae_magnitude"]["mean"]
                ):
                    print(f"  {zone_name}: {zone_stats['mae_magnitude']['mean']:.4f} m/s MAE")

        print(
            f"\nExtreme Events: {len(temporal_results['extreme_events']['high_error_events'])} high-error timesteps detected"
        )

        return comprehensive_results

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
