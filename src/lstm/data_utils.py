"""Data loading and preprocessing utilities for ocean current prediction."""

import numpy as np
import xarray as xr


def load_currents_data(uo_path, vo_path):
    """Load uo and vo current data from NetCDF files."""
    ds_u = xr.open_dataset(uo_path)
    ds_v = xr.open_dataset(vo_path)

    # Extract the current variables (they're named uo_detided and vo_detided)
    uo_data = ds_u["uo_detided"].values  # Shape: (time, lat, lon)
    vo_data = ds_v["vo_detided"].values  # Shape: (time, lat, lon)

    # Get time coordinates
    time_coords = ds_u["time"].values

    # Close datasets
    ds_u.close()
    ds_v.close()

    return uo_data, vo_data, time_coords


def normalize_currents_train_only(uo_train, vo_train):
    """Compute normalization statistics from training data only."""
    # Find valid data points (not NaN) in training data only
    uo_valid = np.isfinite(uo_train)
    vo_valid = np.isfinite(vo_train)

    # Compute statistics on valid training data
    uo_mean = np.mean(uo_train[uo_valid])
    uo_std = np.std(uo_train[uo_valid])
    vo_mean = np.mean(vo_train[vo_valid])
    vo_std = np.std(vo_train[vo_valid])

    stats = {"uo_mean": uo_mean, "uo_std": uo_std, "vo_mean": vo_mean, "vo_std": vo_std}

    return stats


def apply_normalization(uo_data, vo_data, stats):
    """Apply normalization using pre-computed statistics."""
    # Find valid data points (not NaN)
    uo_valid = np.isfinite(uo_data)
    vo_valid = np.isfinite(vo_data)

    # Normalize
    uo_norm = (uo_data - stats["uo_mean"]) / stats["uo_std"]
    vo_norm = (vo_data - stats["vo_mean"]) / stats["vo_std"]

    # Keep invalid values as NaN
    uo_norm = np.where(uo_valid, uo_norm, np.nan)
    vo_norm = np.where(vo_valid, vo_norm, np.nan)

    return uo_norm, vo_norm


def create_ocean_mask_train_only(uo_train, vo_train, min_valid_fraction=0.8):
    """
    Create a spatial mask for ocean points using only training data coverage.
    This prevents data leakage by not considering test data when creating the mask.

    Args:
        uo_train: UO training current data [time, lat, lon]
        vo_train: VO training current data [time, lat, lon]
        min_valid_fraction: Minimum fraction of valid data required per spatial point

    Returns:
        ocean_mask: Boolean array [lat, lon] indicating valid ocean points
    """
    n_timesteps = uo_train.shape[0]

    # Count valid data for each spatial location in training data only
    uo_valid_count = np.isfinite(uo_train).sum(axis=0)  # [lat, lon]
    vo_valid_count = np.isfinite(vo_train).sum(axis=0)  # [lat, lon]

    # A point is valid if BOTH uo and vo have enough data in training set
    uo_fraction = uo_valid_count / n_timesteps
    vo_fraction = vo_valid_count / n_timesteps

    ocean_mask = (uo_fraction >= min_valid_fraction) & (vo_fraction >= min_valid_fraction)

    n_total_points = ocean_mask.size
    n_ocean_points = ocean_mask.sum()

    return ocean_mask


def apply_ocean_mask(data, ocean_mask):
    """
    Apply ocean mask to data, keeping only valid ocean points.

    Args:
        data: Data array [time, channels, lat, lon] or [time, lat, lon]
        ocean_mask: Boolean mask [lat, lon]

    Returns:
        masked_data: Data with only valid ocean points [time, ..., n_ocean_points]
        ocean_indices: Indices of ocean points for reconstruction
    """
    # Get indices of valid ocean points
    ocean_indices = np.where(ocean_mask)

    if data.ndim == 3:  # [time, lat, lon]
        # Extract only ocean points
        masked_data = data[:, ocean_indices[0], ocean_indices[1]]  # [time, n_ocean_points]
    elif data.ndim == 4:  # [time, channels, lat, lon]
        # Extract only ocean points
        masked_data = data[
            :, :, ocean_indices[0], ocean_indices[1]
        ]  # [time, channels, n_ocean_points]
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")

    return masked_data, ocean_indices


def reconstruct_full_grid(predictions, ocean_mask, ocean_indices, fill_value=0.0):
    """
    Reconstruct full spatial grid from ocean-only predictions.

    Args:
        predictions: Predictions for ocean points [..., n_ocean_points] or [..., 2, n_ocean_points]
        ocean_mask: Boolean mask [lat, lon]
        ocean_indices: Indices of ocean points
        fill_value: Value for non-ocean points

    Returns:
        full_predictions: Full grid predictions [..., lat, lon] or [..., 2, lat, lon]
    """
    lat, lon = ocean_mask.shape

    if predictions.ndim == 2:  # [batch, n_ocean_points]
        batch_size = predictions.shape[0]
        full_predictions = np.full((batch_size, lat, lon), fill_value)
        full_predictions[:, ocean_indices[0], ocean_indices[1]] = predictions
    elif predictions.ndim == 3:  # [batch, 2, n_ocean_points]
        batch_size = predictions.shape[0]
        full_predictions = np.full((batch_size, 2, lat, lon), fill_value)
        full_predictions[:, :, ocean_indices[0], ocean_indices[1]] = predictions
    else:
        raise ValueError(f"Unsupported predictions shape: {predictions.shape}")

    return full_predictions


def temporal_train_test_split(uo_data, vo_data, train_fraction=0.8):
    """
    Perform temporal train/test split for time series data.

    Args:
        uo_data: UO current data [time, lat, lon]
        vo_data: VO current data [time, lat, lon]
        train_fraction: Fraction of data to use for training (from beginning)

    Returns:
        (uo_train, vo_train, uo_test, vo_test): Temporally split data
    """
    n_timesteps = uo_data.shape[0]
    split_idx = int(train_fraction * n_timesteps)

    uo_train = uo_data[:split_idx]
    vo_train = vo_data[:split_idx]
    uo_test = uo_data[split_idx:]
    vo_test = vo_data[split_idx:]

    return uo_train, vo_train, uo_test, vo_test
