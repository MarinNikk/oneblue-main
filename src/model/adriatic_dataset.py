import pickle

import numpy as np
import torch
from rich.console import Console
from torch.utils.data import Dataset


console = Console()


class AdriaticDataset(Dataset):
    def __init__(self, data, sequence_length=7, prediction_horizon=1):
        self.seq_len = sequence_length
        self.pred_horizon = prediction_horizon

        # Stack input features: [time, channels, H, W]
        self.inputs = np.stack(
            [
                data["currents_u"],
                data["currents_v"],
                data["wind_u"],
                data["wind_v"],
                data["ssh"],
                data["temp"],
            ],
            axis=1,
        )  # [time, 6, H, W]

        # Targets: only current components (downsampled)
        self.targets = np.stack([data["currents_u"], data["currents_v"]], axis=1)  # [time, 2, H, W]

        # Full-resolution targets for evaluation (if available)
        if "currents_u_full" in data and data["currents_u_full"] is not None:
            self.targets_full = np.stack(
                [data["currents_u_full"], data["currents_v_full"]], axis=1
            )  # [time, 2, full_H, full_W]
            self.full_grid_size = data.get("full_grid_size", None)
            self.uniform_step = data.get("uniform_step", None)
        else:
            self.targets_full = None
            self.full_grid_size = None
            self.uniform_step = None

        # Seasonal features
        self.seasonal = data["seasonal"]  # [time, 2]

        # Find valid sequences (no NaN values in inputs or targets)
        self.valid_indices = self._find_valid_sequences()
        self.n_samples = len(self.valid_indices)

    def _find_valid_sequences(self):
        """Find sequence indices that contain no NaN values."""
        valid_indices = []
        total_sequences = len(self.inputs) - self.seq_len - self.pred_horizon + 1

        for idx in range(total_sequences):
            # Check input sequence for NaN
            x_sequence = self.inputs[idx : idx + self.seq_len]
            # Check targets for NaN (all timesteps from 1 to prediction horizon)
            y_valid_fractions = []
            for h in range(1, self.pred_horizon + 1):
                y_target_h = self.targets[idx + self.seq_len + h - 1]
                y_valid_fractions.append(np.isfinite(y_target_h).mean())
            y_valid_fraction = min(y_valid_fractions)  # Most restrictive check

            # Check if sequence has enough valid data (at least 25% of spatial points valid)
            x_valid_fraction = np.isfinite(x_sequence).mean()

            # Accept sequences with at least 25% valid data (accounting for wind data gaps)
            if x_valid_fraction >= 0.25 and y_valid_fraction >= 0.25:
                valid_indices.append(idx)

        return valid_indices

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Map to actual valid index
        actual_idx = self.valid_indices[idx]

        # Input sequence
        x = self.inputs[actual_idx : actual_idx + self.seq_len]  # [seq_len, 6, H, W]

        # Seasonal context for the sequence
        seasonal = self.seasonal[actual_idx : actual_idx + self.seq_len]  # [seq_len, 2]

        # Targets (all timesteps from 1 to prediction horizon)
        y_list = []
        for h in range(1, self.pred_horizon + 1):
            y_h = self.targets[actual_idx + self.seq_len + h - 1]  # [2, H, W]
            y_list.append(y_h)
        y = np.stack(y_list, axis=0)  # [prediction_horizon, 2, H, W]

        # Full-resolution targets (if available)
        if self.targets_full is not None:
            y_full_list = []
            for h in range(1, self.pred_horizon + 1):
                y_full_h = self.targets_full[
                    actual_idx + self.seq_len + h - 1
                ]  # [2, full_H, full_W]
                y_full_list.append(y_full_h)
            y_full = np.stack(y_full_list, axis=0)  # [prediction_horizon, 2, full_H, full_W]
            y_full = np.nan_to_num(y_full, nan=0.0)
        else:
            y_full = None

        # Handle remaining NaN values (should be minimal after filtering)
        # Replace with zeros only for land/boundary areas
        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        if y_full is not None:
            return (
                torch.FloatTensor(x),
                torch.FloatTensor(seasonal),
                torch.FloatTensor(y),
                torch.FloatTensor(y_full),
            )
        return (torch.FloatTensor(x), torch.FloatTensor(seasonal), torch.FloatTensor(y))


def normalize_data(data, stats_file="normalization_stats.pkl", train_stats=None):
    """Normalize data using provided statistics or compute from data.

    Args:
        data: Dictionary with data to normalize
        stats_file: Path to save/load statistics (only used when train_stats=None)
        train_stats: If provided, use these stats instead of computing from data.

    Returns:
        normalized_data: Normalized data dictionary
        stats: Statistics used for normalization
    """
    variables = ["currents_u", "currents_v", "wind_u", "wind_v", "ssh", "temp"]

    if train_stats is not None:
        stats = train_stats
    else:
        # Compute statistics from this data (for training data only)
        stats = {}

        for var in variables:
            arr = data[var]
            valid_mask = np.isfinite(arr)
            valid_count = valid_mask.sum()

            if valid_count == 0:
                mean, std = 0.0, 1.0
            else:
                valid_data = arr[valid_mask]
                mean = np.mean(valid_data)
                std = np.std(valid_data)

                # Prevent division by zero
                if std < 1e-10:
                    std = 1.0

            stats[var] = {"mean": mean, "std": std}

        # Save statistics (only when computing from training data)
        with open(stats_file, "wb") as f:
            pickle.dump(stats, f)

    # Apply normalization
    normalized_data = data.copy()

    for var in variables:
        arr = data[var]
        mean = stats[var]["mean"]
        std = stats[var]["std"]

        valid_mask = np.isfinite(arr)
        normalized_arr = (arr - mean) / std
        normalized_arr = np.where(valid_mask, normalized_arr, np.nan)
        normalized_data[var] = normalized_arr

    # Filter out spatial locations with too many NaN values
    variables = ["currents_u", "currents_v", "wind_u", "wind_v", "ssh", "temp"]

    # Find spatial locations that have enough valid data (>40% valid across time)
    n_timesteps, height, width = normalized_data["currents_u"].shape
    valid_locations = np.ones((height, width), dtype=bool)

    for var in variables:
        var_data = normalized_data[var]
        valid_fraction = np.isfinite(var_data).sum(axis=0) / n_timesteps
        var_valid_locations = valid_fraction > 0.4
        valid_locations = valid_locations & var_valid_locations

    n_valid_locations = valid_locations.sum()

    if n_valid_locations == 0:
        console.print("[red]No spatial locations with sufficient valid data found![/red]")
        return None, None

    # Apply spatial mask to all variables
    filtered_data = {}
    for var in variables:
        var_data = normalized_data[var].copy()
        var_data[:, ~valid_locations] = np.nan
        filtered_data[var] = var_data

    # Also copy other data (seasonal, dates if present)
    if "seasonal" in normalized_data:
        filtered_data["seasonal"] = normalized_data["seasonal"]
    if "dates" in normalized_data:
        filtered_data["dates"] = normalized_data["dates"]

    # Handle full-resolution data for evaluation
    if "currents_u_full" in data and data["currents_u_full"] is not None:
        u_full = data["currents_u_full"]
        v_full = data["currents_v_full"]

        u_full_norm = (u_full - stats["currents_u"]["mean"]) / stats["currents_u"]["std"]
        v_full_norm = (v_full - stats["currents_v"]["mean"]) / stats["currents_v"]["std"]

        u_full_norm = np.where(np.isfinite(u_full), u_full_norm, np.nan)
        v_full_norm = np.where(np.isfinite(v_full), v_full_norm, np.nan)

        filtered_data["currents_u_full"] = u_full_norm
        filtered_data["currents_v_full"] = v_full_norm
        filtered_data["full_grid_size"] = data.get("full_grid_size", None)
        filtered_data["downsampled_grid_size"] = data.get("downsampled_grid_size", None)
        filtered_data["uniform_step"] = data.get("uniform_step", None)

    return filtered_data, stats
