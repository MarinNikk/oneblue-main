"""
7-Day Mean Baseline Model for Ocean Current Prediction.

This module implements a simple persistence baseline that predicts
tomorrow's current as the mean of the past 7 days. This serves as
the minimum performance threshold - any ML model should beat this.

Physics intuition: Ocean currents have inertia and momentum, so
"tomorrow will look like the recent average" is a reasonable null hypothesis.
"""

import numpy as np

from src.evaluations.evaluation_numpy import calculate_metrics


class SevenDayMeanBaseline:
    """
    Predicts next-day ocean currents as the 7-day rolling mean.

    This is a trivial baseline with:
    - Zero parameters to train
    - No learning
    - Pure temporal smoothing

    We treat U and V components independently since they represent
    orthogonal current directions (eastward and northward).
    """

    def __init__(self, window_size: int = 7):
        """
        Initialize the baseline model.

        Args:
            window_size: Number of past days to average (default: 7)
        """
        self.window_size = window_size
        self.name = f"{window_size}-Day Mean Baseline"

    def predict(self, u_history: np.ndarray, v_history: np.ndarray) -> tuple[float, float]:
        """
        Predict next-day current from historical sequence.

        Args:
            u_history: Array of U (eastward) currents, shape (window_size,) or (window_size, H, W)
            v_history: Array of V (northward) currents, shape (window_size,) or (window_size, H, W)

        Returns:
            Tuple of (u_pred, v_pred) - predicted currents for next day
        """
        if len(u_history) != self.window_size:
            raise ValueError(f"Expected {self.window_size} timesteps, got {len(u_history)}")

        # Simple mean along time axis (axis 0), ignoring NaN values
        u_pred = np.nanmean(u_history, axis=0)
        v_pred = np.nanmean(v_history, axis=0)

        return u_pred, v_pred

    def evaluate_sequence(
        self,
        u_data: np.ndarray,
        v_data: np.ndarray,
        start_idx: int = 7,
    ) -> dict[str, float]:
        """
        Evaluate model on a time series using rolling predictions.

        Args:
            u_data: Full time series of U currents
            v_data: Full time series of V currents
            start_idx: Index to start predictions (default: 7 for 7-day window)

        Returns:
            Dictionary with metrics: {
                'rmse_u', 'rmse_v', 'rmse_magnitude',
                'mae_u', 'mae_v', 'mae_magnitude',
                'r2_u', 'r2_v'
            }

        Process:
            For each timestep t >= start_idx:
                1. Take u_data[t-7:t] as input window
                2. Predict u_data[t] as mean(u_data[t-7:t])
                3. Compare with actual u_data[t]
        """
        n_timesteps = len(u_data)

        # Storage for predictions
        u_predictions = []
        v_predictions = []
        u_targets = []
        v_targets = []

        # Rolling window predictions
        for t in range(start_idx, n_timesteps):
            # Extract window [t-7, t-6, ..., t-1]
            u_window = u_data[t - self.window_size : t]
            v_window = v_data[t - self.window_size : t]

            # Predict timestep t
            u_prediction, v_prediction = self.predict(u_window, v_window)

            # Store predictions and targets
            u_predictions.append(u_prediction)
            v_predictions.append(v_prediction)
            u_targets.append(u_data[t])
            v_targets.append(v_data[t])

        # Convert to arrays for metric calculation
        u_predictions = np.array(u_predictions)
        v_predictions = np.array(v_predictions)
        u_targets = np.array(u_targets)
        v_targets = np.array(v_targets)

        # Filter out NaN values before metrics calculation (land areas)
        u_pred_flat = u_predictions.flatten()
        v_pred_flat = v_predictions.flatten()
        u_true_flat = u_targets.flatten()
        v_true_flat = v_targets.flatten()

        valid_mask = (
            (~np.isnan(u_pred_flat))
            & (~np.isnan(v_pred_flat))
            & (~np.isnan(u_true_flat))
            & (~np.isnan(v_true_flat))
        )

        u_pred_clean = u_pred_flat[valid_mask].reshape(-1, 1)
        v_pred_clean = v_pred_flat[valid_mask].reshape(-1, 1)
        u_true_clean = u_true_flat[valid_mask].reshape(-1, 1)
        v_true_clean = v_true_flat[valid_mask].reshape(-1, 1)

        # Calculate metrics on clean ocean data only
        metrics = calculate_metrics(u_pred_clean, v_pred_clean, u_true_clean, v_true_clean)

        return metrics


def run_mean_baseline_experiment(
    u_data: np.ndarray,
    v_data: np.ndarray,
    window_size: int = 7,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, dict[str, float]]:
    """
    Run complete baseline experiment with train/val/test splits.

    Args:
        u_data: Full U current time series
        v_data: Full V current time series
        window_size: Size of averaging window (default: 7 days)
        split_ratios: (train, val, test) split proportions

    Returns:
        Dictionary with metrics for each split: {
            'train': {...},
            'val': {...},
            'test': {...}
        }
    """
    # Data splitting
    n_total = len(u_data)
    train_end = int(n_total * split_ratios[0])
    val_end = train_end + int(n_total * split_ratios[1])

    # Initialize model
    model = SevenDayMeanBaseline(window_size=window_size)

    # Evaluate on each split
    results = {}

    for split_name, (start, end) in [
        ("train", (0, train_end)),
        ("val", (train_end, val_end)),
        ("test", (val_end, n_total)),
    ]:
        u_split = u_data[start:end]
        v_split = v_data[start:end]

        metrics = model.evaluate_sequence(u_split, v_split, start_idx=window_size)
        results[split_name] = metrics

    return results
