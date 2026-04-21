"""
XGBoost Baseline for Ocean Current Prediction.

Uses 7-day lag features to predict next-day currents.
Tests whether gradient boosting can capture temporal patterns
without explicit sequential modeling.
"""

import numpy as np
from xgboost import XGBRegressor

from src.evaluations.evaluation_numpy import calculate_metrics


class XGBoostBaseline:
    """
    XGBoost baseline using 7-day lag features.

    Treats spatial grid points independently - trains separate models
    for each (lat, lon) location.
    """

    def __init__(self, window_size: int = 7, n_estimators: int = 100):
        """
        Initialize XGBoost baseline.

        Args:
            window_size: Number of past days to use as features (default: 7)
            n_estimators: Number of boosting rounds (default: 100)
        """
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.name = f"XGBoost ({window_size}-day features)"

    def create_features(
        self, data: np.ndarray, start_idx: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create lag features for spatial mean prediction (fast baseline approach).

        Args:
            data: Time series, shape (T, H, W)
            start_idx: Start index for features (default: window_size)

        Returns:
            X: Features, shape (n_samples, window_size)
            y: Targets, shape (n_samples,)

        Uses spatial mean to reduce to a single time series prediction problem.
        """
        if start_idx is None:
            start_idx = self.window_size

        n_timesteps = data.shape[0]

        # Convert to spatial mean time series (ignoring NaN)
        spatial_means = np.nanmean(data, axis=(1, 2))  # Shape: (T,)

        X = []
        y = []

        for t in range(start_idx, n_timesteps):
            # Features: past window_size spatial means
            features = spatial_means[t - self.window_size : t]  # Shape: (window_size,)
            target = spatial_means[t]  # Scalar

            X.append(features)
            y.append(target)

        X = np.array(X)  # (n_samples, window_size)
        y = np.array(y)  # (n_samples,)

        return X, y

    def train(self, u_train: np.ndarray, v_train: np.ndarray, verbose: bool = True) -> None:
        """
        Train XGBoost models on training data.

        Args:
            u_train: U current training data, shape (T, H, W)
            v_train: V current training data, shape (T, H, W)
            verbose: Print training progress
        """
        if verbose:
            print("Creating features from training data...")

        # Create features (using spatial means for fast training)
        X_u, y_u = self.create_features(u_train)
        X_v, y_v = self.create_features(v_train)

        if verbose:
            print("Training XGBoost models...")
            print(f"  Samples: {X_u.shape[0]}")
            print(f"  Features per sample: {X_u.shape[1]}")
            print("  Target: spatial mean prediction")

        # Train one model for U, one for V (across all spatial points)
        # Use smaller, faster configuration for baseline
        self.model_u = XGBRegressor(
            n_estimators=min(50, self.n_estimators),  # Cap at 50 trees for speed
            max_depth=4,  # Reduce depth for faster training
            learning_rate=0.2,  # Higher learning rate to compensate
            random_state=42,
            n_jobs=-1,
            verbosity=1,  # Show progress
            subsample=0.8,  # Use 80% of samples for speed
            colsample_bytree=0.8,  # Use 80% of features for speed
        )

        self.model_v = XGBRegressor(
            n_estimators=min(50, self.n_estimators),
            max_depth=4,
            learning_rate=0.2,
            random_state=42,
            n_jobs=-1,
            verbosity=1,
            subsample=0.8,
            colsample_bytree=0.8,
        )

        if verbose:
            print("  Training U model...")
        self.model_u.fit(X_u, y_u)

        if verbose:
            print("  Training V model...")
        self.model_v.fit(X_v, y_v)

        if verbose:
            print("✓ Training complete")

    def evaluate(
        self, u_data: np.ndarray, v_data: np.ndarray, start_idx: int = None
    ) -> dict[str, float]:
        """
        Evaluate model on data using spatial mean predictions.

        Args:
            u_data: U current data, shape (T, H, W)
            v_data: V current data, shape (T, H, W)
            start_idx: Start index (default: window_size)

        Returns:
            Dictionary with metrics
        """
        if start_idx is None:
            start_idx = self.window_size

        # Create features (spatial means)
        X_u, y_u_true = self.create_features(u_data, start_idx)
        X_v, y_v_true = self.create_features(v_data, start_idx)

        # Predict spatial means
        y_u_pred_mean = self.model_u.predict(X_u)
        y_v_pred_mean = self.model_v.predict(X_v)

        # Reshape for calculate_metrics (expects at least 2D)
        y_u_pred = y_u_pred_mean.reshape(-1, 1)
        y_v_pred = y_v_pred_mean.reshape(-1, 1)
        y_u_true = y_u_true.reshape(-1, 1)
        y_v_true = y_v_true.reshape(-1, 1)

        # Calculate metrics on spatial means
        metrics = calculate_metrics(y_u_pred, y_v_pred, y_u_true, y_v_true)

        return metrics


def run_xgboost_baseline_experiment(
    u_data: np.ndarray,
    v_data: np.ndarray,
    window_size: int = 7,
    n_estimators: int = 100,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, dict[str, float]]:
    """
    Run XGBoost baseline experiment.

    Args:
        u_data: Full U current time series
        v_data: Full V current time series
        window_size: Lag window size
        n_estimators: Number of trees
        split_ratios: Train/val/test split

    Returns:
        Results dict with metrics for each split
    """
    # Split data
    n_total = len(u_data)
    train_end = int(n_total * split_ratios[0])
    val_end = train_end + int(n_total * split_ratios[1])

    u_train, v_train = u_data[:train_end], v_data[:train_end]
    u_val, v_val = u_data[train_end:val_end], v_data[train_end:val_end]
    u_test, v_test = u_data[val_end:], v_data[val_end:]

    # Initialize and train
    model = XGBoostBaseline(window_size=window_size, n_estimators=n_estimators)
    model.train(u_train, v_train, verbose=True)

    # Evaluate
    results = {
        "train": model.evaluate(u_train, v_train),
        "val": model.evaluate(u_val, v_val),
        "test": model.evaluate(u_test, v_test),
    }

    return results
