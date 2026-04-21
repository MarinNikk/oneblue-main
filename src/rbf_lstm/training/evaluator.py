"""Evaluation utilities for RBF-LSTM training."""

from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from rbf_lstm.utils.tracker import AbstractTracker

from ..utils.metrics_history import EpochMetrics


class ModelEvaluator:
    """
    Handles model evaluation during training.

    Separates evaluation logic from training loop for better maintainability.
    """

    def __init__(
        self,
        tracker: AbstractTracker,
        rbf: Any,
        loader: dict[str, DataLoader],
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        datasets: dict[str, dict[str, Any]],
        seq_length: int = 7,
    ):
        """
        Initialize evaluator.

        Args:
            tracker: AbstractTracker for metrices tracking
            rbf: RBF network for computing activations
            train_df: Training DataFrame
            int_to_data_train: Time mapping for training data
            train_time_to_metrics: RBF metrics for training data
            seq_length: Sequence length for predictions
        """
        self.tracker = tracker
        self.rbf = rbf
        self.loader = loader
        self.model = model
        self.loss_fn = (loss_fn,)
        if isinstance(self.loss_fn, tuple):
            self.loss_fn = self.loss_fn[0]
        self.datasets = datasets
        self.seq_length = seq_length
        self.n_clusters = rbf.n_clusters
        self.W_size = 2 * self.n_clusters

        # Pre-compute RBF activations for evaluation (cached)
        lat_lon = self.datasets["test"]["df"][
            self.datasets["test"]["df"]["time"] == self.datasets["test"]["int_to_data"][7]
        ][["longitude", "latitude"]].to_numpy()
        self.X_rbf = rbf._compute_rbf_activations(lat_lon, rbf.cluster_centers)
        self.X_rbf_torch = torch.from_numpy(self.X_rbf).float()

        # Pre-compute all time_to_uv arrays and convert to tensor
        self.time_to_uv_numpy = {}
        self.time_to_uv_torch = {}
        for split in self.datasets:
            self.time_to_uv_numpy[split] = {
                t: self.datasets[split]["df"][self.datasets[split]["df"]["time"] == t][
                    ["uo_detided", "vo_detided"]
                ].to_numpy()
                for t in self.datasets[split]["df"]["time"].unique()
            }

            # Convert to tensor once and store on same device
            self.time_to_uv_torch[split] = {
                t: torch.from_numpy(uv).float() for t, uv in self.time_to_uv_numpy[split].items()
            }

        # Pre-compute ground truth R2 values
        self.time_to_r2 = {}
        for split in self.datasets:
            self.time_to_r2[split] = {
                t: metrics[1] for t, metrics in self.datasets[split]["time_to_metrics"].items()
            }

    def count_r2(self, y_true, y_pred):
        """
        Computes the coefficient of determination (R²) for a set of predictions.

        Args:
            y_true (torch.Tensor): Tensor of true target values, shape (batch_size, n_features).
            y_pred (torch.Tensor): Tensor of predicted values, same shape as y_true.

        Returns:
            torch.Tensor: Tensor containing R² for each batch.

        Formula:
            R² = 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))

        Notes:
            - For a perfect prediction, R² = 1.
            - R² can be negative if the model performs worse than predicting the mean.
        """
        from sklearn.metrics import r2_score

        y_true_flat = y_true.reshape(-1, 2).numpy()
        y_pred_flat = y_pred.reshape(-1, 2).numpy()

        # calculate R² for each column separately
        r2_col0 = r2_score(y_true_flat[:, 0], y_pred_flat[:, 0])
        r2_col1 = r2_score(y_true_flat[:, 1], y_pred_flat[:, 1])

        r2_avg = (r2_col0 + r2_col1) / 2
        return r2_avg

    def evaluate_uv_metrics(self, pred, y_batch, split, batch_size, i_batch, normalized, mean, std):
        # Prepare for U/V evaluation
        y_pred = pred * std + mean if normalized else pred

        y_batch = y_batch.detach().cpu().numpy()  # shape (batch, features)
        pred = pred.detach().cpu().numpy()  # same shape

        Y_pred_tru_list = []
        for coef in y_batch:
            W_true = coef[: 2 * self.n_clusters].reshape(self.n_clusters, -1)
            B_true = coef[2 * self.n_clusters :]
            Y_pred_tru = (self.X_rbf @ W_true) + B_true  # BUG without transpose ???
            Y_pred_tru_list.append(Y_pred_tru)

        Y_pred_UV = []
        for coef in pred:
            W_true = coef[: 2 * self.n_clusters].reshape(self.n_clusters, -1)
            B_true = coef[2 * self.n_clusters :]
            Y_pred_tru = (self.X_rbf @ W_true) + B_true  # BUG without transpose ???
            Y_pred_UV.append(Y_pred_tru)

        y_true = np.vstack(Y_pred_tru_list)  # shape (num_samples, 2)
        y_pred = np.vstack(Y_pred_UV)  # same shape

        return y_true.reshape(-1).tolist(), y_pred.reshape(-1).tolist()

    def evaluate_split(
        self, split: str, epoch: int, normalized: bool, mean: int | None, std: int | None
    ) -> EpochMetrics:
        metrics = EpochMetrics()
        loader = self.loader[split]

        with torch.no_grad():
            test_pred_y, test_true_y = [], []
            test_true_uv, test_pred_uv = [], []
            loss = 0
            n_batches_test = 0
            for i_batch, (X_batch, y_batch) in enumerate(loader):
                device = next(self.model.parameters()).device
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = self.model(X_batch)
                batch_size = len(X_batch)

                loss += self.loss_fn(y_pred, y_batch).item()
                n_batches_test += 1

                test_true_y.extend(y_batch.detach().cpu().numpy().reshape(-1).tolist())
                test_pred_y.extend(y_pred.detach().cpu().numpy().reshape(-1).tolist())

                Y_pred_UV, Y_pred_tru_list = self.evaluate_uv_metrics(
                    y_pred, y_batch, split, batch_size, i_batch, normalized, mean, std
                )

                y_true = np.vstack(Y_pred_tru_list)  # shape (num_samples, 2)
                y_pred = np.vstack(Y_pred_UV)  # same shape

                test_true_uv.extend(y_true.reshape(-1).tolist())
                test_pred_uv.extend(y_pred.reshape(-1).tolist())

        loss /= n_batches_test
        metrics.loss = loss

        test_r2 = r2_score(test_true_y, test_pred_y)
        metrics.r2_lstm = test_r2

        r2_uv = r2_score(test_true_uv, test_pred_uv)
        metrics.r2_uv = r2_uv

        for name, value in asdict(metrics).items():
            self.tracker.log_metric(f"{split}_{name}", value, step=epoch)

        return metrics
