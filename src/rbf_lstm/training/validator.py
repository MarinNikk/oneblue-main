"""Validation utilities (original validation logic from notebook)."""

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score


def validate_overfitting_success(
    train_time_to_coeff: dict,
    int_to_data_train: dict,
    seq_length: int,
    y_pred_batch: torch.Tensor,
    normalized: bool = False,
    mean: float = None,
    std: float = None,
) -> None:
    print("Does overfitting get sucess? Are pred wieight the same as ground true wight?")
    for i in range(y_pred_batch.shape[0]):
        y_pred = y_pred_batch[i]
        if normalized:
            y_pred = y_pred * std + mean
        print(
            np.allclose(
                y_pred.detach().cpu().numpy(),
                train_time_to_coeff[int_to_data_train[i + seq_length]],
                atol=1e-1,
            )
        )


def validate_data_shapes(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    train_df: pd.DataFrame,
    int_to_data_train: dict,
    rbf: Any,
    seq_length: int,
) -> None:
    """Validate data shapes and consistency (validation logic from notebook)."""
    print("Validating data shapes and consistency...")

    print(f"X_tensor shape: {X_tensor.shape}")
    print(f"y_tensor shape: {y_tensor.shape}")

    # Test RBF activation computation (from notebook validation)
    lat_lon = train_df[train_df["time"] == int_to_data_train[seq_length]][
        ["longitude", "latitude"]
    ].to_numpy()
    X = rbf._compute_rbf_activations(lat_lon, rbf.cluster_centers)
    print(f"RBF activations shape: {X.shape}")

    print("Data validation complete!")


def check_r2_consistency(
    train_df: pd.DataFrame,
    int_to_data_train: dict,
    train_time_to_coeff: dict,
    train_time_to_metrics: dict,
    rbf: Any,
    test_index: int = 7,
) -> None:
    """Check R2 consistency (original validation from notebook)."""
    lat_lon = train_df[train_df["time"] == int_to_data_train[test_index]][
        ["longitude", "latitude"]
    ].to_numpy()
    X = rbf._compute_rbf_activations(lat_lon, rbf.cluster_centers)

    first_batch_true = train_time_to_coeff[int_to_data_train[test_index]]
    W_true = first_batch_true[: 2 * rbf.n_clusters].reshape((-1, 200))
    B_true = first_batch_true[2 * rbf.n_clusters :]
    Y_pred_tru = (X @ W_true.T) + B_true

    r2_values = r2_score(
        train_df[train_df["time"] == int_to_data_train[test_index]][
            ["uo_detided", "vo_detided"]
        ].to_numpy(),
        Y_pred_tru,
    )

    print(
        "check is r2 is the same",
        r2_values,
        train_time_to_metrics[int_to_data_train[test_index]][1],
    )
