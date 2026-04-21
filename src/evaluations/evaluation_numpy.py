import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(
    u_pred: np.ndarray,
    v_pred: np.ndarray,
    u_true: np.ndarray,
    v_true: np.ndarray,
) -> dict[str, float]:
    """Calculate metrics using sklearn."""
    # Flatten
    u_pred_flat = u_pred.flatten()
    v_pred_flat = v_pred.flatten()
    u_true_flat = u_true.flatten()
    v_true_flat = v_true.flatten()

    # Component metrics
    mae_u = mean_absolute_error(u_true_flat, u_pred_flat)
    mse_u = mean_squared_error(u_true_flat, u_pred_flat)
    rmse_u = np.sqrt(mse_u)
    r2_u = r2_score(u_true_flat, u_pred_flat)

    mae_v = mean_absolute_error(v_true_flat, v_pred_flat)
    mse_v = mean_squared_error(v_true_flat, v_pred_flat)
    rmse_v = np.sqrt(mse_v)
    r2_v = r2_score(v_true_flat, v_pred_flat)

    # Magnitude metrics
    magnitude_pred = np.sqrt(u_pred_flat**2 + v_pred_flat**2)
    magnitude_true = np.sqrt(u_true_flat**2 + v_true_flat**2)

    mae_magnitude = mean_absolute_error(magnitude_true, magnitude_pred)
    mse_magnitude = mean_squared_error(magnitude_true, magnitude_pred)
    rmse_magnitude = np.sqrt(mse_magnitude)

    return {
        "rmse_u": float(rmse_u),
        "mae_u": float(mae_u),
        "r2_u": float(r2_u),
        "rmse_v": float(rmse_v),
        "mae_v": float(mae_v),
        "r2_v": float(r2_v),
        "rmse_magnitude": float(rmse_magnitude),
        "mae_magnitude": float(mae_magnitude),
        "n_predictions": len(u_pred_flat),
    }
