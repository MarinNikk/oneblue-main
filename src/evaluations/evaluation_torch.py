import torch
import torch.nn.functional as F


def calculate_r2(y_true, y_pred):
    """Calculate R² score."""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return r2.item()


def upsample_predictions(pred, target_size):
    """Upsample predictions to full grid resolution using bilinear interpolation.

    Args:
        pred: Predictions tensor [batch, channels, H, W] or [batch, horizon, channels, H, W]
        target_size: Tuple (target_H, target_W)

    Returns:
        Upsampled predictions with shape [..., target_H, target_W]
    """

    # Handle 5D tensor [batch, horizon, channels, H, W]
    if pred.dim() == 5:
        batch, horizon, channels, H, W = pred.shape
        # Reshape to 4D for interpolation
        pred = pred.view(batch * horizon, channels, H, W)
        upsampled = F.interpolate(pred, size=target_size, mode="bilinear", align_corners=True)
        # Reshape back to 5D
        upsampled = upsampled.view(batch, horizon, channels, target_size[0], target_size[1])
    else:
        # 4D tensor [batch, channels, H, W]
        upsampled = F.interpolate(pred, size=target_size, mode="bilinear", align_corners=True)

    return upsampled


def calculate_r2_full_grid(y_pred, y_true_full, full_grid_size, uniform_step=None):
    """Calculate R² score on full-resolution grid.

    With uniform_step (aspect-ratio preserving downsampling), we can properly
    upsample predictions and compare against ALL points in the full grid.

    Without uniform_step, we fall back to comparing only at sampled points.

    Args:
        y_pred: Downsampled predictions [batch, channels, H, W]
        y_true_full: Full-resolution ground truth [batch, channels, full_H, full_W]
        full_grid_size: Tuple (full_H, full_W)
        uniform_step: The uniform step used for downsampling (enables proper upsampling)

    Returns:
        R² score on full grid
    """
    batch, channels, pred_H, pred_W = y_pred.shape
    full_H, full_W = full_grid_size

    if uniform_step is not None:
        # With uniform step, we can properly upsample predictions to full grid
        # The downsampled grid covers: pred_H * step × pred_W * step of the full grid
        covered_H = min(pred_H * uniform_step, full_H)
        covered_W = min(pred_W * uniform_step, full_W)

        # Upsample predictions to match the covered area of the full grid
        y_pred_upsampled = F.interpolate(
            y_pred, size=(covered_H, covered_W), mode="bilinear", align_corners=True
        )

        # Extract the corresponding region from full ground truth
        y_true_region = y_true_full[:, :, :covered_H, :covered_W]

        # Create mask for valid (non-zero) values
        valid_mask = y_true_region != 0.0

        if valid_mask.sum() == 0:
            return 0.0

        y_true_valid = y_true_region[valid_mask]
        y_pred_valid = y_pred_upsampled[valid_mask]
    else:
        # Fallback: compare only at sampled points (non-uniform step case)
        lat_step = full_H // pred_H
        lon_step = full_W // pred_W
        y_true_sampled = y_true_full[:, :, ::lat_step, ::lon_step][:, :, :pred_H, :pred_W]

        valid_mask = y_true_sampled != 0.0
        if valid_mask.sum() == 0:
            return 0.0

        y_true_valid = y_true_sampled[valid_mask]
        y_pred_valid = y_pred[valid_mask]

    ss_res = torch.sum((y_true_valid - y_pred_valid) ** 2)
    ss_tot = torch.sum((y_true_valid - torch.mean(y_true_valid)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return r2.item()


def calculate_mae_ms(y_true, y_pred, stats):
    """Calculate MAE in m/s (denormalized)."""
    u_stats = stats["currents_u"]
    v_stats = stats["currents_v"]

    # Split U and V components
    u_true = y_true[:, 0] * u_stats["std"] + u_stats["mean"]
    v_true = y_true[:, 1] * v_stats["std"] + v_stats["mean"]
    u_pred = y_pred[:, 0] * u_stats["std"] + u_stats["mean"]
    v_pred = y_pred[:, 1] * v_stats["std"] + v_stats["mean"]

    # Calculate MAE for each component
    mae_u = torch.mean(torch.abs(u_true - u_pred)).item()
    mae_v = torch.mean(torch.abs(v_true - v_pred)).item()

    # Calculate magnitude MAE
    true_magnitude = torch.sqrt(u_true**2 + v_true**2)
    pred_magnitude = torch.sqrt(u_pred**2 + v_pred**2)
    mae_magnitude = torch.mean(torch.abs(true_magnitude - pred_magnitude)).item()

    return mae_u, mae_v, mae_magnitude


def calculate_rmse_ms(y_true, y_pred, stats):
    """Calculate RMSE in m/s (denormalized)."""
    u_stats = stats["currents_u"]
    v_stats = stats["currents_v"]

    # Split U and V components
    u_true = y_true[:, 0] * u_stats["std"] + u_stats["mean"]
    v_true = y_true[:, 1] * v_stats["std"] + v_stats["mean"]
    u_pred = y_pred[:, 0] * u_stats["std"] + u_stats["mean"]
    v_pred = y_pred[:, 1] * v_stats["std"] + v_stats["mean"]

    # Calculate RMSE for each component
    rmse_u = torch.sqrt(torch.mean((u_true - u_pred) ** 2)).item()
    rmse_v = torch.sqrt(torch.mean((v_true - v_pred) ** 2)).item()

    # Calculate magnitude RMSE
    true_magnitude = torch.sqrt(u_true**2 + v_true**2)
    pred_magnitude = torch.sqrt(u_pred**2 + v_pred**2)
    rmse_magnitude = torch.sqrt(torch.mean((true_magnitude - pred_magnitude) ** 2)).item()

    return rmse_u, rmse_v, rmse_magnitude
