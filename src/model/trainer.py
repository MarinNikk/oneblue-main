import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from src.evaluations.evaluation_torch import (
    calculate_mae_ms,
    calculate_r2,
    calculate_r2_full_grid,
    calculate_rmse_ms,
)


console = Console()


def train_model(
    model, train_loader, val_loader, config, stats, full_grid_size=None, uniform_step=None
):
    """Training loop with Rich progress bars.

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration object
        stats: Normalization statistics
        full_grid_size: Optional tuple (full_H, full_W) for full-grid R² evaluation
        uniform_step: The uniform step used for downsampling (for proper upsampling)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Ensure reproducible weight initialization
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_r2 = -np.inf
    early_stop_count = 0
    best_mae_saved = 0
    best_rmse_saved = 0
    best_r2_full_saved = 0

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        train_r2 = 0
        train_mae_u, train_mae_v, train_mae_mag = 0, 0, 0
        train_rmse_u, train_rmse_v, train_rmse_mag = 0, 0, 0

        with Progress() as progress:
            task = progress.add_task(
                f"[green]Epoch {epoch + 1}/{config.epochs} - Training",
                total=len(train_loader),
            )

            for _batch_idx, batch in enumerate(train_loader):
                # Handle both 3-tuple and 4-tuple (with full-res targets)
                if len(batch) == 4:
                    x, seasonal, y, _y_full = batch
                else:
                    x, seasonal, y = batch
                x, seasonal, y = x.to(device), seasonal.to(device), y.to(device)

                optimizer.zero_grad()
                pred = model(x, seasonal)  # [batch, pred_horizon, 2, H, W]

                # Compute loss across all timesteps
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # For metrics, use the final prediction timestep (most important)
                final_pred = pred[:, -1]  # [batch, 2, H, W]
                final_y = y[:, -1]  # [batch, 2, H, W]

                train_r2 += calculate_r2(final_y, final_pred)

                # Calculate physical metrics (m/s) for final timestep
                mae_u, mae_v, mae_mag = calculate_mae_ms(
                    final_y.cpu(), final_pred.cpu().detach(), stats
                )
                rmse_u, rmse_v, rmse_mag = calculate_rmse_ms(
                    final_y.cpu(), final_pred.cpu().detach(), stats
                )

                train_mae_u += mae_u
                train_mae_v += mae_v
                train_mae_mag += mae_mag
                train_rmse_u += rmse_u
                train_rmse_v += rmse_v
                train_rmse_mag += rmse_mag

                progress.advance(task)

        # Validation
        model.eval()
        val_loss = 0
        val_r2 = 0
        val_r2_full = 0
        val_mae_u, val_mae_v, val_mae_mag = 0, 0, 0
        val_rmse_u, val_rmse_v, val_rmse_mag = 0, 0, 0
        has_full_grid = False

        with torch.no_grad():
            for batch in val_loader:
                # Handle both 3-tuple and 4-tuple (with full-res targets)
                if len(batch) == 4:
                    x, seasonal, y, y_full = batch
                    has_full_grid = full_grid_size is not None
                else:
                    x, seasonal, y = batch
                    y_full = None

                x, seasonal, y = x.to(device), seasonal.to(device), y.to(device)
                pred = model(x, seasonal)  # [batch, pred_horizon, 2, H, W]
                val_loss += criterion(pred, y).item()

                # For metrics, use the final prediction timestep
                final_pred = pred[:, -1]  # [batch, 2, H, W]
                final_y = y[:, -1]  # [batch, 2, H, W]

                val_r2 += calculate_r2(final_y, final_pred)

                # Calculate full-grid R² if available
                if has_full_grid and y_full is not None:
                    final_y_full = y_full[:, -1].to(device)  # [batch, 2, full_H, full_W]
                    val_r2_full += calculate_r2_full_grid(
                        final_pred, final_y_full, full_grid_size, uniform_step
                    )

                # Calculate physical metrics (m/s) for final timestep
                mae_u, mae_v, mae_mag = calculate_mae_ms(final_y.cpu(), final_pred.cpu(), stats)
                rmse_u, rmse_v, rmse_mag = calculate_rmse_ms(final_y.cpu(), final_pred.cpu(), stats)

                val_mae_u += mae_u
                val_mae_v += mae_v
                val_mae_mag += mae_mag
                val_rmse_u += rmse_u
                val_rmse_v += rmse_v
                val_rmse_mag += rmse_mag

        # Calculate averages
        n_train = len(train_loader)
        n_val = len(val_loader)

        train_loss /= n_train
        train_r2 /= n_train
        train_mae_u /= n_train
        train_mae_v /= n_train
        train_mae_mag /= n_train
        train_rmse_u /= n_train
        train_rmse_v /= n_train
        train_rmse_mag /= n_train

        val_loss /= n_val
        val_r2 /= n_val
        val_r2_full /= n_val if has_full_grid else 1
        val_mae_u /= n_val
        val_mae_v /= n_val
        val_mae_mag /= n_val
        val_rmse_u /= n_val
        val_rmse_v /= n_val
        val_rmse_mag /= n_val

        scheduler.step(val_loss)

        # Print results with physical units
        table = Table(
            show_header=True,
            header_style="bold magenta",
            title=f"Epoch {epoch + 1} Results",
        )
        table.add_column("Metric", style="cyan", width=12)
        table.add_column("Training", justify="right", width=15)
        table.add_column("Validation", justify="right", width=15)
        table.add_column("Unit", style="dim", width=8)

        table.add_row("Loss", f"{train_loss:.6f}", f"{val_loss:.6f}", "")
        table.add_row("R²", f"{train_r2:.4f}", f"{val_r2:.4f}", "")
        if has_full_grid:
            table.add_row("R² (full)", "-", f"{val_r2_full:.4f}", "")
        table.add_row("MAE U", f"{train_mae_u:.4f}", f"{val_mae_u:.4f}", "m/s")
        table.add_row("MAE V", f"{train_mae_v:.4f}", f"{val_mae_v:.4f}", "m/s")
        table.add_row("MAE |U|", f"{train_mae_mag:.4f}", f"{val_mae_mag:.4f}", "m/s")
        table.add_row("RMSE U", f"{train_rmse_u:.4f}", f"{val_rmse_u:.4f}", "m/s")
        table.add_row("RMSE V", f"{train_rmse_v:.4f}", f"{val_rmse_v:.4f}", "m/s")
        table.add_row("RMSE |U|", f"{train_rmse_mag:.4f}", f"{val_rmse_mag:.4f}", "m/s")
        table.add_row("LR", f"{optimizer.param_groups[0]['lr']:.2e}", "", "")

        console.print(table)

        # Early stopping and model saving
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_mae_saved = val_mae_mag
            best_rmse_saved = val_rmse_mag
            best_r2_full_saved = val_r2_full
            save_dict = {
                "model_state_dict": model.state_dict(),
                "stats": stats,
                "config": config.__dict__,
                "epoch": epoch,
                "val_r2": val_r2,
                "val_mae_magnitude": val_mae_mag,
                "val_rmse_magnitude": val_rmse_mag,
            }
            if has_full_grid:
                save_dict["val_r2_full_grid"] = val_r2_full
                save_dict["full_grid_size"] = full_grid_size
            torch.save(save_dict, "best_model.pth")
            early_stop_count = 0
            r2_msg = f"[green]🎯 New best R²: {best_r2:.4f}"
            if has_full_grid:
                r2_msg += f" | R² (full grid): {val_r2_full:.4f}"
            r2_msg += f" | MAE: {val_mae_mag:.4f} m/s | RMSE: {val_rmse_mag:.4f} m/s[/green]"
            console.print(r2_msg)
        else:
            early_stop_count += 1

        if early_stop_count >= config.early_stopping_patience:
            console.print(f"[yellow]⏹️  Early stopping at epoch {epoch + 1}[/yellow]")
            break

        console.print()

    return best_r2, best_mae_saved, best_rmse_saved, best_r2_full_saved
