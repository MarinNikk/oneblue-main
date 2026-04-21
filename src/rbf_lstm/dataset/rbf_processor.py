"""RBF parameter generation and processing (original functions from notebook)."""

import multiprocessing as mp
from functools import partial
from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from model.RBFNetwork import RBFNetwork


def init_rbf(train_df: pd.DataFrame, int_to_data_train: dict, n_clusters: int = 200) -> Any:
    """Initialize RBF network for first timestep (original function from notebook)."""
    # Initialize and learn RBF for first timestep, use learned centers and scaler for other timesteps
    mask_time_0 = train_df["time"] == int_to_data_train[0]
    time_0_rows = train_df[mask_time_0]
    x_day_0 = time_0_rows[["longitude", "latitude"]].to_numpy()
    y_day_0 = time_0_rows[["uo_detided", "vo_detided"]].to_numpy()
    rbf = RBFNetwork(spred=None, n_clusters=n_clusters)  # Make n_clusters a keyword argument too
    _, _ = rbf.train(x_day_0, y_day_0)
    return rbf


def count_rbf_params(
    df: pd.DataFrame, rbf_template: Any, x: tuple[int, Any]
) -> tuple[Any, np.ndarray, float, float]:
    """Count RBF parameters for a single timestep (original function from notebook)."""
    time = x[1]

    mask_df = df[df["time"] == time]
    rbf_x = mask_df[["longitude", "latitude"]].to_numpy()
    rbf_y = mask_df[["uo_detided", "vo_detided"]].to_numpy()

    local_rbf = type(rbf_template)()  # Create new instance
    local_rbf.__dict__.update(rbf_template.__dict__)  # Copy attributes

    RMSE, R_sqer = local_rbf.train_for_fix_cluster_and_scaler(rbf_x, rbf_y)
    return (
        time,
        np.hstack((local_rbf.model.coef_.flatten(), local_rbf.model.intercept_)),
        RMSE,
        R_sqer,
    )


def process_time_groups(
    time_items: list, df: pd.DataFrame, rbf: Any, process_name: str = ""
) -> tuple[dict, dict, list]:
    """Process RBF parameters for multiple timesteps (original function from notebook)."""
    print(f"Processing {len(time_items)} times for {process_name}...")

    # Create partial function with fixed arguments
    train_func = partial(count_rbf_params, df, rbf)

    with Pool(processes=mp.cpu_count() - 2) as pool:
        print(f"Pool created with {mp.cpu_count() - 2} processes")

        results = list(
            tqdm(pool.imap(train_func, time_items), total=len(time_items), desc=process_name)
        )

    time_to_coeff = {time: pars for time, pars, _, _ in results}
    time_to_metrics = {time: (RMSE, R_sqer) for time, _, RMSE, R_sqer in results}

    return time_to_coeff, time_to_metrics, results
