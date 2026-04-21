"""Input/output utilities for RBF-LSTM pipeline."""

import os
import pickle
from typing import Any

import pandas as pd
import xarray as xr


def save_pickle(obj: Any, filepath: str) -> None:
    """Save object to pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


def load_dataframe(filepath: str) -> pd.DataFrame:
    """Load DataFrame from CSV file."""
    return pd.read_csv(filepath)


def load_ocean_data(data_dir: str, uo_file: str, vo_file: str) -> tuple[xr.Dataset, xr.Dataset]:
    """Load ocean current datasets."""
    uo_path = os.path.join(data_dir, uo_file)
    vo_path = os.path.join(data_dir, vo_file)

    if not os.path.exists(uo_path):
        raise FileNotFoundError(f"UO dataset not found at {uo_path}")
    if not os.path.exists(vo_path):
        raise FileNotFoundError(f"VO dataset not found at {vo_path}")

    uo = xr.open_dataset(uo_path)
    vo = xr.open_dataset(vo_path)

    return uo, vo


def save_rbf_data(
    processed_dir: str,
    split: str,
    time_to_coeff: dict,
    time_to_metrics: dict,
    int_to_data: dict,
    rbf_network: Any = None,
) -> None:
    """Save RBF processing results for a data split."""
    split_dir = os.path.join(processed_dir, split)

    save_pickle(time_to_coeff, os.path.join(split_dir, "time_to_coeff.pkl"))
    save_pickle(time_to_metrics, os.path.join(split_dir, "time_to_metrics.pkl"))
    save_pickle(int_to_data, os.path.join(split_dir, "int_to_data.pkl"))

    # Save RBF network only once (usually with train split)
    if rbf_network is not None:
        save_pickle(rbf_network, os.path.join(processed_dir, "rbf.pkl"))


def load_rbf_data(
    processed_dir: str, split: str, load_rbf_network: bool = False
) -> tuple[dict, dict, dict, Any]:
    """Load RBF processing results for a data split."""
    split_dir = os.path.join(processed_dir, split)

    time_to_coeff = load_pickle(os.path.join(split_dir, "time_to_coeff.pkl"))
    time_to_metrics = load_pickle(os.path.join(split_dir, "time_to_metrics.pkl"))
    int_to_data = load_pickle(os.path.join(split_dir, "int_to_data.pkl"))

    rbf_network = None
    if load_rbf_network:
        rbf_path = os.path.join(processed_dir, "rbf.pkl")
        if os.path.exists(rbf_path):
            rbf_network = load_pickle(rbf_path)

    return time_to_coeff, time_to_metrics, int_to_data, rbf_network
