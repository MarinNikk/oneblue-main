"""Utility functions for CLI commands."""

import os
import random

import numpy as np
import torch
from rich.console import Console


console = Console()


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for CUDA
    os.environ["PYTHONHASHSEED"] = str(seed)

    console.print(f"[green]🌱 Random seed set to {seed} for reproducibility[/green]")


def slice_data(data, start_idx, end_idx):
    """Slice a data dictionary by time indices.
    Handles both numpy arrays and lists.
    """
    sliced = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            # For numpy arrays with time dimension
            if len(val.shape) > 0 and val.shape[0] > 1:
                sliced[key] = val[start_idx:end_idx]
            else:
                sliced[key] = val
        elif isinstance(val, list):
            # For lists (like dates)
            sliced[key] = val[start_idx:end_idx]
        else:
            # For scalars or other types
            sliced[key] = val
    return sliced
