"""LSTM sequence creation and dataset utilities (original functions from notebook)."""

import numpy as np
import torch
from torch.utils.data import Dataset


def create_sequences(
    data: dict, int_to_data: dict, seq_length: int = 7
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sequences for LSTM training (original function from notebook)."""
    # potential bug
    x, y = [], []
    for i in range(len(data) - seq_length):
        xs = [data[int_to_data[i + j]] for j in range(seq_length)]
        x.append(xs)
        y.append(data[int_to_data[i + seq_length]])
    return torch.from_numpy(np.array(x)).float(), torch.from_numpy(np.array(y)).float()


class SeaCurrentDataset(Dataset):
    """Dataset class for sea current data (original class from notebook)."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
