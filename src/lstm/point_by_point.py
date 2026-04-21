"""LSTM model that processes each spatial point independently (good approach)."""

import numpy as np
import torch
import torch.nn as nn

from .data_utils import apply_ocean_mask


class PointByPointLSTM(nn.Module):
    """
    LSTM that processes each spatial point independently.

    This preserves spatial structure by applying the same LSTM to each ocean
    location separately. Much better performance than averaging approach.
    """

    def __init__(
        self,
        input_channels=2,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        prediction_horizon=1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon

        # LSTM that processes each spatial location independently
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Output projection for all prediction horizons
        self.output_proj = nn.Linear(hidden_size, input_channels * prediction_horizon)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)

    def forward(self, x):
        """
        Forward pass that preserves spatial structure.

        Args:
            x: Input tensor [batch_size, seq_len, channels, n_ocean_points]

        Returns:
            Output tensor [batch_size, prediction_horizon, channels, n_ocean_points]
        """
        batch_size, seq_len, channels, n_ocean_points = x.shape

        # Reshape to process each spatial location independently
        # [batch, seq_len, channels, n_ocean_points] -> [batch * n_ocean_points, seq_len, channels]
        x_reshaped = x.permute(0, 3, 1, 2)  # [batch, n_ocean_points, seq_len, channels]
        x_flat = x_reshaped.reshape(batch_size * n_ocean_points, seq_len, channels)

        # Apply LSTM to each spatial location (shared parameters)
        lstm_out, _ = self.lstm(x_flat)  # [batch * n_ocean_points, seq_len, hidden_size]

        # Take last timestep
        last_out = lstm_out[:, -1, :]  # [batch * n_ocean_points, hidden_size]

        # Project to output (all prediction horizons)
        output_flat = self.output_proj(
            last_out
        )  # [batch * n_ocean_points, channels * prediction_horizon]

        # Reshape back to spatial format with prediction horizon
        output = output_flat.reshape(
            batch_size, n_ocean_points, self.prediction_horizon, channels
        )
        output = output.permute(
            0, 2, 3, 1
        )  # [batch, prediction_horizon, channels, n_ocean_points]

        return output

    @staticmethod
    def prepare_sequences_temporal_split(
        uo_data,
        vo_data,
        ocean_mask,
        sequence_length=7,
        prediction_horizon=1,
        train_fraction=0.8,
    ):
        """
        Prepare sequences for PointByPointLSTM with proper temporal split.

        This ensures no data leakage by:
        1. First doing temporal split
        2. Creating sequences only within each split
        3. Returning separate train/test sequences

        Args:
            uo_data: Eastward current data [time, lat, lon]
            vo_data: Northward current data [time, lat, lon]
            ocean_mask: Boolean mask for ocean points [lat, lon]
            sequence_length: Number of input timesteps
            prediction_horizon: Number of days to predict ahead
            train_fraction: Fraction of data for training

        Returns:
            train_data: (sequences, targets) for training
            test_data: (sequences, targets) for testing
            ocean_indices: Indices of ocean points in the mask
        """
        # Stack data
        data = np.stack([uo_data, vo_data], axis=1)  # [time, 2, lat, lon]

        # Temporal split first
        n_timesteps = data.shape[0]
        split_idx = int(train_fraction * n_timesteps)
        train_data = data[:split_idx]  # [train_time, 2, lat, lon]
        test_data = data[split_idx:]  # [test_time, 2, lat, lon]

        # Apply mask to both splits
        train_masked, ocean_indices = apply_ocean_mask(train_data, ocean_mask)
        test_masked, _ = apply_ocean_mask(test_data, ocean_mask)

        # Create sequences within each split
        def create_sequences_from_data(masked_data, split_name):
            sequences, targets = [], []
            # Need enough data for input sequence + all prediction horizons
            max_start = len(masked_data) - sequence_length - prediction_horizon + 1
            for i in range(max_start):
                x = masked_data[i : i + sequence_length]  # [seq_len, 2, n_ocean_points]

                # Collect targets for all prediction horizons
                y_list = []
                for h in range(1, prediction_horizon + 1):
                    y_h = masked_data[
                        i + sequence_length + h - 1
                    ]  # [2, n_ocean_points]
                    y_list.append(y_h)
                y = np.stack(y_list, axis=0)  # [prediction_horizon, 2, n_ocean_points]

                if not (np.isnan(x).any() or np.isnan(y).any()):
                    sequences.append(x)
                    targets.append(y)

            n_ocean_points = ocean_mask.sum()
            if len(sequences) > 0:
                sequences = np.array(
                    sequences
                )  # [n_sequences, seq_len, 2, n_ocean_points]
                targets = np.array(
                    targets
                )  # [n_sequences, prediction_horizon, 2, n_ocean_points]
            else:
                sequences = np.array([]).reshape(
                    0, sequence_length, 2, n_ocean_points
                )
                targets = np.array([]).reshape(
                    0, prediction_horizon, 2, n_ocean_points
                )

            return sequences, targets

        train_sequences, train_targets = create_sequences_from_data(
            train_masked, "Train"
        )
        test_sequences, test_targets = create_sequences_from_data(test_masked, "Test")

        return (
            (train_sequences, train_targets),
            (test_sequences, test_targets),
            ocean_indices,
        )
