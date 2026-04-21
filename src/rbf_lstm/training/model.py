"""LSTM model definition for RBF parameter prediction."""

import torch
import torch.nn as nn


class RBFLSTMModel(nn.Module):
    """
    LSTM model for predicting RBF network parameters from time series data.

    This model takes sequences of RBF parameters and predicts the next timestep's
    parameters for ocean current modeling. The model architecture consists of:
    - Multi-layer LSTM for temporal sequence modeling
    - Fully connected layer for final parameter prediction
    - Optional dropout for regularization

    Args:
        input_size: Dimensionality of input features (RBF parameters + bias)
        hidden_size: Number of features in LSTM hidden state
        num_layers: Number of recurrent layers
        output_size: Dimensionality of output (RBF parameters + bias)
        dropout: Dropout probability for regularization (0.0 = no dropout)
        bidirectional: Whether to use bidirectional LSTM

    Example:
        >>> model = RBFLSTMModel(input_size=402, hidden_size=512, num_layers=2)
        >>> x = torch.randn(batch_size, seq_len, 402)
        >>> output = model(x)  # Shape: (batch_size, 402)
    """

    def __init__(
        self,
        input_size: int = 402,
        hidden_size: int = 512,
        num_layers: int = 3,
        output_size: int = 402,
        dropout: float = 0.6,
        bidirectional: bool = False,
    ):
        super().__init__()

        # Validate inputs
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive integers")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional

        # LSTM layer with optional dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  # Dropout only between layers
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Calculate final hidden size (double if bidirectional)
        final_hidden_size = hidden_size * 2 if bidirectional else hidden_size

        # Fully connected output layer
        self.fc = nn.Linear(final_hidden_size, output_size)

        # Optional dropout before output layer
        self.output_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                torch.nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)

        # Initialize FC layer
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)

        Raises:
            ValueError: If input tensor has wrong dimensions
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input tensor (batch_size, seq_len, input_size), got {x.dim()}D"
            )

        if x.size(-1) != self.input_size:
            raise ValueError(f"Expected input_size {self.input_size}, got {x.size(-1)}")

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take output from last timestep
        # Shape: (batch_size, hidden_size) or (batch_size, 2*hidden_size) if bidirectional
        last_output = lstm_out[:, -1, :]

        # Apply dropout and final linear layer
        out = self.output_dropout(last_output)
        out = self.fc(out)

        return out

    def get_model_info(self) -> dict:
        """Return model configuration and parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "bidirectional": self.bidirectional,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
