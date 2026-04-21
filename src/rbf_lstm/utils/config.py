"""Configuration settings for RBF-LSTM training pipeline."""

import os
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for data processing."""

    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed/rbf-lstm"
    uo_file: str = "adriatic_currents_uo_detided_2023_2024.nc"
    vo_file: str = "adriatic_currents_vo_detided_2023_2024.nc"
    train_ratio: float = 0.70
    test_ratio: float = 0.15  # remaining 0.15 will be validation
    seed: int = 42


@dataclass
class RBFConfig:
    """Configuration for RBF network."""

    n_clusters: int = 200
    spread: float = None  # Will be auto-determined
    n_workers: int = -2  # Use all CPUs except 2

    def get_model_identifier(self, lstm_config) -> str:
        """Generate a unique identifier for the current model configuration."""
        spread_str = f"spread{self.spread:.3f}" if self.spread else "spread_auto"
        return f"clusters{self.n_clusters}_{spread_str}_lstm{lstm_config.hidden_size}x{lstm_config.num_layers}"

    def get_model_save_dir(self, lstm_config) -> str:
        """Get the directory path for saving models with current configuration."""
        model_id = self.get_model_identifier(lstm_config)
        save_dir = os.path.join("models", model_id)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def get_rbf_save_path(self, lstm_config) -> str:
        """Get the path for saving RBF model."""
        return os.path.join(self.get_model_save_dir(lstm_config), "rbf_model.pkl")

    def get_lstm_save_path(self, lstm_config) -> str:
        """Get the path for saving LSTM model."""
        return os.path.join(self.get_model_save_dir(lstm_config), "lstm_model.pth")

    def get_processed_data_dir(self, data_config, lstm_config) -> str:
        """Get the directory for processed data with current configuration."""
        model_id = self.get_model_identifier(lstm_config)
        data_dir = os.path.join(data_config.processed_data_dir, model_id)
        os.makedirs(data_dir, exist_ok=True)
        for split in ["train", "test", "val"]:
            os.makedirs(os.path.join(data_dir, split), exist_ok=True)
        return data_dir

    def check_existing_models(self, data_config, lstm_config) -> dict:
        """Check what models exist for the current configuration."""
        rbf_path = self.get_rbf_save_path(lstm_config)
        lstm_path = self.get_lstm_save_path(lstm_config)
        data_dir = self.get_processed_data_dir(data_config, lstm_config)

        return {
            "rbf_exists": os.path.exists(rbf_path),
            "lstm_exists": os.path.exists(lstm_path),
            "processed_data_exists": os.path.exists(
                os.path.join(data_dir, "train", "time_to_coeff.pkl")
            ),
            "rbf_path": rbf_path,
            "lstm_path": lstm_path,
            "data_dir": data_dir,
            "model_id": self.get_model_identifier(lstm_config),
        }


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""

    seq_length: int = 32
    batch_size: int = 8
    hidden_size: int = 512  # 384
    num_layers: int = 2
    learning_rate: float = 0.0005  # 0.001
    num_epochs: int = 50
    eval_interval: int = 1
    normalized: bool = True
    only_first_batch: bool = False


@dataclass
class Config:
    """Main configuration class."""

    data: DataConfig
    rbf: RBFConfig
    lstm: LSTMConfig

    def __init__(self):
        self.data = DataConfig()
        self.rbf = RBFConfig()
        self.lstm = LSTMConfig()

        # Ensure output directories exist
        os.makedirs(self.data.processed_data_dir, exist_ok=True)
        for split in ["train", "test", "val"]:
            os.makedirs(os.path.join(self.data.processed_data_dir, split), exist_ok=True)

    @property
    def input_size(self) -> int:
        """Calculate LSTM input size based on RBF parameters."""
        return 2 * self.rbf.n_clusters + 2  # W_size + bias

    @property
    def output_size(self) -> int:
        """Calculate LSTM output size based on RBF parameters."""
        return 2 * self.rbf.n_clusters + 2  # W_size + bias

    def get_split_indices(self, total_size: int) -> tuple[int, int, int]:
        """Calculate train/test/val split indices."""
        train_end = int(self.data.train_ratio * total_size) - 1
        test_end = int((self.data.train_ratio + self.data.test_ratio) * total_size) - 1
        val_end = total_size - 1
        return train_end, test_end, val_end
