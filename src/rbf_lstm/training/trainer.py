"""LSTM training logic with improved separation of concerns."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import RBFNetwork
from rbf_lstm.utils.tracker import AbstractTracker
from rbf_lstm.utils.types import SplitData

from ..utils.metrics_history import EpochMetrics
from .evaluator import ModelEvaluator
from .model import RBFLSTMModel


class RBFLSTMTrainer:
    """
    Improved LSTM trainer with better separation of concerns.

    Handles model training with proper evaluation and metrics tracking.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        tracker: AbstractTracker,
        output_size: int,
        loader: DataLoader,
        datasets: dict[str, SplitData],
        number_of_epoch: int,
        rbf: RBFNetwork,
        seq_length: int,
        learning_rate: float = 0.001,
        device: str | None = None,
    ):
        """
        Initialize the RBFLSTM trainer with all required training components.

        Parameters:
            input_size (int): Number of input features per timestep.
            hidden_size (int): Size of the LSTM hidden state.
            num_layers (int): Number of stacked LSTM layers.
            output_size (int): Size of the model output vector.
            tracker (AbstractTracker): Tracking backend used for logging metrics and artifacts.
            loader (DataLoader): DataLoader used during the training loop.
            datasets (Dict[str, SplitData]): Dictionary of dataset splits (e.g. train/val/test).
            number_of_epoch (int): Total number of training epochs.
            rbf (RBFNetwork): RBF network used for evaluation or auxiliary computations.
            seq_length (int): Length of input sequences.
            learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 0.001.
            device (str | None, optional): Target device ('cpu', 'cuda') or None for auto-detection.

        Raises:
            ValueError: If any of the provided hyperparameters are invalid
                        (e.g. non-positive seq_length or hidden_size).
        """

        self.tracker = tracker
        self.loader = loader
        self.datasets = datasets
        self.number_of_epoch = number_of_epoch
        self.rbf = rbf
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.device = device

        self.model = RBFLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
        ).to(self.device)

        model_info = self.model.get_model_info()
        print(f"Model parameters: {model_info['total_parameters']:,}")

        # Loss function and optimizer
        self.loss_fn = nn.HuberLoss()  # nn.MSELoss()
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.01)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )

        # Initialize evaluator
        self.evaluator = ModelEvaluator(
            tracker=tracker,
            rbf=rbf,
            loader=loader,
            model=self.model,
            loss_fn=self.loss_fn,
            datasets=datasets,
            seq_length=seq_length,
        )

    def train_model(
        self,
        normalized: bool = False,
        only_for_first_batch: bool = True,
        eval_interval: int = 1,
        mean: float | None = None,
        std: float | None = None,
    ) -> None:
        """
        Execute the model training loop.

        Handles forward/backward passes, logging, and periodic evaluation.
        Optionally supports normalized data and controlled batch execution.

        Parameters:
            normalized (bool, optional):
                If True, assumes input data has been normalized using provided mean and std.
                Defaults to False.

            only_for_first_batch (bool, optional):
                If True, training will run only on the first batch (useful for debugging
                or quick sanity checks). Defaults to True.

            eval_interval (int, optional):
                Frequency (in epochs) at which validation is performed.
                Example: 1 means evaluate every epoch, 5 means every 5 epochs.
                Defaults to 1.

            mean (float | None, optional):
                Mean value used for normalization. Required if normalized=True.

            std (float | None, optional):
                Standard deviation used for normalization. Required if normalized=True.

        Returns:
            None
        """

        print(f"Starting training for {self.number_of_epoch} epochs...")
        print(f"Evaluation every {eval_interval} epochs")

        num_batches = len(self.loader["train"])
        for epoch in range(self.number_of_epoch):
            self.model.train()
            epoch_loss = 0.0

            # Training loop
            for i_batch, (X_batch, y_batch) in enumerate(self.loader["train"]):
                if only_for_first_batch and i_batch >= 1:
                    break

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                y_pred_batch = self.model(X_batch)
                loss = self.loss_fn(y_pred_batch, y_batch)

                self.tracker.log_metric(
                    "train_loss_step", loss.item(), step=epoch * num_batches + i_batch
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            if epoch % eval_interval == 0:
                self._evaluate_and_log(
                    epoch,
                    normalized,
                    mean,
                    std,
                )

        print("\nTraining completed!")
        self.tracker.log_model(self.model, X_batch)

    def _evaluate_and_log(
        self,
        epoch: int,
        normalized: bool,
        mean: float | None,
        std: float | None,
    ) -> None:
        """Evaluate model and log metrics."""

        self.model.eval()
        train_metrics: EpochMetrics = self.evaluator.evaluate_split(
            "train", epoch, normalized, mean, std
        )
        val_metrics: EpochMetrics = self.evaluator.evaluate_split(
            "val", epoch, normalized, mean, std
        )

        print(f"EPOCH {epoch}")
        print(f"train_metrics: {train_metrics}")
        print(f"val_metrics: {val_metrics}")
        self.model.train()
