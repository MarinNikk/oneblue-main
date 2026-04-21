"""Main execution script for RBF-LSTM training pipeline."""

import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rbf_lstm.utils.tracker import AbstractTracker, MLflowTracker
from rbf_lstm.utils.types import SplitData
from src.model.RBFNetwork import RBFNetwork

from .dataset import (
    SeaCurrentDataset,
    create_sequences,
    init_rbf,
    process_ocean_data,
    process_time_groups,
)
from .training import (
    RBFLSTMTrainer,
    check_r2_consistency,
    validate_data_shapes,
    validate_overfitting_success,
)
from .utils import Config, save_rbf_data


def get_existing_data() -> tuple[dict[str, SplitData], RBFNetwork]:
    print("✓ Found existing processed data, loading...")

    # Load existing data using io_utils functions
    from .utils.io_utils import load_dataframe, load_rbf_data

    datasets = {}
    for split in ["train", "val", "test"]:
        time_to_coeff, time_to_metrics, int_to_data, rbf = load_rbf_data(
            config.data.processed_data_dir, split, load_rbf_network=True
        )
        df = load_dataframe(f"{config.data.processed_data_dir}/{split}/dataframe.csv")
        df["time"] = pd.to_datetime(df["time"])  # Fix data type mismatch

        datasets[split] = {
            "time_to_coeff": time_to_coeff,
            "time_to_metrics": time_to_metrics,
            "int_to_data": int_to_data,
            "rbf": rbf,
            "df": df,
        }

    print(f"✓ Loaded RBF with {rbf.n_clusters} clusters")
    print(f"✓ Loaded {len(datasets['train']['time_to_coeff'])} training coefficient sets")
    return datasets, rbf


def generate_new_data(config: Config) -> tuple[dict[str, SplitData], RBFNetwork]:
    print("No existing data found, processing from scratch...")

    train_df, test_df, val_df, int_to_data_train, int_to_data_test, int_to_data_val = (
        process_ocean_data(config)
    )

    # Step 2: RBF Parameter Generation
    print("\n" + "=" * 50)
    print("STEP 2: RBF PARAMETER GENERATION")
    print("=" * 50)

    print("Initializing RBF network...")
    rbf = init_rbf(train_df, int_to_data_train)

    # Process all dataset splits
    splits = [
        ("train", train_df, int_to_data_train),
        ("test", test_df, int_to_data_test),
        ("val", val_df, int_to_data_val),
    ]

    # Store results for later use
    time_to_coeff_dict = {}
    time_to_metrics_dict = {}

    for split_name, df, int_to_data in splits:
        print(f"Processing RBF parameters for {split_name} data...")
        time_to_coeff, time_to_metrics, _ = process_time_groups(
            list(int_to_data.items()), df, rbf, process_name=split_name
        )

        print(f"Saving RBF parameters for {split_name}...")
        save_rbf_data(
            config.data.processed_data_dir,
            split_name,
            time_to_coeff,
            time_to_metrics,
            int_to_data,
            rbf if split_name == "train" else None,  # Only save RBF with training data
        )

        # Store for later use
        time_to_coeff_dict[split_name] = time_to_coeff
        time_to_metrics_dict[split_name] = time_to_metrics

    # Extract train-specific data for subsequent steps
    datasets = {}
    for split, df, int_to_data in zip(
        ["train", "val", "test"],
        [train_df, test_df, val_df],
        [int_to_data_train, int_to_data_test, int_to_data_val],
        strict=True,
    ):
        datasets[split] = {
            "time_to_coeff": time_to_coeff_dict["train"],
            "time_to_metrics": time_to_metrics_dict["train"],
            "int_to_data": int_to_data,
            "rbf": rbf,
            "df": df,
        }
    return datasets, rbf


def generate_data_loaders(
    datasets: dict[str, SplitData],
    rbf: RBFNetwork,
    batch_size: int,
    seq_length: int,
    normalized: bool,
) -> tuple[dict[str, DataLoader], torch.Tensor | None, torch.Tensor | None]:
    dataloaders = {}
    for split in datasets:
        X_tensor, y_tensor = create_sequences(
            datasets[split]["time_to_coeff"], datasets[split]["int_to_data"], seq_length=seq_length
        )
        # Normalization (if enabled)
        mean, std = None, None
        if normalized:
            combined_tensor = torch.cat([X_tensor.flatten(), y_tensor.flatten()])
            mean = combined_tensor.mean()
            std = combined_tensor.std()

            if std == 0:
                std = 1e-8

            X_tensor = (X_tensor - mean) / std
            y_tensor = (y_tensor - mean) / std

            print("Original X range:", X_tensor.min().item(), X_tensor.max().item())
            print("Original y range:", y_tensor.min().item(), y_tensor.max().item())
            print("Standardized X mean:", X_tensor.mean().item(), "std:", X_tensor.std().item())
            print("Standardized y mean:", y_tensor.mean().item(), "std:", y_tensor.std().item())

        # Create dataset and dataloader
        dataset = SeaCurrentDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size)
        dataloaders[split] = loader

        # Validate data shapes
        validate_data_shapes(
            X_tensor,
            y_tensor,
            datasets[split]["df"],
            datasets[split]["int_to_data"],
            rbf,
            seq_length,
        )
    return dataloaders, mean, std


def main(tracker: AbstractTracker):
    """Main execution function following the exact notebook workflow."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\n" + "=" * 50)
    print("STEP 1: DATA LOADING AND PROCESSING")
    print("=" * 50)

    # Check if processed data already exists
    if os.path.exists(f"{config.data.processed_data_dir}/rbf.pkl"):
        datasets, rbf = get_existing_data()
    else:
        datasets, rbf = generate_new_data(config)

    print("\n" + "=" * 50)
    print("STEP 3: DATA VALIDATION")
    print("=" * 50)
    for split in datasets:
        check_r2_consistency(
            datasets[split]["df"],
            datasets[split]["int_to_data"],
            datasets[split]["time_to_coeff"],
            datasets[split]["time_to_metrics"],
            rbf,
        )

    print("\n" + "=" * 50)
    print("STEP 4: LSTM TRAINING PREPARATION")
    print("=" * 50)

    # Training parameters
    normalized = config.lstm.normalized
    only_for_first_batch = config.lstm.only_first_batch
    seq_length = config.lstm.seq_length
    batch_size = config.lstm.batch_size
    number_of_epoch = config.lstm.num_epochs
    # LSTM parameters
    n_clusters = rbf.n_clusters
    bias = 2
    W_size = 2 * n_clusters
    input_size = W_size + bias
    hidden_size = config.lstm.hidden_size
    num_layers = config.lstm.num_layers
    output_size = W_size + bias

    print("LSTM parameters:")
    print(f"  input_size: {input_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_layers: {num_layers}")
    print(f"  output_size: {output_size}")
    print(f"  seq_length: {seq_length}")
    print(f"  batch_size: {batch_size}")

    print("Creating sequences...")
    dataloaders, mean, std = generate_data_loaders(
        datasets, rbf, batch_size, seq_length, normalized
    )

    print("\n" + "=" * 50)
    print("STEP 5: LSTM TRAINING")
    print("=" * 50)

    trainer = RBFLSTMTrainer(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        tracker=tracker,
        output_size=output_size,
        loader=dataloaders,
        datasets=datasets,
        number_of_epoch=number_of_epoch,
        rbf=rbf,
        seq_length=seq_length,
        learning_rate=config.lstm.learning_rate,
        device=device,
    )

    trainer.train_model(
        normalized=normalized,
        only_for_first_batch=only_for_first_batch,
        eval_interval=config.lstm.eval_interval,
        mean=mean,
        std=std,
    )

    # Get final predictions for validation
    trainer.model.eval()
    with torch.no_grad():
        for i_batch, (X_batch, y_batch) in enumerate(dataloaders["train"]):
            if only_for_first_batch and i_batch >= 1:
                break
            X_batch = X_batch.to(trainer.device)
            y_batch = y_batch.to(trainer.device)
            y_pred_batch = trainer.model(X_batch)

    # Validate overfitting success
    if only_for_first_batch:
        validate_overfitting_success(
            datasets["train"]["time_to_coeff"],
            datasets["train"]["int_to_data"],
            seq_length,
            y_pred_batch,
            normalized,
            mean,
            std,
        )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)


if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    print("Configuration initialized")

    mlflow_tracker = MLflowTracker(experiment_name="oneblue-rbf-lstm", config=config)

    try:
        main(tracker=mlflow_tracker)
    finally:
        mlflow_tracker.end_run()
