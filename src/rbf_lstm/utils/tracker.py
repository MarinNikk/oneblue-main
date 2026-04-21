import json
from dataclasses import asdict

import mlflow

from .config import Config


class AbstractTracker:
    """Interface for logging experiments."""

    def log_config(self, config_obj: Config):
        raise NotImplementedError

    def log_metric(self, key, value, step=None):
        raise NotImplementedError

    def log_model(self, model, artifact_path):
        raise NotImplementedError

    def log_artifact(self, local_path, artifact_path=None):
        raise NotImplementedError


class MLflowTracker(AbstractTracker):
    """Concrete implementation using MLflow."""

    def __init__(self, experiment_name: str, config: Config):
        self.mlflow = mlflow
        self.config = config

        self.mlflow.set_experiment(experiment_name)

        run_name = f"rbf-lstm_c{config.rbf.n_clusters}_bs{config.lstm.batch_size}_h{config.lstm.hidden_size}_lr{config.lstm.learning_rate}_nor{int(config.lstm.normalized)}"

        self.run = self.mlflow.start_run(run_name=run_name)

        self.log_config(config)

    def log_config(self, config_obj: Config):
        """Logs entire Config object as JSON and key parameters as MLflow parameters."""

        conf_dic = asdict(config_obj)
        _ = json.dumps(conf_dic, indent=2)

        self.mlflow.log_param("rbf_n_clusters", config_obj.rbf.n_clusters)
        self.mlflow.log_param("rbf_spread", config_obj.rbf.spread)
        self.mlflow.log_param("lstm_hidden_size", config_obj.lstm.hidden_size)
        self.mlflow.log_param("lstm_num_layers", config_obj.lstm.num_layers)
        self.mlflow.log_param("lstm_batch_size", config_obj.lstm.batch_size)
        self.mlflow.log_param("lstm_learning_rate", config_obj.lstm.learning_rate)
        self.mlflow.log_param("lstm_normalized", int(config_obj.lstm.normalized))
        self.mlflow.log_param("seq_length", config_obj.lstm.seq_length)
        self.mlflow.log_param("num_epochs", config_obj.lstm.num_epochs)

    def log_metric(self, key, value, step=None):
        self.mlflow.log_metric(key, value, step=step)

    def log_model(self, model, sample_input_tensor, artifact_path="model"):
        self.mlflow.pytorch.log_model(
            model,
            input_example=sample_input_tensor.detach().cpu().numpy(),
            name=artifact_path,  # .detach().cpu().numpy()
        )
        print("Model save to mlflow")

    def log_artifact(self, local_path, artifact_path=None):
        self.mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def end_run(self):
        self.mlflow.end_run()
