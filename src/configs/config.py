"""Configuration management with dot notation access"""

from pathlib import Path
from typing import Any

import yaml


class DotDict:
    """Dictionary that supports dot notation access."""

    def __init__(self, data: dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            elif isinstance(value, list):
                # Handle lists that might contain dicts
                setattr(
                    self,
                    key,
                    [DotDict(item) if isinstance(item, dict) else item for item in value],
                )
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        items = dict(self.__dict__.items())
        return f"DotDict({items})"


class Config:
    """Configuration loader with dot notation access."""

    def __init__(self, config_path: str = "config.yaml", secrets_path: str = "secrets.yaml"):
        self.config_path = Path(config_path)
        self.secrets_path = Path(secrets_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load main config
        with open(self.config_path) as f:
            self._raw_config = yaml.safe_load(f)

        # Load and merge secrets if they exist
        if self.secrets_path.exists():
            with open(self.secrets_path) as f:
                secrets = yaml.safe_load(f)
                if secrets:
                    self._merge_dict(self._raw_config, secrets)
        else:
            print(f"Warning: Secrets file not found at {secrets_path}")

        # Convert all nested dicts to support dot notation
        for key, value in self._raw_config.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            elif isinstance(value, list):
                setattr(
                    self,
                    key,
                    [DotDict(item) if isinstance(item, dict) else item for item in value],
                )
            else:
                setattr(self, key, value)

    def _merge_dict(self, base: dict, update: dict):
        """Recursively merge update dict into base dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_dict(base[key], value)
            else:
                base[key] = value

    def __getattr__(self, name):
        # Fallback for attributes not set during init
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return self._raw_config.get(name)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert config back to dictionary."""
        return self._raw_config

    def __repr__(self):
        return f"Config(path='{self.config_path}')"


# Example usage:
if __name__ == "__main__":
    config = Config("config.yaml")

    # Access with dot notation
    print(f"Currents variables: {config.variable_mapping.currents}")
    print(f"SSH variables: {config.variable_mapping.ssh}")
    print(f"Spatial size: {config.spatial_size}")
    print(f"CMEMS username: {config.cmems.username}")
    print(f"Lon min: {config.spatial_bounds.lon_min}")
    print(f"Hidden dims: {config.hidden_dims}")
