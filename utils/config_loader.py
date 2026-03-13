"""Configuration loader utility."""

import yaml
import os
from pathlib import Path

class Config:
    """Configuration manager for the project."""

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def get(self, *keys, default=None):
        """Get nested configuration value."""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value

    def __getitem__(self, key):
        return self.config[key]

    def __repr__(self):
        return f"Config({self.config_path})"

# Global config instance
_config = None

def get_config(config_path="config.yaml"):
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
