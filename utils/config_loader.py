"""Configuration loader utility."""

import yaml
import os
from pathlib import Path

DEFAULT_CONFIG = {
    'data': {
        'raw_dir': 'data/raw',
        'processed_dir': 'data/processed',
        'midi_dir': 'data/midi_files',
        'dataset_size': 10000,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
        'midi_processing': {
            'resolution': 480,
            'max_length': 512,
            'min_length': 32,
        },
        'representation': {
            'note_range': [21, 108],
            'instrument_bins': 16,
            'velocity_bins': 32,
            'duration_bins': 16,
            'time_shift_bins': 100,
        },
    },
    'generation': {
        'num_samples': 100,
        'sequence_length': 512,
        'seed_length': 128,
        'temperature': 0.85,
        'top_k': 50,
        'top_p': 0.9,
        'repetition_penalty': 1.15,
        'output': {
            'tempo': 120,
            'velocity': 80,
        },
    },
    'paths': {
        'generated_midi': 'outputs/generated_midi/',
    },
    'training': {
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 0.001,
        'cpu_optimized': {
            'enabled': True,
            'max_files': 200,
            'generation_samples': 16,
            'train_seq_length': 128,
        },
        'gpu_optimized': {'enabled': True},
        'tpu_optimized': {'enabled': True},
    },
    'models': {
        'lstm': {
            'embedding_dim': 256,
            'layers': 3,
            'units': 512,
            'dropout': 0.3,
            'recurrent_dropout': 0.2,
            'dense_units': [512, 256],
        },
    },
}


def _deep_merge(defaults, override):
    merged = dict(defaults)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

class Config:
    """Configuration manager for the project."""

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        return _deep_merge(DEFAULT_CONFIG, config)

    def get(self, *keys, default=None):
        """Get nested configuration value."""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def require(self, *keys):
        """Get nested configuration value and raise a clear error if missing."""
        value = self.get(*keys, default=None)
        if value is None:
            path = '.'.join(str(key) for key in keys)
            raise KeyError(f"Missing required config key: {path}")
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
    if _config is None or os.path.abspath(_config.config_path) != os.path.abspath(config_path):
        _config = Config(config_path)
    return _config
