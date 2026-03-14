"""Main training script for all models."""

import os
import sys
import yaml
import numpy as np
import pickle
from pathlib import Path
import argparse
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import get_config
from utils.midi_processor import MIDIProcessor, prepare_training_data
from models.lstm.lstm_model import LSTMMusicGenerator, create_callbacks
from models.gru.gru_model import GRUMusicGenerator
from models.vae.vae_model import VAEMusicGenerator
from models.rbm.rbm_model import RBMMusicGenerator
from models.gan.gan_model import GANMusicGenerator


def resolve_runtime_profile(config):
    """Resolve CPU/GPU-aware runtime overrides for faster training."""
    has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    profile = {
        'has_gpu': has_gpu,
        'embedding_dim': 128,
        'num_layers': 2,
        'units': 256,
        'dropout': 0.2,
        'recurrent_dropout': 0.0,
        'dense_units': (256, 128),
        'batch_size': config['training']['batch_size'],
        'epochs': config['training']['epochs'],
    }

    if not has_gpu:
        cpu = config.get('training', 'cpu_optimized', default={})
        if cpu.get('enabled', True):
            profile.update({
                'embedding_dim': cpu.get('embedding_dim', 96),
                'num_layers': cpu.get('layers', 2),
                'units': cpu.get('units', 192),
                'dropout': cpu.get('dropout', 0.2),
                'recurrent_dropout': cpu.get('recurrent_dropout', 0.0),
                'dense_units': tuple(cpu.get('dense_units', [192, 96])),
                'batch_size': cpu.get('batch_size', 32),
                'epochs': cpu.get('epochs', 20),
            })

    return profile


def load_data(data_path, config):
    """Load and prepare training data."""
    print(f"Loading data from {data_path}...")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    sequences = data['sequences']
    vocab_size = data['config']['vocab_size']

    print(f"Total sequences: {len(sequences)}")
    print(f"Vocabulary size: {vocab_size}")

    # Prepare training data
    seq_length = config.get('data', 'midi_processing', 'max_length')
    X, y = prepare_training_data(sequences, vocab_size, seq_length)

    # Split into train/val/test
    train_split = config.get('data', 'train_split')
    val_split = config.get('data', 'val_split')

    n_train = int(len(X) * train_split)
    n_val = int(len(X) * val_split)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab_size

def train_lstm(config, data):
    """Train LSTM model."""
    print("\n" + "="*50)
    print("Training LSTM Model")
    print("="*50)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab_size = data

    # Create model
    lstm_config = config['models']['lstm']
    runtime = resolve_runtime_profile(config)

    if runtime['has_gpu']:
        model_kwargs = {
            'embedding_dim': lstm_config.get('embedding_dim', runtime['embedding_dim']),
            'num_layers': lstm_config.get('layers', runtime['num_layers']),
            'units': lstm_config.get('units', runtime['units']),
            'dropout': lstm_config.get('dropout', runtime['dropout']),
            'recurrent_dropout': lstm_config.get('recurrent_dropout', runtime['recurrent_dropout']),
            'dense_units': tuple(lstm_config.get('dense_units', runtime['dense_units'])),
        }
    else:
        model_kwargs = {
            'embedding_dim': runtime['embedding_dim'],
            'num_layers': runtime['num_layers'],
            'units': runtime['units'],
            'dropout': runtime['dropout'],
            'recurrent_dropout': runtime['recurrent_dropout'],
            'dense_units': runtime['dense_units'],
        }
        print("Running CPU-optimized training profile for feasibility.")

    model = LSTMMusicGenerator(
        vocab_size=vocab_size,
        seq_length=X_train.shape[1],
        **model_kwargs,
    )

    print("\nModel Architecture:")
    model.summary()

    # Compile model
    model.compile_model(learning_rate=config['training']['learning_rate'])

    # Create callbacks
    checkpoint_path = os.path.join(config['paths']['checkpoints'], 'lstm_best.h5')
    log_dir = os.path.join(config['paths']['logs'], 'lstm')
    callbacks = create_callbacks(checkpoint_path, log_dir)

    # Train model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=runtime['epochs'],
        batch_size=runtime['batch_size'],
        callbacks=callbacks
    )

    # Evaluate on test set
    test_loss, test_acc = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Save final model
    final_path = os.path.join(config['paths']['models'], 'lstm', 'lstm_final.h5')
    model.save_model(final_path)

    return model, history

def train_gru(config, data):
    """Train GRU model."""
    print("\n" + "="*50)
    print("Training GRU Model")
    print("="*50)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab_size = data

    gru_config = config['models']['gru']
    runtime = resolve_runtime_profile(config)

    if runtime['has_gpu']:
        model_kwargs = {
            'embedding_dim': gru_config.get('embedding_dim', runtime['embedding_dim']),
            'num_layers': gru_config.get('layers', runtime['num_layers']),
            'units': gru_config.get('units', runtime['units']),
            'dropout': gru_config.get('dropout', runtime['dropout']),
            'recurrent_dropout': gru_config.get('recurrent_dropout', runtime['recurrent_dropout']),
            'dense_units': tuple(gru_config.get('dense_units', runtime['dense_units'])),
        }
    else:
        model_kwargs = {
            'embedding_dim': runtime['embedding_dim'],
            'num_layers': runtime['num_layers'],
            'units': runtime['units'],
            'dropout': runtime['dropout'],
            'recurrent_dropout': runtime['recurrent_dropout'],
            'dense_units': runtime['dense_units'],
        }
        print("Running CPU-optimized training profile for feasibility.")

    model = GRUMusicGenerator(
        vocab_size=vocab_size,
        seq_length=X_train.shape[1],
        **model_kwargs,
    )

    print("\nModel Architecture:")
    model.summary()

    model.compile_model(learning_rate=config['training']['learning_rate'])

    checkpoint_path = os.path.join(config['paths']['checkpoints'], 'gru_best.h5')
    log_dir = os.path.join(config['paths']['logs'], 'gru')
    callbacks = create_callbacks(checkpoint_path, log_dir)

    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=runtime['epochs'],
        batch_size=runtime['batch_size'],
        callbacks=callbacks
    )

    test_loss, test_acc = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    final_path = os.path.join(config['paths']['models'], 'gru', 'gru_final.h5')
    model.save_model(final_path)

    return model, history

def train_vae(config, data):
    """Train VAE model."""
    print("\n" + "="*50)
    print("Training VAE Model")
    print("="*50)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab_size = data

    vae_config = config['models']['vae']
    model = VAEMusicGenerator(
        vocab_size=vocab_size,
        seq_length=X_train.shape[1],
        latent_dim=vae_config['latent_dim']
    )

    model.compile_model(learning_rate=config['training']['learning_rate'])

    checkpoint_path = os.path.join(config['paths']['checkpoints'], 'vae_best.h5')
    log_dir = os.path.join(config['paths']['logs'], 'vae')
    callbacks = create_callbacks(checkpoint_path, log_dir)

    history = model.train(
        X_train, X_val,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=callbacks
    )

    final_prefix = os.path.join(config['paths']['models'], 'vae', 'vae_final')
    model.save_model(final_prefix)

    return model, history

def train_gan(config, data):
    """Train GAN model."""
    print("\n" + "="*50)
    print("Training GAN Model")
    print("="*50)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab_size = data

    gan_config = config['models']['gan']
    model = GANMusicGenerator(
        seq_length=X_train.shape[1],
        vocab_size=vocab_size,
        latent_dim=gan_config['generator']['latent_dim']
    )

    model.compile_models(
        g_learning_rate=config['training']['learning_rate'],
        d_learning_rate=config['training']['learning_rate']
    )

    history = model.train(
        X_train,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        save_interval=10
    )

    final_prefix = os.path.join(config['paths']['models'], 'gan', 'gan_final')
    model.save_models(final_prefix)

    return model, history

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train music generation models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['lstm', 'gru', 'vae', 'rbm', 'gan', 'all'],
                       help='Model to train')
    parser.add_argument('--data', type=str, default='data/processed/sequences.pkl',
                       help='Path to processed data')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)

    # Create output directories
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    os.makedirs(config['paths']['logs'], exist_ok=True)
    os.makedirs(os.path.join(config['paths']['models'], 'lstm'), exist_ok=True)
    os.makedirs(os.path.join(config['paths']['models'], 'gru'), exist_ok=True)
    os.makedirs(os.path.join(config['paths']['models'], 'vae'), exist_ok=True)
    os.makedirs(os.path.join(config['paths']['models'], 'gan'), exist_ok=True)

    # Load data
    data = load_data(args.data, config)

    # Train models
    results = {}

    if args.model in ['lstm', 'all']:
        model, history = train_lstm(config, data)
        results['lstm'] = {'model': model, 'history': history}

    if args.model in ['gru', 'all']:
        model, history = train_gru(config, data)
        results['gru'] = {'model': model, 'history': history}

    if args.model in ['vae', 'all']:
        model, history = train_vae(config, data)
        results['vae'] = {'model': model, 'history': history}

    if args.model in ['gan', 'all']:
        model, history = train_gan(config, data)
        results['gan'] = {'model': model, 'history': history}

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)

    return results

if __name__ == '__main__':
    main()
