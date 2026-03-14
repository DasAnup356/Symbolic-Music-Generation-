"""Main training script for all models."""

import os
import sys
import pickle
import argparse
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import get_config
from utils.midi_processor import prepare_training_data
from models.lstm.lstm_model import LSTMMusicGenerator, create_callbacks
from models.gru.gru_model import GRUMusicGenerator
from models.vae.vae_model import VAEMusicGenerator
from models.gan.gan_model import GANMusicGenerator


def detect_accelerator():
    """Detect TPU/GPU/CPU and return (device, strategy)."""
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return 'tpu', tf.distribute.TPUStrategy(resolver)
    except Exception:
        pass

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return 'gpu', tf.distribute.MirroredStrategy()

    return 'cpu', tf.distribute.get_strategy()


def resolve_runtime_profile(config):
    """Resolve CPU/GPU/TPU-aware runtime overrides for feasible training."""
    device, strategy = detect_accelerator()

    profile = {
        'device': device,
        'strategy': strategy,
        'embedding_dim': config['models']['lstm'].get('embedding_dim', 256),
        'num_layers': config['models']['lstm'].get('layers', 3),
        'units': config['models']['lstm'].get('units', 512),
        'dropout': config['models']['lstm'].get('dropout', 0.3),
        'recurrent_dropout': config['models']['lstm'].get('recurrent_dropout', 0.2),
        'dense_units': tuple(config['models']['lstm'].get('dense_units', [512, 256])),
        'batch_size': config['training']['batch_size'],
        'epochs': config['training']['epochs'],
        'train_seq_length': config.get('data', 'midi_processing', 'max_length'),
    }

    if device == 'cpu':
        cpu = config.get('training', 'cpu_optimized', default={})
        if cpu.get('enabled', True):
            profile.update({
                'embedding_dim': cpu.get('embedding_dim', 64),
                'num_layers': cpu.get('layers', 1),
                'units': cpu.get('units', 96),
                'dropout': cpu.get('dropout', 0.1),
                'recurrent_dropout': cpu.get('recurrent_dropout', 0.0),
                'dense_units': tuple(cpu.get('dense_units', [96])),
                'batch_size': cpu.get('batch_size', 64),
                'epochs': cpu.get('epochs', 6),
                'train_seq_length': cpu.get('train_seq_length', 64),
            })
    elif device == 'gpu':
        gpu = config.get('training', 'gpu_optimized', default={})
        if gpu.get('enabled', True):
            profile.update({
                'batch_size': gpu.get('batch_size', profile['batch_size']),
                'epochs': gpu.get('epochs', profile['epochs']),
                'train_seq_length': gpu.get('train_seq_length', profile['train_seq_length']),
            })
    else:
        tpu = config.get('training', 'tpu_optimized', default={})
        if tpu.get('enabled', True):
            profile.update({
                'batch_size': tpu.get('batch_size', 128),
                'epochs': tpu.get('epochs', profile['epochs']),
                'train_seq_length': tpu.get('train_seq_length', 128),
                'recurrent_dropout': tpu.get('recurrent_dropout', 0.0),
            })

    return profile


def load_data(data_path, config, seq_length=None):
    """Load and prepare training data."""
    print(f"Loading data from {data_path}...")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    sequences = data['sequences']
    vocab_size = data['config']['vocab_size']

    if seq_length is None:
        seq_length = config.get('data', 'midi_processing', 'max_length')

    X, y = prepare_training_data(sequences, vocab_size, seq_length)

    # Shuffle before split for more representative validation
    indices = tf.random.shuffle(tf.range(len(X))).numpy()
    X = X[indices]
    y = y[indices]

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

    print(f"Total sequences: {len(sequences)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab_size


def train_lstm(config, data):
    """Train LSTM model."""
    print("\n" + "=" * 50)
    print("Training LSTM Model")
    print("=" * 50)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab_size = data
    runtime = resolve_runtime_profile(config)

    model_kwargs = {
        'embedding_dim': runtime['embedding_dim'],
        'num_layers': runtime['num_layers'],
        'units': runtime['units'],
        'dropout': runtime['dropout'],
        'recurrent_dropout': runtime['recurrent_dropout'],
        'dense_units': runtime['dense_units'],
    }

    print(f"Detected accelerator: {runtime['device'].upper()}")
    if runtime['device'] == 'cpu':
        print("Running CPU-optimized training profile for feasibility.")

    with runtime['strategy'].scope():
        model = LSTMMusicGenerator(vocab_size=vocab_size, seq_length=X_train.shape[1], **model_kwargs)
        model.compile_model(learning_rate=config['training']['learning_rate'])

    print("\nModel Architecture:")
    model.summary()

    checkpoint_path = os.path.join(config['paths']['checkpoints'], 'lstm_best.h5')
    log_dir = os.path.join(config['paths']['logs'], 'lstm')
    callbacks = create_callbacks(checkpoint_path, log_dir)

    history = model.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=runtime['epochs'],
        batch_size=runtime['batch_size'],
        callbacks=callbacks,
    )

    test_loss, test_acc = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    final_path = os.path.join(config['paths']['models'], 'lstm', 'lstm_final.h5')
    model.save_model(final_path)
    return model, history


def train_gru(config, data):
    """Train GRU model."""
    print("\n" + "=" * 50)
    print("Training GRU Model")
    print("=" * 50)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab_size = data
    runtime = resolve_runtime_profile(config)

    model_kwargs = {
        'embedding_dim': runtime['embedding_dim'],
        'num_layers': runtime['num_layers'],
        'units': runtime['units'],
        'dropout': runtime['dropout'],
        'recurrent_dropout': runtime['recurrent_dropout'],
        'dense_units': runtime['dense_units'],
    }

    with runtime['strategy'].scope():
        model = GRUMusicGenerator(vocab_size=vocab_size, seq_length=X_train.shape[1], **model_kwargs)
        model.compile_model(learning_rate=config['training']['learning_rate'])

    print("\nModel Architecture:")
    model.summary()

    checkpoint_path = os.path.join(config['paths']['checkpoints'], 'gru_best.h5')
    log_dir = os.path.join(config['paths']['logs'], 'gru')
    callbacks = create_callbacks(checkpoint_path, log_dir)

    history = model.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=runtime['epochs'],
        batch_size=runtime['batch_size'],
        callbacks=callbacks,
    )

    test_loss, test_acc = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    final_path = os.path.join(config['paths']['models'], 'gru', 'gru_final.h5')
    model.save_model(final_path)
    return model, history


def train_vae(config, data):
    (X_train, _), (X_val, _), _, vocab_size = data
    vae_config = config['models']['vae']
    model = VAEMusicGenerator(vocab_size=vocab_size, seq_length=X_train.shape[1], latent_dim=vae_config['latent_dim'])
    model.compile_model(learning_rate=config['training']['learning_rate'])
    checkpoint_path = os.path.join(config['paths']['checkpoints'], 'vae_best.h5')
    log_dir = os.path.join(config['paths']['logs'], 'vae')
    callbacks = create_callbacks(checkpoint_path, log_dir)
    history = model.train(X_train, X_val, epochs=config['training']['epochs'], batch_size=config['training']['batch_size'], callbacks=callbacks)
    final_prefix = os.path.join(config['paths']['models'], 'vae', 'vae_final')
    model.save_model(final_prefix)
    return model, history


def train_gan(config, data):
    (X_train, _), _, _, vocab_size = data
    gan_config = config['models']['gan']
    model = GANMusicGenerator(seq_length=X_train.shape[1], vocab_size=vocab_size, latent_dim=gan_config['generator']['latent_dim'])
    model.compile_models(g_learning_rate=config['training']['learning_rate'], d_learning_rate=config['training']['learning_rate'])
    history = model.train(X_train, epochs=config['training']['epochs'], batch_size=config['training']['batch_size'], save_interval=10)
    final_prefix = os.path.join(config['paths']['models'], 'gan', 'gan_final')
    model.save_models(final_prefix)
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train music generation models')
    parser.add_argument('--model', type=str, default='all', choices=['lstm', 'gru', 'vae', 'rbm', 'gan', 'all'])
    parser.add_argument('--data', type=str, default='data/processed/sequences.pkl')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    config = get_config(args.config)

    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    os.makedirs(config['paths']['logs'], exist_ok=True)
    os.makedirs(os.path.join(config['paths']['models'], 'lstm'), exist_ok=True)
    os.makedirs(os.path.join(config['paths']['models'], 'gru'), exist_ok=True)
    os.makedirs(os.path.join(config['paths']['models'], 'vae'), exist_ok=True)
    os.makedirs(os.path.join(config['paths']['models'], 'gan'), exist_ok=True)

    runtime = resolve_runtime_profile(config)
    print(f"Detected accelerator: {runtime['device'].upper()}")
    print(f"Training sequence length: {runtime['train_seq_length']}")
    data = load_data(args.data, config, seq_length=runtime['train_seq_length'])

    if args.model in ['lstm', 'all']:
        train_lstm(config, data)
    if args.model in ['gru', 'all']:
        train_gru(config, data)
    if args.model in ['vae', 'all']:
        train_vae(config, data)
    if args.model in ['gan', 'all']:
        train_gan(config, data)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()
