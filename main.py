"""Main pipeline for symbolic music generation project."""

import os
import sys
import argparse
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import get_config
from preprocessing.preprocess import preprocess_dataset
from train import train_lstm
from generation.generate import generate_music
from evaluation.evaluate import evaluate_generated_music

def run_pipeline(config, steps=['all']):
    """
    Run the complete music generation pipeline.

    Args:
        config: Configuration dictionary
        steps: List of pipeline steps to execute
    """
    print("\n" + "="*80)
    print("SYMBOLIC MUSIC GENERATION PIPELINE")
    print("Deep Learning Techniques for Music Generation - Survey Implementation")
    print("="*80)

    start_time = time.time()

    # Step 1: Preprocessing
    if 'all' in steps or 'preprocess' in steps:
        print("\n[STEP 1/4] Data Preprocessing")
        print("-" * 80)

        sequences = preprocess_dataset(
            config=config,
            max_files=100  # Start with 100 files for testing
        )

        print(f"Preprocessed {len(sequences)} training sequences")

    # Step 2: Training
    if 'all' in steps or 'train' in steps:
        print("\n[STEP 2/4] Model Training")
        print("-" * 80)

        # Load data
        import pickle
        data_path = os.path.join(config.get('data', 'processed_dir'), 'sequences.pkl')

        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)

        sequences = data_dict['sequences']
        vocab_size = data_dict['config']['vocab_size']

        print(f"Training sequences: {len(sequences)}")
        print(f"Vocabulary size: {vocab_size}")

        # Prepare training data
        from utils.midi_processor import prepare_training_data

        seq_length = config.get('data', 'midi_processing', 'max_length')
        X, y = prepare_training_data(sequences, vocab_size, seq_length)

        # Split data
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

        data = ((X_train, y_train), (X_val, y_val), (X_test, y_test), vocab_size)

        # Train LSTM model (3-layer, 512 units)
        model, history = train_lstm(config, data)

        print("Model trained successfully")
        print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

    # Step 3: Generation
    if 'all' in steps or 'generate' in steps:
        print("\n[STEP 3/4] Music Generation")
        print("-" * 80)

        model_path = os.path.join(config['paths']['checkpoints'], 'lstm_best.h5')

        sequences = generate_music(
            config=config,
            model_type='lstm',
            model_path=model_path,
            num_samples=50  # Generate 50 samples for testing
        )

        print(f"Generated {len(sequences)} MIDI files")

    # Step 4: Evaluation
    if 'all' in steps or 'evaluate' in steps:
        print("\n[STEP 4/4] Evaluation")
        print("-" * 80)

        midi_dir = os.path.join(config['paths']['generated_midi'], 'lstm')

        stats = evaluate_generated_music(
            config=config,
            midi_dir=midi_dir
        )

        print("Evaluation complete")

    elapsed_time = time.time() - start_time

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Total execution time: {elapsed_time/60:.2f} minutes")
    print("="*80)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Symbolic Music Generation Pipeline'
    )
    parser.add_argument(
        '--steps',
        nargs='+',
        default=['all'],
        choices=['all', 'preprocess', 'train', 'generate', 'evaluate'],
        help='Pipeline steps to execute'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)

    # Create necessary directories
    os.makedirs(config['data']['raw_dir'], exist_ok=True)
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    os.makedirs(config['data']['midi_dir'], exist_ok=True)
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    os.makedirs(config['paths']['logs'], exist_ok=True)
    os.makedirs(config['paths']['generated_midi'], exist_ok=True)

    # Run pipeline
    run_pipeline(config, steps=args.steps)

if __name__ == '__main__':
    main()
