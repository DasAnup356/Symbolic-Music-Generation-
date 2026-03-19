"""Main pipeline for symbolic music generation project."""

import os
import sys
import argparse
import time
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import get_config
from preprocessing.preprocess import preprocess_dataset
from train import train_lstm, train_gru, train_vae, train_gan, train_cnn, train_rbm, load_data, resolve_runtime_profile
from generation.generate import generate_music
from evaluation.evaluate import evaluate_generated_music
from rag.tutor import setup_tutor, ask_tutor


def resolve_pipeline_runtime(config):
    """Resolve runtime defaults for CPU/GPU/TPU execution."""
    runtime = resolve_runtime_profile(config)
    preprocess_files = config.get('data', 'dataset_size')
    num_samples = config.get('generation', 'num_samples')

    if runtime['device'] == 'cpu':
        cpu_cfg = config.get('training', 'cpu_optimized', default={})
        if cpu_cfg.get('enabled', True):
            preprocess_files = min(preprocess_files, cpu_cfg.get('max_files', 120))
            num_samples = min(num_samples, cpu_cfg.get('generation_samples', 10))
            print(f"CPU-only environment detected. Using lighter pipeline settings: max_files={preprocess_files}, generation_samples={num_samples}")
    else:
        print(f"Accelerator detected: {runtime['device'].upper()}. Using full pipeline settings.")

    return preprocess_files, num_samples


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
    preprocess_max_files, generation_samples = resolve_pipeline_runtime(config)

    # Step 1: Preprocessing
    if 'all' in steps or 'preprocess' in steps:
        print("\n[STEP 1/4] Data Preprocessing")
        print("-" * 80)

        sequences = preprocess_dataset(
            config=config,
            max_files=preprocess_max_files
        )

        print(f"Preprocessed {len(sequences)} training sequences")

    # Step 2: Training
    if 'all' in steps or 'train' in steps:
        print("\n[STEP 2/4] Model Training")
        print("-" * 80)

        runtime = resolve_runtime_profile(config)
        data_path = os.path.join(config.get('data', 'processed_dir'), 'sequences.pkl')
        data = load_data(data_path, config, seq_length=runtime['train_seq_length'])

        # Train models based on config or all
        target_model = config.get('training', 'model', default='lstm')

        if target_model == 'lstm':
            train_lstm(config, data)
        elif target_model == 'gru':
            train_gru(config, data)
        elif target_model == 'vae':
            train_vae(config, data)
        elif target_model == 'gan':
            train_gan(config, data)
        elif target_model == 'cnn':
            train_cnn(config, data)
        elif target_model == 'rbm':
            train_rbm(config, data)
        else:
            # Default to LSTM
            train_lstm(config, data)

        print("Model training complete")

    # Step 3: Generation
    if 'all' in steps or 'generate' in steps:
        print("\n[STEP 3/4] Music Generation")
        print("-" * 80)

        model_path = os.path.join(config['paths']['checkpoints'], 'lstm_best.h5')

        sequences = generate_music(
            config=config,
            model_type='lstm',
            model_path=model_path,
            num_samples=generation_samples
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

def run_tutor(query):
    """Run the AI Tutor with a specific query."""
    try:
        tutor = setup_tutor()
        answer = ask_tutor(query, tutor)
        print("\n" + "="*80)
        print("JULES - AI TUTOR")
        print("="*80)
        print(f"QUESTION: {query}")
        print("-" * 80)
        print(answer)
        print("="*80)
    except Exception as e:
        print(f"Error starting tutor: {e}")
        print("Tip: Make sure to run 'python rag/indexer.py' first to build the knowledge base.")

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
    parser.add_argument(
        '--tutor',
        type=str,
        help='Ask the AI Tutor a question about the project'
    )

    args = parser.parse_args()

    if args.tutor:
        run_tutor(args.tutor)
        return

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
