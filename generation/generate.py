"""Music generation script for all trained models."""

import os
import sys
import argparse
import inspect
import numpy as np
import pickle
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import get_config
from utils.midi_processor import MIDIProcessor
from models.lstm.lstm_model import LSTMMusicGenerator
from models.gru.gru_model import GRUMusicGenerator
from models.vae.vae_model import VAEMusicGenerator
from models.gan.gan_model import GANMusicGenerator


def load_model(model_type, model_path, vocab_size, seq_length, config):
    """Load trained model with correct architecture parameters."""
    print(f"Loading {model_type.upper()} model from {model_path}...")

    if model_type == 'lstm':
        # Resolve runtime profile to get actual architecture used during training
        from train import resolve_runtime_profile
        runtime = resolve_runtime_profile(config)
        
        model = LSTMMusicGenerator(
            vocab_size=vocab_size,
            seq_length=runtime['train_seq_length'],
            embedding_dim=runtime.get('embedding_dim', 512),
            num_layers=runtime.get('layers', 4),
            units=runtime.get('units', 512),
            dense_units=tuple(runtime.get('dense_units', [1024, 512]))
        )
        model.load_model(model_path)
    elif model_type == 'gru':
        model = GRUMusicGenerator(vocab_size, seq_length)
        model.load_model(model_path)
    elif model_type == 'vae':
        model = VAEMusicGenerator(vocab_size, seq_length)
        model.load_model(model_path.replace('.h5', ''))
    elif model_type == 'gan':
        model = GANMusicGenerator(seq_length, vocab_size)
        model.load_models(model_path.replace('.h5', ''))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model

def pick_seed_sequences(data, seed_length, num_samples, vocab_size):
    """Pick random seed sequences from training data."""
    sequences = data['sequences']
    indices = np.random.choice(len(sequences), size=num_samples, replace=True)
    
    seeds = []
    for idx in indices:
        seq = sequences[idx]
        tokens = seq.get('tokens', seq.get('notes', []))
        if len(tokens) >= seed_length:
            seeds.append(tokens[:seed_length])
        else:
            # Pad if too short
            padded = np.zeros(seed_length, dtype=np.int32)
            padded[-len(tokens):] = tokens
            seeds.append(padded)
            
    return np.array(seeds, dtype=np.int32)

def generate_from_lstm_gru(model, seed_sequences, length, temperature, top_k, top_p, repetition_penalty):
    """Generate sequences using LSTM or GRU model."""
    print(f"Generating {len(seed_sequences)} sequences of length {length}...")
    return model.generate_sequences(
        seed_sequences=seed_sequences,
        length=length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )

def generate_from_vae(model, num_samples, length):
    """Generate sequences using VAE model (latent space sampling)."""
    print(f"Generating {num_samples} sequences using VAE latent sampling...")
    return model.generate_sequences(num_samples=num_samples, length=length)

def generate_from_gan(model, num_samples, length):
    """Generate sequences using GAN model."""
    print(f"Generating {num_samples} sequences using GAN noise sampling...")
    return model.generate_samples(num_samples=num_samples)

def save_sequences_as_midi(sequences, output_dir, processor, config):
    """Save generated Performance tokens as MIDI files."""
    print(f"\nSaving {len(sequences)} MIDI files to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    tempo = config.get('generation', 'output', 'tempo', default=120)

    for i, sequence in enumerate(tqdm(sequences)):
        output_path = os.path.join(output_dir, f"generated_{i:04d}.mid")
        processor.sequence_to_midi({'tokens': sequence}, output_path, tempo=tempo)

    print(f"✓ Saved {len(sequences)} MIDI files")

def generate_music(config, model_type, model_path, num_samples=None, output_dir=None):
    """Generate music using trained model."""
    print("\n" + "=" * 70)
    print(f"Music Generation - {model_type.upper()} Model")
    print("=" * 70)

    data_path = os.path.join(config.get('data', 'processed_dir'), 'sequences.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    vocab_size = data['config']['vocab_size']
    # Use training sequence length for the model
    train_seq_length = config.get('data', 'midi_processing', 'max_length')

    print(f"Vocabulary size: {vocab_size}")
    print(f"Model sequence length: {train_seq_length}")

    model = load_model(model_type, model_path, vocab_size, train_seq_length, config)

    if num_samples is None:
        num_samples = config.get('generation', 'num_samples')

    length = config.get('generation', 'sequence_length', default=512)
    seed_length = config.get('generation', 'seed_length', default=64)
    temperature = config.get('generation', 'temperature', default=1.0)
    top_k = config.get('generation', 'top_k', default=50)
    top_p = config.get('generation', 'top_p', default=0.9)
    repetition_penalty = config.get('generation', 'repetition_penalty', default=1.2)

    if model_type in ['lstm', 'gru']:
        seed_sequences = pick_seed_sequences(data, seed_length, num_samples, vocab_size)
        sequences = generate_from_lstm_gru(
            model,
            seed_sequences,
            length,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        )
    elif model_type == 'vae':
        sequences = generate_from_vae(model, num_samples, length)
    elif model_type == 'gan':
        sequences = generate_from_gan(model, num_samples, length)

    midi_cfg = config.get('data', 'midi_processing')
    processor = MIDIProcessor(
        resolution=midi_cfg.get('resolution', 480),
        velocity_bins=midi_cfg.get('velocity_bins', 32),
        time_shift_bins=midi_cfg.get('time_shift_bins', 100),
        max_shift_ms=midi_cfg.get('max_shift_ms', 1000)
    )

    if output_dir is None:
        output_dir = os.path.join(config.get('paths', 'generated_midi'), model_type)

    save_sequences_as_midi(sequences, output_dir, processor, config)

    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"Generated {len(sequences)} sequences")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    return sequences


def main():
    parser = argparse.ArgumentParser(description='Generate music from trained models')
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'gru', 'vae', 'gan'])
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    config = get_config(args.config)
    generate_music(
        config=config,
        model_type=args.model,
        model_path=args.model_path,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
