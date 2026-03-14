"""Music generation script for all trained models."""

import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import get_config
from utils.midi_processor import MIDIProcessor
from models.lstm.lstm_model import LSTMMusicGenerator
from models.gru.gru_model import GRUMusicGenerator
from models.vae.vae_model import VAEMusicGenerator
from models.gan.gan_model import GANMusicGenerator

def load_model(model_type, model_path, vocab_size, seq_length):
    """Load trained model."""
    print(f"Loading {model_type.upper()} model from {model_path}...")

    if model_type == 'lstm':
        model = LSTMMusicGenerator(vocab_size, seq_length)
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

def generate_from_lstm_gru(model, seed_sequence, num_samples, length, temperature):
    """Generate sequences from LSTM/GRU model (batched when supported)."""
    print(f"Generating {num_samples} sequences...")

    if hasattr(model, 'generate_sequences'):
        seed_batch = np.tile(seed_sequence, (num_samples, 1))
        generated = model.generate_sequences(
            seed_sequences=seed_batch,
            length=length,
            temperature=temperature
        )
        return [generated[i] for i in range(generated.shape[0])]

    generated_sequences = []
    for _ in tqdm(range(num_samples)):
        sequence = model.generate_sequence(
            seed_sequence=seed_sequence,
            length=length,
            temperature=temperature
        )
        generated_sequences.append(sequence)

    return generated_sequences

def generate_from_vae(model, num_samples, length):
    """Generate sequences from VAE model."""
    print(f"Generating {num_samples} sequences from latent space...")

    generated_sequences = []
    for i in tqdm(range(num_samples)):
        # Sample from latent space
        z = np.random.normal(0, 1, (1, model.latent_dim))
        generated = model.decode(z)

        # Convert from one-hot to indices
        sequence = np.argmax(generated[0], axis=-1)
        generated_sequences.append(sequence)

    return generated_sequences

def generate_from_gan(model, num_samples, length):
    """Generate sequences from GAN model."""
    print(f"Generating {num_samples} sequences from GAN...")

    # Generate in batches
    batch_size = 32
    generated_sequences = []

    for i in tqdm(range(0, num_samples, batch_size)):
        current_batch = min(batch_size, num_samples - i)
        generated = model.generate(num_samples=current_batch)

        # Convert from one-hot to indices
        for j in range(current_batch):
            sequence = np.argmax(generated[j], axis=-1)
            generated_sequences.append(sequence)

    return generated_sequences

def save_sequences_as_midi(sequences, output_dir, processor, config):
    """Save generated sequences as MIDI files."""
    print(f"\nSaving {len(sequences)} MIDI files to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    tempo = config.get('generation', 'output', 'tempo')
    velocity = config.get('generation', 'output', 'velocity')

    for i, sequence in enumerate(tqdm(sequences)):
        # Create sequence dictionary
        seq_data = {
            'notes': sequence,
            'durations': np.full(len(sequence), processor.resolution),
            'velocities': np.full(len(sequence), velocity),
            'time_shifts': np.arange(len(sequence)) * processor.resolution
        }

        # Save as MIDI
        output_path = os.path.join(output_dir, f"generated_{i:04d}.mid")
        processor.sequence_to_midi(seq_data, output_path, tempo=tempo, velocity=velocity)

    print(f"✓ Saved {len(sequences)} MIDI files")

def generate_music(config, model_type, model_path, num_samples=None, output_dir=None):
    """
    Generate music using trained model.

    Args:
        config: Configuration dictionary
        model_type: Type of model (lstm, gru, vae, gan)
        model_path: Path to trained model
        num_samples: Number of sequences to generate
        output_dir: Directory to save generated MIDI files
    """
    print("\n" + "="*70)
    print(f"Music Generation - {model_type.upper()} Model")
    print("="*70)

    # Load processed data info
    data_path = os.path.join(config.get('data', 'processed_dir'), 'sequences.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    vocab_size = data['config']['vocab_size']
    seq_length = config.get('generation', 'seed_length')

    print(f"Vocabulary size: {vocab_size}")
    print(f"Sequence length: {seq_length}")

    # Load model
    model = load_model(model_type, model_path, vocab_size, seq_length)

    # Generation parameters
    if num_samples is None:
        num_samples = config.get('generation', 'num_samples')

    length = config.get('generation', 'sequence_length')
    temperature = config.get('generation', 'temperature')

    # Create seed sequence for LSTM/GRU
    seed_sequence = np.random.randint(0, vocab_size, size=seq_length)

    # Generate sequences
    if model_type in ['lstm', 'gru']:
        sequences = generate_from_lstm_gru(
            model, seed_sequence, num_samples, length, temperature
        )
    elif model_type == 'vae':
        sequences = generate_from_vae(model, num_samples, length)
    elif model_type == 'gan':
        sequences = generate_from_gan(model, num_samples, length)

    # Initialize MIDI processor
    processor = MIDIProcessor(
        note_range=tuple(config.get('data', 'representation', 'note_range')),
        resolution=config.get('data', 'midi_processing', 'resolution')
    )

    # Save as MIDI files
    if output_dir is None:
        output_dir = os.path.join(
            config.get('paths', 'generated_midi'),
            model_type
        )

    save_sequences_as_midi(sequences, output_dir, processor, config)

    print("\n" + "="*70)
    print("Generation Complete!")
    print("="*70)
    print(f"Generated {len(sequences)} sequences")
    print(f"Output directory: {output_dir}")
    print("="*70)

    return sequences

def main():
    """Main generation function."""
    parser = argparse.ArgumentParser(description='Generate music from trained models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['lstm', 'gru', 'vae', 'gan'],
                       help='Model type to use for generation')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of sequences to generate')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for MIDI files')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)

    # Generate music
    sequences = generate_music(
        config=config,
        model_type=args.model,
        model_path=args.model_path,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
