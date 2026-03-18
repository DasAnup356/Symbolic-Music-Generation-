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
    """Load trained model."""
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


def pick_seed_sequences(processed_data, seq_length, num_samples, vocab_size):
    """Prefer real seed patterns from training data over random tokens."""
    sequences = processed_data.get('sequences', [])
    if sequences:
        seeds = []
        for _ in range(num_samples):
            seq_dict = sequences[np.random.randint(0, len(sequences))]
            seq = seq_dict.get('tokens', seq_dict['notes'])
            seq = np.asarray(seq, dtype=np.int32)
            if len(seq) >= seq_length:
                start = np.random.randint(0, len(seq) - seq_length + 1)
                seed = seq[start:start + seq_length]
            else:
                seed = np.zeros(seq_length, dtype=np.int32)
                seed[-len(seq):] = seq
            seeds.append(seed)
        return np.asarray(seeds, dtype=np.int32)

    return np.random.randint(0, vocab_size, size=(num_samples, seq_length), dtype=np.int32)


def generate_from_lstm_gru(model, seed_sequences, length, temperature, top_k, top_p, repetition_penalty):
    """Generate sequences from LSTM/GRU model (batched when supported)."""
    num_samples = len(seed_sequences)
    print(f"Generating {num_samples} sequences...")

    if hasattr(model, 'generate_sequences'):
        sig = inspect.signature(model.generate_sequences)
        kwargs = {'seed_sequences': seed_sequences, 'length': length, 'temperature': temperature}
        if 'top_k' in sig.parameters:
            kwargs['top_k'] = top_k
        if 'top_p' in sig.parameters:
            kwargs['top_p'] = top_p
        if 'repetition_penalty' in sig.parameters:
            kwargs['repetition_penalty'] = repetition_penalty

        generated = model.generate_sequences(**kwargs)
        return [generated[i] for i in range(generated.shape[0])]

    generated_sequences = []
    for seed in tqdm(seed_sequences):
        sig = inspect.signature(model.generate_sequence)
        kwargs = {'seed_sequence': seed, 'length': length, 'temperature': temperature}
        if 'top_k' in sig.parameters:
            kwargs['top_k'] = top_k
        if 'top_p' in sig.parameters:
            kwargs['top_p'] = top_p
        if 'repetition_penalty' in sig.parameters:
            kwargs['repetition_penalty'] = repetition_penalty

        sequence = model.generate_sequence(**kwargs)
        generated_sequences.append(sequence)

    return generated_sequences


def generate_from_vae(model, num_samples, length):
    """Generate sequences from VAE model."""
    print(f"Generating {num_samples} sequences from latent space...")

    generated_sequences = []
    for _ in tqdm(range(num_samples)):
        z = np.random.normal(0, 1, (1, model.latent_dim))
        generated = model.decode(z)
        sequence = np.argmax(generated[0], axis=-1)
        generated_sequences.append(sequence)

    return generated_sequences


def generate_from_gan(model, num_samples, length):
    """Generate sequences from GAN model."""
    print(f"Generating {num_samples} sequences from GAN...")

    batch_size = 32
    generated_sequences = []

    for i in tqdm(range(0, num_samples, batch_size)):
        current_batch = min(batch_size, num_samples - i)
        generated = model.generate(num_samples=current_batch)
        for j in range(current_batch):
            sequence = np.argmax(generated[j], axis=-1)
            generated_sequences.append(sequence)

    return generated_sequences


def save_sequences_as_midi(sequences, output_dir, processor, config):
    """Save generated sequences as MIDI files with decoded note+instrument tokens."""
    print(f"\nSaving {len(sequences)} MIDI files to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    tempo = config.get('generation', 'output', 'tempo')
    default_velocity = config.get('generation', 'output', 'velocity')

    duration_choices = np.array([
        processor.resolution // 2,
        processor.resolution,
        processor.resolution * 2,
        processor.resolution * 4,
    ], dtype=np.int32)
    duration_probs = np.array([0.20, 0.45, 0.25, 0.10], dtype=np.float64)

    for i, sequence in enumerate(tqdm(sequences)):
        notes = []
        instruments = []
        for token in sequence:
            note_idx, program = processor.decode_token(token)
            notes.append(note_idx)
            instruments.append(program)

        durations = np.random.choice(duration_choices, size=len(notes), p=duration_probs)
        velocities = np.random.randint(max(45, default_velocity - 20), min(127, default_velocity + 20), size=len(notes))
        step_sizes = np.maximum(1, (durations * 0.75).astype(np.int32))
        time_shifts = np.concatenate(([0], np.cumsum(step_sizes[:-1], dtype=np.int32))) if len(notes) else np.array([], dtype=np.int32)

        seq_data = {
            'notes': np.asarray(notes, dtype=np.int32),
            'durations': np.asarray(durations, dtype=np.int32),
            'velocities': np.asarray(velocities, dtype=np.int32),
            'time_shifts': np.asarray(time_shifts, dtype=np.int32),
            'instruments': np.asarray(instruments, dtype=np.int32),
        }

        output_path = os.path.join(output_dir, f"generated_{i:04d}.mid")
        processor.sequence_to_midi(seq_data, output_path, tempo=tempo)

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
    seq_length = config.get('generation', 'seed_length')

    print(f"Vocabulary size: {vocab_size}")
    print(f"Seed length: {seq_length}")

    model = load_model(model_type, model_path, vocab_size, seq_length, config)

    if num_samples is None:
        num_samples = config.get('generation', 'num_samples')

    length = config.get('generation', 'sequence_length')
    temperature = config.get('generation', 'temperature')
    top_k = config.get('generation', 'top_k')
    top_p = config.get('generation', 'top_p')
    repetition_penalty = config.get('generation', 'repetition_penalty', default=1.1)

    if model_type in ['lstm', 'gru']:
        seed_sequences = pick_seed_sequences(data, seq_length, num_samples, vocab_size)
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

    representation_config = config.require('data', 'representation')
    midi_processing_config = config.require('data', 'midi_processing')
    note_range = representation_config.get('note_range', [21, 108])
    if note_range is None:
        raise ValueError("Config key `data.representation.note_range` must be defined.")

    processor = MIDIProcessor(
        note_range=tuple(note_range),
        resolution=midi_processing_config.get('resolution', 480),
        instrument_bins=representation_config.get('instrument_bins', 16),
    )

    if output_dir is None:
        output_dir = os.path.join(config.get('paths', 'generated_midi'), model_type)

    save_sequences_as_midi(sequences, output_dir, processor, config)

    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    generated_lengths = [len(seq) for seq in sequences]
    if generated_lengths:
        print(f"Generated {len(sequences)} sequences")
        print(f"Sequence lengths: min={min(generated_lengths)}, max={max(generated_lengths)}, mean={np.mean(generated_lengths):.1f}")
    else:
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
