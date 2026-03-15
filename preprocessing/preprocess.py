"""Data preprocessing pipeline for MIDI files."""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import get_config
from utils.midi_processor import MIDIProcessor

def download_sample_dataset(output_dir, num_samples=100):
    """
    Download or create sample MIDI dataset for testing.

    Args:
        output_dir: Directory to save MIDI files
        num_samples: Number of sample files to create
    """
    print(f"Creating {num_samples} sample MIDI files for testing...")

    os.makedirs(output_dir, exist_ok=True)

    processor = MIDIProcessor()

    # Create simple test melodies
    for i in tqdm(range(num_samples)):
        # Generate random melody
        notes = np.random.randint(21, 108, size=np.random.randint(50, 200))
        durations = np.random.choice([240, 480, 960], size=len(notes))
        velocities = np.random.randint(60, 100, size=len(notes))
        time_shifts = np.cumsum(durations)

        sequence = {
            'notes': notes - 21,  # Normalize
            'durations': durations,
            'velocities': velocities,
            'time_shifts': time_shifts
        }

        output_path = os.path.join(output_dir, f"sample_{i:04d}.mid")
        processor.sequence_to_midi(sequence, output_path)

    print(f"✓ Created {num_samples} sample MIDI files in {output_dir}")

def preprocess_dataset(config, midi_dir=None, output_path=None, max_files=None):
    """
    Preprocess MIDI dataset.

    Args:
        config: Configuration dictionary
        midi_dir: Directory containing MIDI files
        output_path: Path to save processed data
        max_files: Maximum number of files to process
    """
    if midi_dir is None:
        midi_dir = config.get('data', 'midi_dir')

    if output_path is None:
        output_path = os.path.join(config.get('data', 'processed_dir'), 'sequences.pkl')

    if max_files is None:
        max_files = config.get('data', 'dataset_size')

    print("\n" + "="*70)
    print("MIDI Data Preprocessing Pipeline")
    print("="*70)
    print(f"Source directory: {midi_dir}")
    print(f"Output path: {output_path}")
    print(f"Target dataset size: {max_files} files")
    print("="*70)

    # Initialize processor
    midi_config = config.get('data', 'midi_processing')
    processor = MIDIProcessor(
        note_range=tuple(config.get('data', 'representation', 'note_range')),
        max_length=midi_config.get('max_length'),
        resolution=midi_config.get('resolution'),
        instrument_bins=config.get('data', 'representation', 'instrument_bins', default=16),
    )

    # Check if MIDI directory exists
    if not os.path.exists(midi_dir):
        print(f"\n⚠ MIDI directory not found: {midi_dir}")
        print("Creating sample dataset for testing...")
        os.makedirs(midi_dir, exist_ok=True)
        download_sample_dataset(midi_dir, num_samples=min(100, max_files))

    # Process dataset
    sequences = processor.process_dataset(
        midi_dir=midi_dir,
        output_path=output_path,
        max_files=max_files
    )

    print("\n" + "="*70)
    print("Preprocessing Complete!")
    print("="*70)
    print(f"Processed data saved to: {output_path}")
    print(f"Total training sequences: {len(sequences)}")
    print("="*70)

    return sequences

def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess MIDI dataset')
    parser.add_argument('--midi-dir', type=str, default=None,
                       help='Directory containing MIDI files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for processed data')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--create-samples', action='store_true',
                       help='Create sample MIDI files for testing')

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)

    # Create output directories
    os.makedirs(config.get('data', 'processed_dir'), exist_ok=True)

    if args.create_samples:
        midi_dir = args.midi_dir or config.get('data', 'midi_dir')
        os.makedirs(midi_dir, exist_ok=True)
        download_sample_dataset(midi_dir, num_samples=args.max_files or 100)
    else:
        # Preprocess dataset
        sequences = preprocess_dataset(
            config=config,
            midi_dir=args.midi_dir,
            output_path=args.output,
            max_files=args.max_files
        )

if __name__ == '__main__':
    main()
