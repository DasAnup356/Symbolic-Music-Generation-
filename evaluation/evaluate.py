"""Evaluation metrics for generated music."""

import os
import sys
import numpy as np
import argparse
from collections import Counter
from pathlib import Path
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import get_config
from utils.midi_processor import MIDIProcessor

class MusicEvaluator:
    """Evaluate quality of generated music."""

    def __init__(self):
        self.metrics = {}

    def note_density(self, sequence):
        """Calculate note density (notes per time unit)."""
        return len(sequence) / (len(sequence) + 1)

    def pitch_range(self, sequence):
        """Calculate pitch range (max - min)."""
        return np.max(sequence) - np.min(sequence)

    def pitch_class_entropy(self, sequence):
        """Calculate entropy of pitch class distribution."""
        pitch_classes = sequence % 12
        counts = Counter(pitch_classes)
        total = len(sequence)

        entropy = 0
        for count in counts.values():
            p = count / total
            entropy -= p * np.log2(p + 1e-10)

        return entropy

    def note_transition_matrix(self, sequence):
        """Calculate note transition probabilities."""
        vocab_size = int(np.max(sequence)) + 1
        transitions = np.zeros((vocab_size, vocab_size))

        for i in range(len(sequence) - 1):
            current = int(sequence[i])
            next_note = int(sequence[i + 1])
            transitions[current, next_note] += 1

        # Normalize
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transitions = transitions / row_sums

        return transitions

    def evaluate_sequences(self, sequences):
        """Evaluate a set of sequences."""
        results = {
            'note_density': [],
            'pitch_range': [],
            'pitch_class_entropy': [],
            'sequence_length': []
        }

        for seq in sequences:
            if len(seq) == 0:
                continue
            results['note_density'].append(self.note_density(seq))
            results['pitch_range'].append(self.pitch_range(seq))
            results['pitch_class_entropy'].append(self.pitch_class_entropy(seq))
            results['sequence_length'].append(len(seq))

        # Compute statistics
        stats = {}
        if not results['sequence_length']:
            print("Warning: No non-empty sequences found for evaluation.")
            return {}
        for metric, values in results.items():
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        return stats

    def print_evaluation(self, stats):
        """Print evaluation results."""
        print("\n" + "="*70)
        print("Evaluation Results")
        print("="*70)

        for metric, values in stats.items():
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Mean: {values['mean']:.4f}")
            print(f"  Std:  {values['std']:.4f}")
            print(f"  Min:  {values['min']:.4f}")
            print(f"  Max:  {values['max']:.4f}")

        print("\n" + "="*70)

def evaluate_generated_music(config, sequences_path=None, midi_dir=None):
    """
    Evaluate generated music sequences.

    Args:
        config: Configuration dictionary
        sequences_path: Path to pickled sequences (optional)
        midi_dir: Directory containing MIDI files (optional)
    """
    evaluator = MusicEvaluator()

    # Load sequences
    if sequences_path:
        print(f"Loading sequences from {sequences_path}...")
        with open(sequences_path, 'rb') as f:
            sequences = pickle.load(f)
    elif midi_dir:
        print(f"Loading MIDI files from {midi_dir}...")
        processor = MIDIProcessor()
        sequences = []

        for midi_file in Path(midi_dir).glob("*.mid"):
            seq_data = processor.midi_to_sequence(str(midi_file))
            if seq_data is not None:
                # Support both 'tokens' (new) and 'notes' (legacy)
                tokens = seq_data.get('tokens', seq_data.get('notes', []))
                sequences.append(tokens)
    else:
        raise ValueError("Either sequences_path or midi_dir must be provided")

    print(f"Evaluating {len(sequences)} sequences...")

    # Evaluate
    stats = evaluator.evaluate_sequences(sequences)
    evaluator.print_evaluation(stats)

    return stats

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate generated music')
    parser.add_argument('--sequences', type=str, default=None,
                       help='Path to pickled sequences')
    parser.add_argument('--midi-dir', type=str, default=None,
                       help='Directory containing MIDI files')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')

    args = parser.parse_args()

    config = get_config(args.config)

    stats = evaluate_generated_music(
        config=config,
        sequences_path=args.sequences,
        midi_dir=args.midi_dir
    )

if __name__ == '__main__':
    main()
