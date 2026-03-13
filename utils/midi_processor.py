"""MIDI file processing utilities for music generation."""

import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import pickle
from pathlib import Path
from tqdm import tqdm
import os

class MIDIProcessor:
    """Process MIDI files for deep learning models."""

    def __init__(self, note_range=(21, 108), max_length=512, resolution=480):
        """
        Initialize MIDI processor.

        Args:
            note_range: Tuple of (min_note, max_note) MIDI values
            max_length: Maximum sequence length
            resolution: MIDI ticks per quarter note
        """
        self.note_range = note_range
        self.max_length = max_length
        self.resolution = resolution
        self.note_vocab_size = note_range[1] - note_range[0] + 1

    def midi_to_sequence(self, midi_path):
        """
        Convert MIDI file to sequence representation.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Dictionary with 'notes', 'durations', 'velocities', 'time_shifts'
        """
        try:
            midi = MidiFile(midi_path)

            # Collect all note events
            notes = []
            durations = []
            velocities = []
            time_shifts = []

            current_time = 0
            active_notes = {}

            for track in midi.tracks:
                for msg in track:
                    current_time += msg.time

                    if msg.type == 'note_on' and msg.velocity > 0:
                        # Note start
                        if self.note_range[0] <= msg.note <= self.note_range[1]:
                            active_notes[msg.note] = {
                                'start_time': current_time,
                                'velocity': msg.velocity
                            }

                    elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                        # Note end
                        if msg.note in active_notes:
                            note_info = active_notes.pop(msg.note)
                            duration = current_time - note_info['start_time']

                            # Store note information
                            notes.append(msg.note - self.note_range[0])  # Normalize
                            durations.append(min(duration, self.resolution * 4))  # Cap duration
                            velocities.append(note_info['velocity'])
                            time_shifts.append(note_info['start_time'])

            if len(notes) == 0:
                return None

            return {
                'notes': np.array(notes),
                'durations': np.array(durations),
                'velocities': np.array(velocities),
                'time_shifts': np.array(time_shifts)
            }

        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return None

    def sequence_to_midi(self, sequence, output_path, tempo=120, velocity=80):
        """
        Convert sequence representation back to MIDI file.

        Args:
            sequence: Dictionary with note information
            output_path: Path to save MIDI file
            tempo: Tempo in BPM
            velocity: Default velocity if not in sequence
        """
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)

        # Set tempo
        tempo_value = mido.bpm2tempo(tempo)
        track.append(MetaMessage('set_tempo', tempo=tempo_value, time=0))

        # Add notes
        notes = sequence.get('notes', [])
        durations = sequence.get('durations', [self.resolution] * len(notes))
        velocities = sequence.get('velocities', [velocity] * len(notes))
        time_shifts = sequence.get('time_shifts', list(range(0, len(notes) * self.resolution, self.resolution)))

        # Create note events
        events = []
        for i, (note, duration, vel, time_shift) in enumerate(zip(notes, durations, velocities, time_shifts)):
            note = int(note + self.note_range[0])  # Denormalize
            events.append(('note_on', int(time_shift), note, int(vel)))
            events.append(('note_off', int(time_shift + duration), note, 0))

        # Sort events by time
        events.sort(key=lambda x: x[1])

        # Convert to MIDI messages
        last_time = 0
        for event_type, time, note, vel in events:
            delta_time = time - last_time
            if event_type == 'note_on':
                track.append(Message('note_on', note=note, velocity=vel, time=delta_time))
            else:
                track.append(Message('note_off', note=note, velocity=vel, time=delta_time))
            last_time = time

        # Save MIDI file
        midi.save(output_path)

    def create_training_sequences(self, sequence, seq_length=128, step=32):
        """
        Create overlapping training sequences from a full sequence.

        Args:
            sequence: Dictionary with note information
            seq_length: Length of each training sequence
            step: Step size for sliding window

        Returns:
            List of training sequences
        """
        notes = sequence['notes']
        if len(notes) < seq_length:
            return []

        sequences = []
        for i in range(0, len(notes) - seq_length, step):
            seq_data = {
                'notes': notes[i:i+seq_length],
                'durations': sequence['durations'][i:i+seq_length],
                'velocities': sequence['velocities'][i:i+seq_length],
                'time_shifts': sequence['time_shifts'][i:i+seq_length]
            }
            sequences.append(seq_data)

        return sequences

    def process_dataset(self, midi_dir, output_path, max_files=None):
        """
        Process entire dataset of MIDI files.

        Args:
            midi_dir: Directory containing MIDI files
            output_path: Path to save processed data
            max_files: Maximum number of files to process

        Returns:
            Dictionary with processed sequences
        """
        midi_files = list(Path(midi_dir).rglob("*.mid")) + list(Path(midi_dir).rglob("*.midi"))

        if max_files:
            midi_files = midi_files[:max_files]

        all_sequences = []
        successful = 0

        print(f"Processing {len(midi_files)} MIDI files...")
        for midi_file in tqdm(midi_files):
            sequence = self.midi_to_sequence(str(midi_file))
            if sequence is not None:
                # Create training sequences
                train_seqs = self.create_training_sequences(sequence)
                all_sequences.extend(train_seqs)
                successful += 1

        print(f"\nSuccessfully processed: {successful}/{len(midi_files)} files")
        print(f"Generated {len(all_sequences)} training sequences")

        # Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({
                'sequences': all_sequences,
                'config': {
                    'note_range': self.note_range,
                    'max_length': self.max_length,
                    'resolution': self.resolution,
                    'vocab_size': self.note_vocab_size
                }
            }, f)

        print(f"Saved processed data to {output_path}")
        return all_sequences

    def load_processed_data(self, data_path):
        """Load processed data from file."""
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data

def prepare_training_data(sequences, vocab_size, seq_length=128):
    """
    Prepare sequences for training.

    Args:
        sequences: List of sequence dictionaries
        vocab_size: Size of note vocabulary
        seq_length: Sequence length

    Returns:
        X, y as numpy arrays
    """
    X = []
    y = []

    for seq in sequences:
        notes = seq['notes']
        if len(notes) >= seq_length + 1:
            X.append(notes[:seq_length])
            y.append(notes[seq_length])

    X = np.array(X)
    y = np.array(y)

    # Convert to one-hot encoding
    X_onehot = np.zeros((len(X), seq_length, vocab_size))
    y_onehot = np.zeros((len(y), vocab_size))

    for i, seq in enumerate(X):
        for j, note in enumerate(seq):
            if 0 <= note < vocab_size:
                X_onehot[i, j, int(note)] = 1

    for i, note in enumerate(y):
        if 0 <= note < vocab_size:
            y_onehot[i, int(note)] = 1

    return X_onehot, y_onehot
