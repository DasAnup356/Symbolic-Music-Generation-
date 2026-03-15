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

    def __init__(self, note_range=(21, 108), max_length=512, resolution=480, instrument_bins=16):
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
        self.instrument_bins = instrument_bins
        self.note_vocab_size = note_range[1] - note_range[0] + 1
        self.vocab_size = self.note_vocab_size * self.instrument_bins

    def _program_to_bin(self, program):
        """Map 0..127 program numbers to a smaller instrument-bin vocabulary."""
        return min(max(int(program), 0) // max(1, 128 // self.instrument_bins), self.instrument_bins - 1)

    def _encode_token(self, note_idx, program):
        return self._program_to_bin(program) * self.note_vocab_size + int(note_idx)

    def decode_token(self, token):
        """Decode token back to (note_idx, program)."""
        token = int(token)
        note_idx = token % self.note_vocab_size
        instr_bin = token // self.note_vocab_size
        program = int(instr_bin * (128 // self.instrument_bins))
        return note_idx, min(program, 127)

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

            notes, durations, velocities, time_shifts, instruments = [], [], [], [], []

            # Parse tracks independently to avoid cross-track time corruption.
            for track in midi.tracks:
                current_time = 0
                current_program = 0
                active_notes = {}

                for msg in track:
                    current_time += msg.time

                    if msg.type == 'program_change':
                        current_program = int(msg.program)
                        continue

                    if msg.type == 'note_on' and msg.velocity > 0:
                        if self.note_range[0] <= msg.note <= self.note_range[1]:
                            key = (getattr(msg, 'channel', 0), msg.note)
                            active_notes[key] = {
                                'start_time': current_time,
                                'velocity': int(msg.velocity),
                                'program': current_program,
                            }

                    elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                        key = (getattr(msg, 'channel', 0), msg.note)
                        if key in active_notes:
                            note_info = active_notes.pop(key)
                            duration = max(1, current_time - note_info['start_time'])
                            notes.append(msg.note - self.note_range[0])
                            durations.append(min(duration, self.resolution * 4))
                            velocities.append(int(note_info['velocity']))
                            time_shifts.append(int(note_info['start_time']))
                            instruments.append(int(note_info['program']))

            if len(notes) == 0:
                return None

            order = np.argsort(time_shifts)
            notes = np.array(notes, dtype=np.int32)[order]
            durations = np.array(durations, dtype=np.int32)[order]
            velocities = np.array(velocities, dtype=np.int32)[order]
            time_shifts = np.array(time_shifts, dtype=np.int32)[order]
            instruments = np.array(instruments, dtype=np.int32)[order]

            tokens = np.array([self._encode_token(n, p) for n, p in zip(notes, instruments)], dtype=np.int32)

            return {
                'notes': notes,
                'durations': durations,
                'velocities': velocities,
                'time_shifts': time_shifts,
                'instruments': instruments,
                'tokens': tokens,
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
        instruments = sequence.get('instruments', [0] * len(notes))

        # Use one track per instrument program for richer accompaniment output.
        instrument_tracks = {}
        for note, duration, vel, time_shift, program in zip(notes, durations, velocities, time_shifts, instruments):
            note = int(np.clip(int(note) + self.note_range[0], 0, 127))
            duration = max(1, int(duration))
            vel = int(np.clip(int(vel), 1, 127))
            program = int(np.clip(int(program), 0, 127))

            if program not in instrument_tracks:
                tr = MidiTrack()
                midi.tracks.append(tr)
                channel = program % 16
                if channel == 9:
                    channel = 8
                tr.append(Message('program_change', channel=channel, program=program, time=0))
                instrument_tracks[program] = {
                    'track': tr,
                    'channel': channel,
                    'events': [],
                }

            instrument_tracks[program]['events'].append(('note_on', int(time_shift), note, vel))
            instrument_tracks[program]['events'].append(('note_off', int(time_shift + duration), note, 0))

        for data in instrument_tracks.values():
            events = sorted(data['events'], key=lambda x: x[1])
            last_time = 0
            for event_type, time, note, vel in events:
                delta_time = max(0, int(time - last_time))
                data['track'].append(
                    Message(event_type, channel=data['channel'], note=note, velocity=vel, time=delta_time)
                )
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
            if 'instruments' in sequence:
                seq_data['instruments'] = sequence['instruments'][i:i+seq_length]
            if 'tokens' in sequence:
                seq_data['tokens'] = sequence['tokens'][i:i+seq_length]
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
                train_seqs = self.create_training_sequences(
                    sequence,
                    seq_length=self.max_length
                )
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
                    'vocab_size': self.vocab_size,
                    'note_vocab_size': self.note_vocab_size,
                    'instrument_bins': self.instrument_bins,
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
        notes = np.asarray(seq.get('tokens', seq['notes']), dtype=np.int32)
        if len(notes) < 2:
            continue

        current_seq_len = min(seq_length, len(notes) - 1)
        input_notes = notes[:current_seq_len]
        target_note = notes[current_seq_len]

        if input_notes.shape[0] < seq_length:
            padded = np.zeros(seq_length, dtype=np.int32)
            padded[-input_notes.shape[0]:] = input_notes
            input_notes = padded

        if 0 <= target_note < vocab_size:
            X.append(input_notes)
            y.append(int(target_note))

    if not X:
        raise ValueError(
            "No training samples could be created from the processed sequences. "
            "Try reducing `data.midi_processing.max_length` in config.yaml or "
            "regenerate processed data."
        )

    X = np.asarray(X, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)

    return X, y
