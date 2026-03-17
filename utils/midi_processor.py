"""MIDI file processing utilities for music generation."""

import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import pickle
from pathlib import Path
from tqdm import tqdm
import os

class MIDIProcessor:
    """Process MIDI files for deep learning models using Performance Encoding."""

    def __init__(self, note_range=(21, 108), max_length=1024, resolution=480, 
                 velocity_bins=32, time_shift_bins=100, max_shift_ms=1000):
        """
        Initialize MIDI processor with Performance Encoding.
        """
        self.note_range = note_range
        self.max_length = max_length
        self.resolution = resolution
        self.velocity_bins = velocity_bins
        self.time_shift_bins = time_shift_bins
        self.max_shift_ms = max_shift_ms

        # Token offsets
        self.NOTE_ON_OFFSET = 0
        self.NOTE_OFF_OFFSET = 128
        self.TIME_SHIFT_OFFSET = 256
        self.VELOCITY_OFFSET = 256 + time_shift_bins
        self.INSTRUMENT_OFFSET = 256 + time_shift_bins + velocity_bins
        
        self.vocab_size = self.INSTRUMENT_OFFSET + 128

    def _get_velocity_bin(self, velocity):
        return min(int(velocity) * self.velocity_bins // 128, self.velocity_bins - 1)

    def _get_time_shift_bin(self, ticks):
        # Convert ticks to ms approximately (assuming 120 BPM)
        # ms = ticks * (60000 / (BPM * resolution))
        ms = ticks * (60000 / (120 * self.resolution))
        bin_idx = min(int(ms * self.time_shift_bins / self.max_shift_ms), self.time_shift_bins - 1)
        return max(0, bin_idx)

    def midi_to_sequence(self, midi_path):
        """Convert MIDI to a sequence of Performance tokens."""
        try:
            midi = MidiFile(midi_path)
            events = []
            
            for track in midi.tracks:
                current_tick = 0
                current_program = 0
                for msg in track:
                    current_tick += msg.time
                    if msg.type == 'program_change':
                        current_program = msg.program
                    
                    if msg.type in ['note_on', 'note_off']:
                        events.append({
                            'tick': current_tick,
                            'type': msg.type if msg.velocity > 0 else 'note_off',
                            'note': msg.note,
                            'velocity': msg.velocity,
                            'program': current_program
                        })
            
            if not events:
                return None
                
            # Sort events by time
            events.sort(key=lambda x: x['tick'])
            
            tokens = []
            last_tick = 0
            last_velocity_bin = -1
            last_program = -1
            
            for event in events:
                # 1. Time Shift
                delta_ticks = event['tick'] - last_tick
                if delta_ticks > 0:
                    shift_bin = self._get_time_shift_bin(delta_ticks)
                    tokens.append(self.TIME_SHIFT_OFFSET + shift_bin)
                    last_tick = event['tick']
                
                # 2. Instrument Change
                if event['program'] != last_program:
                    tokens.append(self.INSTRUMENT_OFFSET + event['program'])
                    last_program = event['program']
                
                # 3. Velocity Change (only for note_on)
                if event['type'] == 'note_on':
                    v_bin = self._get_velocity_bin(event['velocity'])
                    if v_bin != last_velocity_bin:
                        tokens.append(self.VELOCITY_OFFSET + v_bin)
                        last_velocity_bin = v_bin
                    
                    # 4. Note On
                    tokens.append(self.NOTE_ON_OFFSET + event['note'])
                else:
                    # 5. Note Off
                    tokens.append(self.NOTE_OFF_OFFSET + event['note'])
                
                if len(tokens) >= self.max_length * 2: # Buffer for processing
                    break
                    
            return {'tokens': np.array(tokens, dtype=np.int32)}

        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return None

    def sequence_to_midi(self, sequence, output_path, tempo=120):
        """Convert Performance tokens back to MIDI."""
        midi = MidiFile()
        # Create a track for each instrument used
        instrument_tracks = {}
        
        tokens = sequence.get('tokens', [])
        current_tick = 0
        current_velocity = 80
        current_program = 0
        
        # Track active notes to ensure they are closed
        active_notes = {} # (program, note) -> start_tick
        
        for token in tokens:
            if self.NOTE_ON_OFFSET <= token < self.NOTE_OFF_OFFSET:
                note = token - self.NOTE_ON_OFFSET
                if current_program not in instrument_tracks:
                    track = MidiTrack()
                    midi.tracks.append(track)
                    track.append(Message('program_change', program=current_program, time=0))
                    instrument_tracks[current_program] = {'track': track, 'events': [], 'last_tick': 0}
                
                instrument_tracks[current_program]['events'].append(
                    Message('note_on', note=note, velocity=current_velocity, time=current_tick)
                )
                active_notes[(current_program, note)] = current_tick
                
            elif self.NOTE_OFF_OFFSET <= token < self.TIME_SHIFT_OFFSET:
                note = token - self.NOTE_OFF_OFFSET
                if (current_program, note) in active_notes:
                    instrument_tracks[current_program]['events'].append(
                        Message('note_off', note=note, velocity=0, time=current_tick)
                    )
                    del active_notes[(current_program, note)]
                    
            elif self.TIME_SHIFT_OFFSET <= token < self.VELOCITY_OFFSET:
                bin_idx = token - self.TIME_SHIFT_OFFSET
                # Reverse _get_time_shift_bin: ms = bin * max_shift / bins
                ms = (bin_idx + 0.5) * self.max_shift_ms / self.time_shift_bins
                ticks = int(ms * (tempo * self.resolution) / 60000)
                current_tick += ticks
                
            elif self.VELOCITY_OFFSET <= token < self.INSTRUMENT_OFFSET:
                v_bin = token - self.VELOCITY_OFFSET
                current_velocity = int((v_bin + 0.5) * 128 / self.velocity_bins)
                
            elif self.INSTRUMENT_OFFSET <= token < self.vocab_size:
                current_program = token - self.INSTRUMENT_OFFSET
        
        # Close any remaining active notes
        for (prog, note), start_tick in active_notes.items():
            if prog in instrument_tracks:
                instrument_tracks[prog]['events'].append(
                    Message('note_off', note=note, velocity=0, time=current_tick)
                )

        # Finalize tracks
        for prog, data in instrument_tracks.items():
            track = data['track']
            events = data['events']
            events.sort(key=lambda x: x.time)
            
            last_t = 0
            for msg in events:
                delta = msg.time - last_t
                msg.time = max(0, delta)
                track.append(msg)
                last_t += msg.time
                
        midi.save(output_path)

    def create_training_sequences(self, sequence, seq_length=128, step=32):
        tokens = sequence.get('tokens', [])
        if len(tokens) <= seq_length:
            return []
            
        sequences = []
        for i in range(0, len(tokens) - seq_length, step):
            sequences.append({'tokens': tokens[i:i+seq_length+1]})
        return sequences

    def process_dataset(self, midi_dir, output_path, max_files=None):
        midi_files = list(Path(midi_dir).rglob("*.mid")) + list(Path(midi_dir).rglob("*.midi"))
        if max_files: midi_files = midi_files[:max_files]
        
        all_sequences = []
        print(f"Processing {len(midi_files)} MIDI files with Performance Encoding...")
        for midi_file in tqdm(midi_files):
            seq = self.midi_to_sequence(str(midi_file))
            if seq:
                all_sequences.extend(self.create_training_sequences(seq, seq_length=128)) # Using default or config
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({
                'sequences': all_sequences,
                'config': {
                    'vocab_size': self.vocab_size,
                    'max_length': self.max_length
                }
            }, f)
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
        # Check for 'tokens' (new format) or 'notes' (legacy format)
        if 'tokens' in seq:
            notes = np.asarray(seq['tokens'], dtype=np.int32)
        elif 'notes' in seq:
            notes = np.asarray(seq['notes'], dtype=np.int32)
        else:
            continue
            
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
