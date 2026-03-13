"""
Quick Start Notebook for Symbolic Music Generation

This notebook demonstrates the complete workflow for the project.
"""

# Cell 1: Setup and Imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add project to path
sys.path.append('..')

from utils.config_loader import get_config
from utils.midi_processor import MIDIProcessor
from models.lstm.lstm_model import LSTMMusicGenerator

print("✓ Imports successful")

# Cell 2: Load Configuration
config = get_config('../config.yaml')

print("Configuration loaded:")
print(f"  Dataset size: {config.get('data', 'dataset_size')}")
print(f"  LSTM layers: {config.get('models', 'lstm', 'layers')}")
print(f"  LSTM units: {config.get('models', 'lstm', 'units')}")
print(f"  Target accuracy: {config.get('training', 'target_metrics', 'validation_accuracy')}")

# Cell 3: Create Sample MIDI Files
from preprocessing.preprocess import download_sample_dataset

midi_dir = '../data/midi_files'
os.makedirs(midi_dir, exist_ok=True)

print("Creating sample MIDI dataset...")
download_sample_dataset(midi_dir, num_samples=100)
print("✓ Sample dataset created")

# Cell 4: Preprocess Data
from preprocessing.preprocess import preprocess_dataset

print("Preprocessing MIDI files...")
sequences = preprocess_dataset(
    config=config,
    max_files=100
)
print(f"✓ Generated {len(sequences)} training sequences")

# Cell 5: Prepare Training Data
import pickle
from utils.midi_processor import prepare_training_data

data_path = '../data/processed/sequences.pkl'
with open(data_path, 'rb') as f:
    data_dict = pickle.load(f)

sequences = data_dict['sequences']
vocab_size = data_dict['config']['vocab_size']

print(f"Vocabulary size: {vocab_size}")
print(f"Total sequences: {len(sequences)}")

# Prepare training data
X, y = prepare_training_data(sequences, vocab_size, seq_length=128)

# Split data
n_train = int(len(X) * 0.8)
n_val = int(len(X) * 0.1)

X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Cell 6: Build and Train LSTM Model
from models.lstm.lstm_model import LSTMMusicGenerator, create_callbacks

print("Building 3-layer LSTM model (512 units each)...")
model = LSTMMusicGenerator(
    vocab_size=vocab_size,
    seq_length=128,
    embedding_dim=256
)

model.compile_model(learning_rate=0.001)
model.summary()

# Create callbacks
checkpoint_path = '../outputs/checkpoints/lstm_notebook.h5'
log_dir = '../outputs/logs/lstm_notebook'
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

callbacks = create_callbacks(checkpoint_path, log_dir)

# Train model (reduced epochs for demo)
print("\nTraining model...")
history = model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=10,  # Reduced for demo
    batch_size=64,
    callbacks=callbacks
)

print("✓ Training complete")

# Cell 7: Plot Training History
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('../outputs/training_history.png', dpi=150)
plt.show()

print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

# Cell 8: Generate Music
print("Generating music sequences...")

# Create seed sequence
seed_sequence = np.random.randint(0, vocab_size, size=128)

# Generate 10 sequences
generated_sequences = []
for i in range(10):
    sequence = model.generate_sequence(
        seed_sequence=seed_sequence,
        length=256,
        temperature=1.0
    )
    generated_sequences.append(sequence)
    print(f"Generated sequence {i+1}/10")

print("✓ Generation complete")

# Cell 9: Save Generated MIDI Files
from utils.midi_processor import MIDIProcessor

processor = MIDIProcessor()
output_dir = '../outputs/generated_midi/notebook'
os.makedirs(output_dir, exist_ok=True)

print(f"Saving {len(generated_sequences)} MIDI files...")
for i, sequence in enumerate(generated_sequences):
    seq_data = {
        'notes': sequence,
        'durations': np.full(len(sequence), 480),
        'velocities': np.full(len(sequence), 80),
        'time_shifts': np.arange(len(sequence)) * 480
    }

    output_path = os.path.join(output_dir, f'generated_{i:04d}.mid')
    processor.sequence_to_midi(seq_data, output_path, tempo=120)

print(f"✓ Saved MIDI files to {output_dir}")

# Cell 10: Evaluate Generated Music
from evaluation.evaluate import MusicEvaluator

evaluator = MusicEvaluator()
stats = evaluator.evaluate_sequences(generated_sequences)
evaluator.print_evaluation(stats)

# Cell 11: Visualize Generated Sequence
plt.figure(figsize=(14, 6))

# Plot first generated sequence
sequence = generated_sequences[0]
plt.subplot(2, 1, 1)
plt.plot(sequence, linewidth=0.5)
plt.xlabel('Time Step')
plt.ylabel('MIDI Note')
plt.title('Generated Melody - Note Sequence')
plt.grid(True, alpha=0.3)

# Plot pitch class histogram
plt.subplot(2, 1, 2)
pitch_classes = sequence % 12
plt.hist(pitch_classes, bins=12, alpha=0.7, edgecolor='black')
plt.xlabel('Pitch Class')
plt.ylabel('Frequency')
plt.title('Pitch Class Distribution')
plt.xticks(range(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/generated_visualization.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("NOTEBOOK COMPLETE")
print("="*60)
print("✓ Model trained successfully")
print("✓ Music generated and saved")
print("✓ Evaluation complete")
print("="*60)
