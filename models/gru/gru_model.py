"""GRU model for symbolic music generation."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

class GRUMusicGenerator:
    """3-layer GRU model with 512 units for music generation."""

    def __init__(self, vocab_size, seq_length=128, embedding_dim=256):
        """
        Initialize GRU music generator.

        Args:
            vocab_size: Size of note vocabulary
            seq_length: Length of input sequences
            embedding_dim: Dimension of embedding layer
        """
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.model = self._build_model()

    def _build_model(self):
        """Build 3-layer GRU model with 512 units each."""
        model = models.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,

            ),

            # First GRU layer - 512 units
            layers.GRU(
                512,
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.2,
                name='gru_layer_1'
            ),
            layers.BatchNormalization(),

            # Second GRU layer - 512 units
            layers.GRU(
                512,
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.2,
                name='gru_layer_2'
            ),
            layers.BatchNormalization(),

            # Third GRU layer - 512 units
            layers.GRU(
                512,
                return_sequences=False,
                dropout=0.3,
                recurrent_dropout=0.2,
                name='gru_layer_3'
            ),
            layers.BatchNormalization(),

            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),

            # Output layer
            layers.Dense(self.vocab_size, activation='softmax')
        ])

        return model

    def compile_model(self, learning_rate=0.001):
        """Compile the model."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, callbacks=None):
        """Train the model."""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def generate_sequence(self, seed_sequence, length=256, temperature=1.0):
        """Generate a sequence of notes."""
        generated = list(seed_sequence)

        for _ in range(length):
            x = np.array(generated[-self.seq_length:]).reshape(1, -1)
            predictions = self.model.predict(x, verbose=0)[0]

            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))

            next_note = np.random.choice(self.vocab_size, p=predictions)
            generated.append(next_note)

        return np.array(generated)

    def save_model(self, filepath):
        """Save model to file."""
        self.model.save(filepath)

    def load_model(self, filepath):
        """Load model from file."""
        self.model = keras.models.load_model(filepath)

    def summary(self):
        """Print model summary."""
        return self.model.summary()
