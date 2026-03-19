"""CNN model for symbolic music generation."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

class CNNMusicGenerator:
    """Convolutional Neural Network for music generation."""

    def __init__(self, vocab_size, seq_length=128, embedding_dim=128,
                 filters=(64, 128, 256), kernel_sizes=(3, 3, 3),
                 dropout=0.3):
        """
        Initialize CNN music generator.

        Args:
            vocab_size: Size of note vocabulary
            seq_length: Length of input sequences
            embedding_dim: Dimension of embedding layer
            filters: List of filters for each Conv1D layer
            kernel_sizes: List of kernel sizes for each Conv1D layer
            dropout: Dropout rate
        """
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.model = self._build_model()

    def _build_model(self):
        """Build the CNN model."""
        model = models.Sequential()
        model.add(layers.Embedding(input_dim=self.vocab_size,
                                 output_dim=self.embedding_dim,
                                 input_shape=(self.seq_length,)))

        for f, k in zip(self.filters, self.kernel_sizes):
            model.add(layers.Conv1D(filters=f, kernel_size=k, padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            model.add(layers.Dropout(self.dropout))
            model.add(layers.MaxPooling1D(pool_size=1, padding='same'))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(self.vocab_size, activation='softmax'))

        return model

    def compile_model(self, learning_rate=0.001):
        """Compile the CNN model."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, callbacks=None):
        """Train the CNN model."""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def generate_sequence(self, seed_sequence, length=256):
        """Generate a sequence using the CNN."""
        generated = list(seed_sequence)
        for _ in range(length):
            x = np.array(generated[-self.seq_length:]).reshape(1, -1)
            predictions = self.model.predict(x, verbose=0)[0]
            next_note = np.random.choice(self.vocab_size, p=predictions)
            generated.append(next_note)
        return np.array(generated)

    def save_model(self, filepath):
        """Save the CNN model."""
        self.model.save(filepath)

    def load_model(self, filepath):
        """Load the CNN model."""
        self.model = keras.models.load_model(filepath)

    def summary(self):
        """Print model summary."""
        return self.model.summary()
