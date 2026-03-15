"""LSTM model for symbolic music generation."""

from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np


class LSTMMusicGenerator:
    """Configurable LSTM model for music generation."""

    def __init__(
        self,
        vocab_size,
        seq_length=128,
        embedding_dim=128,
        num_layers=2,
        units=256,
        dropout=0.2,
        recurrent_dropout=0.0,
        dense_units=(256, 128),
    ):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.dense_units = dense_units
        self.model = self._build_model()

    def _build_model(self):
        """Build configurable multi-layer LSTM model."""
        model = models.Sequential()
        model.add(layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim))

        for i in range(self.num_layers):
            model.add(
                layers.LSTM(
                    self.units,
                    return_sequences=(i < self.num_layers - 1),
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout,
                    name=f"lstm_layer_{i+1}",
                )
            )
            model.add(layers.BatchNormalization())

        for dense_u in self.dense_units:
            model.add(layers.Dense(dense_u, activation="relu"))
            model.add(layers.Dropout(self.dropout))

        model.add(layers.Dense(self.vocab_size, activation="softmax"))
        return model

    def compile_model(self, learning_rate=0.001):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")],
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, callbacks=None):
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def generate_sequence(self, seed_sequence, length=256, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0):
        generated = list(seed_sequence)
        for _ in range(length):
            x = np.array(generated[-self.seq_length:]).reshape(1, -1)
            predictions = self.model.predict(x, verbose=0)[0]
            next_note = self._sample_with_controls(
                predictions,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                recent_notes=generated[-8:],
            )
            generated.append(next_note)
        return np.array(generated)


    def generate_sequences(self, seed_sequences, length=256, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0):
        """Generate multiple sequences in a vectorized batch for speed."""
        generated = np.array(seed_sequences, dtype=np.int32)

        for _ in range(length):
            x = generated[:, -self.seq_length:]
            predictions = self.model.predict(x, verbose=0, batch_size=len(x))
            next_notes = [
                self._sample_with_controls(
                    predictions[i],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    recent_notes=generated[i, -8:],
                )
                for i in range(predictions.shape[0])
            ]
            next_notes = np.array(next_notes, dtype=np.int32).reshape(-1, 1)
            generated = np.concatenate([generated, next_notes], axis=1)

        return generated


    def _sample_with_controls(self, probs, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, recent_notes=None):
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.log(probs + 1e-10) / max(temperature, 1e-5)
        probs = np.exp(probs - np.max(probs))
        probs = probs / np.sum(probs)

        if recent_notes is not None and repetition_penalty > 1.0:
            for n in recent_notes:
                probs[int(n)] /= repetition_penalty
            probs = probs / np.sum(probs)

        if top_k and top_k > 0:
            top_idx = np.argpartition(probs, -top_k)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_idx] = probs[top_idx]
            probs = mask / np.sum(mask)

        if top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_idx]
            cumulative = np.cumsum(sorted_probs)
            cutoff = cumulative > top_p
            if np.any(cutoff):
                first_idx = np.argmax(cutoff)
                sorted_probs[first_idx + 1:] = 0.0
                mask = np.zeros_like(probs)
                mask[sorted_idx] = sorted_probs
                probs = mask / np.sum(mask)

        return np.random.choice(self.vocab_size, p=probs)

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def summary(self):
        return self.model.summary()


def create_callbacks(checkpoint_path, log_dir):
    """Create training callbacks."""
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]
    return callbacks
