"""Enhanced LSTM model with Attention for symbolic music generation."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, seq_len, units)
        et = tf.squeeze(tf.tanh(tf.matmul(x, self.W) + self.b), axis=-1)
        at = tf.nn.softmax(et)
        at = tf.expand_dims(at, axis=-1)
        output = x * at
        return tf.reduce_sum(output, axis=1)

    def get_config(self):
        return super(AttentionLayer, self).get_config()

class LSTMMusicGenerator:
    """Enhanced LSTM model with Attention for high-quality music generation."""

    def __init__(
        self,
        vocab_size,
        seq_length=128,
        embedding_dim=512,
        num_layers=4,
        units=512,
        dropout=0.3,
        recurrent_dropout=0.0,
        dense_units=(1024, 512),
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
        """Build deep LSTM model with Attention."""
        inputs = layers.Input(shape=(self.seq_length,))
        x = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(inputs)
        
        # Multiple LSTM Layers
        for i in range(self.num_layers):
            x = layers.LSTM(
                self.units,
                return_sequences=True,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                name=f"lstm_layer_{i+1}"
            )(x)
            x = layers.BatchNormalization()(x)

        # Attention Mechanism
        x = AttentionLayer()(x)
        
        # Dense Layers
        for i, dense_u in enumerate(self.dense_units):
            x = layers.Dense(dense_u, activation="relu", name=f"dense_{i+1}")(x)
            x = layers.Dropout(self.dropout)(x)

        outputs = layers.Dense(self.vocab_size, activation="softmax")(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def compile_model(self, learning_rate=0.001):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")],
        )

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, callbacks=None):
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

    def generate_sequence(self, seed_sequence, length=512, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.2):
        generated = list(seed_sequence)
        for _ in range(length):
            x = np.array(generated[-self.seq_length:]).reshape(1, -1)
            # Pad if shorter than seq_length
            if x.shape[1] < self.seq_length:
                padded = np.zeros((1, self.seq_length))
                padded[0, -x.shape[1]:] = x[0]
                x = padded

            predictions = self.model.predict(x, verbose=0)[0]
            next_token = self._sample_with_controls(
                predictions,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                recent_notes=generated[-20:],
            )
            generated.append(next_token)
        return np.array(generated)

    def generate_sequences(self, seed_sequences, length=512, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.2):
        """Vectorized generation for efficiency."""
        generated = np.array(seed_sequences, dtype=np.int32)
        
        for _ in range(length):
            x = generated[:, -self.seq_length:]
            # Padding if necessary
            if x.shape[1] < self.seq_length:
                padded = np.zeros((x.shape[0], self.seq_length))
                padded[:, -x.shape[1]:] = x
                x = padded
                
            predictions = self.model.predict(x, verbose=0, batch_size=len(x))
            
            next_tokens = []
            for i in range(predictions.shape[0]):
                token = self._sample_with_controls(
                    predictions[i],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    recent_notes=generated[i, -20:]
                )
                next_tokens.append(token)
            
            next_tokens = np.array(next_tokens, dtype=np.int32).reshape(-1, 1)
            generated = np.concatenate([generated, next_tokens], axis=1)
            
        return generated

    def _sample_with_controls(self, probs, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, recent_notes=None):
        probs = np.asarray(probs, dtype=np.float64)
        
        # Apply repetition penalty
        if recent_notes is not None and repetition_penalty > 1.0:
            for n in recent_notes:
                probs[int(n)] /= repetition_penalty
        
        probs = np.log(probs + 1e-10) / max(temperature, 1e-5)
        probs = np.exp(probs - np.max(probs))
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

        # Normalize again after masking
        probs = probs / np.sum(probs)
        return np.random.choice(self.vocab_size, p=probs)

    def save_model(self, filepath):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath, custom_objects={'AttentionLayer': AttentionLayer})
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
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    return callbacks
