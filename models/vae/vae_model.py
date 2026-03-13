"""Variational Autoencoder (VAE) for symbolic music generation."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

class Sampling(layers.Layer):
    """Sampling layer for VAE latent space."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAEMusicGenerator:
    """Variational Autoencoder for music generation."""

    def __init__(self, vocab_size, seq_length=128, latent_dim=256):
        """
        Initialize VAE music generator.

        Args:
            vocab_size: Size of note vocabulary
            seq_length: Length of input sequences
            latent_dim: Dimension of latent space
        """
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.latent_dim = latent_dim

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.vae = self._build_vae()

    def _build_encoder(self):
        """Build encoder network."""
        inputs = keras.Input(shape=(self.seq_length, self.vocab_size))

        # Encoder layers
        x = layers.LSTM(512, return_sequences=True)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LSTM(256, return_sequences=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # Latent space
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder

    def _build_decoder(self):
        """Build decoder network."""
        latent_inputs = keras.Input(shape=(self.latent_dim,))

        # Decoder layers
        x = layers.Dense(512, activation='relu')(latent_inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.RepeatVector(self.seq_length)(x)
        x = layers.LSTM(256, return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LSTM(512, return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.TimeDistributed(layers.Dense(self.vocab_size, activation='softmax'))(x)

        decoder = models.Model(latent_inputs, outputs, name='decoder')
        return decoder

    def _build_vae(self):
        """Build complete VAE model."""
        inputs = keras.Input(shape=(self.seq_length, self.vocab_size))
        z_mean, z_log_var, z = self.encoder(inputs)
        outputs = self.decoder(z)

        vae = models.Model(inputs, outputs, name='vae')

        # Add KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        vae.add_loss(kl_loss)

        return vae

    def compile_model(self, learning_rate=0.001):
        """Compile the VAE model."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.vae.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, X_val, epochs=100, batch_size=64, callbacks=None):
        """Train the VAE model."""
        history = self.vae.fit(
            X_train, X_train,  # VAE reconstructs input
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def generate_from_latent(self, z=None, num_samples=1):
        """Generate sequences from latent space."""
        if z is None:
            z = np.random.normal(size=(num_samples, self.latent_dim))
        return self.decoder.predict(z, verbose=0)

    def encode(self, sequences):
        """Encode sequences to latent space."""
        z_mean, z_log_var, z = self.encoder.predict(sequences, verbose=0)
        return z

    def decode(self, z):
        """Decode latent vectors to sequences."""
        return self.decoder.predict(z, verbose=0)

    def save_model(self, filepath_prefix):
        """Save encoder and decoder separately."""
        self.encoder.save(f"{filepath_prefix}_encoder.h5")
        self.decoder.save(f"{filepath_prefix}_decoder.h5")
        self.vae.save(f"{filepath_prefix}_vae.h5")

    def load_model(self, filepath_prefix):
        """Load encoder and decoder."""
        self.encoder = keras.models.load_model(f"{filepath_prefix}_encoder.h5")
        self.decoder = keras.models.load_model(f"{filepath_prefix}_decoder.h5")
        self.vae = keras.models.load_model(f"{filepath_prefix}_vae.h5")
