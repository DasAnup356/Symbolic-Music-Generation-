"""Generative Adversarial Network (GAN) for symbolic music generation."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

class GANMusicGenerator:
    """GAN for symbolic music generation."""

    def __init__(self, seq_length=128, vocab_size=88, latent_dim=100):
        """
        Initialize GAN music generator.

        Args:
            seq_length: Length of generated sequences
            vocab_size: Size of note vocabulary
            latent_dim: Dimension of latent noise vector
        """
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()

    def _build_generator(self):
        """Build generator network."""
        model = models.Sequential([
            # Input: latent vector
            layers.Dense(256, input_dim=self.latent_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),

            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),

            layers.Dense(1024),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),

            # Reshape for sequence
            layers.Dense(self.seq_length * 128),
            layers.Reshape((self.seq_length, 128)),

            # LSTM layers
            layers.LSTM(256, return_sequences=True),
            layers.BatchNormalization(),
            layers.LSTM(512, return_sequences=True),
            layers.BatchNormalization(),

            # Output layer
            layers.TimeDistributed(layers.Dense(self.vocab_size, activation='softmax'))
        ], name='generator')

        return model

    def _build_discriminator(self):
        """Build discriminator network."""
        model = models.Sequential([
            # Input: sequence
            layers.LSTM(512, return_sequences=True, 
                       input_shape=(self.seq_length, self.vocab_size)),
            layers.Dropout(0.3),

            layers.LSTM(256, return_sequences=False),
            layers.Dropout(0.3),

            # Dense layers
            layers.Dense(128),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),

            layers.Dense(64),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),

            # Output: real/fake probability
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')

        return model

    def _build_gan(self):
        """Build combined GAN model."""
        self.discriminator.trainable = False

        gan_input = keras.Input(shape=(self.latent_dim,))
        generated_sequence = self.generator(gan_input)
        gan_output = self.discriminator(generated_sequence)

        gan = models.Model(gan_input, gan_output, name='gan')
        return gan

    def compile_models(self, g_learning_rate=0.0002, d_learning_rate=0.0002):
        """Compile generator and discriminator."""
        g_optimizer = keras.optimizers.Adam(learning_rate=g_learning_rate, beta_1=0.5)
        d_optimizer = keras.optimizers.Adam(learning_rate=d_learning_rate, beta_1=0.5)

        self.discriminator.compile(
            optimizer=d_optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.gan.compile(
            optimizer=g_optimizer,
            loss='binary_crossentropy'
        )

    def train(self, X_train, epochs=100, batch_size=64, save_interval=10):
        """
        Train the GAN.

        Args:
            X_train: Training sequences
            epochs: Number of training epochs
            batch_size: Batch size
            save_interval: Interval for saving generated samples

        Returns:
            Training history (discriminator and generator losses)
        """
        history = {'d_loss': [], 'g_loss': [], 'd_acc': []}

        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Train discriminator
            # Select random real sequences
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_sequences = X_train[idx]

            # Generate fake sequences
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_sequences = self.generator.predict(noise, verbose=0)

            # Train discriminator on real and fake
            d_loss_real = self.discriminator.train_on_batch(real_sequences, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_sequences, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, real_labels)

            # Record losses
            history['d_loss'].append(d_loss[0])
            history['d_acc'].append(d_loss[1])
            history['g_loss'].append(g_loss)

            if (epoch + 1) % save_interval == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}")
                print(f"  G Loss: {g_loss:.4f}")

        return history

    def generate(self, num_samples=1, noise=None):
        """
        Generate sequences from random noise.

        Args:
            num_samples: Number of sequences to generate
            noise: Optional pre-specified noise vectors

        Returns:
            Generated sequences
        """
        if noise is None:
            noise = np.random.normal(0, 1, (num_samples, self.latent_dim))

        generated = self.generator.predict(noise, verbose=0)
        return generated

    def save_models(self, filepath_prefix):
        """Save generator and discriminator."""
        self.generator.save(f"{filepath_prefix}_generator.h5")
        self.discriminator.save(f"{filepath_prefix}_discriminator.h5")

    def load_models(self, filepath_prefix):
        """Load generator and discriminator."""
        self.generator = keras.models.load_model(f"{filepath_prefix}_generator.h5")
        self.discriminator = keras.models.load_model(f"{filepath_prefix}_discriminator.h5")
        self.gan = self._build_gan()
