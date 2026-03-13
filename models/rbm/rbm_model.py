"""Restricted Boltzmann Machine (RBM) for symbolic music generation."""

import numpy as np
import tensorflow as tf

class RBMMusicGenerator:
    """Restricted Boltzmann Machine for music generation."""

    def __init__(self, n_visible, n_hidden=512, learning_rate=0.01, k=1):
        """
        Initialize RBM.

        Args:
            n_visible: Number of visible units
            n_hidden: Number of hidden units
            learning_rate: Learning rate for CD-k
            k: Number of Gibbs sampling steps
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.k = k

        # Initialize weights and biases
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.vbias = np.zeros(n_visible)
        self.hbias = np.zeros(n_hidden)

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sample_hidden(self, v):
        """Sample hidden units given visible units."""
        h_prob = self.sigmoid(np.dot(v, self.W) + self.hbias)
        h_sample = (np.random.random(h_prob.shape) < h_prob).astype(float)
        return h_prob, h_sample

    def sample_visible(self, h):
        """Sample visible units given hidden units."""
        v_prob = self.sigmoid(np.dot(h, self.W.T) + self.vbias)
        v_sample = (np.random.random(v_prob.shape) < v_prob).astype(float)
        return v_prob, v_sample

    def contrastive_divergence(self, v0):
        """Perform one step of contrastive divergence."""
        # Positive phase
        h0_prob, h0_sample = self.sample_hidden(v0)

        # Negative phase - k steps of Gibbs sampling
        vk_sample = v0
        for _ in range(self.k):
            hk_prob, hk_sample = self.sample_hidden(vk_sample)
            vk_prob, vk_sample = self.sample_visible(hk_sample)

        hk_prob, _ = self.sample_hidden(vk_sample)

        # Update weights and biases
        positive_grad = np.dot(v0.T, h0_prob)
        negative_grad = np.dot(vk_sample.T, hk_prob)

        self.W += self.learning_rate * (positive_grad - negative_grad) / v0.shape[0]
        self.vbias += self.learning_rate * np.mean(v0 - vk_sample, axis=0)
        self.hbias += self.learning_rate * np.mean(h0_prob - hk_prob, axis=0)

        # Compute reconstruction error
        error = np.mean((v0 - vk_sample) ** 2)
        return error

    def train(self, data, n_epochs=100, batch_size=32, verbose=True):
        """
        Train the RBM using contrastive divergence.

        Args:
            data: Training data (n_samples, n_visible)
            n_epochs: Number of training epochs
            batch_size: Batch size
            verbose: Print training progress

        Returns:
            List of reconstruction errors per epoch
        """
        errors = []
        n_batches = len(data) // batch_size

        for epoch in range(n_epochs):
            epoch_error = 0

            # Shuffle data
            np.random.shuffle(data)

            for i in range(n_batches):
                batch = data[i * batch_size:(i + 1) * batch_size]
                error = self.contrastive_divergence(batch)
                epoch_error += error

            epoch_error /= n_batches
            errors.append(epoch_error)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Reconstruction Error: {epoch_error:.6f}")

        return errors

    def generate(self, n_samples=1, n_gibbs_steps=1000):
        """
        Generate samples using Gibbs sampling.

        Args:
            n_samples: Number of samples to generate
            n_gibbs_steps: Number of Gibbs sampling steps

        Returns:
            Generated samples
        """
        # Start with random visible units
        v = np.random.binomial(1, 0.5, (n_samples, self.n_visible))

        # Run Gibbs sampling
        for _ in range(n_gibbs_steps):
            h_prob, h_sample = self.sample_hidden(v)
            v_prob, v = self.sample_visible(h_sample)

        return v

    def save_model(self, filepath):
        """Save RBM parameters."""
        np.savez(filepath,
                 W=self.W,
                 vbias=self.vbias,
                 hbias=self.hbias,
                 n_visible=self.n_visible,
                 n_hidden=self.n_hidden)

    def load_model(self, filepath):
        """Load RBM parameters."""
        data = np.load(filepath)
        self.W = data['W']
        self.vbias = data['vbias']
        self.hbias = data['hbias']
        self.n_visible = int(data['n_visible'])
        self.n_hidden = int(data['n_hidden'])

class CRBMMusicGenerator:
    """Conditional RBM for temporal music generation."""

    def __init__(self, n_visible, n_hidden=512, n_visible_temporal=100, learning_rate=0.01):
        """
        Initialize Conditional RBM.

        Args:
            n_visible: Number of visible units (current frame)
            n_hidden: Number of hidden units
            n_visible_temporal: Number of visible units from previous frames
            learning_rate: Learning rate
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_visible_temporal = n_visible_temporal
        self.learning_rate = learning_rate

        # Initialize weights
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.A = np.random.normal(0, 0.01, (n_visible_temporal, n_hidden))
        self.B = np.random.normal(0, 0.01, (n_visible_temporal, n_visible))

        self.vbias = np.zeros(n_visible)
        self.hbias = np.zeros(n_hidden)

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sample_hidden(self, v, v_prev):
        """Sample hidden units given current and previous visible units."""
        h_activation = np.dot(v, self.W) + np.dot(v_prev, self.A) + self.hbias
        h_prob = self.sigmoid(h_activation)
        h_sample = (np.random.random(h_prob.shape) < h_prob).astype(float)
        return h_prob, h_sample

    def sample_visible(self, h, v_prev):
        """Sample visible units given hidden units and previous visible."""
        v_activation = np.dot(h, self.W.T) + np.dot(v_prev, self.B) + self.vbias
        v_prob = self.sigmoid(v_activation)
        v_sample = (np.random.random(v_prob.shape) < v_prob).astype(float)
        return v_prob, v_sample

    def generate_sequence(self, length=256, v_init=None):
        """
        Generate a sequence using the CRBM.

        Args:
            length: Length of sequence to generate
            v_init: Initial visible state

        Returns:
            Generated sequence
        """
        if v_init is None:
            v_init = np.random.binomial(1, 0.5, (1, self.n_visible_temporal))

        sequence = []
        v_prev = v_init

        for _ in range(length):
            # Sample hidden
            h_prob, h_sample = self.sample_hidden(np.zeros((1, self.n_visible)), v_prev)

            # Sample visible
            v_prob, v_sample = self.sample_visible(h_sample, v_prev)

            sequence.append(v_sample[0])

            # Update previous visible (shift window)
            v_prev = np.roll(v_prev, -self.n_visible)
            v_prev[0, -self.n_visible:] = v_sample[0]

        return np.array(sequence)
