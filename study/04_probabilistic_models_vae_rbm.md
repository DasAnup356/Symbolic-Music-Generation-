# 🌌 Module 04: Probabilistic Models (VAE & RBM)

Some models don't just predict the next note. They learn the "probability distribution" of the whole song at once. This project explores two key architectures for this: **Variational Autoencoders (VAE)** and **Restricted Boltzmann Machines (RBM)**.

## 🌈 Variational Autoencoders (VAE) (`models/vae/vae_model.py`)
A VAE doesn't just copy an input MIDI song. It compresses it into a small "Latent Space".

### 1. Encoder (The Compressor)
A series of LSTM layers that turn a whole song (e.g., 256 tokens) into a small "Vector" (e.g., 256 dimensions).
*   **Trick:** Instead of a single point, it learns a **mean** and a **standard deviation**. This creates a continuous, smooth space.

### 2. Decoder (The Reconstructor)
Takes a random point from this Latent Space and tries to "uncompress" it back into music.

### 🎨 Creativity with VAEs: Latent Space Interpolation
You can take two different songs, find their "Vectors" in the Latent Space, and find the points *between* them. This lets you **blend** two songs together!

## ⚡ Restricted Boltzmann Machines (RBM) (`models/rbm/rbm_model.py`)
An RBM is an "Energy-Based" model. It has:
*   **Visible Layer:** The actual MIDI notes.
*   **Hidden Layer:** "Hidden" features (patterns) the model learns.

### 1. Training with Contrastive Divergence
Instead of "Backpropagation" (used in LSTMs), RBMs use a Gibbs Sampling process to align the "Hidden Layer" with the "Visible Layer".

### 2. CRBM (Conditional RBM)
To generate music over time, we use a CRBM. It looks at the *past* notes to decide how the *current* hidden features should behave.

---
**Next Step:** [Module 05: Adversarial Networks (GAN)](./05_adversarial_networks_gan.md)
