# 🗺️ Symbolic Music Generation: Bottom-Up Study Roadmap

Welcome! This roadmap is designed to take you from the very basics of music representation to state-of-the-art deep generative models.

## 🛤️ The Learning Path

### Phase 1: Foundations
*   **[01: MIDI and Representation](./01_midi_and_representation.md)**
    *   What is symbolic music?
    *   The MIDI protocol.
    *   How to represent notes for a computer (Piano Rolls, Performance Encoding).

### Phase 2: Data Engineering
*   **[02: Preprocessing Pipeline](./02_preprocessing_pipeline.md)**
    *   Parsing MIDI files.
    *   Tokenization and vocabularies.
    *   Creating training sequences with sliding windows.

### Phase 3: Sequential & Structural Modeling
*   **[03: RNNs (LSTM/GRU) and CNNs](./03_rnns_and_cnns.md)**
    *   Why sequences matter in music.
    *   Vanishing gradients and the need for Memory (LSTM).
    *   Using CNNs for finding melodic motifs.

### Phase 4: Probabilistic & Latent Models
*   **[04: Probabilistic Models (VAE & RBM)](./04_probabilistic_models_vae_rbm.md)**
    *   Generative vs. Discriminative models.
    *   Variational Autoencoders (VAE) and latent spaces.
    *   Restricted Boltzmann Machines (RBM) for music.

### Phase 5: Adversarial Learning
*   **[05: Adversarial Networks (GAN)](./05_adversarial_networks_gan.md)**
    *   The Generator and the Discriminator.
    *   Training GANs for symbolic music.
    *   Challenges in discrete sequence generation.

### Phase 6: Success Metrics
*   **[06: Evaluation Metrics](./06_evaluation_metrics.md)**
    *   How do we know if the music is "good"?
    *   Note density, pitch range, and entropy.
    *   Subjective vs. Objective evaluation.

---

## 🛠️ How to use this guide
1.  Read the modules in order.
2.  Explore the corresponding code in the repository.
3.  Use the AI Tutor (`python main.py --tutor "your question"`) to clarify concepts.
