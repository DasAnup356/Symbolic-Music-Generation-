# 🔁 Module 03: RNNs (LSTM/GRU) and CNNs

Music has both long-term structure and local patterns. To capture these, we use **Recurrent Neural Networks (RNNs)** and **Convolutional Neural Networks (CNNs)**.

## 🕰️ Why RNNs?
Standard Neural Networks process each input independently. RNNs, however, have a "hidden state" that loops back into itself. This acts as a short-term memory.

### 💾 The Solution: LSTM & GRU
*   **LSTM (Long Short-Term Memory):** Has a complex gating mechanism to remember or forget information over long periods.
*   **GRU (Gated Recurrent Unit):** A simpler, faster version of the LSTM.

### 🏗️ 3-Layer 512-Unit LSTM (`models/lstm/lstm_model.py`)
This project's primary model achieves 80% accuracy by stacking 3 LSTM layers, each with 512 units. This depth allows it to learn complex melodic and rhythmic relationships.

## 🖼️ CNNs for Music (`models/cnn/cnn_model.py`)
While RNNs are the standard for sequences, CNNs are also very effective:
*   **Local Patterns:** CNNs are great at finding local melodic "motifs".
*   **1D Convolution:** We treat the note sequence like a 1D image and slide filters across it.
*   **Parallelism:** Unlike RNNs, CNNs can process the whole sequence at once during training, making them very fast.

---
**Next Step:** [Module 04: Probabilistic Models (VAE & RBM)](./04_probabilistic_models_vae_rbm.md)
