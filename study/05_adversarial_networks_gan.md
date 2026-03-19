# ⚔️ Module 05: Adversarial Networks (GAN)

Generative Adversarial Networks (GANs) are the "Wild West" of Deep Learning. They're based on a competition between two models: **The Generator** and **The Discriminator**.

## 🎨 The Generator (`models/gan/gan_model.py`)
Think of the Generator as an **Art Forger**.
1.  Takes a "Random Noise Vector" (e.g., 100 random numbers).
2.  Passes it through series of layers (Dense + LSTM).
3.  Tries to create a 256-note sequence that looks like real music.

## 🕵️ The Discriminator (`models/gan/gan_model.py`)
Think of the Discriminator as an **Art Critic** or a **Detective**.
1.  Takes a MIDI sequence as input.
2.  Learns to distinguish between "Real" music (from the dataset) and "Fake" music (from the Generator).
3.  Outputs a single probability: `0` (Fake) or `1` (Real).

## ⚔️ The Competition
*   The **Generator** is penalized when the Discriminator catches its fakes.
*   The **Discriminator** is penalized when the Generator successfully tricks it.

### 🎮 The End Game: Nash Equilibrium
Eventually, the Generator gets so good that the Discriminator can only guess 50/50. At this point, the Generator has learned to create "Real" music!

## 🚧 Challenges in Symbolic GANs
Standard GANs work well with continuous data (like images). MIDI is **discrete** (you can't have "Note 60.5").
*   This project uses Gumbel-Softmax or specific sampling techniques to make the Generator differentiable during training.

---
**Next Step:** [Module 06: Evaluation Metrics](./06_evaluation_metrics.md)
