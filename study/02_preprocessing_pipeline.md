# 🏗️ Module 02: Preprocessing Pipeline

Before training a model, we need to convert thousands of MIDI files into a format that a Neural Network can ingest: **Training Sequences**.

## 🛠️ The Pipeline (`preprocessing/preprocess.py`)

### 1. File Discovery
We look for `.mid` or `.midi` files in a source directory (`data/midi_files`).

### 2. MIDI to Token Conversion (`utils/midi_processor.py`)
Each file is parsed using the `mido` library.
1.  Extract note events.
2.  Quantize time (align to a grid).
3.  Assign instrument bins.
4.  Encode as a sequence of integer tokens.

### 3. Sequence Generation (The Sliding Window)
We don't feed an entire song at once. Instead, we use a **Sliding Window**:
*   **Input Window:** A fixed number of tokens (e.g., 128 or 256).
*   **Step Size:** How many tokens to move the window.

If we have a song with 1000 notes and a window of 128, we might create 10-20 different training samples from that single song!

## 💾 Saving Data
All these sequences are packed into a `sequences.pkl` file (a Python Pickle file) in `data/processed/`.

## ⚙️ Configuration
See `config.yaml` under `data`:
*   `dataset_size: 10000` (We aim to process 10K files!)
*   `train_split: 0.85` (85% of data for training, 10% for validation, 5% for testing).

## 🧪 Try it Yourself
Run this command to see the preprocessing in action:
```bash
python preprocessing/preprocess.py --max-files 10 --create-samples
```
This will create 10 sample MIDI files and process them.

---
**Next Step:** [Module 03: RNNs and CNNs](./03_rnns_and_cnns.md)
