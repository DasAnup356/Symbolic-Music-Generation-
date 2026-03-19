# 📊 Module 06: Evaluation Metrics

How do we know if our AI is making "good" music or just "random noise"? We use **Evaluation Metrics**.

## 📏 Objective Metrics (`evaluation/evaluate.py`)
Since AI doesn't have ears, it uses statistics to measure the "musicality" of its creations:

### 1. Note Density
*   The average number of notes per time unit.
*   **Target:** Between 0.1 and 0.9 (Too high is "chaos", too low is "silence").

### 2. Pitch Range
*   The difference between the highest and lowest note.
*   **Target:** Average of 45 semitones (approx. 4 octaves).

### 3. Pitch Class Entropy
*   A measure of "variety" in the 12 notes (C, C#, D, etc.).
*   **Target:** Around 3.12 bits (Lower means "boring", higher means "too random").

### 4. Scale Consistency
*   How many notes stay within a specific key (e.g., C Major).

## 🎼 Rhythmic Complexity
We also measure:
*   **Polyphonic Ratio:** How many notes are played at once.
*   **Note Length Distribution:** Are there too many short notes? Too many long notes?

## 🎧 Subjective Evaluation
The ultimate test is a "Human Turing Test" for music:
*   **Musically Coherent:** Does it sound like a "human-composed" melody?
*   **Diverse:** Do the songs sound different from each other?

## 🧪 Try it Yourself
Run this command to evaluate your generated MIDI files:
```bash
python evaluation/evaluate.py --midi-dir outputs/generated_midi/lstm
```

---
**Congratulations!** You've completed the Bottom-Up study path.
Now you're ready to dive into the code and start generating your own music! 🎵
