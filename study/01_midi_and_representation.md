# 🎼 Module 01: MIDI and Representation

To teach a computer how to make music, we first need to decide how to talk to it about music. This is called **Symbolic Music Representation**.

## 🎹 What is MIDI?
MIDI (**Musical Instrument Digital Interface**) is not an audio format. It's a series of **messages**. Instead of recording sound, it records actions:
*   "Note 60 (Middle C) was pressed at 100 velocity."
*   "Note 60 was released 480 ticks later."

Think of it like digital sheet music or a player piano roll.

## 🧮 How Neural Networks "See" Music
Neural networks need numbers. There are several ways to represent MIDI as numbers:

### 1. Piano Roll (Image-like)
A 2D grid where one axis is time and the other is pitch.
*   **Pros:** Easy for CNNs to process.
*   **Cons:** Very sparse (lots of zeros), doesn't handle duration well.

### 2. Event-Based (Sequence)
A list of events: `[Note_On, Delta_Time, Note_Off]`.
*   **Pros:** Efficient, captures the natural flow of MIDI.
*   **Cons:** Can be long and complex.

### 3. Performance Encoding (Used in this project)
This project uses an optimized encoding in `utils/midi_processor.py`. It combines note pitch and instrument info into a single **token**.
*   **Pitch Range:** Typically MIDI notes 21 to 108 (Standard Piano).
*   **Vocab size:** Total number of possible tokens (Pitch x Instrument Bins).

## 🧩 Vocabulary in this Project
Check `config.yaml` under `representation`:
*   `note_range`: [21, 108]
*   `instrument_bins`: 16

The computer treats music like a language where each note-instrument combination is a "word".

---
**Next Step:** [Module 02: Preprocessing Pipeline](./02_preprocessing_pipeline.md)
