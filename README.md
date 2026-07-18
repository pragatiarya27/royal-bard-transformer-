# 📜 Royal Bard Transformer

A decoder-style Transformer, built from scratch in TensorFlow/Keras, trained to generate fairy-tale text one word at a time.

This project is a follow-up to an earlier RNN/LSTM-based text generator. The goal was to swap the recurrent architecture for a Transformer and compare how self-attention handles context versus a recurrent model that tends to "forget" the beginning of longer sequences.

## Architecture

- **Token + Positional Embedding** — a custom Keras layer that adds a learned positional embedding to each token embedding, since Transformers have no inherent sense of word order.
- **Transformer Block** — multi-head self-attention (causal-masked) → residual + LayerNorm → feed-forward network → residual + LayerNorm.
- **Output head** — the last token's representation is passed through a Dense softmax layer over the vocabulary to predict the next word.

```
tokens → [Token + Positional Embedding] → [Transformer Block] → last token → Dense(softmax) → next word
```

## Pipeline

1. **Cleaning** — lowercase + strip punctuation.
2. **Tokenization** — Keras `Tokenizer`, vocab size 2000, OOV handling.
3. **N-gram sequence generation** — each tale is expanded into `(seen-so-far → next word)` training pairs.
4. **Padding** — pre-padded to a fixed length so the target always sits at the last position.
5. **Training** — `sparse_categorical_crossentropy` loss, Adam optimizer, 30 epochs.
6. **Generation** — autoregressive sampling with temperature scaling (and optional top-k filtering) to control creativity vs. coherence.

## Experiments

A few architecture ablations were run on top of the base model to build intuition for how each hyperparameter affects output quality:

| Experiment | Variants tested |
|---|---|
| Transformer block depth | 1 block vs. 2 stacked blocks |
| Embedding dimension | 32 vs. 128 |
| Attention heads | 1 vs. 8 |
| Training duration | 30 vs. 60 epochs |

## Sample output

```
Seed: "Once upon a time in a faraway kingdom there lived"

T=0.4 (low):  ...a kind and gentle king who ruled the land with wisdom and grace...
T=1.0 (mid):  ...a curious young girl who dreamed of sailing past the misty mountains...
T=1.5 (high): ...a shimmering fox spirit who whispered riddles to the wandering stars...
```

*(Replace with your own generated samples once you run the notebook.)*

## Tech stack

- Python, TensorFlow / Keras
- NumPy, Pandas
- Trained on a small built-in corpus of 50 fairy-tale snippets (swap in your own `tales.csv` to use a larger dataset)

## Getting started

```bash
pip install tensorflow numpy pandas
jupyter notebook royal_bard_transformer.ipynb
```

Run all cells top to bottom. Training takes roughly 1–3 minutes on CPU.

## Why this project

Built to move past "black box" usage of Transformer-based LLMs and understand the mechanics directly — implementing self-attention, positional embeddings, and autoregressive generation from the ground up on a small, fast-to-train dataset.
