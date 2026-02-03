# Language Models from Scratch

This repository contains my from-scratch implementations of core neural network and language model components, inspired by Andrej Karpathy’s *Neural Networks: Zero to Hero* series.

I worked through the underlying mechanics of these models in Python, added my own explanations, extended suggested exercises, and validated components through testing.  
The goal of this project is a deep, practical understanding of how language models work under the hood.

Files are organized by concept rather than by training pipeline.

---

## ✨ What This Repository Covers

### 🔹 Foundations
- **Micrograd** – A minimal automatic differentiation engine
- **Backpropagation** – Explicit forward and backward passes
- **Batch Normalization** – Forward and backward implementations
- **MLP** – Fully connected neural networks built step by step

### 🔹 Statistical Language Models
- **Bigram Language Model**
- **Trigram Language Model**
- Log-likelihood loss, probability estimation, and sampling

### 🔹 Neural Language Models
- **Neural Bigram Model**
- **WaveNet-style MLP**
- **Transformer Language Model**
  - Self-attention
  - Token and positional embeddings
  - Causal masking

### 🔹 Tokenization
- **Byte Pair Encoding (BPE)** implemented from first principles
- Custom **minBPE tokenizer**
- Tokenizer training, encoding, decoding, and tests

---

## Disclaimer

- This project is exploratory and intended for educational and analytical purposes only
- Implementations prioritize clarity and learning over optimization
- Models are trained on small datasets and are not production-ready
