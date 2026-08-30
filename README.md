# Transformer-Based Small Language Model for Multilingual Text

A compact decoder-only Transformer language model built from scratch for multilingual text, focused on **parameter efficiency, tokenizer design, and reproducible evaluation**.

> **Result:** 1,827,968 parameters and **1.7067 bits/byte (bpb)** on the development evaluation, compared with a reported 2.3718 bpb byte-tokenizer baseline — a **28.0% relative improvement**.

## Highlights

- Decoder-only GPT architecture in PyTorch
- RoPE positional encoding
- RMSNorm
- SwiGLU feed-forward blocks
- Tied token embedding / LM head weights
- SentencePiece Unigram tokenizer with 8,192 vocabulary and byte fallback
- Plain Adam and Muon + AdamW optimizer options for ablation
- Bits-per-byte evaluation for tokenizer-independent comparison

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the included smoke evaluation:

```bash
python evaluate.py --checkpoint ckpt.pt --text_file data/sample_eval.txt
```

> `data/sample_eval.txt` is a small smoke-test file included in this repository. Its bpb is **not** the reported development-set result above.

Generate text from a prompt:

```bash
python generate.py --checkpoint ckpt.pt --prompt "भारत एक" --seed 42
```

Train from scratch (requires your training corpus):

```bash
python train.py --data ../data/train_corpus.txt --steps 2000 --batch 8 --optimizer adam_all --out ckpt.pt
```

Run the basic tests:

```bash
python -m unittest discover -s tests -v
```

## Architecture

```text
Input text
   ↓
SentencePiece Unigram tokenizer (8,192 vocab)
   ↓
Token embeddings
   ↓
4 × Transformer blocks
   ├─ RMSNorm
   ├─ Causal self-attention + RoPE
   ├─ Residual connection
   ├─ RMSNorm
   ├─ SwiGLU MLP
   └─ Residual connection
   ↓
RMSNorm → vocabulary logits
```

| Component | Value |
|---|---:|
| Parameters | 1,827,968 |
| Layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 128 |
| Context length | 256 |
| Vocabulary size | 8,192 |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| FFN | SwiGLU |
| Weight tying | Yes |

The training code keeps the original **2,000-step** and **2M-parameter** constraints.

## Results

| Experiment | Dev bpb |
|---|---:|
| Byte-tokenizer baseline | 2.3718 |
| Final model | **1.7067** |
| Muon + AdamW ablation | 1.7210 |

Lower bpb is better. The final run was selected because plain Adam scored better than the recorded Muon + AdamW ablation at roughly equal wall time.

## Evaluation

The project uses **bits per byte (bpb)** rather than bits per token, avoiding direct dependence on tokenizer sequence length. The evaluator also checks tokenizer round-trip correctness and uses a sliding context window.

## Repository Structure

```text
├── model.py          # Transformer architecture
├── tokenizer.py      # SentencePiece / byte tokenizer
├── train.py          # Training loop
├── evaluate.py       # bpb evaluation
├── generate.py       # Simple text generation CLI
├── muon.py           # Muon optimizer
├── ckpt.pt           # Model checkpoint
├── tok_v8192.model   # SentencePiece model
├── tok_v8192.vocab   # Vocabulary
├── data/
│   └── sample_eval.txt # Small smoke-test evaluation file
├── tests/            # Basic model/tokenizer tests
├── NOTES.md          # Design decisions
├── RUNLOG.md         # Experiment log
├── SUMMARY.html      # Project summary
├── requirements.txt  # Python dependencies
└── .gitignore
```

## Design Choices

The model uses RoPE, RMSNorm, SwiGLU, tied embeddings, and a lossless Unigram tokenizer while staying below the 2M-parameter budget. The tokenizer and architecture changes were evaluated under the project's fixed training constraints.

## Author

**Atul1127**

Built with Python, PyTorch, and SentencePiece.
