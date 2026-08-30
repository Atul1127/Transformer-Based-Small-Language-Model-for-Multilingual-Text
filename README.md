# Transformer-Based Small Language Model for Multilingual Text

A compact **decoder-only Transformer language model built from scratch in PyTorch** for multilingual text, focused on parameter efficiency, tokenizer design, and reproducible evaluation.

> **Result:** **1,827,968 parameters** and **1.7067 bits/byte (bpb)** on the development evaluation, compared with a reported **2.3718 bpb** byte-tokenizer baseline — a **28.0% relative improvement**.

## Highlights

- Decoder-only GPT-style architecture implemented in PyTorch
- **RoPE** positional encoding and **RMSNorm**
- **SwiGLU** feed-forward blocks
- Tied token embedding / LM-head weights
- **SentencePiece Unigram tokenizer** with 8,192 vocabulary and byte fallback
- Plain Adam and Muon + AdamW optimizer options for ablation
- **Bits-per-byte (bpb)** evaluation for tokenizer-independent comparison
- Reproducible text generation with configurable temperature, top-k, and seed

## Architecture

![Model Architecture](docs/architecture.svg)

| Component | Value |
|---|---:|
| Parameters | **1,827,968** |
| Layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 128 |
| Context length | 256 |
| Vocabulary size | 8,192 |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| FFN | SwiGLU |
| Weight tying | Yes |

## Results

| Experiment | Dev bpb |
|---|---:|
| Byte-tokenizer baseline | 2.3718 |
| Final model | **1.7067** |
| Muon + AdamW ablation | 1.7210 |

**Lower bpb is better.** The final checkpoint uses plain Adam because it achieved the best recorded result under the fixed experimental constraints.

### Training setup

- Training steps: **2,000**
- Batch size: **8**
- Context length: **256**
- Training device: CPU
- Parameter budget: **2M maximum**
- Final optimizer: Adam

## Generation Demo

The released checkpoint supports both English and Hindi prompts. Generation is a qualitative demonstration; **bpb is the primary quantitative evaluation metric** for this constrained model.

### English

```text
Prompt: The state

Generated:
The state. The war rate of the United States was shosed by the South Atlantic Ocean, and the U.S. U.S. Constitution in the Atlantic Ocean, and the western North Sea.
```

### Hindi

```text
Prompt: भारत एक

Generation is supported through the same tokenizer + Transformer pipeline.
```

> Generation quality is limited by the intentionally small model size and short training budget. Outputs are shown as generated rather than manually edited.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the included smoke evaluation:

```bash
python evaluate.py --checkpoint ckpt.pt --text_file data/sample_eval.txt
```

Generate text:

```bash
python generate.py --checkpoint ckpt.pt --prompt "भारत एक" --temperature 0.7 --top_k 40 --max_new_tokens 50 --seed 42
```

The generator prints the prompt, sampling configuration, and generated text in a structured terminal format.

Train from scratch (requires your training corpus):

```bash
python train.py --data ../data/train_corpus.txt --steps 2000 --batch 8 --optimizer adam_all --out ckpt.pt
```

Run the basic tests:

```bash
python -m unittest discover -s tests -v
```

## Evaluation

The project uses **bits per byte (bpb)** rather than bits per token, reducing dependence on tokenizer sequence length. The evaluator checks tokenizer round-trip correctness and evaluates the model with a sliding context window.

The final verified development evaluation scored **1.7067 bpb** over **37,272 scored tokens** after 2,000 training steps.

## Experiments and Ablation

The experiment log records the progression from a byte-tokenizer baseline to the final Unigram-tokenized Transformer. The final model combines RoPE, RMSNorm, SwiGLU, scaled residual initialization, weight tying, and a lossless Unigram tokenizer while remaining below the 2M-parameter budget.

A Muon + AdamW run achieved **1.7210 bpb**, while the Adam-only run achieved **1.7067 bpb** at approximately equal wall time. This comparison uses the tested learning-rate configurations and should not be interpreted as a universal optimizer ranking.

See [`RUNLOG.md`](RUNLOG.md) for the detailed experiment record.

## Limitations

This project is intentionally constrained to approximately **2M parameters and 2,000 training steps**. As a result:

- Generated text is not expected to match the coherence or factual reliability of modern large language models.
- The training corpus and compute budget are small compared with production-scale language-model training.
- bpb measures language-modeling efficiency but does not establish instruction following, reasoning, or downstream task performance.
- The optimizer ablation compares selected configurations rather than a full hyperparameter sweep.

These constraints are part of the project's experimental setting rather than claims of production-scale LLM performance.

## Repository Structure

```text
├── model.py          # Transformer architecture
├── tokenizer.py      # SentencePiece / byte tokenizer
├── train.py          # Training loop
├── evaluate.py       # bpb evaluation
├── generate.py       # Text generation CLI
├── muon.py           # Muon optimizer
├── ckpt.pt           # Final model checkpoint
├── tok_v8192.model   # SentencePiece model
├── tok_v8192.vocab   # Vocabulary
├── docs/
│   └── architecture.svg # Architecture diagram
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
