# Transformer-Based Small Language Model for Multilingual Text

A **1.83M-parameter decoder-only Transformer built from scratch in PyTorch** for multilingual language modeling under a strict compute budget.

The project studies how much language-modeling efficiency a tiny model can obtain from careful choices in **architecture, tokenization, and optimization**.

> **Best recorded development result:** **1.7067 bits/byte (bpb)**, compared with **2.3718 bpb** for the byte-tokenizer baseline — a **28.0% relative improvement**.

## Highlights

- **1,827,968 parameters** under a 2M parameter budget
- 4-layer decoder-only Transformer with **RoPE, RMSNorm, SwiGLU, and tied embeddings**
- **SentencePiece Unigram + byte fallback** for multilingual UTF-8 text
- Adam and Muon + AdamW optimizer configurations
- Sliding-window **bits-per-byte (bpb)** evaluation
- Reproducible CLI generation with temperature, top-k, and seed controls
- Optional **Gradio demo** for interactive inference
- Unit tests for model behavior, parameter budget, context length, and UTF-8 tokenization

## Results

| Experiment | Dev bpb | Result |
|---|---:|---|
| Byte-tokenizer baseline | 2.3718 | Reference |
| Unigram reference | 1.9326 | Tokenization improvement |
| Muon + AdamW | 1.7210 | Optimizer ablation |
| **Final Unigram + Adam** | **1.7067** | **Best recorded** |

The final checkpoint was trained for **2,000 steps on CPU** and evaluated on **37,272 scored tokens**. Lower bpb is better.

The optimizer comparison is limited to the configurations tested in this project; it is not a general benchmark of Adam versus Muon.

## Architecture

![Model Architecture](docs/architecture.svg)

| Component | Configuration |
|---|---|
| Model | Decoder-only GPT-style Transformer |
| Layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 128 |
| Context length | 256 tokens |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Feed-forward | SwiGLU |
| Attention | Causal scaled dot-product attention |
| Weight tying | Token embedding ↔ LM head |
| Tokenizer | SentencePiece Unigram |
| Vocabulary | 8,192 + byte fallback |

The Transformer is implemented directly with PyTorch modules rather than using a pretrained Transformer stack.

## Tokenization

The tokenizer uses **SentencePiece Unigram with byte fallback**, giving compact subword representations while retaining a lossless path for arbitrary UTF-8 text.

The evaluator verifies:

```text
text → encode → decode == original text
```

A raw byte tokenizer is also available as a fallback when no trained SentencePiece model is present.

## Training

The training loop supports:

1. **Adam** over all parameters
2. **Muon + AdamW** split between Transformer matrices and remaining parameters

It also includes:

- linear learning-rate warmup
- cosine learning-rate decay
- global gradient clipping
- configurable random seed and model dimensions
- explicit 2M-parameter guard
- checkpoint metadata and training-loss recording

The released repository does **not** include the original training corpus. To retrain, provide your own compatible UTF-8 corpus:

```bash
python train.py \
  --data path/to/train_corpus.txt \
  --steps 2000 \
  --batch 8 \
  --optimizer adam_all \
  --out ckpt.pt
```

## Evaluation

The primary metric is **bits per byte (bpb)** rather than bits per token, making comparisons less dependent on tokenizer sequence length.

Standard evaluation:

```bash
python evaluate.py --checkpoint ckpt.pt --text_file data/sample_eval.txt
```

Language-specific evaluation:

```bash
python evaluate_languages.py \
  --checkpoint ckpt.pt \
  --english data/sample_eval_english.txt \
  --hindi data/sample_eval_hindi.txt
```

The language-specific files are intentionally small smoke-test datasets. Their bpb values are useful for checking the multilingual pipeline, **not as statistically stable language benchmarks**.

## Generation

Run deterministic CLI generation with a fixed seed:

```bash
python generate.py \
  --checkpoint ckpt.pt \
  --prompt "भारत एक" \
  --temperature 0.7 \
  --top_k 40 \
  --max_new_tokens 50 \
  --seed 42
```

The model is intentionally tiny, so generation quality is limited. **bpb is the primary quantitative result; generated text is qualitative only.**

Example recorded English generation:

```text
Prompt: The state
Temperature: 0.7
Top-k: 40
Seed: 42

Generated:
The state. The war rate of the United States was shosed by the South Atlantic Ocean, and the U.S. U.S. Constitution in the Atlantic Ocean, and the western North Sea.
```

## Interactive Demo

The repository includes an optional Gradio interface for trying the released checkpoint.

Install core dependencies and the demo dependency:

```bash
pip install -r requirements.txt
pip install -r requirements-demo.txt
```

Launch:

```bash
python app.py
```

The demo exposes prompt, temperature, top-k, maximum generation length, and seed controls. It is an **experimental small language model**, not a chatbot or production inference system.

## Reproducibility

The released checkpoint stores model configuration, training-step count, random seed, tokenizer information, and training-loss data.

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

The current test suite covers:

- model forward pass and loss shape
- 2M parameter limit
- context-length enforcement
- lossless UTF-8 byte-tokenizer round trips
- byte-tokenizer vocabulary size

### Verified local smoke checks

The repository has been locally verified with:

- **5/5 unit tests passing**
- checkpoint loading and standard evaluation working
- English and Hindi evaluation utilities working
- Gradio application launching successfully

## Repository Structure

```text
├── model.py                 # Transformer architecture
├── tokenizer.py             # SentencePiece / byte tokenizer
├── train.py                 # Training loop
├── train_unigram.py         # SentencePiece training utility
├── evaluate.py              # Standard bpb evaluator
├── evaluate_languages.py    # Per-language bpb evaluator
├── generate.py              # CLI generation
├── app.py                   # Optional Gradio demo
├── muon.py                  # Muon optimizer
├── ckpt.pt                  # Released checkpoint
├── tok_v8192.model          # SentencePiece model
├── tok_v8192.vocab          # SentencePiece vocabulary
├── docs/                    # Architecture diagram
├── data/                    # Evaluation smoke-test data
├── tests/                   # Unit tests
├── NOTES.md                 # Design decisions
├── RUNLOG.md                # Experiment log
├── requirements.txt         # Core dependencies
└── requirements-demo.txt    # Optional demo dependency
```

## Design Rationale

- **RoPE:** positional information without learned absolute position embeddings.
- **RMSNorm:** lightweight normalization for pre-norm Transformer blocks.
- **SwiGLU:** stronger feed-forward expressiveness within the parameter budget.
- **Weight tying:** avoids duplicating token embedding and output projection weights.
- **Unigram + byte fallback:** improves tokenization efficiency while preserving UTF-8 round trips.
- **Scaled residual initialization:** helps control residual-stream variance in the shallow network.

## Limitations

This project intentionally operates far below the scale of modern language models.

- 1.83M parameters limits generation quality and learned knowledge.
- 2,000 training steps and CPU training provide a small compute budget.
- bpb measures language-modeling efficiency, not instruction following, reasoning, factuality, or downstream task performance.
- The optimizer experiment is an ablation rather than a full hyperparameter sweep.
- The headline result is a development-set result under a fixed experimental protocol.

These limitations are part of the experimental design and are reported explicitly.

## Tech Stack

**Python · PyTorch · SentencePiece · Transformer · RoPE · RMSNorm · SwiGLU · Muon · AdamW · bpb · Gradio**

## Author

**Atul1127**

Built from scratch as an experiment in efficient multilingual language modeling.
