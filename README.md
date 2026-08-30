# Transformer-Based Small Language Model for Multilingual Text

A compact decoder-only Transformer language model built from scratch for multilingual text, with a focus on **parameter efficiency, tokenizer design, and reproducible evaluation**.

> **Result:** 1,827,968 parameters and **1.7067 bits/byte (bpb)** on the project's development evaluation, compared with a reported 2.3718 bpb byte-tokenizer baseline — a **28.0% relative improvement**. See `NOTES.md` and `RUNLOG.md` for the experiment record.

## Highlights

- **Decoder-only GPT architecture** implemented in PyTorch.
- **RoPE** positional encoding instead of learned absolute position embeddings.
- **RMSNorm** for normalization.
- **SwiGLU** feed-forward blocks with parameter-aware hidden dimension scaling.
- **Weight tying** between token embeddings and the LM head.
- **SentencePiece Unigram tokenizer** with an 8,192-token vocabulary, byte fallback, and lossless round-trip behavior.
- **Residual-output scaled initialization** for stable training in a small model.
- Training supports both **plain Adam** and a **Muon + AdamW** split so optimizer ablations can be compared under the same training loop.
- Evaluation reports **bits per byte**, making results comparable across tokenizer choices.

## Architecture

```text
Input text
   ↓
SentencePiece Unigram tokenizer (8,192 vocab)
   ↓
Token embeddings (tied to output head)
   ↓
4 × Transformer blocks
   ├─ RMSNorm
   ├─ Causal self-attention + RoPE
   ├─ Residual connection
   ├─ RMSNorm
   ├─ SwiGLU MLP
   └─ Residual connection
   ↓
RMSNorm
   ↓
Vocabulary logits
```

Default model configuration:

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
| Dropout | 0.0 |
| Weight tying | Yes |

The training code enforces a **2,000-step** and **2M-parameter** ceiling from the original project constraints. fileciteturn3file0

## Training

The training loop uses:

- linear learning-rate warmup
- cosine learning-rate decay
- global gradient-norm clipping
- deterministic seed control
- configurable model size and optimizer choice

Example:

```bash
python train.py \
  --data ../data/train_corpus.txt \
  --steps 2000 \
  --batch 8 \
  --optimizer adam_all \
  --out ckpt.pt
```

The current project notes identify the plain-Adam run as the final selected configuration because it scored better in the recorded ablation than the Muon + AdamW setup at roughly equal wall time. fileciteturn5file0

## Evaluation

Evaluation uses **bits per byte (bpb)** rather than bits per token. This avoids giving one tokenizer an unfair advantage simply because it creates shorter or longer token sequences.

```bash
python evaluate.py \
  --checkpoint ckpt.pt \
  --text_file ../data/dev_eval.txt
```

Example output:

```json
{"bpb": 1.7067, "n_params": 1827968, "steps": 2000, "tokens_in_eval": ..., "tokens_scored": ...}
```

The evaluator also verifies that the tokenizer is lossless and scores the text with a sliding context window, so the reported metric remains consistent across runs. fileciteturn6file0

## Experiment Summary

| Experiment | Dev bpb | Relative result |
|---|---:|---:|
| Byte-tokenizer baseline | 2.3718 | Baseline |
| Final multilingual tokenizer + architecture | **1.7067** | **28.0% lower bpb** |
| Muon + AdamW ablation | 1.7210 | Worse than final run |

The project notes attribute the Muon result primarily to limited tuning and the small batch size used in this experiment. The final implementation therefore favors the simpler optimizer configuration rather than adding complexity without a measured gain. fileciteturn5file0

## Repository Structure

```text
.
├── model.py          # Transformer architecture and model configuration
├── tokenizer.py      # SentencePiece tokenizer wrapper
├── train.py          # Training loop and optimizer/schedule controls
├── evaluate.py       # Reproducible bpb evaluation
├── muon.py           # Muon optimizer implementation
├── ckpt.pt           # Saved model checkpoint
├── tok_v8192.model   # SentencePiece model
├── tok_v8192.vocab   # Token vocabulary
├── NOTES.md          # Design decisions and experiment rationale
├── RUNLOG.md         # Recorded training runs
└── SUMMARY.html      # Project summary artifact
```

## Why This Project Matters

This project is less about scaling a large model and more about understanding the **trade-offs that matter when compute and parameter budgets are tight**:

1. Tokenization can strongly affect multilingual sequence efficiency.
2. Architectural choices such as RoPE, RMSNorm, and SwiGLU can be evaluated within a fixed parameter budget.
3. A better optimizer is only useful if it improves the actual measured objective.
4. Evaluation should be defined in a tokenizer-independent unit when comparing tokenization strategies.

## Reproducibility

The training script records the seed, model configuration, number of steps, and training loss curve in the checkpoint. The project also keeps a run log and design notes so experiments can be compared rather than judged from a single final number. fileciteturn3file0

## Next Improvements

Potential extensions include:

- add a small inference/generation CLI
- add automated tests for tokenizer round-trip and model shape invariants
- add loss/bpb visualization from `RUNLOG.md`
- add a benchmark table for multiple language subsets
- add GitHub Actions for linting and smoke tests

## Author

**Atul1127**

Built with Python, PyTorch, and SentencePiece.
