"""Score a checkpoint in bits per byte (bpb)."""
import argparse
import json
import math
import os

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = Config()
    for k, v in ckpt["config"].items():
        setattr(cfg, k, v)
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def load_checkpoint_tokenizer(ckpt):
    """Load the tokenizer recorded by newer checkpoints and verify vocab size.
    Older checkpoints fall back to tokenizer.py's normal discovery behavior.
    """
    name = ckpt.get("tokenizer_model")
    if name:
        path = name if os.path.isabs(name) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"checkpoint expects tokenizer '{name}', but it was not found at {path}")
        tok = tokenizer_mod.load(path)
    else:
        tok = tokenizer_mod.load()
    if tok.vocab_size != ckpt["config"]["vocab_size"]:
        raise ValueError(
            f"tokenizer/model vocab mismatch: tokenizer={tok.vocab_size}, "
            f"checkpoint={ckpt['config']['vocab_size']}")
    return tok


@torch.no_grad()
def bits_per_byte(model, cfg, tok, text):
    n_bytes = len(text.encode("utf-8"))
    if n_bytes == 0:
        raise ValueError("eval text is empty")
    id_list = tok.encode(text)
    if tok.decode(id_list) != text:
        raise ValueError("tokenizer is not lossless: decode(encode(text)) != text")
    ids = torch.tensor(id_list, dtype=torch.long)
    if cfg.block_size < 2:
        raise ValueError("block_size must be >= 2")
    block, stride = cfg.block_size, max(1, cfg.block_size // 2)
    total_nll, n_scored = 0.0, 0
    scored = 1
    while scored < len(ids):
        start = max(0, scored - stride)
        end = min(len(ids), start + block)
        window = ids[start:end]
        logits, _ = model(window[None, :])
        logp = torch.log_softmax(logits[0], dim=-1)
        targets = ids[start + 1:end]
        nll = -logp[torch.arange(len(targets)), targets]
        offset = scored - (start + 1)
        total_nll += nll[offset:].sum().item()
        n_scored += len(nll) - offset
        scored = end
    if n_scored == 0:
        raise ValueError("nothing scored: tokenizer produced fewer than 2 tokens")
    return total_nll / math.log(2) / n_bytes, n_scored, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt.pt")
    ap.add_argument("--text_file", required=True)
    args = ap.parse_args()
    model, cfg, ckpt = load_model(args.checkpoint)
    tok = load_checkpoint_tokenizer(ckpt)
    with open(args.text_file, encoding="utf-8") as f:
        text = f.read()
    bpb, n_scored, n_tokens = bits_per_byte(model, cfg, tok, text)
    print(json.dumps({
        "bpb": round(bpb, 4),
        "n_params": model.n_params(),
        "steps": ckpt.get("steps"),
        "tokens_in_eval": n_tokens,
        "tokens_scored": n_scored,
    }))


if __name__ == "__main__":
    main()
