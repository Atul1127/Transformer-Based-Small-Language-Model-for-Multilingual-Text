"""Small command-line text generator.

Usage:
    python generate.py --checkpoint ckpt.pt --prompt "भारत एक" --seed 42
"""
import argparse
import os

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod


def load_model(checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = Config()
    for key, value in ckpt["config"].items():
        setattr(cfg, key, value)
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

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
    if tok.vocab_size != cfg.vocab_size:
        raise ValueError(
            f"tokenizer/model vocab mismatch: tokenizer={tok.vocab_size}, "
            f"checkpoint={cfg.vocab_size}")
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=80, temperature=0.8, top_k=40):
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")

    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    if ids.size(1) == 0:
        raise ValueError("prompt must contain at least one encodable token")

    for _ in range(max_new_tokens):
        context = ids[:, -model.cfg.block_size:]
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

    return tok.decode(ids[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text with the small multilingual Transformer")
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for reproducible generation")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tok = load_model(args.checkpoint)
    print(generate(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k))


if __name__ == "__main__":
    main()
