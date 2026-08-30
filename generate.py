"""Small command-line text generator.

Usage:
    python generate.py --checkpoint ckpt.pt --prompt "भारत एक"
"""
import argparse

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
    return model, tokenizer_mod.load()


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=80, temperature=0.8, top_k=40):
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)

    for _ in range(max_new_tokens):
        context = ids[:, -model.cfg.block_size:]
        logits, _ = model(context)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

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
    args = parser.parse_args()

    model, tok = load_model(args.checkpoint)
    print(generate(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k))


if __name__ == "__main__":
    main()
