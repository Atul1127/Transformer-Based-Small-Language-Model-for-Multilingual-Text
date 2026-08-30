"""Train the small multilingual Transformer.

Uses Muon + AdamW or plain Adam, with linear warmup, cosine decay,
global gradient clipping, and the project's 2,000-step / 2M-parameter caps.
"""
import argparse
import math
import os
import time

import torch

from model import GPT, Config
from muon import Muon
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    if len(ids) <= block:
        raise ValueError(
            f"training corpus is too short: need more than {block} tokens, "
            f"got {len(ids)}")
    ix = torch.randint(len(ids) - block, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def lr_at(step, total_steps, peak_lr, warmup_steps, min_lr_ratio):
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = peak_lr * min_lr_ratio
    return min_lr + coeff * (peak_lr - min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--n_layer", type=int, default=None)
    ap.add_argument("--n_head", type=int, default=None)
    ap.add_argument("--n_embd", type=int, default=None)
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--use_swiglu", type=int, default=None, choices=[0, 1])
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--adamw_lr", type=float, default=3e-3)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--optimizer", default="muon_adamw",
                     choices=["muon_adamw", "adam_all"],
                     help="adam_all = plain Adam over every param, same LR schedule")
    ap.add_argument("--tokenizer_model", default=None)
    args = ap.parse_args()

    if not 1 <= args.steps <= MAX_STEPS:
        raise ValueError(f"steps must be between 1 and {MAX_STEPS}")
    if args.batch < 1:
        raise ValueError("batch must be >= 1")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if args.min_lr_ratio < 0:
        raise ValueError("min_lr_ratio must be >= 0")

    torch.manual_seed(args.seed)
    device = "cpu"

    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    tok = tokenizer_mod.load(args.tokenizer_model)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    if args.n_layer is not None: cfg.n_layer = args.n_layer
    if args.n_head is not None: cfg.n_head = args.n_head
    if args.n_embd is not None: cfg.n_embd = args.n_embd
    if args.block_size is not None: cfg.block_size = args.block_size
    if args.use_swiglu is not None: cfg.use_swiglu = bool(args.use_swiglu)

    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params  (cap {MAX_PARAMS:,})")
    if n > MAX_PARAMS:
        raise ValueError(
            f"model has {n:,} params; cap is {MAX_PARAMS:,} -- "
            "shrink n_embd/vocab_size/n_layer")

    muon_group = list(model.muon_params())
    adamw_group = list(model.adamw_params())
    n_muon = sum(p.numel() for p in muon_group)
    n_adamw = sum(p.numel() for p in adamw_group)
    print(f"optimizer split: {n_muon:,} params -> Muon, "
          f"{n_adamw:,} params -> AdamW")

    if args.optimizer == "muon_adamw":
        muon_opt = Muon(muon_group, lr=args.muon_lr, momentum=0.95)
        adamw_opt = torch.optim.AdamW(adamw_group, lr=args.adamw_lr,
                                       weight_decay=args.weight_decay,
                                       betas=(0.9, 0.95))
        optimizers = [muon_opt, adamw_opt]
    else:
        adam_opt = torch.optim.Adam(model.parameters(), lr=args.adamw_lr,
                                    betas=(0.9, 0.95))
        optimizers = [adam_opt]

    model.train()
    t0 = time.time()
    losses = []
    for step in range(args.steps):
        cur_muon_lr = lr_at(step, args.steps, args.muon_lr,
                             args.warmup_steps, args.min_lr_ratio)
        cur_adamw_lr = lr_at(step, args.steps, args.adamw_lr,
                              args.warmup_steps, args.min_lr_ratio)
        if args.optimizer == "muon_adamw":
            for g in muon_opt.param_groups: g["lr"] = cur_muon_lr
            for g in adamw_opt.param_groups: g["lr"] = cur_adamw_lr
        else:
            for g in optimizers[0].param_groups: g["lr"] = cur_adamw_lr

        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
        for opt in optimizers: opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers: opt.step()
        losses.append(loss.item())

        s = step + 1
        if s % args.log_every == 0 or s == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            cur_lr = optimizers[-1].param_groups[0]["lr"]
            print(f"step {s:5d}  loss {avg:.4f}  lr {cur_lr:.2e}  "
                  f"({(time.time()-t0)/s*1000:.0f} ms/step)")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "seed": args.seed,
                "tokenizer_model": os.path.basename(tok.model_path)
                if hasattr(tok, "model_path") else None,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
