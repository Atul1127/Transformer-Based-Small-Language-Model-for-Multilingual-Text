"""Evaluate the checkpoint on multiple language-specific text files.

Usage:
    python evaluate_languages.py --english data/sample_eval.txt --hindi data/sample_eval_hindi.txt

The script reuses evaluate.py so bpb remains tokenizer-independent. It is
intended for transparent reporting; it does not manufacture language labels
or compare files with different evaluation protocols.
"""
import argparse
import json

from evaluate import bits_per_byte, load_checkpoint_tokenizer, load_model


def score(model, cfg, tok, path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    bpb, scored, tokens = bits_per_byte(model, cfg, tok, text)
    return {"bpb": round(bpb, 4), "tokens_in_eval": tokens, "tokens_scored": scored}


def main():
    parser = argparse.ArgumentParser(description="Per-language bpb evaluation")
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--english")
    parser.add_argument("--hindi")
    args = parser.parse_args()
    if not args.english and not args.hindi:
        parser.error("provide at least one of --english or --hindi")

    model, cfg, ckpt = load_model(args.checkpoint)
    tok = load_checkpoint_tokenizer(ckpt)
    results = {"checkpoint": args.checkpoint, "n_params": model.n_params(), "steps": ckpt.get("steps")}
    if args.english:
        results["english"] = score(model, cfg, tok, args.english)
    if args.hindi:
        results["hindi"] = score(model, cfg, tok, args.hindi)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
