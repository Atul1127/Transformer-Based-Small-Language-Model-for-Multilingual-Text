"""Minimal Gradio demo for the released multilingual checkpoint.

Run after installing Gradio:
    pip install gradio
    python app.py

The app presents the model as a small experimental LM rather than a
production chatbot. Generation is CPU-friendly but can be slow.
"""
import os

import gradio as gr
import torch

from generate import generate, load_model

CHECKPOINT = os.environ.get("CHECKPOINT", "ckpt.pt")
MODEL, TOKENIZER = load_model(CHECKPOINT)


def predict(prompt, temperature, top_k, max_new_tokens, seed):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a prompt first.")
    torch.manual_seed(int(seed))
    return generate(
        MODEL,
        TOKENIZER,
        prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
    )


with gr.Blocks(title="Small Multilingual Transformer") as demo:
    gr.Markdown(
        """# Small Multilingual Transformer
A ~1.83M-parameter decoder-only Transformer trained from scratch.

**Experimental model:** outputs may be incoherent because the model was intentionally trained under a strict compute and parameter budget."""
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                value="भारत एक",
                lines=4,
                placeholder="Enter an English or Hindi prompt...",
            )
            generate_button = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
            top_k = gr.Slider(0, 100, value=40, step=1, label="Top-k")
            max_new_tokens = gr.Slider(1, 200, value=50, step=1, label="Max new tokens")
            seed = gr.Number(value=42, precision=0, label="Seed")

    output = gr.Textbox(label="Generated text", lines=10)
    generate_button.click(predict, [prompt, temperature, top_k, max_new_tokens, seed], output)
    prompt.submit(predict, [prompt, temperature, top_k, max_new_tokens, seed], output)


if __name__ == "__main__":
    demo.launch()
