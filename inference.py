#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ─── Configuration ───
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
ADAPTER_REPO = "omk4rr/DiceplineAI"  # Your HF LoRA repo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 4-bit Quantization Config ───
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ─── Tokenizer ───
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# ─── Load Base Model ───
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config
)

# ─── Attach LoRA Adapter ───
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, device_map="auto")
model.eval()

# ─── Inference Function ───
def ask_discepline(prompt: str, max_new_tokens: int = 200) -> str:
    persona = (
        "You are Discepline AI—an upbeat, insightful coach who channels the wisdom "
        "of James Clear, Robert Greene, Maxwell Maltz, and others.\n\n"
    )
    input_text = f"{persona}### User:\n{prompt}\n### Discepline AI:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            no_repeat_ngram_size=3
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    return response.strip()

# ─── CLI / Direct Execution ───
def main():
    import sys
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        prompt = "How can I build discipline to stay consistent every day?"
    response = ask_discepline(prompt)
    print(f"\n🧠 Discepline AI:\n{response}\n")

if __name__ == "__main__":
    main()
