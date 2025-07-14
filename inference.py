#!/usr/bin/env python
# coding: utf-8

import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rag_utils import retrieve_similar   # RAG helper

# ─── Configuration ───
BASE_MODEL   = "mistralai/Mistral-7B-v0.1"
ADAPTER_REPO = "omk4rr/DiceplineAI"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ─── 4‑bit Quant Config ───
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ─── Tokenizer & Model ───
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config
)
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, device_map="auto")
model.eval()

# ─── Few‑Shot Loader ───
with open("few_shot_qa.jsonl", "r") as f:
    few_shot = [json.loads(line) for line in f]

def build_prompt(user_q: str, use_rag: bool = True, k: int = 3) -> str:
    persona = (
        "You are Discepline AI—an upbeat, stoic coach inspired by Greene, Deida, Maltz, Carnegie.\n\n"
    )
    if use_rag:
        sims = retrieve_similar(user_q, k=k)
        examples = "\n".join(f"Q: {ex['question']}\nA: {ex['answer']}\n" for ex in sims)
    else:
        examples = "\n".join(f"Q: {ex['question']}\nA: {ex['answer']}\n" for ex in few_shot)
    return (
        f"{persona}"
        f"Here are some examples of how you answer:\n\n"
        f"{examples}\n"
        f"Now answer this:\nQ: {user_q}\nA:"
    )

# ─── Inference Function ───
def ask_discepline(prompt: str, max_new_tokens: int = 200, use_rag: bool = True) -> str:
    input_text = build_prompt(prompt, use_rag=use_rag)
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
    
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

# ─── CLI / Main ───
def main():
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        prompt = "How can I build discipline to stay consistent every day?"
    answer = ask_discepline(prompt, use_rag=True)
    print(f"\n🧠 Discepline AI:\n{answer}\n")

if __name__ == "__main__":
    main()
