# 🧠 DisceplineAI — Your Personal Discipline Coach

Welcome to **DisceplineAI**, a lean yet powerful LoRA‑tuned Mistral‑7B model trained to think and advise like the greatest self‑help gurus. Fine‑tuned on the core teachings of:

- **The 48 Laws of Power** (Robert Greene)  
- **The Way of the Superior Man** (David Deida)  
- **Psycho‑Cybernetics** (Maxwell Maltz)  
- **How to Win Friends and Influence People** (Dale Carnegie)  

_DisceplineAI doesn’t regurgitate quotes—it **embodies** these teachings in its own words, delivering razor‑sharp, actionable guidance._

---

## 📋 Table of Contents

1. [Key Features](#key-features)  
2. [Quickstart](#quickstart)  
3. [Installation](#installation)  
4. [Usage](#usage)  
   - [CLI](#cli)  
   - [Python API](#python-api)  
5. [Model Details](#model-details)  
6. [Training Notebook](#training-notebook)  
7. [Folder Structure](#folder-structure)  
8. [Contributing](#contributing)  
9. [License](#license)  

---

## 🔑 Key Features

- **Persona‑Driven Advice**: Channels the mindset of four legendary self‑help authors.  
- **4‑bit QLoRA Efficiency**: Runs on consumer‑grade GPUs with minimal memory.  
- **Original Output**: No verbatim quoting—fresh, concise paraphrasing.  
- **Easy Integration**: CLI tool, Python API, or web demo via Gradio/Streamlit.  

---

## 🚀 Quickstart

```bash
# 1. Clone this repo
git clone https://github.com/omk4rr/DisceplineAI.git
cd DisceplineAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the CLI
python inference.py "How do I build lasting habits?"
```
## 📥 Installation
```bash
# (Optional) Create & activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install project dependencies
pip install -r requirements.txt
```
## 🛠️ Usage
 CLI
```bash
python inference.py "What’s the secret to influence people?"
```

## Python API
```bash
from inference import ask_discepline

answer = ask_discepline("How can I overcome procrastination?")
print("DisceplineAI:", answer)
```

## 🤖 Model Details
Base: mistralai/Mistral-7B-v0.1

Fine‑Tune Method: LoRA + QLoRA (4‑bit)

Adapter Repo: omk4rr/DisceplineAI

Training Data: Passages and paraphrases derived from the four self‑help classics listed above.

## 📂 Folder Structure
```bash
DisceplineAI/
├── DiceplineAI.ipynb            # Colab notebook: data prep & fine‑tuning
├── inference.py                 # CLI & importable inference script
├── requirements.txt             # Python dependencies
├── adapter_config.json          # LoRA adapter configuration
├── adapter_model.safetensors    # Trained LoRA weights
├── .gitattributes               # Git LFS config for large files
├── README.md                    # This detailed overview
```
🤝 Contributing
We welcome pull requests to:

Improve prompts and persona

Add new self‑help book adapters

Build a web UI (Gradio/Streamlit)

Optimize performance (8‑bit, CPU offload)

Please fork the repo, create a feature branch, and submit a PR – we’ll review ASAP.

# 📜 License
Distributed under the MIT License. See LICENSE for details.


# DisceplineAI doesn’t quote self‑help books—it thinks like them.


