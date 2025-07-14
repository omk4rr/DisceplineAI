# rag_utils.py

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1) Load & embed your Q&A bank
embedder = SentenceTransformer("all-MiniLM-L6-v2")
with open("few_shot_qa.jsonl", "r") as f:
    qa = [json.loads(line) for line in f]

questions  = [ex["question"] for ex in qa]
embeddings = embedder.encode(questions, convert_to_numpy=True)

# 2) Build FAISS index
dim   = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# 3) Retrieval function
def retrieve_similar(user_q: str, k: int = 5):
    vec = embedder.encode([user_q], convert_to_numpy=True)
    _, I = index.search(vec, k)
    return [qa[i] for i in I[0]]
