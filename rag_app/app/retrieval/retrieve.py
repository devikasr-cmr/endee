import json
import numpy as np
from sentence_transformers import SentenceTransformer

LOCAL_VECTOR_STORE = "data/processed/vectors.json"


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_context(query, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode(query)

    with open(LOCAL_VECTOR_STORE, "r", encoding="utf-8") as f:
        data = json.load(f)

    scored = []
    for item in data:
        vec = np.array(item["vector"])
        score = cosine_similarity(query_vec, vec)
        scored.append((score, item["metadata"]["text"]))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [text for _, text in scored[:top_k]]


if __name__ == "__main__":
    results = retrieve_context("What is Endee?")
    print("\n🔍 Retrieved Contexts:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r}")
