import os
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from app.utils.endee_client import EndeeClient

RAW_DOCS_PATH = "data/raw_docs"
PROCESSED_PATH = "data/processed"
LOCAL_VECTOR_STORE = os.path.join(PROCESSED_PATH, "vectors.json")

COLLECTION_NAME = "rag_docs"
EMBEDDING_DIM = 384
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def load_documents(path):
    documents = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ RAW_DOCS_PATH not found: {path}")

    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            full_path = os.path.join(path, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    documents.append({
                        "source": filename,
                        "text": text
                    })

    return documents


def chunk_text(text, chunk_size, overlap):
    text = text.strip()
    if not text:
        return []

    # Small document → single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def ingest_documents():
    print("🔹 Loading documents...")
    docs = load_documents(RAW_DOCS_PATH)

    if not docs:
        print("⚠️ No documents found to ingest")
        return []

    print(f"📚 Found {len(docs)} document(s)")

    print("🔹 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🔹 Connecting to Endee...")
    client = EndeeClient()

    print(f"🔹 Creating / validating collection '{COLLECTION_NAME}'")
    client.create_collection(COLLECTION_NAME, EMBEDDING_DIM)

    all_chunks = []
    local_vectors = []

    print("🔹 Chunking, embedding & inserting vectors...")
    for doc in tqdm(docs):
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"📄 {doc['source']} → {len(chunks)} chunk(s)")

        if not chunks:
            continue

        embeddings = model.encode(chunks)

        if len(embeddings) != len(chunks):
            print(f"⚠️ Embedding mismatch for {doc['source']}")
            continue

        for i, emb in enumerate(embeddings):
            metadata = {
                "source": doc["source"],
                "chunk_id": i,
                "text": chunks[i]
            }

            # Store in Endee (vector DB)
            client.insert(
                collection=COLLECTION_NAME,
                vector=emb.tolist(),
                metadata=metadata
            )

            # Store locally for retrieval
            local_vectors.append({
                "vector": emb.tolist(),
                "metadata": metadata
            })

            all_chunks.append(metadata)

    # Save local vector cache
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    with open(LOCAL_VECTOR_STORE, "w", encoding="utf-8") as f:
        json.dump(local_vectors, f, indent=2)

    print(f"💾 Saved {len(local_vectors)} vectors locally at {LOCAL_VECTOR_STORE}")

    return all_chunks


if __name__ == "__main__":
    data = ingest_documents()
    print(f"\n✅ Generated {len(data)} chunks with embeddings")
