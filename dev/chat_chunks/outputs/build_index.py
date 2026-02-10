import os
import json
import numpy as np
import faiss
import openai
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from same directory as this script
load_dotenv(Path(__file__).parent / ".env")

# Use old OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Paths relative to this script
HERE = Path(__file__).parent
# Find all chunk files in this directory matching *_chunks.txt
CHUNK_FILES = sorted(HERE.glob("*_chunks.txt"))
OUT_INDEX = HERE / "index.faiss"
OUT_META = HERE / "chunks.json"


def main():

    chunks = []
    # Read and combine all chunk files in this folder
    for f in CHUNK_FILES:
        text = f.read_text(encoding="utf-8")
        file_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        print(f"Loaded {len(file_chunks)} chunks from {f.name}")
        chunks.extend(file_chunks)

    embeddings = []

    print(f"Embedding {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):

        # Try preferred model first, then fall back to an older embedding model if permission denied
        vec = None
        last_exc = None
        for model_name in ("text-embedding-3-small", "text-embedding-ada-002"):
            try:
                resp = openai.Embedding.create(
                    model=model_name,
                    input=chunk
                )
                vec = resp["data"][0]["embedding"]
                if i % 200 == 0:
                    print(f"using embedding model: {model_name}")
                break
            except Exception as e:
                last_exc = e
                # try next model
                continue
        if vec is None:
            # re-raise last exception for visibility
            raise last_exc
        embeddings.append(vec)

        if i % 50 == 0:
            print(f"{i}/{len(chunks)}")

    dim = len(embeddings[0])
    vectors = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, str(OUT_INDEX))

    OUT_META.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Index: {OUT_INDEX}, Metadata: {OUT_META}")


if __name__ == "__main__":
    main()
