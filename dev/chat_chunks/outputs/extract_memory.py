import json
import os
from typing import Iterable

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

try:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    _OPENAI_NEW = True
except Exception:
    import openai

    client = openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    _OPENAI_NEW = False


HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CHUNKS_PATH = os.path.join(HERE, "chunks.json")
DEFAULT_OUT_PATH = os.path.join(HERE, "memories.json")


def _iter_batches(chunks: list[str], max_chunks_per_batch: int = 8) -> Iterable[list[str]]:
    """Yield small batches of chunks (smaller = faster API calls)."""
    batch: list[str] = []
    for c in chunks:
        if not isinstance(c, str):
            continue
        c = c.strip()
        if not c:
            continue
        batch.append(c)
        if len(batch) >= max_chunks_per_batch:
            yield batch
            batch = []
    if batch:
        yield batch


def _extract_json_list(text: str) -> list[str]:
    """Best-effort parse: expects a JSON list of strings."""
    text = (text or "").strip()
    try:
        data = json.loads(text)
    except Exception:
        # Try to salvage the first JSON array in the output.
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except Exception:
            return []

    if not isinstance(data, list):
        return []
    out: list[str] = []
    for item in data:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out


def extract_memories_from_chunks(chunks: list[str], model: str = "gpt-5-mini") -> list[str]:
    """Extracts deduped, stable memories from WhatsApp chat chunks."""
    memories: list[str] = []
    seen: set[str] = set()

    batches = list(_iter_batches(chunks))
    print(f"Processing {len(chunks)} chunks in {len(batches)} batches...")

    for batch_idx, batch in enumerate(batches, 1):
        print(f"  Batch {batch_idx}/{len(batches)} ({len(batch)} chunks)...", end=" ", flush=True)
        
        chats = "\n\n---\n\n".join(batch)
        prompt = (
            "From these WhatsApp chat excerpts, extract durable factual information "
            "about Alex and the people around him (relationships, names, recurring preferences, "
            "stable constraints like routines or rules).\n\n"
            "Rules:\n"
            "- Only stable facts likely to remain true\n"
            "- No emotions\n"
            "- No temporary events (one-off plans, dates/times, logistics)\n"
            "- No opinions\n"
            "- Short sentences\n"
            "- Third person\n"
            "- Output ONLY a JSON array of strings\n\n"
            "Chats:\n"
            f"{chats}"
        )

        try:
            # Support both new v1 OpenAI client and legacy openai package
            if _OPENAI_NEW:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You extract structured memory. Output only JSON."},
                        {"role": "user", "content": prompt},
                    ],
                )
                choice = resp.choices[0] if hasattr(resp, "choices") else resp["choices"][0]
                content = choice.message.content if hasattr(choice, "message") else choice["message"]["content"]
            else:
                resp = client.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You extract structured memory. Output only JSON."},
                        {"role": "user", "content": prompt},
                    ],
                )
                choice = resp["choices"][0]
                content = choice["message"]["content"]
            
            batch_memories = _extract_json_list(content)
            new_count = 0
            for m in batch_memories:
                key = m.lower()
                if key not in seen:
                    seen.add(key)
                    memories.append(m)
                    new_count += 1
            print(f"{new_count} new memories")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    return memories


def main() -> None:
    chunks_path = os.getenv("CHUNKS_JSON_PATH", DEFAULT_CHUNKS_PATH)
    out_path = os.getenv("MEMORIES_OUT_PATH", DEFAULT_OUT_PATH)
    model = os.getenv("MEMORY_MODEL", "gpt-5-mini")
    max_chunks_env = os.getenv("MAX_CHUNKS")

    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not isinstance(chunks, list):
        raise ValueError("chunks.json must be a JSON array")

    print(f"Loaded {len(chunks)} chunks")
    
    if max_chunks_env:
        try:
            limit = int(max_chunks_env)
            chunks = chunks[:limit]
            print(f"Limited to first {len(chunks)} chunks (MAX_CHUNKS={max_chunks_env})")
        except Exception:
            pass

    memories = extract_memories_from_chunks(chunks, model=model)
    
    print(f"\nWriting {len(memories)} memories to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)

    print(f"✓ Done! Wrote {len(memories)} memories to {out_path}")


if __name__ == "__main__":
    main()
