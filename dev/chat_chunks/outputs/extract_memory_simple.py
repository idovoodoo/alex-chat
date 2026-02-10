import json
import os
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
CHAT_LOGS_DIR = os.path.join(HERE, "..", "..", "chat_logs")
DEFAULT_OUT_PATH = os.path.join(HERE, "memories.json")


def extract_memories_from_text(chat_text: str, chat_name: str, model: str = "gpt-5-mini") -> list[str]:
    """Extract memories from a full chat log in one API call."""
    print(f"Processing {chat_name}...", end=" ", flush=True)
    
    # Truncate if needed (most models have ~128k token limit, roughly 500k chars)
    max_chars = 400000
    if len(chat_text) > max_chars:
        print(f"(truncating from {len(chat_text)} to {max_chars} chars)...", end=" ", flush=True)
        chat_text = chat_text[:max_chars]
    
    prompt = (
        "From this WhatsApp chat log, extract up to 400 concise, durable factual statements "
        "about Alex and the people around him (relationships, names, usernames, pets, recurring "
        "preferences, routines, household rules, possessions, regular activities).\n\n"
        "Rules:\n"
        "- Only stable facts likely to remain true over time\n"
        "- No emotions, no opinions\n"
        "- Exclude one-off events, timestamps, and ephemeral logistics\n"
        "- Prefer very short sentences (5-12 words) in third person\n"
        "- If you can infer a username, nickname, or account id, include it (short)\n"
        "- Output ONLY a JSON array of strings (no surrounding text)\n\n"
        f"Chat log:\n{chat_text}"
    )

    # Wrap call in a small retry loop and parse
    last_err = None
    for attempt in range(1, 4):
        try:
            if _OPENAI_NEW:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You extract structured factual memory from chat logs. Output only JSON."},
                        {"role": "user", "content": prompt},
                    ],
                )
                choice = resp.choices[0] if hasattr(resp, "choices") else resp["choices"][0]
                content = choice.message.content if hasattr(choice, "message") else choice["message"]["content"]
            else:
                resp = client.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You extract structured factual memory from chat logs. Output only JSON."},
                        {"role": "user", "content": prompt},
                    ],
                )
                choice = resp["choices"][0]
                content = choice["message"]["content"]

            # Parse JSON from response
            content = (content or "").strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(line for line in lines if not line.startswith("```"))

            memories = json.loads(content)
            if not isinstance(memories, list):
                raise ValueError(f"Expected JSON array, got {type(memories)}")

            result = [m.strip() for m in memories if isinstance(m, str) and m.strip()]
            print(f"{len(result)} memories")
            return result
        except Exception as e:
            last_err = e
            print(f"Attempt {attempt} failed: {e}")
    print(f"ERROR: all attempts failed: {last_err}")
    return []


def main() -> None:
    out_path = os.getenv("MEMORIES_OUT_PATH", DEFAULT_OUT_PATH)
    model = os.getenv("MEMORY_MODEL", "gpt-5-mini")
    
    all_memories: list[str] = []
    seen: set[str] = set()
    
    print(f"Reading chat logs from {CHAT_LOGS_DIR}...")
    
    # Process each chat log file
    for filename in ["abi-chat.txt", "steve-chat.txt"]:
        filepath = os.path.join(CHAT_LOGS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  Skipping {filename} (not found)")
            continue
        
        with open(filepath, "r", encoding="utf-8") as f:
            chat_text = f.read()
        
        print(f"  {filename}: {len(chat_text)} chars")
        
        memories = extract_memories_from_text(chat_text, filename, model=model)

        # Dedupe and checkpoint after each file
        new_count = 0
        for m in memories:
            key = m.lower()
            if key not in seen:
                seen.add(key)
                all_memories.append(m)
                new_count += 1
        print(f"  Added {new_count} new unique memories from {filename}")

        # checkpoint
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_memories, f, ensure_ascii=False, indent=2)
            print(f"  Checkpointed {len(all_memories)} memories to {out_path}")
        except Exception as e:
            print(f"  Warning: failed to write checkpoint: {e}")
    
    print(f"\nWriting {len(all_memories)} unique memories to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_memories, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Done! Wrote {len(all_memories)} memories to {out_path}")


if __name__ == "__main__":
    main()
