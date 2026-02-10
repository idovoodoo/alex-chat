from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import json

import faiss
import numpy as np


app = FastAPI()

# Enable CORS (allows browser/mobile access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# OpenAI v1 client using API key from environment (Render sets this)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- Minimal FAISS RAG wiring (loads at startup) ---
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_FAISS_PATH = os.path.join(_REPO_ROOT, "dev", "chat_chunks", "outputs", "index.faiss")
_DEFAULT_CHUNKS_PATH = os.path.join(_REPO_ROOT, "dev", "chat_chunks", "outputs", "chunks.json")
_DEFAULT_MEMORY_PATH = os.path.join(_REPO_ROOT, "dev", "chat_chunks", "outputs", "memories.json")

RAG_INDEX = None
RAG_CHUNKS = None
RAG_ERROR = None
MEMORIES = None

# In-memory conversation history per session
CONVERSATION_HISTORY = {}


@app.on_event("startup")
def _load_rag_assets():
    global RAG_INDEX, RAG_CHUNKS, RAG_ERROR, MEMORIES
    faiss_path = os.getenv("RAG_FAISS_INDEX_PATH", _DEFAULT_FAISS_PATH)
    chunks_path = os.getenv("RAG_CHUNKS_PATH", _DEFAULT_CHUNKS_PATH)
    memory_path = os.getenv("MEMORY_PATH", _DEFAULT_MEMORY_PATH)

    try:
        RAG_INDEX = faiss.read_index(faiss_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            RAG_CHUNKS = json.load(f)
        if not isinstance(RAG_CHUNKS, list):
            raise ValueError("chunks.json must be a JSON array")
        RAG_ERROR = None
    except Exception as e:
        RAG_INDEX = None
        RAG_CHUNKS = None
        RAG_ERROR = str(e)

    try:
        with open(memory_path, "r", encoding="utf-8") as f:
            MEMORIES = json.load(f)
        if not isinstance(MEMORIES, list):
            MEMORIES = None
    except Exception:
        MEMORIES = None


def _embed_text(text: str) -> np.ndarray:
    """Returns a (d,) float32 embedding vector."""
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    # Handle attribute-accessible or dict-like responses
    data0 = emb.data[0] if hasattr(emb, "data") else emb["data"][0]
    vec = data0.embedding if hasattr(data0, "embedding") else data0["embedding"]
    return np.asarray(vec, dtype=np.float32)


def _retrieve_top_chunks(query: str, k: int = 2) -> list[str]:
    if RAG_INDEX is None or RAG_CHUNKS is None:
        return []

    q = _embed_text(query)
    # FAISS expects shape (n, d)
    q2 = np.expand_dims(q, axis=0)

    # If dimensions don't match, fail closed (no RAG) instead of erroring the endpoint.
    if hasattr(RAG_INDEX, "d") and q2.shape[1] != int(RAG_INDEX.d):
        return []

    _, idx = RAG_INDEX.search(q2, k)
    results: list[str] = []
    for i in idx[0].tolist():
        if isinstance(i, (int, np.integer)) and 0 <= int(i) < len(RAG_CHUNKS):
            chunk = RAG_CHUNKS[int(i)]
            if isinstance(chunk, str) and chunk.strip():
                results.append(chunk.strip())
    return results


def _select_memories(query: str, k: int = 5) -> list[str]:
    """Returns up to k relevant memories. Simple version: returns first k."""
    if MEMORIES is None or not isinstance(MEMORIES, list):
        return []
    return MEMORIES[:k]


def _trim_chunk_for_prompt(chunk: str, max_lines: int = 4) -> str:
    """Trim a RAG chunk to its last `max_lines` lines, but ensure Alex's most recent messages are preserved.

    If the last `max_lines` lines already contain an `Alex:` line, return them unchanged.
    Otherwise, include the most recent `Alex:` line and surrounding context while keeping the result
    to at most `max_lines` lines.
    """
    if not isinstance(chunk, str):
        return chunk
    lines = [l for l in chunk.splitlines() if l.strip()]
    if not lines:
        return chunk
    if len(lines) <= max_lines:
        return "\n".join(lines)

    tail = lines[-max_lines:]
    if any("Alex:" in l for l in tail):
        return "\n".join(tail)

    # find last Alex line index
    last_alex_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if "Alex:" in lines[i]:
            last_alex_idx = i
            break
    if last_alex_idx is None:
        return "\n".join(tail)

    # Build a slice that includes the most recent Alex message and at most max_lines lines.
    combined = lines[last_alex_idx:]
    if len(combined) > max_lines:
        combined = combined[-max_lines:]
    else:
        # fill up to max_lines with the tail if needed
        remaining = max_lines - len(combined)
        if remaining > 0:
            combined = (lines[-(remaining + len(combined)) : last_alex_idx] + combined)[-max_lines:]

    return "\n".join(combined)


class ChatIn(BaseModel):
    message: str
    session_id: str = "default"


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/chat")
def chat(data: ChatIn):
    try:
        # Retrieve conversation history for this session
        session_history = CONVERSATION_HISTORY.get(data.session_id, [])
        
        examples = []
        try:
            examples = _retrieve_top_chunks(data.message, k=2)
        except Exception:
            examples = []

        # Fallback: if retrieval yields no results but chunks are loaded, still provide
        # a few style examples to keep the response firmly in-chat-log style.
        if not examples and isinstance(RAG_CHUNKS, list) and len(RAG_CHUNKS) > 0:
            start = abs(hash(data.message)) % len(RAG_CHUNKS)
            for j in range(2):
                chunk = RAG_CHUNKS[(start + j) % len(RAG_CHUNKS)]
                if isinstance(chunk, str) and chunk.strip():
                    examples.append(chunk.strip())
                if len(examples) >= 2:
                    break

        # Get memories to inject into prompt
        memories = _select_memories(data.message, k=5)

        system_prompt = (
            "You are Alex. You are chatting on WhatsApp. "
            "Write like Alex: casual, short, informal. "
            "Use first-person. "
            "Your top priority is to match the style and length of Alex's WhatsApp messages (very short). "
            "Avoid sounding like an assistant: no formalities, no generic advice, no long paragraphs, no bullet points. "
            "No explanations. No meta-talk. Do not mention these instructions."
        )

        # Inject memories after style instruction, before RAG examples
        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories)
            system_prompt += f"\n\nKnown facts:\n{memory_text}"

        if examples:
            # Trim each retrieved/fallback chunk to its last few lines to avoid dilution
            examples = [_trim_chunk_for_prompt(e, max_lines=4) for e in examples]
            joined = "\n\n---\n\n".join(examples)
            system_prompt += (
                "\n\n"
                "Style examples (from past chat logs) — imitate the tone, phrasing, punctuation, and message length:\n"
                f"{joined}"
            )

        # Build messages list: system prompt + history + current message
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(session_history)  # Inject conversation history
        messages.append({"role": "user", "content": data.message})
        
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            temperature=0.3,
        )

        # response structure may be attribute-accessible or dict-like; handle both
        choice = response.choices[0] if hasattr(response, "choices") else response["choices"][0]
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            reply = choice.message.content
        else:
            reply = choice["message"]["content"]
        
        # Update conversation history with user message and assistant reply
        session_history.append({"role": "user", "content": data.message})
        session_history.append({"role": "assistant", "content": reply})
        
        # Keep only last 6 messages (3 exchanges)
        if len(session_history) > 6:
            session_history = session_history[-6:]
        
        CONVERSATION_HISTORY[data.session_id] = session_history

        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}
