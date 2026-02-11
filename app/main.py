from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
import google.generativeai as genai
import os
import json
import psycopg2
from dotenv import load_dotenv
import logging
from urllib.parse import urlparse
from urllib.parse import parse_qs, urlencode, urlunparse
import socket
import re

import faiss
import numpy as np
import math

# Basic logging - configure before any logging calls
logging.basicConfig(level=logging.INFO)

# Determine repo root early for .env loading
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load environment variables from .env file
# Try multiple locations: repo root, dev/chat_chunks/outputs
_DOTENV_LOADED_FROM = None
_DOTENV_LOAD_RESULT = None
_dotenv_locations = [
    os.path.join(_REPO_ROOT, ".env"),
    os.path.join(_REPO_ROOT, "dev", "chat_chunks", "outputs", ".env"),
]
for _dotenv_path in _dotenv_locations:
    if os.path.exists(_dotenv_path):
        _DOTENV_LOADED_FROM = _dotenv_path
        _DOTENV_LOAD_RESULT = bool(load_dotenv(_dotenv_path))
        logging.info(f"Loaded .env from: {_dotenv_path} (success={_DOTENV_LOAD_RESULT})")
        break
else:
    logging.warning("No .env file found in expected locations")


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

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# --- Minimal FAISS RAG wiring (loads at startup) ---
_DEFAULT_FAISS_PATH = os.path.join(_REPO_ROOT, "dev", "chat_chunks", "outputs", "index.faiss")
_DEFAULT_CHUNKS_PATH = os.path.join(_REPO_ROOT, "dev", "chat_chunks", "outputs", "chunks.json")

RAG_INDEX = None
RAG_CHUNKS = None
RAG_ERROR = None
MEMORIES = None
DB_CONN = None
DB_LAST_ERROR = None

# In-memory conversation history per session
CONVERSATION_HISTORY = {}


@app.on_event("startup")
def _load_rag_assets():
    global RAG_INDEX, RAG_CHUNKS, RAG_ERROR, MEMORIES, DB_CONN
    faiss_path = os.getenv("RAG_FAISS_INDEX_PATH", _DEFAULT_FAISS_PATH)
    chunks_path = os.getenv("RAG_CHUNKS_PATH", _DEFAULT_CHUNKS_PATH)

    try:
        RAG_INDEX = faiss.read_index(faiss_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
    from datetime import datetime
            RAG_CHUNKS = json.load(f)
        if not isinstance(RAG_CHUNKS, list):
            raise ValueError("chunks.json must be a JSON array")
        RAG_ERROR = None
    except Exception as e:
        RAG_INDEX = None
        RAG_CHUNKS = None
        RAG_ERROR = str(e)

    # Connect to Supabase PostgreSQL and load core_memories
    try:
        db_url = os.getenv("SUPABASE_DB_URL")
        if db_url:
            logging.info("SUPABASE_DB_URL found, attempting connection...")

            # Supabase Postgres typically requires SSL. If the URL doesn't specify it,
            # default to sslmode=require.
            try:
                p = urlparse(db_url)
                q = parse_qs(p.query)
                if "sslmode" not in q:
                    q["sslmode"] = ["require"]
                    db_url = urlunparse(p._replace(query=urlencode(q, doseq=True)))
            except Exception:
                pass

            DB_CONN = psycopg2.connect(db_url, connect_timeout=10)
            logging.info("Database connection successful")
            # Load core_memories into MEMORIES
            with DB_CONN.cursor() as cur:
    LAST_LIFE_RECALL = None
                cur.execute("SELECT content FROM core_memories ORDER BY id")
                rows = cur.fetchall()
                MEMORIES = [row[0] for row in rows if row[0]]
            # Log how many memories were loaded
            try:
                count = len(MEMORIES) if isinstance(MEMORIES, list) else 0
                logging.info(f"Loaded {count} core_memories from database")
            except Exception:
                logging.info("Loaded core_memories from database (count unknown)")
        else:
            logging.warning("SUPABASE_DB_URL environment variable not set")
            MEMORIES = None
    except Exception as e:
        # Log the exception for debugging and clear connection/memories
        logging.exception("Failed to connect to Supabase/Postgres or load core_memories")
        global DB_LAST_ERROR
        DB_LAST_ERROR = f"{type(e).__name__}: {e}"
        DB_CONN = None
        MEMORIES = None
    finally:
        # Diagnostic log whether DB_CONN appears connected
        if DB_CONN:
            logging.info("Database connection established")
            DB_LAST_ERROR = None
        else:
            logging.info("Database not connected")


@app.get("/db_status")
def _db_status():
    """Return whether the application can reach the configured Supabase/Postgres DB."""
    try:
        if DB_CONN is None:
            return {"connected": False, "core_memories_count": 0, "error": DB_LAST_ERROR}
        with DB_CONN.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        # include loaded memories count for diagnostics
        try:
            count = len(MEMORIES) if isinstance(MEMORIES, list) else 0
        except Exception:
            count = 0
        return {"connected": True, "core_memories_count": int(count), "error": None}
    except Exception:
        return {"connected": False, "core_memories_count": 0, "error": DB_LAST_ERROR}


@app.get("/debug/db")
def _debug_db():
    """Detailed database diagnostics for debugging."""
    supabase_db_url = os.getenv("SUPABASE_DB_URL")
    parsed = None
    try:
        parsed = urlparse(supabase_db_url) if supabase_db_url else None
    except Exception:
        parsed = None

    sslmode = None
    if parsed and parsed.query:
        try:
            sslmode = (parse_qs(parsed.query).get("sslmode") or [None])[0]
        except Exception:
            sslmode = None

    # Detect Render environment (best-effort)
    render_detected = any(
        os.getenv(k)
        for k in (
            "RENDER",
            "RENDER_SERVICE_ID",
            "RENDER_SERVICE_NAME",
            "RENDER_EXTERNAL_URL",
            "RENDER_INTERNAL_HOSTNAME",
        )
    )

    diagnostics = {
        "render_detected": bool(render_detected),
        "cwd": os.getcwd(),
        "repo_root": _REPO_ROOT,
        "dotenv_loaded_from": _DOTENV_LOADED_FROM,
        "dotenv_load_result": _DOTENV_LOAD_RESULT,
        "dotenv_locations": [{"path": p, "exists": os.path.exists(p)} for p in _dotenv_locations],
        "env_var_set": bool(supabase_db_url),
        "env_keys_with_supabase": [k for k in os.environ.keys() if "SUPABASE" in k.upper()],
        # Safe DB URL hints (no credentials)
        "db_url_scheme": (parsed.scheme if parsed else None),
        "db_url_host": (parsed.hostname if parsed else None),
        "db_url_port": (parsed.port if parsed else None),
        "db_url_db": (parsed.path.lstrip("/") if (parsed and parsed.path) else None),
        "db_url_sslmode": sslmode,
        "db_conn_object": DB_CONN is not None,
        "db_last_error": DB_LAST_ERROR,
        "memories_loaded": isinstance(MEMORIES, list),
        "memories_count": len(MEMORIES) if isinstance(MEMORIES, list) else 0,
        # Also show whether other common envs are set (booleans only)
        "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "gemini_api_key_set": bool(os.getenv("GEMINI_API_KEY")),
    }
    
    # DNS resolution check (helps debug Render/Supabase hostname issues)
    if parsed and parsed.hostname:
        try:
            diagnostics["dns_lookup"] = socket.gethostbyname(parsed.hostname)
        except Exception as e:
            diagnostics["dns_lookup"] = f"failed: {type(e).__name__}: {e}"
    else:
        diagnostics["dns_lookup"] = None

    # Try a test query
    if DB_CONN:
        try:
            with DB_CONN.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            diagnostics["test_query"] = "success"
        except Exception as e:
            diagnostics["test_query"] = f"failed: {str(e)}"
    else:
        diagnostics["test_query"] = "no connection"
    
    # Try to count core_memories in DB
    if DB_CONN:
        try:
            with DB_CONN.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM core_memories")
                count = cur.fetchone()[0]
            diagnostics["db_row_count"] = int(count)
        except Exception as e:
            diagnostics["db_row_count"] = f"error: {str(e)}"
    else:
        diagnostics["db_row_count"] = "no connection"

    # Include last life-memory recall info if available
    try:
        if LAST_LIFE_RECALL is None:
            diagnostics["last_life_memory_recall"] = None
        else:
            # Make timestamp JSON-serializable
            lr = LAST_LIFE_RECALL
            diagnostics["last_life_memory_recall"] = {
                "time_utc": lr.get("time").isoformat() if lr.get("time") else None,
                "session_id": lr.get("session_id"),
                "query_snippet": (lr.get("query")[:200] + "...") if lr.get("query") and len(lr.get("query")) > 200 else lr.get("query"),
                "results_count": int(lr.get("results_count") or 0),
            }
    except Exception:
        diagnostics["last_life_memory_recall"] = None

    return diagnostics


def estimate_tokens(text: str) -> int:
    """Rudimentary token estimator when exact usage isn't available.

    This uses a simple heuristic (roughly 4 chars per token). If the
    model response includes explicit usage fields we'll use those instead.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _embed_text(text: str) -> tuple[np.ndarray, int]:
    """Returns a (d,) float32 embedding vector and an estimated/observed token count.

    Returns (vector, tokens_used). If the embedding response includes a usage
    field we use it; otherwise we fall back to `estimate_tokens`.
    """
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    # Handle attribute-accessible or dict-like responses
    data0 = emb.data[0] if hasattr(emb, "data") else emb["data"][0]
    vec = data0.embedding if hasattr(data0, "embedding") else data0["embedding"]

    # Try to extract usage if present
    tokens_used = None
    if hasattr(emb, "usage"):
        u = emb.usage
        tokens_used = getattr(u, "prompt_tokens", None) or getattr(u, "total_tokens", None)
    elif isinstance(emb, dict) and "usage" in emb:
        u = emb["usage"]
        tokens_used = u.get("prompt_tokens") or u.get("total_tokens")

    if tokens_used is None:
        tokens_used = estimate_tokens(text)

    return np.asarray(vec, dtype=np.float32), int(tokens_used)


def _retrieve_top_chunks(query: str, k: int = 2) -> tuple[list[str], int]:
    """Returns (results, tokens_used_by_embedding).

    If FAISS or chunks aren't available returns ([], 0).
    """
    if RAG_INDEX is None or RAG_CHUNKS is None:
        return [], 0

    q, emb_tokens = _embed_text(query)
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
    return results, int(emb_tokens)


def _select_memories(query: str, k: int = 5) -> list[str]:
    """Returns up to k relevant memories. Simple version: returns first k."""
    if MEMORIES is None or not isinstance(MEMORIES, list):
        return []
    return MEMORIES[:k]


def _reload_memories():
    """Reload memories from Supabase database into the global MEMORIES variable."""
    global MEMORIES
    try:
        if DB_CONN:
            with DB_CONN.cursor() as cur:
                cur.execute("SELECT content FROM core_memories ORDER BY id")
                rows = cur.fetchall()
                MEMORIES = [row[0] for row in rows if row[0]]
        else:
            MEMORIES = None
    except Exception:
        MEMORIES = None

    # Log how many memories are currently loaded
    try:
        count = len(MEMORIES) if isinstance(MEMORIES, list) else 0
        logging.info(f"Reloaded core_memories: {count} items in MEMORIES")
    except Exception:
        logging.info("Reloaded core_memories (count unknown)")


_RECALL_TRIGGERS = {
    "remember",
    "remind",
    "when",
    "trip",
    "vacation",
    "holiday",
    "school",
    "college",
    "uni",
    "university",
    "high school",
    "back then",
    "used to",
    "last time",
}


def _message_suggests_recall(message: str) -> bool:
    if not isinstance(message, str) or not message.strip():
        return False
    m = message.lower()
    return any(t in m for t in _RECALL_TRIGGERS)


def _search_life_memories(message: str, limit: int = 3) -> list[str]:
    """Simple keyword search against life_memories.content (ILIKE)."""
    if DB_CONN is None:
        return []
    if not isinstance(message, str) or not message.strip():
        return []

    # Extract a few meaningful keywords from the message.
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9']+", message)]
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "do",
        "did",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "i'm",
        "im",
        "in",
        "is",
        "it",
        "like",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "that",
        "the",
        "their",
        "then",
        "there",
        "they",
        "this",
        "to",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "who",
        "why",
        "with",
        "you",
        "your",
        "remember",
        "remind",
        "trip",
        "school",
        "back",
        "time",
        "used",
    }

    keywords: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        if len(t) < 4:
            continue
        if t in stop:
            continue
        if t in seen:
            continue
        seen.add(t)
        keywords.append(t)
        if len(keywords) >= 6:
            break

    if not keywords:
        return []

    clauses = " OR ".join(["content ILIKE %s"] * len(keywords))
    params = [f"%{k}%" for k in keywords]

    try:
        with DB_CONN.cursor() as cur:
            cur.execute(
                f"SELECT content FROM life_memories WHERE ({clauses}) LIMIT %s",
                (*params, int(limit)),
            )
            rows = cur.fetchall()
        return [r[0] for r in rows if r and isinstance(r[0], str) and r[0].strip()]
    except Exception:
        return []


def _summarize_conversation(session_history: list) -> str:
    """Summarize a conversation into 1-2 short factual memory lines.
    
    Returns third-person, factual statements with no emotions, opinions, or temporary details.
    Returns empty string if the conversation is too short or summarization fails.
    """
    if not session_history or len(session_history) < 2:
        return ""
    
    # Build conversation text
    conversation_text = ""
    for msg in session_history:
        if msg["role"] == "user":
            conversation_text += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            conversation_text += f"Alex: {msg['content']}\n"
    
    # Create summarization prompt
    summary_prompt = (
        "Summarize this WhatsApp conversation into 1-2 short factual memory lines. "
        "Requirements:\n"
        "- Write in third person\n"
        "- Include only facts, no emotions or opinions\n"
        "- Exclude temporary details (times, dates, immediate plans)\n"
        "- Focus on lasting facts about Alex's life, relationships, or preferences\n"
        "- Each line should be a complete, standalone sentence\n"
        "- If there are no important lasting facts, return NONE\n\n"
        f"Conversation:\n{conversation_text}\n\n"
        "Memory (1-2 lines only):"
    )
    
    try:
        # Use OpenAI to generate summary
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.3,
            max_tokens=100
        )
        
        summary = response.choices[0].message.content.strip()
        
        # Skip if model says there's nothing important
        if summary.upper() == "NONE" or not summary:
            return ""
        
        return summary
    except Exception:
        return ""


def _save_memory(memory_line: str):
    """Append a memory line to Supabase core_memories table and reload MEMORIES.
    
    Args:
        memory_line: A single memory string or multiple lines separated by newlines
    """
    if not memory_line or not memory_line.strip():
        return
    
    try:
        if DB_CONN:
            # Split multi-line memory into individual lines
            new_memories = [line.strip() for line in memory_line.split("\n") if line.strip()]
            
            # Insert each memory into the database
            with DB_CONN.cursor() as cur:
                for mem in new_memories:
                    cur.execute("INSERT INTO core_memories (content) VALUES (%s)", (mem,))
            DB_CONN.commit()
            
            # Reload global MEMORIES
            _reload_memories()
    except Exception:
        pass


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
    user_name: str = "User"
    model: str = "gpt-5-mini"


# Mount static files directory
_STATIC_DIR = os.path.join(_REPO_ROOT, "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Directory where server-side chat logs are stored (safe, controlled)
CHAT_LOGS_DIR = os.path.join(_REPO_ROOT, "dev", "chat_logs")


@app.get("/")
def root():
    """Serve the chat UI."""
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"status": "ok"}


@app.post("/chat")
def chat(data: ChatIn):
    try:
        # Retrieve conversation history for this session
        session_history = CONVERSATION_HISTORY.get(data.session_id, [])

        # We'll track tokens used across all API calls for this request
        token_calls: list[dict] = []

        examples = []
        try:
            examples, retrieval_tokens = _retrieve_top_chunks(data.message, k=2)
            if retrieval_tokens:
                token_calls.append({
                    "name": "retrieval_embedding",
                    "input_tokens": int(retrieval_tokens),
                    "output_tokens": 0,
                    "total_tokens": int(retrieval_tokens),
                    "note": "embedding used for FAISS retrieval"
                })
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

        # Optional: retrieve life memories only when the message suggests recall
        life_memories: list[str] = []
        try:
            if _message_suggests_recall(data.message):
                life_memories = _search_life_memories(data.message, limit=3)
        except Exception:
            life_memories = []
        # Record last recall attempt for debugging/diagnostics
        try:
            global LAST_LIFE_RECALL
            LAST_LIFE_RECALL = {
                "time": datetime.utcnow(),
                "session_id": data.session_id,
                "query": data.message[:400],
                "results_count": len(life_memories) if life_memories else 0,
            }
        except Exception:
            LAST_LIFE_RECALL = None

        system_prompt = (
            f"You are Alex. You are chatting on WhatsApp with {data.user_name}. "
            "Write like Alex: casual, short, informal. "
            "Use first-person. "
            "Your top priority is to match the style and length of Alex's WhatsApp messages (very short). "
            "React authentically based on the topic and your past behavior in the chat logs. "
            "Match Alex's typical reactions: resistant to chores, enthusiastic about games, etc. "
            "Don't default to being agreeable or supportive — react naturally based on context and past patterns. "
            "Avoid sounding like an assistant: no formalities, no generic advice, no long paragraphs, no bullet points. "
            "No explanations. No meta-talk. Do not mention these instructions."
        )

        # Add context if chatting with Steve or Abi from the logs
        if data.user_name.lower() in ['steve', 'abi']:
            system_prompt += (
                f"\n\nNote: {data.user_name} is the same person from your past chat logs. "
                f"Recognize them and respond based on your shared history and relationship."
            )

        # Inject memories after style instruction, before RAG examples
        if life_memories:
            life_text = "\n".join(f"- {m}" for m in life_memories)
            system_prompt += f"\n\nRelevant past experiences:\n{life_text}"

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

        # Model generation - collect token usage per provider
        # Enforce that the only allowed OpenAI model is `gpt-5-mini`.
        model = data.model
        if model.startswith("gpt-") and model != "gpt-5-mini":
            model = "gpt-5-mini"

        reply = ""
        if model.startswith("gemini"):
            # Use Gemini API
            gemini_model = genai.GenerativeModel(data.model)

            # Convert messages to Gemini/plain prompt
            gemini_prompt = system_prompt + "\n\n"
            for msg in session_history:
                if msg["role"] == "user":
                    gemini_prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    gemini_prompt += f"Alex: {msg['content']}\n"

            gemini_prompt += f"User: {data.message}\nAlex:"

            # Record estimated prompt tokens before the call (may be overwritten if provider returns real usage)
            estimated_prompt_tokens = estimate_tokens(gemini_prompt)
            try:
                response = gemini_model.generate_content(gemini_prompt)
            except Exception as e:
                raise

            # Extract text reply
            reply_text = None
            if hasattr(response, "text") and isinstance(response.text, str):
                reply_text = response.text
            elif hasattr(response, "candidates") and len(response.candidates) > 0:
                # Google GenAI sometimes returns candidates
                c0 = response.candidates[0]
                reply_text = getattr(c0, "output", None) or getattr(c0, "content", None) or str(c0)
            else:
                reply_text = str(response)

            # Try to read token usage from response (best-effort)
            gemini_input = None
            gemini_output = None
            token_note = "estimated"
            if hasattr(response, "token_usage"):
                tu = response.token_usage
                gemini_input = getattr(tu, "input_tokens", None) or getattr(tu, "prompt_tokens", None)
                gemini_output = getattr(tu, "output_tokens", None) or getattr(tu, "completion_tokens", None)
                token_note = "reported"
            elif isinstance(response, dict) and "tokenUsage" in response:
                tu = response["tokenUsage"]
                gemini_input = tu.get("inputTokens") or tu.get("promptTokens")
                gemini_output = tu.get("outputTokens") or tu.get("completionTokens")
                token_note = "reported"

            if gemini_input is None:
                gemini_input = estimated_prompt_tokens
            if gemini_output is None:
                gemini_output = estimate_tokens(reply_text)

            token_calls.append({
                "name": "gemini_generate",
                "input_tokens": int(gemini_input),
                "output_tokens": int(gemini_output),
                "total_tokens": int(gemini_input) + int(gemini_output),
                "note": token_note,
            })

            reply = reply_text
        else:
            # Use OpenAI API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )

            # response structure may be attribute-accessible or dict-like; handle both
            choice = response.choices[0] if hasattr(response, "choices") else response["choices"][0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                reply = choice.message.content
            else:
                reply = choice["message"]["content"]

            # Try to extract usage from the response
            usage = None
            if hasattr(response, "usage"):
                usage = response.usage
            elif isinstance(response, dict) and "usage" in response:
                usage = response["usage"]

            if usage is not None:
                # usage may be attribute-like or dict-like
                if hasattr(usage, "prompt_tokens") or hasattr(usage, "completion_tokens"):
                    input_t = getattr(usage, "prompt_tokens", None) or getattr(usage, "prompt_tokens", None)
                    output_t = getattr(usage, "completion_tokens", None) or getattr(usage, "completion_tokens", None)
                    total_t = getattr(usage, "total_tokens", None) or getattr(usage, "total_tokens", None)
                else:
                    input_t = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                    output_t = usage.get("completion_tokens") if isinstance(usage, dict) else None
                    total_t = usage.get("total_tokens") if isinstance(usage, dict) else None

                # fallbacks
                if input_t is None:
                    input_t = estimate_tokens(json.dumps(messages))
                if output_t is None:
                    output_t = estimate_tokens(reply)
                if total_t is None:
                    total_t = int(input_t) + int(output_t)

                token_calls.append({
                    "name": "openai_chat",
                    "input_tokens": int(input_t),
                    "output_tokens": int(output_t),
                    "total_tokens": int(total_t),
                    "note": "reported" if (hasattr(usage, "total_tokens") or (isinstance(usage, dict) and "total_tokens" in usage)) else "estimated",
                })
            else:
                # No usage reported: estimate
                in_est = estimate_tokens(json.dumps(messages))
                out_est = estimate_tokens(reply)
                token_calls.append({
                    "name": "openai_chat",
                    "input_tokens": int(in_est),
                    "output_tokens": int(out_est),
                    "total_tokens": int(in_est) + int(out_est),
                    "note": "estimated",
                })
        
        # Update conversation history with user message and assistant reply
        session_history.append({"role": "user", "content": data.message})
        session_history.append({"role": "assistant", "content": reply})
        
        # Keep only last 6 messages (3 exchanges)
        if len(session_history) > 6:
            session_history = session_history[-6:]
        
        CONVERSATION_HISTORY[data.session_id] = session_history

        # Aggregate totals
        total_input = sum(int(c.get("input_tokens", 0)) for c in token_calls)
        total_output = sum(int(c.get("output_tokens", 0)) for c in token_calls)
        total_all = sum(int(c.get("total_tokens", 0)) for c in token_calls)

        return {
            "reply": reply,
            "tokens": {
                "calls": token_calls,
                "input_tokens": int(total_input),
                "output_tokens": int(total_output),
                "total_tokens": int(total_all),
            },
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/chat_logs")
def list_chat_logs():
    """Return a list of server-side chat log files (name, size, mtime)."""
    try:
        files = []
        if os.path.isdir(CHAT_LOGS_DIR):
            for fname in sorted(os.listdir(CHAT_LOGS_DIR)):
                # Only expose simple text files to avoid surprises
                if not fname.lower().endswith('.txt'):
                    continue
                # Prevent path traversal by rejecting names with separators
                if '/' in fname or '\\' in fname or '..' in fname:
                    continue
                path = os.path.join(CHAT_LOGS_DIR, fname)
                try:
                    st = os.stat(path)
                    files.append({
                        'name': fname,
                        'size': int(st.st_size),
                        'mtime': int(st.st_mtime),
                    })
                except Exception:
                    continue
        return {'files': files}
    except Exception as e:
        return {'error': str(e)}


@app.delete("/chat_logs/{name}")
def delete_chat_log(name: str):
    """Delete a server-side chat log file by name. Returns 404 if not found."""
    # Basic validation to avoid path traversal
    if '/' in name or '\\' in name or '..' in name:
        raise HTTPException(status_code=400, detail='Invalid filename')

    path = os.path.join(CHAT_LOGS_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail='File not found')

    try:
        os.remove(path)
        return {'status': 'ok', 'deleted': name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class NewChatIn(BaseModel):
    session_id: str = "default"


@app.post("/new_chat")
def new_chat(data: NewChatIn):
    """End the current chat session and start a new one.
    
    Summarizes the previous conversation and saves it to Supabase core_memories.
    """
    try:
        # Get the conversation history for this session
        session_history = CONVERSATION_HISTORY.get(data.session_id, [])
        
        # If there's conversation history, summarize and save it
        if session_history:
            summary = _summarize_conversation(session_history)
            if summary:
                _save_memory(summary)
        
        # Clear the conversation history for this session
        if data.session_id in CONVERSATION_HISTORY:
            del CONVERSATION_HISTORY[data.session_id]
        
        return {
            "status": "ok",
            "message": "New chat started",
            "summary_saved": bool(session_history and summary)
        }
    except Exception as e:
        return {"error": str(e)}