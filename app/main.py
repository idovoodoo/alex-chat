from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
try:
    # openai>=1.x
    from openai import OpenAI  # type: ignore
    _OPENAI_V1 = True
except Exception:  # pragma: no cover
    # openai<1.x fallback
    OpenAI = None  # type: ignore
    _OPENAI_V1 = False
    import openai  # type: ignore
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
from datetime import datetime

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
if _OPENAI_V1:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai


def _openai_chat_completion(*, model: str, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None):
    """Compatibility wrapper for OpenAI chat completions across SDK versions."""
    if _OPENAI_V1:
        kwargs = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            # GPT-5 models use `max_completion_tokens` instead of `max_tokens`.
            if isinstance(model, str) and model.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
        resp = client.chat.completions.create(**kwargs)
        # v1: reply in resp.choices[0].message.content
        choice0 = resp.choices[0]
        reply_text = choice0.message.content
        usage = getattr(resp, "usage", None)
        return reply_text, usage

    # openai<1.x
    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = client.ChatCompletion.create(**kwargs)
    reply_text = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage")
    return reply_text, usage


def _openai_embedding(*, model: str, input_text: str):
    """Compatibility wrapper for OpenAI embeddings across SDK versions."""
    if _OPENAI_V1:
        resp = client.embeddings.create(model=model, input=input_text)
        data0 = resp.data[0]
        vec = data0.embedding
        usage = getattr(resp, "usage", None)
        return vec, usage

    resp = client.Embedding.create(model=model, input=input_text)
    vec = resp["data"][0]["embedding"]
    usage = resp.get("usage")
    return vec, usage

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
LAST_LIFE_RECALL = None
LIFE_RECALL_DEBUG = None
LAST_NEW_CHAT_DEBUG = None

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
            # Load core memories from unified memories table
            with DB_CONN.cursor() as cur:
                cur.execute("SELECT content FROM memories WHERE type = 'core' ORDER BY id")
                rows = cur.fetchall()
                MEMORIES = [row[0] for row in rows if row[0]]
            # Log how many memories were loaded
            try:
                count = len(MEMORIES) if isinstance(MEMORIES, list) else 0
                logging.info(f"Loaded {count} core memories from database")
            except Exception:
                logging.info("Loaded core memories from database (count unknown)")
        else:
            logging.warning("SUPABASE_DB_URL environment variable not set")
            MEMORIES = None
    except Exception as e:
        # Log the exception for debugging and clear connection/memories
        logging.exception("Failed to connect to Supabase/Postgres or load core memories")
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


def _ensure_db_connection():
    """Ensure database connection is alive, reconnect if needed."""
    global DB_CONN, DB_LAST_ERROR
    
    # If no connection exists, try to establish one
    if DB_CONN is None:
        try:
            db_url = os.getenv("SUPABASE_DB_URL")
            if not db_url:
                return False
            
            # Add sslmode if not present
            try:
                p = urlparse(db_url)
                q = parse_qs(p.query)
                if "sslmode" not in q:
                    q["sslmode"] = ["require"]
                    db_url = urlunparse(p._replace(query=urlencode(q, doseq=True)))
            except Exception:
                pass
            
            DB_CONN = psycopg2.connect(db_url, connect_timeout=10)
            DB_LAST_ERROR = None
            logging.info("Database connection re-established")
            return True
        except Exception as e:
            DB_LAST_ERROR = f"{type(e).__name__}: {e}"
            logging.error(f"Failed to establish database connection: {DB_LAST_ERROR}")
            return False
    
    # Connection exists, check if it's alive
    try:
        with DB_CONN.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return True
    except Exception:
        # Connection is dead, close it and try to reconnect
        try:
            DB_CONN.close()
        except Exception:
            pass
        DB_CONN = None
        
        # Try to reconnect
        return _ensure_db_connection()


@app.get("/db_status")
def _db_status():
    """Return whether the application can reach the configured Supabase/Postgres DB."""
    try:
        if not _ensure_db_connection():
            return {"connected": False, "cached_core_memories_count": 0, "error": DB_LAST_ERROR}
        with DB_CONN.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        # include loaded memories count for diagnostics
        try:
            count = len(MEMORIES) if isinstance(MEMORIES, list) else 0
        except Exception:
            count = 0
        return {"connected": True, "cached_core_memories_count": int(count), "error": None}
    except Exception:
        return {"connected": False, "cached_core_memories_count": 0, "error": DB_LAST_ERROR}


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
        "_section_environment": "=== ENVIRONMENT ===",
        "render_detected": bool(render_detected),
        "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        "gemini_api_key_set": bool(os.getenv("GEMINI_API_KEY")),
        
        "_section_database": "=== DATABASE ===",
        "db_url_scheme": (parsed.scheme if parsed else None),
        "db_url_host": (parsed.hostname if parsed else None),
        "db_url_port": (parsed.port if parsed else None),
        "db_conn_alive": DB_CONN is not None,
        "db_last_error": DB_LAST_ERROR,
        
        "_section_memory_counts": "=== MEMORY COUNTS ===",
        "cached_core_memories": len(MEMORIES) if isinstance(MEMORIES, list) else 0,
        
        "_section_patterns": "=== DETECTION PATTERNS ===",
        "recall_triggers": len(_RECALL_TRIGGERS),
        "remember_when_patterns": len(_REMEMBER_WHEN_PATTERNS),
        "past_question_patterns": len(_PAST_QUESTION_PATTERNS),
    }

    # Include last new-chat (conversation summarization) info
    diagnostics["_section_last_new_chat"] = "=== LAST NEW CHAT (MEMORY EXTRACTION) ==="
    try:
        diagnostics["last_new_chat"] = LAST_NEW_CHAT_DEBUG if isinstance(LAST_NEW_CHAT_DEBUG, dict) else "No /new_chat calls yet"
    except Exception as e:
        diagnostics["last_new_chat"] = f"error: {str(e)}"
    
    # Try to count memories in DB (unified table)
    if _ensure_db_connection():
        try:
            with DB_CONN.cursor() as cur:
                # Count core memories
                cur.execute("SELECT COUNT(*) FROM memories WHERE type = 'core'")
                core_count = cur.fetchone()[0]
                # Count life memories
                cur.execute("SELECT COUNT(*) FROM memories WHERE type = 'life'")
                life_count = cur.fetchone()[0]
            diagnostics["db_core_memories"] = int(core_count)
            diagnostics["db_life_memories"] = int(life_count)
        except Exception as e:
            diagnostics["db_core_memories"] = f"error: {str(e)}"
            diagnostics["db_life_memories"] = f"error: {str(e)}"
    else:
        diagnostics["db_core_memories"] = "connection failed"
        diagnostics["db_life_memories"] = "connection failed"

    # Include last life-memory recall info if available
    diagnostics["_section_last_recall"] = "=== LAST LIFE MEMORY SEARCH ==="
    try:
        if LIFE_RECALL_DEBUG is not None and isinstance(LIFE_RECALL_DEBUG, dict):
            # Format nicely for readability
            recall_info = {
                "query": LIFE_RECALL_DEBUG.get("search_message", "N/A"),
                "keywords_extracted": LIFE_RECALL_DEBUG.get("keywords", []),
                "is_past_question": LIFE_RECALL_DEBUG.get("is_past_question", False),
                "is_remember_when": LIFE_RECALL_DEBUG.get("is_remember_when", False),
                "recall_triggered": LIFE_RECALL_DEBUG.get("recall_triggered", False),
                "results_found": LIFE_RECALL_DEBUG.get("results_count", 0),
                "llm_bypassed": LIFE_RECALL_DEBUG.get("llm_bypassed", False),
                "bypass_reason": LIFE_RECALL_DEBUG.get("bypass_reason", "N/A"),
            }
            if LIFE_RECALL_DEBUG.get("error"):
                recall_info["error"] = LIFE_RECALL_DEBUG["error"]
            if LIFE_RECALL_DEBUG.get("results_preview"):
                recall_info["results_preview"] = LIFE_RECALL_DEBUG["results_preview"]
            diagnostics["last_recall_attempt"] = recall_info
        else:
            diagnostics["last_recall_attempt"] = "No searches yet"
    except Exception as e:
        diagnostics["last_recall_attempt"] = f"error: {str(e)}"

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
    vec, usage = _openai_embedding(model="text-embedding-3-small", input_text=text)

    tokens_used = None
    if usage is not None:
        if isinstance(usage, dict):
            tokens_used = usage.get("prompt_tokens") or usage.get("total_tokens")
        else:
            tokens_used = getattr(usage, "prompt_tokens", None) or getattr(usage, "total_tokens", None)

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
        if _ensure_db_connection():
            with DB_CONN.cursor() as cur:
                cur.execute("SELECT content FROM memories WHERE type = 'core' ORDER BY id")
                rows = cur.fetchall()
                MEMORIES = [row[0] for row in rows if row[0]]
        else:
            MEMORIES = None
    except Exception:
        MEMORIES = None

    # Log how many memories are currently loaded
    try:
        count = len(MEMORIES) if isinstance(MEMORIES, list) else 0
        logging.info(f"Reloaded core memories: {count} items in MEMORIES")
    except Exception:
        logging.info("Reloaded core memories (count unknown)")


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

# "Do you remember when" style patterns
_REMEMBER_WHEN_PATTERNS = [
    r"\bdo\s+you\s+remember\s+(when|that|the)",
    r"\bremember\s+when\b",
    r"\bremember\s+that\s+time\b",
    r"\bthat\s+time\s+(when|we|you)\b",
    r"\bback\s+when\b",
    r"\byou\s+remember\s+(when|that|the)\b",
]

# Past factual question patterns
_PAST_QUESTION_PATTERNS = [
    r"\b(where|what|when|who|how)\s+(did|have|has|was|were|had)\b",
    r"\b(where|what|when|who)\s+have\s+you\s+(been|gone|done|seen|visited)\b",
    r"\bhave\s+you\s+(been|gone|visited|done|seen)\b",
    r"\bdid\s+you\s+(go|visit|travel|see|do)\b",
    r"\bwhere\s+(have|did)\b",
    r"\bwhat\s+(happened|did you do)\b",
    r"\bwhen\s+(did|was|were)\b",
]


def _is_past_factual_question(message: str) -> bool:
    """Detect if the message is asking about past events/experiences."""
    if not isinstance(message, str) or not message.strip():
        return False
    m = message.lower()
    for pattern in _PAST_QUESTION_PATTERNS:
        if re.search(pattern, m):
            return True
    return False


def _is_remember_when_prompt(message: str) -> bool:
    """Detect 'do you remember when' style prompts."""
    if not isinstance(message, str) or not message.strip():
        return False
    m = message.lower()
    for pattern in _REMEMBER_WHEN_PATTERNS:
        if re.search(pattern, m):
            return True
    return False


def _generate_clarification_question(message: str) -> str:
    """Generate a casual clarification question in Alex's style."""
    options = [
        "which time?",
        "when was that?",
        "what year?",
        "when?",
        "which one?",
        "need more details",
        "refresh my memory?",
    ]
    # Pick based on message hash for consistency
    return options[abs(hash(message)) % len(options)]


def _message_suggests_recall(message: str) -> bool:
    global LIFE_RECALL_DEBUG
    if not isinstance(message, str) or not message.strip():
        return False
    m = message.lower()
    triggered = any(t in m for t in _RECALL_TRIGGERS)
    matched_triggers = [t for t in _RECALL_TRIGGERS if t in m] if triggered else []
    
    if LIFE_RECALL_DEBUG is None:
        LIFE_RECALL_DEBUG = {}
    LIFE_RECALL_DEBUG["recall_triggered"] = triggered
    LIFE_RECALL_DEBUG["matched_triggers"] = matched_triggers
    
    logging.info(f"Life memory recall check: triggered={triggered}, matched={matched_triggers}")
    return triggered


def _search_life_memories(message: str, limit: int = 3) -> list[str]:
    """Simple keyword search against memories table WHERE type='life'."""
    global LIFE_RECALL_DEBUG
    
    if LIFE_RECALL_DEBUG is None:
        LIFE_RECALL_DEBUG = {}
    
    if not _ensure_db_connection():
        LIFE_RECALL_DEBUG["error"] = "DB connection unavailable"
        logging.warning("Life memory search skipped: DB connection unavailable")
        return []
    if not isinstance(message, str) or not message.strip():
        LIFE_RECALL_DEBUG["error"] = "empty message"
        logging.warning("Life memory search skipped: empty message")
        return []
    
    LIFE_RECALL_DEBUG["search_message"] = message[:100]
    logging.info(f"Starting life memory search for message: {message[:100]}...")

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
        LIFE_RECALL_DEBUG["keywords"] = []
        LIFE_RECALL_DEBUG["error"] = "no keywords extracted"
        logging.info("Life memory search: no keywords extracted from message")
        return []

    LIFE_RECALL_DEBUG["keywords"] = keywords
    logging.info(f"Life memory search keywords: {keywords}")
    clauses = " OR ".join(["content ILIKE %s"] * len(keywords))
    params = [f"%{k}%" for k in keywords]

    try:
        with DB_CONN.cursor() as cur:
            query = f"SELECT content FROM memories WHERE type = 'life' AND ({clauses}) LIMIT %s"
            LIFE_RECALL_DEBUG["sql_query"] = query
            LIFE_RECALL_DEBUG["sql_params"] = params
            logging.info(f"Executing life memory query with {len(keywords)} keywords, limit={limit}")
            cur.execute(query, (*params, int(limit)))
            rows = cur.fetchall()
        results = [r[0] for r in rows if r and isinstance(r[0], str) and r[0].strip()]
        LIFE_RECALL_DEBUG["results_count"] = len(results)
        LIFE_RECALL_DEBUG["results_preview"] = [r[:100] + "..." if len(r) > 100 else r for r in results[:3]]
        logging.info(f"Life memory search returned {len(results)} results")
        return results
    except Exception as e:
        LIFE_RECALL_DEBUG["error"] = f"{type(e).__name__}: {e}"
        logging.error(f"Life memory search failed: {type(e).__name__}: {e}")
        return []


def _summarize_conversation(session_history: list) -> str:
    """Extract ONE durable factual memory from conversation.
    
    Returns a single third-person, present-tense factual statement.
    No emotions, no temporary details, no speculation.
    Returns empty string if no durable fact exists or extraction fails.
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
    
    # Create strict summarization prompt
    summary_prompt = (
        "Extract ONE durable factual memory from this conversation.\n"
        "Requirements:\n"
        "- Third person\n"
        "- Present tense\n"
        "- No emotions\n"
        "- No temporary details\n"
        "- No speculation\n"
        "- Return ONLY ONE sentence\n"
        "- If no durable fact exists, return: NONE\n\n"
        f"Conversation:\n{conversation_text}\n\n"
        "Memory:"
    )
    
    try:
        # GPT-5 models can spend a lot of tokens on reasoning; keep enough headroom
        # so we reliably get an actual output sentence.
        summary, _ = _openai_chat_completion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.2,
            max_tokens=256,
        )

        summary = (summary or "").strip()

        # Retry once if the model produced no visible output (e.g. used budget on reasoning).
        if not summary:
            summary, _ = _openai_chat_completion(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            summary = (summary or "").strip()
        
        # Skip if model says there's nothing important
        if summary.upper() == "NONE" or not summary:
            return ""
        
        # Take only the first line if multiple were returned
        first_line = summary.split('\n')[0].strip()
        return first_line
    except Exception:
        return ""


def _check_duplicate_memory(new_memory: str, threshold: float = 0.85, check_life_memories: bool = True) -> bool:
    """Check if a memory is a duplicate using cosine similarity.
    
    Args:
        new_memory: The memory text to check
        threshold: Cosine similarity threshold (default 0.85)
        check_life_memories: If True, check against life memories; if False, check against MEMORIES
    
    Returns:
        True if duplicate found (similarity > threshold), False otherwise
    """
    if not new_memory or not new_memory.strip():
        return True  # Empty memory is considered duplicate
    
    # Get memory cache
    if check_life_memories:
        # Query life memories from database
        try:
            if _ensure_db_connection():
                with DB_CONN.cursor() as cur:
                    cur.execute("SELECT content FROM memories WHERE type = 'life' ORDER BY id")
                    rows = cur.fetchall()
                    memory_cache = [row[0] for row in rows if row[0]]
            else:
                memory_cache = []
        except Exception:
            memory_cache = []
    else:
        memory_cache = MEMORIES
    
    if not memory_cache or not isinstance(memory_cache, list) or len(memory_cache) == 0:
        return False  # No existing memories, so not a duplicate
    
    try:
        # Generate embedding for new memory
        new_embedding, _ = _embed_text(new_memory)
        
        # Compare against all existing memories
        for existing_memory in memory_cache:
            if not existing_memory or not isinstance(existing_memory, str):
                continue
            
            # Generate embedding for existing memory
            existing_embedding, _ = _embed_text(existing_memory)
            
            # Calculate cosine similarity
            dot_product = np.dot(new_embedding, existing_embedding)
            norm_new = np.linalg.norm(new_embedding)
            norm_existing = np.linalg.norm(existing_embedding)
            
            if norm_new == 0 or norm_existing == 0:
                continue
            
            similarity = dot_product / (norm_new * norm_existing)
            
            # Check if above threshold
            if similarity > threshold:
                logging.info(f"Duplicate memory detected (similarity: {similarity:.3f})")
                return True
        
        return False
    except Exception as e:
        logging.error(f"Error checking duplicate memory: {type(e).__name__}: {e}")
        return False  # On error, proceed with insert to avoid data loss


def _save_memory(memory_line: str):
    """Append a memory line to Supabase memories table and reload MEMORIES.
    
    Args:
        memory_line: A single memory string or multiple lines separated by newlines
    """
    if not memory_line or not memory_line.strip():
        return
    
    try:
        if _ensure_db_connection():
            # Split multi-line memory into individual lines
            new_memories = [line.strip() for line in memory_line.split("\n") if line.strip()]
            
            # Insert each memory into the database
            with DB_CONN.cursor() as cur:
                for mem in new_memories:
                    cur.execute("INSERT INTO memories (content, type) VALUES (%s, %s)", (mem, 'core'))
            DB_CONN.commit()
            
            # Reload global MEMORIES
            _reload_memories()
    except Exception:
        pass


def _save_life_memory(memory_line: str):
    """Append a memory line to Supabase memories table with type='life'.
    
    Args:
        memory_line: A single memory string or multiple lines separated by newlines
    """
    if not memory_line or not memory_line.strip():
        return
    
    try:
        if _ensure_db_connection():
            # Split multi-line memory into individual lines
            new_memories = [line.strip() for line in memory_line.split("\n") if line.strip()]
            
            # Insert each memory into the database
            with DB_CONN.cursor() as cur:
                for mem in new_memories:
                    cur.execute("INSERT INTO memories (content, type) VALUES (%s, %s)", (mem, 'life'))
            DB_CONN.commit()
            logging.info(f"Saved {len(new_memories)} life memories to database")
    except Exception as e:
        logging.error(f"Failed to save life_memory: {type(e).__name__}: {e}")


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
        # Retrieve (or create) conversation history for this session.
        # Keep a stable, per-session list so we never lose prior turns.
        session_history = CONVERSATION_HISTORY.get(data.session_id)
        if not isinstance(session_history, list):
            session_history = []
            CONVERSATION_HISTORY[data.session_id] = session_history

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

        # Optional: retrieve life memories only when the message suggests recall or is a past factual question
        life_memories: list[str] = []
        global LIFE_RECALL_DEBUG
        LIFE_RECALL_DEBUG = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": data.message[:100]
        }
        
        # Check what type of question this is
        suggests_recall = _message_suggests_recall(data.message)
        is_past_q = _is_past_factual_question(data.message)
        is_remember = _is_remember_when_prompt(data.message)
        
        LIFE_RECALL_DEBUG["suggests_recall"] = suggests_recall
        LIFE_RECALL_DEBUG["is_past_question"] = is_past_q
        LIFE_RECALL_DEBUG["is_remember_when"] = is_remember
        
        try:
            if suggests_recall or is_past_q or is_remember:
                life_memories = _search_life_memories(data.message, limit=3)
                LIFE_RECALL_DEBUG["search_executed"] = True
                LIFE_RECALL_DEBUG["results_count"] = len(life_memories)
                logging.info(f"Life memories retrieved: {len(life_memories)} items")
            else:
                LIFE_RECALL_DEBUG["search_executed"] = False
                LIFE_RECALL_DEBUG["skipped_reason"] = "message does not suggest recall, past question, or remember_when"
                logging.info("Life memory search skipped: message does not suggest recall or past question")
        except Exception as e:
            LIFE_RECALL_DEBUG["error"] = f"{type(e).__name__}: {e}"
            logging.error(f"Life memory retrieval error: {type(e).__name__}: {e}")
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

        # Check if this is a past factual question with no supporting memories
        is_past_question = _is_past_factual_question(data.message)
        is_remember_when = _is_remember_when_prompt(data.message)
        LIFE_RECALL_DEBUG["is_past_question"] = is_past_question
        LIFE_RECALL_DEBUG["is_remember_when"] = is_remember_when
        LIFE_RECALL_DEBUG["llm_bypassed"] = False
        
        # If "do you remember when" style with no memories, ask for clarification
        if is_remember_when and not life_memories:
            LIFE_RECALL_DEBUG["llm_bypassed"] = True
            LIFE_RECALL_DEBUG["bypass_reason"] = "remember_when prompt with no matching memories - asking for clarification"
            
            reply = _generate_clarification_question(data.message)
            
            # Update history
            session_history.append({"role": "user", "content": data.message})
            session_history.append({"role": "assistant", "content": reply})
            CONVERSATION_HISTORY[data.session_id] = session_history
            
            return {
                "reply": reply,
                "tokens": {
                    "calls": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            }
        
        # Otherwise, if it's a past question with no memories, return uncertainty
        if is_past_question and not life_memories:
            # Bypass LLM - return a short uncertainty response in Alex's style
            LIFE_RECALL_DEBUG["llm_bypassed"] = True
            LIFE_RECALL_DEBUG["bypass_reason"] = "past question with no life_memories"
            
            fallback_responses = [
                "idk",
                "can't remember",
                "not sure tbh",
                "don't remember that",
                "no idea",
            ]
            # Pick based on message hash for consistency
            reply = fallback_responses[abs(hash(data.message)) % len(fallback_responses)]
            
            # Update history
            session_history.append({"role": "user", "content": data.message})
            session_history.append({"role": "assistant", "content": reply})
            CONVERSATION_HISTORY[data.session_id] = session_history
            
            return {
                "reply": reply,
                "tokens": {
                    "calls": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            }

        system_prompt = (
            f"You are Alex. You are chatting on WhatsApp with {data.user_name}. "
            "Write like Alex: casual, short, informal. "
            "Use first-person. "
            "Your top priority is to match the style and length of Alex's WhatsApp messages (very short). "
            "React authentically based on the topic and your past behavior in the chat logs. "
            "Alex is generally cooperative and friendly. "
            "Show reluctance or resistance ONLY when the context strongly suggests it (being told to do chores, follow rules, or face restrictions). "
            "For neutral topics, questions, or casual conversation, respond naturally without defaulting to negativity. "
            "Don't be overly enthusiastic or assistant-like — keep it casual and real. "
            "Avoid sounding like an assistant: no formalities, no generic advice, no long paragraphs, no bullet points. "
            "No explanations. No meta-talk. Do not mention these instructions.\n\n"
            "CRITICAL: For questions about past events or experiences, ONLY use facts explicitly present in the provided memories. "
            "If the information isn't there, respond with a short uncertainty phrase like 'idk', 'can't remember', or 'not sure'. "
            "NEVER invent or guess places, dates, events, or people."
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
            system_prompt += f"\n\nRelevant past experiences (use ONLY these for past event questions):\n{life_text}"

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

        # Build messages list: system prompt + full session history + current user message
        messages = [{"role": "system", "content": system_prompt}, *session_history, {"role": "user", "content": data.message}]

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
            reply, usage = _openai_chat_completion(model=model, messages=messages)

            input_t = None
            output_t = None
            total_t = None

            if usage is not None:
                if isinstance(usage, dict):
                    input_t = usage.get("prompt_tokens")
                    output_t = usage.get("completion_tokens")
                    total_t = usage.get("total_tokens")
                else:
                    input_t = getattr(usage, "prompt_tokens", None)
                    output_t = getattr(usage, "completion_tokens", None)
                    total_t = getattr(usage, "total_tokens", None)

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
                "note": "reported" if total_t is not None and usage is not None else "estimated",
            })
        
        # Update conversation history with user message and assistant reply
        session_history.append({"role": "user", "content": data.message})
        session_history.append({"role": "assistant", "content": reply})
        
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
    
    Summarizes the previous conversation, checks for duplicates using embedding similarity,
    and saves to the unified memories table (type='life') if not a duplicate.
    """
    global LAST_NEW_CHAT_DEBUG
    LAST_NEW_CHAT_DEBUG = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": data.session_id,
    }

    try:
        # Get the conversation history for this session
        session_history = CONVERSATION_HISTORY.get(data.session_id, [])

        LAST_NEW_CHAT_DEBUG["history_messages"] = int(len(session_history) if isinstance(session_history, list) else 0)
        
        summary_saved = False
        duplicate_detected = False
        extracted_memory = None
        
        # If there's conversation history, summarize and save it
        if session_history:
            # Extract ONE durable factual memory
            summary = _summarize_conversation(session_history)
            extracted_memory = summary

            LAST_NEW_CHAT_DEBUG["extracted_memory"] = summary
            
            # Log extracted memory for debugging
            if summary:
                logging.info(f"Extracted memory: {summary}")
            else:
                logging.info("No durable memory extracted from conversation")
                LAST_NEW_CHAT_DEBUG["note"] = "no durable memory extracted"
            
            if summary:
                # Check for duplicates using cosine similarity (threshold 0.85) against life memories
                is_duplicate = _check_duplicate_memory(summary, threshold=0.85, check_life_memories=True)
                
                if is_duplicate:
                    duplicate_detected = True
                    logging.info(f"Memory not saved (duplicate): {summary}")
                else:
                    # Not a duplicate, save to memories table (type='life')
                    _save_life_memory(summary)
                    summary_saved = True
                    logging.info(f"New life_memory saved: {summary}")
        else:
            LAST_NEW_CHAT_DEBUG["note"] = "no session history"
        
        # Clear the conversation history for this session
        if data.session_id in CONVERSATION_HISTORY:
            del CONVERSATION_HISTORY[data.session_id]

        LAST_NEW_CHAT_DEBUG["summary_saved"] = bool(summary_saved)
        LAST_NEW_CHAT_DEBUG["duplicate_detected"] = bool(duplicate_detected)
        
        return {
            "status": "ok",
            "message": "New chat started",
            "summary_saved": summary_saved,
            "duplicate_detected": duplicate_detected,
            "extracted_memory": extracted_memory
        }
    except Exception as e:
        logging.error(f"Error in new_chat: {type(e).__name__}: {e}")
        try:
            LAST_NEW_CHAT_DEBUG["error"] = f"{type(e).__name__}: {e}"
        except Exception:
            pass
        return {"error": str(e)}