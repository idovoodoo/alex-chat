from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
# OpenRouter is used for chat generation. OpenAI is used for embeddings.
from openai import OpenAI  # type: ignore
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
import hashlib
import unicodedata

# Basic logging - configure before any logging calls
# Only show WARNING and above in server logs (INFO goes to browser console via /debug/last_console)
logging.basicConfig(level=logging.WARNING)

# Suppress httpx INFO logs (HTTP requests to OpenRouter)
logging.getLogger("httpx").setLevel(logging.WARNING)

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


# OpenRouter exposes OpenAI-compatible Chat Completions and Embeddings APIs.
_OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-5.6-luna")
# Reasoning effort per task ("low" | "medium" | "high"). Chat defaults to low for
# snappy replies; memory extraction defaults to high for better judgment.
_OPENROUTER_CHAT_EFFORT = os.getenv("OPENROUTER_CHAT_REASONING", "low")
_OPENROUTER_MEMORY_EFFORT = os.getenv("OPENROUTER_MEMORY_REASONING", "high")
_OPENROUTER_TAG_EFFORT = os.getenv("OPENROUTER_TAG_REASONING", "low")
_OPENROUTER_CLEANUP_EFFORT = os.getenv("OPENROUTER_CLEANUP_REASONING", "medium")
_OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "openai/text-embedding-3-small")
openrouter_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=_OPENROUTER_BASE_URL,
)
openai_embedding_client = openrouter_client


def _openrouter_chat_completion(*, model: str, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None, reasoning_effort: str | None = None):
    """Call OpenRouter through its OpenAI-compatible Chat Completions API.

    reasoning_effort: optional "low" | "medium" | "high" — passed through
    OpenRouter's `reasoning` parameter to control how hard the model thinks.
    """
    kwargs = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    resp = openrouter_client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    reply_text = getattr(msg, "content", None) or getattr(msg, "reasoning_content", None) or ""
    return reply_text, getattr(resp, "usage", None)


def _openai_embeddings(*, input_texts: list[str]):
    """Create one or more embeddings using OpenAI's embeddings API."""
    if not input_texts:
        return [], None
    response = openai_embedding_client.embeddings.create(
        model=_OPENAI_EMBEDDING_MODEL,
        input=input_texts,
    )
    vectors = [item.embedding for item in response.data]
    if len(vectors) != len(input_texts):
        raise ValueError(
            f"OpenAI returned {len(vectors)} vectors for {len(input_texts)} texts"
        )
    return vectors, getattr(response, "usage", None)


def _normalize_memory_text(text: str) -> str:
    """Normalize memory text for duplicate detection.
    
    Applies:
    - Unicode normalization (NFKC)
    - Lowercase
    - Whitespace normalization
    - Simple verb canonicalization
    - Sorted person lists for consistent ordering
    
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    # Unicode normalize (NFKC combines compatibility forms)
    text = unicodedata.normalize("NFKC", text)
    
    # Lowercase
    text = text.lower()
    
    # Normalize whitespace (collapse multiple spaces, strip)
    text = " ".join(text.split())
    
    # Simple verb canonicalization (past/progressive to simple form)
    # "used to live" -> "lived", "is living" -> "live", etc.
    verb_replacements = {
        r"\bused to live\b": "lived",
        r"\bused to work\b": "worked",
        r"\bis living\b": "live",
        r"\bare living\b": "live",
        r"\bwas living\b": "lived",
        r"\bwere living\b": "lived",
        r"\bis working\b": "work",
        r"\bare working\b": "work",
        r"\bwas working\b": "worked",
        r"\bwere working\b": "worked",
    }
    
    for pattern, replacement in verb_replacements.items():
        text = re.sub(pattern, replacement, text)
    
    # Sort person names in "X and Y" patterns for consistency
    # This handles "Steve and Alex" vs "Alex and Steve"
    def sort_names(match):
        parts = match.group(0).split(" and ")
        if len(parts) == 2:
            # Sort alphabetically
            sorted_parts = sorted([p.strip() for p in parts])
            return " and ".join(sorted_parts)
        return match.group(0)
    
    # Match patterns like "Name and Name" (capitalized words)
    text = re.sub(r'\b[A-Z][a-z]+ and [A-Z][a-z]+\b', sort_names, text, flags=0)
    # Re-lowercase after name sorting
    text = text.lower()
    
    return text


def _compute_memory_hash(text: str) -> str:
    """Compute SHA256 hash of normalized memory text.
    
    Args:
        text: Memory text (will be normalized before hashing)
    
    Returns:
        Hex digest of SHA256 hash
    """
    normalized = _normalize_memory_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# --- Minimal FAISS RAG wiring (loads at startup) ---
_DEFAULT_FAISS_PATH = os.path.join(_REPO_ROOT, "dev", "chat_chunks", "outputs", "index.faiss")
_DEFAULT_CHUNKS_PATH = os.path.join(_REPO_ROOT, "dev", "chat_chunks", "outputs", "chunks.json")

RAG_INDEX = None
RAG_CHUNKS = None
RAG_ERROR = None
MEMORIES = None
CORE_MEMORY_TEXTS: list[str] | None = None
CORE_MEMORY_EMBEDDINGS: np.ndarray | None = None  # shape (n, d), row-normalized
CORE_MEMORY_EMBEDDINGS_ERROR: str | None = None
LIFE_MEMORY_TEXTS: list[str] | None = None
LIFE_MEMORY_EMBEDDINGS: np.ndarray | None = None  # shape (n, d), row-normalized
LIFE_MEMORY_EMBEDDINGS_ERROR: str | None = None
DB_CONN = None
DB_LAST_ERROR = None
LAST_LIFE_RECALL = None
LIFE_RECALL_DEBUG = None
CORE_MEMORY_DEBUG = None
LAST_NEW_CHAT_DEBUG = None
LAST_DEBUG_CONSOLE: str | None = None

# Per-session progress state for /new_chat operations (frontend can poll)
NEW_CHAT_PROGRESS: dict = {}

# In-memory conversation history per session
CONVERSATION_HISTORY = {}
USER_NAME_BY_SESSION = {}  # Track user_name for pronoun replacement in memories
# Per-session token logs (temporary runtime-only, cleared after /new_chat)
SESSION_TOKEN_LOGS: dict = {}


# ===========================================================================================
# CONFIGURATION: Retrieval and Memory Injection Settings
# ===========================================================================================
# These parameters control how many memories/chunks are retrieved and injected into prompts,
# and the similarity thresholds used to filter relevant results.

# --- RAG (Retrieval-Augmented Generation) Configuration ---
# Controls style example retrieval from chat log chunks via FAISS semantic search
RAG_CHUNKS_TO_RETRIEVE = 3          # Number of chat log chunks to retrieve for style examples
RAG_FALLBACK_EXAMPLES = 3            # Number of random chunks to use when retrieval fails
RAG_CHUNK_MAX_LINES = 4              # Max lines to keep from each chunk (keeps most recent exchanges)

# --- Core Memory Configuration ---
# Controls factual memory injection (e.g., "Alex lives in Birmingham", "Alex's dad is Steve")
CORE_MEMORY_MAX_INJECT = 6           # Maximum core memories to inject into prompt
CORE_MEMORY_MIN_SIMILARITY = 0.22    # Absolute minimum cosine similarity to consider relevant
CORE_MEMORY_RELATIVE_GATE = 0.85     # Relative threshold: keep results >= (best_score * this value)

# Questions about identity can be phrased very differently from the stored fact
# (for example, "where are you from?" vs. "Alex is Norwegian."). Keep these
# facts available even when embedding similarity is weaker than an unrelated
# location or style example.
IDENTITY_QUESTION_PATTERNS = (
    r"\bwhat(?:'s| is)\s+(?:your|alex'?s)\s+(?:nationality|ethnicity)\b",
    r"\bwhat\s+nationality\b",
    r"\bwhere\s+(?:are|is)\s+(?:you|alex)\s+from\b",
    r"\bwhat\s+(?:country|place)\s+(?:are|is)\s+(?:you|alex)\s+from\b",
)
IDENTITY_MEMORY_PATTERNS = (
    r"\b(?:nationality|ethnicity)\b",
    r"\b(?:norwegian|british|english|scottish|welsh|irish|swedish|danish|finnish|german|french|spanish|italian|polish)\b",
)
FACT_QUESTION_PATTERNS = (
    r"^(?:who|what|where|when|which|how)\b",
    r"\b(?:do|does|did|is|are|was|were|has|have|had)\s+(?:you|alex)\b",
    r"\b(?:tell me|remind me)\s+(?:about|who|what|where|when)\b",
)

# --- Life Memory Configuration: Recall Mode ---
# Explicit recall triggers (e.g., "remember when", past factual questions)
# Retrieves detailed episodic memories with moderate filtering
LIFE_RECALL_MAX_INJECT = 6           # Maximum life memories for recall-triggered queries
LIFE_RECALL_MIN_SIMILARITY = 0.22    # Absolute minimum similarity threshold for recall mode
LIFE_RECALL_RELATIVE_GATE = 0.85     # Relative threshold: keep results >= (best_score * this value)

# --- Life Memory Configuration: Contextual Mode ---
# Passive background injection on every message (when recall mode doesn't trigger)
# Uses strict filtering to minimize prompt bloat and only inject highly relevant memories
LIFE_CONTEXTUAL_MAX_INJECT = 2       # Maximum life memories for passive contextual injection
LIFE_CONTEXTUAL_MIN_SIMILARITY = 0.35  # Stricter threshold for contextual mode (vs 0.22 for recall)

# --- Duplicate Detection Configuration ---
# Controls similarity threshold for detecting duplicate memories before insertion
DUPLICATE_SIMILARITY_THRESHOLD = 0.85  # Cosine similarity above which memories are considered duplicates

# --- Debug and Logging Configuration ---
DEBUG_CONSOLE_TRUNCATE_LIMIT = 16000  # Max characters for debug console output before truncation

# ===========================================================================================


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
            
            # Run database migrations (add normalized_hash column if missing)
            try:
                with DB_CONN.cursor() as cur:
                    # Check if normalized_hash column exists
                    cur.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name='memories' AND column_name='normalized_hash'
                    """)
                    if not cur.fetchone():
                        logging.info("Adding normalized_hash column to memories table...")
                        cur.execute("ALTER TABLE memories ADD COLUMN normalized_hash TEXT")
                        DB_CONN.commit()
                        logging.info("Migration complete: normalized_hash column added")
                    else:
                        logging.info("normalized_hash column already exists")

                    # Attempt to create a UNIQUE index on (type, normalized_hash).
                    # But first check for existing duplicate hashes which would prevent index creation.
                    try:
                        cur.execute("SELECT normalized_hash, COUNT(*) FROM memories WHERE normalized_hash IS NOT NULL GROUP BY normalized_hash HAVING COUNT(*) > 1")
                        dup_rows = cur.fetchall()
                        if dup_rows and len(dup_rows) > 0:
                            # There are duplicates; do not attempt to create unique index.
                            logging.warning("Cannot create unique index idx_memories_type_hash: duplicate normalized_hash values exist. Please deduplicate before adding a unique constraint.")
                            logging.warning("Example SQL to find duplicates: SELECT normalized_hash, COUNT(*) FROM memories WHERE normalized_hash IS NOT NULL GROUP BY normalized_hash HAVING COUNT(*) > 1;")
                        else:
                            try:
                                cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_type_hash ON memories(type, normalized_hash)")
                                DB_CONN.commit()
                                logging.info("Unique index idx_memories_type_hash created (or already exists)")
                            except Exception as ie:
                                logging.warning(f"Failed to create unique index idx_memories_type_hash: {type(ie).__name__}: {ie}")
                                try:
                                    DB_CONN.rollback()
                                except Exception:
                                    pass
                    except Exception as e:
                        logging.warning(f"Failed to check for duplicate normalized_hash values: {type(e).__name__}: {e}")
                        try:
                            DB_CONN.rollback()
                        except Exception:
                            pass
            except Exception as e:
                logging.warning(f"Migration warning: {type(e).__name__}: {e}")
                # Continue even if migration fails
                try:
                    DB_CONN.rollback()
                except Exception:
                    pass
            
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

    # Build in-memory similarity index for core memories (best-effort)
    try:
        _build_core_memory_embeddings()
    except Exception:
        # Never crash startup due to memory embedding.
        pass

    # Build in-memory similarity index for life memories (best-effort)
    try:
        _build_life_memory_embeddings()
    except Exception:
        # Never crash startup due to life memory embedding.
        pass


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
        "openrouter_api_key_set": bool(os.getenv("OPENROUTER_API_KEY")),
        "openrouter_model": _OPENROUTER_MODEL,
        "openrouter_embedding_model": _OPENAI_EMBEDDING_MODEL,
        
        "_section_database": "=== DATABASE ===",
        "db_url_scheme": (parsed.scheme if parsed else None),
        "db_url_host": (parsed.hostname if parsed else None),
        "db_url_port": (parsed.port if parsed else None),
        "db_conn_alive": DB_CONN is not None,
        "db_last_error": DB_LAST_ERROR,
        
        "_section_memory_counts": "=== MEMORY COUNTS ===",
        "cached_core_memories": len(MEMORIES) if isinstance(MEMORIES, list) else 0,
        "core_memory_embeddings_built": bool(CORE_MEMORY_EMBEDDINGS is not None and CORE_MEMORY_TEXTS is not None),
        "core_memory_embeddings_count": int(len(CORE_MEMORY_TEXTS) if isinstance(CORE_MEMORY_TEXTS, list) else 0),
        "life_memory_embeddings_built": bool(LIFE_MEMORY_EMBEDDINGS is not None and LIFE_MEMORY_TEXTS is not None),
        "life_memory_embeddings_count": int(len(LIFE_MEMORY_TEXTS) if isinstance(LIFE_MEMORY_TEXTS, list) else 0),
        "life_memory_embeddings_error": LIFE_MEMORY_EMBEDDINGS_ERROR,
        "core_memory_embeddings_error": CORE_MEMORY_EMBEDDINGS_ERROR,
        
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
    
    # Show current conversation history state (for debugging)
    diagnostics["_section_session_state"] = "=== ACTIVE SESSIONS ==="
    try:
        session_info = {}
        for session_id, history in CONVERSATION_HISTORY.items():
            if isinstance(history, list):
                session_info[session_id] = {
                    "message_count": len(history),
                    "last_3_messages": [
                        f"{msg.get('role', '?')}: {str(msg.get('content', ''))[:50]}..."
                        for msg in history[-3:]
                    ] if len(history) > 0 else []
                }
            else:
                session_info[session_id] = "invalid (not a list)"
        diagnostics["active_sessions"] = session_info if session_info else "No active sessions"
    except Exception as e:
        diagnostics["active_sessions"] = f"error: {str(e)}"
    
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
                "retrieval_mode": LIFE_RECALL_DEBUG.get("retrieval_mode", "N/A"),
                "semantic_search_always_enabled": LIFE_RECALL_DEBUG.get("semantic_search_always_enabled", False),
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

    # Also print to server console (Render logs) so this is visible without UI.
    try:
        payload = json.dumps(diagnostics, ensure_ascii=False, default=str)
        if len(payload) > 12000:
            payload = payload[:12000] + "…(truncated)"
        logging.info(f"/debug/db diagnostics: {payload}")
    except Exception:
        logging.info("/debug/db diagnostics: (failed to serialize)")

    return diagnostics


@app.get("/debug/last_console")
def _debug_last_console():
    """Return the last DEBUG_CONSOLE payload (for frontend-only browser logging).

    The frontend can fetch this and `console.log` it so server logs are not noisy.
    """
    try:
        if LAST_DEBUG_CONSOLE:
            try:
                return json.loads(LAST_DEBUG_CONSOLE)
            except Exception:
                return {"raw": LAST_DEBUG_CONSOLE}
        return {"raw": None}
    except Exception:
        return {"raw": None}


def estimate_tokens(text: str) -> int:
    """Rudimentary token estimator when exact usage isn't available.

    This uses a simple heuristic (roughly 4 chars per token). If the
    model response includes explicit usage fields we'll use those instead.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _embed_text(text: str, embedding_type: str = "query") -> tuple[np.ndarray, int]:
    """Returns a (d,) float32 embedding vector and an estimated/observed token count.

    Returns (vector, tokens_used). If the embedding response includes a usage
    field we use it; otherwise we fall back to `estimate_tokens`.
    """
    vectors, usage = _openai_embeddings(input_texts=[text])
    vec = vectors[0]

    tokens_used = None
    if usage is not None:
        if isinstance(usage, dict):
            tokens_used = usage.get("prompt_tokens") or usage.get("total_tokens")
        else:
            tokens_used = getattr(usage, "prompt_tokens", None) or getattr(usage, "total_tokens", None)

    if tokens_used is None:
        tokens_used = estimate_tokens(text)

    return np.asarray(vec, dtype=np.float32), int(tokens_used)


def _build_core_memory_embeddings() -> None:
    """Build a row-normalized embedding matrix for core memories (type='core').

    This is called at startup so each memory entry is embedded once (best-effort).
    Query-time selection embeds the user message and does a cosine similarity search.
    """
    global CORE_MEMORY_TEXTS, CORE_MEMORY_EMBEDDINGS, CORE_MEMORY_EMBEDDINGS_ERROR

    CORE_MEMORY_TEXTS = None
    CORE_MEMORY_EMBEDDINGS = None
    CORE_MEMORY_EMBEDDINGS_ERROR = None

    if MEMORIES is None or not isinstance(MEMORIES, list):
        return

    memory_texts: list[str] = [m.strip() for m in MEMORIES if isinstance(m, str) and m.strip()]
    if not memory_texts:
        CORE_MEMORY_TEXTS = []
        CORE_MEMORY_EMBEDDINGS = None
        return

    vectors: list[np.ndarray] = []
    kept_texts: list[str] = []
    expected_d: int | None = None

    try:
        raw_vectors, _usage = _openai_embeddings(input_texts=memory_texts)
        for m, raw_vector in zip(memory_texts, raw_vectors):
            v = np.asarray(raw_vector, dtype=np.float32)
            if v.ndim != 1:
                continue
            if expected_d is None:
                expected_d = int(v.shape[0])
            elif int(v.shape[0]) != expected_d:
                continue
            n = float(np.linalg.norm(v))
            if n <= 0:
                continue
            vectors.append((v / n).astype(np.float32))
            kept_texts.append(m)

        if not vectors:
            CORE_MEMORY_TEXTS = []
            CORE_MEMORY_EMBEDDINGS = None
            return

        CORE_MEMORY_TEXTS = kept_texts
        CORE_MEMORY_EMBEDDINGS = np.vstack(vectors).astype(np.float32)
        logging.info(f"Embedded {len(kept_texts)} core memories for similarity search")
        try:
            _log_debug_to_console("core_mem_build")
        except Exception:
            pass
    except Exception as e:
        CORE_MEMORY_TEXTS = []
        CORE_MEMORY_EMBEDDINGS = None
        CORE_MEMORY_EMBEDDINGS_ERROR = f"{type(e).__name__}: {e}"
        logging.exception("Failed to build core memory embeddings")


def _build_life_memory_embeddings() -> None:
    """Build a row-normalized embedding matrix for life memories (type='life').

    This is called at startup and after /new_chat to cache embeddings for duplicate checking.
    Query-time duplicate checking embeds only the new memory and does cosine similarity.
    """
    global LIFE_MEMORY_TEXTS, LIFE_MEMORY_EMBEDDINGS, LIFE_MEMORY_EMBEDDINGS_ERROR

    LIFE_MEMORY_TEXTS = None
    LIFE_MEMORY_EMBEDDINGS = None
    LIFE_MEMORY_EMBEDDINGS_ERROR = None

    if not _ensure_db_connection():
        LIFE_MEMORY_EMBEDDINGS_ERROR = DB_LAST_ERROR or "database connection unavailable"
        logging.warning("Cannot build life memory embeddings: DB connection unavailable")
        return

    try:
        with DB_CONN.cursor() as cur:
            cur.execute("SELECT content FROM memories WHERE type = 'life' ORDER BY id")
            rows = cur.fetchall()
        
        memory_texts: list[str] = [row[0].strip() for row in rows if row[0] and row[0].strip()]
        
        if not memory_texts:
            LIFE_MEMORY_TEXTS = []
            LIFE_MEMORY_EMBEDDINGS = None
            logging.info("No life memories to embed")
            return

        vectors: list[np.ndarray] = []
        kept_texts: list[str] = []
        expected_d: int | None = None

        raw_vectors, _usage = _openai_embeddings(input_texts=memory_texts)
        for m, raw_vector in zip(memory_texts, raw_vectors):
            v = np.asarray(raw_vector, dtype=np.float32)
            if v.ndim != 1:
                continue
            if expected_d is None:
                expected_d = int(v.shape[0])
            elif int(v.shape[0]) != expected_d:
                continue
            n = float(np.linalg.norm(v))
            if n <= 0:
                continue
            vectors.append((v / n).astype(np.float32))
            kept_texts.append(m)

        if not vectors:
            LIFE_MEMORY_TEXTS = []
            LIFE_MEMORY_EMBEDDINGS = None
            LIFE_MEMORY_EMBEDDINGS_ERROR = (
                f"failed to embed all {len(memory_texts)} life memories"
            )
            logging.error(f"Life memory embedding failed: {LIFE_MEMORY_EMBEDDINGS_ERROR}")
            return

        LIFE_MEMORY_TEXTS = kept_texts
        LIFE_MEMORY_EMBEDDINGS = np.vstack(vectors).astype(np.float32)
        logging.info(f"Embedded {len(kept_texts)} life memories for duplicate checking")
    except Exception as e:
        LIFE_MEMORY_TEXTS = []
        LIFE_MEMORY_EMBEDDINGS = None
        LIFE_MEMORY_EMBEDDINGS_ERROR = f"{type(e).__name__}: {e}"
        logging.exception(f"Failed to build life memory embeddings: {type(e).__name__}: {e}")


def _ensure_life_memory_cache_current() -> None:
    """Refresh the life-memory embedding cache when the database row count changed."""
    if not _ensure_db_connection():
        return

    try:
        with DB_CONN.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM memories WHERE type = 'life'")
            db_count = int(cur.fetchone()[0])
        cached_count = len(LIFE_MEMORY_TEXTS) if isinstance(LIFE_MEMORY_TEXTS, list) else -1
        if cached_count != db_count:
            logging.info(
                f"Refreshing life memory cache: database={db_count}, cached={cached_count}"
            )
            _build_life_memory_embeddings()
    except Exception as e:
        logging.warning(f"Could not verify life memory cache: {type(e).__name__}: {e}")


def _retrieve_top_chunks(query: str, k: int = RAG_CHUNKS_TO_RETRIEVE) -> tuple[list[str], int]:
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


def _log_debug_to_console(tag: str = "") -> None:
    """Dump relevant debug globals to server console as JSON for easy viewing.

    Includes LAST_NEW_CHAT_DEBUG, LIFE_RECALL_DEBUG, LAST_LIFE_RECALL, DB_LAST_ERROR,
    and core memory embedding counts.
    """
    try:
        payload = {
            "tag": tag,
            "LAST_NEW_CHAT_DEBUG": LAST_NEW_CHAT_DEBUG if isinstance(LAST_NEW_CHAT_DEBUG, dict) else None,
            "LIFE_RECALL_DEBUG": LIFE_RECALL_DEBUG if isinstance(LIFE_RECALL_DEBUG, dict) else None,
            "CORE_MEMORY_DEBUG": CORE_MEMORY_DEBUG if isinstance(CORE_MEMORY_DEBUG, dict) else None,
            "LAST_LIFE_RECALL": LAST_LIFE_RECALL if isinstance(LAST_LIFE_RECALL, dict) else LAST_LIFE_RECALL,
            "DB_LAST_ERROR": DB_LAST_ERROR,
            "core_memory_embeddings_built": bool(CORE_MEMORY_EMBEDDINGS is not None and CORE_MEMORY_TEXTS is not None),
            "core_memory_embeddings_count": int(len(CORE_MEMORY_TEXTS) if isinstance(CORE_MEMORY_TEXTS, list) else 0),
            "life_memory_embeddings_built": bool(LIFE_MEMORY_EMBEDDINGS is not None and LIFE_MEMORY_TEXTS is not None),
            "life_memory_embeddings_count": int(len(LIFE_MEMORY_TEXTS) if isinstance(LIFE_MEMORY_TEXTS, list) else 0),
            "life_memory_embeddings_error": LIFE_MEMORY_EMBEDDINGS_ERROR,
        }
        s = json.dumps(payload, ensure_ascii=False, default=str)
        if len(s) > DEBUG_CONSOLE_TRUNCATE_LIMIT:
            s = s[:DEBUG_CONSOLE_TRUNCATE_LIMIT] + "…(truncated)"
        # Store the latest debug console payload in-memory so the frontend
        # can fetch and print it to the browser console. Do NOT log to server.
        try:
            global LAST_DEBUG_CONSOLE
            LAST_DEBUG_CONSOLE = s
        except Exception:
            pass
    except Exception:
        logging.exception("DEBUG_CONSOLE: failed to serialize debug state")


def _select_memories(query: str, k: int = CORE_MEMORY_MAX_INJECT) -> list[str]:
    """Return up to k *relevant* core memories by embedding similarity.

    Embeds the user message on each call, then retrieves the top matches from
    the startup-built core memory embedding matrix.

    To avoid injecting unrelated memories, applies conservative similarity gates
    and may return an empty list.
    """
    global CORE_MEMORY_DEBUG
    CORE_MEMORY_DEBUG = {
        "query": query[:160] if isinstance(query, str) else "",
        "fact_question": False,
        "identity_question": False,
        "best_similarity": None,
        "selected": [],
    }
    if not isinstance(query, str) or not query.strip():
        return []
    if CORE_MEMORY_EMBEDDINGS is None or CORE_MEMORY_TEXTS is None:
        return []
    if not CORE_MEMORY_TEXTS:
        return []

    try:
        k_int = int(k)
    except Exception:
        k_int = 6
    # Requirement: inject only the top 1–6 relevant memories
    k_int = max(1, min(6, k_int))

    n = int(CORE_MEMORY_EMBEDDINGS.shape[0])
    if n <= 0:
        return []

    is_fact_question = any(
        re.search(pattern, query.strip().lower()) for pattern in FACT_QUESTION_PATTERNS
    )
    CORE_MEMORY_DEBUG["fact_question"] = is_fact_question

    # Identity questions need a deterministic lexical safeguard because the
    # wording of the question may share no words with the stored fact.
    identity_memories: list[str] = []
    if any(re.search(pattern, query.lower()) for pattern in IDENTITY_QUESTION_PATTERNS):
        identity_memories = [
            text for text in CORE_MEMORY_TEXTS
            if any(re.search(pattern, text.lower()) for pattern in IDENTITY_MEMORY_PATTERNS)
        ]
    CORE_MEMORY_DEBUG["identity_question"] = bool(identity_memories)

    # Requirement: embed the user message on each query
    q, _tok = _embed_text(query)
    qn = float(np.linalg.norm(q))
    if qn <= 0:
        return []
    q_unit = (q / qn).astype(np.float32)

    sims = CORE_MEMORY_EMBEDDINGS @ q_unit  # cosine similarity due to normalization
    k0 = min(k_int, n)

    top_idx = np.argpartition(-sims, range(k0))[:k0]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    best = float(sims[int(top_idx[0])])
    CORE_MEMORY_DEBUG["best_similarity"] = best

    # Conservative filters to avoid unrelated injections.
    abs_min = CORE_MEMORY_MIN_SIMILARITY
    is_identity_question = bool(identity_memories)
    # For fact questions, allow a lower-confidence semantic match through as a
    # fallback. This handles wording changes such as "where do you live?" vs.
    # "Alex lives in Birmingham", without weakening ordinary chat retrieval.
    fact_question_min = 0.10
    if best < abs_min and not (is_identity_question or (is_fact_question and best >= fact_question_min)):
        return []

    rel_gate = max(fact_question_min if is_fact_question else abs_min, best * CORE_MEMORY_RELATIVE_GATE)
    selection_limit = 3 if is_fact_question and not is_identity_question else k_int

    picked: list[str] = []
    for text in identity_memories:
        if text not in picked:
            picked.append(text)
        if len(picked) >= selection_limit:
            break

    for i in top_idx.tolist():
        s = float(sims[int(i)])
        if s < rel_gate:
            continue
        if 0 <= int(i) < len(CORE_MEMORY_TEXTS):
            if CORE_MEMORY_TEXTS[int(i)] not in picked:
                picked.append(CORE_MEMORY_TEXTS[int(i)])
        if len(picked) >= k_int:
            break

    # Console log for debugging (kept short to avoid noise)
    try:
        CORE_MEMORY_DEBUG["similarity_gate"] = rel_gate
        CORE_MEMORY_DEBUG["selected"] = [p[:160] for p in picked]
        if picked:
            preview = [p[:120] + ("…" if len(p) > 120 else "") for p in picked]
            logging.info(f"Core memories selected: count={len(picked)} best_sim={best:.3f} gate={rel_gate:.3f} preview={preview}")
        else:
            logging.info(f"Core memories selected: none (best_sim={best:.3f} < gate)")
    except Exception:
        pass
    return picked


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

    # Keep similarity index in sync (best-effort)
    try:
        _build_core_memory_embeddings()
    except Exception:
        pass


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


def _search_life_memories(message: str, limit: int = LIFE_RECALL_MAX_INJECT, threshold: float = LIFE_RECALL_MIN_SIMILARITY) -> list[str]:
    """Semantic search against life memories using pre-computed embeddings.
    
    Embeds the user message and retrieves the top matches from the cached
    life memory embedding matrix (similar to core memory selection).
    
    Args:
        message: The user message to search against
        limit: Maximum number of memories to return (1-6)
        threshold: Minimum similarity threshold
    """
    global LIFE_RECALL_DEBUG
    
    if LIFE_RECALL_DEBUG is None:
        LIFE_RECALL_DEBUG = {}

    # Memories may have been added outside this process since startup.
    _ensure_life_memory_cache_current()
    
    if not isinstance(message, str) or not message.strip():
        LIFE_RECALL_DEBUG["error"] = "empty message"
        logging.warning("Life memory search skipped: empty message")
        return []
    
    if LIFE_MEMORY_EMBEDDINGS is None or LIFE_MEMORY_TEXTS is None:
        LIFE_RECALL_DEBUG["error"] = (
            "life memory embeddings not available"
            + (f": {LIFE_MEMORY_EMBEDDINGS_ERROR}" if LIFE_MEMORY_EMBEDDINGS_ERROR else "")
        )
        logging.warning("Life memory search skipped: embeddings not built")
        return []
    
    if not LIFE_MEMORY_TEXTS:
        LIFE_RECALL_DEBUG["error"] = "no life memories available"
        return []
    
    LIFE_RECALL_DEBUG["search_message"] = message[:100]
    logging.info(f"Starting semantic life memory search for message: {message[:100]}...")
    
    try:
        k_int = int(limit)
    except Exception:
        k_int = 6
    k_int = max(1, min(6, k_int))
    
    n = int(LIFE_MEMORY_EMBEDDINGS.shape[0])
    if n <= 0:
        return []
    
    try:
        # Embed the user message
        q, _tok = _embed_text(message)
        qn = float(np.linalg.norm(q))
        if qn <= 0:
            return []
        q_unit = (q / qn).astype(np.float32)
        
        # Cosine similarity via matrix multiply
        sims = LIFE_MEMORY_EMBEDDINGS @ q_unit
        k0 = min(k_int, n)
        
        top_idx = np.argpartition(-sims, range(k0))[:k0]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        
        best = float(sims[int(top_idx[0])])
        
        # Similarity threshold (configurable for different retrieval modes)
        abs_min = float(threshold)
        if best < abs_min:
            LIFE_RECALL_DEBUG["error"] = f"best similarity {best:.3f} below threshold {abs_min}"
            logging.info(f"Life memory search: no results above threshold (best={best:.3f})")
            return []
        
        rel_gate = max(abs_min, best * LIFE_RECALL_RELATIVE_GATE)
        
        picked: list[str] = []
        for i in top_idx.tolist():
            s = float(sims[int(i)])
            if s < rel_gate:
                continue
            if 0 <= int(i) < len(LIFE_MEMORY_TEXTS):
                picked.append(LIFE_MEMORY_TEXTS[int(i)])
            if len(picked) >= k_int:
                break
        
        LIFE_RECALL_DEBUG["results_count"] = len(picked)
        LIFE_RECALL_DEBUG["results_preview"] = [r[:100] + "..." if len(r) > 100 else r for r in picked[:3]]
        LIFE_RECALL_DEBUG["best_similarity"] = best
        LIFE_RECALL_DEBUG["threshold"] = rel_gate
        
        if picked:
            logging.info(f"Life memories selected: count={len(picked)} best_sim={best:.3f} gate={rel_gate:.3f}")
        else:
            logging.info(f"Life memories selected: none (best_sim={best:.3f} < gate={rel_gate:.3f})")
        
        return picked
    except Exception as e:
        LIFE_RECALL_DEBUG["error"] = f"{type(e).__name__}: {e}"
        logging.error(f"Life memory semantic search failed: {type(e).__name__}: {e}")
        return []


def _summarize_conversation(session_history: list, user_name: str = "User") -> str:
    """Extract 0..N durable personal memories from conversation.

    Returns one short factual sentence per line (no bullets/numbering).
    Returns empty string if no valid personal/contextual memory exists or extraction fails.
    """
    global LAST_NEW_CHAT_DEBUG
    
    if not session_history or len(session_history) < 2:
        if isinstance(LAST_NEW_CHAT_DEBUG, dict):
            LAST_NEW_CHAT_DEBUG["summarize_skipped"] = "insufficient messages"
        logging.info("_summarize_conversation: skipped (insufficient messages)")
        return ""
    
    # Build conversation text
    conversation_text = ""
    for msg in session_history:
        if msg["role"] == "user":
            conversation_text += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            conversation_text += f"Alex: {msg['content']}\n"
    
    if isinstance(LAST_NEW_CHAT_DEBUG, dict):
        LAST_NEW_CHAT_DEBUG["conversation_text_length"] = len(conversation_text)
        LAST_NEW_CHAT_DEBUG["conversation_preview"] = conversation_text[:200] + "..." if len(conversation_text) > 200 else conversation_text
    
    # Create strict summarization prompt
    # OpenRouter needs sufficient output budget for structured memory extraction.
    summary_prompt = (
        "Convert this chat into personal memory entries.\n\n"
        "Rules:\n"
        "- Output 0 to 2 memories, EACH on its own line. Maximum 3 memories total.\n"
        "- Each memory must be about the user, Alex, or someone in their lives.\n"
        "- Each memory must describe a DURABLE fact: preferences, traits, past experiences, future plans with dates, relationships, or ongoing situations.\n"
        "- Use only information stated by the USER. Ignore anything said by Alex/assistant.\n"
        "- Do NOT output general knowledge, definitions, or explanations.\n"
        "- Do NOT restate or summarize AI answers.\n"
        "- If no valid personal/contextual memory exists, output exactly: NONE\n\n"
        "CRITICAL - MERGE related details into ONE memory:\n"
        "- If multiple sentences describe the SAME event or outing, merge them into a single detailed sentence.\n"
        "- Do NOT split one event into multiple micro-entries (e.g., 'went for a walk' + 'climbed a tree' = one memory).\n"
        "- Preserve meaningful narrative detail within that single sentence (e.g., 'went for a walk and climbed a tree in the park').\n"
        "- Only create a second memory if it describes a genuinely separate, unrelated event or fact.\n\n"
        "CRITICAL - 6-MONTH DURABILITY TEST:\n"
        "- ONLY save facts that would still matter in 6 months.\n"
        "- Ask: Would this fact be relevant or useful to recall months or years from now?\n"
        "- If the answer is NO, do NOT save it.\n"
        "- Examples to EXCLUDE: typos, minor edits, UI tweaks, debug actions, temporary tasks, session-specific events, what someone is doing right now.\n"
        "- Examples to INCLUDE: lasting preferences, significant life events, major plans with dates, personality traits, relationships, hobbies.\n\n"
        "CRITICAL - EXCLUDE IMMEDIATE/TRANSIENT ACTIVITIES:\n"
        "- Do NOT capture what someone is doing RIGHT NOW (e.g., 'Steve is going to the shops', 'Abi is playing TF2').\n"
        "- Do NOT capture temporary states, current locations, or immediate plans without specific dates.\n"
        "- Do NOT capture trivial events: typos, minor code changes, UI adjustments, debug sessions, temporary bug fixes.\n"
        "- ONLY capture: preferences (likes/dislikes), past events, future plans with dates/context, habits, relationships, or ongoing situations.\n"
        "- Examples to EXCLUDE: 'X is playing Y', 'X is at the store', 'X fixed a typo', 'X updated the UI', 'X is debugging'.\n"
        "- Examples to INCLUDE: 'Steve loves playing football', 'Alex went skiing in France in 2025', 'Abi is planning to visit Japan in March'.\n\n"
        "CRITICAL - Use actual names, NOT pronouns:\n"
        f"- The user's name is '{user_name}'. Replace 'I' with '{user_name}'.\n"
        "- Replace 'we' with actual names (e.g., 'Alex and Abi' not 'we').\n"
        "- Infer who 'we' refers to from context (Alex + user + anyone mentioned).\n"
        "- Example: If user says 'we went to Legoland with Marilyn', write 'Alex went to Legoland with {user_name} and Marilyn'.\n\n"
        "Output format:\n"
        "- Return ONLY the sentences (or NONE).\n"
        "- One sentence per line. No numbering, no bullet points, no commentary, no quotes.\n\n"
        f"Conversation:\n{conversation_text}\n\n"
        "Return ONLY the sentences (or NONE):"
    )
    
    try:
        logging.info("_summarize_conversation: calling LLM for memory extraction...")
        if isinstance(LAST_NEW_CHAT_DEBUG, dict):
            LAST_NEW_CHAT_DEBUG["llm_call_1"] = "started"
        
        # Use a generous output budget for memory extraction.
        summary, usage = _openrouter_chat_completion(
            model=_OPENROUTER_MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=4096,
            reasoning_effort=_OPENROUTER_MEMORY_EFFORT,
        )
        
        # Track token usage
        if isinstance(LAST_NEW_CHAT_DEBUG, dict) and usage:
            if isinstance(usage, dict):
                LAST_NEW_CHAT_DEBUG["llm_call_1_tokens"] = {
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0)
                }
            else:
                LAST_NEW_CHAT_DEBUG["llm_call_1_tokens"] = {
                    "input": getattr(usage, "prompt_tokens", 0),
                    "output": getattr(usage, "completion_tokens", 0),
                    "total": getattr(usage, "total_tokens", 0)
                }
        
        summary = (summary or "").strip()
        logging.info(f"_summarize_conversation: LLM returned: '{summary}'")
        if isinstance(LAST_NEW_CHAT_DEBUG, dict):
            LAST_NEW_CHAT_DEBUG["llm_call_1_result"] = summary
        
        # Skip if model says there's nothing important or returned empty
        if summary.upper() == "NONE" or not summary:
            logging.info("_summarize_conversation: no durable memory found")
            if isinstance(LAST_NEW_CHAT_DEBUG, dict):
                LAST_NEW_CHAT_DEBUG["final_result"] = "NONE or empty"
            return ""

        # Some reasoning-capable responses leak internal analysis markers into
        # the normal content field.  Remove complete think blocks, but reject
        # an unclosed think block rather than storing its analysis as memory.
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.IGNORECASE | re.DOTALL)
        if re.search(r"<think>|</think>", summary, flags=re.IGNORECASE):
            summary = re.sub(r"<\/?think>.*", "", summary, flags=re.IGNORECASE | re.DOTALL)
        summary = re.sub(r"```(?:text|markdown)?\s*", "", summary, flags=re.IGNORECASE)
        summary = summary.replace("```", "").strip()
        if not summary:
            if isinstance(LAST_NEW_CHAT_DEBUG, dict):
                LAST_NEW_CHAT_DEBUG["final_result"] = "empty after removing reasoning markup"
            return ""
        
        # Split to candidate lines; enforce 1 sentence per line; filter junk.
        lines_in = [ln.strip() for ln in summary.split("\n") if ln.strip()]
        if not lines_in:
            return ""
        if len(lines_in) == 1 and lines_in[0].upper() == "NONE":
            logging.info("_summarize_conversation: no valid memory found (NONE)")
            if isinstance(LAST_NEW_CHAT_DEBUG, dict):
                LAST_NEW_CHAT_DEBUG["final_result"] = "NONE"
            return ""

        bad_phrases = (
            # General knowledge / definitions
            " is defined as ",
            " refers to ",
            " means ",
            " in general ",
            " generally ",
            " as an ai ",
            " i am an ai ",
            # Immediate/transient activities (present continuous)
            " is going to ",
            " is playing ",
            " is at the ",
            " is at ",
            " is doing ",
            " is watching ",
            " is eating ",
            " is drinking ",
            " is buying ",
            " is working on ",
            " is reading ",
            " is listening to ",
            " are going to ",
            " are playing ",
            " are at the ",
            " are at ",
            " are doing ",
            " are watching ",
            # Trivial/short-term events (6-month durability filter)
            " fixed a typo ",
            " typo ",
            " updated the ui ",
            " changed the ui ",
            " ui change ",
            " debugging ",
            " debug ",
            " is debugging ",
            " temporary ",
            " for now ",
            " right now ",
            " at the moment ",
            " currently ",
            " session ",
            " minor edit ",
            " small change ",
            " quick fix ",
            " tweaked ",
            " adjusted ",
        )

        # Models occasionally ignore the output-only instruction and return a
        # heading or analysis fragment instead of a memory.  These must never
        # reach the database as durable life memories.
        bad_line_prefixes = (
            "looking at what ",
            "looking at the conversation",
            "here are the memories",
            "here are some memories",
            "possible memories:",
            "memories:",
            "analysis:",
            "answer:",
            "output:",
            "sure,",
        )
        bad_line_patterns = (
            r"^\s*(?:[-*•]|\d+[.)])\s*$",       # empty bullet/number marker
            r"^\s*(?:[-*•]|\d+[.)])\s+",       # markdown/list prefix
            r"^\s*(?:analysis|reasoning|thoughts?)\s*:",
            r"^\s*<\/?think\b",
        )

        cleaned: list[str] = []
        seen_norm: set[str] = set()

        for ln in lines_in:
            if ln.upper() == "NONE":
                continue
            if ln.startswith("-"):
                ln = ln.lstrip("- ").strip()
            if any(re.search(pattern, ln, flags=re.IGNORECASE) for pattern in bad_line_patterns):
                continue
            # Enforce one sentence
            ln = re.split(r"(?<=[.!?])\s+", ln, maxsplit=1)[0].strip()
            if not ln:
                continue

            # Reject headings, meta-commentary, and incomplete fragments such
            # as "Looking at what Steve actually said:".  A colon-terminated
            # line is not a complete memory sentence.
            lower_line = ln.lower()
            if lower_line.startswith(bad_line_prefixes) or ln.endswith(":"):
                continue
            if len(ln.split()) < 4:
                continue

            lower = f" {ln.lower()} "
            if any(p in lower for p in bad_phrases):
                continue
            # Durable project memories can legitimately contain several
            # concrete details. Keep them intact instead of silently dropping
            # otherwise valid extraction results.
            if len(ln.split()) > 60:
                continue

            norm = re.sub(r"\s+", " ", ln.lower()).strip()
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            cleaned.append(ln)
            if len(cleaned) >= 5:
                break

        if not cleaned:
            return ""

        # Post-process: replace any remaining pronouns with actual names
        final_cleaned: list[str] = []
        for line in cleaned:
            # Replace pronouns (case-insensitive, word boundaries)
            line_fixed = line
            # Replace "I" with user_name (careful with word boundaries)
            line_fixed = re.sub(r"\bI\b", user_name, line_fixed)
            line_fixed = re.sub(r"\bI'm\b", f"{user_name} is", line_fixed, flags=re.IGNORECASE)
            line_fixed = re.sub(r"\bI've\b", f"{user_name} has", line_fixed, flags=re.IGNORECASE)
            # Replace "we" with "Alex and {user_name}" if no other names visible in context
            if re.search(r"\bwe\b", line_fixed, flags=re.IGNORECASE):
                # Simple heuristic: if "Alex" is already in the sentence, just replace "we" with "they"
                # otherwise replace with "Alex and {user_name}"
                if "Alex" in line_fixed:
                    line_fixed = re.sub(r"\bwe\b", "they", line_fixed, flags=re.IGNORECASE)
                    line_fixed = re.sub(r"\bWe\b", "They", line_fixed)
                else:
                    line_fixed = re.sub(r"\bwe\b", f"Alex and {user_name}", line_fixed, flags=re.IGNORECASE)
                    line_fixed = re.sub(r"\bWe\b", f"Alex and {user_name}", line_fixed)
            final_cleaned.append(line_fixed)

        out = "\n".join(final_cleaned)
        logging.info(f"_summarize_conversation: extracted {len(final_cleaned)} memory line(s)")
        if isinstance(LAST_NEW_CHAT_DEBUG, dict):
            LAST_NEW_CHAT_DEBUG["final_result"] = final_cleaned
        return out
    except Exception as e:
        logging.error(f"_summarize_conversation: error: {type(e).__name__}: {e}")
        if isinstance(LAST_NEW_CHAT_DEBUG, dict):
            LAST_NEW_CHAT_DEBUG["summarize_error"] = f"{type(e).__name__}: {e}"
        return ""


def _get_tags_for_candidates(candidate_lines: list[str]) -> dict:
    """Ask the LLM to provide 1-3 short tags for each candidate memory line.

    Returns a mapping {memory_line: [tag1, tag2, ...]}.
    """
    if not candidate_lines:
        return {}

    try:
        prompt = (
            "For each of the following short personal memory sentences, provide 1 to 3 short tags (single words or very short phrases) that describe the topic."
            " Return ONLY a JSON array of arrays where each element is the list of tags for the corresponding memory in order.\n\n"
            "Memories:\n"
        )
        for ln in candidate_lines:
            prompt += f"- {ln}\n"

        summary, usage = _openrouter_chat_completion(
            model=_OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            reasoning_effort=_OPENROUTER_TAG_EFFORT,
        )

        tags_map: dict = {}
        resp_text = (summary or "").strip()
        if not resp_text:
            return {}

        # Try to parse JSON first
        try:
            parsed = json.loads(resp_text)
            # parsed expected to be list of lists
            for i, ln in enumerate(candidate_lines):
                if i < len(parsed) and isinstance(parsed[i], list):
                    tags_map[ln] = [str(t).strip() for t in parsed[i] if str(t).strip()]
                else:
                    tags_map[ln] = []
            if isinstance(LAST_NEW_CHAT_DEBUG, dict):
                LAST_NEW_CHAT_DEBUG["tags_extraction_result"] = parsed
            return tags_map
        except Exception:
            # fallback: parse line-by-line CSV or one-line JSON-ish
            lines = [l.strip() for l in resp_text.splitlines() if l.strip()]
            for i, ln in enumerate(candidate_lines):
                if i < len(lines):
                    parts = [p.strip() for p in re.split(r"[,;]", lines[i]) if p.strip()]
                    tags_map[ln] = parts
                else:
                    tags_map[ln] = []
            if isinstance(LAST_NEW_CHAT_DEBUG, dict):
                LAST_NEW_CHAT_DEBUG["tags_extraction_fallback"] = lines
            return tags_map
    except Exception as e:
        logging.error(f"_get_tags_for_candidates error: {type(e).__name__}: {e}")
        if isinstance(LAST_NEW_CHAT_DEBUG, dict):
            LAST_NEW_CHAT_DEBUG["tags_extraction_error"] = f"{type(e).__name__}: {e}"
        return {}


def _check_duplicate_memory(new_memory: str, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD, check_life_memories: bool = True) -> bool:
    """Check if a memory is a duplicate using normalized hash first, then cosine similarity.
    
    Args:
        new_memory: The memory text to check
        threshold: Cosine similarity threshold
        check_life_memories: If True, check against life memories; if False, check against MEMORIES
    
    Returns:
        True if duplicate found (hash match or similarity > threshold), False otherwise
    """
    if not new_memory or not new_memory.strip():
        return True  # Empty memory is considered duplicate
    
    # STEP 1: Check for exact normalized hash match in database (fast, deterministic)
    if check_life_memories:
        try:
            new_hash = _compute_memory_hash(new_memory)
            if _ensure_db_connection():
                try:
                    with DB_CONN.cursor() as cur:
                        cur.execute(
                            "SELECT COUNT(*) FROM memories WHERE type = 'life' AND normalized_hash = %s",
                            (new_hash,)
                        )
                        count = cur.fetchone()[0]
                        if count > 0:
                            logging.info(f"Duplicate life memory detected (exact normalized hash match)")
                            return True
                except Exception as e:
                    # If the normalized_hash column doesn't exist, skip hash-check gracefully
                    msg = str(e).lower()
                    if 'normalized_hash' in msg or 'undefinedcolumn' in msg or 'column \"normalized_hash\"' in msg:
                        logging.warning("Hash duplicate check skipped: normalized_hash column missing in DB")
                    else:
                        logging.warning(f"Hash duplicate check failed: {type(e).__name__}: {e}")
        except Exception as e:
            logging.warning(f"Hash duplicate check preparation failed: {type(e).__name__}: {e}")
            # Continue to embedding check on error
    
    # STEP 2: Check for paraphrase duplicates using embeddings    
    if check_life_memories:
        # Use pre-computed life memory embeddings (90%+ token reduction)
        if LIFE_MEMORY_EMBEDDINGS is None or LIFE_MEMORY_TEXTS is None or len(LIFE_MEMORY_TEXTS) == 0:
            return False  # No existing memories, so not a duplicate
        
        try:
            # Embed new memory (SINGLE API call)
            new_vec, _ = _embed_text(new_memory)
            norm = float(np.linalg.norm(new_vec))
            if norm <= 0:
                return False
            
            new_unit = (new_vec / norm).astype(np.float32)
            
            # Matrix multiply: O(1) operation after pre-computation
            sims = LIFE_MEMORY_EMBEDDINGS @ new_unit
            
            max_sim = float(sims.max())
            
            if max_sim > threshold:
                logging.info(f"Duplicate life memory detected (similarity: {max_sim:.3f})")
                return True
            
            return False
        except Exception as e:
            logging.error(f"Error checking duplicate life memory: {type(e).__name__}: {e}")
            return False  # On error, proceed with insert to avoid data loss
    else:
        # Check against core memories (MEMORIES)
        memory_cache = MEMORIES
        if not memory_cache or not isinstance(memory_cache, list) or len(memory_cache) == 0:
            return False
        
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
                    logging.info(f"Duplicate core memory detected (similarity: {similarity:.3f})")
                    return True
            
            return False
        except Exception as e:
            logging.error(f"Error checking duplicate core memory: {type(e).__name__}: {e}")
            return False


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
    global LAST_NEW_CHAT_DEBUG
    
    if not memory_line or not memory_line.strip():
        if isinstance(LAST_NEW_CHAT_DEBUG, dict):
            LAST_NEW_CHAT_DEBUG["save_life_memory_skipped"] = "empty memory"
        return
    
    try:
        if _ensure_db_connection():
            # Split multi-line memory into individual lines
            new_memories = [line.strip() for line in memory_line.split("\n") if line.strip()]

            if isinstance(LAST_NEW_CHAT_DEBUG, dict):
                LAST_NEW_CHAT_DEBUG["db_connection"] = "established"
                LAST_NEW_CHAT_DEBUG["memories_to_insert"] = new_memories

            # Insert each memory into the database with normalized_hash when available.
            # If the DB schema hasn't been migrated (column missing), fall back to the legacy insert.
            with DB_CONN.cursor() as cur:
                for mem in new_memories:
                    mem_hash = _compute_memory_hash(mem)
                    logging.info(f"Executing INSERT for life memory: '{mem}' (hash: {mem_hash[:16]}...)")
                    try:
                        # Try modern insert with normalized_hash + ON CONFLICT
                        cur.execute(
                            "INSERT INTO memories (content, type, normalized_hash) VALUES (%s, %s, %s) ON CONFLICT (type, normalized_hash) DO NOTHING",
                            (mem, 'life', mem_hash)
                        )
                    except Exception as e:
                        # If column missing or migration not applied, fall back quietly to legacy insert
                        msg = str(e).lower()
                        if 'normalized_hash' in msg or 'undefinedcolumn' in msg or 'column \"normalized_hash\"' in msg:
                            logging.warning("normalized_hash column missing; falling back to legacy insert for life memories")
                            try:
                                cur.execute("INSERT INTO memories (content, type) VALUES (%s, %s)", (mem, 'life'))
                            except Exception as e2:
                                logging.error(f"Failed legacy insert for life memory: {type(e2).__name__}: {e2}")
                        else:
                            logging.error(f"Failed to insert life memory: {type(e).__name__}: {e}")
            try:
                DB_CONN.commit()
            except Exception as e:
                logging.error(f"DB commit failed after inserting life memories: {type(e).__name__}: {e}")
            logging.info(f"Saved {len(new_memories)} life memories to database (committed)")

            # Incrementally append new embeddings to in-memory cache (avoid full rebuild)
            try:
                global LIFE_MEMORY_TEXTS, LIFE_MEMORY_EMBEDDINGS
                if LIFE_MEMORY_TEXTS is None:
                    LIFE_MEMORY_TEXTS = []
                    LIFE_MEMORY_EMBEDDINGS = None

                for mem in new_memories:
                    try:
                        vec, _tok = _embed_text(mem)
                        norm = float(np.linalg.norm(vec))
                        if norm <= 0:
                            continue
                        v_norm = (vec / norm).astype(np.float32)
                        if LIFE_MEMORY_EMBEDDINGS is None:
                            LIFE_MEMORY_EMBEDDINGS = np.expand_dims(v_norm, axis=0)
                            LIFE_MEMORY_TEXTS = [mem]
                        else:
                            LIFE_MEMORY_EMBEDDINGS = np.vstack([LIFE_MEMORY_EMBEDDINGS, v_norm])
                            LIFE_MEMORY_TEXTS.append(mem)
                    except Exception:
                        continue
                logging.info(f"Updated life memory cache with {len(new_memories)} new embedding(s)")
            except Exception as e:
                logging.error(f"Failed to update life memory cache: {type(e).__name__}: {e}")

            if isinstance(LAST_NEW_CHAT_DEBUG, dict):
                LAST_NEW_CHAT_DEBUG["db_insert_count"] = len(new_memories)
                LAST_NEW_CHAT_DEBUG["db_commit"] = "success"
        else:
            logging.error("DB connection failed in _save_life_memory")
            if isinstance(LAST_NEW_CHAT_DEBUG, dict):
                LAST_NEW_CHAT_DEBUG["db_connection"] = "failed"
    except Exception as e:
        logging.error(f"Failed to save life_memory: {type(e).__name__}: {e}")
        if isinstance(LAST_NEW_CHAT_DEBUG, dict):
            LAST_NEW_CHAT_DEBUG["save_life_memory_error"] = f"{type(e).__name__}: {e}"


def _trim_chunk_for_prompt(chunk: str, max_lines: int = RAG_CHUNK_MAX_LINES) -> str:
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
    model: str = "openrouter"


# Mount static files directory
_STATIC_DIR = os.path.join(_REPO_ROOT, "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# Mount a developer images directory (so we can serve images placed in dev/images)
_IMAGES_DIR = os.path.join(_REPO_ROOT, "dev", "images")
if os.path.isdir(_IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=_IMAGES_DIR), name="images")

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
            examples, retrieval_tokens = _retrieve_top_chunks(data.message, k=RAG_CHUNKS_TO_RETRIEVE)
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
            for j in range(RAG_FALLBACK_EXAMPLES):
                chunk = RAG_CHUNKS[(start + j) % len(RAG_CHUNKS)]
                if isinstance(chunk, str) and chunk.strip():
                    examples.append(chunk.strip())
                if len(examples) >= RAG_FALLBACK_EXAMPLES:
                    break

        # Get memories to inject into prompt (top 1–6, relevant only)
        memories = _select_memories(data.message, k=CORE_MEMORY_MAX_INJECT)

        # Retrieve life memories semantically for every message. Explicit recall
        # questions use the broader recall threshold; ordinary messages use a
        # stricter contextual threshold so unrelated memories stay out.
        life_memories: list[str] = []
        global LIFE_RECALL_DEBUG
        LIFE_RECALL_DEBUG = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": data.message[:100]
        }
        try:
            _log_debug_to_console("chat_life_recall_start")
        except Exception:
            pass
        
        # Check what type of question this is
        suggests_recall = _message_suggests_recall(data.message)
        is_past_q = _is_past_factual_question(data.message)
        is_remember = _is_remember_when_prompt(data.message)
        
        LIFE_RECALL_DEBUG["suggests_recall"] = suggests_recall
        LIFE_RECALL_DEBUG["is_past_question"] = is_past_q
        LIFE_RECALL_DEBUG["is_remember_when"] = is_remember
        LIFE_RECALL_DEBUG["core_memories_selected"] = len(memories)
        LIFE_RECALL_DEBUG["core_memory_previews"] = [m[:160] for m in memories]
        LIFE_RECALL_DEBUG["identity_question"] = any(
            re.search(pattern, data.message.lower()) for pattern in IDENTITY_QUESTION_PATTERNS
        )
        LIFE_RECALL_DEBUG["identity_memory_selected"] = [
            m[:160] for m in memories
            if any(re.search(pattern, m.lower()) for pattern in IDENTITY_MEMORY_PATTERNS)
        ]
        
        explicit_recall = suggests_recall or is_past_q or is_remember
        LIFE_RECALL_DEBUG["retrieval_mode"] = "recall" if explicit_recall else "contextual"
        LIFE_RECALL_DEBUG["semantic_search_always_enabled"] = True
        try:
            if explicit_recall:
                life_memories = _search_life_memories(
                    data.message,
                    limit=LIFE_RECALL_MAX_INJECT,
                    threshold=LIFE_RECALL_MIN_SIMILARITY,
                )
            else:
                life_memories = _search_life_memories(
                    data.message,
                    limit=LIFE_CONTEXTUAL_MAX_INJECT,
                    threshold=LIFE_CONTEXTUAL_MIN_SIMILARITY,
                )
            LIFE_RECALL_DEBUG["search_executed"] = True
            LIFE_RECALL_DEBUG["results_count"] = len(life_memories)
            logging.info(
                f"Life memories retrieved: {len(life_memories)} items "
                f"(mode={'recall' if explicit_recall else 'contextual'})"
            )
        except Exception as e:
            LIFE_RECALL_DEBUG["error"] = f"{type(e).__name__}: {e}"
            logging.error(f"Life memory retrieval error: {type(e).__name__}: {e}")
            life_memories = []
        try:
            _log_debug_to_console("chat_life_recall_done")
        except Exception:
            pass
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
        
        # Otherwise, if it's a past question with no memories, ask clarifying question
        if is_past_question and not life_memories:
            # Bypass LLM - ask for clarification instead of saying "idk"
            LIFE_RECALL_DEBUG["llm_bypassed"] = True
            LIFE_RECALL_DEBUG["bypass_reason"] = "past question with no life_memories - asking for clarification"
            
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
            "If memories are provided below, USE THEM naturally in your responses when relevant. "
            "For a simple yes/no question, do not give a bare one-word answer when relevant memories are provided: answer directly and add one useful remembered detail. "
            "If the information isn't in the memories, ask a short clarifying question (e.g., 'when was that?', 'who was there?', 'where did we go?') instead of just saying 'idk'. "
            "NEVER invent or guess places, dates, events, or people. "
            "Stored identity facts are authoritative: do not infer nationality, ethnicity, or origin from where someone currently lives, where the conversation happens, or the language being used. "
            "For identity questions such as nationality or where someone is from, answer from the relevant known fact when one is provided."
        )
        # Language guardrails: narrow British tone without overriding RAG style weighting
        system_prompt += (
            "\n\nLanguage Guardrails:\n"
            "- RAG style examples take priority.\n"
            "- Apply these constraints only when the model would otherwise default to generic or American phrasing.\n"
            "- Follow the tone demonstrated in the style examples above.\n"
            "- Use British English spelling (e.g., colour, favourite, realise).\n"
            "- Avoid American slang or idioms.\n"
            "- Avoid post-2021 internet or Gen-Z slang.\n"
            "- Tone reflects Midlands, England (2001–2021 era).\n"
            "- If unsure, default to standard British English.\n"
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
            system_prompt += (
                f"\n\nRelevant past experiences (background context only):\n{life_text}\n"
                "Use these as background context, not a script. "
                "Do not quote or restate them verbatim. "
                "Reconstruct events naturally in your own conversational tone. "
                "Integrate relevant details fluidly and only when they relate to what the user just said."
            )

        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories)
            system_prompt += f"\n\nKnown facts:\n{memory_text}"

        # The style corpus contains many terse yes/no replies.  Without an
        # explicit composition rule, those examples can overpower retrieved
        # life memories and the model answers only the literal question.  Ask
        # for a compact anecdotal answer when supporting context exists, while
        # keeping the no-invention rule intact.
        if life_memories or memories:
            system_prompt += (
                "\n\nContextual reply rule:\n"
                "- If the user's question can be answered yes/no and the context above supports it, do not stop at a bare yes or no.\n"
                "- Answer directly first, then add one specific relevant detail from the provided context.\n"
                "- When it feels natural, finish with one genuine follow-up question that keeps the conversation going.\n"
                "- Keep this to roughly 1–3 casual sentences, not a long explanation.\n"
                "- Only use details explicitly supported by the context; never invent a story, date, place, person, or opinion.\n"
                "- Example shape (do not copy it literally): 'Yeah, I have — [relevant remembered detail]. Do you still [related activity]?'\n"
            )

        LIFE_RECALL_DEBUG["life_memories_injected"] = len(life_memories)
        LIFE_RECALL_DEBUG["life_memory_previews"] = [m[:160] for m in life_memories]
        LIFE_RECALL_DEBUG["known_facts_injected"] = bool(memories)
        LIFE_RECALL_DEBUG["prompt_memory_injection"] = bool(life_memories or memories)
        try:
            _log_debug_to_console("chat_prompt_memory_injection")
        except Exception:
            pass

        if examples:
            # Trim each retrieved/fallback chunk to its last few lines to avoid dilution
            examples = [_trim_chunk_for_prompt(e, max_lines=RAG_CHUNK_MAX_LINES) for e in examples]
            joined = "\n\n---\n\n".join(examples)
            system_prompt += (
                "\n\n"
                "Style examples (from past chat logs) — imitate the tone, phrasing, punctuation, and message length:\n"
                f"{joined}"
            )

        # Build messages list: system prompt + full session history + current user message
        messages = [{"role": "system", "content": system_prompt}, *session_history, {"role": "user", "content": data.message}]

        # Model generation - collect token usage per provider
        model = data.model
        if model == "openrouter" or model.startswith("gpt-") or model.startswith("gemini"):
            model = _OPENROUTER_MODEL

        reply = ""
        if True:
            reply, usage = _openrouter_chat_completion(model=model, messages=messages, reasoning_effort=_OPENROUTER_CHAT_EFFORT)

            input_t = None
            output_t = None
            total_t = None
            if usage is not None:
                if isinstance(usage, dict):
                    input_t = usage.get("prompt_tokens") or usage.get("input_tokens")
                    output_t = usage.get("completion_tokens") or usage.get("output_tokens")
                    total_t = usage.get("total_tokens")
                else:
                    input_t = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
                    output_t = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
                    total_t = getattr(usage, "total_tokens", None)

            if input_t is None:
                input_t = estimate_tokens(json.dumps(messages))
            if output_t is None:
                output_t = estimate_tokens(reply)
            if total_t is None:
                total_t = int(input_t) + int(output_t)

            token_calls.append({
                "name": "openrouter_chat",
                "input_tokens": int(input_t),
                "output_tokens": int(output_t),
                "total_tokens": int(total_t),
                "note": "reported" if usage is not None else "estimated",
            })
        else:
            # Compatibility fallback: all chat requests still use OpenRouter.
            reply, usage = _openrouter_chat_completion(model=_OPENROUTER_MODEL, messages=messages, reasoning_effort=_OPENROUTER_CHAT_EFFORT)

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
                "name": "openrouter_chat",
                "input_tokens": int(input_t),
                "output_tokens": int(output_t),
                "total_tokens": int(total_t),
                "note": "reported" if usage is not None else "estimated",
            })
        
        # Update conversation history with user message and assistant reply
        session_history.append({"role": "user", "content": data.message})
        session_history.append({"role": "assistant", "content": reply})
        
        CONVERSATION_HISTORY[data.session_id] = session_history
        # Track user_name for memory extraction
        USER_NAME_BY_SESSION[data.session_id] = data.user_name

        # Record token calls for this session (keep runtime-only log per session)
        try:
            lst = SESSION_TOKEN_LOGS.get(data.session_id)
            if not isinstance(lst, list):
                lst = []
                SESSION_TOKEN_LOGS[data.session_id] = lst
            # store a lightweight snapshot: timestamp and calls
            lst.append({
                "time": datetime.utcnow().isoformat(),
                "calls": token_calls,
            })
            logging.info(f"[TOKEN_DEBUG] Recorded {len(token_calls)} token call(s) for session {data.session_id}. Total entries: {len(lst)}")
        except Exception as e:
            logging.error(f"[TOKEN_DEBUG] Error recording token calls: {type(e).__name__}: {e}")

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
        _log_debug_to_console("new_chat_started")
    except Exception:
        pass

    try:
        # Get the conversation history for this session
        session_history = CONVERSATION_HISTORY.get(data.session_id, [])

        # Initialize progress state for frontend polling
        try:
            NEW_CHAT_PROGRESS[data.session_id] = {
                "status": "started",
                "progress": 0,
                "message": "starting new chat",
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception:
            pass

        LAST_NEW_CHAT_DEBUG["history_messages"] = int(len(session_history) if isinstance(session_history, list) else 0)
        
        summary_saved = False
        duplicate_detected = False
        extracted_memory = None
        
        # If there's conversation history, summarize and save it
        if session_history:
            # Extract 0..N personal memory lines (one per line)
            NEW_CHAT_PROGRESS[data.session_id]["status"] = "summarizing"
            NEW_CHAT_PROGRESS[data.session_id]["progress"] = 10
            NEW_CHAT_PROGRESS[data.session_id]["message"] = "summarizing conversation"
            try:
                _log_debug_to_console("new_chat_summarize_start")
            except Exception:
                pass

            # Get user_name for pronoun replacement
            user_name = USER_NAME_BY_SESSION.get(data.session_id, "User")
            summary = _summarize_conversation(session_history, user_name=user_name)
            NEW_CHAT_PROGRESS[data.session_id]["progress"] = 30
            extracted_memory = summary

            LAST_NEW_CHAT_DEBUG["extracted_memory"] = summary
            
            # Log extracted memory for debugging
            if summary:
                logging.info(f"Extracted memory: {summary}")
            else:
                logging.info("No durable memory extracted from conversation")
                LAST_NEW_CHAT_DEBUG["note"] = "no durable memory extracted"
            
            if summary:
                # Split into separate DB entries (one per line) and apply duplicate check per entry.
                candidate_lines = [ln.strip() for ln in str(summary).split("\n") if ln.strip()]
                
                # Deduplicate candidate_lines using normalized text (prevent duplicates within same summary)
                seen_hashes = set()
                deduped_lines = []
                for ln in candidate_lines:
                    ln_hash = _compute_memory_hash(ln)
                    if ln_hash not in seen_hashes:
                        seen_hashes.add(ln_hash)
                        deduped_lines.append(ln)
                
                original_count = len(candidate_lines)
                candidate_lines = deduped_lines
                
                if original_count > len(candidate_lines):
                    logging.info(f"Deduped candidate_lines: {original_count} -> {len(candidate_lines)}")
                    if isinstance(LAST_NEW_CHAT_DEBUG, dict):
                        LAST_NEW_CHAT_DEBUG["deduped_count"] = original_count - len(candidate_lines)
                
                NEW_CHAT_PROGRESS[data.session_id]["status"] = "checking_duplicates"
                NEW_CHAT_PROGRESS[data.session_id]["progress"] = 40
                NEW_CHAT_PROGRESS[data.session_id]["message"] = f"{len(candidate_lines)} candidate(s) to check"
                LAST_NEW_CHAT_DEBUG["candidate_memory_lines"] = candidate_lines

                # Detect ambiguous "we" references and ask clarification immediately
                clarification_questions: list[str] = []
                ambiguous_patterns = [
                    r"\bremember when we\b",
                    r"\bwe (are|did|went|were|have|had|will|visited|traveled|went to|went skiing)\b",
                ]
                for ln in candidate_lines:
                    low = ln.lower()
                    for pat in ambiguous_patterns:
                        if re.search(pat, low):
                            q = f'Who does "we" refer to in the memory: "{ln}"?'
                            clarification_questions.append(q)
                            break

                if clarification_questions:
                    LAST_NEW_CHAT_DEBUG["clarification_needed"] = True
                    LAST_NEW_CHAT_DEBUG["clarification_questions"] = clarification_questions
                    # Append assistant follow-up(s) into the session history so the user sees them
                    session_history = CONVERSATION_HISTORY.get(data.session_id, [])
                    for q in clarification_questions:
                        try:
                            session_history.append({"role": "assistant", "content": q})
                        except Exception:
                            pass
                    CONVERSATION_HISTORY[data.session_id] = session_history
                    # Update progress and return early so we don't clear history or save ambiguous memories
                    NEW_CHAT_PROGRESS[data.session_id]["status"] = "awaiting_clarification"
                    NEW_CHAT_PROGRESS[data.session_id]["progress"] = 50
                    NEW_CHAT_PROGRESS[data.session_id]["message"] = "awaiting clarification for ambiguous 'we' references"
                    # Do not clear session history; return to caller with clarification info
                    LAST_NEW_CHAT_DEBUG["note"] = "clarification requested for ambiguous 'we'"
                    return {
                        "status": "ok",
                        "message": "Clarification requested",
                        "summary_saved": False,
                        "duplicate_detected": False,
                        "extracted_memory": extracted_memory,
                        "clarification_needed": True,
                        "clarification_questions": clarification_questions,
                    }

                # Do not request or attach tags anymore; skip tag extraction
                tags_map = {}

                saved_lines: list[str] = []
                skipped_duplicates: list[str] = []
                LAST_NEW_CHAT_DEBUG["duplicate_check"] = "started"

                total = len(candidate_lines)
                for idx, line in enumerate(candidate_lines, start=1):
                    try:
                        is_duplicate = _check_duplicate_memory(
                            line, 
                            threshold=DUPLICATE_SIMILARITY_THRESHOLD, 
                            check_life_memories=True
                        )
                    except Exception:
                        is_duplicate = False

                    if is_duplicate:
                        duplicate_detected = True
                        skipped_duplicates.append(line)
                        # update progress
                        try:
                            NEW_CHAT_PROGRESS[data.session_id]["message"] = f"skipped duplicate ({len(skipped_duplicates)})"
                            NEW_CHAT_PROGRESS[data.session_id]["progress"] = 40 + int((idx/total) * 40)
                        except Exception:
                            pass
                        continue

                    try:
                        # Do not pass tags when saving life memories
                        _save_life_memory(line)
                        saved_lines.append(line)
                        summary_saved = True
                        # update progress
                        try:
                            NEW_CHAT_PROGRESS[data.session_id]["message"] = f"saved {len(saved_lines)} of {total}"
                            NEW_CHAT_PROGRESS[data.session_id]["progress"] = 40 + int((idx/total) * 40)
                        except Exception:
                            pass
                    except Exception:
                        pass

                LAST_NEW_CHAT_DEBUG["duplicate_check_result"] = {
                    "saved": len(saved_lines),
                    "skipped_duplicates": len(skipped_duplicates),
                }
                LAST_NEW_CHAT_DEBUG["saved_lines"] = saved_lines
                LAST_NEW_CHAT_DEBUG["skipped_duplicates"] = skipped_duplicates
                # Build a DB summary for frontend diagnostics
                try:
                    db_summary = {
                        "db_connected": bool(DB_CONN is not None),
                        "saved_lines": int(len(saved_lines)),
                        "skipped_duplicates": int(len(skipped_duplicates)),
                        "db_last_error": DB_LAST_ERROR,
                    }
                except Exception:
                    db_summary = None
                LAST_NEW_CHAT_DEBUG["db_summary"] = db_summary
                try:
                    _log_debug_to_console("new_chat_saved_lines")
                except Exception:
                    pass
                # mark complete
                try:
                    NEW_CHAT_PROGRESS[data.session_id]["status"] = "done"
                    NEW_CHAT_PROGRESS[data.session_id]["progress"] = 100
                    NEW_CHAT_PROGRESS[data.session_id]["message"] = f"saved {len(saved_lines)} new memory(ies), skipped {len(skipped_duplicates)} duplicates"
                    NEW_CHAT_PROGRESS[data.session_id]["completed_at"] = datetime.utcnow().isoformat()
                except Exception:
                    pass
        else:
            LAST_NEW_CHAT_DEBUG["note"] = "no session history"
        
        # Rebuild life memory embeddings cache (resync with DB)
        try:
            NEW_CHAT_PROGRESS[data.session_id]["status"] = "rebuilding_cache"
            NEW_CHAT_PROGRESS[data.session_id]["progress"] = 85
            NEW_CHAT_PROGRESS[data.session_id]["message"] = "resyncing life memory cache"
            _build_life_memory_embeddings()
            logging.info("Life memory cache rebuilt after /new_chat")
        except Exception as e:
            logging.error(f"Failed to rebuild life memory cache: {type(e).__name__}: {e}")
        
        # Clear the conversation history for this session
        if data.session_id in CONVERSATION_HISTORY:
            del CONVERSATION_HISTORY[data.session_id]
        # Also clear user_name tracking
        if data.session_id in USER_NAME_BY_SESSION:
            del USER_NAME_BY_SESSION[data.session_id]

        # Optionally keep progress for a short time; we do not auto-delete here.
        LAST_NEW_CHAT_DEBUG["summary_saved"] = bool(summary_saved)
        LAST_NEW_CHAT_DEBUG["duplicate_detected"] = bool(duplicate_detected)
        # Before clearing session history, if we have token logs for this session,
        # compute totals and write them to the server console (as requested).
        token_summary = None
        try:
            toklog = SESSION_TOKEN_LOGS.get(data.session_id)
            logging.info(f"[TOKEN_DEBUG] session_id={data.session_id}, toklog_exists={toklog is not None}, toklog_type={type(toklog).__name__ if toklog is not None else 'None'}, toklog_length={len(toklog) if isinstance(toklog, list) else 'N/A'}")
            
            if isinstance(toklog, list) and toklog:
                # Flatten all calls across the session
                flat_calls = []
                for entry in toklog:
                    calls = entry.get('calls') or []
                    for c in calls:
                        flat_calls.append(c)

                total_in = sum(int(c.get('input_tokens', 0)) for c in flat_calls)
                total_out = sum(int(c.get('output_tokens', 0)) for c in flat_calls)
                total_sum = sum(int(c.get('total_tokens', 0)) for c in flat_calls)

                token_summary = {
                    'session_id': data.session_id,
                    'entries': len(toklog),
                    'calls_total': len(flat_calls),
                    'input_tokens': int(total_in),
                    'output_tokens': int(total_out),
                    'total_tokens': int(total_sum),
                    'per_call': flat_calls,
                }

                # Log the token summary to the server console (Render logs)
                logging.info(f"Session token summary: {json.dumps(token_summary, default=str, ensure_ascii=False)}")
                
                # Add token_summary to progress state so frontend can display it
                try:
                    st = NEW_CHAT_PROGRESS.get(data.session_id)
                    if not isinstance(st, dict):
                        st = {
                            "status": "started",
                            "progress": 0,
                            "message": "starting new chat",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                        NEW_CHAT_PROGRESS[data.session_id] = st
                    st["token_summary"] = token_summary
                    # include DB summary if available
                    try:
                        st["db_summary"] = LAST_NEW_CHAT_DEBUG.get("db_summary") if isinstance(LAST_NEW_CHAT_DEBUG, dict) else None
                    except Exception:
                        st["db_summary"] = None
                except Exception:
                    pass
                
                # clear the runtime token log for this session now we've reported it
                del SESSION_TOKEN_LOGS[data.session_id]
            else:
                logging.info(f"[TOKEN_DEBUG] No token log to report for session {data.session_id}")
        except Exception as e:
            logging.error(f"[TOKEN_DEBUG] Error generating token summary: {type(e).__name__}: {e}")

        # Ensure the progress state always reaches a terminal status so the frontend
        # polling loop can stop and print token summaries.
        try:
            st = NEW_CHAT_PROGRESS.get(data.session_id)
            if isinstance(st, dict):
                cur_status = st.get("status")
                if cur_status not in ("done", "error", "awaiting_clarification"):
                    st["status"] = "done"
                    st["progress"] = 100
                    if not st.get("message"):
                        if not session_history:
                            st["message"] = "nothing to summarize"
                        elif not extracted_memory:
                            st["message"] = "no durable memory extracted"
                        else:
                            st["message"] = "completed"
                    st["completed_at"] = datetime.utcnow().isoformat()
        except Exception:
            pass

        return {
            "status": "ok",
            "message": "New chat started",
            "summary_saved": summary_saved,
            "duplicate_detected": duplicate_detected,
            "extracted_memory": extracted_memory,
            "token_summary": token_summary,
            "db_summary": LAST_NEW_CHAT_DEBUG.get("db_summary") if isinstance(LAST_NEW_CHAT_DEBUG, dict) else None
        }
    except Exception as e:
        logging.error(f"Error in new_chat: {type(e).__name__}: {e}")
        try:
            LAST_NEW_CHAT_DEBUG["error"] = f"{type(e).__name__}: {e}"
        except Exception:
            pass
        try:
            # mark progress as failed
            NEW_CHAT_PROGRESS[data.session_id] = {
                "status": "error",
                "progress": NEW_CHAT_PROGRESS.get(data.session_id, {}).get("progress", 0),
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception:
            pass
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Life Memory Cleanup endpoints
# ---------------------------------------------------------------------------

# Simple header-token guard. If ADMIN_TOKEN is not set, the endpoints are
# open (fine for local/dev). Set ADMIN_TOKEN env var in production.
_ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

def _check_admin(request_token: str | None) -> bool:
    """Return True if the request should be allowed."""
    if not _ADMIN_TOKEN:
        return True  # open when env var not configured
    return request_token == _ADMIN_TOKEN


class _LifeMemoryCleanupApplyRequest(BaseModel):
    proposed: list[dict]  # each has at least {"content": str}


@app.post("/admin/life_memories/propose")
def life_memories_propose(x_admin_token: str | None = None):
    """Fetch all life memories and ask the LLM to produce a cleaned proposal."""
    from fastapi import Request as _Request  # local import to avoid shadowing
    if not _check_admin(x_admin_token):
        raise HTTPException(status_code=403, detail="Forbidden")

    if not _ensure_db_connection():
        raise HTTPException(status_code=503, detail="DB unavailable")

    try:
        with DB_CONN.cursor() as cur:
            cur.execute("SELECT id, content FROM memories WHERE type = 'life' ORDER BY id")
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB read error: {e}")

    if not rows:
        return {"original": [], "proposed": [], "summary": {"before": 0, "after": 0, "merged": 0, "deleted": 0}}

    before_count = len(rows)
    original_list = [{"content": row[1].strip()} for row in rows]
    numbered = "\n".join(f"{i+1}. {row[1].strip()}" for i, row in enumerate(rows))

    cleanup_prompt = (
        "You are cleaning up a personal memory database.\n\n"
        "Input memories (one per line, numbered):\n"
        f"{numbered}\n\n"
        "Task:\n"
        "- Merge near-duplicate or fragmented entries that describe the same event into ONE coherent memory.\n"
        "- Remove true duplicates entirely.\n"
        "- Preserve meaningful narrative detail — do NOT reduce to vague summaries.\n"
        "- Do NOT invent new facts; only recombine or rewrite using the text already present.\n"
        "- The output set should have fewer, higher-quality memories than the input.\n\n"
        "Return ONLY valid JSON with this exact structure (no markdown, no commentary):\n"
        "{\n"
        '  "proposed": [\n'
        '    { "content": "...", "tags": null, "year": null, "created_at": null }\n'
        "  ],\n"
        '  "summary": { "before": N, "after": M, "merged": X, "deleted": Y }\n'
        "}\n"
        f"before (N) must equal {before_count}. after (M) is the count of proposed memories."
    )

    try:
        _cleanup_model = _OPENROUTER_MODEL
        raw, _usage = _openrouter_chat_completion(
            model=_cleanup_model,
            messages=[{"role": "user", "content": cleanup_prompt}],
            max_tokens=4096,
            reasoning_effort=_OPENROUTER_CLEANUP_EFFORT,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    raw = (raw or "").strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"```$", "", raw).strip()

    if not raw:
        raise HTTPException(
            status_code=502,
            detail=(
                f"LLM ({_cleanup_model}) returned an empty response. "
                "This can happen with reasoning models when `content` is None. "
                "Check that the model name is correct and the API key has access."
            ),
        )

    try:
        result = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {e}. Raw: {raw[:300]}")

    # Ensure summary reflects reality
    proposed = result.get("proposed", [])
    after_count = len(proposed)
    summary = result.get("summary", {})
    summary["before"] = before_count
    summary["after"] = after_count
    result["summary"] = summary

    result["original"] = original_list
    logging.info(f"life_memories_propose: before={before_count} after={after_count}")
    return result


@app.post("/admin/life_memories/apply")
def life_memories_apply(body: _LifeMemoryCleanupApplyRequest, x_admin_token: str | None = None):
    """Replace all life memories with the approved proposed set."""
    if not _check_admin(x_admin_token):
        raise HTTPException(status_code=403, detail="Forbidden")

    if not _ensure_db_connection():
        raise HTTPException(status_code=503, detail="DB unavailable")

    new_contents = [m["content"].strip() for m in body.proposed if m.get("content", "").strip()]
    if not new_contents:
        raise HTTPException(status_code=400, detail="proposed list is empty")

    try:
        with DB_CONN.cursor() as cur:
            # --- DB backup before any destructive change ---
            try:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS life_memory_backups "
                    "(id SERIAL PRIMARY KEY, backed_up_at TIMESTAMPTZ DEFAULT NOW(), memories TEXT NOT NULL)"
                )
                cur.execute("SELECT content FROM memories WHERE type = 'life' ORDER BY id")
                existing = [r[0] for r in cur.fetchall()]
                cur.execute(
                    "INSERT INTO life_memory_backups (memories) VALUES (%s)",
                    (json.dumps(existing),)
                )
            except Exception as bk_err:
                logging.warning(f"life_memories_apply: backup step failed (continuing): {bk_err}")
            # --- Replace ---
            cur.execute("DELETE FROM memories WHERE type = 'life'")
            for content in new_contents:
                mem_hash = _compute_memory_hash(content)
                try:
                    cur.execute(
                        "INSERT INTO memories (content, type, normalized_hash) VALUES (%s, %s, %s) ON CONFLICT (type, normalized_hash) DO NOTHING",
                        (content, "life", mem_hash),
                    )
                except Exception:
                    cur.execute("INSERT INTO memories (content, type) VALUES (%s, %s)", (content, "life"))
        DB_CONN.commit()
    except Exception as e:
        try:
            DB_CONN.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"DB write error: {e}")

    # Rebuild embedding cache
    try:
        _build_life_memory_embeddings()
    except Exception:
        pass

    logging.info(f"life_memories_apply: inserted {len(new_contents)} memories")
    return {"ok": True, "inserted": len(new_contents)}


@app.get("/new_chat_progress/{session_id}")
def new_chat_progress(session_id: str):
    """Return the current progress state for a `/new_chat` operation for a session.

    Frontend can poll this endpoint to power a popup/progress bar.
    """
    try:
        state = NEW_CHAT_PROGRESS.get(session_id)
        if state is None:
            return {"session_id": session_id, "status": "not_found"}
        return {"session_id": session_id, "state": state}
    except Exception as e:
        logging.error(f"new_chat_progress error: {type(e).__name__}: {e}")
        return {"session_id": session_id, "status": "error", "error": str(e)}