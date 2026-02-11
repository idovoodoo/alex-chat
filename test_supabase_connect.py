import os
import sys
import psycopg2
from dotenv import load_dotenv

# Try loading .env from common locations (repo root, dev/chat_chunks/outputs)
root_dotenv = os.path.join(os.path.dirname(__file__), ".env")
alt_dotenv = os.path.join(os.path.dirname(__file__), "dev", "chat_chunks", "outputs", ".env")
if os.path.exists(alt_dotenv):
    load_dotenv(alt_dotenv)
elif os.path.exists(root_dotenv):
    load_dotenv(root_dotenv)
else:
    # fallback to default loader which looks in CWD
    load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
if not SUPABASE_DB_URL:
    print("SUPABASE_DB_URL not found in environment or .env")
    sys.exit(2)

print("Found SUPABASE_DB_URL; attempting to connect...")
try:
    conn = psycopg2.connect(SUPABASE_DB_URL, connect_timeout=10)
    cur = conn.cursor()
    try:
        # Try to count rows in core_memories if it exists
        cur.execute("SELECT count(*) FROM core_memories;")
        cnt = cur.fetchone()[0]
        print(f"Connection successful. core_memories rows: {cnt}")
    except Exception as e:
        # If table doesn't exist, fallback to simple test query
        try:
            cur.execute("SELECT 1;")
            print("Connected but querying `core_memories` failed:", str(e))
        except Exception as e2:
            print("Connected attempt failed on queries:", str(e2))
    finally:
        cur.close()
        conn.close()
        print("Connection closed.")
    sys.exit(0)
except Exception as e:
    print("Connection failed:", str(e))
    sys.exit(1)
