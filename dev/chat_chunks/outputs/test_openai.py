import os
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("Testing OpenAI API access...")
print(f"API key loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
print(f"Key prefix: {os.getenv('OPENAI_API_KEY')[:20]}..." if os.getenv('OPENAI_API_KEY') else "No key")

try:
    # Try a simple embedding call
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input="test"
    )
    print(f"✓ Successfully created embedding")
    print(f"  Dimension: {len(resp.data[0].embedding)}")
    print(f"  Model: {resp.model}")
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nTrying text-embedding-ada-002 instead...")
    try:
        resp = client.embeddings.create(
            model="text-embedding-ada-002",
            input="test"
        )
        print(f"✓ text-embedding-ada-002 works!")
        print(f"  Dimension: {len(resp.data[0].embedding)}")
    except Exception as e2:
        print(f"✗ Also failed: {e2}")
