from dotenv import load_dotenv
import os
from pathlib import Path
p = Path(__file__).parent / '.env'
loaded = load_dotenv(p)
print('loaded=', loaded)
print('has_key=', bool(os.getenv('OPENAI_API_KEY')))