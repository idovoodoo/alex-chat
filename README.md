## Alex Chat

Alex Chat - a small scaffold for a chat assistant project.

## What this repo contains

- Initial repository scaffold for the Alex Chat project

## Getting started

1. Install dependencies with `pip install -r requirements.txt`.
2. Create a file named `.env` in the repository root (next to `README.md`).
3. Add your OpenRouter key:

	`OPENROUTER_API_KEY=your_openrouter_api_key_here`

	   The app uses the OpenRouter model `openai/gpt-5.6-luna`. To select another model supported by OpenRouter, add:

		`OPENROUTER_MODEL=openai/gpt-5.6-luna`

	The default API endpoint is `https://openrouter.ai/api/v1`. The app also uses OpenRouter's `openai/text-embedding-3-small` embedding model for RAG and memory search. To override the embedding model, set `OPENAI_EMBEDDING_MODEL=openai/text-embedding-3-small`.

4. Start the app with `uvicorn app.main:app --reload` or `start.sh`.

The `.env` file is for local development only and must not be committed. For Render, add `OPENROUTER_API_KEY` in **Dashboard → your service → Environment → Environment Variables**, then redeploy. All application LLM calls—including chat, embeddings, memory extraction, tagging, and cleanup—use OpenRouter.

## License
This project is available under the MIT license — see `LICENSE`.
