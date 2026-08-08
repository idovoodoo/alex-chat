## Alex Chat

Alex Chat - a small scaffold for a chat assistant project.

## What this repo contains

- Initial repository scaffold for the Alex Chat project

## Getting started

1. Install dependencies with `pip install -r requirements.txt`.
2. Create a file named `.env` in the repository root (next to `README.md`).
3. Add your MiniMax key:

	`MINIMAX_API_KEY=your_minimax_api_key_here`

	   The app uses the MiniMax model `MiniMax-M3`. To select another model supported by your MiniMax account, add:

		`MINIMAX_MODEL=MiniMax-M3`

	The default API endpoint is `https://api.minimax.io/v1`. The app also uses MiniMax's `embo-01` embedding model for RAG and memory search. If your account uses another MiniMax endpoint, set it with:

	`MINIMAX_BASE_URL=https://api.minimax.io/v1`

	To override the embedding model, set `MINIMAX_EMBEDDING_MODEL=embo-01`.

4. Start the app with `uvicorn app.main:app --reload` or `start.sh`.

The `.env` file is for local development only and must not be committed. For Render, add `MINIMAX_API_KEY` in **Dashboard → your service → Environment → Environment Variables**, then redeploy. All application LLM calls—including chat, embeddings, memory extraction, tagging, and cleanup—use MiniMax.

## License
This project is available under the MIT license — see `LICENSE`.
