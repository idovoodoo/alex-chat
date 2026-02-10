# Development-only workspace

Use this `dev/` folder for experimental code, local tools, and features that should not be deployed to Render.

Guidelines:

- Keep all non-deployable files under `dev/` or `local_dev/`.
- Add any secrets to your local environment or a `.env` file (which is ignored).
- When ready to deploy code, move or cherry-pick files into the main project tree.
