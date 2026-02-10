#!/usr/bin/env bash

# Start Uvicorn serving the FastAPI app. Use PORT from environment (Render sets this).
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
