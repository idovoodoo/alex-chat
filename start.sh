#!/usr/bin/env bash

# Start Uvicorn serving the FastAPI app. Use PORT from environment (Render sets this).
# Suppress access logs (all debug output goes to browser console via /debug/last_console)
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --log-level warning --no-access-log
