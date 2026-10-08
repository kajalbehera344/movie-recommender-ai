#!/usr/bin/env bash
# Starts the FastAPI backend (background) and the Streamlit frontend (foreground).
# First run also creates the venv and installs dependencies (needs internet, ~3-5 min).
# Usage:  bash run_demo.sh      (Ctrl+C stops both)
set -e
cd "$(dirname "$0")"

if [ ! -f .env ] || grep -q "your_tmdb_api_key_here" .env; then
  echo "Put your TMDB API key in .env first (free key: https://www.themoviedb.org/settings/api)"
  exit 1
fi

if [ ! -x venv/bin/python ]; then
  # Find a Python 3.10-3.12 (the pinned libraries don't support 3.13 yet)
  PY=""
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)'; then
      PY="$c"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.10, 3.11 or 3.12 is required. Install it from https://www.python.org/downloads/"
    exit 1
  fi
  echo "First run: creating virtual environment and installing dependencies (a few minutes)..."
  "$PY" -m venv venv
  venv/bin/python -m pip install --upgrade pip -q
  venv/bin/python -m pip install -r requirements.txt || {
    rm -rf venv
    echo "Dependency install failed - see the messages above. Check your internet connection and run again."
    exit 1
  }
fi

echo "Starting backend on http://127.0.0.1:8000 ..."
venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
trap 'kill $BACKEND_PID 2>/dev/null' EXIT

# Wait for the backend to load the pickles before opening the UI
for _ in $(seq 1 60); do
  curl -s http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  sleep 1
done

echo "Starting frontend on http://localhost:8501 ..."
venv/bin/python -m streamlit run app.py --server.port 8501
