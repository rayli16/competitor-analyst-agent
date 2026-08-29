# ---- builder: install dependencies, may need to compile from source ----
FROM python:3.14-slim AS builder

WORKDIR /app

# gcc only exists in this stage. Python 3.14 is recent enough that not every
# pinned package is guaranteed to have a pre-built wheel for it yet — if pip
# falls back to compiling one from source, this is what makes that possible.
# It never reaches the runtime image below.
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Copied alone, before the rest of the app, so this layer — the slow one —
# only rebuilds when dependencies actually change, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---- runtime: only what's needed to run the app ----
FROM python:3.14-slim

WORKDIR /app

# PYTHONUNBUFFERED matters here specifically: main.py's except blocks call
# logger.exception() for real diagnosis, and buffered stdout would delay or
# drop that output from `docker logs` / the platform's log stream.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# The installed venv, nothing else from the builder stage — no gcc, no pip
# cache, no apt state.
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

RUN useradd --create-home appuser
COPY --chown=appuser:appuser app/ ./app/
USER appuser

EXPOSE 8000

# Shell form, not exec-form CMD, so ${PORT:-8000} actually expands — most
# hosting platforms inject PORT at runtime and expect the app to bind to it,
# not a fixed value; 8000 is just the local-run default. `exec` in front
# replaces the shell process with uvicorn instead of running it as a child,
# so uvicorn becomes PID 1 and receives SIGTERM directly — without it,
# `docker stop`/a platform shutdown would signal the shell, not the app.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
