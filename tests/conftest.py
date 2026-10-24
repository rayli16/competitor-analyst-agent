import os

# app.agent and app.discovery build their Tavily client at import time, and
# the clients fail fast without a key (by design — see get_tavily_api_keys).
# Unit tests never make network calls, so a dummy value is enough to import.
# setdefault, not assignment: a real key already in the environment is left
# alone, and load_dotenv() won't override whatever is set here.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-dummy-key")
os.environ.setdefault("TAVILY_API_KEY", "test-dummy-key")

# Forced, not setdefault: tests drive the real pipeline with fake searches,
# and with tracing on, a local run (real Langfuse keys in .env) would send
# those fake traces to the real project.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
