import os

# app.agent and app.discovery build their Tavily client at import time, and
# the clients fail fast without a key (by design — see get_tavily_api_keys).
# Unit tests never make network calls, so a dummy value is enough to import.
# setdefault, not assignment: a real key already in the environment is left
# alone, and load_dotenv() won't override whatever is set here.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-dummy-key")
os.environ.setdefault("TAVILY_API_KEY", "test-dummy-key")
