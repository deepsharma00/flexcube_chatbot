"""
Central configuration — reads from .env file in the project root.
No external dependencies (no python-dotenv needed).

All other modules import OLLAMA_MODEL and OLLAMA_URL from here.
To change the model or server, edit the .env file:

    OLLAMA_MODEL=qwen2.5:7b
    OLLAMA_URL=http://217.216.78.230:11434
"""

import os
from pathlib import Path


def LoadEnvFile():
    """
    Read .env file from the same directory as this config.py.
    Sets values as environment variables. Skips comments and blank lines.
    Does NOT override existing env vars (system env takes priority).
    """
    EnvPath = Path(__file__).parent / ".env"
    if not EnvPath.exists():
        return

    with open(EnvPath, "r", encoding="utf-8") as F:
        for Line in F:
            Line = Line.strip()
            if not Line or Line.startswith("#"):
                continue
            if "=" not in Line:
                continue
            Key, Value = Line.split("=", 1)
            Key = Key.strip()
            Value = Value.strip()
            # Don't override existing env vars
            if Key not in os.environ:
                os.environ[Key] = Value


# Load .env on import
LoadEnvFile()

# Central config values — every module reads these
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
