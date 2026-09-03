"""
D Desk AI — Configuration
Centralized settings for the FastAPI microservice architecture.
"""

import os
from pathlib import Path


def _load_env_file() -> None:
    """Load simple KEY=VALUE pairs from the project .env file if present."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file()

class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", "d-desk-ai-secret-key-2024")

    # Groq / LLM
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1536"))
    GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.4"))

    # Classifier
    DUPLICATE_THRESHOLD = 0.75

    # Firebase / Cloud Firestore (required)
    FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
    FIREBASE_CREDENTIALS_JSON = os.getenv("FIREBASE_CREDENTIALS_JSON", "")
    FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "d-desk-ai")

    # Demo Account Seeding (read from environment — never hardcode)
    DEMO_ADMIN_EMAIL = os.getenv("DEMO_ADMIN_EMAIL", "")
    DEMO_ADMIN_PASSWORD = os.getenv("DEMO_ADMIN_PASSWORD", "")
    DEMO_DEFAULT_PASSWORD = os.getenv("DEMO_DEFAULT_PASSWORD", "")

    # CORS
    CORS_ORIGINS = ["*"]
