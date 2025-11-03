import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения из текущей директории и из корня проекта
load_dotenv()
project_root_env = Path(__file__).resolve().parents[1] / ".env"
if project_root_env.exists():
    load_dotenv(dotenv_path=project_root_env, override=False)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_PROXY_URL = os.getenv("TELEGRAM_PROXY_URL", "").strip()  # optional, e.g. http://user:pass@host:port or socks5://user:pass@host:port

# LLM config (supports OpenAI and OpenRouter)
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip()
_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "").strip()

_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
_OPENROUTER_BASE_URL = os.getenv("OPENROUTER_API_BASE_URL", "").strip() or os.getenv("OPENROUTER_BASE_URL", "").strip()
_MODEL_NAME = os.getenv("MODEL_NAME", "").strip()

# Effective values used by llm.py
OPENAI_API_KEY = _OPENAI_API_KEY or _OPENROUTER_API_KEY
OPENAI_MODEL = _MODEL_NAME or _OPENAI_MODEL or _OPENROUTER_MODEL or "gpt-4o-mini"
OPENAI_BASE_URL = _OPENAI_BASE_URL or _OPENROUTER_BASE_URL

# If OpenRouter key detected and base_url is not set, default to OpenRouter endpoint
if not OPENAI_BASE_URL and OPENAI_API_KEY.startswith("sk-or-"):
    OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
DEBUG_LLM = os.getenv("DEBUG_LLM", "").strip()

def env_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


def validate_config() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")


