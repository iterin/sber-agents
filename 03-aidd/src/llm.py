from typing import Optional, List, Dict
import os
from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    DEBUG_LLM,
    env_bool,
    LLM_TIMEOUT_S,
    LLM_RETRY_COUNT,
)
from collections import deque
from src.memory import ChatMemory

SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"


def load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are a helpful assistant. Keep responses concise."


def build_messages_with_context(user_text: str, history: Optional[ChatMemory]) -> List[Dict[str, str]]:
    system_prompt = load_system_prompt()
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        for role, content in history:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})
    return messages


def generate_reply(user_text: str, history: Optional[ChatMemory] = None) -> str:
    if not user_text:
        return ""

    # Mock mode if no key
    if not OPENAI_API_KEY:
        if env_bool(DEBUG_LLM or ""):
            print("[LLM DEBUG] No OPENAI_API_KEY; returning mock response")
        return f"[mock] {user_text}"

    try:
        from openai import OpenAI  # lazy import to avoid hard dependency at startup
    except Exception as e:
        if env_bool(DEBUG_LLM or ""):
            print(f"[LLM DEBUG] openai import failed: {type(e).__name__}: {e}")
        return f"[mock] {user_text}"

    system_prompt = load_system_prompt()
    try:
        client_kwargs = {"api_key": OPENAI_API_KEY, "timeout": LLM_TIMEOUT_S, "max_retries": LLM_RETRY_COUNT}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        if env_bool(DEBUG_LLM or ""):
            masked = OPENAI_API_KEY[:4] + "..." if OPENAI_API_KEY else ""
            print(
                f"[LLM DEBUG] init client base_url={OPENAI_BASE_URL or 'default'} model={OPENAI_MODEL} key={masked} timeout={LLM_TIMEOUT_S}s retries={LLM_RETRY_COUNT}"
            )
        client = OpenAI(**client_kwargs)
        messages = build_messages_with_context(user_text, history)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.2,
        )
        if env_bool(DEBUG_LLM or ""):
            print("[LLM DEBUG] response received")
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        # Minimal diagnostic to console, user still gets mock reply to avoid crashes
        print(f"[LLM ERROR] {type(e).__name__}: {e}")
        # If key present but failed, return polite fallback; if no key, mock already returned earlier
        if OPENAI_API_KEY:
            return "Извините, сервис временно недоступен. Попробуйте позже."
        return f"[mock] {user_text}"


