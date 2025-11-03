from typing import Optional
import os
from openai import OpenAI

SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


def load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are a helpful assistant. Keep responses concise."


def generate_reply(user_text: str) -> str:
    if not user_text:
        return ""

    # Mock mode if no key
    if not OPENAI_API_KEY:
        return f"[mock] {user_text}"

    client = OpenAI(api_key=OPENAI_API_KEY)
    system_prompt = load_system_prompt()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


