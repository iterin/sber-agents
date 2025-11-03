from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

# Храним последние ~20 пар (user/assistant) = 40 сообщений (role, content)
ChatMessage = Tuple[str, str]
ChatMemory = Deque[ChatMessage]

_chat_id_to_history: Dict[int, ChatMemory] = defaultdict(lambda: deque(maxlen=40))


def get_history(chat_id: int) -> ChatMemory:
    return _chat_id_to_history[chat_id]


def reset_history(chat_id: int) -> None:
    _chat_id_to_history.pop(chat_id, None)


