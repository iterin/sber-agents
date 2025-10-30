# Техническое видение проекта

## Технологии

**Основные технологии:**
- **Python 3.11+** - основной язык разработки
- **uv** - управление зависимостями и виртуальным окружением
- **aiogram 3.x** - фреймворк для Telegram Bot API (polling)
- **openai** - клиент для работы с LLM через Openrouter
- **python-dotenv** - для работы с переменными окружения
- **Make** - автоматизация сборки и запуска

## Принципы разработки

**Принципы:**
- **KISS** (Keep It Simple, Stupid) - максимальная простота решений
- **YAGNI** (You Aren't Gonna Need It) - реализуем только то, что нужно сейчас
- **Монолитная архитектура** - весь код в одном месте, никаких микросервисов
- **Прямолинейный код** - минимум абстракций, максимум читаемости
- **Быстрый старт** - от идеи до рабочего прототипа за минимальное время

**Что НЕ делаем:**
- Не создаем сложные архитектурные паттерны
- Не делаем преждевременную оптимизацию
- Не добавляем функции "на будущее"
- Не усложняем без крайней необходимости

## Структура проекта

```
/
├── src/
│   ├── bot.py          # Основной файл бота, инициализация aiogram
│   ├── handlers.py     # Обработчики команд и сообщений Telegram
│   ├── llm.py          # Работа с LLM через OpenRouter
│   └── config.py       # Загрузка конфигурации из .env
├── .env                # Переменные окружения (токены, настройки)
├── .env.example        # Пример конфигурации
├── pyproject.toml      # Конфигурация проекта для uv
├── Makefile            # Команды для запуска и управления
└── README.md           # Документация по запуску
```

**Принцип:** Всего 4 Python-файла в одной папке `src/`. Никаких пакетов, подпакетов, сложной иерархии.

## Архитектура проекта

**Компоненты:**

1. **bot.py** - точка входа
   - Инициализирует aiogram Bot и Dispatcher
   - Регистрирует handlers
   - Запускает polling

2. **handlers.py** - обработка событий
   - `/start` - приветствие и очистка истории
   - Обработчик всех текстовых сообщений → добавление в историю → вызов LLM → сохранение ответа в историю
   - Хранит историю диалогов в памяти: `dict[int, list]` (chat_id → список сообщений)

3. **llm.py** - интеграция с LLM
   - Один метод: `get_response(message_history: list) -> str`
   - Отправляет запрос в OpenRouter через openai client с историей сообщений
   - Возвращает ответ

4. **config.py** - конфигурация
   - Класс Config с полями: `TELEGRAM_TOKEN`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL`, `SYSTEM_PROMPT`
   - Загрузка из .env через python-dotenv

**Поток данных:**
```
Telegram → handlers.py (добавить в историю) → llm.py → OpenRouter → 
llm.py → handlers.py (сохранить ответ в историю) → Telegram
```

**Принцип:** Никакой DI, никаких интерфейсов, никаких слоев абстракции. Просто прямые вызовы функций.

## Модель данных

**Хранение в памяти (без БД):**

Глобальный словарь в `handlers.py`:
```python
chat_conversations: dict[int, list[dict]] = {}
```

**Структура истории диалога:**
```python
chat_conversations[chat_id] = [
    {"role": "system", "content": "системный промпт"},
    {"role": "user", "content": "сообщение пользователя"},
    {"role": "assistant", "content": "ответ LLM"},
    ...
]
```

**Операции:**
- При `/start` - очищаем историю для данного чата
- При новом сообщении - добавляем в список
- Передаем весь список в LLM для контекста
- При перезапуске бота - вся история теряется

**Принцип:** Максимальная простота. Никаких БД, файлов, сериализации. История живет только в runtime.

## Работа с LLM

**Используемая библиотека:** `openai` (официальный Python client, асинхронная версия)

**Настройка:**
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL  # https://openrouter.ai/api/v1
)
```

**Основной метод в llm.py:**
```python
async def get_response(message_history: list[dict]) -> str:
    response = await client.chat.completions.create(
        model=config.MODEL,  # например "openai/gpt-oss-20b:free"
        messages=message_history
    )
    return response.choices[0].message.content
```

**Параметры из .env:**
- `OPENAI_API_KEY` - ключ от OpenRouter
- `OPENAI_BASE_URL` - `https://openrouter.ai/api/v1`
- `MODEL` - название модели (например `openai/gpt-oss-20b:free`)
- `SYSTEM_PROMPT` - роль/инструкция для LLM

**Обработка ошибок:**
- try/except для сетевых ошибок
- Возврат простого сообщения об ошибке пользователю

**Принцип:** Асинхронный запрос-ответ. Никакого retry, никаких очередей, никакого streaming.

## Сценарии работы

**Сценарий 1: Первый запуск**
1. Пользователь отправляет `/start`
2. Бот отвечает приветственным сообщением
3. История диалога инициализируется с системным промптом

**Сценарий 2: Диалог**
1. Пользователь пишет текстовое сообщение
2. Бот добавляет сообщение в историю чата
3. Бот отправляет историю чата в LLM
4. Бот получает ответ и добавляет его в историю чата
5. Бот отправляет ответ пользователю

**Сценарий 3: Сброс контекста**
1. Пользователь отправляет `/start`
2. История диалога очищается
3. Начинается новый диалог

**Ограничения:**
- Бот работает только с текстом (не обрабатывает фото, файлы, голосовые)
- Один пользователь не блокирует других (асинхронность)
- При перезапуске бота все истории теряются

## Подход к конфигурированию

**Файл .env** (не коммитится в git):
```bash
TELEGRAM_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openrouter_api_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL=openai/gpt-oss-20b:free
SYSTEM_PROMPT=Ты дружелюбный ассистент, который помогает пользователю.
```

**Файл .env.example** (коммитится):
```bash
TELEGRAM_TOKEN=
OPENAI_API_KEY=
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL=openai/gpt-oss-20b:free
SYSTEM_PROMPT=Ты дружелюбный ассистент.
```

**config.py:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    MODEL = os.getenv("MODEL")
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")

config = Config()
```

**Принципы:**
- Все секреты только в .env
- Нет YAML, JSON, TOML конфигов
- Нет окружений (dev/prod)
- Нет валидации на старте (упадет при первом использовании если что-то не так)

## Подход к логгированию

**Используем встроенный logging Python:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

**Что логируем:**
- Старт/остановка бота
- Входящие сообщения от пользователей (chat_id + текст)
- Ошибки при вызове LLM
- Исключения

**Что НЕ логируем:**
- Содержимое ответов LLM (избыточно для MVP)
- Детальные трейсы успешных операций
- Метрики, аналитика

**Вывод:** Только в stdout/stderr (консоль)

**Принципы:**
- Без внешних библиотек (structlog и т.п.)
- Без файлов, ротации логов
- Без отправки в внешние системы
- Простой текстовый формат


